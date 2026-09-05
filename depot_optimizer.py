#!/usr/bin/env python3
"""
depot_optimizer.py
==================

Standalone depot-location optimizer.

Pipeline
--------
1. Load customer data (lat, lon, code, volume) from CSV.               [pandas]
2. Generate candidate depot sites with K-Means on the unit sphere.     [scipy.cluster.vq]
3. Compute a quick nearest-candidate baseline assignment with vq().    [scipy.cluster.vq]
4. Build a haversine distance matrix (customers x candidates).        [math / numpy]
5. Solve a MILP that jointly:
      - maximises the number of customers served within the radius
      - minimises the number of depots opened
      - minimises the volume-weighted transport distance
   using PuLP + the bundled CBC solver.                                [PuLP]
6. Export a results workbook.                                          [pandas -> openpyxl]
7. Draw a static bubble chart (bubble size ~ volume).                  [matplotlib]
8. Emit a self-contained interactive Leaflet/OSM HTML map with
   volume-scaled markers and a fixed-radius service circle per depot.  [Leaflet.js + OSM]

Run `python depot_optimizer.py --help` for all options, or
`python depot_optimizer.py --generate-sample` to create a demo CSV first.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2, vq

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EARTH_R_KM = 6371.0
REQUIRED_COLS = ["Customer_code", "customer_geocode_lat", "customer_geocode_long", "volume"]


# ----------------------------------------------------------------------------
# 1. Geometry helpers  (math / numpy)
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km. Inputs broadcast like numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_R_KM * c


def latlon_to_unit_sphere(lat, lon):
    """Project lat/lon onto the unit sphere so that Euclidean K-Means behaves
    sensibly near the poles / dateline (used only for candidate generation)."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def unit_sphere_to_latlon(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    norm = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arcsin(np.clip(z / norm, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return np.column_stack([lat, lon])


def distance_matrix_km(cust_latlon, cand_latlon):
    """(n_customers, n_candidates) haversine distance matrix, fully vectorised."""
    lat1 = cust_latlon[:, 0][:, None]
    lon1 = cust_latlon[:, 1][:, None]
    lat2 = cand_latlon[:, 0][None, :]
    lon2 = cand_latlon[:, 1][None, :]
    return haversine_km(lat1, lon1, lat2, lon2)


# ----------------------------------------------------------------------------
# 2 & 3. Candidate generation + baseline assignment  (scipy.cluster.vq)
# ----------------------------------------------------------------------------
def generate_candidates(cust_latlon, n_candidates, seed=42):
    """K-Means (on the unit sphere) to propose n_candidates depot sites.

    Using MORE candidates than the final expected depot count gives the MILP
    real choices to pick from; it decides which candidates to actually open.
    """
    xyz = latlon_to_unit_sphere(cust_latlon[:, 0], cust_latlon[:, 1])
    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(xyz, k=n_candidates, seed=rng, minit="++")
    cand_latlon = unit_sphere_to_latlon(centroids)
    return cand_latlon, labels


def baseline_vq_assignment(cust_latlon, cand_latlon):
    """Cheap nearest-candidate assignment (no radius/capacity awareness) used
    only as a naive baseline to contrast against the MILP result."""
    xyz_cust = latlon_to_unit_sphere(cust_latlon[:, 0], cust_latlon[:, 1])
    xyz_cand = latlon_to_unit_sphere(cand_latlon[:, 0], cand_latlon[:, 1])
    idx, _ = vq(xyz_cust, xyz_cand)
    return idx


# ----------------------------------------------------------------------------
# 4 & 5. MILP  (PuLP + CBC)
# ----------------------------------------------------------------------------
def solve_milp(dist_km, volume, max_radius_km, max_depots=None, capacity=None,
               w_unassigned=None, w_depot=None, w_dist=1.0, time_limit_sec=120,
               msg=False):
    """
    Facility-location MILP.

    Sets
        i in customers, j in candidate depots
    Decision variables
        y_j     = 1 if depot j is opened
        x_ij    = 1 if customer i is served by depot j   (only defined where
                  dist_ij <= max_radius_km -- infeasible pairs get no variable
                  at all, which keeps the model small)
        u_i     = 1 if customer i ends up unassigned (slack)
    Constraints
        sum_j x_ij + u_i == 1                         for every customer i
        x_ij <= y_j                                    for every feasible (i,j)
        sum_j y_j <= max_depots                        (optional hard cap)
        sum_i volume_i * x_ij <= capacity_j            (optional, per depot)
    Objective (single weighted MILP that folds in all three goals at once)
        minimise   W_UNASSIGNED * sum(u_i)                  -- maximise coverage
                 + W_DEPOT      * sum(y_j)                  -- minimise #depots
                 + w_dist       * sum(volume_i * dist_ij * x_ij)  -- min. cost
    The weights are auto-scaled (unless overridden) so that coverage always
    dominates depot-count, which always dominates raw distance -- i.e. the
    solver will never leave someone unserved just to save a depot, and will
    never open an extra depot just to shave a few km.
    """
    import pulp

    n, m = dist_km.shape
    feasible = np.argwhere(dist_km <= max_radius_km)  # (i, j) pairs
    if feasible.size == 0:
        raise ValueError(
            f"No customer is within {max_radius_km} km of any candidate depot. "
            "Increase --max-radius or --candidates."
        )

    # ---- auto-scale objective weights so coverage >> depot-count >> distance
    max_dist_cost = float(volume.sum()) * float(max_radius_km)  # worst-case distance term
    if w_depot is None:
        w_depot = max(1.0, max_dist_cost * 10.0)
    if w_unassigned is None:
        w_unassigned = max(1.0, w_depot * (m + 1) * 10.0)

    prob = pulp.LpProblem("depot_location", pulp.LpMinimize)

    y = {j: pulp.LpVariable(f"open_{j}", cat="Binary") for j in range(m)}
    x = {(i, j): pulp.LpVariable(f"assign_{i}_{j}", cat="Binary") for i, j in feasible}
    u = {i: pulp.LpVariable(f"unassigned_{i}", cat="Binary") for i in range(n)}

    # coverage constraint per customer
    by_customer = {i: [] for i in range(n)}
    for i, j in feasible:
        by_customer[i].append(j)
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in by_customer[i]) + u[i] == 1, f"cover_{i}"

    # link assignment to open depots
    for i, j in feasible:
        prob += x[(i, j)] <= y[j], f"link_{i}_{j}"

    # optional hard cap on number of depots opened
    if max_depots is not None:
        prob += pulp.lpSum(y[j] for j in range(m)) <= int(max_depots), "max_depots"

    # optional capacity constraint (volume served per depot)
    if capacity is not None:
        by_depot = {j: [] for j in range(m)}
        for i, j in feasible:
            by_depot[j].append(i)
        cap = capacity if hasattr(capacity, "__len__") else [capacity] * m
        for j in range(m):
            prob += (
                pulp.lpSum(volume[i] * x[(i, j)] for i in by_depot[j]) <= cap[j] * y[j],
                f"capacity_{j}",
            )

    # objective
    prob += (
        w_unassigned * pulp.lpSum(u.values())
        + w_depot * pulp.lpSum(y.values())
        + w_dist * pulp.lpSum(volume[i] * float(dist_km[i, j]) * x[(i, j)] for i, j in feasible)
    )

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    status = prob.solve(solver)

    y_val = np.array([int(round(y[j].value())) for j in range(m)])
    assign = np.full(n, -1, dtype=int)
    for i, j in feasible:
        if round(x[(i, j)].value()) == 1:
            assign[i] = j

    return {
        "status": pulp.LpStatus[status],
        "opened": y_val,
        "assignment": assign,
        "weights": {"w_unassigned": w_unassigned, "w_depot": w_depot, "w_dist": w_dist},
    }


# ----------------------------------------------------------------------------
# 6. Results workbook  (pandas -> openpyxl)
# ----------------------------------------------------------------------------
def build_results_df(df, cand_latlon, assign, dist_km, opened_mask):
    out = df.copy()
    out["depot_index"] = assign
    out["Depot_Name"] = [
        f"Depot_{int(j) + 1}" if j != -1 else "Unassigned" for j in assign
    ]
    lat = np.where(assign != -1, cand_latlon[assign.clip(min=0), 0], np.nan)
    lon = np.where(assign != -1, cand_latlon[assign.clip(min=0), 1], np.nan)
    out["Depot_Lat"] = lat
    out["Depot_Lon"] = lon
    dists = np.array(
        [dist_km[i, assign[i]] if assign[i] != -1 else np.nan for i in range(len(assign))]
    )
    out["Distance_from_Customer_km"] = dists
    out["Time_hr"] = out["Distance_from_Customer_km"] / 40.0
    return out


def export_excel(out_df, depot_summary_df, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Customer_Assignments")
        depot_summary_df.to_excel(writer, index=False, sheet_name="Depot_Summary")


# ----------------------------------------------------------------------------
# 7. Static bubble chart  (matplotlib)
# ----------------------------------------------------------------------------
def plot_bubble_chart(out_df, cand_latlon, opened_idx, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9))

    n_depots = len(opened_idx)
    # matplotlib.cm.get_cmap() was removed in newer matplotlib (>=3.9);
    # matplotlib.colormaps[...] is the current API.
    cmap = matplotlib.colormaps["tab20"].resampled(max(n_depots, 1))
    depot_color = {d: cmap(k) for k, d in enumerate(opened_idx)}

    vol = out_df["volume"].to_numpy(dtype=float)
    vol_safe = np.clip(vol, 1e-6, None)
    sizes = 15.0 + 300.0 * np.sqrt(vol_safe) / np.sqrt(vol_safe.max())

    assigned_mask = out_df["depot_index"].to_numpy() != -1
    colors = [
        depot_color.get(d, (0.6, 0.6, 0.6, 1.0)) for d in out_df["depot_index"]
    ]

    ax.scatter(
        out_df["customer_geocode_long"], out_df["customer_geocode_lat"],
        s=sizes, c=colors, alpha=0.6, linewidths=0.3, edgecolors="black",
        label="_nolegend_",
    )
    ax.scatter(
        out_df.loc[~assigned_mask, "customer_geocode_long"],
        out_df.loc[~assigned_mask, "customer_geocode_lat"],
        s=25, c="red", marker="x", label="Unassigned",
    )

    for k, d in enumerate(opened_idx):
        lat, lon = cand_latlon[d]
        ax.scatter(lon, lat, marker="*", s=500, c=[depot_color[d]],
                   edgecolors="black", linewidths=1.2, zorder=5)
        ax.annotate(f"Depot_{d + 1}", (lon, lat), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, fontweight="bold")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Depot Optimization — bubble size ~ customer volume")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 8. Interactive Leaflet / OSM map
# ----------------------------------------------------------------------------
def build_leaflet_html(out_df, cand_latlon, opened_idx, depot_summary_df,
                        circle_radius_km, path):
    vol = out_df["volume"].to_numpy(dtype=float)
    vol_safe = np.clip(vol, 1e-6, None)
    max_vol = float(vol_safe.max())

    customers = []
    for _, r in out_df.iterrows():
        radius_px = 3 + 12 * math.sqrt(max(r["volume"], 0) / max_vol) if max_vol > 0 else 4
        customers.append({
            "lat": float(r["customer_geocode_lat"]),
            "lon": float(r["customer_geocode_long"]),
            "code": str(r["Customer_code"]),
            "volume": float(r["volume"]),
            "depot": str(r["Depot_Name"]),
            "assigned": bool(r["depot_index"] != -1),
            "radius_px": radius_px,
        })

    depots = []
    summary_by_idx = depot_summary_df.set_index("depot_index").to_dict("index")
    for d in opened_idx:
        lat, lon = cand_latlon[d]
        info = summary_by_idx.get(d, {})
        depots.append({
            "lat": float(lat),
            "lon": float(lon),
            "name": f"Depot_{d + 1}",
            "customers": int(info.get("n_customers", 0)),
            "volume": float(info.get("total_volume", 0.0)),
        })

    data_json = json.dumps({"customers": customers, "depots": depots,
                             "circle_radius_m": circle_radius_km * 1000.0})

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Depot Optimization Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin:0; padding:0; height:100%; }}
  #map {{ width:100%; height:100%; }}
  .legend {{ background:white; padding:8px 10px; font-family:sans-serif; font-size:13px;
             box-shadow:0 0 6px rgba(0,0,0,0.3); border-radius:4px; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
const DATA = {data_json};

const map = L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

const bounds = [];

DATA.customers.forEach(c => {{
  const color = c.assigned ? '#2b8cbe' : '#e31a1c';
  L.circleMarker([c.lat, c.lon], {{
    radius: c.radius_px, color: color, weight: 1, fillOpacity: 0.6
  }}).bindPopup(
    `<b>${{c.code}}</b><br/>Volume: ${{c.volume}}<br/>Depot: ${{c.depot}}`
  ).addTo(map);
  bounds.push([c.lat, c.lon]);
}});

DATA.depots.forEach(d => {{
  L.marker([d.lat, d.lon]).bindPopup(
    `<b>${{d.name}}</b><br/>Customers: ${{d.customers}}<br/>Total volume: ${{d.volume}}`
  ).addTo(map);
  L.circle([d.lat, d.lon], {{
    radius: DATA.circle_radius_m, color: '#238b45', fillOpacity: 0.05, weight: 1
  }}).addTo(map);
  bounds.push([d.lat, d.lon]);
}});

if (bounds.length > 0) {{ map.fitBounds(bounds, {{padding:[20,20]}}); }} else {{ map.setView([0,0], 2); }}

const legend = L.control({{position: 'bottomright'}});
legend.onAdd = function() {{
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML = `
    <div><span style="color:#2b8cbe;">&#9679;</span> Assigned customer (size ~ volume)</div>
    <div><span style="color:#e31a1c;">&#9679;</span> Unassigned customer</div>
    <div><span style="color:#238b45;">&#9711;</span> Depot service radius (${{Math.round(DATA.circle_radius_m/1000)}} km)</div>
    <div>&#128205; Depot location</div>`;
  return div;
}};
legend.addTo(map);
</script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ----------------------------------------------------------------------------
# Sample data generator (for trying the pipeline without real data)
# ----------------------------------------------------------------------------
def generate_sample_csv(path, n=500, seed=1):
    rng = np.random.default_rng(seed)
    hubs = np.array([[19.076, 72.877], [28.613, 77.209], [12.972, 77.594],
                      [22.573, 88.364], [17.385, 78.487]])  # Mumbai, Delhi, Blr, Kolkata, Hyd
    hub_idx = rng.integers(0, len(hubs), size=n)
    lat = hubs[hub_idx, 0] + rng.normal(0, 0.6, n)
    lon = hubs[hub_idx, 1] + rng.normal(0, 0.6, n)
    volume = rng.lognormal(mean=3.0, sigma=0.8, size=n).round(1)
    df = pd.DataFrame({
        "Customer_code": [f"C{i:05d}" for i in range(n)],
        "customer_geocode_lat": lat,
        "customer_geocode_long": lon,
        "volume": volume,
    })
    df.to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Depot location optimizer (KMeans + MILP)")
    ap.add_argument("--input", default="customers.csv", help="Input CSV path")
    ap.add_argument("--output-dir", default="output", help="Directory for all outputs")
    ap.add_argument("--candidates", type=int, default=12,
                     help="Number of candidate depot sites offered to the MILP")
    ap.add_argument("--max-radius", type=float, default=100.0,
                     help="Max service radius in km (feasibility cutoff for assignment)")
    ap.add_argument("--circle-radius", type=float, default=20.0,
                     help="Radius (km) drawn around each depot on the Leaflet map")
    ap.add_argument("--min-customers", type=int, default=1,
                     help="Hide depots serving fewer than this many customers in outputs")
    ap.add_argument("--max-depots", type=int, default=None,
                     help="Optional hard cap on number of depots the MILP may open")
    ap.add_argument("--capacity", type=float, default=None,
                     help="Optional max total volume per depot (uncapacitated if omitted)")
    ap.add_argument("--w-unassigned", type=float, default=None, help="Override coverage weight")
    ap.add_argument("--w-depot", type=float, default=None, help="Override depot-count weight")
    ap.add_argument("--w-dist", type=float, default=1.0, help="Distance-cost weight")
    ap.add_argument("--time-limit", type=int, default=120, help="CBC solver time limit (sec)")
    ap.add_argument("--solver-msg", action="store_true", help="Show CBC solver log")
    ap.add_argument("--generate-sample", action="store_true",
                     help="Write a synthetic demo CSV to --input and exit")
    args = ap.parse_args()

    if args.generate_sample:
        generate_sample_csv(args.input)
        print(f"Sample CSV written to {args.input}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. load
    df = pd.read_csv(args.input)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Input CSV is missing required column(s): {missing}")
    df = df.dropna(subset=["customer_geocode_lat", "customer_geocode_long", "volume"]).reset_index(drop=True)
    cust_latlon = df[["customer_geocode_lat", "customer_geocode_long"]].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    # ---- 2. candidate depots
    n_candidates = min(args.candidates, len(df))
    cand_latlon, _ = generate_candidates(cust_latlon, n_candidates)
    print(f"Generated {n_candidates} candidate depot sites via K-Means.")

    # ---- 3. naive baseline (vq) for comparison
    baseline_assign = baseline_vq_assignment(cust_latlon, cand_latlon)
    baseline_dist = haversine_km(
        cust_latlon[:, 0], cust_latlon[:, 1],
        cand_latlon[baseline_assign, 0], cand_latlon[baseline_assign, 1],
    )
    baseline_covered = int((baseline_dist <= args.max_radius).sum())
    print(f"Naive nearest-candidate baseline (vq): {baseline_covered}/{len(df)} "
          f"customers would be within {args.max_radius} km.")

    # ---- 4. distance matrix
    dist_km = distance_matrix_km(cust_latlon, cand_latlon)

    # ---- 5. MILP
    print("Solving facility-location MILP with PuLP/CBC ...")
    result = solve_milp(
        dist_km, volume, args.max_radius,
        max_depots=args.max_depots, capacity=args.capacity,
        w_unassigned=args.w_unassigned, w_depot=args.w_depot, w_dist=args.w_dist,
        time_limit_sec=args.time_limit, msg=args.solver_msg,
    )
    print(f"Solver status: {result['status']}  |  weights used: {result['weights']}")

    opened_all = np.where(result["opened"] == 1)[0]
    assign = result["assignment"]

    # ---- 6. results dataframe + depot summary
    out_df = build_results_df(df, cand_latlon, assign, dist_km, result["opened"])
    depot_summary = (
        out_df[out_df["depot_index"] != -1]
        .groupby("depot_index")
        .agg(n_customers=("Customer_code", "count"), total_volume=("volume", "sum"),
             avg_distance_km=("Distance_from_Customer_km", "mean"))
        .reset_index()
    )
    depot_summary["Depot_Name"] = depot_summary["depot_index"].apply(lambda j: f"Depot_{int(j) + 1}")
    depot_summary["Depot_Lat"] = cand_latlon[depot_summary["depot_index"], 0]
    depot_summary["Depot_Lon"] = cand_latlon[depot_summary["depot_index"], 1]

    opened_idx = [j for j in opened_all
                  if depot_summary.set_index("depot_index")["n_customers"].get(j, 0) >= args.min_customers]

    n_unassigned = int((assign == -1).sum())
    print(f"MILP result: {len(opened_all)} depot(s) opened, "
          f"{len(df) - n_unassigned}/{len(df)} customers served, {n_unassigned} unassigned.")
    print(depot_summary[["Depot_Name", "n_customers", "total_volume", "avg_distance_km"]]
          .to_string(index=False))

    # ---- 7 & 8. outputs
    excel_path = os.path.join(args.output_dir, "Customer_Depot_Analysis.xlsx")
    export_excel(out_df, depot_summary, excel_path)

    png_path = os.path.join(args.output_dir, "depot_bubble_chart.png")
    plot_bubble_chart(out_df, cand_latlon, opened_idx, png_path)

    html_path = os.path.join(args.output_dir, "depot_map.html")
    build_leaflet_html(out_df, cand_latlon, opened_idx, depot_summary,
                        args.circle_radius, html_path)

    print("\nOutputs written:")
    print(f"  - {excel_path}")
    print(f"  - {png_path}")
    print(f"  - {html_path}")


if __name__ == "__main__":
    main()
