#!/usr/bin/env python3
"""
depot_streamlit_app.py
=======================
Browser-based front end for depot_optimizer.py — upload a CSV, set
parameters, click Run, get results in the page (table, chart, interactive
map) plus a download button for the Excel workbook.

This file contains NO new optimization logic — it only imports and calls
the functions already written (and tested) in depot_optimizer.py, so the
two stay in sync automatically.

Run locally:
    streamlit run depot_streamlit_app.py

Deploy for free:
    Push this repo to GitHub, then deploy on https://share.streamlit.io
    (Streamlit Community Cloud) pointing at this file. See DEPLOY.md.
"""
import io

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

import depot_optimizer as do

st.set_page_config(page_title="Depot Optimizer", layout="wide")
st.title("Depot Optimization")
st.caption(
    "K-Means candidate generation (scipy) + a MILP (PuLP/CBC) that jointly "
    "maximizes coverage, minimizes depot count, and minimizes volume-weighted distance."
)

# ----------------------------------------------------------------------------
# Sidebar — inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    use_sample = st.checkbox("Use a generated sample dataset instead", value=False)

    st.header("2. Parameters")
    candidates = st.number_input("Candidate depot sites", min_value=1, max_value=200, value=12, step=1)
    max_radius = st.number_input("Max service radius (km)", min_value=1.0, max_value=5000.0, value=100.0, step=5.0)
    circle_radius = st.number_input("Map circle radius (km)", min_value=1.0, max_value=5000.0, value=20.0, step=1.0)
    min_customers = st.number_input("Hide depots below this customer count", min_value=0, value=1, step=1)

    with st.expander("Advanced (optional overrides)"):
        max_depots = st.number_input("Hard cap on # depots (0 = no cap)", min_value=0, value=0, step=1)
        capacity = st.number_input("Max volume per depot (0 = uncapacitated)", min_value=0.0, value=0.0, step=100.0)
        w_unassigned = st.number_input("Coverage weight override (0 = auto)", min_value=0.0, value=0.0)
        w_depot = st.number_input("Depot-count weight override (0 = auto)", min_value=0.0, value=0.0)
        w_dist = st.number_input("Distance weight", min_value=0.0, value=1.0)
        time_limit = st.number_input("Solver time limit (sec)", min_value=5, value=120, step=5)

    run_btn = st.button("Run analysis", type="primary")

st.markdown(
    "Required CSV columns: `Customer_code`, `customer_geocode_lat`, "
    "`customer_geocode_long`, `volume`."
)

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
df = None
if use_sample:
    buf = io.StringIO()
    tmp_path = "/tmp/_depot_sample.csv"
    do.generate_sample_csv(tmp_path, n=400)
    df = pd.read_csv(tmp_path)
elif uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

if df is not None:
    missing = [c for c in do.REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {missing}")
        st.stop()
    df = df.dropna(subset=["customer_geocode_lat", "customer_geocode_long", "volume"]).reset_index(drop=True)
    st.write(f"Loaded **{len(df)}** customers.")
    st.dataframe(df.head(10), use_container_width=True)

# ----------------------------------------------------------------------------
# Run pipeline
# ----------------------------------------------------------------------------
if run_btn:
    if df is None:
        st.warning("Upload a CSV (or check 'Use a generated sample dataset') first.")
        st.stop()

    cust_latlon = df[["customer_geocode_lat", "customer_geocode_long"]].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    with st.spinner("Generating candidate depot sites (K-Means)..."):
        n_candidates = min(int(candidates), len(df))
        cand_latlon, _ = do.generate_candidates(cust_latlon, n_candidates)

    baseline_assign = do.baseline_vq_assignment(cust_latlon, cand_latlon)
    baseline_dist = do.haversine_km(
        cust_latlon[:, 0], cust_latlon[:, 1],
        cand_latlon[baseline_assign, 0], cand_latlon[baseline_assign, 1],
    )
    baseline_covered = int((baseline_dist <= max_radius).sum())
    st.info(
        f"Naive nearest-candidate baseline (vq): {baseline_covered}/{len(df)} "
        f"customers would be within {max_radius} km (for comparison only)."
    )

    dist_km = do.distance_matrix_km(cust_latlon, cand_latlon)

    with st.spinner("Solving facility-location MILP with PuLP/CBC... this can take a while"):
        try:
            result = do.solve_milp(
                dist_km, volume, max_radius,
                max_depots=(int(max_depots) or None),
                capacity=(capacity or None),
                w_unassigned=(w_unassigned or None),
                w_depot=(w_depot or None),
                w_dist=w_dist,
                time_limit_sec=int(time_limit),
                msg=False,
            )
        except Exception as e:
            st.error(f"Solver failed: {e}")
            st.stop()

    st.success(f"Solver status: {result['status']}")

    opened_all = np.where(result["opened"] == 1)[0]
    assign = result["assignment"]

    out_df = do.build_results_df(df, cand_latlon, assign, dist_km, result["opened"])
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

    n_by_idx = depot_summary.set_index("depot_index")["n_customers"].to_dict()
    opened_idx = [j for j in opened_all if n_by_idx.get(j, 0) >= int(min_customers)]

    n_unassigned = int((assign == -1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Depots opened", len(opened_all))
    c2.metric("Customers served", f"{len(df) - n_unassigned}/{len(df)}")
    c3.metric("Unassigned", n_unassigned)

    st.subheader("Depot summary")
    st.dataframe(
        depot_summary[["Depot_Name", "n_customers", "total_volume", "avg_distance_km",
                        "Depot_Lat", "Depot_Lon"]],
        use_container_width=True,
    )

    st.subheader("Customer assignments (first 200 rows)")
    st.dataframe(out_df.head(200), use_container_width=True)

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Customer_Assignments")
        depot_summary.to_excel(writer, index=False, sheet_name="Depot_Summary")
    st.download_button(
        "Download results (Excel)", data=excel_buf.getvalue(),
        file_name="Customer_Depot_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Bubble chart (bubble size ~ volume)")
    png_path = "/tmp/_depot_chart.png"
    do.plot_bubble_chart(out_df, cand_latlon, opened_idx, png_path)
    st.image(png_path, use_container_width=True)

    st.subheader("Interactive map")
    html_path = "/tmp/_depot_map.html"
    do.build_leaflet_html(out_df, cand_latlon, opened_idx, depot_summary, circle_radius, html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        st_html(f.read(), height=600, scrolling=False)
