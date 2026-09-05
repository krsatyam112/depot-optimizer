# Going live

`depot_streamlit_app.py` is a browser UI (upload CSV → set parameters → Run)
built on top of the same `depot_optimizer.py` logic — no duplicated code, so
fixes to one apply to the other.

## Try it locally first

```bash
pip install -r requirements.txt
streamlit run depot_streamlit_app.py
```

Opens at `http://localhost:8501`. Check the "Use a generated sample dataset"
box in the sidebar to try it without your own CSV.

## Free option: Streamlit Community Cloud (simplest)

1. Push this folder to a **public** (or private, on paid plans) GitHub repo.
   Make sure it contains: `depot_streamlit_app.py`, `depot_optimizer.py`,
   `requirements.txt`.
2. Go to https://share.streamlit.io → sign in with GitHub → **New app**.
3. Pick the repo/branch, set **Main file path** to `depot_streamlit_app.py`.
4. Click **Deploy**. You get a public URL like
   `https://<something>.streamlit.app` in a couple of minutes.
5. Any push to the branch auto-redeploys.

Notes / limits:
- Free tier apps sleep after inactivity and wake on the next visit (~30s cold
  start).
- CBC (PuLP's bundled solver) runs fine there — no extra system packages
  needed.
- If you need it always-warm, or private with SSO, that needs a paid tier or
  one of the options below.

## Other options if you outgrow the free tier

- **Hugging Face Spaces** (Streamlit SDK) — same idea, free tier, generous
  compute, good if Streamlit Cloud is unavailable in your region.
- **Render / Railway** — `Dockerfile` or a native Python web service running
  `streamlit run depot_streamlit_app.py --server.port $PORT
  --server.address 0.0.0.0`. Gives you a real always-on box, small monthly cost.
- **Your own server / VM** — run the same `streamlit run` command behind
  nginx + a systemd service, or in a Docker container, for full control.

## Minimal Dockerfile (for Render/Railway/your own server)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY depot_optimizer.py depot_streamlit_app.py ./
EXPOSE 8501
CMD ["streamlit", "run", "depot_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## A note on this build

I couldn't install `streamlit` or `pulp` in the sandbox I used to write this
(no network access there), so I syntax-checked `depot_streamlit_app.py` and
reused the already-tested `depot_optimizer.py` functions directly, but I
have not run the actual Streamlit UI end-to-end. Please run it locally
(`streamlit run depot_streamlit_app.py`) before deploying, and let me know
if anything needs a fix.
