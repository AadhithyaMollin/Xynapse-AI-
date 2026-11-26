# Dashboard deployment

Predictions are computational hypotheses, not experimental findings.

## Run locally from a fresh clone

1. Install Python 3.12 and clone the shared repository.
2. From the repository root, create and activate a virtual environment:
   ```sh
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the declared packages:
   ```sh
   python -m pip install -r requirements.txt
   ```
4. Start the cached-Parquet dashboard:
   ```sh
   streamlit run app/app.py
   ```
5. The app uses `data/processed/predictions.parquet` when present, otherwise the tracked `demo_data/predictions.parquet` bundle. The demo selection is explicitly marked `DEMO - NOT REAL`. To select a different prediction directory, set `XYNAPSE_DATA_DIR` to a directory containing `predictions.parquet` before starting Streamlit.
6. The demo heatmap includes the Tier-1 `esm_llr` feature. The 3D view is opt-in and may make a GET request to AlphaFold DB if a structure is not cached; cached demo PDB files are in `demo_data/structures/`.

The demo bundle is for dashboard operation and is not a model distribution. Reproducing inference requires the separately distributed model artifacts and their matching `FROZEN.json`; see [`SHARING_ARTIFACTS.md`](SHARING_ARTIFACTS.md). Do not infer model availability from the presence of demo predictions.

## Streamlit Community Cloud (steps only; not run here)

1. Publish the repository branch containing the dashboard and `demo_data/` bundle, then select that repository and branch in Streamlit Community Cloud.
2. Set the app entry point to `app/app.py`.
3. Use the repository's `requirements.txt` as the dependency file and select a Python runtime compatible with Python 3.12.
4. Leave `XYNAPSE_DATA_DIR` unset to use the tracked demo fallback, or configure it only when the deployment environment provides a compatible directory containing `predictions.parquet`.
5. Confirm the demo warning and disclaimer appear after deployment. Do not upload raw datasets or model binaries into the source repository.
6. If the optional 3D view is enabled, permit outbound GET access only to `alphafold.ebi.ac.uk`; otherwise keep the view unavailable and use the cached demo PDB files.

Predictions are computational hypotheses, not experimental findings.
