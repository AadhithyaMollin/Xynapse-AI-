# Xynapse-AI-

<!-- XYNAPSE-ML:BEGIN -->
## ML mutation predictions

Predictions are computational hypotheses, not experimental findings.

The dashboard predicts single-substitution stability (`ddg_pred`, kcal/mol; positive means destabilizing) and function impact (`p_impact`), with ranked classes and evidence-linked mechanism hints. The available model uses ESM-2 `facebook/esm2_t12_35M_UR50D` (`esm2_t12_35M`) and Tier-1 features. The current bundle provides predictions for PTEN (`P60484`) and TP53 (`P04637`) only; the dashboard labels tracked example predictions `DEMO - NOT REAL`. There is no Tier-2 model in this build.

### Setup and dashboard

See [`docs/DEPLOY.md`](docs/DEPLOY.md) for a fresh-clone setup and optional Streamlit Community Cloud steps. To launch locally after installing `requirements.txt`:

```sh
streamlit run app/app.py
```

The dashboard reads cached Parquet predictions, preferring `data/processed/` and falling back to `demo_data/`. Set `XYNAPSE_DATA_DIR` to select another directory containing `predictions.parquet`.

### Reproduce saved artifacts

Run the full unit suite and static/user-facing audits:

```sh
python -m pytest -q
python scripts/make_reports.py
python scripts/leakage_audit.py
python scripts/claims_audit.py
```

`make_reports.py` regenerates the derived report tables and model cards from the saved local predictions, OOF tables, model metadata, and test-evaluation registry. Source-acquisition inspection artifacts, frozen one-time test tables, and local-only raw/processed Parquet inputs are retained as saved evidence rather than rebuilt from unavailable raw files. Data acquisition and model fitting scripts are documented with their inputs in `docs/data_cards/` and `docs/model_cards/`.

See [`docs/limitations.md`](docs/limitations.md), [`docs/label_definitions.md`](docs/label_definitions.md), [`docs/DEPLOY.md`](docs/DEPLOY.md), and [`docs/SHARING_ARTIFACTS.md`](docs/SHARING_ARTIFACTS.md). Large datasets and model binaries are not part of this Git repository.

Predictions are computational hypotheses, not experimental findings.
<!-- XYNAPSE-ML:END -->
