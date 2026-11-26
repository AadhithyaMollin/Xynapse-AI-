#!/usr/bin/env python3
"""Data gate for complete Tier-1 features across all current proteins."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def run_gate() -> int:
    from xynapse.features import model_short_name, read_feature_model
    from xynapse.schema import MUT_ESM_COLS, MUT_KEYS, MUT_PHYS_COLS, TIER1_EXTRA_COLS
    from xynapse.validate import validate_table

    proteins = pd.read_parquet(ROOT / "data/interim/train_proteins.parquet")
    features_path = ROOT / "data/processed/features_tier1.parquet"
    features = pd.read_parquet(features_path)
    full_model, short_model = read_feature_model(features_path)
    expected_short = model_short_name(full_model)
    report: list[tuple[str, bool, str]] = []
    problems = validate_table(features, "features_tier1", proteins)
    report.append(("Tier-1 table validates against the contract", not problems, "; ".join(problems) or "validator PASS"))
    expected_rows = int((proteins["length"] * 19).sum())
    report.append(("every protein has exactly 19 x L mutation rows",
                   len(features) == expected_rows and set(features.protein_id) == set(proteins.protein_id),
                   f"rows={len(features)}; expected={expected_rows}; proteins={features.protein_id.nunique()}"))
    tier1_columns = MUT_ESM_COLS + MUT_PHYS_COLS + TIER1_EXTRA_COLS
    no_nan = not features[tier1_columns].isna().any().any()
    finite = bool(np.isfinite(features[tier1_columns].to_numpy(dtype=float)).all())
    report.append(("Tier-1 columns have no missing or non-finite values", no_nan and finite,
                   f"columns={len(tier1_columns)}"))
    one_model = full_model == "facebook/esm2_t12_35M_UR50D" and short_model == expected_short == "esm2_t12_35M"
    report.append(("features metadata uses one configured ESM-2 35M model", one_model,
                   f"full={full_model}; short={short_model}"))

    runtime = pd.read_csv(ROOT / "reports/tables/phase_04_feature_runtime.csv")
    manifest = pd.read_csv(ROOT / "reports/tables/training_protein_manifest.csv")
    request_path = ROOT / "reports/teammate_request.md"
    direction = pd.read_csv(ROOT / "reports/tables/label_direction_check.csv")
    summary = json.loads((ROOT / "reports/tables/phase_04_feature_summary.json").read_text())
    report.append(("per-protein runtime log and availability manifest cover every protein",
                   set(runtime.protein_id) == set(proteins.protein_id)
                   and set(manifest.protein_id) == set(proteins.protein_id),
                   f"runtime rows={len(runtime)}; manifest rows={len(manifest)}"))
    report.append(("BIO-1/BIO-2 request includes required Tier-2 and msa_logodds fields",
                   request_path.is_file() and "msa_logodds" in request_path.read_text()
                   and "pos_structure" in request_path.read_text(),
                   f"request={request_path.relative_to(ROOT)}"))
    mean_rho = float(direction["spearman_project_label_esm_llr"].mean())
    report.append(("full function-label ESM direction check has negative mean Spearman",
                   len(direction) > 0 and np.isfinite(mean_rho) and mean_rho < 0,
                   f"assays={len(direction)}; mean={mean_rho:.9f}"))
    report.append(("summary agrees with output row/model metadata and measured runtime",
                   summary["n_feature_rows"] == len(features)
                   and summary["esm_model"] == full_model
                   and summary["esm_model_short"] == short_model
                   and summary["runtime_seconds"] > 0,
                   f"runtime={summary['runtime_seconds']:.2f}s; caches reused={summary['cache_reused_count']}"))

    cache_suffix = full_model.rsplit("/", 1)[-1]
    cache_dir = ROOT / "data/interim/esm"
    cache_files = sorted(cache_dir.glob(f"*_{cache_suffix}.parquet"))
    cache_models = set()
    for path in cache_files:
        import pyarrow.parquet as pq
        metadata = pq.read_schema(path).metadata or {}
        raw = metadata.get(b"esm_model")
        if raw is not None:
            cache_models.add(raw.decode("utf-8"))
    report.append(("per-protein ESM caches contain one ESM model name",
                   bool(cache_files) and cache_models == {full_model},
                   f"cache files={len(cache_files)}; models={sorted(cache_models)}"))

    pytest_path = ROOT / "reports/gates/phase_04_pytest.txt"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                            capture_output=True, text=True)
    pytest_path.write_text(result.stdout + result.stderr)
    report.append(("pytest -q passes", result.returncode == 0,
                   result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr.strip()))

    lines = [DISCLAIMER] + [f"{'PASS' if ok else 'FAIL'}: {name} | {details}" for name, ok, details in report]
    gate_path = ROOT / "reports/gates/phase_04_gate.txt"
    gate_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(ok for _, ok, _ in report) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
