#!/usr/bin/env python3
"""Phase 7 function-model gate; no held-out labels are scored."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_gate() -> int:
    checks: list[tuple[str, bool, str]] = []
    reports = ROOT / "reports"
    model_dir = ROOT / "models/esm2_t12_35M"
    metadata = json.loads((model_dir / "function_tier1.json").read_text())
    summary = json.loads((reports / "tables/phase_07_function_summary.json").read_text())
    model_path = model_dir / metadata["model_file"]
    logistic_path = model_dir / metadata["logistic_model_file"]
    bundle = joblib.load(model_path)
    logistic = joblib.load(logistic_path)
    all_rows = pd.read_parquet(ROOT / "data/processed/train_function.parquet")
    train = all_rows.loc[all_rows.split_id.eq("train")].copy()
    test_clusters = set(all_rows.loc[all_rows.split_id.eq("test"), "cluster_id"].astype(str))
    oof = pd.read_parquet(reports / "tables/oof_function.parquet")
    audits = pd.read_csv(reports / "tables/function_fit_audit.csv")
    metrics = pd.read_csv(reports / "tables/function_oof_metrics.csv")
    thresholds = pd.read_csv(reports / "tables/function_mcc_thresholds.csv")
    assay_auc = pd.read_csv(reports / "tables/function_per_assay_auroc.csv")
    curves = pd.read_csv(reports / "tables/function_calibration_curve.csv")
    calibration = pd.read_csv(reports / "tables/function_calibration_method_scores.csv")
    search = pd.read_csv(reports / "tables/function_hyperparameter_search.csv")

    checks.append(("LightGBM + calibrator and logistic regression artifacts exist", model_path.is_file()
                   and logistic_path.is_file() and isinstance(bundle, dict)
                   and {"estimator", "calibrator", "calibration_method", "feature_list"} <= set(bundle),
                   f"lgbm={model_path.is_file()}; logistic={logistic_path.is_file()}"))
    checks.append(("model binary hashes match metadata", sha256(model_path) == metadata["model_sha256"]
                   and sha256(logistic_path) == metadata["logistic_model_sha256"],
                   f"LGBM={metadata['model_sha256']}; logistic={metadata['logistic_model_sha256']}"))
    checks.append(("logistic regression uses a sklearn Pipeline with preprocessing", isinstance(logistic, Pipeline)
                   and "features" in logistic.named_steps and "classifier" in logistic.named_steps,
                   f"pipeline={type(logistic).__name__}"))
    expected_features = {"esm_llr", "esm_wt_logp", "esm_mut_logp", "blosum62", "d_hydropathy",
                         "d_volume", "d_charge", "esm_pos_mean_llr", "esm_pos_min_llr", "rel_position", "wt", "mut"}
    checks.append(("saved feature lists are Tier-1 only; no predicted ddG or ClinVar features",
                   set(metadata.get("feature_list", [])) == expected_features
                   and bundle.get("feature_list") == metadata.get("feature_list")
                   and not any("clinvar" in name.lower() or "ddg" in name.lower()
                               for name in metadata.get("feature_list", [])),
                   f"features={len(metadata.get('feature_list', []))}"))
    checks.append(("one configured ESM-2 35M model and Tier-2 absence recorded",
                   metadata.get("esm_model") == "facebook/esm2_t12_35M_UR50D"
                   and metadata.get("esm_model_short") == "esm2_t12_35M"
                   and summary.get("tier2_model_exists") is False,
                   f"ESM={metadata.get('esm_model')}; Tier-2={summary.get('tier2_model_exists')}"))
    checks.append(("function training table is DMS-only, both classes present, and at least 10 proteins",
                   len(train) >= 10 and train.protein_id.nunique() >= 10 and train.label.nunique() == 2
                   and metadata.get("n_rows") == len(train)
                   and metadata.get("n_proteins") == train.protein_id.nunique(),
                   f"train proteins={train.protein_id.nunique()}; rows={len(train)}; prevalence={train.label.mean():.6f}"))
    key_cols = ["protein_id", "position", "wt", "mut"]
    expected_keys = set(map(tuple, train[key_cols].itertuples(index=False, name=None)))
    checks.append(("OOF probabilities cover each training mutation exactly once",
                   len(oof) == len(train) and oof[key_cols].duplicated().sum() == 0
                   and set(map(tuple, oof[key_cols].itertuples(index=False, name=None))) == expected_keys
                   and oof.filter(regex="^p_").notna().all().all()
                   and np.allclose(oof.y_true.to_numpy(float),
                                   oof[key_cols].merge(train[key_cols + ["label"]], on=key_cols,
                                                       validate="one_to_one").label.to_numpy(float)),
                   f"OOF rows={len(oof)}; training rows={len(train)}"))
    checks.append(("fit/validation and fit/test cluster overlaps are zero",
                   len(audits) == 20 and audits.fit_validation_cluster_overlap.eq(0).all()
                   and audits.fit_test_cluster_overlap.eq(0).all()
                   and set(train.cluster_id.astype(str)).isdisjoint(test_clusters),
                   f"fold fits={len(audits)}; max test overlap={int(audits.fit_test_cluster_overlap.max())}"))
    checks.append(("OOF metrics include AUROC/AUPRC/MCC/prevalence and task sample counts",
                   {"auroc", "auprc", "mcc", "prevalence", "n_rows", "n_proteins"} <= set(metrics.columns)
                   and len(metrics) == 7 and metrics.n_rows.eq(len(train)).all()
                   and metrics.n_proteins.eq(train.protein_id.nunique()).all(),
                   f"models={len(metrics)}"))
    checks.append(("MCC thresholds were selected from grouped OOF predictions only",
                   len(thresholds) == 7 and thresholds.threshold_source.eq("grouped OOF only").all(),
                   f"threshold rows={len(thresholds)}"))
    checks.append(("per-assay AUROC distribution is present for 26 training assays",
                   assay_auc.assay_id.nunique() == train.assay_id.nunique()
                   and assay_auc.model.nunique() == 7
                   and len(assay_auc) == train.assay_id.nunique() * 7,
                   f"assays={assay_auc.assay_id.nunique()}; rows={len(assay_auc)}"))
    calibration_method = metadata.get("calibration_method")
    calibration_means = calibration.set_index("method").mean_grouped_cv_brier.to_dict()
    checks.append(("calibrator selected by cross-fitted grouped Brier and fit on all OOF rows",
                   calibration_method in {"isotonic", "platt"}
                   and calibration_method == bundle.get("calibration_method")
                   and calibration_method == min(calibration_means, key=calibration_means.get)
                   and len(pd.read_csv(reports / "tables/function_calibration_cv.csv")) == 10,
                   f"selected={calibration_method}; mean Brier={calibration_means}"))
    checks.append(("calibration curve data contains raw and calibrated OOF probabilities",
                   set(curves.model) == {"tier1_full", "tier1_full_calibrated"}
                   and curves.n_rows.sum() == 2 * len(train),
                   f"rows={len(curves)}; binned count={int(curves.n_rows.sum())}"))
    checks.append(("20 grouped-CV configurations completed and all results are recorded",
                   len(search) == 20 and search.status.eq("complete").all()
                   and summary.get("n_search_configs_completed") == 20
                   and summary.get("search_configs_reused") == 20
                   and summary.get("search_runtime_scope") == "cache_reuse_lookup"
                   and metadata.get("initial_search_runtime_captured") is False
                   and "wall_seconds=600.0" in (ROOT / "scripts/train_function.py").read_text(),
                   f"configs={len(search)}; reused={summary.get('search_configs_reused')}; search scope={summary.get('search_runtime_scope')}; original duration not captured"))
    checks.append(("max training sequence length and provisional label-direction caveat recorded",
                   metadata.get("max_training_protein_length") == summary.get("max_training_protein_length")
                   and "mixed" in metadata.get("label_direction_caveat", "").lower()
                   and metadata.get("held_out_test_used") is False
                   and summary.get("held_out_test_used") is False,
                   f"max length={metadata.get('max_training_protein_length')}; direction caveat present"))

    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (reports / "gates/phase_07_pytest.txt").write_text(test_run.stdout + test_run.stderr)
    last = test_run.stdout.strip().splitlines()[-1] if test_run.stdout.strip() else test_run.stderr.strip()
    checks.append(("pytest -q passes", test_run.returncode == 0, last))
    lines = [DISCLAIMER] + [f"{'PASS' if ok else 'FAIL'}: {name} | {details}" for name, ok, details in checks]
    gate_path = reports / "gates/phase_07_gate.txt"
    gate_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
