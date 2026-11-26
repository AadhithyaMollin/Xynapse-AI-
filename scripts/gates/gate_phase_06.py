#!/usr/bin/env python3
"""Phase 6 stability-model gate; test labels are never scored here."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    metadata_path = model_dir / "stability_tier1.json"
    model_path = model_dir / "stability_tier1.joblib"
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads((reports / "tables/phase_06_stability_summary.json").read_text())
    split_table = pd.read_parquet(ROOT / "data/processed/train_ddg.parquet")
    train = split_table.loc[split_table.split_id.eq("train")].copy()
    test_clusters = set(split_table.loc[split_table.split_id.eq("test"), "cluster_id"].astype(str))
    oof = pd.read_parquet(reports / "tables/oof_ddg.parquet")
    audits = pd.read_csv(reports / "tables/stability_fit_audit.csv")
    metrics = pd.read_csv(reports / "tables/stability_oof_metrics.csv")
    baselines = pd.read_csv(reports / "tables/stability_baseline_cv.csv")
    sign_check = json.loads((reports / "tables/stability_baseline_sign_check.json").read_text())
    search = pd.read_csv(reports / "tables/stability_hyperparameter_search.csv")
    per_protein = pd.read_csv(reports / "tables/stability_per_protein_oof.csv")

    checks.append(("Tier-1 model metadata and ignored model binary exist", model_path.is_file()
                   and metadata_path.is_file() and metadata.get("tier") == "tier1",
                   f"model={model_path.is_file()}; metadata tier={metadata.get('tier')}"))
    checks.append(("model binary SHA-256 matches tracked metadata", model_path.is_file()
                   and sha256(model_path) == metadata.get("model_sha256"), metadata.get("model_sha256", "missing")))
    checks.append(("model metadata uses the configured ESM-2 35M model",
                   metadata.get("esm_model") == "facebook/esm2_t12_35M_UR50D"
                   and metadata.get("esm_model_short") == "esm2_t12_35M",
                   f"{metadata.get('esm_model')} / {metadata.get('esm_model_short')}"))
    expected_features = {"esm_llr", "esm_wt_logp", "esm_mut_logp", "blosum62", "d_hydropathy",
                         "d_volume", "d_charge", "esm_pos_mean_llr", "esm_pos_min_llr", "rel_position", "wt", "mut"}
    checks.append(("saved feature list is exactly Tier-1 and excludes ClinVar data",
                   set(metadata.get("feature_list", [])) == expected_features
                   and not any("clinvar" in name.lower() for name in metadata.get("feature_list", [])),
                   f"features={len(metadata.get('feature_list', []))}"))
    checks.append(("Tier-2 model was not created without sufficient complete Tier-2 inputs",
                   summary.get("tier2_model_exists") is False and not (model_dir / "stability_tier2.joblib").exists(),
                   f"tier2_model_exists={summary.get('tier2_model_exists')}"))
    checks.append(("at least 10 non-case-study training proteins and rows recorded",
                   train.protein_id.nunique() >= 10 and len(train) >= 10
                   and metadata.get("n_proteins") == train.protein_id.nunique()
                   and metadata.get("n_rows") == len(train),
                   f"proteins={train.protein_id.nunique()}; rows={len(train)}"))

    key_cols = ["protein_id", "position", "wt", "mut"]
    checks.append(("OOF predictions cover every training mutation exactly once",
                   len(oof) == len(train) and not oof.duplicated(key_cols).any()
                   and set(map(tuple, oof[key_cols].itertuples(index=False, name=None)))
                   == set(map(tuple, train[key_cols].itertuples(index=False, name=None)))
                   and oof.filter(like="pred_").notna().all().all(),
                   f"oof_rows={len(oof)}; training_rows={len(train)}"))
    checks.append(("OOF labels are the corresponding training labels only",
                   np.allclose(oof.y_true.to_numpy(float),
                               oof[key_cols].merge(train[key_cols + ["ddg"]], on=key_cols, validate="one_to_one").ddg.to_numpy(float)),
                   f"oof rows={len(oof)}"))
    checks.append(("no fit fold overlaps its validation or any held-out test cluster",
                   len(audits) == 15 and audits.fit_validation_cluster_overlap.eq(0).all()
                   and audits.fit_test_cluster_overlap.eq(0).all()
                   and set(train.cluster_id.astype(str)).isdisjoint(test_clusters),
                   f"fit rows audited={len(audits)}; fit-test overlap max={int(audits.fit_test_cluster_overlap.max())}"))
    checks.append(("stability metrics include n_proteins and n_rows for all ablations/baselines",
                   {"n_proteins", "n_rows", "spearman", "pearson", "rmse"} <= set(metrics.columns)
                   and len(metrics) == 7 and metrics.n_rows.eq(len(train)).all(),
                   f"metrics rows={len(metrics)}"))
    checks.append(("all four oriented baselines avoid an all-fold sign contradiction",
                   sign_check.get("training_only") is True
                   and not sign_check.get("all_folds_negative_baselines")
                   and set(baselines.baseline.unique()) == {"negative_blosum62", "abs_d_hydropathy", "abs_d_volume", "negative_esm_llr"},
                   f"baseline mean Spearman={sign_check.get('baseline_mean_spearman')}"))
    checks.append(("20 grouped-CV hyperparameter configurations completed under 10 minutes",
                   len(search) == 20 and search.status.eq("complete").all()
                   and int(summary.get("n_search_configs_completed", 0)) == 20
                   and float(summary.get("search_runtime_seconds", 601)) <= 600,
                   f"completed={int(search.status.eq('complete').sum())}; runtime={summary.get('search_runtime_seconds')}s"))
    checks.append(("per-protein Spearman table reports qualifying proteins", 
                   len(per_protein) > 0 and per_protein.n_qualifying_proteins.notna().all(),
                   f"models={per_protein.model.nunique()}; qualifying={per_protein.groupby('model').protein_id.nunique().min()}..{per_protein.groupby('model').protein_id.nunique().max()}"))
    proteins = pd.read_parquet(ROOT / "data/interim/train_proteins.parquet")
    expected_max = int(proteins.loc[proteins.protein_id.isin(train.protein_id.unique()), "length"].max())
    checks.append(("metadata records true maximum training protein length and no test evaluation",
                   metadata.get("max_training_protein_length") == expected_max
                   and metadata.get("held_out_test_used") is False
                   and summary.get("held_out_test_used") is False,
                   f"max_length={expected_max}; held_out_test_used={metadata.get('held_out_test_used')}"))
    checks.append(("summary records complete finite OOF coverage and no model tier mixing",
                   summary.get("n_oof_rows") == len(train) and summary.get("n_oof_missing") == 0
                   and summary.get("esm_model_short") == "esm2_t12_35M",
                   f"oof={summary.get('n_oof_rows')}; missing={summary.get('n_oof_missing')}"))

    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (reports / "gates/phase_06_pytest.txt").write_text(test_run.stdout + test_run.stderr)
    last = test_run.stdout.strip().splitlines()[-1] if test_run.stdout.strip() else test_run.stderr.strip()
    checks.append(("pytest -q passes", test_run.returncode == 0, last))

    lines = [DISCLAIMER] + [f"{'PASS' if ok else 'FAIL'}: {name} | {details}" for name, ok, details in checks]
    path = reports / "gates/phase_06_gate.txt"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
