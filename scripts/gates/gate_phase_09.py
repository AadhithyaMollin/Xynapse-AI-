#!/usr/bin/env python3
"""Data gate for Phase 9 ranked case-study predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
EVIDENCE_DISCLAIMER = "Computational prediction; not experimental evidence."


def run_gate() -> int:
    from xynapse.features import read_feature_model
    from xynapse.freeze import verify_frozen
    from xynapse.mechanism import apply_mechanism_rules
    from xynapse.schema import MUT_KEYS
    from xynapse.validate import validate_table

    checks: list[tuple[str, bool, str]] = []
    processed = ROOT / "data/processed"
    predictions = pd.read_parquet(processed / "predictions.parquet")
    annotations = pd.read_parquet(processed / "mut_features.parquet")
    proteins = pd.read_parquet(processed / "proteins.parquet")
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    rules = yaml.safe_load((ROOT / "configs/mechanism_rules.yaml").read_text())
    model_name, model_short = read_feature_model(processed / "features_tier1.parquet")
    problems = validate_table(predictions, "predictions", proteins=proteins)
    checks.append(("validator passes for complete prediction contract", not problems,
                   "; ".join(problems) if problems else f"rows={len(predictions)}"))

    expected_counts = proteins.set_index("protein_id")["length"].mul(19).to_dict()
    actual_counts = predictions.groupby("protein_id").size().to_dict()
    checks.append(("prediction rows equal 19 times protein length for every protein",
                   set(actual_counts) == set(expected_counts)
                   and all(actual_counts.get(pid) == expected_counts[pid] for pid in actual_counts),
                   f"actual={actual_counts}; expected={expected_counts}"))
    stability_meta = json.loads((ROOT / "models/esm2_t12_35M/stability_tier1.json").read_text())
    function_meta = json.loads((ROOT / "models/esm2_t12_35M/function_tier1.json").read_text())
    expected_ddg_confidence = {
        protein_id: "low" if int(length) > int(stability_meta["max_training_protein_length"]) else "medium"
        for protein_id, length in proteins.set_index("protein_id")["length"].items()
    }
    expected_function_confidence = {
        protein_id: "low" if int(length) > int(function_meta["max_training_protein_length"]) else "medium"
        for protein_id, length in proteins.set_index("protein_id")["length"].items()
    }
    confidence_ok = all(
        set(group.ddg_confidence) == {expected_ddg_confidence[protein_id]}
        and set(group.confidence) == {expected_function_confidence[protein_id]}
        for protein_id, group in predictions.groupby("protein_id")
    )
    checks.append(("confidence fields follow frozen training-length limits for Tier 1",
                   confidence_ok,
                   f"ddg={expected_ddg_confidence}; function={expected_function_confidence}"))
    finite_ok = (
        predictions[["ddg_pred", "p_impact"]].notna().all().all()
        and np.isfinite(predictions[["ddg_pred", "p_impact"]].to_numpy(float)).all()
    )
    checks.append(("ddG and function probabilities are finite for all mutations",
                   bool(finite_ok), f"rows={len(predictions)}"))
    checks.append(("all rows have a rank class and required evidence disclaimer",
                   predictions.rank_class.notna().all()
                   and predictions.evidence_text.fillna("").str.contains(EVIDENCE_DISCLAIMER, regex=False).all()
                   and predictions.evidence_text.fillna("").str.contains(DISCLAIMER, regex=False).all(),
                   f"rows={len(predictions)}"))
    checks.append(("all prediction rows use Tier 1 and one configured ESM-2 model",
                   set(predictions.tier) == {"tier1"} and set(predictions.esm_model) == {model_name}
                   and model_name == "facebook/esm2_t12_35M_UR50D" and model_short == "esm2_t12_35M",
                   f"tiers={sorted(predictions.tier.unique())}; models={sorted(predictions.esm_model.unique())}"))

    joined = predictions.merge(annotations, on=MUT_KEYS, how="left", validate="one_to_one", suffixes=("", "_annotation"))
    derived = [apply_mechanism_rules(row, config["thresholds"], config.get("mechanism", {}), rules)
               for row in joined.to_dict(orient="records")]
    checks.append(("mechanism flags re-derive exactly from mutation annotations and frozen rules",
                   derived == predictions.mechanism_flags.tolist(),
                   f"rows_checked={len(derived)}"))
    counts_path = ROOT / "reports/tables/rank_class_counts.csv"
    counts = pd.read_csv(counts_path) if counts_path.is_file() else pd.DataFrame()
    reconstructed = (predictions.groupby(["protein_id", "rank_class"]).size().rename("n_mutations").reset_index())
    counts_ok = not counts.empty and set(counts.columns) == set(reconstructed.columns)
    if counts_ok:
        counts_ok = reconstructed.sort_values(list(reconstructed.columns)).reset_index(drop=True).equals(
            counts.sort_values(list(counts.columns)).reset_index(drop=True)
        )
    checks.append(("rank class count CSV matches prediction rows", counts_ok,
                   f"rows={len(counts)}"))

    frozen_path = ROOT / "models/esm2_t12_35M/FROZEN.json"
    ok_hashes, hash_problems = verify_frozen(frozen_path, frozen_path.parent)
    frozen = json.loads(frozen_path.read_text())
    checks.append(("frozen artifacts remain unchanged and configured thresholds match freeze",
                   ok_hashes and not hash_problems and frozen.get("thresholds") == config["thresholds"],
                   f"problems={hash_problems}"))
    summary_path = ROOT / "reports/tables/phase_09_prediction_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    checks.append(("prediction summary exists and reports the complete case-study set",
                   summary.get("n_rows") == len(predictions)
                   and summary.get("n_proteins") == predictions.protein_id.nunique()
                   and summary.get("disclaimer") == DISCLAIMER,
                   f"n_rows={summary.get('n_rows')}; n_proteins={summary.get('n_proteins')}"))

    import subprocess
    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (ROOT / "reports/gates/phase_09_pytest.txt").write_text(test_run.stdout + test_run.stderr)
    last = test_run.stdout.strip().splitlines()[-1] if test_run.stdout.strip() else test_run.stderr.strip()
    checks.append(("pytest -q passes", test_run.returncode == 0, last))
    lines = [DISCLAIMER] + [f"{'PASS' if passed else 'FAIL'}: {name} | {detail}" for name, passed, detail in checks]
    (ROOT / "reports/gates/phase_09_gate.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
