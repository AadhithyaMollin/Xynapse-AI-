#!/usr/bin/env python3
"""Phase 8 gate for frozen models and the completed one-time test evaluations."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
EXPECTED_EVALUATIONS = {
    "stability_cluster_test",
    "function_cluster_test",
    "case_study_dms_P60484",
    "case_study_dms_P04637",
    "clinvar_external_test",
}
EXPECTED_TEST_OUTPUTS = (
    "test_stability_cluster.csv",
    "test_stability_per_protein.csv",
    "test_function_cluster.csv",
    "test_function_per_assay.csv",
    "test_case_study_P60484.csv",
    "test_case_study_P60484_variants.csv",
    "test_case_study_P60484_worst_disagreements.csv",
    "test_case_study_P04637.csv",
    "test_case_study_P04637_variants.csv",
    "test_case_study_P04637_worst_disagreements.csv",
    "test_clinvar_external.csv",
    "test_clinvar_counts_by_gene_label.csv",
)


def run_gate() -> int:
    from xynapse.freeze import verify_frozen

    checks: list[tuple[str, bool, str]] = []
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    thresholds = config["thresholds"]
    evidence = pd.read_csv(ROOT / "reports/tables/threshold_evidence.csv")
    ddg = pd.read_parquet(ROOT / "reports/tables/oof_ddg.parquet")
    function = pd.read_parquet(ROOT / "reports/tables/oof_function.parquet")
    frozen_path = ROOT / "models/esm2_t12_35M/FROZEN.json"
    frozen = json.loads(frozen_path.read_text())
    ok_hashes, hash_problems = verify_frozen(frozen_path, ROOT / "models/esm2_t12_35M")

    evidence_values = evidence.set_index("threshold").value.to_dict()
    threshold_keys = ["ddg_damaging", "ddg_stabilizing", "p_impact_high", "p_impact_low"]
    values_match = True
    for key in threshold_keys:
        expected, actual = evidence_values[key], thresholds[key]
        if pd.isna(expected) and actual is None:
            continue
        if expected is None or actual is None or not np.isclose(float(expected), float(actual), rtol=0, atol=1e-10):
            values_match = False
    checks.append(("four config thresholds match saved OOF evidence", values_match,
                   f"thresholds={thresholds}"))
    checks.append(("stability thresholds are derived from 45,411 grouped OOF rows",
                   len(ddg) == 45411 and set(ddg.cv_fold.unique()) == {0, 1, 2, 3, 4}
                   and evidence.loc[evidence.threshold.str.startswith("ddg_"), "n_rows"].eq(len(ddg)).all(),
                   f"rows={len(ddg)}; folds={sorted(ddg.cv_fold.unique().tolist())}"))
    stabilizing = evidence.set_index("threshold").loc["ddg_stabilizing"]
    checks.append(("beneficial threshold is disabled only below the 50-positive minimum",
                   (int(stabilizing.n_positive) >= 50 and thresholds["ddg_stabilizing"] is not None)
                   or (int(stabilizing.n_positive) < 50 and thresholds["ddg_stabilizing"] is None),
                   f"stabilizing positives={int(stabilizing.n_positive)}; cutoff={thresholds['ddg_stabilizing']}"))
    low_row = evidence.set_index("threshold").loc["p_impact_low"]
    low_valid = (thresholds["p_impact_low"] is None and low_row.n_assigned_negative == 0) or (
        thresholds["p_impact_low"] is not None and low_row.observed_npv >= low_row.target_npv)
    checks.append(("p_impact_low is null when OOF NPV 0.90 is unreachable",
                   low_valid and int(low_row.n_rows) == len(function),
                   f"cutoff={thresholds['p_impact_low']}; target NPV={low_row.target_npv}; observed={low_row.observed_npv}"))
    checks.append(("function thresholds use calibrated grouped OOF predictions",
                   len(function) == 71962 and function.p_impact_calibrated.notna().all()
                   and int(evidence.set_index("threshold").loc["p_impact_high", "n_rows"]) == len(function),
                   f"function OOF rows={len(function)}"))
    checks.append(("freeze model inventory hashes all artifacts and still matches disk",
                   ok_hashes and len(frozen.get("model_files_sha256", {})) == 5 and not hash_problems,
                   f"files={len(frozen.get('model_files_sha256', {}))}; problems={hash_problems}"))
    stability_meta = json.loads((ROOT / "models/esm2_t12_35M/stability_tier1.json").read_text())
    function_meta = json.loads((ROOT / "models/esm2_t12_35M/function_tier1.json").read_text())
    checks.append(("frozen threshold and feature inventories match model metadata",
                   frozen.get("thresholds") == thresholds
                   and frozen.get("feature_lists", {}).get("stability") == stability_meta["feature_list"]
                   and frozen.get("feature_lists", {}).get("function") == function_meta["feature_list"],
                   f"thresholds frozen={frozen.get('thresholds')}"))
    checks.append(("freeze uses only the configured ESM-2 35M model",
                   frozen.get("esm_model") == "facebook/esm2_t12_35M_UR50D"
                   and frozen.get("esm_model_short") == "esm2_t12_35M",
                   f"{frozen.get('esm_model')}"))

    registry_path = ROOT / "reports/test_eval_registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    evaluations = registry.get("evaluations", {})
    registry_complete = set(evaluations) == EXPECTED_EVALUATIONS and all(
        evaluations[name].get("status") == "completed" for name in EXPECTED_EVALUATIONS
    ) and not registry.get("failures")
    checks.append(("the five named test evaluations are recorded exactly once and completed",
                   registry_complete,
                   f"names={sorted(evaluations)}; statuses={{{', '.join(f'{k}: {v.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115))}' for k, v in sorted(evaluations.items()))}}}"))
    outputs_dir = ROOT / "reports/tables"
    missing_outputs = [name for name in EXPECTED_TEST_OUTPUTS if not (outputs_dir / name).is_file()]
    empty_outputs = [name for name in EXPECTED_TEST_OUTPUTS if (outputs_dir / name).is_file()
                     and (outputs_dir / name).stat().st_size == 0]
    checks.append(("all one-time test evaluation tables are saved and non-empty",
                   not missing_outputs and not empty_outputs,
                   f"missing={missing_outputs}; empty={empty_outputs}"))
    summary_path = outputs_dir / "phase_08_test_evaluation_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    checks.append(("saved evaluation summary covers all five completed evaluations",
                   set(summary.get("evaluations", {})) == EXPECTED_EVALUATIONS,
                   f"names={sorted(summary.get('evaluations', {}))}"))

    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (ROOT / "reports/gates/phase_08_pytest.txt").write_text(test_run.stdout + test_run.stderr)
    last = test_run.stdout.strip().splitlines()[-1] if test_run.stdout.strip() else test_run.stderr.strip()
    checks.append(("pytest -q passes", test_run.returncode == 0, last))
    lines = [DISCLAIMER] + [f"{'PASS' if passed else 'FAIL'}: {name} | {details}" for name, passed, details in checks]
    (ROOT / "reports/gates/phase_08_gate.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
