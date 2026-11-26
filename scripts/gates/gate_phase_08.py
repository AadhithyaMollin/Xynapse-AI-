#!/usr/bin/env python3
"""Phase 8 freeze gate; held-out evaluations must still be untouched."""
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
    keys = ["ddg_damaging", "ddg_stabilizing", "p_impact_high", "p_impact_low"]
    values_match = True
    for key in keys:
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
    checks.append(("frozen threshold and feature inventories match model metadata",
                   frozen.get("thresholds") == thresholds
                   and frozen.get("feature_lists", {}).get("stability")
                   == json.loads((ROOT / "models/esm2_t12_35M/stability_tier1.json").read_text())["feature_list"]
                   and frozen.get("feature_lists", {}).get("function")
                   == json.loads((ROOT / "models/esm2_t12_35M/function_tier1.json").read_text())["feature_list"],
                   f"thresholds frozen={frozen.get('thresholds')}"))
    checks.append(("freeze uses only the configured ESM-2 35M model",
                   frozen.get("esm_model") == "facebook/esm2_t12_35M_UR50D"
                   and frozen.get("esm_model_short") == "esm2_t12_35M",
                   f"{frozen.get('esm_model')}"))
    registry = ROOT / "reports/test_eval_registry.json"
    registry_empty = not registry.exists() or not json.loads(registry.read_text()).get("evaluations")
    test_reports = list((ROOT / "reports/tables").glob("test_*.csv"))
    checks.append(("no test evaluation has occurred before freeze publication", registry_empty and not test_reports,
                   f"registry_exists={registry.exists()}; test_report_files={len(test_reports)}"))

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
