#!/usr/bin/env python3
"""Reproduce the frozen P60484 Phase 9 predictions from a fresh output directory."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def run_gate() -> int:
    from xynapse.pipeline import run_pipeline
    from xynapse.schema import MUT_KEYS

    checks: list[tuple[str, bool, str]] = []
    phase9 = pd.read_parquet(ROOT / "data/processed/predictions.parquet")
    expected = phase9.loc[phase9.protein_id.eq("P60484")].sort_values(MUT_KEYS).reset_index(drop=True)
    with tempfile.TemporaryDirectory(prefix="xynapse-phase10-p60484-") as temp_dir:
        actual, summary = run_pipeline("P60484", esm_model="dev", out=temp_dir, project_root=ROOT)
        actual = actual.sort_values(MUT_KEYS).reset_index(drop=True)
        output_files = ["predictions.parquet", "features_tier1.parquet", "pipeline_summary.json"]
        files_exist = all((Path(temp_dir) / name).is_file() for name in output_files)
        row_keys_match = expected[MUT_KEYS].equals(actual[MUT_KEYS])
        max_ddg_delta = float(np.max(np.abs(expected.ddg_pred.to_numpy(float) - actual.ddg_pred.to_numpy(float))))
        max_p_delta = float(np.max(np.abs(expected.p_impact.to_numpy(float) - actual.p_impact.to_numpy(float))))
        checks.append(("fresh pipeline output contains all expected artifacts", files_exist,
                       f"files={output_files}"))
        checks.append(("pipeline mutation keys match Phase 9 P60484 rows", row_keys_match,
                       f"rows={len(actual)}"))
        checks.append(("pipeline ddg_pred matches frozen Phase 9 predictions within 1e-6",
                       len(actual) == len(expected) and max_ddg_delta <= 1e-6,
                       f"max_abs_delta={max_ddg_delta:.12g}"))
        checks.append(("pipeline p_impact matches frozen Phase 9 predictions within 1e-6",
                       len(actual) == len(expected) and max_p_delta <= 1e-6,
                       f"max_abs_delta={max_p_delta:.12g}"))
        checks.append(("pipeline model identifier, class and disclaimers are consistent",
                       set(actual.esm_model) == {"facebook/esm2_t12_35M_UR50D"}
                       and actual.rank_class.notna().all()
                       and actual.evidence_text.str.contains(DISCLAIMER, regex=False).all(),
                       f"rows={len(actual)}; tiers={sorted(actual.tier.unique())}"))
        summary_payload = {
            "n_rows": int(len(actual)),
            "protein_id": "P60484",
            "max_abs_ddg_delta_vs_phase_09": max_ddg_delta,
            "max_abs_p_impact_delta_vs_phase_09": max_p_delta,
            "pipeline_summary": summary,
            "fresh_output_files": output_files,
            "disclaimer": DISCLAIMER,
        }
    import subprocess
    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (ROOT / "reports/gates/phase_10_pytest.txt").write_text(test_run.stdout + test_run.stderr)
    last = test_run.stdout.strip().splitlines()[-1] if test_run.stdout.strip() else test_run.stderr.strip()
    checks.append(("pytest -q passes", test_run.returncode == 0, last))
    summary_path = ROOT / "reports/tables/phase_10_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    lines = [DISCLAIMER] + [f"{'PASS' if passed else 'FAIL'}: {name} | {detail}" for name, passed, detail in checks]
    (ROOT / "reports/gates/phase_10_gate.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
