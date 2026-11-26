#!/usr/bin/env python3
"""Phase 13 final verification gate; does not evaluate held-out labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.verify_all import DISCLAIMER, run_all  # noqa: E402


def run_gate() -> int:
    shared = ROOT.parent / "xynapse-shared"
    results = run_all(shared)
    lines = [DISCLAIMER] + [f"{'PASS' if ok else 'FAIL'}: {name} | {detail}" for name, ok, detail in results]
    report = ROOT / "reports/gates/phase_13_gate.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "checks": len(results),
        "passed": sum(ok for _, ok, _ in results),
        "failed": sum(not ok for _, ok, _ in results),
        "held_out_test_evaluations_run": 0,
        "disclaimer": DISCLAIMER,
    }
    summary_path = ROOT / "reports/tables/phase_13_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
