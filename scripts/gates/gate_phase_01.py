#!/usr/bin/env python3
"""Phase 1 gate: helper tests, data-contract validation, and staged-tree policy."""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_check(name: str, command: list[str], cwd: Path = ROOT) -> tuple[bool, str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    detail = (result.stdout + result.stderr).strip()
    return result.returncode == 0, detail or f"exit code {result.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, default=ROOT.parent / "xynapse-shared")
    args = parser.parse_args()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("pytest", *run_check("pytest", [sys.executable, "-m", "pytest", "-q"])))
    checks.append(("contract table validator", *run_check("validator", [sys.executable, "scripts/validate_tables.py"])))
    if args.shared.is_dir():
        checks.append(("pre_push_check", *run_check("pre_push_check", [sys.executable, str(ROOT / "scripts/pre_push_check.py"), "--repo", str(args.shared)])))
    else:
        checks.append(("pre_push_check", False, f"shared clone not found: {args.shared}"))

    lines = [
        "Predictions are computational hypotheses, not experimental findings.",
        f"Phase 1 gate run: {datetime.now(timezone.utc).isoformat()}",
    ]
    for name, passed, detail in checks:
        lines.append(f"{'PASS' if passed else 'FAIL'}: {name} — {detail.replace(chr(10), ' | ')}")
    report = ROOT / "reports/gates/phase_01_gate.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Gate report: {report}")
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
