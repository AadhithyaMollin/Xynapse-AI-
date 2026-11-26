#!/usr/bin/env python3
"""Phase 3 label/protein contract gate; ClinVar remains external-only."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def run_gate() -> int:
    from xynapse.schema import MUT_KEYS
    from xynapse.validate import validate_table

    proteins = pd.read_parquet(ROOT / "data/interim/train_proteins.parquet")
    ddg = pd.read_parquet(ROOT / "data/processed/labels_ddg.parquet")
    function = pd.read_parquet(ROOT / "data/processed/labels_function.parquet")
    clinvar = pd.read_parquet(ROOT / "data/processed/external_clinvar.parquet")
    prevalence = pd.read_csv(ROOT / "reports/tables/label_prevalence.csv")
    sign = pd.read_csv(ROOT / "reports/tables/sign_check_megascale_labels.csv")
    summary = json.loads((ROOT / "reports/tables/phase_03_label_build_summary.json").read_text())

    checks: list[tuple[str, bool, str]] = []
    checks.append(("train_proteins contract and sequence lengths", not validate_table(proteins, "proteins"),
                   f"n={len(proteins)}"))
    for name, frame, key in (("labels_ddg", ddg, "ddg"),
                             ("labels_function", function, "label"),
                             ("external_clinvar", clinvar, "label")):
        problems = validate_table(frame, name, proteins)
        checks.append((f"{name} validator PASS", not problems, "; ".join(problems) or f"rows={len(frame)}"))
        duplicates = int(frame.duplicated(MUT_KEYS).sum())
        checks.append((f"{name} has unique mutation keys", duplicates == 0, f"duplicate_keys={duplicates}"))

    checks.append(("at least 10 training proteins in each model-label table",
                   ddg["protein_id"].nunique() >= 10 and function["protein_id"].nunique() >= 10,
                   f"ddG={ddg['protein_id'].nunique()}; function={function['protein_id'].nunique()}"))
    checks.append(("ClinVar external table has observed mapped proteins and at least 10 variants",
                   clinvar["protein_id"].nunique() >= 1 and len(clinvar) >= 10,
                   f"external-only proteins={clinvar['protein_id'].nunique()}; variants={len(clinvar)}; constrained to proteins.parquet mappings"))
    checks.append(("all output protein IDs resolve against train_proteins",
                   set(ddg.protein_id) | set(function.protein_id) | set(clinvar.protein_id) <= set(proteins.protein_id),
                   f"train_proteins={len(proteins)}"))
    checks.append(("label prevalence table covers all three outputs",
                   set(prevalence.dataset) == {"labels_ddg", "labels_function", "external_clinvar"},
                   f"rows={len(prevalence)}"))
    checks.append(("written project-ddG X-to-proline sign check is positive",
                   len(sign) == 1 and float(sign.iloc[0].mean_project_ddg) > 0,
                   f"n={int(sign.iloc[0].n_x_to_proline)}; mean={float(sign.iloc[0].mean_project_ddg):.9f}" if len(sign) else "missing"))
    checks.append(("summary records no dropped selected proteins and all label validators passed",
                   not summary["dropped_proteins"] and all(not x for x in summary["validator_results"].values())
                   and not summary["protein_validator"],
                   f"dropped={len(summary['dropped_proteins'])}; runtime={summary['runtime_seconds']:.2f}s"))

    pytest_file = ROOT / "reports/gates/phase_03_pytest.txt"
    pytest_file.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                            capture_output=True, text=True)
    pytest_file.write_text(result.stdout + result.stderr)
    checks.append(("pytest -q passes", result.returncode == 0,
                   result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr.strip()))

    report = [DISCLAIMER] + [f"{'PASS' if ok else 'FAIL'}: {label} | {details}" for label, ok, details in checks]
    path = ROOT / "reports/gates/phase_03_gate.txt"
    path.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
