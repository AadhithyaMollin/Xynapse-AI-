#!/usr/bin/env python3
"""Phase 12 gate: report reproducibility, audits, and tracked demo/documentation checks."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_gate() -> int:
    checks: list[tuple[str, bool, str]] = []
    try:
        from scripts.claims_audit import audit as claims_audit
        claim_results = claims_audit(ROOT)
        checks.extend((f"claims audit: {name}", passed, detail) for name, passed, detail in claim_results)
    except Exception as exc:
        checks.append(("claims audit executes", False, f"{type(exc).__name__}: {exc}"))

    try:
        from scripts.leakage_audit import audit as leakage_audit
        leakage_results = leakage_audit()
        checks.extend((f"leakage audit: {name}", passed, detail) for name, passed, detail in leakage_results)
    except Exception as exc:
        checks.append(("leakage audit executes", False, f"{type(exc).__name__}: {exc}"))

    try:
        from scripts.make_reports import run as make_reports
        make_reports()
        manifest_path = ROOT / "reports/tables/generated_report_manifest.json"
        first = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_hashes = {name: item["sha256"] for name, item in first["files"].items()}
        make_reports()
        second = json.loads(manifest_path.read_text(encoding="utf-8"))
        second_hashes = {name: item["sha256"] for name, item in second["files"].items()}
        report_reproducible = first_hashes == second_hashes and first["files"] == second["files"]
        checks.append(("derived reports and model cards reproduce byte-for-byte from saved outputs",
                       report_reproducible,
                       f"artifacts={len(second_hashes)}; stable_hashes={report_reproducible}"))
    except Exception as exc:
        checks.append(("report generator executes reproducibly", False, f"{type(exc).__name__}: {exc}"))

    demo_dir = ROOT / "demo_data"
    manifest_path = demo_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files_ok = True
        file_details = []
        for relative, detail in manifest["files"].items():
            path = ROOT / relative
            size_ok = path.is_file() and path.stat().st_size == detail["bytes"]
            cap = 8 * 1024 * 1024 if path.name == "predictions.parquet" else 2 * 1024 * 1024
            hash_ok = size_ok and path.stat().st_size <= cap and _sha(path) == detail["sha256"]
            files_ok = files_ok and hash_ok
            file_details.append(f"{relative}:{path.stat().st_size if path.exists() else 'missing'}:{hash_ok}")
        checks.append(("demo bundle files match manifest hashes and individual size caps",
                       files_ok, "; ".join(file_details)))
        demo = pd.read_parquet(demo_dir / "predictions.parquet")
        proteins = {str(key): int(len(value)) for key, value in demo.groupby("protein_id")}
        row_counts_ok = proteins == manifest["proteins"] and set(proteins) == {"P60484", "P04637"}
        row_counts_ok = row_counts_ok and all(len(frame) == 19 * int(frame.position.max())
                                               for _, frame in demo.groupby("protein_id"))
        columns_ok = {"esm_llr", "ddg_pred", "p_impact", "rank_class", "evidence_text"}.issubset(demo.columns)
        checks.append(("demo contains full 19 x L predictions for both case studies with dashboard metrics",
                       row_counts_ok and columns_ok,
                       f"rows={len(demo)}; proteins={proteins}; required_columns={columns_ok}"))
        checks.append(("demo manifest and data card carry the disclaimer and AlphaFold source evidence",
                       manifest.get("disclaimer") == DISCLAIMER
                       and (ROOT / "docs/data_cards/alphafold_demo.md").is_file()
                       and DISCLAIMER in (ROOT / "docs/data_cards/alphafold_demo.md").read_text(encoding="utf-8"),
                       f"structures={len(manifest.get('structures', []))}; card_present={(ROOT / 'docs/data_cards/alphafold_demo.md').is_file()}"))
    except Exception as exc:
        checks.append(("demo bundle inspection succeeds", False, f"{type(exc).__name__}: {exc}"))

    model_dir = ROOT / "models/esm2_t12_35M"
    for filename in ("stability_tier1.json", "function_tier1.json"):
        try:
            meta = json.loads((model_dir / filename).read_text(encoding="utf-8"))
            checks.append((f"{filename} reports model counts and Tier-1 metadata",
                           meta.get("tier") == "tier1" and int(meta.get("n_proteins", 0)) >= 10
                           and int(meta.get("n_rows", 0)) > 0 and bool(meta.get("feature_list")),
                           f"proteins={meta.get('n_proteins')}; rows={meta.get('n_rows')}; tier={meta.get('tier')}"))
        except Exception as exc:
            checks.append((f"{filename} is readable", False, f"{type(exc).__name__}: {exc}"))

    case_files = [ROOT / f"reports/tables/test_case_study_{protein}{suffix}.csv"
                  for protein in ("P60484", "P04637")
                  for suffix in ("", "_variants", "_worst_disagreements")]
    checks.append(("case-study summary, comparison, and disagreement outputs exist for both proteins",
                   all(path.is_file() for path in case_files),
                   f"files={[path.name for path in case_files if path.is_file()]}"))
    required_docs = [ROOT / "docs/DEPLOY.md", ROOT / "docs/SHARING_ARTIFACTS.md", ROOT / "docs/limitations.md"]
    checks.append(("deployment, artifact-sharing, and limitations docs exist with disclaimer",
                   all(path.is_file() and DISCLAIMER in path.read_text(encoding="utf-8") for path in required_docs),
                   f"docs={[path.relative_to(ROOT).as_posix() for path in required_docs]}"))

    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    pytest_output = test_run.stdout + test_run.stderr
    (ROOT / "reports/gates/phase_12_pytest.txt").write_text(pytest_output, encoding="utf-8")
    final_line = next((line for line in reversed(test_run.stdout.splitlines()) if line.strip()), "no pytest summary")
    checks.append(("pytest -q passes", test_run.returncode == 0, final_line))

    lines = [DISCLAIMER] + [f"{'PASS' if passed else 'FAIL'}: {name} | {detail}" for name, passed, detail in checks]
    gate_path = ROOT / "reports/gates/phase_12_gate.txt"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "disclaimer": DISCLAIMER,
        "checks": len(checks),
        "passed": sum(passed for _, passed, _ in checks),
        "failed": sum(not passed for _, passed, _ in checks),
        "pytest_exit_code": test_run.returncode,
        "demo_manifest_sha256": _sha(manifest_path) if manifest_path.is_file() else None,
    }
    summary_path = ROOT / "reports/tables/phase_12_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
