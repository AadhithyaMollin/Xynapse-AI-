#!/usr/bin/env python3
"""Run final repository checks without fitting models or reevaluating test sets."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def _last_lines(text: str, limit: int = 3) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-limit:]) if lines else "no output"


def _subprocess_check(label: str, command: list[str], *, cwd: Path, timeout: int = 180) -> tuple[str, bool, str]:
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        detail = _last_lines(result.stdout + "\n" + result.stderr)
        return label, result.returncode == 0, f"exit={result.returncode}; seconds={time.monotonic() - started:.3f}; {detail}"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return label, False, f"timeout after {timeout}s; {_last_lines(stdout + stderr)}"
    except Exception as exc:  # report a failed check rather than aborting the remaining audit
        return label, False, f"{type(exc).__name__}: {exc}"


def _contract_check() -> tuple[str, bool, str]:
    import pandas as pd
    from xynapse.schema import TABLES
    from xynapse.validate import validate_table

    roots = (ROOT / "data", ROOT / "demo_data")
    files = sorted({path.resolve() for folder in roots if folder.exists() for path in folder.rglob("*.parquet")})
    protein_path = ROOT / "data/interim/train_proteins.parquet"
    proteins = pd.read_parquet(protein_path) if protein_path.is_file() else None
    failures = []
    checked = []
    for path in files:
        name = "proteins" if path.name == "train_proteins.parquet" else path.stem
        if name not in TABLES:
            continue
        table_proteins = proteins
        if name == "proteins":
            table_proteins = None
        elif table_proteins is None and (path.parent / "proteins.parquet").is_file():
            table_proteins = pd.read_parquet(path.parent / "proteins.parquet")
        problems = validate_table(pd.read_parquet(path), name, table_proteins)
        checked.append(path.relative_to(ROOT).as_posix())
        if problems:
            failures.append(f"{path.relative_to(ROOT)}: {problems}")
    ok = bool(checked) and not failures
    return "contract validator on every known data/demo Parquet table", ok, f"checked={len(checked)}; failures={failures}"


def _row_count_check() -> tuple[str, bool, str]:
    import pandas as pd

    processed = ROOT / "data/processed/predictions.parquet"
    demo = ROOT / "demo_data/predictions.parquet"
    proteins_path = ROOT / "data/interim/train_proteins.parquet"
    try:
        predictions = pd.read_parquet(processed)
        demo_predictions = pd.read_parquet(demo)
        proteins = pd.read_parquet(proteins_path)
        lengths = dict(zip(proteins.protein_id.astype(str), proteins.length.astype(int)))
        per_protein = predictions.groupby(predictions.protein_id.astype(str)).size().to_dict()
        complete = bool(per_protein) and all(key in lengths and count == 19 * lengths[key]
                                             for key, count in per_protein.items())
        matching_demo = set(demo_predictions.protein_id.astype(str)) <= set(per_protein)
        demo_full = all(len(group) == 19 * int(group.position.max())
                        for _, group in demo_predictions.groupby(demo_predictions.protein_id.astype(str)))
        ok = complete and matching_demo and demo_full
        return "prediction and demo row counts", ok, (
            f"predictions={len(predictions)}; demo={len(demo_predictions)}; "
            f"proteins={len(per_protein)}; complete_19xL={complete}; demo_complete={demo_full}"
        )
    except Exception as exc:
        return "prediction and demo row counts", False, f"{type(exc).__name__}: {exc}"


def _frozen_check() -> tuple[str, bool, str]:
    from xynapse.freeze import verify_frozen

    model_dir = ROOT / "models/esm2_t12_35M"
    path = model_dir / "FROZEN.json"
    try:
        ok, problems = verify_frozen(path, model_dir)
        frozen = json.loads(path.read_text(encoding="utf-8"))
        ok = ok and frozen.get("esm_model") == "facebook/esm2_t12_35M_UR50D"
        return "frozen model hash and ESM model identity", bool(ok), f"files={len(frozen.get('model_files_sha256', {}))}; problems={problems}"
    except Exception as exc:
        return "frozen model hash and ESM model identity", False, f"{type(exc).__name__}: {exc}"


def _pipeline_reproduction_check() -> tuple[str, bool, str]:
    import numpy as np
    import pandas as pd
    from xynapse.pipeline import run_pipeline
    from xynapse.schema import MUT_KEYS

    try:
        expected_path = ROOT / "data/processed/predictions.parquet"
        expected_all = pd.read_parquet(expected_path)
        expected = expected_all.loc[expected_all.protein_id.astype(str).eq("P60484")].sort_values(MUT_KEYS).reset_index(drop=True)
        with tempfile.TemporaryDirectory(prefix="xynapse-final-pipeline-") as output:
            actual, _ = run_pipeline("P60484", esm_model="dev", out=output, project_root=ROOT)
            actual = actual.sort_values(MUT_KEYS).reset_index(drop=True)
            key_match = expected[MUT_KEYS].equals(actual[MUT_KEYS])
            ddg_delta = float(np.max(np.abs(expected.ddg_pred.to_numpy(float) - actual.ddg_pred.to_numpy(float))))
            impact_delta = float(np.max(np.abs(expected.p_impact.to_numpy(float) - actual.p_impact.to_numpy(float))))
            outputs = all((Path(output) / name).is_file() for name in ("predictions.parquet", "features_tier1.parquet", "pipeline_summary.json"))
            ok = key_match and outputs and len(actual) == len(expected) and ddg_delta <= 1e-6 and impact_delta <= 1e-6
            return "pipeline reproduction matches saved P60484 predictions", ok, (
                f"rows={len(actual)}; outputs={outputs}; max_abs_ddg_delta={ddg_delta:.12g}; "
                f"max_abs_p_impact_delta={impact_delta:.12g}"
            )
    except Exception as exc:
        return "pipeline reproduction matches saved P60484 predictions", False, f"{type(exc).__name__}: {exc}"


def _clean_clone_payloads(root: Path) -> tuple[list[Path], list[Path]]:
    data_payloads = [p for p in (root / "data").rglob("*") if p.is_file() and p.name != ".gitkeep"] if (root / "data").exists() else []
    model_binaries = [p for p in (root / "models").rglob("*")
                      if p.is_file() and p.suffix.lower() in {".joblib", ".pkl", ".pt", ".bin"}] if (root / "models").exists() else []
    return data_payloads, model_binaries


def _clean_tree_check(repo: Path) -> tuple[str, bool, str]:
    try:
        branch_result = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"],
                                       check=True, capture_output=True, text=True)
        branch = branch_result.stdout.strip()
        if not branch:
            return "clean-clone pytest without local data/model binaries", False, "source clone is detached"
        with tempfile.TemporaryDirectory(prefix="xynapse-clean-clone-") as temp:
            destination = Path(temp) / "clone"
            cloned = subprocess.run(["git", "clone", "--quiet", "--single-branch", "--branch", branch,
                                     str(repo), str(destination)], capture_output=True, text=True, timeout=120)
            if cloned.returncode:
                return "clean-clone pytest without local data/model binaries", False, _last_lines(cloned.stderr)
            data_payloads, model_binaries = _clean_clone_payloads(destination)
            if data_payloads or model_binaries:
                return "clean-clone pytest without local data/model binaries", False, (
                    f"unexpected data={len(data_payloads)}; model_binaries={len(model_binaries)}"
                )
            result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=destination,
                                    capture_output=True, text=True, timeout=180)
            detail = _last_lines(result.stdout + "\n" + result.stderr)
            return "clean-clone pytest without local data/model binaries", result.returncode == 0, (
                f"branch={branch}; clean_data=True; model_binaries=0; exit={result.returncode}; {detail}"
            )
    except Exception as exc:
        return "clean-clone pytest without local data/model binaries", False, f"{type(exc).__name__}: {exc}"


def run_all(repo: Path) -> list[tuple[str, bool, str]]:
    repo = repo.expanduser().resolve()
    results: list[tuple[str, bool, str]] = []
    results.append(_subprocess_check("pytest -q", [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, timeout=240))
    try:
        results.append(_contract_check())
    except Exception as exc:
        results.append(("contract validator on every known data/demo Parquet table", False, f"{type(exc).__name__}: {exc}"))
    results.append(_subprocess_check("leakage audit", [sys.executable, "scripts/leakage_audit.py"], cwd=ROOT))
    results.append(_subprocess_check("claims audit", [sys.executable, "scripts/claims_audit.py"], cwd=ROOT))
    results.append(_row_count_check())
    results.append(_frozen_check())
    results.append(_pipeline_reproduction_check())
    results.append(_subprocess_check("dashboard AppTest", [sys.executable, "-m", "pytest", "-q", "tests/test_dashboard.py"], cwd=ROOT, timeout=180))
    results.append(_subprocess_check("pre_push_check", [sys.executable, "scripts/pre_push_check.py", "--repo", str(repo)], cwd=ROOT))
    results.append(_clean_tree_check(repo))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT.parent / "xynapse-shared",
                        help="shared clone used for staged-policy and clean-clone checks")
    args = parser.parse_args(argv)
    results = run_all(args.repo)
    lines = [DISCLAIMER] + [f"{'PASS' if ok else 'FAIL'}: {name} | {detail}" for name, ok, detail in results]
    report = ROOT / "reports/gates/phase_13_verify_all.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
