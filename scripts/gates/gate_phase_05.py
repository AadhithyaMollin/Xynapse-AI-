#!/usr/bin/env python3
"""Phase 5 gate for real sequence clustering and grouped splits."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def run_gate() -> int:
    from xynapse.schema import MUT_KEYS
    from xynapse.splits import CASE_STUDY_IDS, parse_cluster_tsv
    from xynapse.validate import validate_table

    checks: list[tuple[str, bool, str]] = []
    report_path = ROOT / "reports/gates/phase_05_gate.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_parquet(ROOT / "data/interim/train_proteins.parquet")
    assignments = pd.read_parquet(ROOT / "data/interim/split_assignments.parquet")
    ddg = pd.read_parquet(ROOT / "data/processed/labels_ddg.parquet")
    function = pd.read_parquet(ROOT / "data/processed/labels_function.parquet")
    train_ddg = pd.read_parquet(ROOT / "data/processed/train_ddg.parquet")
    train_function = pd.read_parquet(ROOT / "data/processed/train_function.parquet")
    clusters = parse_cluster_tsv(ROOT / "data/interim/mmseqs_train_proteins_cluster.tsv",
                                 set(proteins.protein_id.astype(str)))

    checks.append(("one cluster assignment per protein; observed MMseqs TSV parsed", 
                   len(assignments) == len(proteins) and assignments.protein_id.is_unique
                   and set(assignments.protein_id) == set(proteins.protein_id)
                   and len(clusters) == len(proteins),
                   f"proteins={len(proteins)}; clusters={assignments.cluster_id.nunique()}"))
    train_clusters = set(assignments.loc[assignments.split_id.eq("train"), "cluster_id"])
    test_clusters = set(assignments.loc[assignments.split_id.eq("test"), "cluster_id"])
    checks.append(("train and test clusters are disjoint", train_clusters.isdisjoint(test_clusters),
                   f"train={len(train_clusters)}; test={len(test_clusters)}"))
    train_cv = assignments.loc[assignments.split_id.eq("train")]
    cv_group_counts = train_cv.groupby("cluster_id").cv_fold.nunique(dropna=True)
    checks.append(("every training cluster occurs in exactly one CV validation fold",
                   train_cv.cv_fold.notna().all() and cv_group_counts.eq(1).all(),
                   f"folds={sorted(train_cv.cv_fold.dropna().astype(int).unique().tolist())}"))
    checks.append(("test proteins have no CV fold", assignments.loc[assignments.split_id.eq("test"), "cv_fold"].isna().all(),
                   f"test proteins={int(assignments.split_id.eq('test').sum())}"))

    external_path = ROOT / "data/processed/external_clinvar.parquet"
    external_ids = set(pd.read_parquet(external_path).protein_id.astype(str))
    held_out_ids = (set(CASE_STUDY_IDS) | external_ids) & set(proteins.protein_id.astype(str))
    held_out_clusters = set(assignments.loc[assignments.protein_id.isin(held_out_ids), "cluster_id"])
    train_ids = set(assignments.loc[assignments.split_id.eq("train"), "protein_id"])
    checks.append(("case-study and ClinVar-test clusters are absent from training",
                   not held_out_clusters & train_clusters and not held_out_ids & train_ids,
                   f"held-out proteins={len(held_out_ids)}; held-out clusters={len(held_out_clusters)}"))

    for name, labels, training in (("ddG", ddg, train_ddg), ("function", function, train_function)):
        problems = validate_table(labels, "labels_ddg" if name == "ddG" else "labels_function", proteins)
        checks.append((f"{name} label table validates and has unique keys", not problems and not labels.duplicated(MUT_KEYS).any(),
                       f"rows={len(labels)}; proteins={labels.protein_id.nunique()}; validator={problems or 'PASS'}"))
        fit_rows = training.loc[training.split_id.eq("train")]
        train_protein_count = fit_rows.protein_id.nunique()
        checks.append((f"{name} training set has at least 10 proteins", train_protein_count >= 10,
                       f"train_proteins={train_protein_count}; train_rows={len(fit_rows)}; test_rows={int(training.split_id.eq('test').sum())}"))
        checks.append((f"{name} joined training table has cluster/split/CV columns",
                       {"cluster_id", "split_id", "cv_fold"} <= set(training.columns)
                       and not training.cluster_id.isna().any() and not training.split_id.isna().any(),
                       f"rows={len(training)}"))
        fold_clusters = training.loc[training.split_id.eq("train")].groupby("cluster_id").cv_fold.nunique(dropna=True)
        checks.append((f"{name} training table does not split a cluster across CV folds",
                       fold_clusters.eq(1).all(), f"training_clusters={len(fold_clusters)}"))

    summary_path = ROOT / "reports/tables/split_summary.csv"
    cluster_card = ROOT / "docs/data_cards/splits.md"
    summary = pd.read_csv(summary_path)
    checks.append(("split summary and data card exist", len(summary) > 0 and cluster_card.is_file()
                   and DISCLAIMER in cluster_card.read_text(),
                   f"summary_rows={len(summary)}"))
    ptest = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    pytest_path = ROOT / "reports/gates/phase_05_pytest.txt"
    pytest_path.write_text(ptest.stdout + ptest.stderr)
    last = ptest.stdout.strip().splitlines()[-1] if ptest.stdout.strip() else ptest.stderr.strip()
    checks.append(("pytest -q passes", ptest.returncode == 0, last))

    lines = [DISCLAIMER] + [f"{'PASS' if ok else 'FAIL'}: {name} | {details}" for name, ok, details in checks]
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
