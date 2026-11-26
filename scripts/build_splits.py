#!/usr/bin/env python3
"""Run configured MMseqs2 clustering and build grouped train/CV tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xynapse.splits import CASE_STUDY_IDS, assign_splits, attach_assignments, make_training_table, parse_cluster_tsv, summarize_splits  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs/config.yaml").read_text())


def write_fasta(proteins: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in proteins.sort_values("protein_id").itertuples(index=False):
            sequence = str(row.sequence).upper()
            if not sequence or any(letter not in "ACDEFGHIKLMNPQRSTVWY" for letter in sequence):
                raise ValueError(f"{row.protein_id}: sequence is empty or contains non-canonical amino acids")
            handle.write(f">{row.protein_id}\n{sequence}\n")


def run_mmseqs(fasta: Path, prefix: Path, temp_dir: Path, binary: str, identity: float, coverage: float) -> tuple[Path, str, list[str], bool, float]:
    version = subprocess.run([binary, "version"], check=True, capture_output=True, text=True).stdout.strip()
    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    metadata_path = Path(f"{prefix}_run.json")
    command = [binary, "easy-cluster", str(fasta), str(prefix), str(temp_dir),
               "--min-seq-id", str(identity), "-c", str(coverage), "--cov-mode", "0"]
    input_hash = sha256(fasta)
    if cluster_tsv.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("input_sha256") == input_hash and metadata.get("version") == version
                and metadata.get("identity") == identity and metadata.get("coverage") == coverage
                and metadata.get("command") == command):
            return cluster_tsv, version, command, True, 0.0
    prefix.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    (prefix.parent / "mmseqs_easy_cluster.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"MMseqs2 failed ({completed.returncode}): {completed.stderr[-4000:]}")
    if not cluster_tsv.is_file():
        raise RuntimeError(f"MMseqs2 returned success but did not create expected TSV: {cluster_tsv}")
    metadata_path.write_text(json.dumps({
        "version": version,
        "input_sha256": input_hash,
        "identity": identity,
        "coverage": coverage,
        "command": command,
        "command_display": " ".join(command),
        "runtime_seconds": elapsed,
    }, indent=2) + "\n")
    return cluster_tsv, version, command, False, elapsed


def build(binary: str | None = None) -> dict:
    started = time.perf_counter()
    config = load_config()
    mmseqs = binary or shutil.which("mmseqs")
    if not mmseqs:
        raise RuntimeError("MMseqs2 executable not found; install the project-declared mmseqs2 dependency")
    proteins_path = ROOT / "data/interim/train_proteins.parquet"
    proteins = pd.read_parquet(proteins_path)
    protein_ids = set(proteins.protein_id.astype(str))
    fasta = ROOT / "data/interim/train_proteins.fasta"
    write_fasta(proteins, fasta)
    prefix = ROOT / "data/interim/mmseqs_train_proteins"
    cluster_tsv, version, command, reused, mmseqs_runtime = run_mmseqs(
        fasta, prefix, ROOT / "data/interim/mmseqs_tmp", mmseqs,
        float(config["splits"]["identity"]), float(config["splits"]["coverage"]),
    )
    # This parser validates the observed TSV is exactly representative<TAB>member,
    # with one assignment for every FASTA ID and no unrecognized header/row.
    cluster_map = parse_cluster_tsv(cluster_tsv, protein_ids)
    external_path = ROOT / "data/processed/external_clinvar.parquet"
    external_ids = set(pd.read_parquet(external_path).protein_id.astype(str)) if external_path.exists() else set()
    assignments = assign_splits(
        proteins, cluster_map, case_study_ids=CASE_STUDY_IDS & protein_ids,
        external_test_ids=external_ids, test_fraction=float(config["splits"]["test_fraction_of_clusters"]),
        seed=int(config["seed"]), n_folds=5,
    )

    ddg_path = ROOT / "data/processed/labels_ddg.parquet"
    function_path = ROOT / "data/processed/labels_function.parquet"
    features_path = ROOT / "data/processed/features_tier1.parquet"
    ddg = pd.read_parquet(ddg_path)
    function = pd.read_parquet(function_path)
    features = pd.read_parquet(features_path)
    ddg_updated = attach_assignments(ddg, assignments)
    function_updated = attach_assignments(function, assignments)
    for task, labels in (("ddg", ddg_updated), ("function", function_updated)):
        task_train = labels.loc[labels.split_id.eq("train"), "protein_id"].nunique()
        if task_train < 10:
            raise RuntimeError(f"fewer than 10 training proteins survive the cluster filter for {task}: {task_train}")
    ddg_train = make_training_table(features, ddg, assignments)
    function_train = make_training_table(features, function, assignments)
    if not ddg_train.loc[ddg_train.split_id.eq("train"), "protein_id"].nunique() >= 10:
        raise RuntimeError("fewer than 10 ddG training proteins in joined table")
    if not function_train.loc[function_train.split_id.eq("train"), "protein_id"].nunique() >= 10:
        raise RuntimeError("fewer than 10 function training proteins in joined table")

    assignments_path = ROOT / "data/interim/split_assignments.parquet"
    assignments.to_parquet(assignments_path, index=False)
    ddg_updated.to_parquet(ddg_path, index=False)
    function_updated.to_parquet(function_path, index=False)
    ddg_train.to_parquet(ROOT / "data/processed/train_ddg.parquet", index=False)
    function_train.to_parquet(ROOT / "data/processed/train_function.parquet", index=False)
    summary = summarize_splits(assignments, ddg_updated, function_updated)
    summary.to_csv(ROOT / "reports/tables/split_summary.csv", index=False)
    cluster_summary = assignments.groupby("cluster_id", as_index=False).agg(
        n_proteins=("protein_id", "nunique"), split_id=("split_id", "first"), cv_fold=("cv_fold", "first"))
    cluster_summary.to_csv(ROOT / "reports/tables/mmseqs_cluster_summary.csv", index=False)

    card = {
        "version": version,
        "command": command,
        "command_display": " ".join(command),
        "identity": float(config["splits"]["identity"]),
        "coverage": float(config["splits"]["coverage"]),
        "cov_mode": 0,
        "input_fasta": str(fasta.relative_to(ROOT)),
        "input_fasta_sha256": sha256(fasta),
        "cluster_tsv": str(cluster_tsv.relative_to(ROOT)),
        "cluster_tsv_sha256": sha256(cluster_tsv),
        "retrieval_date": None,
        "n_proteins": len(proteins),
        "n_clusters": int(assignments.cluster_id.nunique()),
        "n_case_external_test_ids": len(CASE_STUDY_IDS | external_ids),
        "n_test_clusters": int(assignments.loc[assignments.split_id.eq("test"), "cluster_id"].nunique()),
        "mmseqs_runtime_seconds": round(mmseqs_runtime, 6),
        "mmseqs_result_reused": reused,
        "build_runtime_seconds": round(time.perf_counter() - started, 6),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    card["retrieval_date"] = card["created_utc"][:10]
    (ROOT / "reports/tables/phase_05_split_summary.json").write_text(json.dumps(card, indent=2) + "\n")
    doc = [
        "# Sequence clustering and split data card", "",
        "Predictions are computational hypotheses, not experimental findings.", "",
        "## Inputs and reproducibility", "",
        f"- MMseqs2 version: `{version}`.",
        f"- Exact command: `{card['command_display']}`.",
        f"- Input FASTA: `{card['input_fasta']}`; SHA-256 `{card['input_fasta_sha256']}`.",
        f"- Observed MMseqs cluster TSV: `{card['cluster_tsv']}`; SHA-256 `{card['cluster_tsv_sha256']}`.",
        f"- Retrieval/build date (UTC): {card['retrieval_date']}.",
        "- TSV inspection: two tab-separated fields per row (representative, member); every FASTA ID occurs exactly once as a member.",
        f"- Measured proteins: {card['n_proteins']}; clusters: {card['n_clusters']}; test clusters: {card['n_test_clusters']}.",
        "- Split policy: complete case-study/external-test clusters plus a seeded 15% sample of other clusters are test; remaining clusters are training. Training clusters are assigned to five shuffled GroupKFold validation folds with seed 42.",
        "- All training artifacts remain local/ignored. See `reports/tables/split_summary.csv` for counts by task and split.",
        "",
    ]
    (ROOT / "docs/data_cards/splits.md").write_text("\n".join(doc))
    return card


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmseqs", help="MMseqs2 executable; default: discovered on PATH")
    args = parser.parse_args()
    result = build(args.mmseqs)
    print(json.dumps(result, indent=2))
    print("Predictions are computational hypotheses, not experimental findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
