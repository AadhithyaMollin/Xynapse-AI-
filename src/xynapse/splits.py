"""Sequence-clustered train/test assignment and mutation-table assembly."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .schema import MUT_KEYS

CASE_STUDY_IDS = frozenset({"P60484", "P04637"})


def parse_cluster_tsv(path: str | Path, protein_ids: set[str]) -> dict[str, str]:
    """Parse MMseqs representative/member TSV after validating its real row format."""
    mapping: dict[str, str] = {}
    rows = 0
    with Path(path).open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 2 or not row[0] or not row[1]:
                raise ValueError(f"cluster TSV line {line_number}: expected two non-empty tab-separated fields")
            representative, member = row
            if representative not in protein_ids or member not in protein_ids:
                raise ValueError(
                    f"cluster TSV line {line_number}: unknown ID(s) {representative!r}, {member!r}; "
                    "expected FASTA protein IDs"
                )
            if member in mapping:
                raise ValueError(f"cluster TSV line {line_number}: duplicate member {member!r}")
            mapping[member] = representative
            rows += 1
    missing = protein_ids - set(mapping)
    if missing:
        raise ValueError(f"cluster TSV omits {len(missing)} protein(s): {sorted(missing)[:5]}")
    if set(mapping) != protein_ids or rows != len(protein_ids):
        raise ValueError("cluster TSV does not assign each FASTA protein exactly once")
    return mapping


def assign_splits(
    proteins: pd.DataFrame,
    cluster_map: dict[str, str],
    *,
    case_study_ids: set[str] | frozenset[str] = CASE_STUDY_IDS,
    external_test_ids: set[str] | frozenset[str] = frozenset(),
    test_fraction: float = 0.15,
    seed: int = 42,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Assign whole clusters to train/test and training clusters to one CV fold."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    required = {"protein_id"}
    if not required <= set(proteins.columns):
        raise ValueError("proteins table must include protein_id")
    ids = set(proteins["protein_id"].astype(str))
    if proteins["protein_id"].duplicated().any():
        raise ValueError("proteins table contains duplicate protein_id values")
    if set(cluster_map) != ids:
        raise ValueError("cluster map must cover exactly the proteins table")
    requested_cases = set(case_study_ids) | set(external_test_ids)
    missing_cases = requested_cases - ids
    if missing_cases:
        raise ValueError(f"case-study/external-test proteins missing from proteins table: {sorted(missing_cases)}")

    cluster_by_id = {protein_id: str(cluster_map[protein_id]) for protein_id in ids}
    case_clusters = {cluster_by_id[protein_id] for protein_id in requested_cases}
    other_clusters = sorted(set(cluster_by_id.values()) - case_clusters)
    if not other_clusters:
        raise ValueError("no non-case-study clusters remain for a train/test split")
    n_test = max(1, int(math.ceil(len(other_clusters) * test_fraction)))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(other_clusters))
    test_clusters = case_clusters | {other_clusters[index] for index in order[:n_test]}

    records = []
    for protein_id in sorted(ids):
        cluster_id = cluster_by_id[protein_id]
        split_id = "test" if cluster_id in test_clusters else "train"
        records.append({"protein_id": protein_id, "cluster_id": cluster_id, "split_id": split_id})
    assignments = pd.DataFrame(records)
    train = assignments.loc[assignments.split_id.eq("train")].copy()
    train_clusters = set(train.cluster_id)
    if len(train_clusters) < n_folds:
        raise ValueError(f"need at least {n_folds} training clusters for grouped CV, found {len(train_clusters)}")
    fold_by_cluster: dict[str, int] = {}
    fold_splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    x = np.zeros((len(train), 1), dtype=np.uint8)
    for fold, (_, validation_indices) in enumerate(fold_splitter.split(x, groups=train.cluster_id)):
        for cluster_id in train.iloc[validation_indices].cluster_id.unique():
            if cluster_id in fold_by_cluster:
                raise AssertionError(f"cluster {cluster_id} was assigned to multiple validation folds")
            fold_by_cluster[str(cluster_id)] = fold
    if set(fold_by_cluster) != train_clusters:
        raise AssertionError("not every training cluster received a CV fold")
    assignments["cv_fold"] = assignments.cluster_id.map(fold_by_cluster).astype("Int64")
    if assignments.loc[assignments.split_id.eq("test"), "cv_fold"].notna().any():
        raise AssertionError("test proteins must not have a CV fold")
    return assignments


def attach_assignments(table: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Add cluster and split identifiers, preserving existing table column order."""
    mapping = assignments.set_index("protein_id")[["cluster_id", "split_id"]]
    result = table.copy()
    for column in ("cluster_id", "split_id"):
        values = result["protein_id"].map(mapping[column])
        if values.isna().any():
            missing = sorted(result.loc[values.isna(), "protein_id"].astype(str).unique())
            raise ValueError(f"no split assignment for protein(s): {missing[:5]}")
        if column in result:
            result[column] = values.to_numpy()
        else:
            result[column] = values.to_numpy()
    return result


def make_training_table(features: pd.DataFrame, labels: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Join mutation labels to Tier-1 features and append split/CV metadata."""
    if features.duplicated(MUT_KEYS).any() or labels.duplicated(MUT_KEYS).any():
        raise ValueError("feature and label tables must have unique mutation keys")
    updated_labels = attach_assignments(labels, assignments)
    merged = features.merge(updated_labels, on=MUT_KEYS, how="inner", validate="one_to_one")
    cv_map = assignments.set_index("protein_id")["cv_fold"]
    merged["cv_fold"] = merged["protein_id"].map(cv_map).astype("Int64")
    if merged.empty:
        raise ValueError("feature/label join produced no training rows")
    return merged.sort_values(["protein_id", "position", "mut"], kind="stable").reset_index(drop=True)


def summarize_splits(assignments: pd.DataFrame, ddg: pd.DataFrame, function: pd.DataFrame) -> pd.DataFrame:
    """Summarize protein/cluster/row counts by task, split, and training fold."""
    records: list[dict] = []
    for task, labels in (("ddg", ddg), ("function", function)):
        joined = labels[["protein_id"]].merge(
            assignments[["protein_id", "cluster_id", "split_id", "cv_fold"]],
            on="protein_id", how="left", validate="many_to_one",
        )
        if joined[["cluster_id", "split_id"]].isna().any().any():
            raise ValueError(f"{task}: labels have proteins without split assignments")
        for split_id, split_rows in joined.groupby("split_id", sort=True):
            records.append({
                "task": task,
                "split_id": split_id,
                "cv_fold": "all",
                "n_clusters": int(split_rows.cluster_id.nunique()),
                "n_proteins": int(split_rows.protein_id.nunique()),
                "n_rows": int(len(split_rows)),
            })
            if split_id == "train":
                for fold, fold_rows in split_rows.groupby("cv_fold", sort=True, dropna=True):
                    records.append({
                        "task": task,
                        "split_id": "train_cv_validation",
                        "cv_fold": int(fold),
                        "n_clusters": int(fold_rows.cluster_id.nunique()),
                        "n_proteins": int(fold_rows.protein_id.nunique()),
                        "n_rows": int(len(fold_rows)),
                    })
    return pd.DataFrame.from_records(records)
