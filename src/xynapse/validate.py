"""Check a table against the data contract in schema.py.
validate_table returns a list of problems ([] means the table is OK)."""
from __future__ import annotations

import pandas as pd

from .schema import AA, TABLES


def validate_table(df: pd.DataFrame, name: str, proteins: pd.DataFrame | None = None) -> list[str]:
    if name not in TABLES:
        raise KeyError(f"'{name}' is not a table in the data contract")
    spec = TABLES[name]

    missing = [c for c in spec["required"] if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    problems = []
    keys = spec["keys"]
    if df[keys].isna().any().any():
        problems.append("null values in key columns")
    n_dup = int(df.duplicated(keys).sum())
    if n_dup:
        problems.append(f"{n_dup} duplicate rows for key {keys}")

    level = spec["level"]
    if level == "protein":
        bad = df[df["length"] != df["sequence"].str.len()]
        if len(bad):
            problems.append(f"length != len(sequence) for {len(bad)} proteins")
        return problems

    if (df["position"] < 1).any():
        problems.append("position must be 1-based (found values < 1)")
    if (~df["wt"].isin(list(AA))).any():
        problems.append("wt has letters outside the 20 canonical amino acids")
    if level == "mutation":
        if (~df["mut"].isin(list(AA))).any():
            problems.append("mut has letters outside the 20 canonical amino acids")
        if (df["mut"] == df["wt"]).any():
            problems.append("some rows have mut == wt")

    if proteins is not None:
        seqs = proteins.set_index("protein_id")["sequence"]
        for pid, g in df.groupby("protein_id"):
            if pid not in seqs.index:
                problems.append(f"{pid}: not found in proteins table")
                continue
            seq = seqs[pid]
            L = len(seq)
            in_range = g[(g["position"] >= 1) & (g["position"] <= L)]
            n_out = len(g) - len(in_range)
            if n_out:
                problems.append(f"{pid}: {n_out} rows with position outside 1..{L}")
            expected_wt = in_range["position"].map(lambda p: seq[p - 1])
            n_bad = int((in_range["wt"] != expected_wt).sum())
            if n_bad:
                problems.append(f"{pid}: wt letter does not match the sequence in {n_bad} rows")
            if spec["complete"]:
                expected = L if level == "position" else 19 * L
                if len(g) != expected:
                    problems.append(f"{pid}: expected {expected} rows, found {len(g)}")
    return problems
