"""Join mutation-level and position-level feature tables into mut_features.

Contract (see schema.TABLES["mut_features"]):
- one row per MUT_KEYS, exactly 19 * L rows per protein;
- mutation tables merge on MUT_KEYS, position tables left-join on POS_KEYS;
- only contract columns are kept;
- no filling or imputation: a missing position row or a wt mismatch becomes NaN;
- the ESM model name is stored in Parquet schema metadata.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    MUT_ESM_COLS,
    MUT_EVO_COLS,
    MUT_KEYS,
    MUT_PHYS_COLS,
    POS_EVO_COLS,
    POS_FUNC_COLS,
    POS_KEYS,
    POS_STRUCT_COLS,
    TABLES,
)

POSITION_TABLES = {
    "pos_structure": POS_STRUCT_COLS,
    "pos_evolution": POS_EVO_COLS,
    "pos_function": POS_FUNC_COLS,
}

FEATURE_COLUMNS = TABLES["mut_features"]["required"]


def _check_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    if df.duplicated(keys).any():
        n = int(df.duplicated(keys).sum())
        raise ValueError(f"{name}: {n} duplicate rows for key {keys}")


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def build_mut_features(
    mut_esm: pd.DataFrame,
    *,
    mut_phys: pd.DataFrame | None = None,
    pos_tables: dict[str, pd.DataFrame] | None = None,
    esm_model: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the mut_features table.

    Parameters
    ----------
    mut_esm : mutation-level table with MUT_KEYS + MUT_ESM_COLS.
    mut_phys : optional mutation-level table with MUT_KEYS + MUT_PHYS_COLS.
    pos_tables : optional mapping of contract name -> position table
        (``pos_structure``, ``pos_evolution``, ``pos_function``).
    esm_model : model name to record; stored in the returned report.

    Returns
    -------
    (features, report) where report holds the ESM model name, proteins seen in
    position tables but not in mut_esm, and per-column NaN counts.
    """
    _check_columns(mut_esm, MUT_KEYS + MUT_ESM_COLS, "mut_esm")
    _check_unique(mut_esm, MUT_KEYS, "mut_esm")

    out = mut_esm[MUT_KEYS + MUT_ESM_COLS].copy()
    proteins = set(out["protein_id"].unique())

    if mut_phys is not None:
        _check_columns(mut_phys, MUT_KEYS + MUT_PHYS_COLS, "mut_phys")
        _check_unique(mut_phys, MUT_KEYS, "mut_phys")
        out = out.merge(
            mut_phys[MUT_KEYS + MUT_PHYS_COLS],
            on=MUT_KEYS,
            how="left",
            validate="one_to_one",
        )

    pos_tables = pos_tables or {}
    unknown = set(pos_tables) - set(POSITION_TABLES)
    if unknown:
        raise ValueError(f"unknown position tables: {sorted(unknown)}")

    ignored: dict[str, list[str]] = {}
    for name, frame in pos_tables.items():
        value_cols = POSITION_TABLES[name]
        _check_columns(frame, POS_KEYS + value_cols, name)
        _check_unique(frame, POS_KEYS, name)

        extra_ids = sorted(set(frame["protein_id"].unique()) - proteins)
        if extra_ids:
            ignored[name] = extra_ids

        out = out.merge(
            frame[POS_KEYS + value_cols],
            on=POS_KEYS,
            how="left",
            validate="many_to_one",
        )

    # msa_logodds is Tier-2 and may be delivered later; never impute it.
    if MUT_EVO_COLS[0] not in out.columns:
        out[MUT_EVO_COLS[0]] = np.nan

    out = out[FEATURE_COLUMNS].reset_index(drop=True)

    report = {
        "esm_model": esm_model,
        "ignored_proteins": ignored,
        "nan_report": nan_report(out),
    }
    return out, report


def nan_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value counts, in table column order."""
    rows = []
    for column in df.columns:
        n_missing = int(df[column].isna().sum())
        rows.append({
            "column": column,
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / max(len(df), 1), 4),
            "dtype": str(df[column].dtype),
        })
    return pd.DataFrame(rows, columns=["column", "n_missing", "pct_missing", "dtype"])


def save_mut_features(df: pd.DataFrame, path: str | Path, esm_model: str) -> None:
    """Write mut_features to Parquet with the ESM model in schema metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df[FEATURE_COLUMNS], preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"esm_model"] = esm_model.encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)
