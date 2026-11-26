"""Tier-1 mutation feature assembly and ESM model metadata helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .physchem import add_physchem
from .schema import MUT_ESM_COLS, MUT_KEYS, MUT_PHYS_COLS, TIER1_EXTRA_COLS

ESM_MODEL_METADATA_KEY = b"esm_model"
ESM_MODEL_SHORT_METADATA_KEY = b"esm_model_short"


def model_short_name(model_name: str) -> str:
    base = model_name.rstrip("/").split("/")[-1]
    suffix = "_UR50D"
    if not base.startswith("esm2_") or not base.endswith(suffix):
        raise ValueError(f"Unsupported ESM-2 model name for project short-name mapping: {model_name}")
    return base[:-len(suffix)]


def add_tier1_features(esm: pd.DataFrame, sequence: str) -> pd.DataFrame:
    """Add label-free per-position summaries, relative position, and physchem."""
    required = MUT_KEYS + MUT_ESM_COLS
    missing = [column for column in required if column not in esm.columns]
    if missing:
        raise ValueError(f"Tier-1 assembly missing input columns: {missing}")
    if esm["protein_id"].nunique() != 1:
        raise ValueError("Tier-1 assembly requires exactly one protein per input frame")
    sequence = str(sequence).strip().upper()
    protein_id = str(esm["protein_id"].iloc[0])
    if not sequence:
        raise ValueError("Tier-1 assembly requires a non-empty sequence")
    if len(esm) != 19 * len(sequence):
        raise ValueError(f"{protein_id}: expected {19 * len(sequence)} mutations, found {len(esm)}")
    if esm.duplicated(MUT_KEYS).any():
        raise ValueError(f"{protein_id}: duplicate mutation keys in ESM features")
    counts = esm.groupby("position").size()
    if len(counts) != len(sequence) or not counts.reindex(range(1, len(sequence) + 1), fill_value=0).eq(19).all():
        raise ValueError(f"{protein_id}: each position must have exactly 19 substitution rows")
    if not np.isfinite(esm[MUT_ESM_COLS].to_numpy(dtype=float)).all():
        raise ValueError(f"{protein_id}: non-finite ESM features")
    wt_by_position = esm.groupby("position")["wt"].first()
    for position, wt in wt_by_position.items():
        if position < 1 or position > len(sequence) or sequence[int(position) - 1] != wt:
            raise ValueError(f"{protein_id}: WT mismatch at position {position}")

    position_summary = (esm.groupby(["protein_id", "position"], as_index=False)
                        .agg(esm_pos_mean_llr=("esm_llr", "mean"),
                             esm_pos_min_llr=("esm_llr", "min")))
    out = esm.merge(position_summary, on=["protein_id", "position"], how="left", validate="many_to_one")
    out["rel_position"] = out["position"].astype(float) / float(len(sequence))
    out = add_physchem(out)
    columns = MUT_KEYS + MUT_ESM_COLS + MUT_PHYS_COLS + TIER1_EXTRA_COLS
    if out[columns].isna().any().any() or not np.isfinite(out[TIER1_EXTRA_COLS].to_numpy(dtype=float)).all():
        raise ValueError(f"{protein_id}: Tier-1 output contains missing or non-finite values")
    return out[columns].sort_values(MUT_KEYS).reset_index(drop=True)


def save_features_tier1(frame: pd.DataFrame, path: str | Path, model_name: str) -> None:
    """Write Tier-1 Parquet with full and short ESM model identifiers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = MUT_KEYS + MUT_ESM_COLS + MUT_PHYS_COLS + TIER1_EXTRA_COLS
    table = pa.Table.from_pandas(frame[columns], preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[ESM_MODEL_METADATA_KEY] = model_name.encode("utf-8")
    metadata[ESM_MODEL_SHORT_METADATA_KEY] = model_short_name(model_name).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


def read_feature_model(path: str | Path) -> tuple[str, str]:
    metadata = pq.read_schema(path).metadata or {}
    full = metadata.get(ESM_MODEL_METADATA_KEY)
    short = metadata.get(ESM_MODEL_SHORT_METADATA_KEY)
    if full is None or short is None:
        raise KeyError(f"{path}: ESM model metadata is incomplete")
    return full.decode("utf-8"), short.decode("utf-8")
