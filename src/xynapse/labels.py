"""Source-to-project label conversions; all positions are 1-based."""
from __future__ import annotations

import math

import pandas as pd


def stability_project_ddg(delta_g_wt: float, delta_g_mut: float) -> float:
    """Convert source deltaG values to project sign: positive means destabilizing."""
    wt = float(delta_g_wt)
    mutant = float(delta_g_mut)
    if not math.isfinite(wt) or not math.isfinite(mutant):
        raise ValueError("deltaG values must be finite")
    return wt - mutant


def function_project_label(source_binary_label: int, label_1_is_tolerated: bool) -> int:
    """Map an assay binary value to project label 1=damaging."""
    value = float(source_binary_label)
    if not math.isfinite(value) or value not in (0.0, 1.0):
        raise ValueError(f"source binary label must be 0 or 1, got {source_binary_label!r}")
    return int(1 - value if label_1_is_tolerated else value)


def within_assay_percentile(scores: pd.Series) -> pd.Series:
    """Return average-tie percentile ranks without comparing across assays."""
    numeric = pd.to_numeric(scores, errors="coerce")
    return numeric.rank(method="average", pct=True)
