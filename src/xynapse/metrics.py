"""Regression metrics that safely handle NaN and constant inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _paired_finite(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"metric arrays must have equal shape, got {true.shape} and {pred.shape}")
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def correlation(y_true, y_pred, method: str = "spearman") -> float:
    """Return a finite-pair correlation, or NaN for insufficient/constant values."""
    true, pred = _paired_finite(y_true, y_pred)
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    if len(true) < 2 or np.unique(true).size < 2 or np.unique(pred).size < 2:
        return float("nan")
    result = spearmanr(true, pred).statistic if method == "spearman" else pearsonr(true, pred).statistic
    return float(result) if np.isfinite(result) else float("nan")


def spearman(y_true, y_pred) -> float:
    return correlation(y_true, y_pred, "spearman")


def pearson(y_true, y_pred) -> float:
    return correlation(y_true, y_pred, "pearson")


def rmse(y_true, y_pred) -> float:
    true, pred = _paired_finite(y_true, y_pred)
    if not len(true):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(true - pred))))


def regression_metrics(y_true, y_pred) -> dict[str, float | int]:
    true, pred = _paired_finite(y_true, y_pred)
    return {
        "n": int(len(true)),
        "spearman": spearman(true, pred),
        "pearson": pearson(true, pred),
        "rmse": rmse(true, pred),
    }


def per_protein_spearman(
    frame: pd.DataFrame,
    *,
    protein_col: str = "protein_id",
    true_col: str = "y_true",
    pred_col: str = "y_pred",
    min_variants: int = 30,
) -> tuple[pd.DataFrame, int]:
    """Return per-protein correlations with the qualifying protein count."""
    missing = {protein_col, true_col, pred_col} - set(frame.columns)
    if missing:
        raise ValueError(f"per-protein metrics missing columns: {sorted(missing)}")
    records = []
    for protein_id, group in frame.groupby(protein_col, sort=True):
        true, pred = _paired_finite(group[true_col], group[pred_col])
        if len(true) < min_variants:
            continue
        value = spearman(true, pred)
        records.append({"protein_id": protein_id, "n_variants": len(true), "spearman": value})
    result = pd.DataFrame(records, columns=["protein_id", "n_variants", "spearman"])
    return result, int(len(result))
