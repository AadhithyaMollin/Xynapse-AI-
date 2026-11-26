"""Binary classification metrics and assay-level summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, matthews_corrcoef,
                             roc_auc_score)


def finite_binary_pairs(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(score, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"label and score arrays must have equal shape, got {y.shape} and {p.shape}")
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    valid = np.isin(y, [0.0, 1.0])
    return y[valid].astype(int), p[valid]


def best_mcc_threshold(y_true, score) -> tuple[float, float]:
    """Maximize MCC over every distinct score boundary in O(n log n) time."""
    y, p = finite_binary_pairs(y_true, score)
    if not len(y) or np.unique(y).size < 2:
        return float("nan"), float("nan")
    order = np.argsort(-p, kind="mergesort")
    sorted_scores = p[order]
    sorted_labels = y[order]
    cumulative_positive = np.cumsum(sorted_labels, dtype=np.int64)
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True]) + 1
    boundary_thresholds = sorted_scores[group_ends - 1]
    extra_thresholds = np.array([0.0, 0.5, 1.0, np.nextafter(sorted_scores[0], np.inf)])
    candidates = np.unique(np.concatenate((boundary_thresholds, extra_thresholds)))
    counts = np.searchsorted(-sorted_scores, -candidates, side="right")
    true_positive = np.zeros(len(counts), dtype=float)
    has_predicted_positive = counts > 0
    true_positive[has_predicted_positive] = cumulative_positive[counts[has_predicted_positive] - 1]
    predicted_positive = counts.astype(float)
    total_positive = float(y.sum())
    total_negative = float(len(y) - y.sum())
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive
    numerator = true_positive * true_negative - false_positive * false_negative
    denominator = np.sqrt((true_positive + false_positive) * (true_positive + false_negative)
                          * (true_negative + false_positive) * (true_negative + false_negative))
    mcc_values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    maximum = float(np.max(mcc_values))
    best_candidates = candidates[np.isclose(mcc_values, maximum, rtol=0.0, atol=1e-15)]
    threshold = float(min(best_candidates, key=lambda value: (abs(value - 0.5), value)))
    return threshold, maximum


def binary_metrics(y_true, score, threshold: float | None = None) -> dict[str, float | int]:
    y, p = finite_binary_pairs(y_true, score)
    if len(y) == 0:
        return {"n_rows": 0, "n_positive": 0, "n_negative": 0, "prevalence": float("nan"),
                "auroc": float("nan"), "auprc": float("nan"), "mcc": float("nan"),
                "threshold": float("nan"), "brier": float("nan")}
    if threshold is None or not np.isfinite(threshold):
        threshold, _ = best_mcc_threshold(y, p)
    has_both = np.unique(y).size == 2
    is_probability = bool(np.all((p >= 0.0) & (p <= 1.0)))
    return {
        "n_rows": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "mcc": float(matthews_corrcoef(y, p >= threshold)) if has_both and np.isfinite(threshold) else float("nan"),
        "threshold": float(threshold),
        "brier": float(brier_score_loss(y, p)) if is_probability else float("nan"),
    }


def calibration_curve_data(y_true, probability, *, n_bins: int = 10) -> pd.DataFrame:
    y, p = finite_binary_pairs(y_true, probability)
    if not len(y):
        return pd.DataFrame(columns=["bin", "mean_predicted_value", "fraction_of_positives", "n_rows"])
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    records = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            records.append({"bin": int(bin_id), "mean_predicted_value": float(p[mask].mean()),
                            "fraction_of_positives": float(y[mask].mean()), "n_rows": int(mask.sum())})
    return pd.DataFrame(records)


def per_assay_auroc(frame: pd.DataFrame, score_column: str, *, min_per_class: int = 1) -> pd.DataFrame:
    required = {"assay_id", "label", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"per-assay AUROC missing columns: {sorted(missing)}")
    records = []
    for assay_id, group in frame.groupby("assay_id", sort=True):
        y, p = finite_binary_pairs(group.label, group[score_column])
        positives, negatives = int(y.sum()), int(len(y) - y.sum())
        value = float(roc_auc_score(y, p)) if positives >= min_per_class and negatives >= min_per_class else float("nan")
        records.append({"assay_id": assay_id, "auroc": value, "n_rows": len(y),
                        "n_positive": positives, "n_negative": negatives})
    return pd.DataFrame(records)
