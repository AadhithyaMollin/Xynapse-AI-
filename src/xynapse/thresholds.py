"""Threshold derivation helpers that consume OOF predictions only."""
from __future__ import annotations

import numpy as np

from .classification_metrics import best_mcc_threshold


def largest_threshold_for_npv(y_true, probability, target_npv: float = 0.90) -> tuple[float | None, float, int]:
    """Find the largest p cutoff assigning tolerated (negative) with requested NPV."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("labels and probabilities must have equal shape")
    valid = np.isfinite(y) & np.isfinite(p) & np.isin(y, [0, 1])
    y, p = y[valid].astype(int), p[valid]
    if not 0.0 < target_npv <= 1.0:
        raise ValueError("target_npv must be in (0, 1]")
    candidates = np.unique(np.concatenate(([0.0, 1.0], np.clip(p, 0.0, 1.0))))
    eligible: list[tuple[float, float, int]] = []
    for cutoff in candidates:
        predicted_negative = p <= cutoff
        n_assigned = int(predicted_negative.sum())
        if not n_assigned:
            continue
        npv = float((y[predicted_negative] == 0).mean())
        if npv >= target_npv:
            eligible.append((float(cutoff), npv, n_assigned))
    if not eligible:
        return None, float("nan"), 0
    return max(eligible, key=lambda item: item[0])


def derive_oof_thresholds(ddg_true, ddg_pred, impact_true, impact_probability,
                          *, ddg_label_cutoff: float = 1.0,
                          min_stabilizing_positives: int = 50,
                          target_npv: float = 0.90) -> tuple[dict, list[dict]]:
    """Derive damaging, stabilizing, and function thresholds from OOF arrays."""
    ddg_true = np.asarray(ddg_true, dtype=float)
    ddg_pred = np.asarray(ddg_pred, dtype=float)
    impact_true = np.asarray(impact_true, dtype=float)
    impact_probability = np.asarray(impact_probability, dtype=float)
    if ddg_true.shape != ddg_pred.shape or impact_true.shape != impact_probability.shape:
        raise ValueError("each OOF target and prediction pair must have equal shapes")
    ddg_damage_target = (ddg_true >= ddg_label_cutoff).astype(int)
    damage_cut, damage_mcc = best_mcc_threshold(ddg_damage_target, ddg_pred)
    ddg_stable_target = (ddg_true <= -ddg_label_cutoff).astype(int)
    n_stable_positive = int(ddg_stable_target.sum())
    stable_score_cut, stable_mcc = best_mcc_threshold(ddg_stable_target, -ddg_pred)
    stable_cut = (-stable_score_cut if n_stable_positive >= min_stabilizing_positives
                  and np.isfinite(stable_score_cut) else None)
    impact_high, impact_mcc = best_mcc_threshold(impact_true, impact_probability)
    impact_low, low_npv, n_low = largest_threshold_for_npv(impact_true, impact_probability, target_npv)
    thresholds = {
        "ddg_damaging": float(damage_cut),
        "ddg_stabilizing": float(stable_cut) if stable_cut is not None else None,
        "p_impact_high": float(impact_high),
        "p_impact_low": float(impact_low) if impact_low is not None else None,
    }
    records = [
        {"threshold": "ddg_damaging", "value": thresholds["ddg_damaging"],
         "criterion": f"MCC max for true ddG >= {ddg_label_cutoff:g} kcal/mol",
         "mcc": damage_mcc, "n_rows": int(len(ddg_true)), "n_positive": int(ddg_damage_target.sum()),
         "target_npv": np.nan, "observed_npv": np.nan, "n_assigned_negative": 0},
        {"threshold": "ddg_stabilizing", "value": thresholds["ddg_stabilizing"],
         "criterion": (f"MCC max for true ddG <= {-ddg_label_cutoff:g} kcal/mol; null if positives < {min_stabilizing_positives}"),
         "mcc": stable_mcc if n_stable_positive >= min_stabilizing_positives else np.nan,
         "n_rows": int(len(ddg_true)), "n_positive": n_stable_positive,
         "target_npv": np.nan, "observed_npv": np.nan, "n_assigned_negative": 0},
        {"threshold": "p_impact_high", "value": thresholds["p_impact_high"],
         "criterion": "MCC max for OOF calibrated function probability",
         "mcc": impact_mcc, "n_rows": int(len(impact_true)), "n_positive": int(impact_true.sum()),
         "target_npv": np.nan, "observed_npv": np.nan, "n_assigned_negative": 0},
        {"threshold": "p_impact_low", "value": thresholds["p_impact_low"],
         "criterion": f"largest OOF probability cutoff with NPV >= {target_npv:.2f}; null if unreachable",
         "mcc": np.nan, "n_rows": int(len(impact_true)), "n_positive": int(impact_true.sum()),
         "target_npv": target_npv, "observed_npv": low_npv, "n_assigned_negative": n_low},
    ]
    return thresholds, records
