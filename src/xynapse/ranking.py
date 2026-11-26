"""Frozen-threshold mutation classes, confidence, and fixed-template evidence."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .mechanism import ANNOTATED_FLAGS

DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
EVIDENCE_DISCLAIMER = "Computational prediction; not experimental evidence."


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rank_mutation(
    row: Mapping[str, Any],
    thresholds: Mapping[str, float | None],
    ddg_confidence: str,
    *,
    models_available: bool = True,
    features_available: bool = True,
) -> str:
    """Apply the specified class order; unavailable inputs always yield uncertain."""
    ddg = _number(row, "ddg_pred")
    p_impact = _number(row, "p_impact")
    if not models_available or not features_available or ddg is None or p_impact is None:
        return "uncertain"
    ddg_damaging = thresholds.get("ddg_damaging")
    p_high = thresholds.get("p_impact_high")
    p_low = thresholds.get("p_impact_low")
    ddg_stabilizing = thresholds.get("ddg_stabilizing")
    if (p_high is not None and p_impact >= float(p_high)) or (
        ddg_damaging is not None and ddg >= float(ddg_damaging)
    ):
        return "potentially damaging"
    if (ddg_stabilizing is not None and p_low is not None
            and ddg <= float(ddg_stabilizing) and p_impact <= float(p_low)
            and ddg_confidence != "low"):
        return "potentially beneficial (hypothesis)"
    if (p_low is not None and ddg_damaging is not None
            and p_impact <= float(p_low) and ddg < float(ddg_damaging)):
        return "tolerated"
    return "uncertain"


def confidence_fields(
    protein_length: int,
    stability_max_length: int,
    function_max_length: int,
    tier: str,
    row: Mapping[str, Any],
    confidence_config: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    """Return ddG confidence, function confidence, and explicit reason strings."""
    reasons: list[str] = []
    if protein_length > stability_max_length:
        ddg_confidence = "low"
        reasons.append(f"protein length {protein_length} exceeds stability-training maximum {stability_max_length}")
    elif tier == "tier2":
        ddg_confidence = "high"
    else:
        ddg_confidence = "medium"

    if protein_length > function_max_length:
        function_confidence = "low"
        reasons.append(f"protein length {protein_length} exceeds function-training maximum {function_max_length}")
    elif tier == "tier2":
        function_confidence = "high"
        n_eff = _number(row, "n_eff")
        plddt = _number(row, "plddt")
        if n_eff is not None and n_eff <= float(confidence_config.get("n_eff_low", 5)):
            function_confidence = "low"
            reasons.append(f"n_eff {n_eff:g} is at or below configured low-confidence threshold")
        if plddt is not None and plddt < float(confidence_config.get("plddt_low", 70)):
            function_confidence = "low"
            reasons.append(f"pLDDT {plddt:g} is below configured low-confidence threshold")
    else:
        function_confidence = "medium"
    return ddg_confidence, function_confidence, reasons


def evidence_text(row: Mapping[str, Any], reasons: list[str] | None = None) -> str:
    """Build deterministic evidence text from row values and fixed templates only."""
    ddg = _number(row, "ddg_pred")
    p_impact = _number(row, "p_impact")
    esm_llr = _number(row, "esm_llr")
    def fmt(value: float | None, digits: int = 3) -> str:
        return "unavailable" if value is None else f"{value:.{digits}f}"

    parts = [
        f"Predicted ddG: {fmt(ddg)} kcal/mol (positive is destabilizing).",
        f"Predicted function-impact probability: {fmt(p_impact)}.",
        f"ESM-2 log-likelihood ratio: {fmt(esm_llr)}.",
        f"Mechanism hints: {row.get('mechanism_flags') or 'none'}.",
        f"Model tier: {row.get('tier', 'unavailable')}; ddG confidence: {row.get('ddg_confidence', 'unavailable')}; function confidence: {row.get('confidence', 'unavailable')}.",
    ]
    optional = (("rsasa", "relative solvent accessibility"), ("plddt", "pLDDT"),
                ("cons_entropy", "conservation entropy"), ("cons_weighted", "weighted conservation"),
                ("n_eff", "effective sequence count"), ("msa_logodds", "MSA log-odds"))
    for key, label in optional:
        value = _number(row, key)
        if key in row and value is not None:
            parts.append(f"{label}: {fmt(value)}.")
    annotations = []
    for key in ANNOTATED_FLAGS:
        if key in row and row.get(key) is not None:
            annotations.append(f"{key}={row[key]}")
    distance = _number(row, "dist_to_nearest_site")
    if distance is not None:
        annotations.append(f"dist_to_nearest_site={fmt(distance)}")
    if annotations:
        parts.append("Available annotation evidence: " + ", ".join(annotations) + ".")
    if row.get("plddt") is not None:
        plddt = _number(row, "plddt")
        low = plddt is not None and plddt < 70
        if low and row.get("mechanism_flags"):
            parts.append("Annotation-based mechanism hint carries low structural confidence.")
    if reasons:
        parts.append("Confidence notes: " + "; ".join(reasons) + ".")
    parts.extend([EVIDENCE_DISCLAIMER, DISCLAIMER])
    return " ".join(parts)
