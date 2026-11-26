"""Rule-based mechanism hints from annotations and model outputs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


BINDING_FLAGS = ("is_binding_site", "is_metal", "is_dna_binding")
ANNOTATED_FLAGS = ("is_active_site", *BINDING_FLAGS, "is_interface", "is_disulfide")


def _present(row: Mapping[str, Any], key: str) -> bool:
    if key not in row:
        return False
    value = row[key]
    if value is None:
        return False
    try:
        if value != value:  # NaN / pandas.NA-safe in conjunction with bool below
            return False
    except (TypeError, ValueError):
        return False
    return True


def _true(row: Mapping[str, Any], key: str) -> bool:
    if not _present(row, key):
        return False
    value = row[key]
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _number(row: Mapping[str, Any], key: str) -> float | None:
    if not _present(row, key):
        return None
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def apply_mechanism_rules(
    row: Mapping[str, Any],
    thresholds: Mapping[str, float | None],
    mechanism_config: Mapping[str, Any] | None = None,
    rules_config: Mapping[str, Any] | None = None,
) -> str:
    """Return semicolon-separated rule names; absent annotations never imply a flag."""
    mechanism_config = mechanism_config or {}
    rules = (rules_config or {}).get("rules", {})
    active = lambda name: rules.get(name, {}).get("enabled", True)
    p_high = _number(row, "p_impact")
    ddg = _number(row, "ddg_pred")
    high_cutoff = _number(thresholds, "p_impact_high")
    ddg_cutoff = _number(thresholds, "ddg_damaging")
    high_impact = p_high is not None and high_cutoff is not None and p_high >= high_cutoff
    damaging_stability = ddg is not None and ddg_cutoff is not None and ddg >= ddg_cutoff
    flags: list[str] = []

    if active("catalytic_site") and _true(row, "is_active_site") and high_impact:
        flags.append("catalytic_site")
    if active("binding_site") and any(_true(row, key) for key in BINDING_FLAGS) and high_impact:
        flags.append("binding_site")
    buried_cutoff = mechanism_config.get("buried_rsasa")
    rsasa = _number(row, "rsasa")
    if active("folding_stability") and damaging_stability:
        if "rsasa" not in row or rsasa is not None and buried_cutoff is not None and rsasa <= float(buried_cutoff):
            flags.append("folding_stability")
    if active("interface") and _true(row, "is_interface") and high_impact:
        flags.append("interface")
    if active("disulfide") and _true(row, "is_disulfide") and (high_impact or damaging_stability):
        flags.append("disulfide")

    allosteric = rules.get("allosteric_hypothesis", {})
    minimum_distance = allosteric.get("minimum_site_distance")
    distance = _number(row, "dist_to_nearest_site")
    has_site_annotation = any(_true(row, key) for key in ANNOTATED_FLAGS)
    if (active("allosteric_hypothesis") and mechanism_config.get("allosteric", False)
            and minimum_distance is not None and distance is not None
            and distance > float(minimum_distance) and high_impact and not has_site_annotation):
        flags.append("allosteric_hypothesis")
    return ";".join(flags)
