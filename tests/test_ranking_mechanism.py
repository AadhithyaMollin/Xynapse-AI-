import pytest

from xynapse.mechanism import apply_mechanism_rules
from xynapse.ranking import confidence_fields, evidence_text, rank_mutation


THRESHOLDS = {"p_impact_high": 0.6, "p_impact_low": 0.2,
              "ddg_damaging": 1.0, "ddg_stabilizing": -1.0}


def test_each_annotation_mechanism_rule_and_missing_annotation_behavior():
    assert apply_mechanism_rules(
        {"is_active_site": True, "p_impact": 0.8, "ddg_pred": 0.0}, THRESHOLDS
    ) == "catalytic_site"
    assert apply_mechanism_rules(
        {"is_metal": True, "p_impact": 0.8, "ddg_pred": 0.0}, THRESHOLDS
    ) == "binding_site"
    assert apply_mechanism_rules(
        {"ddg_pred": 1.2, "rsasa": 0.2, "p_impact": 0.1}, THRESHOLDS,
        {"buried_rsasa": 0.25},
    ) == "folding_stability"
    assert apply_mechanism_rules(
        {"is_interface": True, "p_impact": 0.8, "ddg_pred": 0.0}, THRESHOLDS
    ) == "interface"
    assert apply_mechanism_rules(
        {"is_disulfide": True, "p_impact": 0.1, "ddg_pred": 1.2}, THRESHOLDS
    ) == "folding_stability;disulfide"
    assert apply_mechanism_rules({"p_impact": 0.9, "ddg_pred": 1.4}, THRESHOLDS) == "folding_stability"
    assert apply_mechanism_rules({"p_impact": 0.9}, THRESHOLDS) == ""


def test_folding_stability_requires_buried_accessibility_when_value_is_present():
    assert apply_mechanism_rules(
        {"ddg_pred": 1.2, "rsasa": 0.8}, THRESHOLDS, {"buried_rsasa": 0.25}
    ) == ""
    assert apply_mechanism_rules(
        {"ddg_pred": 1.2, "rsasa": None}, THRESHOLDS, {"buried_rsasa": 0.25}
    ) == ""


def test_allosteric_hypothesis_is_disabled_by_default_and_requires_configured_distance():
    row = {"p_impact": 0.9, "dist_to_nearest_site": 25.0}
    assert apply_mechanism_rules(row, THRESHOLDS) == ""
    rules = {"rules": {"allosteric_hypothesis": {"enabled": True, "minimum_site_distance": 10.0}}}
    assert apply_mechanism_rules(row, THRESHOLDS, {"allosteric": True}, rules) == "allosteric_hypothesis"
    assert apply_mechanism_rules(
        {**row, "is_active_site": True}, THRESHOLDS, {"allosteric": True}, rules
    ) == "catalytic_site"
    rules["rules"]["allosteric_hypothesis"]["minimum_site_distance"] = None
    assert apply_mechanism_rules(row, THRESHOLDS, {"allosteric": True}, rules) == ""


def test_ranking_order_and_unavailable_class_thresholds():
    assert rank_mutation({"ddg_pred": 1.0, "p_impact": 0.1}, THRESHOLDS, "medium") == "potentially damaging"
    assert rank_mutation({"ddg_pred": -1.2, "p_impact": 0.1}, THRESHOLDS, "medium") == "potentially beneficial (hypothesis)"
    assert rank_mutation({"ddg_pred": -0.1, "p_impact": 0.1}, THRESHOLDS, "medium") == "tolerated"
    assert rank_mutation({"ddg_pred": -0.1, "p_impact": 0.3}, THRESHOLDS, "medium") == "uncertain"
    no_low_cutoff = {**THRESHOLDS, "p_impact_low": None}
    assert rank_mutation({"ddg_pred": -2, "p_impact": 0.0}, no_low_cutoff, "medium") == "uncertain"
    assert rank_mutation({"ddg_pred": -2, "p_impact": 0.0}, THRESHOLDS, "low") == "tolerated"
    assert rank_mutation({"ddg_pred": 1.2, "p_impact": 0.1}, THRESHOLDS, "low") == "potentially damaging"
    assert rank_mutation({"ddg_pred": 0.0}, THRESHOLDS, "medium") == "uncertain"
    assert rank_mutation({"ddg_pred": 0.0, "p_impact": 0.1}, THRESHOLDS, "medium", models_available=False) == "uncertain"


def test_confidence_uses_training_lengths_and_tier_inputs():
    ddg, function, reasons = confidence_fields(100, 72, 465, "tier1", {}, {})
    assert (ddg, function) == ("low", "medium")
    assert any("exceeds stability-training maximum" in reason for reason in reasons)
    ddg, function, _ = confidence_fields(500, 72, 465, "tier2", {"n_eff": 4, "plddt": 69},
                                          {"n_eff_low": 5, "plddt_low": 70})
    assert (ddg, function) == ("low", "low")
    ddg, function, reasons = confidence_fields(400, 72, 465, "tier2", {"n_eff": 4, "plddt": 69},
                                               {"n_eff_low": 5, "plddt_low": 70})
    assert (ddg, function) == ("low", "low")
    assert any("n_eff" in reason for reason in reasons)
    assert any("pLDDT" in reason for reason in reasons)
    ddg, function, _ = confidence_fields(50, 72, 465, "tier1", {}, {})
    assert (ddg, function) == ("medium", "medium")


def test_evidence_text_uses_row_values_and_disclaimers():
    text = evidence_text({"ddg_pred": 1.23456, "p_impact": 0.8, "esm_llr": -2.0,
                          "mechanism_flags": "catalytic_site", "tier": "tier1",
                          "ddg_confidence": "low", "confidence": "medium", "plddt": 60,
                          "is_active_site": True})
    assert "1.235 kcal/mol" in text
    assert "catalytic_site" in text
    assert "low structural confidence" in text
    assert "Computational prediction; not experimental evidence." in text
    assert "Predictions are computational hypotheses, not experimental findings." in text


def test_non_numeric_values_do_not_assign_flags_or_classes():
    assert apply_mechanism_rules({"is_active_site": None, "p_impact": "unknown"}, THRESHOLDS) == ""
    assert rank_mutation({"ddg_pred": "bad", "p_impact": 1}, THRESHOLDS, "medium") == "uncertain"
