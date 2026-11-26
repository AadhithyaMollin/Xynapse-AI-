import pandas as pd
import pytest
from xynapse.schema import AA, MUT_ESM_COLS, MUT_KEYS, MUT_PHYS_COLS, TIER1_EXTRA_COLS, TABLES, mutation_label, check_columns


def test_alphabet():
    assert len(AA) == 20 and len(set(AA)) == 20


def test_label():
    assert mutation_label("R", 175, "H") == "R175H"


def test_check_columns():
    df = pd.DataFrame({"protein_id": ["P1"], "position": [1]})
    check_columns(df, ["protein_id", "position"])
    with pytest.raises(ValueError):
        check_columns(df, ["wt"])


def test_tier1_features_contract_is_additive_and_complete():
    assert TIER1_EXTRA_COLS == ["esm_pos_mean_llr", "esm_pos_min_llr", "rel_position"]
    assert TABLES["features_tier1"]["level"] == "mutation"
    assert TABLES["features_tier1"]["complete"] is True
    assert TABLES["features_tier1"]["required"] == MUT_KEYS + MUT_ESM_COLS + MUT_PHYS_COLS + TIER1_EXTRA_COLS


def test_external_clinvar_contract_is_additive():
    assert TABLES["external_clinvar"]["level"] == "mutation"
    assert TABLES["external_clinvar"]["required"] == MUT_KEYS + [
        "label", "review_stars", "clinvar_variation_id", "gene"
    ]
