import numpy as np
import pytest

from xynapse.mutations import generate_mutations
from xynapse.features import add_tier1_features, model_short_name, read_feature_model, save_features_tier1
from xynapse.schema import MUT_ESM_COLS, MUT_KEYS, MUT_PHYS_COLS, TIER1_EXTRA_COLS

MODEL = "facebook/esm2_t12_35M_UR50D"


def synthetic_esm():
    frame = generate_mutations("SYNTH", "AC")
    frame["esm_llr"] = np.arange(len(frame), dtype=float)
    frame["esm_wt_logp"] = -1.0
    frame["esm_mut_logp"] = frame["esm_llr"] - 1.0
    return frame[MUT_KEYS + MUT_ESM_COLS]


def test_tier1_extras_are_position_summaries_and_complete():
    output = add_tier1_features(synthetic_esm(), "AC")
    assert len(output) == 38
    assert list(output.columns) == MUT_KEYS + MUT_ESM_COLS + MUT_PHYS_COLS + TIER1_EXTRA_COLS
    assert output.loc[output.position.eq(1), "esm_pos_mean_llr"].nunique() == 1
    assert output.loc[output.position.eq(1), "esm_pos_min_llr"].nunique() == 1
    assert output.loc[output.position.eq(1), "esm_pos_mean_llr"].iloc[0] == 9.0
    assert output.loc[output.position.eq(1), "esm_pos_min_llr"].iloc[0] == 0.0
    assert output.loc[output.position.eq(2), "esm_pos_mean_llr"].iloc[0] == 28.0
    assert output.rel_position.tolist()[:19] == [0.5] * 19
    assert np.isfinite(output[TIER1_EXTRA_COLS].to_numpy()).all()


def test_tier1_rejects_incomplete_or_inconsistent_esm():
    frame = synthetic_esm().iloc[:-1]
    with pytest.raises(ValueError, match="expected 38 mutations"):
        add_tier1_features(frame, "AC")
    frame = synthetic_esm()
    frame.loc[0, "wt"] = "D"
    with pytest.raises(ValueError, match="WT mismatch"):
        add_tier1_features(frame, "AC")


def test_model_short_name_and_metadata_roundtrip(tmp_path):
    assert model_short_name(MODEL) == "esm2_t12_35M"
    assert model_short_name("facebook/esm2_t33_650M_UR50D") == "esm2_t33_650M"
    frame = add_tier1_features(synthetic_esm(), "AC")
    path = tmp_path / "features.parquet"
    save_features_tier1(frame, path, MODEL)
    assert read_feature_model(path) == (MODEL, "esm2_t12_35M")
