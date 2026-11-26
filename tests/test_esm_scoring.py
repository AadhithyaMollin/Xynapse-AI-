import numpy as np
import pandas as pd
import pytest
import torch

from xynapse.esm_scoring import (
    load_esm,
    read_esm_model,
    save_mut_esm,
    score_protein,
)
from xynapse.schema import AA, MUT_ESM_COLS, MUT_KEYS, PROTEIN_COLS
from xynapse.validate import validate_table

MODEL = "facebook/esm2_t6_8M_UR50D"  # smallest ESM-2: fast, no GPU needed
SEQ = "MKTAYIAKQRQ"
PID = "TEST"


@pytest.fixture(scope="module")
def scored():
    return score_protein(PID, SEQ, MODEL, batch_size=8)


def test_row_count_and_columns(scored):
    assert len(scored) == 19 * len(SEQ)
    assert list(scored.columns) == MUT_KEYS + MUT_ESM_COLS


def test_llr_is_difference(scored):
    assert np.allclose(
        scored["esm_llr"],
        scored["esm_mut_logp"] - scored["esm_wt_logp"],
        atol=1e-6,
    )


def test_logp_values_are_valid(scored):
    for column in ("esm_wt_logp", "esm_mut_logp"):
        values = scored[column].to_numpy()
        assert np.isfinite(values).all()
        assert (values <= 1e-6).all()


def test_manual_forward_pass_matches(scored):
    tokenizer, model = load_esm(MODEL, device="cpu")
    position = 3

    input_ids = tokenizer(SEQ, return_tensors="pt")["input_ids"].clone()
    input_ids[0, position] = tokenizer.mask_token_id
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    logp = torch.log_softmax(logits[0, position].float(), dim=-1)

    group = scored[scored["position"] == position]
    wt = SEQ[position - 1]
    assert abs(float(logp[tokenizer.convert_tokens_to_ids(wt)])
               - group["esm_wt_logp"].iloc[0]) < 1e-4
    for _, row in group.iterrows():
        expected = float(logp[tokenizer.convert_tokens_to_ids(row["mut"])])
        assert abs(expected - row["esm_mut_logp"]) < 1e-4


def test_matches_contract(scored):
    proteins = pd.DataFrame(
        [{
            "protein_id": PID,
            "uniprot_id": PID,
            "sequence": SEQ,
            "length": len(SEQ),
            "structure_source": "test",
            "mean_plddt": 90.0,
        }],
        columns=PROTEIN_COLS,
    )
    assert validate_table(scored, "mut_esm", proteins) == []


def test_metadata_round_trip(scored, tmp_path):
    path = tmp_path / "mut_esm.parquet"
    save_mut_esm(scored, path, MODEL)
    assert read_esm_model(path) == MODEL

    back = pd.read_parquet(path)
    assert list(back.columns) == MUT_KEYS + MUT_ESM_COLS
    assert len(back) == len(scored)


def test_too_long_sequence_raises():
    with pytest.raises(ValueError, match="max_len"):
        score_protein(PID, "M" * 1023, MODEL)
