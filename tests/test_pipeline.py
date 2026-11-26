import numpy as np
import pandas as pd
import pytest

from xynapse import pipeline
from xynapse.mutations import generate_mutations
from xynapse.schema import MUT_KEYS, PROTEIN_COLS, PRED_COLS


def _proteins_frame(accession="P12345", sequence="ACD"):
    return pd.DataFrame([{
        "protein_id": accession, "uniprot_id": accession, "sequence": sequence,
        "length": len(sequence), "structure_source": None, "mean_plddt": None,
    }], columns=PROTEIN_COLS)


def _fake_esm(protein_id, sequence, model_name, **kwargs):
    frame = generate_mutations(protein_id, sequence)
    frame["esm_llr"] = np.linspace(-1, 1, len(frame))
    frame["esm_wt_logp"] = -2.0
    frame["esm_mut_logp"] = frame["esm_llr"] - 2.0
    return frame[MUT_KEYS + ["esm_llr", "esm_wt_logp", "esm_mut_logp"]]


def _fake_predictions(frame, context):
    result = frame[MUT_KEYS].copy()
    result["ddg_pred"] = 0.0
    result["p_impact"] = 0.5
    result["rank_class"] = "uncertain"
    result["mechanism_flags"] = ""
    result["evidence_text"] = "Predictions are computational hypotheses, not experimental findings."
    result["confidence"] = "medium"
    result["ddg_confidence"] = "medium"
    result["tier"] = "tier1"
    result["esm_model"] = pipeline.ESM_MODEL
    return result[MUT_KEYS + PRED_COLS + ["ddg_confidence", "tier", "esm_model"]]


def _project(tmp_path):
    processed = tmp_path / "data/processed"
    interim = tmp_path / "data/interim"
    processed.mkdir(parents=True)
    interim.mkdir(parents=True)
    _proteins_frame().to_parquet(interim / "train_proteins.parquet", index=False)
    return tmp_path


def test_parse_fasta_and_reject_bad_response():
    header, sequence = pipeline._parse_fasta(">sp|P12345|SAMPLE\nAC\nD\n")
    assert header == "sp|P12345|SAMPLE"
    assert sequence == "ACD"
    with pytest.raises(ValueError, match="FASTA header"):
        pipeline._parse_fasta("ACD")


def test_pipeline_uses_verified_local_sequence_and_resumable_esm_cache(tmp_path, monkeypatch):
    root = _project(tmp_path)
    out = tmp_path / "out"
    score_calls = []
    monkeypatch.setattr(pipeline, "score_protein", lambda *args, **kwargs: (score_calls.append(args[0]) or _fake_esm(*args, **kwargs)))
    monkeypatch.setattr(pipeline, "load_prediction_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "predict_mutations", _fake_predictions)

    first, first_summary = pipeline.run_pipeline("P12345", out=out, project_root=root)
    second, second_summary = pipeline.run_pipeline("P12345", out=out, project_root=root)
    assert len(first) == len(second) == 57
    assert score_calls == ["P12345"]
    assert first_summary["sequence_source"].startswith("local_sequence:")
    assert first_summary["data_tier"] == "tier1_only"
    assert second_summary["n_mutations"] == 57
    assert (out / "predictions.parquet").is_file()
    assert (out / "features_tier1.parquet").is_file()
    assert (out / "pipeline_summary.json").is_file()
    pd.testing.assert_frame_equal(first, second)


def test_pipeline_rejects_non_project_esm_model(tmp_path):
    with pytest.raises(ValueError, match="unsupported ESM model"):
        pipeline.run_pipeline("P12345", esm_model="large", out=tmp_path / "out", project_root=tmp_path)
