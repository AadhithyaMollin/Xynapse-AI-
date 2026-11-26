import json

import numpy as np
import pytest

from xynapse.freeze import evaluate_once, freeze_models, verify_frozen
from xynapse.thresholds import derive_oof_thresholds, largest_threshold_for_npv


def test_largest_npv_threshold_and_unreachable_case():
    threshold, npv, count = largest_threshold_for_npv([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.9)
    assert threshold == pytest.approx(0.2)
    assert npv == pytest.approx(1.0)
    assert count == 2
    threshold, npv, count = largest_threshold_for_npv([0, 1], [0.4, 0.3], 0.95)
    assert threshold is None
    assert np.isnan(npv)
    assert count == 0


def test_thresholds_use_oof_and_disable_stabilizing_class_below_minimum():
    values, evidence = derive_oof_thresholds(
        [-2, -1.5, 0, 0.5, 1, 2], [-2.1, -1.4, 0.1, 0.4, 1.2, 1.8],
        [0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], min_stabilizing_positives=50,
    )
    assert values["ddg_damaging"] is not None
    assert values["ddg_stabilizing"] is None
    assert 0 <= values["p_impact_high"] <= 1
    assert values["p_impact_low"] == pytest.approx(0.2)
    assert len(evidence) == 4


def test_freeze_records_all_files_and_detects_mutation(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "a.joblib").write_bytes(b"model-a")
    (model_dir / "b.json").write_text('{"features": ["x"]}')
    frozen = freeze_models(model_dir, {"p_high": 0.8}, {"task": ["x"]},
                           esm_model="facebook/esm2_t12_35M_UR50D")
    ok, problems = verify_frozen(model_dir / "FROZEN.json", model_dir)
    assert ok and not problems
    assert set(frozen["model_files_sha256"]) == {"a.joblib", "b.json"}
    (model_dir / "a.joblib").write_bytes(b"changed")
    ok, problems = verify_frozen(model_dir / "FROZEN.json", model_dir)
    assert not ok
    assert any("SHA-256 mismatch: a.joblib" in problem for problem in problems)


def test_evaluate_once_refuses_second_run_and_records_result(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    calls = []

    def evaluator():
        calls.append(1)
        return {"n": 3}

    assert evaluate_once("heldout", evaluator, path) == {"n": 3}
    with pytest.raises(RuntimeError, match="already recorded"):
        evaluate_once("heldout", evaluator, path)
    assert calls == [1]
    entry = json.loads(path.read_text())["evaluations"]["heldout"]
    assert entry["status"] == "completed"
    assert entry["result"] == {"n": 3}


def test_evaluate_once_override_requires_reason_and_logs_it(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    evaluate_once("external", lambda: {"run": 1}, path)
    monkeypatch.setenv("XYNAPSE_ALLOW_REEVAL", "1")
    monkeypatch.delenv("XYNAPSE_REEVAL_REASON", raising=False)
    with pytest.raises(RuntimeError, match="REASON is required"):
        evaluate_once("external", lambda: {"run": 2}, path)
    monkeypatch.setenv("XYNAPSE_REEVAL_REASON", "unit test exercise")
    assert evaluate_once("external", lambda: {"run": 2}, path) == {"run": 2}
    entry = json.loads(path.read_text())["evaluations"]["external"]
    assert entry["re_evaluation_reason"] == "unit test exercise"
    assert entry["result"] == {"run": 2}
