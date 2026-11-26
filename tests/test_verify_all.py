from scripts.make_final_report import render
from scripts.verify_all import _clean_clone_payloads, _last_lines


def test_clean_clone_policy_allows_placeholders_and_model_metadata(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/.gitkeep").write_text("")
    (tmp_path / "models/esm2_t12_35M").mkdir(parents=True)
    (tmp_path / "models/esm2_t12_35M/FROZEN.json").write_text("{}")

    data_files, model_files = _clean_clone_payloads(tmp_path)

    assert data_files == []
    assert model_files == []


def test_clean_clone_policy_rejects_data_and_model_binaries(tmp_path):
    data_path = tmp_path / "data/processed/labels.parquet"
    model_path = tmp_path / "models/esm2_t12_35M/model.joblib"
    data_path.parent.mkdir(parents=True)
    model_path.parent.mkdir(parents=True)
    data_path.write_bytes(b"data")
    model_path.write_bytes(b"model")

    data_files, model_files = _clean_clone_payloads(tmp_path)

    assert data_files == [data_path]
    assert model_files == [model_path]


def test_last_lines_returns_compact_command_summary():
    assert _last_lines("one\ntwo\nthree\n", limit=2) == "two | three"
    assert _last_lines("\n") == "no output"


def test_final_report_is_sourced_and_carries_required_sections():
    report = render()

    assert "Predictions are computational hypotheses, not experimental findings." in report
    assert "ProteinGym direction remains unresolved" in report
    assert "No Tier-2 model was trained" in report
    assert "Frozen thresholds and plan parameters" in report
    assert "reports/gates/phase_13_gate.txt" in report
    assert "single recorded evaluations" in report
