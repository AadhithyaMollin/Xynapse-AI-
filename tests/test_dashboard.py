from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.app import DISCLAIMER, csv_bytes

ROOT = Path(__file__).resolve().parents[1]


def _require_local_predictions():
    predictions_path = ROOT / "data/processed/predictions.parquet"
    features_path = ROOT / "data/processed/features_tier1.parquet"
    if not predictions_path.is_file() or not features_path.is_file():
        pytest.skip("requires git-ignored predictions/features in data/processed; Phase 11 gate covers data-dependent UI checks")


def test_app_renders_banner_heatmap_table_and_case_study_selection(monkeypatch):
    _require_local_predictions()
    data_dir = ROOT / "data/processed"
    monkeypatch.setenv("XYNAPSE_DATA_DIR", str(data_dir))
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    assert not app.exception
    assert any(DISCLAIMER in warning.value for warning in app.warning)
    assert any(DISCLAIMER in caption.value for caption in app.caption)
    assert len(app.tabs) >= 5
    assert len(app.get("plotly_chart")) >= 1
    assert len(app.get("download_button")) == 1

    protein_selector = next(widget for widget in app.selectbox if widget.label == "Protein")
    protein_selector.set_value("P60484").run()
    assert not app.exception
    assert any("P60484" in heading.value for heading in app.subheader)
    assert any("Heatmap metric" == widget.label for widget in app.selectbox)
    assert app.get("dataframe")[1].value.shape[0] == 19 * 403
    assert "Showing 7,657 of 7,657 substitutions." in [caption.value for caption in app.caption]


def test_csv_download_payload_contains_all_p60484_mutations():
    _require_local_predictions()
    predictions = pd.read_parquet(ROOT / "data/processed/predictions.parquet")
    pten = predictions.loc[predictions.protein_id.eq("P60484")]
    payload = csv_bytes(pten)
    downloaded = pd.read_csv(BytesIO(payload))
    assert len(downloaded) == 19 * 403
    assert len(downloaded) == len(pten)
    assert set(downloaded.rank_class) == set(pten.rank_class)


def test_data_directory_falls_back_to_demo_when_primary_has_no_predictions(tmp_path, monkeypatch):
    from app.app import resolve_data_dir

    monkeypatch.delenv("XYNAPSE_DATA_DIR", raising=False)
    project_root = tmp_path
    (project_root / "data/processed").mkdir(parents=True)
    (project_root / "demo_data").mkdir()
    (project_root / "demo_data/predictions.parquet").write_bytes(b"placeholder")
    assert resolve_data_dir(project_root) == project_root / "demo_data"


def test_structure_coloring_uses_per_residue_prediction_in_display_b_factor():
    from app.app import color_pdb_by_residue

    pdb_line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 50.00           C"
    predictions = pd.DataFrame({"position": [1, 1], "p_impact": [0.6, 0.8]})
    colored, minimum, maximum = color_pdb_by_residue(pdb_line, predictions, "p_impact")
    assert float(colored.splitlines()[0][60:66]) == 0.7
    assert minimum == maximum == 0.7


def test_alphafold_fetch_checks_observed_api_fields_and_caches_structure(tmp_path, monkeypatch):
    import io
    import json
    from app.app import load_alpha_fold_pdb

    payload = [{"uniprotAccession": "P60484", "uniprotStart": 1,
                "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P60484-F1-model_v6.pdb"}]
    responses = [
        io.BytesIO(json.dumps(payload).encode("utf-8")),
        io.BytesIO(b"HEADER    TEST STRUCTURE\nATOM  test\n"),
    ]
    for response in responses:
        response.status = 200
        response.headers = {}
        response.__enter__ = lambda self=response: self
        response.__exit__ = lambda *args: False
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return responses.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    pdb_text, error = load_alpha_fold_pdb("P60484", tmp_path)
    assert error is None
    assert "ATOM  " in pdb_text
    assert len(calls) == 2
    assert (tmp_path / "P60484.pdb").is_file()
    cached, error = load_alpha_fold_pdb("P60484", tmp_path)
    assert error is None and cached == pdb_text
    assert len(calls) == 2
