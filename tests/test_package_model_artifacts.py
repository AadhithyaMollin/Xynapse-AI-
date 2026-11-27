import json
import zipfile

import joblib
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from scripts.package_model_artifacts import FROZEN_PURPOSES, HANDOFF_NAMES, build_bundle, sha256


def _make_frozen_model_dir(path):
    path.mkdir()
    checksums = {}
    for name in FROZEN_PURPOSES:
        artifact = path / name
        if name == "function_tier1.joblib":
            bundle = {
                "estimator": object(),
                "calibrator": IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0, 1]),
                "calibration_method": "isotonic",
            }
            joblib.dump(bundle, artifact)
        elif name == "function_tier1.json":
            artifact.write_text(json.dumps({"calibration_method": "isotonic"}), encoding="utf-8")
        else:
            artifact.write_bytes(f"synthetic:{name}".encode())
        checksums[name] = sha256(artifact)
    frozen = {
        "esm_model": "facebook/esm2_t12_35M_UR50D",
        "esm_model_short": "esm2_t12_35M",
        "feature_lists": {"stability": [], "function": []},
        "model_files_sha256": checksums,
        "thresholds": {"p_impact_low": None},
    }
    (path / "FROZEN.json").write_text(json.dumps(frozen), encoding="utf-8")
    return frozen


def _write_oof(path):
    pd.DataFrame({
        "p_tier1_full": [0.05, 0.12, 0.25, 0.35, 0.55, 0.62, 0.81, 0.94],
        "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
    }).to_parquet(path, index=False)


def test_builds_frozen_bundle_and_exact_three_file_team_handoff(tmp_path):
    model_dir = tmp_path / "model"
    _make_frozen_model_dir(model_dir)
    oof_path = tmp_path / "oof.parquet"
    _write_oof(oof_path)
    archive_path = tmp_path / "frozen.zip"
    manifest_path = tmp_path / "manifest.json"

    result = build_bundle(model_dir, archive_path, manifest_path, oof_path)

    assert archive_path.is_file()
    assert result["archive_sha256"] == sha256(archive_path)
    assert result["selected_calibration_method"] == "isotonic"
    assert result["team_handoff"]["files"]["function_logistic_tier1.joblib"]["calibration_method"] == "Platt scaling / logistic regression"
    assert result["team_handoff"]["files"]["function_logistic_tier1.joblib"]["n_oof_rows_fit"] == 8
    assert "not the selected production isotonic calibrator" in result["team_handoff"]["files"]["function_logistic_tier1.joblib"]["purpose"]

    handoff_dir = archive_path.parent / "team_handoff"
    assert {path.name for path in handoff_dir.glob("*.joblib")} == set(HANDOFF_NAMES)
    assert isinstance(joblib.load(handoff_dir / "function_logistic_tier1.joblib"), LogisticRegression)
    assert isinstance(joblib.load(archive_path.parent / "function_calibrator_tier1.joblib"), IsotonicRegression)

    handoff_archive = archive_path.parent / result["team_handoff"]["archive"]
    with zipfile.ZipFile(handoff_archive) as archive:
        names = set(archive.namelist())
        assert set(HANDOFF_NAMES).issubset(names)
        embedded = json.loads(archive.read("TEAM_HANDOFF_MANIFEST.json"))
        assert embedded["files"]["function_logistic_tier1.joblib"]["sha256"] == result["team_handoff"]["files"]["function_logistic_tier1.joblib"]["sha256"]
        assert "Do not overwrite it or merge this handoff directory" in archive.read("README.md").decode()
    assert json.loads(manifest_path.read_text())["team_handoff"]["archive_sha256"] == result["team_handoff"]["archive_sha256"]


def test_bundle_refuses_any_frozen_hash_mismatch(tmp_path):
    model_dir = tmp_path / "model"
    _make_frozen_model_dir(model_dir)
    oof_path = tmp_path / "oof.parquet"
    _write_oof(oof_path)
    with (model_dir / "stability_tier1.joblib").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="frozen hash mismatch"):
        build_bundle(model_dir, tmp_path / "frozen.zip", tmp_path / "manifest.json", oof_path)
