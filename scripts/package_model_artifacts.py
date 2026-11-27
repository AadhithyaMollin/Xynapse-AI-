#!/usr/bin/env python3
"""Package frozen models and the three-file teammate handoff from saved OOF outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHORT = "esm2_t12_35M"
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
FROZEN_PURPOSES = {
    "stability_tier1.joblib": "Tier-1 LightGBM stability regression estimator.",
    "function_tier1.joblib": (
        "Tier-1 function inference bundle: fitted function estimator plus selected isotonic calibrator."
    ),
    "function_logistic_tier1.joblib": (
        "Frozen auxiliary logistic-regression comparison baseline; this is not the selected calibrator."
    ),
    "stability_tier1.json": "Stability feature inventory, training metadata, and estimator filename.",
    "function_tier1.json": "Function feature inventory, calibration metadata, and estimator filename.",
}
HANDOFF_NAMES = (
    "stability_tier1.joblib",
    "function_tier1.joblib",
    "function_logistic_tier1.joblib",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _verified_frozen_files(model_dir: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    frozen_path = model_dir / "FROZEN.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"missing frozen inventory: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("esm_model_short") != MODEL_SHORT:
        raise ValueError(f"unexpected ESM model short name: {frozen.get('esm_model_short')!r}")
    expected = frozen.get("model_files_sha256")
    if not isinstance(expected, dict) or set(expected) != set(FROZEN_PURPOSES):
        raise ValueError("FROZEN.json model_files_sha256 does not match the expected frozen inventory")
    entries: dict[str, dict[str, Any]] = {}
    for name, purpose in FROZEN_PURPOSES.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen artifact: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected[name]:
            raise ValueError(f"frozen hash mismatch for {name}: expected {expected[name]}, got {actual_hash}")
        entries[name] = {
            "bytes": path.stat().st_size,
            "frozen": True,
            "purpose": purpose,
            "sha256": actual_hash,
        }
    frozen_path_hash = sha256(frozen_path)
    entries["FROZEN.json"] = {
        "bytes": frozen_path.stat().st_size,
        "frozen": True,
        "purpose": "Frozen inventory of model-file hashes, thresholds, feature lists, and ESM identity.",
        "sha256": frozen_path_hash,
    }
    return frozen, expected, entries


def _extract_selected_calibrator(model_dir: Path, output_dir: Path, frozen_files: dict[str, str]) -> tuple[Path, str]:
    import joblib

    function_bundle_path = model_dir / "function_tier1.joblib"
    if sha256(function_bundle_path) != frozen_files["function_tier1.joblib"]:
        raise ValueError("function model changed before calibrator extraction")
    function_meta = json.loads((model_dir / "function_tier1.json").read_text(encoding="utf-8"))
    function_bundle = joblib.load(function_bundle_path)
    if not isinstance(function_bundle, dict) or not {"estimator", "calibrator", "calibration_method"}.issubset(function_bundle):
        raise ValueError("function_tier1.joblib lacks the estimator/calibrator inference bundle")
    method = function_bundle["calibration_method"]
    if method != function_meta.get("calibration_method"):
        raise ValueError("function bundle calibration method disagrees with function metadata")
    if not hasattr(function_bundle["calibrator"], "predict"):
        raise ValueError("selected function calibrator does not expose predict")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "function_calibrator_tier1.joblib"
    joblib.dump(function_bundle["calibrator"], output)
    return output, method


def _build_team_handoff(
    model_dir: Path,
    handoff_dir: Path,
    handoff_archive: Path,
    oof_path: Path,
    frozen: dict[str, Any],
    frozen_files: dict[str, str],
) -> dict[str, Any]:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    oof_path = oof_path.expanduser().resolve()
    if not oof_path.is_file():
        raise FileNotFoundError(f"missing saved function OOF table: {oof_path}")
    oof = pd.read_parquet(oof_path)
    required = {"y_true", "p_tier1_full"}
    if not required.issubset(oof.columns):
        raise ValueError(f"function OOF table missing columns {sorted(required - set(oof.columns))}")
    scores = oof["p_tier1_full"].to_numpy(dtype=float)
    labels = oof["y_true"].to_numpy(dtype=int)
    if len(scores) == 0 or not np.isfinite(scores).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("function OOF scores/labels are empty, non-finite, or not binary")

    # This is an alternative Platt calibrator fit only to grouped OOF scores;
    # production inference retains the selected isotonic calibrator in the frozen bundle.
    platt = LogisticRegression(max_iter=1000, random_state=42)
    platt.fit(scores.reshape(-1, 1), labels)

    handoff_dir = handoff_dir.expanduser().resolve()
    handoff_archive = handoff_archive.expanduser().resolve()
    handoff_dir.mkdir(parents=True, exist_ok=True)
    for name in HANDOFF_NAMES[:2]:
        shutil.copy2(model_dir / name, handoff_dir / name)
    platt_path = handoff_dir / "function_logistic_tier1.joblib"
    joblib.dump(platt, platt_path)

    oof_hash = sha256(oof_path)
    handoff_files = {
        "stability_tier1.joblib": {
            "bytes": (handoff_dir / HANDOFF_NAMES[0]).stat().st_size,
            "purpose": "Trained stability regression model.",
            "sha256": sha256(handoff_dir / HANDOFF_NAMES[0]),
            "source_frozen_sha256": frozen_files["stability_tier1.joblib"],
        },
        "function_tier1.joblib": {
            "bytes": (handoff_dir / HANDOFF_NAMES[1]).stat().st_size,
            "purpose": "Trained function prediction inference bundle (estimator plus selected isotonic calibration).",
            "sha256": sha256(handoff_dir / HANDOFF_NAMES[1]),
            "source_frozen_sha256": frozen_files["function_tier1.joblib"],
        },
        "function_logistic_tier1.joblib": {
            "bytes": platt_path.stat().st_size,
            "calibration_method": "Platt scaling / logistic regression",
            "n_oof_rows_fit": int(len(oof)),
            "oof_table_sha256": oof_hash,
            "purpose": "Standalone alternative function probability calibrator fit from saved grouped OOF scores; not the selected production isotonic calibrator.",
            "sha256": sha256(platt_path),
        },
    }
    handoff_manifest = {
        "disclaimer": DISCLAIMER,
        "esm_model": frozen["esm_model"],
        "esm_model_short": MODEL_SHORT,
        "files": handoff_files,
        "format_version": 1,
        "notes": [
            "The exact three requested filenames are present at the archive root.",
            "function_logistic_tier1.joblib is a Platt probability calibrator fit on OOF scores, not the frozen logistic feature-model baseline.",
            "The selected production calibrator is isotonic and is embedded in function_tier1.joblib; the logistic calibrator is an alternative for teammate workflows.",
            "Do not copy this handoff set over models/esm2_t12_35M: its function_logistic_tier1.joblib intentionally has a different role from the frozen file of the same name.",
            "The two primary model files retain their frozen SHA-256 values. No test-set rows were used to fit the Platt calibrator.",
        ],
        "source_frozen_inventory_sha256": sha256(model_dir / "FROZEN.json"),
        "thresholds": frozen.get("thresholds", {}),
    }
    readme = (
        "# Teammate model files\n\n"
        f"{DISCLAIMER}\n\n"
        "This handoff contains the requested `stability_tier1.joblib`, `function_tier1.joblib`, "
        "and `function_logistic_tier1.joblib`. The third file is a Platt/logistic probability "
        "calibrator fit from saved grouped OOF predictions. It is an alternative: production "
        "inference selects isotonic calibration, already embedded in `function_tier1.joblib`. "
        "The canonical frozen `function_logistic_tier1.joblib` in the model directory is a separate "
        "logistic-regression feature baseline. Do not overwrite it or merge this handoff directory "
        "into the canonical model directory. See `TEAM_HANDOFF_MANIFEST.json` for hashes and roles.\n"
    )
    handoff_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(handoff_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in HANDOFF_NAMES:
            archive.write(handoff_dir / name, name)
        archive.writestr("TEAM_HANDOFF_MANIFEST.json", _json_bytes(handoff_manifest))
        archive.writestr("README.md", readme)
    return {
        "archive": handoff_archive.name,
        "archive_bytes": handoff_archive.stat().st_size,
        "archive_sha256": sha256(handoff_archive),
        "files": handoff_files,
        "manifest": handoff_manifest,
        "oof_table": oof_path.name,
    }


def build_bundle(
    model_dir: Path,
    output_path: Path,
    manifest_path: Path,
    oof_path: Path | None = None,
) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    frozen, frozen_files, entries = _verified_frozen_files(model_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator_path, selected_method = _extract_selected_calibrator(model_dir, output_path.parent, frozen_files)
    entries[calibrator_path.name] = {
        "bytes": calibrator_path.stat().st_size,
        "derived_from": "function_tier1.joblib",
        "derived_from_sha256": frozen_files["function_tier1.joblib"],
        "frozen": False,
        "calibration_method": selected_method,
        "purpose": "Standalone copy of the selected production calibrator for handoff; selected calibration is isotonic.",
        "sha256": sha256(calibrator_path),
    }

    bundle_manifest = {
        "disclaimer": DISCLAIMER,
        "esm_model": frozen["esm_model"],
        "esm_model_short": MODEL_SHORT,
        "files": entries,
        "format_version": 1,
        "notes": [
            "Verify each SHA-256 before loading any serialized estimator.",
            "The frozen function_tier1.joblib contains the production estimator and selected isotonic calibrator.",
            "The frozen function_logistic_tier1.joblib is a comparison baseline, not the calibrator.",
            "function_calibrator_tier1.joblib is a standalone copy of the selected isotonic calibrator.",
            "The package does not contain training data or ESM weights.",
        ],
        "thresholds": frozen.get("thresholds", {}),
    }
    readme = (
        "# Frozen Xynapse AI model bundle\n\n"
        f"{DISCLAIMER}\n\n"
        f"ESM model: `{frozen['esm_model']}` (`{MODEL_SHORT}`). The production function estimator and "
        "selected isotonic calibrator are bundled in `function_tier1.joblib`; the standalone selected "
        "calibrator is `function_calibrator_tier1.joblib`. In this frozen inventory, "
        "`function_logistic_tier1.joblib` is an auxiliary logistic-regression feature baseline. "
        "Verify against `MODEL_BUNDLE_MANIFEST.json` and `FROZEN.json` before loading.\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in frozen_files:
            archive.write(model_dir / name, f"{MODEL_SHORT}/{name}")
        archive.write(model_dir / "FROZEN.json", f"{MODEL_SHORT}/FROZEN.json")
        archive.write(calibrator_path, f"{MODEL_SHORT}/{calibrator_path.name}")
        archive.writestr(f"{MODEL_SHORT}/MODEL_BUNDLE_MANIFEST.json", _json_bytes(bundle_manifest))
        archive.writestr(f"{MODEL_SHORT}/MODEL_BUNDLE_README.md", readme)

    handoff = None
    if oof_path is not None:
        handoff = _build_team_handoff(
            model_dir=model_dir,
            handoff_dir=output_path.parent / "team_handoff",
            handoff_archive=output_path.parent / "xynapse_ai_team_model_files_esm2_t12_35M.zip",
            oof_path=oof_path,
            frozen=frozen,
            frozen_files=frozen_files,
        )
    result = {
        "archive": output_path.name,
        "archive_bytes": output_path.stat().st_size,
        "archive_sha256": sha256(output_path),
        "disclaimer": DISCLAIMER,
        "esm_model": frozen["esm_model"],
        "esm_model_short": MODEL_SHORT,
        "files": entries,
        "frozen_inventory_sha256": sha256(model_dir / "FROZEN.json"),
        "manifest_version": 2,
        "notes": bundle_manifest["notes"],
        "selected_calibration_method": selected_method,
        "team_handoff": handoff,
        "thresholds": frozen.get("thresholds", {}),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_json_bytes(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / MODEL_SHORT)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "models" / f"{MODEL_SHORT}_handoff" / "xynapse_ai_esm2_t12_35M_model_bundle.zip",
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/model_artifact_bundle.json")
    parser.add_argument("--oof", type=Path, default=ROOT / "reports/tables/oof_function.parquet")
    args = parser.parse_args(argv)
    result = build_bundle(args.model_dir, args.out, args.manifest, args.oof)
    print(f"PASS: all {len(FROZEN_PURPOSES)} frozen model/metadata hashes match FROZEN.json")
    print(f"Frozen model bundle: {args.out.expanduser().resolve()} ({result['archive_bytes']} bytes)")
    if result["team_handoff"]:
        handoff = result["team_handoff"]
        print("PASS: teammate handoff contains the exact three requested filenames")
        print(f"Team handoff: {args.out.expanduser().resolve().parent / handoff['archive']} ({handoff['archive_bytes']} bytes)")
        print(f"Team handoff SHA-256: {handoff['archive_sha256']}")
        print("The handoff function_logistic file is a Platt calibrator fit from saved OOF rows; production keeps the selected isotonic calibrator.")
    print(f"Manifest: {args.manifest.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
