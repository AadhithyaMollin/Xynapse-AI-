"""Apply frozen Tier-1 models and ranking policy to mutation-feature rows."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from .features import read_feature_model
from .freeze import verify_frozen
from .mechanism import apply_mechanism_rules
from .ranking import confidence_fields, evidence_text, rank_mutation
from .schema import MUT_KEYS, PRED_COLS

ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
ESM_SHORT = "esm2_t12_35M"


def _prepare_features(frame: pd.DataFrame, feature_list: list[str], categories: Mapping[str, list[str]]) -> pd.DataFrame:
    missing = [column for column in feature_list if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction input is missing frozen model features: {missing}")
    prepared = frame[feature_list].copy()
    for column in ("wt", "mut"):
        if column in prepared:
            values = prepared[column].astype("string")
            allowed = categories.get(column, [])
            prepared[column] = pd.Categorical(values.where(values.isin(allowed)), categories=allowed)
    return prepared


def load_prediction_context(model_dir: str | Path, config_path: str | Path, feature_path: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    frozen_path = model_dir / "FROZEN.json"
    ok, problems = verify_frozen(frozen_path, model_dir)
    if not ok:
        raise RuntimeError(f"frozen model hash verification failed: {problems}")
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("esm_model") != ESM_MODEL or frozen.get("esm_model_short") != ESM_SHORT:
        raise RuntimeError("frozen model does not match the configured ESM-2 35M model")
    feature_model, feature_short = read_feature_model(feature_path)
    if (feature_model, feature_short) != (ESM_MODEL, ESM_SHORT):
        raise RuntimeError("Tier-1 feature table ESM metadata does not match the frozen model")

    stability_meta = json.loads((model_dir / "stability_tier1.json").read_text())
    function_meta = json.loads((model_dir / "function_tier1.json").read_text())
    if frozen.get("feature_lists", {}).get("stability") != stability_meta.get("feature_list"):
        raise RuntimeError("stability feature list disagrees with frozen inventory")
    if frozen.get("feature_lists", {}).get("function") != function_meta.get("feature_list"):
        raise RuntimeError("function feature list disagrees with frozen inventory")
    stability_model = joblib.load(model_dir / stability_meta["model_file"])
    function_bundle = joblib.load(model_dir / function_meta["model_file"])
    if function_bundle.get("calibration_method") != function_meta.get("calibration_method"):
        raise RuntimeError("function calibrator method disagrees with frozen metadata")
    config = yaml.safe_load(Path(config_path).read_text())
    rules_config = yaml.safe_load((Path(config_path).parent / "mechanism_rules.yaml").read_text())
    if frozen.get("thresholds") != config.get("thresholds"):
        raise RuntimeError("configured thresholds differ from frozen thresholds")
    return {
        "frozen": frozen,
        "stability_meta": stability_meta,
        "stability_model": stability_model,
        "function_meta": function_meta,
        "function_bundle": function_bundle,
        "config": config,
        "rules_config": rules_config,
    }


def _calibrated_probabilities(bundle: Mapping[str, Any], raw: np.ndarray) -> np.ndarray:
    calibrator = bundle["calibrator"]
    method = bundle["calibration_method"]
    if method == "isotonic":
        result = calibrator.predict(raw)
    else:
        result = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)


def predict_mutations(frame: pd.DataFrame, context: Mapping[str, Any]) -> pd.DataFrame:
    """Score a complete mutation frame; preserve available annotation evidence."""
    missing_keys = [column for column in MUT_KEYS if column not in frame]
    if missing_keys:
        raise ValueError(f"prediction input missing mutation keys: {missing_keys}")
    if frame.duplicated(MUT_KEYS).any():
        raise ValueError("prediction input contains duplicate mutation keys")
    stability_meta = context["stability_meta"]
    function_meta = context["function_meta"]
    stability_x = _prepare_features(frame, stability_meta["feature_list"], stability_meta["category_lists"])
    function_x = _prepare_features(frame, function_meta["feature_list"], function_meta["category_lists"])
    ddg = np.asarray(context["stability_model"].predict(stability_x), dtype=float)
    bundle = context["function_bundle"]
    raw_probability = np.asarray(bundle["estimator"].predict_proba(function_x)[:, 1], dtype=float)
    p_impact = _calibrated_probabilities(bundle, raw_probability)
    if not (np.isfinite(ddg).all() and np.isfinite(p_impact).all()):
        raise RuntimeError("frozen model produced non-finite predictions")

    config = context["config"]
    thresholds = config["thresholds"]
    stability_max = int(stability_meta["max_training_protein_length"])
    function_max = int(function_meta["max_training_protein_length"])
    output = frame.copy().reset_index(drop=True)
    output["ddg_pred"] = ddg
    output["p_impact"] = p_impact
    output["tier"] = "tier1"
    output["esm_model"] = ESM_MODEL
    confidence_rows = []
    for row in output.to_dict(orient="records"):
        protein_length = int(row.get("protein_length", row.get("length", 0)))
        ddg_conf, func_conf, reasons = confidence_fields(
            protein_length, stability_max, function_max, "tier1", row, config.get("confidence", {})
        )
        confidence_rows.append((ddg_conf, func_conf, reasons))
    output["ddg_confidence"] = [values[0] for values in confidence_rows]
    output["confidence"] = [values[1] for values in confidence_rows]
    mechanism_config = config.get("mechanism", {})
    output["mechanism_flags"] = [
        apply_mechanism_rules(row, thresholds, mechanism_config, context["rules_config"])
        for row in output.to_dict(orient="records")
    ]
    output["rank_class"] = [
        rank_mutation(row, thresholds, row["ddg_confidence"], models_available=True, features_available=True)
        for row in output.to_dict(orient="records")
    ]
    output["evidence_text"] = [
        evidence_text({**row, "mechanism_flags": flags, "ddg_confidence": ddg_conf,
                       "confidence": func_conf, "tier": "tier1"}, reasons)
        for row, flags, (ddg_conf, func_conf, reasons)
        in zip(output.to_dict(orient="records"), output["mechanism_flags"], confidence_rows, strict=True)
    ]
    columns = MUT_KEYS + PRED_COLS + ["ddg_confidence", "tier", "esm_model"]
    return output[columns].sort_values(MUT_KEYS).reset_index(drop=True)
