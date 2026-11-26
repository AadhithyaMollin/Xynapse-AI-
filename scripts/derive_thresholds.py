#!/usr/bin/env python3
"""Derive decision thresholds from grouped OOF predictions and freeze all models."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xynapse.freeze import freeze_models  # noqa: E402
from xynapse.thresholds import derive_oof_thresholds  # noqa: E402


def update_threshold_config(config_path: Path, values: dict) -> None:
    original = config_path.read_text()
    updated = original
    for key, value in values.items():
        pattern = re.compile(rf"(?m)^(\s+{re.escape(key)}:\s*)(null|[-+0-9.eE]+)(\s*(?:#.*)?)$")
        match = pattern.search(updated)
        if not match:
            raise ValueError(f"existing thresholds.{key} entry not found in config")
        current = match.group(2)
        if current != "null":
            old_value = yaml.safe_load(current)
            if value is None and old_value is None:
                continue
            if value is not None and old_value is not None and np.isclose(float(old_value), float(value), rtol=0, atol=1e-10):
                continue
            raise ValueError(f"config thresholds.{key} already has a different non-null value: {current}")
        replacement = "null" if value is None else format(float(value), ".12g")
        updated = updated[:match.start(2)] + replacement + updated[match.end(2):]
    config_path.write_text(updated)


def run() -> dict:
    run_started = time.perf_counter()
    config_path = ROOT / "configs/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    ddg = pd.read_parquet(ROOT / "reports/tables/oof_ddg.parquet")
    function = pd.read_parquet(ROOT / "reports/tables/oof_function.parquet")
    if not {"y_true", "pred_tier1_full"} <= set(ddg.columns):
        raise RuntimeError("stability grouped OOF file lacks the target or tier1_full prediction")
    if not {"y_true", "p_impact_calibrated"} <= set(function.columns):
        raise RuntimeError("function grouped OOF file lacks the target or calibrated probability")
    thresholds, evidence = derive_oof_thresholds(
        ddg.y_true.to_numpy(float), ddg.pred_tier1_full.to_numpy(float),
        function.y_true.to_numpy(int), function.p_impact_calibrated.to_numpy(float),
        ddg_label_cutoff=float(config["ranking"]["ddg_label_cutoff"]),
        min_stabilizing_positives=50, target_npv=0.90,
    )
    evidence_path = ROOT / "reports/tables/threshold_evidence.csv"
    evidence_frame = pd.DataFrame(evidence)
    evidence_frame.to_csv(evidence_path, index=False)
    update_threshold_config(config_path, thresholds)
    updated_config = yaml.safe_load(config_path.read_text())
    if updated_config["thresholds"] != thresholds:
        for key, value in thresholds.items():
            actual = updated_config["thresholds"][key]
            if value is None and actual is None:
                continue
            if value is None or actual is None or not np.isclose(float(value), float(actual), rtol=0, atol=1e-10):
                raise AssertionError(f"saved config threshold differs for {key}: {actual} vs {value}")
    stability_meta = json.loads((ROOT / "models/esm2_t12_35M/stability_tier1.json").read_text())
    function_meta = json.loads((ROOT / "models/esm2_t12_35M/function_tier1.json").read_text())
    feature_lists = {"stability": stability_meta["feature_list"], "function": function_meta["feature_list"]}
    saved_thresholds = updated_config["thresholds"]
    frozen = freeze_models(ROOT / "models/esm2_t12_35M", saved_thresholds, feature_lists,
                           esm_model="facebook/esm2_t12_35M_UR50D")
    summary = {
        "thresholds": saved_thresholds,
        "stability_oof_rows": int(len(ddg)),
        "stability_damaging_positive_rows": int((ddg.y_true >= float(config["ranking"]["ddg_label_cutoff"])).sum()),
        "stability_stabilizing_positive_rows": int((ddg.y_true <= -float(config["ranking"]["ddg_label_cutoff"])).sum()),
        "function_oof_rows": int(len(function)),
        "function_label_prevalence": float(function.y_true.mean()),
        "calibration": function_meta["calibration_method"],
        "model_files_frozen": len(frozen["model_files_sha256"]),
        "runtime_seconds": round(time.perf_counter() - run_started, 6),
        "frozen_at_utc": frozen["frozen_at_utc"],
    }
    (ROOT / "reports/tables/phase_08_freeze_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Predictions are computational hypotheses, not experimental findings.")
    return summary


if __name__ == "__main__":
    run()
