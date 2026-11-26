#!/usr/bin/env python3
"""Create case-study mutation predictions with frozen models and thresholds."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xynapse.features import read_feature_model
from xynapse.prediction import ESM_MODEL, load_prediction_context, predict_mutations
from xynapse.schema import MUT_KEYS
from xynapse.validate import validate_table


def run() -> dict:
    started = time.monotonic()
    processed = ROOT / "data/processed"
    features_path = processed / "features_tier1.parquet"
    annotations_path = processed / "mut_features.parquet"
    proteins_path = processed / "proteins.parquet"
    model_dir = ROOT / "models/esm2_t12_35M"
    config_path = ROOT / "configs/config.yaml"

    t0 = time.monotonic()
    tier1 = pd.read_parquet(features_path)
    annotations = pd.read_parquet(annotations_path)
    proteins = pd.read_parquet(proteins_path)
    feature_model, short_name = read_feature_model(features_path)
    if feature_model != ESM_MODEL or short_name != "esm2_t12_35M":
        raise RuntimeError("Tier-1 feature table has an unexpected ESM model identifier")
    case_study_ids = set(annotations["protein_id"].astype(str).unique())
    tier1 = tier1.loc[tier1.protein_id.astype(str).isin(case_study_ids)].copy()
    feature_keys = set(tier1[MUT_KEYS].itertuples(index=False, name=None))
    annotation_keys = set(annotations[MUT_KEYS].itertuples(index=False, name=None))
    if feature_keys != annotation_keys:
        raise RuntimeError("Tier-1 features and teammate annotations do not have identical mutation keys")
    frame = tier1.merge(annotations, on=MUT_KEYS, how="inner", validate="one_to_one", suffixes=("", "_annotation"))
    protein_lengths = proteins.set_index("protein_id")["length"].to_dict()
    frame["protein_length"] = frame["protein_id"].map(protein_lengths)
    if frame["protein_length"].isna().any():
        raise RuntimeError("a case-study protein is missing from the protein contract table")
    input_seconds = time.monotonic() - t0

    t0 = time.monotonic()
    context = load_prediction_context(model_dir, config_path, features_path)
    predictions = predict_mutations(frame, context)
    predict_seconds = time.monotonic() - t0
    problems = validate_table(predictions, "predictions", proteins=proteins)
    if problems:
        raise ValueError("prediction table contract failed: " + "; ".join(problems))
    out_path = processed / "predictions.parquet"
    predictions.to_parquet(out_path, index=False)
    counts = (predictions.groupby(["protein_id", "rank_class"], as_index=False)
              .size().rename(columns={"size": "n_mutations"}))
    counts_path = ROOT / "reports/tables/rank_class_counts.csv"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(counts_path, index=False)

    summary = {
        "n_rows": int(len(predictions)),
        "n_proteins": int(predictions.protein_id.nunique()),
        "rows_per_protein": {str(key): int(value) for key, value in predictions.groupby("protein_id").size().items()},
        "rank_class_counts": {str(key): int(value) for key, value in predictions.rank_class.value_counts().items()},
        "tier": sorted(predictions.tier.unique().tolist()),
        "esm_model": sorted(predictions.esm_model.unique().tolist()),
        "input_merge_seconds": input_seconds,
        "model_inference_and_ranking_seconds": predict_seconds,
        "total_seconds": time.monotonic() - started,
        "output": "data/processed/predictions.parquet",
        "rank_counts": "reports/tables/rank_class_counts.csv",
        "disclaimer": "Predictions are computational hypotheses, not experimental findings.",
    }
    summary_path = ROOT / "reports/tables/phase_09_prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run()
