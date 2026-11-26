#!/usr/bin/env python3
"""Regenerate derived report tables and model cards from saved outputs only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xynapse.schema import MUT_KEYS as MUTATION_KEYS

DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
MODEL_DIR = ROOT / "models/esm2_t12_35M"
REPORT_DIR = ROOT / "reports/tables"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def regenerate_label_prevalence(processed: Path) -> pd.DataFrame:
    records = []
    for filename, table_name, positive_definition in (
        ("labels_ddg.parquet", "labels_ddg", "ddg >= 1.0 kcal/mol"),
        ("labels_function.parquet", "labels_function", "label == 1"),
        ("external_clinvar.parquet", "external_clinvar", "label == 1"),
    ):
        frame = pd.read_parquet(processed / filename)
        positive = frame.ddg.ge(1.0) if "ddg" in frame else frame.label.eq(1)
        records.append({
            "dataset": table_name,
            "n_rows": int(len(frame)),
            "n_proteins": int(frame.protein_id.nunique()),
            "positive_definition": positive_definition,
            "n_positive": int(positive.sum()),
            "prevalence": float(positive.mean()) if len(frame) else float("nan"),
        })
    return pd.DataFrame(records, columns=["dataset", "n_rows", "n_proteins", "positive_definition", "n_positive", "prevalence"])


def regenerate_rank_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    return (predictions.groupby(["protein_id", "rank_class"], as_index=False)
            .size().rename(columns={"size": "n_mutations"})
            .sort_values(["protein_id", "rank_class"]).reset_index(drop=True))


def regenerate_split_summary(processed: Path) -> pd.DataFrame:
    records = []
    for task, filename in (("ddg", "train_ddg.parquet"), ("function", "train_function.parquet")):
        frame = pd.read_parquet(processed / filename)
        for split_id in ("test", "train"):
            group = frame.loc[frame.split_id.eq(split_id)]
            if not group.empty:
                records.append({"task": task, "split_id": split_id, "cv_fold": "all",
                                "n_clusters": int(group.cluster_id.nunique()),
                                "n_proteins": int(group.protein_id.nunique()), "n_rows": int(len(group))})
        training = frame.loc[frame.split_id.eq("train")]
        for fold, group in training.groupby("cv_fold", sort=True):
            records.append({"task": task, "split_id": "train_cv_validation", "cv_fold": int(fold),
                            "n_clusters": int(group.cluster_id.nunique()),
                            "n_proteins": int(group.protein_id.nunique()), "n_rows": int(len(group))})
    return pd.DataFrame(records, columns=["task", "split_id", "cv_fold", "n_clusters", "n_proteins", "n_rows"])


def _metric_records(task: str, frame: pd.DataFrame, labels: np.ndarray,
                    predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    if task == "ddg":
        from xynapse.metrics import regression_metrics
        records = []
        for model, values in predictions.items():
            scores = regression_metrics(labels, values)
            records.append({"model": model, **scores, "n_rows": scores["n"],
                            "n_proteins": int(frame.protein_id.nunique()), "tier": "tier1"})
        return pd.DataFrame(records)
    from xynapse.classification_metrics import best_mcc_threshold, binary_metrics
    records = []
    for model, values in predictions.items():
        threshold, _ = best_mcc_threshold(labels, values)
        scores = binary_metrics(labels, values, threshold=threshold)
        records.append({"model": model, **scores, "n_proteins": int(frame.protein_id.nunique()), "tier": "tier1"})
    return pd.DataFrame(records)


def regenerate_model_cards(model_dir: Path, registry: dict[str, Any], card_dir: Path) -> list[Path]:
    stability = json.loads((model_dir / "stability_tier1.json").read_text())
    function = json.loads((model_dir / "function_tier1.json").read_text())
    evaluations = registry.get("evaluations", {})
    stability_test = evaluations.get("stability_cluster_test", {}).get("result", {}).get("metrics", [])
    function_test = evaluations.get("function_cluster_test", {}).get("result", {}).get("metrics", [])
    stability_row = next((row for row in stability_test if row.get("model") == "tier1_full"), {})
    function_row = next((row for row in function_test if row.get("model") == "tier1_full_calibrated"), {})
    card_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for task, meta, oof, test in (
        ("stability", stability, stability.get("oof_metrics", {}), stability_row),
        ("function", function, function.get("oof_metrics_calibrated", {}), function_row),
    ):
        feature_lines = "\n".join(f"- `{feature}`" for feature in meta["feature_list"])
        metrics = "\n".join(f"- `{key}`: {value}" for key, value in sorted(oof.items()))
        test_metrics = "\n".join(f"- `{key}`: {value}" for key, value in sorted(test.items())) or "- Not recorded."
        content = (
            f"# {task.title()} model card\n\n{DISCLAIMER}\n\n"
            f"## Model\n\n- Task: `{meta['task']}`\n- Tier: `{meta['tier']}`\n"
            f"- ESM model: `{meta['esm_model']}` (`{meta['esm_model_short']}`)\n"
            f"- Training proteins: {meta['n_proteins']}\n- Training rows: {meta['n_rows']}\n"
            f"- Maximum training protein length: {meta['max_training_protein_length']} aa\n"
            f"- Training-label SHA-256: `{meta.get('training_labels_sha256', 'unavailable')}`\n"
            f"- Feature-table SHA-256: `{meta.get('features_sha256', 'unavailable')}`\n\n"
            f"## Features\n\n{feature_lines}\n\n## Grouped out-of-fold metrics\n\n{metrics}\n\n"
            f"## Held-out cluster metrics\n\n{test_metrics}\n\n"
            "## Known limits\n\n"
            "Tier-2 model unavailable. Results are computational hypotheses; training labels and assays have heterogeneous sources."
            " The stability model was trained on short domains; predictions beyond the recorded maximum length are extrapolations.\n\n"
            f"See [`docs/limitations.md`](../limitations.md).\n\n{DISCLAIMER}\n"
        )
        path = card_dir / f"{task}_tier1.md"
        path.write_text(content, encoding="utf-8")
        generated.append(path)
    return generated


def run() -> list[Path]:
    processed = ROOT / "data/processed"
    predictions_path = processed / "predictions.parquet"
    predictions = pd.read_parquet(predictions_path)
    registry_path = ROOT / "reports/test_eval_registry.json"
    registry = json.loads(registry_path.read_text())
    generated: list[Path] = []

    paths = {
        "label_prevalence": REPORT_DIR / "label_prevalence.csv",
        "rank_class_counts": REPORT_DIR / "rank_class_counts.csv",
        "split_summary": REPORT_DIR / "split_summary.csv",
    }
    frames = {
        "label_prevalence": regenerate_label_prevalence(processed),
        "rank_class_counts": regenerate_rank_counts(predictions),
        "split_summary": regenerate_split_summary(processed),
    }
    for key, frame in frames.items():
        _write_csv(frame, paths[key])
        generated.append(paths[key])

    from xynapse.classification_metrics import (best_mcc_threshold, binary_metrics,
                                                calibration_curve_data, per_assay_auroc)
    from xynapse.metrics import per_protein_spearman
    ddg_oof = pd.read_parquet(REPORT_DIR / "oof_ddg.parquet")
    ddg_train = pd.read_parquet(processed / "train_ddg.parquet")
    ddg_frame = ddg_oof.merge(
        ddg_train[MUTATION_KEYS + ["blosum62", "d_hydropathy", "d_volume", "esm_llr"]],
        on=MUTATION_KEYS, how="left", validate="one_to_one",
    )
    ddg_scores = {
        "negative_blosum62": -ddg_frame.blosum62.to_numpy(float),
        "abs_d_hydropathy": np.abs(ddg_frame.d_hydropathy.to_numpy(float)),
        "abs_d_volume": np.abs(ddg_frame.d_volume.to_numpy(float)),
        "negative_esm_llr": -ddg_frame.esm_llr.to_numpy(float),
        "esm_only": ddg_oof.pred_esm_only.to_numpy(float),
        "esm_plus_physchem": ddg_oof.pred_esm_plus_physchem.to_numpy(float),
        "tier1_full": ddg_oof.pred_tier1_full.to_numpy(float),
    }
    ddg_metrics = _metric_records("ddg", ddg_frame, ddg_oof.y_true.to_numpy(float), ddg_scores)
    out_path = REPORT_DIR / "stability_oof_metrics.csv"
    _write_csv(ddg_metrics, out_path)
    generated.append(out_path)
    per_protein = []
    for model, scores in ddg_scores.items():
        group_frame = pd.DataFrame({"protein_id": ddg_frame.protein_id, "y_true": ddg_oof.y_true, "y_pred": scores})
        per_protein_metrics, n_qualifying = per_protein_spearman(group_frame, min_variants=30)
        per_protein_metrics["model"] = model
        per_protein_metrics["n_qualifying_proteins"] = n_qualifying
        per_protein.append(per_protein_metrics)
    out_path = REPORT_DIR / "stability_per_protein_oof.csv"
    _write_csv(pd.concat(per_protein, ignore_index=True), out_path)
    generated.append(out_path)

    function_oof = pd.read_parquet(REPORT_DIR / "oof_function.parquet")
    function_train = pd.read_parquet(processed / "train_function.parquet")
    function_frame = function_oof.merge(
        function_train[MUTATION_KEYS + ["blosum62", "esm_llr"]],
        on=MUTATION_KEYS, how="left", validate="one_to_one",
    )
    function_scores = {
        "negative_esm_llr": -function_frame.esm_llr.to_numpy(float),
        "negative_blosum62": -function_frame.blosum62.to_numpy(float),
        "esm_only": function_oof.p_esm_only.to_numpy(float),
        "esm_plus_physchem": function_oof.p_esm_plus_physchem.to_numpy(float),
        "tier1_full": function_oof.p_tier1_full.to_numpy(float),
        "logistic_tier1": function_oof.p_logistic_tier1.to_numpy(float),
        "tier1_full_calibrated": function_oof.p_impact_calibrated.to_numpy(float),
    }
    function_metrics = _metric_records("function", function_frame, function_oof.y_true.to_numpy(int), function_scores)
    out_path = REPORT_DIR / "function_oof_metrics.csv"
    _write_csv(function_metrics, out_path)
    generated.append(out_path)
    assay_frames = []
    calibration_frames = []
    for model, scores in function_scores.items():
        assay_input = function_oof[["assay_id", "y_true"]].copy().rename(columns={"y_true": "label"})
        assay_input["score"] = scores
        assay_result = per_assay_auroc(assay_input, "score")
        assay_result["model"] = model
        assay_frames.append(assay_result)
        if model in {"tier1_full", "tier1_full_calibrated"}:
            calibration = calibration_curve_data(function_oof.y_true.to_numpy(int), scores, n_bins=10)
            calibration["model"] = model
            calibration_frames.append(calibration)
    out_path = REPORT_DIR / "function_per_assay_auroc.csv"
    _write_csv(pd.concat(assay_frames, ignore_index=True), out_path)
    generated.append(out_path)
    out_path = REPORT_DIR / "function_calibration_curve.csv"
    _write_csv(pd.concat(calibration_frames, ignore_index=True), out_path)
    generated.append(out_path)

    model_cards = regenerate_model_cards(MODEL_DIR, registry, ROOT / "docs/model_cards")
    generated.extend(model_cards)
    manifest = {
        "generated_from_saved_outputs": True,
        "generator": "scripts/make_reports.py",
        "files": {str(path.relative_to(ROOT)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                  for path in sorted(generated)},
        "disclaimer": DISCLAIMER,
    }
    manifest_path = ROOT / "reports/tables/generated_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated.append(manifest_path)
    print(f"Regenerated {len(generated)} deterministic report artifacts from saved tables/models.")
    for path in generated:
        print(f"  {path.relative_to(ROOT)}")
    return generated


if __name__ == "__main__":
    run()
