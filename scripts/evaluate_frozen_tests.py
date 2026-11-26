#!/usr/bin/env python3
"""Evaluate frozen models once on held-out, case-study, and ClinVar sets."""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from xynapse.classification_metrics import binary_metrics, per_assay_auroc
from xynapse.features import read_feature_model
from xynapse.freeze import evaluate_once, verify_frozen
from xynapse.metrics import per_protein_spearman, regression_metrics, spearman
from xynapse.schema import MUT_KEYS
from scripts.train_stability import prepare_x
from scripts.train_function import apply_calibrator

MODEL_DIR = ROOT / "models/esm2_t12_35M"
REPORTS = ROOT / "reports/tables"
REGISTRY_PATH = ROOT / "reports/test_eval_registry.json"


def load_frozen_context() -> dict:
    ok, problems = verify_frozen(MODEL_DIR / "FROZEN.json", MODEL_DIR)
    if not ok:
        raise RuntimeError(f"frozen model hash verification failed: {problems}")
    frozen = json.loads((MODEL_DIR / "FROZEN.json").read_text())
    if frozen.get("esm_model") != "facebook/esm2_t12_35M_UR50D":
        raise RuntimeError("FROZEN.json does not name the configured ESM-2 35M model")
    stability_meta = json.loads((MODEL_DIR / "stability_tier1.json").read_text())
    function_meta = json.loads((MODEL_DIR / "function_tier1.json").read_text())
    stability_model = joblib.load(MODEL_DIR / stability_meta["model_file"])
    function_bundle = joblib.load(MODEL_DIR / function_meta["model_file"])
    function_logistic = joblib.load(MODEL_DIR / function_meta["logistic_model_file"])
    if not isinstance(function_bundle, dict) or function_bundle.get("calibration_method") != function_meta["calibration_method"]:
        raise RuntimeError("function model/calibrator bundle disagrees with metadata")
    feature_model, short = read_feature_model(ROOT / "data/processed/features_tier1.parquet")
    if feature_model != frozen["esm_model"] or short != frozen["esm_model_short"]:
        raise RuntimeError("feature table ESM metadata disagrees with frozen model")
    return {
        "frozen": frozen,
        "stability_meta": stability_meta,
        "stability_model": stability_model,
        "function_meta": function_meta,
        "function_bundle": function_bundle,
        "function_logistic": function_logistic,
        "features": pd.read_parquet(ROOT / "data/processed/features_tier1.parquet"),
        "function_oof_thresholds": pd.read_csv(REPORTS / "function_mcc_thresholds.csv").set_index("model").threshold_mcc_max.to_dict(),
        "case_study_assays": pd.read_csv(ROOT / "reports/tables/case_study_assays.csv"),
        "case_study_sequences": pd.read_parquet(ROOT / "data/interim/train_proteins.parquet").set_index("protein_id").sequence.to_dict(),
        "direction": json.loads((REPORTS.parent / "tables/proteingym_direction_precheck_summary.json").read_text()),
    }


def stability_prediction(context: dict, frame: pd.DataFrame) -> np.ndarray:
    meta = context["stability_meta"]
    x = prepare_x(frame, meta["feature_list"], meta["category_lists"])
    pred = np.asarray(context["stability_model"].predict(x), dtype=float)
    if not np.isfinite(pred).all():
        raise RuntimeError("stability model produced non-finite predictions")
    return pred


def function_predictions(context: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    meta = context["function_meta"]
    bundle = context["function_bundle"]
    features = meta["feature_list"]
    x = prepare_x(frame, features, meta["category_lists"])
    raw = np.asarray(bundle["estimator"].predict_proba(x)[:, 1], dtype=float)
    calibrated = apply_calibrator(bundle["calibrator"], bundle["calibration_method"], raw)
    logistic = np.asarray(context["function_logistic"].predict_proba(frame[features])[:, 1], dtype=float)
    if not (np.isfinite(raw).all() and np.isfinite(calibrated).all() and np.isfinite(logistic).all()):
        raise RuntimeError("function model produced non-finite probabilities")
    return calibrated, logistic


def stability_test(context: dict) -> dict:
    labels = pd.read_parquet(ROOT / "data/processed/train_ddg.parquet")
    test = labels.loc[labels.split_id.eq("test")].copy()
    if test.empty:
        raise RuntimeError("stability held-out cluster test is empty")
    if set(test.cluster_id.astype(str)) & set(labels.loc[labels.split_id.eq("train"), "cluster_id"].astype(str)):
        raise AssertionError("stability test clusters overlap the training split")
    predictions = {
        "negative_blosum62": -test.blosum62.to_numpy(float),
        "abs_d_hydropathy": np.abs(test.d_hydropathy.to_numpy(float)),
        "abs_d_volume": np.abs(test.d_volume.to_numpy(float)),
        "negative_esm_llr": -test.esm_llr.to_numpy(float),
        "tier1_full": stability_prediction(context, test),
    }
    records = []
    per_protein = []
    for model_name, values in predictions.items():
        metrics = regression_metrics(test.ddg, values)
        records.append({"model": model_name, **metrics, "n_rows": len(test),
                        "n_proteins": test.protein_id.nunique(), "tier": "tier1"})
        pp, _ = per_protein_spearman(pd.DataFrame({
            "protein_id": test.protein_id, "y_true": test.ddg, "y_pred": values,
        }), min_variants=30)
        pp["model"] = model_name
        per_protein.append(pp)
    pd.DataFrame(records).to_csv(REPORTS / "test_stability_cluster.csv", index=False)
    pd.concat(per_protein, ignore_index=True).to_csv(REPORTS / "test_stability_per_protein.csv", index=False)
    return {"n_rows": len(test), "n_proteins": int(test.protein_id.nunique()),
            "n_clusters": int(test.cluster_id.nunique()), "metrics": records}


def function_test(context: dict) -> dict:
    labels = pd.read_parquet(ROOT / "data/processed/train_function.parquet")
    test = labels.loc[labels.split_id.eq("test")].copy()
    if test.empty:
        raise RuntimeError("function held-out cluster test is empty")
    if set(test.cluster_id.astype(str)) & set(labels.loc[labels.split_id.eq("train"), "cluster_id"].astype(str)):
        raise AssertionError("function test clusters overlap the training split")
    calibrated, logistic = function_predictions(context, test)
    scores = {
        "negative_esm_llr": -test.esm_llr.to_numpy(float),
        "negative_blosum62": -test.blosum62.to_numpy(float),
        "logistic_tier1": logistic,
        "tier1_full_calibrated": calibrated,
    }
    thresholds = context["function_oof_thresholds"]
    records = []
    per_assay = []
    for model_name, values in scores.items():
        metric_name = model_name
        threshold = float(thresholds[metric_name])
        metric = binary_metrics(test.label, values, threshold=threshold)
        records.append({"model": model_name, **metric, "n_proteins": test.protein_id.nunique(), "tier": "tier1"})
        assay_frame = test[["assay_id", "label"]].copy()
        assay_frame["score"] = values
        assay_auc = per_assay_auroc(assay_frame, "score")
        assay_auc["model"] = model_name
        per_assay.append(assay_auc)
    pd.DataFrame(records).to_csv(REPORTS / "test_function_cluster.csv", index=False)
    pd.concat(per_assay, ignore_index=True).to_csv(REPORTS / "test_function_per_assay.csv", index=False)
    return {"n_rows": len(test), "n_proteins": int(test.protein_id.nunique()),
            "n_clusters": int(test.cluster_id.nunique()), "metrics": records}


def build_case_study_frame(context: dict, protein_id: str) -> tuple[pd.DataFrame, list[dict]]:
    assays = context["case_study_assays"].loc[
        context["case_study_assays"].case_study_protein_id.eq(protein_id)].copy()
    if assays.empty:
        raise RuntimeError(f"no saved case-study assays for {protein_id}")
    sequence = str(context["case_study_sequences"][protein_id])
    direction = context["direction"]
    label_1_is_tolerated = bool(direction["label_1_is_tolerated_by_pooled_mean"])
    output = []
    assay_metadata = []
    token = re.compile(r"^([A-Z])(\d+)([A-Z])$")
    archive_path = ROOT / "data/raw/proteingym/DMS_ProteinGym_substitutions.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for assay in assays.itertuples(index=False):
            member = f"DMS_ProteinGym_substitutions/{assay.DMS_filename}"
            if member not in members:
                raise FileNotFoundError(f"case-study assay archive member missing: {member}")
            frame = pd.read_csv(archive.open(member), low_memory=False)
            label_column = str(assay.binary_label_column)
            grouped: dict[tuple[int, str, str], list[tuple[int, float]]] = {}
            rejected = {"non_single": 0, "invalid_label_or_score": 0, "wt_mismatch": 0,
                        "conflicting_duplicate_labels": 0}
            for row in frame.itertuples(index=False):
                match = token.fullmatch(str(getattr(row, "mutant", "")))
                if not match:
                    rejected["non_single"] += 1
                    continue
                wt, position, mut = match.group(1), int(match.group(2)), match.group(3)
                raw_label = pd.to_numeric(pd.Series([getattr(row, label_column, np.nan)]), errors="coerce").iloc[0]
                score = pd.to_numeric(pd.Series([getattr(row, "DMS_score", np.nan)]), errors="coerce").iloc[0]
                if pd.isna(raw_label) or float(raw_label) not in (0.0, 1.0) or pd.isna(score) or not np.isfinite(float(score)):
                    rejected["invalid_label_or_score"] += 1
                    continue
                if position < 1 or position > len(sequence) or sequence[position - 1] != wt:
                    rejected["wt_mismatch"] += 1
                    continue
                project_label = int(1 - int(raw_label) if label_1_is_tolerated else int(raw_label))
                grouped.setdefault((position, wt, mut), []).append((project_label, float(score)))
            for key, observations in grouped.items():
                observed_labels = {label for label, _ in observations}
                if len(observed_labels) != 1:
                    rejected["conflicting_duplicate_labels"] += 1
                    continue
                position, wt, mut = key
                output.append({"protein_id": protein_id, "position": position, "wt": wt, "mut": mut,
                               "label": next(iter(observed_labels)),
                               "DMS_score": float(np.mean([score for _, score in observations])),
                               "assay_id": assay.DMS_id,
                               "phenotype": assay.raw_DMS_phenotype_name,
                               "selection_type": assay.selection_type,
                               "coarse_selection_type": assay.coarse_selection_type})
            assay_metadata.append({"assay_id": assay.DMS_id, "phenotype": assay.raw_DMS_phenotype_name,
                                  "selection_type": assay.selection_type,
                                  "coarse_selection_type": assay.coarse_selection_type,
                                  "n_mutations": len(grouped), "rejected": rejected})
    result = pd.DataFrame(output)
    if result.empty:
        raise RuntimeError(f"no single-mutant case-study labels remained for {protein_id}")
    if result.duplicated(["protein_id", "assay_id", "position", "wt", "mut"]).any():
        raise RuntimeError(f"case-study {protein_id} assay table has duplicate mutation keys")
    return result, assay_metadata


def case_study_test(context: dict, protein_id: str) -> dict:
    frame, assay_metadata = build_case_study_frame(context, protein_id)
    features = context["features"]
    joined = frame.merge(features, on=MUT_KEYS, how="inner", validate="many_to_one")
    if len(joined) != len(frame):
        raise RuntimeError(f"{protein_id}: {len(frame) - len(joined)} DMS rows lacked Tier-1 features")
    calibrated, _ = function_predictions(context, joined)
    joined["p_impact"] = calibrated
    joined["predicted_damaging"] = joined.p_impact >= float(context["frozen"]["thresholds"]["p_impact_high"])
    threshold = float(context["frozen"]["thresholds"]["p_impact_high"])
    per_assay = []
    for assay_id, group in joined.groupby("assay_id", sort=True):
        class_metrics = binary_metrics(group.label, group.p_impact, threshold=threshold)
        per_assay.append({"protein_id": protein_id, "assay_id": assay_id,
                          "phenotype": group.phenotype.iloc[0], "selection_type": group.selection_type.iloc[0],
                          "coarse_selection_type": group.coarse_selection_type.iloc[0],
                          **class_metrics,
                          "spearman_p_impact_DMS_score": spearman(group.DMS_score, group.p_impact)})
    worst = joined.assign(abs_label_probability_error=lambda f: (f.p_impact - f.label).abs())
    worst = worst.sort_values("abs_label_probability_error", ascending=False).groupby("assay_id", sort=True).head(20)
    variant_path = REPORTS / f"test_case_study_{protein_id}_variants.csv"
    joined[[*MUT_KEYS, "assay_id", "phenotype", "selection_type", "coarse_selection_type",
            "DMS_score", "label", "p_impact", "predicted_damaging"]].to_csv(variant_path, index=False)
    worst_path = REPORTS / f"test_case_study_{protein_id}_worst_disagreements.csv"
    worst[[*MUT_KEYS, "assay_id", "phenotype", "label", "p_impact", "DMS_score",
           "abs_label_probability_error"]].to_csv(worst_path, index=False)
    summary_path = REPORTS / f"test_case_study_{protein_id}.csv"
    pd.DataFrame(per_assay).to_csv(summary_path, index=False)
    return {"protein_id": protein_id, "n_assays": len(per_assay), "n_rows": len(joined),
            "assay_metrics": per_assay, "assay_metadata": assay_metadata}


def clinvar_test(context: dict) -> dict:
    labels = pd.read_parquet(ROOT / "data/processed/external_clinvar.parquet")
    joined = labels.merge(context["features"], on=MUT_KEYS, how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise RuntimeError(f"{len(labels) - len(joined)} ClinVar variants lacked Tier-1 features")
    calibrated, _ = function_predictions(context, joined)
    joined["p_impact"] = calibrated
    threshold = float(context["frozen"]["thresholds"]["p_impact_high"])
    overall = binary_metrics(joined.label, joined.p_impact, threshold=threshold)
    per_gene = []
    count_records = []
    for gene, gene_frame in joined.groupby("gene", sort=True):
        for label in (0, 1):
            count_records.append({"gene": gene, "label": label,
                                  "n_variants": int(gene_frame.label.eq(label).sum())})
    counts = pd.DataFrame(count_records)
    for gene, group in joined.groupby("gene", sort=True):
        per_gene.append({"gene": gene, **binary_metrics(group.label, group.p_impact, threshold=threshold)})
    pd.DataFrame([{"scope": "overall", **overall, "n_proteins": joined.protein_id.nunique()}]
                 + [{"scope": "gene", **record, "n_proteins": joined.loc[joined.gene.eq(record["gene"]), "protein_id"].nunique()}
                    for record in per_gene]).to_csv(REPORTS / "test_clinvar_external.csv", index=False)
    counts.to_csv(REPORTS / "test_clinvar_counts_by_gene_label.csv", index=False)
    return {"n_variants": len(joined), "n_proteins": int(joined.protein_id.nunique()),
            "overall_metrics": overall, "per_gene_metrics": per_gene,
            "counts_by_gene_label": counts.to_dict(orient="records")}


def run() -> dict:
    run_started = time.perf_counter()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        existing = json.loads(REGISTRY_PATH.read_text()).get("evaluations", {})
        if existing:
            raise RuntimeError(f"test registry is not empty; refusing an implicit re-evaluation: {sorted(existing)}")
    context = load_frozen_context()
    results = {}
    evaluators = [
        ("stability_cluster_test", lambda: stability_test(context)),
        ("function_cluster_test", lambda: function_test(context)),
        ("case_study_dms_P60484", lambda: case_study_test(context, "P60484")),
        ("case_study_dms_P04637", lambda: case_study_test(context, "P04637")),
        ("clinvar_external_test", lambda: clinvar_test(context)),
    ]
    failures = []
    for name, evaluator in evaluators:
        try:
            results[name] = evaluate_once(name, evaluator, REGISTRY_PATH)
            print(f"PASS: evaluated once: {name}", flush=True)
        except Exception as exc:
            failures.append({"name": name, "error_type": type(exc).__name__, "error": str(exc)})
            print(f"FAIL: {name}: {type(exc).__name__}: {exc}", flush=True)
    summary = {"evaluations": results, "failures": failures,
               "frozen_at_utc": context["frozen"]["frozen_at_utc"],
               "esm_model": context["frozen"]["esm_model"],
               "runtime_seconds": round(time.perf_counter() - run_started, 6),
               "disclaimer": "Predictions are computational hypotheses, not experimental findings."}
    (ROOT / "reports/tables/phase_08_test_evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print("Predictions are computational hypotheses, not experimental findings.")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
