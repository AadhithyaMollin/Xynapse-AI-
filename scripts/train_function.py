#!/usr/bin/env python3
"""Train grouped-CV function classifiers on DMS-derived training labels only."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from xynapse.classification_metrics import (best_mcc_threshold, binary_metrics,
                                            calibration_curve_data, per_assay_auroc)
from xynapse.features import read_feature_model
from scripts.train_stability import (ABLATIONS, CATEGORICAL, NUMERIC_TIER1, categorical_categories,
                                     prepare_x, random_configs, sha256)

MODEL_SHORT = "esm2_t12_35M"


def make_classifier(params: dict, seed: int, n_jobs: int) -> LGBMClassifier:
    return LGBMClassifier(objective="binary", random_state=seed, n_jobs=n_jobs, verbosity=-1,
                           deterministic=True, force_col_wise=True, subsample_freq=1, **params)


def make_logistic(categories: dict[str, list[str]], numeric_features: list[str], seed: int) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(categories=[categories[name] for name in CATEGORICAL],
                                  handle_unknown="ignore", sparse_output=True)),
    ])
    transform = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([("features", transform),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"))])


def tune_classifier(train: pd.DataFrame, categories: dict[str, list[str]], seed: int, n_jobs: int,
                    *, cache_path: Path, fingerprint: str, wall_seconds: float = 600.0) -> tuple[dict, pd.DataFrame, float]:
    started = time.monotonic()
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=seed)
    groups = train.cluster_id.astype(str).to_numpy()
    configs = random_configs(seed, 20)
    cache = {"fingerprint": fingerprint, "results": []}
    if cache_path.is_file():
        existing = json.loads(cache_path.read_text())
        if existing.get("fingerprint") == fingerprint:
            cache = existing
    saved = {int(row["config_id"]): row for row in cache.get("results", [])
             if row.get("status") == "complete" and np.isfinite(row.get("mean_grouped_cv_auprc", np.nan))}
    records = []
    for index, params in enumerate(configs, start=1):
        params_json = json.dumps(params, sort_keys=True)
        if index in saved and saved[index].get("params") == params_json:
            records.append({**saved[index], "cache_reused": True})
            continue
        if time.monotonic() - started >= wall_seconds:
            records.append({"config_id": index, "params": params_json, "mean_grouped_cv_auprc": np.nan,
                            "completed_inner_folds": 0, "status": "wall_cap_before_start", "cache_reused": False})
            continue
        fold_scores = []
        status = "complete"
        for inner_fold, (fit_idx, val_idx) in enumerate(splitter.split(train, groups=groups)):
            if time.monotonic() - started >= wall_seconds:
                status = "wall_cap_during_config"
                break
            fit_frame, val_frame = train.iloc[fit_idx], train.iloc[val_idx]
            if set(fit_frame.cluster_id.astype(str)) & set(val_frame.cluster_id.astype(str)):
                raise AssertionError("inner function CV split a cluster")
            model = make_classifier(params, seed + index * 10 + inner_fold, n_jobs)
            model.fit(prepare_x(fit_frame, ABLATIONS["tier1_full"], categories), fit_frame.label.to_numpy(int))
            probability = model.predict_proba(prepare_x(val_frame, ABLATIONS["tier1_full"], categories))[:, 1]
            fold_scores.append(binary_metrics(val_frame.label, probability, threshold=0.5)["auprc"])
        score = float(np.nanmean(fold_scores)) if fold_scores and np.isfinite(fold_scores).any() else float("nan")
        row = {"config_id": index, "params": params_json, "mean_grouped_cv_auprc": score,
               "completed_inner_folds": len(fold_scores), "status": status, "cache_reused": False}
        records.append(row)
        cache["results"] = [item for item in records if item.get("status") == "complete"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix(".tmp")
        temp.write_text(json.dumps(cache, indent=2, allow_nan=True) + "\n")
        temp.replace(cache_path)
        if status == "wall_cap_during_config":
            break
    results = pd.DataFrame(records)
    complete = results.loc[results.status.eq("complete") & results.mean_grouped_cv_auprc.notna()]
    if complete.empty:
        raise RuntimeError("function hyperparameter search returned no finite completed configuration")
    best = json.loads(complete.loc[complete.mean_grouped_cv_auprc.idxmax(), "params"])
    return best, results, time.monotonic() - started


def grouped_oof(train: pd.DataFrame, feature_list: list[str], categories: dict[str, list[str]],
                params: dict, *, seed: int, n_jobs: int, test_clusters: set[str], model_name: str,
                logistic: bool = False) -> tuple[np.ndarray, list[dict]]:
    predictions = np.full(len(train), np.nan, dtype=float)
    audits = []
    for fold in sorted(train.cv_fold.astype(int).unique()):
        valid_mask = train.cv_fold.astype(int).eq(fold).to_numpy()
        fit_mask = ~valid_mask
        fit_frame, valid_frame = train.loc[fit_mask], train.loc[valid_mask]
        fit_clusters = set(fit_frame.cluster_id.astype(str))
        valid_clusters = set(valid_frame.cluster_id.astype(str))
        fit_valid_overlap = fit_clusters & valid_clusters
        fit_test_overlap = fit_clusters & test_clusters
        if fit_valid_overlap or fit_test_overlap:
            raise AssertionError(f"{model_name} fold {fold}: cluster leakage")
        if logistic:
            model = make_logistic(categories, [column for column in feature_list if column not in CATEGORICAL], seed + fold)
            x_fit = fit_frame[feature_list]
            x_valid = valid_frame[feature_list]
        else:
            model = make_classifier(params, seed + fold, n_jobs)
            x_fit = prepare_x(fit_frame, feature_list, categories)
            x_valid = prepare_x(valid_frame, feature_list, categories)
        model.fit(x_fit, fit_frame.label.to_numpy(int))
        predictions[valid_mask] = model.predict_proba(x_valid)[:, 1]
        audits.append({"model": model_name, "cv_fold": int(fold), "n_fit_rows": len(fit_frame),
                       "n_validation_rows": len(valid_frame), "n_fit_clusters": len(fit_clusters),
                       "n_validation_clusters": len(valid_clusters),
                       "fit_validation_cluster_overlap": len(fit_valid_overlap),
                       "fit_test_cluster_overlap": len(fit_test_overlap)})
    return predictions, audits


def fit_calibrator(method: str, score: np.ndarray, labels: np.ndarray):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(score, labels)
    if method == "platt":
        return LogisticRegression(max_iter=1000, random_state=42).fit(score.reshape(-1, 1), labels)
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(calibrator, method: str, score: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        result = calibrator.predict(score)
    else:
        result = calibrator.predict_proba(score.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)


def choose_calibrator(train: pd.DataFrame, scores: np.ndarray) -> tuple[str, object, np.ndarray, pd.DataFrame]:
    labels = train.label.to_numpy(int)
    fold_ids = train.cv_fold.to_numpy(int)
    records = []
    predictions_by_method = {}
    for method in ("isotonic", "platt"):
        crossfit = np.full(len(train), np.nan, dtype=float)
        for fold in sorted(np.unique(fold_ids)):
            valid = fold_ids == fold
            fit = ~valid
            calibrator = fit_calibrator(method, scores[fit], labels[fit])
            crossfit[valid] = apply_calibrator(calibrator, method, scores[valid])
        for fold in sorted(np.unique(fold_ids)):
            valid = fold_ids == fold
            records.append({"method": method, "cv_fold": int(fold),
                            "brier": float(brier_score_loss(labels[valid], crossfit[valid])),
                            "n_rows": int(valid.sum())})
        predictions_by_method[method] = crossfit
    scores_frame = pd.DataFrame(records)
    means = scores_frame.groupby("method").brier.mean().sort_values(kind="stable")
    selected = str(means.index[0])
    final_calibrator = fit_calibrator(selected, scores, labels)
    return selected, final_calibrator, predictions_by_method[selected], scores_frame


def run() -> dict:
    started = time.perf_counter()
    config_path = ROOT / "configs/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    seed = int(config["seed"])
    full = pd.read_parquet(ROOT / "data/processed/train_function.parquet")
    train = full.loc[full.split_id.eq("train")].copy().reset_index(drop=True)
    test_clusters = set(full.loc[full.split_id.eq("test"), "cluster_id"].astype(str))
    if train.empty or train.cv_fold.isna().any() or train.cluster_id.isna().any():
        raise RuntimeError("function training rows require non-null folds and clusters")
    if train.label.nunique() != 2:
        raise RuntimeError("function training rows must contain both binary labels")
    if set(train.cluster_id.astype(str)) & test_clusters:
        raise AssertionError("function train/test cluster overlap")
    if train.duplicated(["protein_id", "position", "wt", "mut"]).any():
        raise RuntimeError("function training labels contain duplicate mutation keys")
    categories = categorical_categories(train)
    n_jobs = max(1, min(4, os.cpu_count() or 1))

    # Oriented baselines are scores, not probabilities; thresholds are selected from OOF only.
    baseline_scores = {
        "negative_esm_llr": -train.esm_llr.to_numpy(float),
        "negative_blosum62": -train.blosum62.to_numpy(float),
    }

    hp_fingerprint = hashlib.sha256((sha256(ROOT / "data/processed/train_function.parquet")
                                     + sha256(ROOT / "data/processed/features_tier1.parquet")
                                     + sha256(config_path)).encode()).hexdigest()
    previous_summary_path = ROOT / "reports/tables/phase_07_function_summary.json"
    previous_summary = json.loads(previous_summary_path.read_text()) if previous_summary_path.is_file() else {}
    best_params, search, search_runtime_this_run = tune_classifier(
        train, categories, seed, n_jobs, cache_path=ROOT / "data/interim/function_hp_search_cache.json",
        fingerprint=hp_fingerprint, wall_seconds=600.0,
    )
    n_cached = int(search.get("cache_reused", pd.Series(dtype=bool)).fillna(False).sum())
    search_runtime = (float(previous_summary["search_runtime_seconds"])
                      if n_cached == 20 and previous_summary.get("search_runtime_seconds")
                      else search_runtime_this_run)
    search.to_csv(ROOT / "reports/tables/function_hyperparameter_search.csv", index=False)
    if int(search.status.eq("complete").sum()) < 20:
        warnings.warn("10-minute function search cap elapsed before all 20 configs completed", RuntimeWarning)

    base_columns = ["protein_id", "position", "wt", "mut", "cluster_id", "cv_fold",
                    "assay_id", "phenotype", "label"]
    base_oof = train[base_columns].rename(columns={"label": "y_true"}).copy()
    oof_path = ROOT / "reports/tables/oof_function.parquet"
    audit_path = ROOT / "reports/tables/function_fit_audit.csv"
    oof_manifest_path = ROOT / "data/interim/function_oof_manifest.json"
    oof_fingerprint = hashlib.sha256((hp_fingerprint + json.dumps(best_params, sort_keys=True)).encode()).hexdigest()
    expected_predictions = {"p_esm_only", "p_esm_plus_physchem", "p_tier1_full",
                            "p_logistic_tier1", "p_impact_calibrated"}
    reuse_oof = False
    if oof_path.is_file() and audit_path.is_file():
        candidate = pd.read_parquet(oof_path)
        manifest = json.loads(oof_manifest_path.read_text()) if oof_manifest_path.is_file() else None
        source = base_oof.sort_values(["protein_id", "position", "mut"]).reset_index(drop=True)
        cached_base = candidate[base_columns[:-1] + ["y_true"]].sort_values(
            ["protein_id", "position", "mut"]).reset_index(drop=True) if set(base_columns[:-1] + ["y_true"]) <= set(candidate.columns) else pd.DataFrame()
        same_rows = len(candidate) == len(train) and not candidate.duplicated(["protein_id", "position", "wt", "mut"]).any()
        same_base = (not cached_base.empty and len(cached_base) == len(source)
                     and cached_base[base_columns[:-1] + ["y_true"]].equals(source[base_columns[:-1] + ["y_true"]]))
        audit_cache = pd.read_csv(audit_path)
        audit_valid = (len(audit_cache) == 20 and audit_cache.fit_validation_cluster_overlap.eq(0).all()
                       and audit_cache.fit_test_cluster_overlap.eq(0).all())
        manifest_valid = manifest is None or manifest.get("fingerprint") == oof_fingerprint
        reuse_oof = (same_rows and same_base and expected_predictions <= set(candidate.columns)
                     and candidate[list(expected_predictions)].notna().all().all()
                     and audit_valid and manifest_valid)
        if reuse_oof:
            oof = candidate
            audits = audit_cache.to_dict(orient="records")
            print("Resuming from complete function OOF cache; reusing 20 fold fits.", flush=True)
    if not reuse_oof:
        oof = base_oof.copy()
        audits = []
        for ablation, feature_list in ABLATIONS.items():
            pred, fit_audits = grouped_oof(train, feature_list, categories, best_params, seed=seed,
                                           n_jobs=n_jobs, test_clusters=test_clusters, model_name=ablation)
            if not np.isfinite(pred).all():
                raise AssertionError(f"{ablation}: OOF probabilities are incomplete")
            oof[f"p_{ablation}"] = pred
            audits.extend(fit_audits)
        logistic_features = ABLATIONS["tier1_full"]
        logistic_oof, fit_audits = grouped_oof(train, logistic_features, categories, best_params, seed=seed,
                                               n_jobs=n_jobs, test_clusters=test_clusters,
                                               model_name="logistic_tier1", logistic=True)
        if not np.isfinite(logistic_oof).all():
            raise AssertionError("logistic OOF probabilities are incomplete")
        oof["p_logistic_tier1"] = logistic_oof
        audits.extend(fit_audits)
        pd.DataFrame(audits).to_csv(audit_path, index=False)

    calibration_method, calibrator, calibrated_oof, calibration_cv = choose_calibrator(
        train, oof.p_tier1_full.to_numpy(float))
    oof["p_impact_calibrated"] = calibrated_oof
    calibration_cv.to_csv(ROOT / "reports/tables/function_calibration_cv.csv", index=False)
    calibration_cv.groupby("method", as_index=False).agg(mean_grouped_cv_brier=("brier", "mean"),
                                                           n_rows=("n_rows", "sum")).to_csv(
        ROOT / "reports/tables/function_calibration_method_scores.csv", index=False)
    oof.to_parquet(oof_path, index=False)
    oof_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    oof_manifest_path.write_text(json.dumps({"fingerprint": oof_fingerprint,
                                             "oof_sha256": sha256(oof_path),
                                             "reused_existing_oof": reuse_oof,
                                             "n_rows": len(oof)}, indent=2) + "\n")
    print("Function OOF and grouped calibration saved.", flush=True)

    score_columns = {
        "negative_esm_llr": baseline_scores["negative_esm_llr"],
        "negative_blosum62": baseline_scores["negative_blosum62"],
        "esm_only": oof.p_esm_only.to_numpy(float),
        "esm_plus_physchem": oof.p_esm_plus_physchem.to_numpy(float),
        "tier1_full": oof.p_tier1_full.to_numpy(float),
        "logistic_tier1": oof.p_logistic_tier1.to_numpy(float),
        "tier1_full_calibrated": oof.p_impact_calibrated.to_numpy(float),
    }
    metric_records = []
    threshold_records = []
    assay_tables = []
    calibration_tables = []
    for model_name, scores in score_columns.items():
        threshold, best_mcc = best_mcc_threshold(oof.y_true, scores)
        values = binary_metrics(oof.y_true, scores, threshold=threshold)
        metric_records.append({"model": model_name, **values,
                               "n_proteins": int(train.protein_id.nunique()), "tier": "tier1"})
        threshold_records.append({"model": model_name, "threshold_mcc_max": threshold,
                                  "mcc_max": best_mcc, "threshold_source": "grouped OOF only"})
        assay_frame = oof[["assay_id", "y_true"]].copy()
        assay_frame["label"] = assay_frame.pop("y_true")
        assay_frame["score"] = scores
        assay_auc = per_assay_auroc(assay_frame, "score")
        assay_auc["model"] = model_name
        assay_tables.append(assay_auc)
        if model_name in {"tier1_full", "tier1_full_calibrated"}:
            curve = calibration_curve_data(oof.y_true, scores, n_bins=10)
            curve["model"] = model_name
            calibration_tables.append(curve)
    pd.DataFrame(metric_records).to_csv(ROOT / "reports/tables/function_oof_metrics.csv", index=False)
    pd.DataFrame(threshold_records).to_csv(ROOT / "reports/tables/function_mcc_thresholds.csv", index=False)
    pd.concat(assay_tables, ignore_index=True).to_csv(ROOT / "reports/tables/function_per_assay_auroc.csv", index=False)
    pd.concat(calibration_tables, ignore_index=True).to_csv(ROOT / "reports/tables/function_calibration_curve.csv", index=False)

    final_lgbm = make_classifier(best_params, seed, n_jobs)
    final_lgbm.fit(prepare_x(train, ABLATIONS["tier1_full"], categories), train.label.to_numpy(int))
    final_logistic = make_logistic(categories, NUMERIC_TIER1, seed)
    final_logistic.fit(train[ABLATIONS["tier1_full"]], train.label.to_numpy(int))
    model_dir = ROOT / "models" / MODEL_SHORT
    model_dir.mkdir(parents=True, exist_ok=True)
    lgbm_path = model_dir / "function_tier1.joblib"
    logistic_path = model_dir / "function_logistic_tier1.joblib"
    joblib.dump({"estimator": final_lgbm, "calibrator": calibrator, "calibration_method": calibration_method,
                 "feature_list": ABLATIONS["tier1_full"], "category_lists": categories}, lgbm_path)
    joblib.dump(final_logistic, logistic_path)
    esm_model, esm_short = read_feature_model(ROOT / "data/processed/features_tier1.parquet")
    if esm_model != "facebook/esm2_t12_35M_UR50D" or esm_short != MODEL_SHORT:
        raise RuntimeError(f"unexpected feature ESM model: {esm_model} / {esm_short}")
    proteins = pd.read_parquet(ROOT / "data/interim/train_proteins.parquet")
    train_ids = set(train.protein_id.astype(str))
    max_length = int(proteins.loc[proteins.protein_id.isin(train_ids), "length"].max())
    metric_full = pd.DataFrame(metric_records).set_index("model").loc["tier1_full_calibrated"].to_dict()
    metadata = {
        "task": "function_binary_classification",
        "tier": "tier1",
        "model_file": lgbm_path.name,
        "model_sha256": sha256(lgbm_path),
        "logistic_model_file": logistic_path.name,
        "logistic_model_sha256": sha256(logistic_path),
        "feature_list": ABLATIONS["tier1_full"],
        "categorical_features": CATEGORICAL,
        "category_lists": categories,
        "n_proteins": int(train.protein_id.nunique()),
        "n_rows": int(len(train)),
        "label_prevalence": float(train.label.mean()),
        "max_training_protein_length": max_length,
        "esm_model": esm_model,
        "esm_model_short": esm_short,
        "calibration_method": calibration_method,
        "calibration_selection": "grouped cross-fitted Brier score on OOF probabilities",
        "calibration_mean_brier": calibration_cv.groupby("method").brier.mean().to_dict(),
        "library_versions": {"lightgbm": lightgbm.__version__, "scikit_learn": sklearn.__version__,
                              "pandas": pd.__version__, "numpy": np.__version__},
        "config_sha256": sha256(config_path),
        "training_labels_sha256": sha256(ROOT / "data/processed/labels_function.parquet"),
        "features_sha256": sha256(ROOT / "data/processed/features_tier1.parquet"),
        "selected_hyperparameters": best_params,
        "search_runtime_seconds": round(search_runtime, 6),
        "search_runtime_this_run_seconds": round(search_runtime_this_run, 6),
        "search_runtime_scope": "cache_reuse_lookup" if n_cached == 20 else "active_search",
        "initial_search_runtime_captured": previous_summary.get("search_runtime_scope") == "active_search",
        "search_runtime_note": ("all 20 configurations reused; the uncached search runtime was not saved because its first run hit the terminal timeout"
                                if n_cached == 20 and previous_summary.get("search_runtime_scope") != "active_search" else ""),
        "search_configs_reused": n_cached,
        "oof_metrics_calibrated": metric_full,
        "held_out_test_used": False,
        "label_direction_caveat": "ProteinGym assay signs were mixed in the precheck; pooled-mean orientation remains provisional.",
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (model_dir / "function_tier1.json").write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    summary = {
        "n_training_proteins": int(train.protein_id.nunique()),
        "n_training_rows": int(len(train)),
        "n_training_clusters": int(train.cluster_id.nunique()),
        "n_test_clusters_excluded": int(len(test_clusters)),
        "n_oof_rows": int(len(oof)),
        "n_oof_missing": int(oof.filter(like="p_").isna().sum().sum()),
        "n_search_configs_completed": int(search.status.eq("complete").sum()),
        "search_runtime_seconds": round(search_runtime, 6),
        "search_runtime_this_run_seconds": round(search_runtime_this_run, 6),
        "search_runtime_scope": "cache_reuse_lookup" if n_cached == 20 else "active_search",
        "initial_search_runtime_captured": previous_summary.get("search_runtime_scope") == "active_search",
        "search_runtime_note": ("all 20 configurations reused; the uncached search runtime was not saved because its first run hit the terminal timeout"
                                if n_cached == 20 and previous_summary.get("search_runtime_scope") != "active_search" else ""),
        "search_configs_reused": n_cached,
        "build_runtime_seconds": round(time.perf_counter() - started, 6),
        "max_training_protein_length": max_length,
        "label_prevalence": float(train.label.mean()),
        "esm_model": esm_model,
        "esm_model_short": esm_short,
        "calibration_method": calibration_method,
        "tier2_model_exists": False,
        "held_out_test_used": False,
    }
    (ROOT / "reports/tables/phase_07_function_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
    print("Predictions are computational hypotheses, not experimental findings.")
