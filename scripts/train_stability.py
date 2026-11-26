#!/usr/bin/env python3
"""Train and cross-validate the Tier-1 stability regressor on training clusters only."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import lightgbm
import yaml
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xynapse.features import read_feature_model  # noqa: E402
from xynapse.metrics import per_protein_spearman, regression_metrics, spearman  # noqa: E402
from xynapse.schema import AA  # noqa: E402

MODEL_SHORT = "esm2_t12_35M"
NUMERIC_TIER1 = [
    "esm_llr", "esm_wt_logp", "esm_mut_logp", "blosum62", "d_hydropathy",
    "d_volume", "d_charge", "esm_pos_mean_llr", "esm_pos_min_llr", "rel_position",
]
CATEGORICAL = ["wt", "mut"]
ABLATIONS = {
    "esm_only": ["esm_llr", "esm_wt_logp", "esm_mut_logp", *CATEGORICAL],
    "esm_plus_physchem": ["esm_llr", "esm_wt_logp", "esm_mut_logp", "blosum62",
                           "d_hydropathy", "d_volume", "d_charge", *CATEGORICAL],
    "tier1_full": [*NUMERIC_TIER1, *CATEGORICAL],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def categorical_categories(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {column: sorted(frame[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL}


def prepare_x(frame: pd.DataFrame, features: list[str], categories: dict[str, list[str]]) -> pd.DataFrame:
    x = frame[features].copy()
    for column in CATEGORICAL:
        if column in x:
            values = x[column].astype("string")
            values = values.where(values.isin(categories[column]))
            x[column] = pd.Categorical(values, categories=categories[column])
    return x


def make_model(params: dict, seed: int, n_jobs: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression", random_state=seed, n_jobs=n_jobs, verbosity=-1,
        deterministic=True, force_col_wise=True, subsample_freq=1,
        **params,
    )


def random_configs(seed: int, n: int = 20) -> list[dict]:
    space = {
        "num_leaves": [7, 15, 31, 63],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "n_estimators": [200, 400, 800],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0, 1, 10],
    }
    rng = random.Random(seed)
    candidates: dict[tuple, dict] = {}
    while len(candidates) < n:
        candidate = {name: rng.choice(values) for name, values in space.items()}
        candidates[tuple(candidate.values())] = candidate
    return list(candidates.values())


def cross_validated_predictions(
    frame: pd.DataFrame,
    feature_list: list[str],
    categories: dict[str, list[str]],
    params: dict,
    *,
    seed: int,
    n_jobs: int,
    expected_test_clusters: set[str],
    model_label: str,
) -> tuple[np.ndarray, list[dict]]:
    prediction = np.full(len(frame), np.nan, dtype=float)
    audits = []
    folds = sorted(frame.cv_fold.dropna().astype(int).unique())
    for fold in folds:
        validation_mask = frame.cv_fold.astype("Int64").eq(fold).to_numpy()
        train_mask = ~validation_mask
        fit_clusters = set(frame.loc[train_mask, "cluster_id"].astype(str))
        validation_clusters = set(frame.loc[validation_mask, "cluster_id"].astype(str))
        overlap = fit_clusters & validation_clusters
        test_overlap = fit_clusters & expected_test_clusters
        if overlap or test_overlap:
            raise AssertionError(f"{model_label} fold {fold}: cluster leakage; fit/validation={overlap}, test={test_overlap}")
        x_fit = prepare_x(frame.loc[train_mask], feature_list, categories)
        x_val = prepare_x(frame.loc[validation_mask], feature_list, categories)
        y_fit = frame.loc[train_mask, "ddg"].to_numpy(dtype=float)
        model = make_model(params, seed + int(fold), n_jobs)
        model.fit(x_fit, y_fit)
        prediction[validation_mask] = model.predict(x_val)
        audits.append({
            "model": model_label, "cv_fold": int(fold),
            "n_fit_rows": int(train_mask.sum()), "n_validation_rows": int(validation_mask.sum()),
            "n_fit_clusters": len(fit_clusters), "n_validation_clusters": len(validation_clusters),
            "fit_validation_cluster_overlap": len(overlap), "fit_test_cluster_overlap": len(test_overlap),
        })
    return prediction, audits


def tune_params(train: pd.DataFrame, categories: dict[str, list[str]], seed: int, n_jobs: int,
                wall_seconds: float = 600.0, cache_path: Path | None = None,
                data_fingerprint: str | None = None) -> tuple[dict, pd.DataFrame, float]:
    started = time.monotonic()
    groups = train.cluster_id.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=seed)
    configs = random_configs(seed, 20)
    cache_path = cache_path or ROOT / "data/interim/stability_hp_search_cache.json"
    fingerprint = data_fingerprint or "unspecified"
    cache = {"fingerprint": fingerprint, "results": []}
    if cache_path.is_file():
        existing = json.loads(cache_path.read_text())
        if existing.get("fingerprint") == fingerprint:
            cache = existing
    saved = {int(row["config_id"]): row for row in cache.get("results", [])
             if row.get("status") == "complete" and np.isfinite(row.get("mean_grouped_cv_spearman", np.nan))}
    records = []
    for index, params in enumerate(configs, start=1):
        if index in saved and saved[index].get("params") == json.dumps(params, sort_keys=True):
            records.append({**saved[index], "cache_reused": True})
            continue
        if time.monotonic() - started >= wall_seconds:
            records.append({"config_id": index, "params": json.dumps(params, sort_keys=True),
                            "mean_grouped_cv_spearman": np.nan, "completed_inner_folds": 0,
                            "status": "wall_cap_before_start"})
            continue
        scores = []
        status = "complete"
        for inner_fold, (fit_idx, val_idx) in enumerate(splitter.split(train, groups=groups)):
            if time.monotonic() - started >= wall_seconds:
                status = "wall_cap_during_config"
                break
            fit_frame, val_frame = train.iloc[fit_idx], train.iloc[val_idx]
            if set(fit_frame.cluster_id) & set(val_frame.cluster_id):
                raise AssertionError("inner grouped CV split a cluster")
            model = make_model(params, seed + index * 10 + inner_fold, n_jobs)
            model.fit(prepare_x(fit_frame, ABLATIONS["tier1_full"], categories), fit_frame.ddg.to_numpy(float))
            pred = model.predict(prepare_x(val_frame, ABLATIONS["tier1_full"], categories))
            scores.append(spearman(val_frame.ddg, pred))
        score = float(np.nanmean(scores)) if scores and np.isfinite(scores).any() else float("nan")
        row = {"config_id": index, "params": json.dumps(params, sort_keys=True),
               "mean_grouped_cv_spearman": score, "completed_inner_folds": len(scores),
               "status": status, "cache_reused": False}
        records.append(row)
        cache["results"] = [item for item in records if item.get("status") == "complete"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(cache, indent=2, allow_nan=True) + "\n")
        temp_path.replace(cache_path)
        if status == "wall_cap_during_config":
            break
    results = pd.DataFrame(records)
    complete = results.loc[results.status.eq("complete") & results.mean_grouped_cv_spearman.notna()]
    if complete.empty:
        raise RuntimeError("hyperparameter search produced no complete finite-score configuration")
    best_index = int(complete.mean_grouped_cv_spearman.idxmax())
    best = json.loads(complete.loc[best_index, "params"])
    return best, results, time.monotonic() - started


def run() -> dict:
    run_started = time.perf_counter()
    config_path = ROOT / "configs/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    seed = int(config["seed"])
    train_table = pd.read_parquet(ROOT / "data/processed/train_ddg.parquet")
    train = train_table.loc[train_table.split_id.eq("train")].copy().reset_index(drop=True)
    test_clusters = set(train_table.loc[train_table.split_id.eq("test"), "cluster_id"].astype(str))
    if train.empty or train.cluster_id.isna().any() or train.cv_fold.isna().any():
        raise RuntimeError("training rows must have non-null cluster IDs and CV folds")
    if train.cluster_id.nunique() < 5:
        raise RuntimeError("fewer than five training clusters are available")
    if train_table.loc[train_table.split_id.eq("test"), "cluster_id"].isna().any():
        raise RuntimeError("test clusters are missing identifiers")
    if set(train.cluster_id.astype(str)) & test_clusters:
        raise AssertionError("train/test cluster overlap")
    if train.duplicated(["protein_id", "position", "wt", "mut"]).any():
        raise RuntimeError("training label table contains duplicate mutation keys")
    if not set(NUMERIC_TIER1 + CATEGORICAL + ["ddg", "cluster_id", "cv_fold"]) <= set(train.columns):
        raise RuntimeError("training table lacks one or more required Tier-1 or split columns")
    categories = categorical_categories(train)
    n_jobs = max(1, min(4, os.cpu_count() or 1))

    # OOF baseline predictions use training rows only.
    baseline_functions = {
        "negative_blosum62": lambda f: -f.blosum62.to_numpy(float),
        "abs_d_hydropathy": lambda f: np.abs(f.d_hydropathy.to_numpy(float)),
        "abs_d_volume": lambda f: np.abs(f.d_volume.to_numpy(float)),
        "negative_esm_llr": lambda f: -f.esm_llr.to_numpy(float),
    }
    baseline_records = []
    baseline_predictions = {}
    for name, score_fn in baseline_functions.items():
        pred = score_fn(train)
        baseline_predictions[name] = pred
        for fold in sorted(train.cv_fold.astype(int).unique()):
            subset = train.cv_fold.astype(int).eq(fold)
            baseline_metrics = regression_metrics(train.loc[subset, "ddg"], pred[subset])
            baseline_records.append({"baseline": name, "cv_fold": int(fold),
                                     **baseline_metrics, "n_rows": baseline_metrics["n"],
                                     "n_proteins": int(train.loc[subset, "protein_id"].nunique())})
    baseline_cv = pd.DataFrame(baseline_records)
    baseline_cv.to_csv(ROOT / "reports/tables/stability_baseline_cv.csv", index=False)
    contradicted = [name for name, group in baseline_cv.groupby("baseline")
                    if group.spearman.notna().all() and group.spearman.lt(0).all()]
    (ROOT / "reports/tables/stability_baseline_sign_check.json").write_text(json.dumps({
        "expected_orientation": "higher baseline score means more destabilizing",
        "baseline_mean_spearman": baseline_cv.groupby("baseline").spearman.mean().to_dict(),
        "all_folds_negative_baselines": contradicted,
        "training_only": True,
    }, indent=2) + "\n")
    if contradicted:
        raise RuntimeError(f"baseline sign contradicts expected orientation in every fold: {contradicted}")

    hp_fingerprint = hashlib.sha256((sha256(ROOT / "data/processed/train_ddg.parquet")
                                     + sha256(ROOT / "data/processed/features_tier1.parquet")
                                     + sha256(config_path)).encode()).hexdigest()
    previous_summary_path = ROOT / "reports/tables/phase_06_stability_summary.json"
    previous_summary = json.loads(previous_summary_path.read_text()) if previous_summary_path.is_file() else {}
    best_params, search, search_runtime_this_run = tune_params(
        train, categories, seed, n_jobs, wall_seconds=600.0,
        cache_path=ROOT / "data/interim/stability_hp_search_cache.json",
        data_fingerprint=hp_fingerprint,
    )
    n_cached_configs = int(search.get("cache_reused", pd.Series(dtype=bool)).fillna(False).sum())
    search_runtime = (float(previous_summary["search_runtime_seconds"])
                      if n_cached_configs == 20 and previous_summary.get("search_runtime_seconds")
                      else search_runtime_this_run)
    search.to_csv(ROOT / "reports/tables/stability_hyperparameter_search.csv", index=False)
    if int((search.status == "complete").sum()) < 20:
        warnings.warn("10-minute search cap elapsed before all 20 configurations completed", RuntimeWarning)

    all_oof = pd.DataFrame({
        "protein_id": train.protein_id,
        "position": train.position,
        "wt": train.wt,
        "mut": train.mut,
        "cluster_id": train.cluster_id,
        "cv_fold": train.cv_fold.astype(int),
        "y_true": train.ddg.astype(float),
    })
    audits: list[dict] = []
    for ablation, feature_list in ABLATIONS.items():
        pred, fold_audits = cross_validated_predictions(
            train, feature_list, categories, best_params, seed=seed, n_jobs=n_jobs,
            expected_test_clusters=test_clusters, model_label=ablation,
        )
        if not np.isfinite(pred).all():
            raise AssertionError(f"{ablation}: OOF predictions do not cover every training row exactly once")
        all_oof[f"pred_{ablation}"] = pred
        audits.extend(fold_audits)
    pd.DataFrame(audits).to_csv(ROOT / "reports/tables/stability_fit_audit.csv", index=False)
    all_oof.to_parquet(ROOT / "reports/tables/oof_ddg.parquet", index=False)

    metrics_records = []
    per_protein_tables = []
    for model_name in [*baseline_functions, *[f"pred_{name}" for name in ABLATIONS]]:
        pred_col = model_name if model_name in baseline_functions else model_name
        if model_name in baseline_functions:
            values = baseline_predictions[model_name]
        else:
            values = all_oof[pred_col].to_numpy(float)
        metrics = regression_metrics(train.ddg, values)
        metrics_records.append({
            "model": model_name.replace("pred_", ""),
            **metrics,
            "n_rows": metrics["n"],
            "n_proteins": int(train.protein_id.nunique()),
            "tier": "tier1",
        })
        pp, n_qual = per_protein_spearman(pd.DataFrame({
            "protein_id": train.protein_id, "y_true": train.ddg, "y_pred": values,
        }), min_variants=30)
        pp["model"] = model_name.replace("pred_", "")
        pp["n_qualifying_proteins"] = n_qual
        per_protein_tables.append(pp)
    metrics_frame = pd.DataFrame(metrics_records)
    metrics_frame.to_csv(ROOT / "reports/tables/stability_oof_metrics.csv", index=False)
    pp_frame = pd.concat(per_protein_tables, ignore_index=True)
    pp_frame.to_csv(ROOT / "reports/tables/stability_per_protein_oof.csv", index=False)

    final_model = make_model(best_params, seed, n_jobs)
    final_model.fit(prepare_x(train, ABLATIONS["tier1_full"], categories), train.ddg.to_numpy(float))
    model_dir = ROOT / "models" / MODEL_SHORT
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "stability_tier1.joblib"
    joblib.dump(final_model, model_path)
    model_id_path = ROOT / "data/processed/features_tier1.parquet"
    esm_model, esm_short = read_feature_model(model_id_path)
    if esm_short != MODEL_SHORT or "35M" not in esm_model:
        raise RuntimeError(f"unexpected ESM feature model: {esm_model} ({esm_short})")
    proteins = pd.read_parquet(ROOT / "data/interim/train_proteins.parquet")
    train_protein_ids = set(train.protein_id.astype(str))
    max_length = int(proteins.loc[proteins.protein_id.isin(train_protein_ids), "length"].max())
    metadata = {
        "task": "stability_regression",
        "tier": "tier1",
        "model_file": model_path.name,
        "model_sha256": sha256(model_path),
        "feature_list": ABLATIONS["tier1_full"],
        "categorical_features": CATEGORICAL,
        "category_lists": categories,
        "n_proteins": int(train.protein_id.nunique()),
        "n_rows": int(len(train)),
        "max_training_protein_length": max_length,
        "esm_model": esm_model,
        "esm_model_short": esm_short,
        "library_versions": {"lightgbm": lightgbm.__version__, "scikit_learn": sklearn.__version__,
                              "pandas": pd.__version__, "numpy": np.__version__},
        "config_sha256": sha256(config_path),
        "training_labels_sha256": sha256(ROOT / "data/processed/labels_ddg.parquet"),
        "features_sha256": sha256(model_id_path),
        "selected_hyperparameters": best_params,
        "search_runtime_seconds": round(search_runtime, 6),
        "search_runtime_this_run_seconds": round(search_runtime_this_run, 6),
        "search_configs_reused": n_cached_configs,
        "oof_metrics": metrics_frame.loc[metrics_frame.model.eq("tier1_full")].iloc[0].to_dict(),
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "held_out_test_used": False,
    }
    (model_dir / "stability_tier1.json").write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    summary = {
        "n_training_proteins": int(train.protein_id.nunique()),
        "n_training_rows": int(len(train)),
        "n_training_clusters": int(train.cluster_id.nunique()),
        "n_test_clusters_excluded": int(len(test_clusters)),
        "n_oof_rows": int(len(all_oof)),
        "n_oof_missing": int(all_oof.filter(like="pred_").isna().sum().sum()),
        "n_search_configs_completed": int((search.status == "complete").sum()),
        "search_runtime_seconds": round(search_runtime, 6),
        "search_runtime_this_run_seconds": round(search_runtime_this_run, 6),
        "search_configs_reused": n_cached_configs,
        "build_runtime_seconds": round(time.perf_counter() - run_started, 6),
        "max_training_protein_length": max_length,
        "esm_model": esm_model,
        "esm_model_short": esm_short,
        "tier2_model_exists": False,
        "held_out_test_used": False,
    }
    (ROOT / "reports/tables/phase_06_stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
    print("Predictions are computational hypotheses, not experimental findings.")
