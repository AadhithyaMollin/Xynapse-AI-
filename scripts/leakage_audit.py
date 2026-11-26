#!/usr/bin/env python3
"""Static/data leakage checks for grouped splits, frozen artifacts, and OOF coverage."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
MUT_KEYS = ["protein_id", "position", "wt", "mut"]


def check_static_training_fits(root: Path = ROOT) -> tuple[bool, str]:
    suspicious = []
    checked = 0
    fit_call = re.compile(r"\.fit\s*\(")
    allowed_fit_tokens = ("train", "fit_frame", "x_fit", "score")
    for path in (root / "scripts/train_stability.py", root / "scripts/train_function.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if fit_call.search(line):
                checked += 1
                if not any(token in line for token in allowed_fit_tokens) or any(
                    token in line for token in ("test_frame", "test_data", "all_rows", "full_table")
                ):
                    suspicious.append(f"{path.name}:{line_number}: {line.strip()}")
    return not suspicious and checked > 0, f"fit_calls={checked}; suspicious={suspicious}"


def audit() -> list[tuple[str, bool, str]]:
    processed = ROOT / "data/processed"
    assignments = pd.read_parquet(ROOT / "data/interim/split_assignments.parquet")
    checks = []
    for task, filename in (("stability", "train_ddg.parquet"), ("function", "train_function.parquet")):
        frame = pd.read_parquet(processed / filename)
        train_clusters = set(frame.loc[frame.split_id.eq("train"), "cluster_id"].astype(str))
        test_clusters = set(frame.loc[frame.split_id.eq("test"), "cluster_id"].astype(str))
        checks.append((f"{task} cluster sets are disjoint across train/test",
                       not train_clusters.intersection(test_clusters),
                       f"train_clusters={len(train_clusters)}; test_clusters={len(test_clusters)}"))
    case_ids = {"P60484", "P04637"}
    case_clusters = set(assignments.loc[assignments.protein_id.isin(case_ids), "cluster_id"].astype(str))
    for task, filename in (("stability", "train_ddg.parquet"), ("function", "train_function.parquet")):
        train = pd.read_parquet(processed / filename).loc[lambda frame: frame.split_id.eq("train")]
        has_case_id = bool(set(train.protein_id.astype(str)).intersection(case_ids))
        has_case_cluster = bool(set(train.cluster_id.astype(str)).intersection(case_clusters))
        checks.append((f"{task} training excludes case-study proteins and their clusters",
                       not has_case_id and not has_case_cluster,
                       f"case_ids={has_case_id}; shared_case_clusters={has_case_cluster}"))
    clinvar_ids = set(pd.read_parquet(processed / "external_clinvar.parquet").protein_id.astype(str))
    clinvar_clusters = set(assignments.loc[assignments.protein_id.astype(str).isin(clinvar_ids), "cluster_id"].astype(str))
    for task, filename in (("stability", "train_ddg.parquet"), ("function", "train_function.parquet")):
        train = pd.read_parquet(processed / filename).loc[lambda frame: frame.split_id.eq("train")]
        shared = set(train.cluster_id.astype(str)).intersection(clinvar_clusters)
        checks.append((f"{task} training excludes all ClinVar-test clusters", not shared,
                       f"shared_clusters={sorted(shared)}"))

    for name, filename in (("labels_ddg", "labels_ddg.parquet"),
                           ("labels_function", "labels_function.parquet"),
                           ("external_clinvar", "external_clinvar.parquet")):
        frame = pd.read_parquet(processed / filename)
        duplicates = int(frame.duplicated(MUT_KEYS).sum())
        checks.append((f"{name} has no duplicate mutation keys", duplicates == 0,
                       f"duplicate_keys={duplicates}"))

    feature_lists = {}
    for task in ("stability", "function"):
        metadata = json.loads((ROOT / f"models/esm2_t12_35M/{task}_tier1.json").read_text())
        feature_lists[task] = metadata["feature_list"]
    forbidden_features = [feature for features in feature_lists.values() for feature in features
                          if any(token in feature.lower() for token in ("clinvar", "review_stars", "clinical"))]
    checks.append(("model feature inventories contain no ClinVar-derived fields",
                   not forbidden_features, f"forbidden_features={forbidden_features}"))

    static_ok, static_detail = check_static_training_fits()
    checks.append(("training fit calls are confined to fold/train variables (static check)", static_ok, static_detail))

    registry = json.loads((ROOT / "reports/test_eval_registry.json").read_text())
    expected_evaluations = {"stability_cluster_test", "function_cluster_test", "case_study_dms_P60484",
                            "case_study_dms_P04637", "clinvar_external_test"}
    evaluations = registry.get("evaluations", {})
    registry_ok = set(evaluations) == expected_evaluations and all(
        entry.get("status") == "completed" and "re_evaluation_reason" not in entry
        for entry in evaluations.values()
    ) and not registry.get("failures")
    checks.append(("test registry contains one completed, non-repeated entry per approved evaluation",
                   registry_ok, f"entries={sorted(evaluations)}"))

    from xynapse.freeze import verify_frozen
    frozen_path = ROOT / "models/esm2_t12_35M/FROZEN.json"
    frozen_ok, hash_problems = verify_frozen(frozen_path, frozen_path.parent)
    checks.append(("frozen model hashes match all model artifacts", frozen_ok and not hash_problems,
                   f"problems={hash_problems}"))

    from xynapse.features import read_feature_model
    from xynapse.esm_scoring import read_esm_model
    esm_names = {
        read_feature_model(processed / "features_tier1.parquet")[0],
        read_esm_model(processed / "mut_esm.parquet"),
        json.loads((ROOT / "models/esm2_t12_35M/stability_tier1.json").read_text())["esm_model"],
        json.loads((ROOT / "models/esm2_t12_35M/function_tier1.json").read_text())["esm_model"],
        json.loads(frozen_path.read_text())["esm_model"],
    }
    checks.append(("feature tables and model metadata use one ESM model",
                   len(esm_names) == 1 and "facebook/esm2_t12_35M_UR50D" in esm_names,
                   f"models={sorted(esm_names)}"))

    for task, filename in (("stability", "train_ddg.parquet"), ("function", "train_function.parquet")):
        training = pd.read_parquet(processed / filename).loc[lambda frame: frame.split_id.eq("train")]
        oof = pd.read_parquet(ROOT / f"reports/tables/oof_{task if task == 'function' else 'ddg'}.parquet")
        keys = MUT_KEYS
        expected = set(training[keys].itertuples(index=False, name=None))
        observed = set(oof[keys].itertuples(index=False, name=None))
        duplicates = int(oof.duplicated(keys).sum())
        checks.append((f"{task} OOF covers each training mutation exactly once",
                       expected == observed and duplicates == 0,
                       f"training_rows={len(training)}; oof_rows={len(oof)}; duplicate_oof={duplicates}"))
    return checks


def main() -> int:
    checks = audit()
    print(DISCLAIMER)
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name} | {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
