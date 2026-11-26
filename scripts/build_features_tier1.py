#!/usr/bin/env python3
"""Resumably build complete Tier-1 features for all Phase 3 proteins."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import yaml

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", value)


def build() -> dict[str, Path]:
    from xynapse.esm_scoring import load_esm, read_esm_model, resolve_device, save_mut_esm, score_protein
    from xynapse.features import add_tier1_features, model_short_name, save_features_tier1
    from xynapse.schema import (MUT_ESM_COLS, MUT_KEYS, MUT_PHYS_COLS, POS_EVO_COLS,
                               POS_FUNC_COLS, POS_KEYS, POS_STRUCT_COLS, TIER1_EXTRA_COLS)
    from xynapse.validate import validate_table

    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    model_name = str(config["esm"]["dev_model"])
    short = model_short_name(model_name)
    seed = int(config.get("seed", 42))
    model_file_suffix = model_name.rsplit("/", 1)[-1]
    device = resolve_device()
    proteins_path = ROOT / "data/interim/train_proteins.parquet"
    proteins = pd.read_parquet(proteins_path)
    protein_ids = proteins["protein_id"].astype(str).tolist()
    sanitized = {protein_id: safe_id(protein_id) for protein_id in protein_ids}
    reverse = {}
    for protein_id, filename_id in sanitized.items():
        if filename_id in reverse and reverse[filename_id] != protein_id:
            raise ValueError(f"Sanitized identifier collision: {reverse[filename_id]} and {protein_id}")
        reverse[filename_id] = protein_id

    output_paths = {
        "features": ROOT / "data/processed/features_tier1.parquet",
        "runtime": ROOT / "reports/tables/phase_04_feature_runtime.csv",
        "manifest": ROOT / "reports/tables/training_protein_manifest.csv",
        "request": ROOT / "reports/teammate_request.md",
        "direction": ROOT / "reports/tables/label_direction_check.csv",
        "summary": ROOT / "reports/tables/phase_04_feature_summary.json",
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = ROOT / "data/interim/esm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    model_components = load_esm(model_name, device=device)
    all_esm = []
    runtime_rows = []
    for row in proteins.itertuples(index=False):
        protein_id = str(row.protein_id)
        sequence = str(row.sequence).upper()
        cache = cache_dir / f"{sanitized[protein_id]}_{model_file_suffix}.parquet"
        started = time.perf_counter()
        reused = cache.exists()
        if reused:
            cache_model = read_esm_model(cache)
            if cache_model != model_name:
                raise ValueError(f"Refusing mixed ESM cache: {cache} has {cache_model}")
            esm = pd.read_parquet(cache)
        else:
            esm = score_protein(protein_id, sequence, model_name, batch_size=16,
                                device=device, model_components=model_components)
            save_mut_esm(esm, cache, model_name)
        single_protein = proteins[proteins["protein_id"].eq(protein_id)]
        validation = validate_table(esm, "mut_esm", single_protein)
        if validation:
            raise ValueError(f"ESM cache/table failed contract for {protein_id}: {validation}")
        if len(esm) != 19 * len(sequence):
            raise ValueError(f"{protein_id}: ESM cache has {len(esm)} rows, expected {19 * len(sequence)}")
        all_esm.append(esm[MUT_KEYS + MUT_ESM_COLS])
        runtime_rows.append({
            "protein_id": protein_id, "length": len(sequence), "rows": len(esm),
            "cache_reused": reused, "runtime_seconds": time.perf_counter() - started,
            "esm_model": model_name, "esm_model_short": short,
        })

    esm_all = pd.concat(all_esm, ignore_index=True)
    if esm_all["protein_id"].nunique() != len(proteins):
        raise ValueError("ESM feature set does not cover every training protein")
    feature_frames = []
    for row in proteins.itertuples(index=False):
        pid = str(row.protein_id)
        subset = esm_all[esm_all["protein_id"].eq(pid)]
        feature_frames.append(add_tier1_features(subset, str(row.sequence)))
    features = pd.concat(feature_frames, ignore_index=True)
    expected_rows = int((proteins["length"] * 19).sum())
    if len(features) != expected_rows:
        raise ValueError(f"Expected {expected_rows} complete Tier-1 rows, found {len(features)}")
    tier1_cols = MUT_ESM_COLS + MUT_PHYS_COLS + TIER1_EXTRA_COLS
    if features[tier1_cols].isna().any().any() or not np.isfinite(features[tier1_cols].to_numpy(dtype=float)).all():
        raise ValueError("Tier-1 features contain missing/non-finite values")
    if features.duplicated(MUT_KEYS).any():
        raise ValueError("Tier-1 feature table contains duplicate mutation keys")
    save_features_tier1(features, output_paths["features"], model_name)

    pd.DataFrame(runtime_rows).to_csv(output_paths["runtime"], index=False)
    teammate_proteins_path = ROOT / "data/processed/proteins.parquet"
    teammate_proteins = pd.read_parquet(teammate_proteins_path)
    teammate_sequences = dict(zip(teammate_proteins["protein_id"].astype(str),
                                  teammate_proteins["sequence"].astype(str).str.upper()))
    teammate_tables = {}
    for table_name in ("pos_structure", "pos_evolution", "pos_function"):
        table_path = ROOT / f"data/processed/{table_name}.parquet"
        teammate_tables[table_name] = set(pd.read_parquet(table_path)["protein_id"].astype(str).unique()) if table_path.exists() else set()
    ddg_ids = set(pd.read_csv(ROOT / "reports/tables/proposed_training_ddg_domains.csv")["domain"].astype(str))
    function_ids = set(pd.read_csv(ROOT / "reports/tables/proposed_training_function_assays.csv")["DMS_id"].astype(str))
    manifest_rows = []
    for row in proteins.itertuples(index=False):
        pid = str(row.protein_id)
        dataset = "Tsuboyama" if pid in ddg_ids else "ProteinGym" if pid in function_ids else "case_study"
        matching = [name for name, ids in teammate_tables.items() if pid in ids]
        sequence_match = ("NA" if pid not in teammate_sequences else
                          "yes" if str(row.sequence).upper() == teammate_sequences[pid] else "no")
        manifest_rows.append({"protein_id": pid, "dataset": dataset, "length": int(row.length),
                              "tier2_tables_present": ";".join(matching),
                              "sequence_equals_teammate": sequence_match})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_paths["manifest"], index=False)

    request_candidates = proteins[~proteins["protein_id"].isin(["P60484", "P04637"])].copy()
    request_candidates = request_candidates.merge(manifest[["protein_id", "dataset"]],
                                                  on="protein_id", how="left", validate="one_to_one")
    request_candidates = request_candidates.sample(n=min(15, len(request_candidates)), random_state=seed)
    request_lines = [
        "# BIO-1/BIO-2 position-feature request",
        "",
        DISCLAIMER,
        "",
        "Please provide position tables for the following sampled training proteins where feasible. IDs are kept as their dataset identifiers unless an accession is explicitly verified by exact sequence match. Include `protein_id`, 1-based `position`, and `wt` in every position table.",
        "",
        "Requested position-table columns:",
        "",
        f"- `pos_structure`: {', '.join(POS_KEYS + POS_STRUCT_COLS)}",
        f"- `pos_evolution`: {', '.join(POS_KEYS + POS_EVO_COLS)}",
        f"- `pos_function`: {', '.join(POS_KEYS + POS_FUNC_COLS)}",
        f"- Per-mutation evolution: {', '.join(MUT_KEYS + ['msa_logodds'])}",
        "",
        "Training protein requests (dataset ID followed by exact sequence):",
        "",
    ]
    for row in request_candidates.sort_values("protein_id").itertuples(index=False):
        request_lines.extend([f"## {row.protein_id} ({row.dataset}, length {row.length})",
                              "", f"`{row.sequence}`", ""])
    request_lines.extend([
        "Please also provide a per-mutation `msa_logodds` table and review the PTEN `pos_evolution` table, whose saved entropy is degenerate at most positions. The PTEN/TP53 `pos_function.dist_to_nearest_site` field is null in the current teammate table; please fill it where supported by the annotation source, otherwise retain null and state why. No Tier-2 biology features were computed by this pipeline.",
        "",
    ])
    output_paths["request"].write_text("\n".join(request_lines))

    labels = pd.read_parquet(ROOT / "data/processed/labels_function.parquet")
    paired = labels.merge(features[["protein_id", "position", "wt", "mut", "esm_llr"]],
                          on=["protein_id", "position", "wt", "mut"], how="inner", validate="one_to_one")
    direction_rows = []
    for assay_id, group in paired.groupby("assay_id", sort=True):
        rho = scipy.stats.spearmanr(group["label"], group["esm_llr"]).statistic if len(group) >= 2 else np.nan
        direction_rows.append({"assay_id": assay_id, "phenotype": group["phenotype"].iloc[0],
                               "n_mutations": len(group), "spearman_project_label_esm_llr": float(rho) if pd.notna(rho) else np.nan})
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(output_paths["direction"], index=False)
    mean_rho = float(direction["spearman_project_label_esm_llr"].mean())
    if not np.isfinite(mean_rho) or mean_rho >= 0:
        raise ValueError(f"Full function-label direction check must have a negative mean Spearman; found {mean_rho}")

    elapsed = time.perf_counter() - started_all
    summary = {
        "runtime_seconds": elapsed, "n_proteins": len(proteins), "n_feature_rows": len(features),
        "esm_model": model_name, "esm_model_short": short, "device": device,
        "cache_reused_count": int(pd.DataFrame(runtime_rows)["cache_reused"].sum()),
        "tier1_nan_count": int(features[tier1_cols].isna().sum().sum()),
        "direction_assays": len(direction), "mean_spearman_project_label_esm_llr": mean_rho,
        "direction_pass": True, "teammate_manifest_tier2_available": int(manifest["tier2_tables_present"].ne("").sum()),
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    print(DISCLAIMER)
    print(f"Tier-1 model: {model_name} ({short}) on {device}")
    print(f"Proteins: {len(proteins)}; rows: {len(features)}; caches reused: {summary['cache_reused_count']}")
    print(f"Function assays in direction check: {len(direction)}; mean Spearman: {mean_rho:.6f}")
    print(f"Measured runtime: {elapsed:.2f} s")
    for key, path in output_paths.items():
        print(f"{key}: {path}")
    return output_paths


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
