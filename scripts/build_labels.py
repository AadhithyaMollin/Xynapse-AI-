#!/usr/bin/env python3
"""Build Phase 3 protein and label tables from the inspected source files."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")
SINGLE_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
WT_TOKENS = {"wt", "wildtype", "wild_type", "wild-type"}


def is_reliable(delta_g: object, ci: object, ci_max: float = 0.5,
                dg_min: float = -1.0, dg_max: float = 5.0) -> bool:
    if pd.isna(delta_g) or pd.isna(ci):
        return False
    delta_g, ci = float(delta_g), float(ci)
    return math.isfinite(delta_g) and math.isfinite(ci) and ci <= ci_max and dg_min <= delta_g <= dg_max


def parse_single(value: object) -> tuple[str, int, str] | None:
    match = SINGLE_RE.fullmatch(str(value))
    return (match.group(1), int(match.group(2)), match.group(3)) if match else None


def add_protein(records: dict[str, dict], protein_id: str, sequence: str,
                uniprot_id: str | None = None) -> bool:
    seq = str(sequence).upper()
    if len(seq) > 1022 or not seq or set(seq) - AA:
        return False
    if protein_id in records:
        prior = records[protein_id]
        if prior["sequence"] != seq or prior["uniprot_id"] != uniprot_id:
            raise ValueError(f"Protein identifier collision for {protein_id}")
        return True
    records[protein_id] = {
        "protein_id": protein_id, "uniprot_id": uniprot_id, "sequence": seq,
        "length": len(seq), "structure_source": None, "mean_plddt": np.nan,
    }
    return True


def read_training_proteins() -> tuple[pd.DataFrame, dict[str, dict], list[dict]]:
    proteins_path = ROOT / "data/processed/proteins.parquet"
    case = pd.read_parquet(proteins_path)
    record_map: dict[str, dict] = {}
    dropped: list[dict] = []
    for row in case.itertuples(index=False):
        if row.protein_id in {"P04637", "P60484"}:
            if not add_protein(record_map, str(row.protein_id), str(row.sequence), str(row.uniprot_id)):
                raise ValueError(f"Case-study sequence is unsupported: {row.protein_id}")

    singles = pd.read_csv(ROOT / "data/raw/tsuboyama/Single_DMS_list.csv",
                          usecols=["name", "aa_seq", "DMS_group"], dtype=str)
    selected_domains = pd.read_csv(ROOT / "reports/tables/proposed_training_ddg_domains.csv")
    domain_map = singles.set_index("name")["aa_seq"].to_dict()
    for domain in selected_domains["domain"]:
        if domain not in domain_map:
            raise ValueError(f"Selected stability domain missing from Single_DMS_list: {domain}")
        if not add_protein(record_map, str(domain), domain_map[domain], None):
            dropped.append({"protein_id": str(domain), "source": "Tsuboyama", "reason": "length_or_noncanonical_sequence"})

    reference = pd.read_csv(ROOT / "data/raw/proteingym/DMS_substitutions.csv", dtype=str)
    chosen = pd.read_csv(ROOT / "reports/tables/proposed_training_function_assays.csv", dtype=str)
    ref_index = reference.set_index("DMS_id", drop=False)
    for assay_id in chosen["DMS_id"]:
        if assay_id not in ref_index.index:
            raise ValueError(f"Selected ProteinGym assay missing from reference: {assay_id}")
        row = ref_index.loc[assay_id]
        if not add_protein(record_map, str(assay_id), str(row["target_seq"]), None):
            dropped.append({"protein_id": str(assay_id), "source": "ProteinGym", "reason": "length_or_noncanonical_sequence"})

    frame = pd.DataFrame(record_map.values()).sort_values("protein_id").reset_index(drop=True)
    return frame, record_map, dropped


def build_stability_labels(protein_map: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    proposal = pd.read_csv(ROOT / "reports/tables/proposed_training_ddg_domains.csv")
    offsets = {str(row.domain): (0 if pd.isna(row.constant_offset) else int(row.constant_offset))
               for row in proposal.itertuples(index=False)}
    selected = set(offsets) & set(protein_map)
    if len(selected) < 10:
        raise ValueError(f"Fewer than 10 supported Tsuboyama proteins remain: {len(selected)}")
    wt_rows: dict[str, list[float]] = {domain: [] for domain in selected}
    observations: dict[tuple[str, int, str, str], list[tuple[float, float]]] = {}
    rejected = {"unknown_name_class": 0, "unreliable_mutation": 0, "wt_mismatch_after_offset": 0}
    input_path = ROOT / "data/raw/tsuboyama/Tsuboyama2023_Dataset1_20230416.csv"
    started = time.perf_counter()
    from xynapse.labels import stability_project_ddg
    for chunk in pd.read_csv(input_path,
                             usecols=["name", "deltaG", "deltaG_95CI"],
                             chunksize=100_000, low_memory=False):
        for row in chunk.itertuples(index=False):
            name = str(row.name)
            if name in selected:
                if is_reliable(row.deltaG, row.deltaG_95CI):
                    wt_rows[name].append(float(row.deltaG))
                continue
            domain, separator, token = name.rpartition("_")
            if not separator or domain not in selected:
                continue
            if token.lower() in WT_TOKENS:
                if is_reliable(row.deltaG, row.deltaG_95CI):
                    wt_rows[domain].append(float(row.deltaG))
                continue
            parsed = parse_single(token)
            if parsed is None:
                rejected["unknown_name_class"] += 1
                continue
            if not is_reliable(row.deltaG, row.deltaG_95CI):
                rejected["unreliable_mutation"] += 1
                continue
            wt, reported_position, mut = parsed
            position = reported_position - offsets[domain]
            sequence = protein_map[domain]["sequence"]
            if position < 1 or position > len(sequence) or sequence[position - 1] != wt:
                rejected["wt_mismatch_after_offset"] += 1
                continue
            key = (domain, position, wt, mut)
            observations.setdefault(key, []).append((float(row.deltaG), float(row.deltaG_95CI)))

    wt_reference = {}
    for domain in selected:
        values = wt_rows[domain]
        if not values:
            raise ValueError(f"No reliable WT reference found while building labels for {domain}")
        wt_reference[domain] = float(pd.Series(values, dtype="float64").median())
    grouped = []
    for (protein_id, position, wt, mut), values in sorted(observations.items()):
        ddgs = [stability_project_ddg(wt_reference[protein_id], delta_g) for delta_g, _ in values]
        grouped.append({
            "protein_id": protein_id, "position": position, "wt": wt, "mut": mut,
            "ddg": float(np.mean(ddgs)), "source": "megascale", "split_id": None,
            "cluster_id": None, "n_obs": len(values), "domain_name": protein_id,
            "deltaG_95CI": float(np.mean([ci for _, ci in values])),
        })
    elapsed = time.perf_counter() - started
    return pd.DataFrame(grouped), {"runtime_seconds": elapsed, "rejected": rejected,
                                   "n_wt_rows_by_domain": {k: len(v) for k, v in wt_rows.items()},
                                   "n_unique_mutations": len(grouped),
                                   "n_reliable_observations": sum(len(v) for v in observations.values()),
                                   "n_collapsed_observation_rows": sum(max(0, len(v) - 1) for v in observations.values())}


def build_function_labels(protein_map: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    import zipfile
    from xynapse.labels import function_project_label, within_assay_percentile

    proposal = pd.read_csv(ROOT / "reports/tables/proposed_training_function_assays.csv", dtype=str)
    audit = pd.read_csv(ROOT / "reports/tables/proteingym_assay_summary.csv", dtype=str).set_index("DMS_id")
    direction = json.loads((ROOT / "reports/tables/proteingym_direction_precheck_summary.json").read_text())
    label_1_is_tolerated = bool(direction["label_1_is_tolerated_by_pooled_mean"])
    archive_path = ROOT / "data/raw/proteingym/DMS_ProteinGym_substitutions.zip"
    started = time.perf_counter()
    output = []
    rejected = {"non_single_or_unparsed": 0, "unlabeled_or_nonfinite_score": 0,
                "wt_mismatch_after_offset": 0, "conflicting_duplicate_labels": 0}
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for assay in proposal.itertuples(index=False):
            assay_id = str(assay.DMS_id)
            if assay_id not in protein_map:
                continue
            if assay_id not in audit.index:
                raise ValueError(f"Assay audit entry missing: {assay_id}")
            metadata = audit.loc[assay_id]
            member = f"DMS_ProteinGym_substitutions/{metadata['DMS_filename']}"
            if member not in members:
                raise ValueError(f"Selected assay member missing from archive: {member}")
            frame = pd.read_csv(archive.open(member), low_memory=False)
            label_column = str(metadata["binary_label_column"])
            if label_column not in frame.columns or "DMS_score" not in frame.columns or "mutant" not in frame.columns:
                raise ValueError(f"Selected assay {assay_id} missing mutant, DMS_score, or {label_column}")
            if not metadata["numbering_verified"].lower() == "true":
                raise ValueError(f"Selected assay numbering is not verified: {assay_id}")
            offset = 0 if pd.isna(metadata["numbering_offset"]) else int(float(metadata["numbering_offset"]))
            sequence = protein_map[assay_id]["sequence"]
            per_key: dict[tuple[str, int, str, str], list[tuple[int, float]]] = {}
            for row in frame.itertuples(index=False):
                parsed = parse_single(getattr(row, "mutant", None))
                if parsed is None:
                    rejected["non_single_or_unparsed"] += 1
                    continue
                wt, reported_position, mut = parsed
                actual_position = reported_position - offset
                label_value = pd.to_numeric(pd.Series([getattr(row, label_column, np.nan)]), errors="coerce").iloc[0]
                score = pd.to_numeric(pd.Series([getattr(row, "DMS_score", np.nan)]), errors="coerce").iloc[0]
                if pd.isna(label_value) or float(label_value) not in (0.0, 1.0) or pd.isna(score) or not math.isfinite(float(score)):
                    rejected["unlabeled_or_nonfinite_score"] += 1
                    continue
                if actual_position < 1 or actual_position > len(sequence) or sequence[actual_position - 1] != wt:
                    rejected["wt_mismatch_after_offset"] += 1
                    continue
                key = (assay_id, actual_position, wt, mut)
                project_label = function_project_label(int(label_value), label_1_is_tolerated)
                per_key.setdefault(key, []).append((project_label, float(score)))
            assay_rows = []
            for key, values in per_key.items():
                labels = {label for label, _ in values}
                if len(labels) != 1:
                    rejected["conflicting_duplicate_labels"] += 1
                    continue
                protein_id, position, wt, mut = key
                assay_rows.append({
                    "protein_id": protein_id, "position": position, "wt": wt, "mut": mut,
                    "label": next(iter(labels)), "score_raw": float(np.mean([score for _, score in values])),
                    "source": "proteingym", "split_id": None, "cluster_id": None,
                    "assay_id": assay_id, "phenotype": metadata["raw_DMS_phenotype_name"],
                })
            if not assay_rows:
                raise ValueError(f"No valid labeled single substitutions for {assay_id}")
            ranks = within_assay_percentile(pd.Series([row["score_raw"] for row in assay_rows]))
            for row, rank in zip(assay_rows, ranks):
                row["score_norm"] = float(rank)
                output.append(row)
    elapsed = time.perf_counter() - started
    result = pd.DataFrame(output)
    if result.duplicated(["protein_id", "position", "wt", "mut"]).any():
        raise ValueError("Function labels contain duplicate mutation keys across selected assays")
    return result, {"runtime_seconds": elapsed, "rejected": rejected,
                    "n_unique_mutations": len(result), "label_1_is_tolerated": label_1_is_tolerated,
                    "direction_evidence_status": direction["direction_evidence_status"]}


def build() -> dict[str, Path]:
    from xynapse.labels import stability_project_ddg
    from xynapse.schema import PROTEIN_COLS
    from xynapse.validate import validate_table

    started = time.perf_counter()
    proteins, protein_map, dropped_proteins = read_training_proteins()
    ddg, ddg_meta = build_stability_labels(protein_map)
    function, function_meta = build_function_labels(protein_map)
    clinvar = pd.read_parquet(ROOT / "data/interim/clinvar_external_candidates.parquet")
    if not set(clinvar["protein_id"]).issubset(set(proteins["protein_id"])):
        raise ValueError("ClinVar external proteins are missing from train_proteins")

    outputs = {
        "proteins": ROOT / "data/interim/train_proteins.parquet",
        "ddg": ROOT / "data/processed/labels_ddg.parquet",
        "function": ROOT / "data/processed/labels_function.parquet",
        "clinvar": ROOT / "data/processed/external_clinvar.parquet",
        "prevalence": ROOT / "reports/tables/label_prevalence.csv",
        "sign_check": ROOT / "reports/tables/sign_check_megascale_labels.csv",
        "summary": ROOT / "reports/tables/phase_03_label_build_summary.json",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    proteins = proteins[PROTEIN_COLS]
    clinvar = clinvar[["protein_id", "position", "wt", "mut", "label", "review_stars",
                       "clinvar_variation_id", "gene"]]
    proteins.to_parquet(outputs["proteins"], index=False)
    ddg.to_parquet(outputs["ddg"], index=False)
    function.to_parquet(outputs["function"], index=False)
    clinvar.to_parquet(outputs["clinvar"], index=False)

    tables = [("labels_ddg", ddg), ("labels_function", function), ("external_clinvar", clinvar)]
    validator_results = {name: validate_table(frame, name, proteins) for name, frame in tables}
    protein_validation = validate_table(proteins, "proteins")
    if protein_validation or any(validator_results.values()):
        raise ValueError(f"Contract validation failed: proteins={protein_validation}; labels={validator_results}")

    prevalence_rows = []
    ddg_positive = ddg["ddg"].ge(1.0)
    for name, frame, positive in [
        ("labels_ddg", ddg, ddg_positive),
        ("labels_function", function, function["label"].eq(1)),
        ("external_clinvar", clinvar, clinvar["label"].eq(1)),
    ]:
        prevalence_rows.append({"dataset": name, "n_rows": len(frame),
                                "n_proteins": frame["protein_id"].nunique(),
                                "positive_definition": "ddg >= 1.0 kcal/mol" if name == "labels_ddg" else "label == 1",
                                "n_positive": int(positive.sum()),
                                "prevalence": float(positive.mean()) if len(frame) else float("nan")})
    pd.DataFrame(prevalence_rows).to_csv(outputs["prevalence"], index=False)
    proline = ddg[ddg["mut"].eq("P")]
    proline_mean = float(proline["ddg"].mean()) if len(proline) else float("nan")
    pd.DataFrame([{"n_x_to_proline": len(proline), "mean_project_ddg": proline_mean}]).to_csv(outputs["sign_check"], index=False)
    if not math.isfinite(proline_mean) or proline_mean <= 0:
        raise ValueError(f"Written project-ddG proline sign check is not positive: {proline_mean}")
    for name, frame in (("labels_ddg", ddg), ("labels_function", function), ("external_clinvar", clinvar)):
        if frame.duplicated(["protein_id", "position", "wt", "mut"]).any():
            raise ValueError(f"Duplicate mutation keys in {name}")
        n_proteins = frame["protein_id"].nunique()
        if name == "external_clinvar":
            if len(frame) < 10 or n_proteins < 1:
                raise ValueError(f"ClinVar external table has too few variants or mapped proteins: {len(frame)} / {n_proteins}")
        elif n_proteins < 10:
            raise ValueError(f"Fewer than 10 training proteins in {name}: {n_proteins}")

    summary = {
        "runtime_seconds": time.perf_counter() - started,
        "n_train_proteins": len(proteins), "n_selected_stability_domains": len(pd.read_csv(ROOT / "reports/tables/proposed_training_ddg_domains.csv")),
        "n_selected_function_assays": len(pd.read_csv(ROOT / "reports/tables/proposed_training_function_assays.csv")),
        "n_labels_ddg": len(ddg), "n_ddg_proteins": int(ddg["protein_id"].nunique()),
        "n_labels_function": len(function), "n_function_proteins": int(function["protein_id"].nunique()),
        "n_external_clinvar": len(clinvar), "n_external_proteins": int(clinvar["protein_id"].nunique()),
        "external_protein_minimum_scope_note": "ClinVar is external-only and Phase 2 mapped only proteins present in the read-only teammate protein table; it is not subject to the 10-training-protein minimum.",
        "ddg_build": ddg_meta, "function_build": function_meta,
        "validator_results": {k: v for k, v in validator_results.items()},
        "protein_validator": protein_validation, "proline_sign_mean_project_ddg": proline_mean,
        "proline_sign_n": len(proline), "dropped_proteins": dropped_proteins,
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("Predictions are computational hypotheses, not experimental findings.")
    print(f"Training proteins: {len(proteins)}")
    print(f"Stability labels: {len(ddg)} rows / {ddg['protein_id'].nunique()} proteins")
    print(f"Function labels: {len(function)} rows / {function['protein_id'].nunique()} proteins")
    print(f"ClinVar external labels: {len(clinvar)} rows / {clinvar['protein_id'].nunique()} proteins")
    print(f"Written project-ddG X-to-P mean: {proline_mean:.9f} over {len(proline)} rows")
    print(f"Measured build runtime: {summary['runtime_seconds']:.2f} s")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Reserved for a future fully parameterized output root")
    args = parser.parse_args()
    if args.out is not None:
        parser.error("--out is not yet supported; outputs follow the configured Phase 3 contract paths")
    try:
        build()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
