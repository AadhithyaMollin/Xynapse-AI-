#!/usr/bin/env python3
"""Inspect ProteinGym single-substitution assays and infer label direction from ESM-2."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xynapse.schema import AA

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
SINGLE_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
REFERENCE_FIELDS = [
    "DMS_id", "DMS_filename", "UniProt_ID", "target_seq", "seq_len",
    "selection_assay", "selection_type", "coarse_selection_type",
    "raw_DMS_phenotype_name", "raw_DMS_directionality", "raw_DMS_mutant_column",
    "DMS_binarization_cutoff", "DMS_binarization_method", "raw_mut_offset",
]
# UniProt JSON fetched during this run mapped P04637 to P53_HUMAN and P60484
# to PTEN_HUMAN; these reference identifiers catch assays with non-identical constructs.
CASE_REFERENCE_IDS = {"P04637": ("P04637", "P53_HUMAN"), "P60484": ("P60484", "PTEN_HUMAN")}


def parse_single_mutation(value: object) -> tuple[str, int, str] | None:
    match = SINGLE_RE.fullmatch(str(value))
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def assess_numbering(frame: pd.DataFrame, sequence: str, label_col: str | None) -> tuple[float, int | None, float, list[tuple[int, str, str, str, object]]]:
    records = []
    for row in frame.itertuples(index=False):
        parsed = parse_single_mutation(getattr(row, "mutant", None))
        if parsed is None:
            continue
        wt, position, mut = parsed
        label = getattr(row, label_col, np.nan) if label_col else np.nan
        records.append((position, wt, mut, getattr(row, "mutant"), label))
    if not records:
        return 0.0, None, 0.0, records
    exact_count = sum(1 for pos, wt, *_ in records if 1 <= pos <= len(sequence) and sequence[pos - 1] == wt)
    exact_rate = exact_count / len(records)
    offset_candidates = sorted({
        pos - (idx + 1)
        for pos, wt, *_ in records
        for idx, aa in enumerate(sequence)
        if aa == wt
    })
    best_offset, best_rate = None, 0.0
    for offset in offset_candidates:
        rate = sum(1 for pos, wt, *_ in records
                   if 1 <= pos - offset <= len(sequence) and sequence[pos - offset - 1] == wt) / len(records)
        if rate > best_rate:
            best_offset, best_rate = offset, rate
    if exact_rate >= 0.99:
        return exact_rate, 0, exact_rate, records
    return exact_rate, best_offset, best_rate, records


def find_binary_label_column(frame: pd.DataFrame) -> str | None:
    candidates = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
        if len(values) > 0 and set(values).issubset({0, 1}):
            candidates.append(column)
    if "DMS_score_bin" in candidates:
        return "DMS_score_bin"
    return candidates[0] if len(candidates) == 1 else None


def sanitized_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", value)


def _case_study_ids(proteins: pd.DataFrame) -> dict[str, str]:
    result = {}
    for pid in ("P60484", "P04637"):
        rows = proteins[proteins["protein_id"] == pid]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one verified case-study sequence for {pid}")
        result[pid] = str(rows.iloc[0]["sequence"]).upper()
    return result


def identify_case_study(source: dict, sequence: str, case_sequences: dict[str, str]) -> str | None:
    for protein_id, case_sequence in case_sequences.items():
        if sequence == case_sequence:
            return protein_id
    name_text = " ".join(str(source.get(field, "")) for field in ("DMS_id", "DMS_filename", "UniProt_ID"))
    for protein_id, identifiers in CASE_REFERENCE_IDS.items():
        if any(identifier in name_text for identifier in identifiers):
            return protein_id
    return None


def inspect(reference_path: Path, archive_path: Path, proteins_path: Path,
            out_dir: Path, precheck: bool = True, max_precheck: int = 5) -> dict[str, Path]:
    ref = pd.read_csv(reference_path, dtype={"DMS_id": str, "DMS_filename": str, "target_seq": str})
    missing = [field for field in REFERENCE_FIELDS if field not in ref.columns]
    if missing:
        raise ValueError(f"Reference is missing relied-on columns: {missing}")
    proteins = pd.read_parquet(proteins_path)
    case_sequences = _case_study_ids(proteins)
    assay_rows = []
    case_rows = []
    valid_assays = []
    with zipfile.ZipFile(archive_path) as archive:
        archive_members = set(archive.namelist())
        for source in ref.to_dict("records"):
            seq = str(source["target_seq"]).upper()
            seq_len = len(seq)
            seq_digest = hashlib.sha256(seq.encode("ascii", errors="ignore")).hexdigest()
            file_member = f"DMS_ProteinGym_substitutions/{source['DMS_filename']}"
            if seq_len > 500:
                record = {**{field: source[field] for field in REFERENCE_FIELDS},
                          "binary_label_column": None, "n_single_rows": 0, "n_valid_single_rows": 0, "n_unique_single_mutations": 0,
                          "numbering_match_rate": 0.0, "numbering_offset": None,
                          "offset_match_rate": 0.0, "numbering_verified": False,
                          "sequence_sha256": seq_digest, "case_study": False,
                          "rejection_reason": "length_gt_500"}
                assay_rows.append(record)
                continue
            if file_member not in archive_members:
                record = {**{field: source[field] for field in REFERENCE_FIELDS},
                          "binary_label_column": None, "n_single_rows": 0, "n_valid_single_rows": 0, "n_unique_single_mutations": 0,
                          "numbering_match_rate": 0.0, "numbering_offset": None,
                          "offset_match_rate": 0.0, "numbering_verified": False,
                          "sequence_sha256": seq_digest, "case_study": False,
                          "rejection_reason": "assay_file_missing"}
                assay_rows.append(record)
                continue
            assay = pd.read_csv(archive.open(file_member), low_memory=False)
            label_col = find_binary_label_column(assay)
            exact_rate, offset, offset_rate, records = assess_numbering(assay, seq, label_col)
            verified = exact_rate >= 0.99 or (offset is not None and offset_rate >= 0.99)
            selected_offset = offset or 0
            valid_records = [
                item for item in records
                if label_col is not None
                and pd.notna(item[4])
                and float(item[4]) in (0.0, 1.0)
                and 1 <= item[0] - selected_offset <= len(seq)
                and seq[item[0] - selected_offset - 1] == item[1]
            ]
            unique_mutations = {(wt, pos - selected_offset, mut) for pos, wt, mut, _, _ in valid_records}
            case_pid = identify_case_study(source, seq, case_sequences)
            reason = ""
            if set(seq) - set(AA):
                reason = "noncanonical_target_sequence"
            elif label_col is None:
                reason = "no_unique_binary_label_column"
            elif not verified:
                reason = "numbering_unverified"
            elif not unique_mutations:
                reason = "no_single_mutations"
            record = {**{field: source[field] for field in REFERENCE_FIELDS},
                      "binary_label_column": label_col,
                      "n_single_rows": len(records),
                      "n_valid_single_rows": len(valid_records),
                      "n_unique_single_mutations": len(unique_mutations),
                      "numbering_match_rate": exact_rate,
                      "numbering_offset": offset if offset not in (None, 0) else None,
                      "offset_match_rate": offset_rate,
                      "numbering_verified": verified,
                      "sequence_sha256": seq_digest,
                      "case_study": case_pid is not None,
                      "case_study_protein_id": case_pid,
                      "rejection_reason": reason}
            assay_rows.append(record)
            if case_pid is not None:
                case_rows.append(record)
            if not reason:
                valid_assays.append((source, assay, label_col, valid_records, selected_offset, seq_digest))

    assay_summary = pd.DataFrame(assay_rows).drop(columns=["target_seq"])
    assay_summary_path = out_dir / "proteingym_assay_summary.csv"
    case_path = out_dir / "case_study_assays.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    assay_summary.to_csv(assay_summary_path, index=False)
    pd.DataFrame(case_rows, columns=assay_summary.columns).to_csv(case_path, index=False)

    # Stability is modeled separately; only non-Stability assay classes are used
    # to define function labels. Keep the excluded assays in the full audit table.
    function_assays = [item for item in valid_assays
                       if str(item[0]["coarse_selection_type"]).strip().lower() != "stability"]
    candidates = {}
    for item in function_assays:
        source, assay, label_col, records, offset, seq_digest = item
        if source["DMS_id"] not in candidates:
            candidates[source["DMS_id"]] = item
    # One assay per exact target sequence; prefer the most unique valid single mutants,
    # then DMS_id alphabetically, following the fixed D4 rule.
    best_by_sequence = {}
    for item in function_assays:
        source, assay, label_col, records, offset, seq_digest = item
        prior = best_by_sequence.get(seq_digest)
        count = len({(wt, pos - offset, mut) for pos, wt, mut, _, _ in records})
        candidate_key = (-count, source["DMS_id"])
        if prior is None:
            best_by_sequence[seq_digest] = item
        else:
            prior_source, _, _, prior_records, prior_offset, _ = prior
            prior_count = len({(wt, pos - prior_offset, mut) for pos, wt, mut, _, _ in prior_records})
            if candidate_key < (-prior_count, prior_source["DMS_id"]):
                best_by_sequence[seq_digest] = item
    training_pool = [item for item in best_by_sequence.values() if not item[0]["DMS_id"] in assay_summary.loc[assay_summary["case_study"], "DMS_id"].tolist()]
    training_pool = [item for item in training_pool if len(str(item[0]["target_seq"])) <= 500]
    eligible_frame = pd.DataFrame([{
        "DMS_id": item[0]["DMS_id"], "DMS_filename": item[0]["DMS_filename"],
        "UniProt_ID": item[0]["UniProt_ID"], "target_seq": item[0]["target_seq"],
        "seq_len": len(str(item[0]["target_seq"])),
        "selection_assay": item[0]["selection_assay"], "selection_type": item[0]["selection_type"],
        "coarse_selection_type": item[0]["coarse_selection_type"],
        "raw_DMS_phenotype_name": item[0]["raw_DMS_phenotype_name"],
        "n_single_rows": len(item[3]), "n_valid_single_rows": len(item[3]),
        "n_unique_single_mutations": len({(w, p-item[4], m) for p,w,m,_,_ in item[3]}),
        "sequence_sha256": item[5],
    } for item in training_pool])
    if eligible_frame.empty:
        raise ValueError("No eligible non-case-study ProteinGym assays")
    selected = eligible_frame.sample(n=min(30, len(eligible_frame)), random_state=42).sort_values("DMS_id")
    proposed_path = out_dir / "proposed_training_function_assays.csv"
    selected.drop(columns=["target_seq"]).to_csv(proposed_path, index=False)

    precheck_path = out_dir / "proteingym_direction_precheck.csv"
    precheck_rows = []
    if precheck:
        from xynapse.esm_scoring import resolve_device, save_mut_esm, score_protein
        proposed_ids = set(selected.loc[selected["seq_len"] <= 300, "DMS_id"])
        selected_precheck = [item for item in best_by_sequence.values()
                             if len(str(item[0]["target_seq"])) <= 300
                             and item[0]["DMS_id"] in proposed_ids
                             and item[0]["DMS_id"] not in assay_summary.loc[assay_summary["case_study"], "DMS_id"].tolist()]
        selected_precheck = sorted(selected_precheck, key=lambda item: item[0]["DMS_id"])[:max_precheck]
        if not selected_precheck:
            raise ValueError("No non-case-study eligible assays of length <= 300 for label precheck")
        for source, assay, label_col, records, offset, seq_digest in selected_precheck:
            dms_id = str(source["DMS_id"])
            seq = str(source["target_seq"]).upper()
            safe_id = sanitized_id(dms_id)
            if len(safe_id) > 160:
                raise ValueError(f"Sanitized identifier too long: {dms_id}")
            cache_path = ROOT / "data/interim/esm" / f"{safe_id}_esm2_t12_35M_UR50D.parquet"
            if cache_path.exists():
                scored = pd.read_parquet(cache_path)
            else:
                scored = score_protein(dms_id, seq, MODEL_NAME, batch_size=16, device=resolve_device())
                save_mut_esm(scored, cache_path, MODEL_NAME)
            llr = scored.set_index(["wt", "position", "mut"])["esm_llr"]
            obs = []
            for pos, wt, mut, mutant_text, _ in records:
                actual_pos = pos - offset
                if actual_pos < 1 or actual_pos > len(seq):
                    continue
                if pd.isna(_):
                    continue
                key = (wt, actual_pos, mut)
                if key not in llr.index:
                    continue
                obs.append((float(_), float(llr.loc[key]), mut))
            if len(obs) < 3:
                raise ValueError(f"Too few aligned binary labels for ESM precheck: {dms_id}")
            labels = np.asarray([row[0] for row in obs], dtype=float)
            scores = np.asarray([row[1] for row in obs], dtype=float)
            rho = float(spearmanr(labels, scores).statistic)
            proline = [index for index, row in enumerate(obs) if row[2] == "P"]
            n_label1_proline = int(labels[proline].sum()) if proline else 0
            precheck_rows.append({
                "DMS_id": dms_id, "DMS_filename": source["DMS_filename"],
                "raw_DMS_phenotype_name": source["raw_DMS_phenotype_name"],
                "selection_type": source["selection_type"],
                "coarse_selection_type": source["coarse_selection_type"],
                "binary_label_column": label_col, "n_aligned_single_mutants": len(obs),
                "spearman_binary_label_esm_llr": rho,
                "n_proline_substitutions": len(proline),
                "n_proline_label_1": n_label1_proline,
                "assay_rho_positive": rho > 0,
            })
        precheck_df = pd.DataFrame(precheck_rows)
        rhos = precheck_df["spearman_binary_label_esm_llr"].dropna()
        mean_rho = float(rhos.mean()) if len(rhos) else float("nan")
        label_1_is_tolerated = bool(mean_rho > 0)
        precheck_df["label_1_is_tolerated_global"] = label_1_is_tolerated
        precheck_df["proline_labelled_damaging_fraction"] = precheck_df.apply(
            lambda row: ((row["n_proline_substitutions"] - row["n_proline_label_1"])
                         if label_1_is_tolerated else row["n_proline_label_1"])
            / row["n_proline_substitutions"] if row["n_proline_substitutions"] else float("nan"), axis=1)
        precheck_df.to_csv(precheck_path, index=False)
        signs_disagree = bool((rhos > 0).nunique() > 1) if len(rhos) else True
        direction_summary = {
            "n_assays": int(len(rhos)), "mean_spearman_binary_label_esm_llr": mean_rho,
            "mean_absolute_spearman": float(rhos.abs().mean()) if len(rhos) else float("nan"),
            "assay_signs_disagree": signs_disagree,
            "label_1_is_tolerated_by_pooled_mean": label_1_is_tolerated,
            "project_label_formula": "1 - source_binary_label" if label_1_is_tolerated else "source_binary_label",
            "direction_evidence_status": "mixed_assay_signs" if signs_disagree else "consistent_assay_signs",
            "policy_note": "Pooled-mean direction is used to continue at owner request; mixed assay signs remain an unresolved limitation.",
        }
        (out_dir / "proteingym_direction_precheck_summary.json").write_text(
            json.dumps(direction_summary, indent=2) + "\n")
        if signs_disagree:
            print("WARNING: Assay correlations have mixed signs; pooled-mean orientation is used only by owner-approved override.")
    else:
        pd.DataFrame().to_csv(precheck_path, index=False)

    print(DISCLAIMER)
    print(f"Assays in reference: {len(ref)}")
    print(f"Valid assays (<=500 aa, binary label, numbering verified): {len(valid_assays)}")
    print(f"Non-Stability function assay pool: {len(function_assays)}")
    print(f"Case-study assays: {len(case_rows)}")
    print(f"One-per-sequence non-case-study pool: {len(training_pool)}")
    print(f"Proposed function training assays: {len(selected)}")
    if precheck_rows:
        print(f"Direction precheck assays: {len(precheck_rows)}")
        print(f"Mean Spearman(label, ESM llr): {precheck_df['spearman_binary_label_esm_llr'].mean():.6f}")
        print(f"Label direction by pooled mean: {'1=tolerated; invert to project-damaging' if mean_rho > 0 else '1=damaging'}")
    return {"summary": assay_summary_path, "case_study": case_path,
            "proposed": proposed_path, "precheck": precheck_path,
            "direction_summary": out_dir / "proteingym_direction_precheck_summary.json"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "data/raw/proteingym/DMS_substitutions.csv")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/proteingym/DMS_ProteinGym_substitutions.zip")
    parser.add_argument("--proteins", type=Path, default=ROOT / "data/processed/proteins.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/tables")
    parser.add_argument("--no-precheck", action="store_true")
    parser.add_argument("--max-precheck", type=int, default=5)
    args = parser.parse_args()
    try:
        inspect(args.reference, args.archive, args.proteins, args.out,
                precheck=not args.no_precheck, max_precheck=args.max_precheck)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
