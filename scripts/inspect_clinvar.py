#!/usr/bin/env python3
"""Inspect ClinVar missense records for the project proteins; no model fitting."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
SOURCE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
USECOLS = ["VariationID", "GeneSymbol", "Assembly", "Type", "Name",
           "ClinicalSignificance", "ReviewStatus"]
AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
PROTEIN_CHANGE_RE = re.compile(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)")
SIGNIFICANCE_LABELS = {
    "Pathogenic": 1, "Likely pathogenic": 1, "Pathogenic/Likely pathogenic": 1,
    "Benign": 0, "Likely benign": 0, "Benign/Likely benign": 0,
}
REVIEW_STARS = {
    "criteria provided, single submitter": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "reviewed by expert panel": 3,
    "practice guideline": 4,
    "criteria provided, conflicting classifications": -1,
    "no assertion criteria provided": 0,
    "no classification for the single variant": 0,
    "no classification provided": 0,
    "no classifications from unflagged records": 0,
    "-": 0,
}


def parse_protein_change(value: object) -> tuple[str, int, str] | None:
    match = PROTEIN_CHANGE_RE.search(str(value))
    if not match:
        return None
    wt3, pos, mut3 = match.groups()
    wt, mut = AA3_TO_1.get(wt3.title()), AA3_TO_1.get(mut3.title())
    if wt is None or mut is None or wt == mut:
        return None
    return wt, int(pos), mut


def star_mapping(observed_statuses: set[str]) -> dict[str, int]:
    unknown = observed_statuses - REVIEW_STARS.keys()
    if unknown:
        raise ValueError(f"Unmapped observed ClinVar review statuses: {sorted(unknown)}")
    return {status: REVIEW_STARS[status] for status in sorted(observed_statuses)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_uniprot_records(proteins: pd.DataFrame, raw_dir: Path) -> tuple[dict[str, str], dict[str, dict]]:
    gene_to_protein: dict[str, str] = {}
    evidence = {}
    for row in proteins.itertuples(index=False):
        protein_id = str(row.protein_id)
        url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        if data.get("primaryAccession") != protein_id:
            raise ValueError(f"UniProt accession mismatch for requested {protein_id}")
        sequence = str(data.get("sequence", {}).get("value", "")).upper()
        if sequence != str(row.sequence).upper():
            raise ValueError(f"UniProt sequence does not exactly match proteins.parquet for {protein_id}")
        # The fetched response showed the gene symbol under genes[].geneName.value.
        symbols = sorted({gene["geneName"]["value"] for gene in data.get("genes", [])
                          if isinstance(gene, dict) and isinstance(gene.get("geneName"), dict)
                          and gene["geneName"].get("value")})
        if not symbols:
            raise ValueError(f"No gene-name value found in UniProt JSON for {protein_id}")
        for symbol in symbols:
            if symbol in gene_to_protein and gene_to_protein[symbol] != protein_id:
                raise ValueError(f"Gene symbol collision: {symbol}")
            gene_to_protein[symbol] = protein_id
        cache = raw_dir / f"uniprot_{protein_id}.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(response.content)
        evidence[protein_id] = {
            "url": url, "sha256": hashlib.sha256(response.content).hexdigest(),
            "size": len(response.content), "gene_symbols": symbols,
            "uniprot_id": data.get("uniProtkbId"), "sequence_length": len(sequence),
            "cache_path": str(cache.relative_to(ROOT)),
        }
    return gene_to_protein, evidence


def inspect(clinvar_path: Path, proteins_path: Path, out_dir: Path, raw_dir: Path) -> dict[str, Path]:
    proteins = pd.read_parquet(proteins_path)
    gene_to_protein, uniprot_evidence = fetch_uniprot_records(proteins, raw_dir)
    sequences = {str(row.protein_id): str(row.sequence).upper() for row in proteins.itertuples(index=False)}
    protein_to_gene = {protein_id: gene for gene, protein_id in gene_to_protein.items()}

    category_counts: dict[str, dict[str, int]] = {c: {} for c in ("Assembly", "Type", "ClinicalSignificance", "ReviewStatus")}
    candidate_frames = []
    total_rows = 0
    for chunk in pd.read_csv(clinvar_path, sep="\t", compression="gzip", usecols=USECOLS,
                             chunksize=250_000, low_memory=False, dtype={"GeneSymbol": str, "Name": str}):
        total_rows += len(chunk)
        for column in category_counts:
            counts = chunk[column].fillna("<NA>").astype(str).value_counts()
            target = category_counts[column]
            for value, count in counts.items():
                target[value] = target.get(value, 0) + int(count)
        selected = chunk[
            chunk["Assembly"].eq("GRCh38")
            & chunk["Type"].eq("single nucleotide variant")
            & chunk["GeneSymbol"].isin(gene_to_protein)
        ].copy()
        if len(selected):
            parsed = selected["Name"].map(parse_protein_change)
            selected = selected.loc[parsed.notna()].copy()
            selected["_change"] = parsed.loc[parsed.notna()]
            candidate_frames.append(selected)

    out_dir.mkdir(parents=True, exist_ok=True)
    categories_path = out_dir / "clinvar_category_values.csv"
    pd.DataFrame([
        {"column": column, "value": value, "count": count}
        for column, values in category_counts.items() for value, count in sorted(values.items())
    ]).to_csv(categories_path, index=False)

    observed_statuses = set(category_counts["ReviewStatus"])
    observed_statuses.discard("<NA>")
    mapping = star_mapping(observed_statuses)
    mapping_path = out_dir / "clinvar_review_status_mapping.csv"
    pd.DataFrame([{"review_status": status, "review_stars": mapping[status],
                   "included_by_minimum_star_filter": mapping[status] >= 1}
                  for status in sorted(mapping)]).to_csv(mapping_path, index=False)

    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(columns=USECOLS + ["_change"])
    mismatch_rows = []
    accepted = []
    for gene, protein_id in gene_to_protein.items():
        gene_rows = candidates[candidates["GeneSymbol"].eq(gene)]
        mismatches = 0
        for _, row in gene_rows.iterrows():
            wt, pos, mut = row["_change"]
            seq = sequences[protein_id]
            if pos < 1 or pos > len(seq) or seq[pos - 1] != wt:
                mismatches += 1
            else:
                accepted.append({
                    "protein_id": protein_id, "position": pos, "wt": wt, "mut": mut,
                    "gene": gene, "variation_id": int(row["VariationID"]),
                    "clinical_significance": str(row["ClinicalSignificance"]),
                    "review_status": str(row["ReviewStatus"]),
                })
        mismatch_rate = mismatches / len(gene_rows) if len(gene_rows) else 0.0
        mismatch_rows.append({"gene": gene, "protein_id": protein_id,
                              "n_parsed_candidates": len(gene_rows), "n_wt_mismatches": mismatches,
                              "wt_mismatch_rate": mismatch_rate,
                              "sequence_length": len(sequences[protein_id])})
        if mismatch_rate > 0.05:
            raise ValueError(f"ClinVar WT mismatch rate exceeds 5% for {gene}: {mismatch_rate:.6f}")
    mismatch_path = out_dir / "clinvar_wt_mismatch_rates.csv"
    pd.DataFrame(mismatch_rows).to_csv(mismatch_path, index=False)

    frame = pd.DataFrame(accepted)
    if frame.empty:
        raise ValueError("No ClinVar candidates mapped to project proteins after sequence checks")
    frame["label"] = frame["clinical_significance"].map(SIGNIFICANCE_LABELS)
    frame["review_stars"] = frame["review_status"].map(mapping)
    conflict_status = frame["review_stars"].eq(-1)
    frame = frame[~conflict_status & frame["review_stars"].ge(1) & frame["label"].notna()].copy()
    frame["label"] = frame["label"].astype("int8")
    key_cols = ["protein_id", "position", "wt", "mut"]
    duplicate_groups = frame.groupby(key_cols, dropna=False)["label"].nunique()
    conflict_keys = set(duplicate_groups[duplicate_groups > 1].index.tolist())
    if conflict_keys:
        frame["_key"] = list(zip(*(frame[c] for c in key_cols)))
        before_conflict = len(frame)
        frame = frame[~frame["_key"].isin(conflict_keys)].drop(columns="_key")
        conflict_rows = before_conflict - len(frame)
    else:
        conflict_rows = 0
    if not frame.empty:
        frame = frame.sort_values(key_cols + ["review_stars", "variation_id"], ascending=[True, True, True, True, False, True])
        frame = frame.drop_duplicates(key_cols, keep="first")
        frame = frame.rename(columns={"variation_id": "clinvar_variation_id"})
        final_cols = key_cols + ["label", "review_stars", "clinvar_variation_id", "gene"]
        frame = frame[final_cols].sort_values(key_cols).reset_index(drop=True)

    parquet_path = ROOT / "data/interim/clinvar_external_candidates.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    counts_path = out_dir / "clinvar_external_counts.csv"
    counts = frame.groupby(["gene", "label", "review_stars"], dropna=False).size().rename("count").reset_index()
    counts.to_csv(counts_path, index=False)
    summary_path = out_dir / "clinvar_inspection_summary.json"
    summary = {
        "source_url": SOURCE_URL, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_size_bytes": clinvar_path.stat().st_size, "source_sha256": sha256(clinvar_path),
        "total_rows_scanned": total_rows, "gene_to_protein": gene_to_protein,
        "uniprot_evidence": uniprot_evidence,
        "parsed_gene_candidates": len(candidates), "sequence_checked_rows": len(accepted) + sum(x["n_wt_mismatches"] for x in mismatch_rows),
        "retained_after_significance_review_filters_and_dedupe": len(frame),
        "conflicting_label_rows_dropped": conflict_rows,
        "star_mapping": mapping,
        "significance_mapping": SIGNIFICANCE_LABELS,
        "assembly_filter": "GRCh38", "type_filter": "single nucleotide variant",
        "minimum_review_stars": 1,
        "duplicate_rule": "drop conflicting labels; for consistent duplicates retain the highest review star then smallest VariationID",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(DISCLAIMER)
    print(f"ClinVar rows scanned: {total_rows}")
    print(f"Mapped genes: {len(gene_to_protein)}; parsed candidates: {len(candidates)}")
    print(f"Retained variants after filters/deduplication: {len(frame)}")
    print(f"Conflicting-label rows dropped: {conflict_rows}")
    print(f"Summary: {summary_path}")
    return {"categories": categories_path, "mapping": mapping_path, "mismatch": mismatch_path,
            "counts": counts_path, "summary": summary_path, "parquet": parquet_path}


def main() -> int:
    try:
        inspect(ROOT / "data/raw/clinvar/variant_summary.txt.gz",
                ROOT / "data/processed/proteins.parquet", ROOT / "reports/tables",
                ROOT / "data/raw/clinvar")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
