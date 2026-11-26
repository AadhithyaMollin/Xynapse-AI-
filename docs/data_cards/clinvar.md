# ClinVar external-test data card

Predictions are computational hypotheses, not experimental findings.

## Source and retrieval

- File: `variant_summary.txt.gz` from `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz`.
- Retrieved 2026-11-26; Content-Length was checked before download: 458,570,768 bytes. Downloaded size: 458,570,768 bytes. SHA-256: `1d910abfb5b576d92abcf042e118dd7ae1c6cf9e91382931093ae5555816206a`.
- Download manifest: `data/raw/clinvar/variant_summary.txt.gz.manifest.json`.
- All 9,356,209 rows were read in chunks for category census; category counts by the header-observed columns `Assembly`, `Type`, `ClinicalSignificance`, and `ReviewStatus` are in `reports/tables/clinvar_category_values.csv`.

## Protein and gene mapping

Only proteins already present in `data/processed/proteins.parquet` were considered. UniProt JSON responses were fetched and inspected during this run; their observed `genes[].geneName.value` fields yielded `TP53` for `P04637` and `PTEN` for `P60484`. Each fetched UniProt sequence exactly equals the corresponding project protein sequence. Response URLs, byte sizes, hashes, identifiers, and sequence lengths are recorded in `reports/tables/clinvar_inspection_summary.json`; raw responses are cached under `data/raw/clinvar/`.

The ClinVar `GeneSymbol` and `Name` columns were used to find candidate variants. Protein changes were parsed with the specified three-letter amino-acid form `(p.<WT><position><mutant>)`; noncanonical/stop/extension/frameshift/deletion-like changes and synonymous changes do not produce a parsed missense substitution. The parsed WT letter was checked against the 1-based position in the exact project protein sequence. The saved mismatch report contains 1,449 parsed TP53 candidates and 1,210 parsed PTEN candidates, with zero WT mismatches for either gene (`reports/tables/clinvar_wt_mismatch_rates.csv`).

## Filters and labels

The selected rows require `Assembly == GRCh38`, `Type == single nucleotide variant`, a parsed amino-acid substitution, one of the project gene symbols, one of the explicitly retained clinical-significance values, and at least one review star. Significance mapping is exact-string only:

- `Pathogenic`, `Likely pathogenic`, `Pathogenic/Likely pathogenic` → label 1.
- `Benign`, `Likely benign`, `Benign/Likely benign` → label 0.
- Other values are excluded.

The review-status values were collected from the fetched file before mapping. The complete observed mapping is in `reports/tables/clinvar_review_status_mapping.csv`: single submitter → 1; multiple submitters with no conflicts → 2; expert panel → 3; practice guideline → 4; no-assertion/no-classification statuses and `-` → 0; conflicting classifications → excluded (`-1`). No unobserved status was assigned a star value.

Duplicate mutation keys with conflicting labels are dropped. For same-label duplicates, the highest review-star record is kept, then the smallest observed `VariationID`; the output id column is `clinvar_variation_id`. The run dropped zero conflicting-label rows and retained 646 unique variants after filters and deduplication. Per-gene, label, and review-star counts are in `reports/tables/clinvar_external_counts.csv`; the filtered local candidate Parquet is `data/interim/clinvar_external_candidates.parquet`.

## Intended use and limitations

ClinVar is reserved as an external test source, not as a training label or feature. The current mapped genes are the two project proteins, PTEN and TP53. The saved output counts are data-processing counts, not model performance. No model was fitted or evaluated in this phase.
