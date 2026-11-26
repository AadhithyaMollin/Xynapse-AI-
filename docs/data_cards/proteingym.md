# ProteinGym substitution data card

Predictions are computational hypotheses, not experimental findings.

## Source and retrieval

Retrieved 2026-11-26. The download page was fetched from `https://proteingym.org/download`; its JavaScript bundle was fetched from `https://proteingym.org/assets/index-e72aeca0.js`. The bundle was inspected to obtain the ProteinGym v1.3 base URL and filenames; the size inventory is saved in `reports/tables/proteingym_download_inventory.csv`.

Files used:

- Reference: `DMS_substitutions.csv`, source URL `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv`, 208,734 bytes, SHA-256 `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308`.
- Assay bundle: `DMS_ProteinGym_substitutions.zip`, source URL `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip`, 43,021,128 bytes, SHA-256 `3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921`.

The reference has 217 assay rows. The inspector relies on the observed reference columns `DMS_id`, `DMS_filename`, `UniProt_ID`, `target_seq`, `seq_len`, `selection_assay`, `selection_type`, `coarse_selection_type`, `raw_DMS_phenotype_name`, `raw_DMS_directionality`, `raw_DMS_mutant_column`, `DMS_binarization_cutoff`, `DMS_binarization_method`, and `raw_mut_offset`. Assay files were read for the `mutant` field and a detected binary field; the five direction-precheck files used `DMS_score_bin`. The tracked summaries retain target sequence length and SHA-256 rather than raw sequences; this avoids secret-scanner false positives on amino-acid strings while preserving sequence identity evidence.

## TLS note

TLS certificate verification fails for `marks.hms.harvard.edu` in this environment. The pre-approved exception was applied only to GET/HEAD requests to that exact host, with no credentials. A warning was printed, and the SHA-256 of each downloaded file is recorded in its manifest. No other host used disabled TLS verification.

## Filters, counts, and selection

The reference contains 217 assays. The current audit marks 159 assays eligible by length at most 500, availability of a binary label, and verified numbering. Numbering is verified at 99% or greater either exactly or after a single constant offset; only aligned single substitutions with binary labels are counted. Noncanonical target sequences and assays with missing/ambiguous binary-label columns are rejected.

Six assays are set aside as PTEN/TP53 case-study data. Sequence identity against the project protein table and verified reference names were used; the UniProt JSON retrieved in this run maps `P04637` to `P53_HUMAN` and `P60484` to `PTEN_HUMAN`. The rows are listed in `reports/tables/case_study_assays.csv`.

Assays whose observed `coarse_selection_type` is `Stability` are excluded from the function-training pool so the function task does not include the separately modeled stability category. There are 94 valid non-Stability assays before case-study exclusion and sequence deduplication; the resulting non-case-study one-assay-per-sequence pool contains 70 candidates. A deterministic seed-42 sample of 30 is saved in `reports/tables/proposed_training_function_assays.csv`. The proposal retains the actual `raw_DMS_phenotype_name`, `selection_type`, and `coarse_selection_type` values per assay. These are candidate selections, not a fitted model.

## Label direction precheck

Five of the proposed non-case-study assays of length at most 300 were scored with the cached ESM-2 35M model. The saved `reports/tables/proteingym_direction_precheck.csv` records each assay, its raw phenotype name, aligned single-mutant count, Spearman correlation between its binary label and ESM LLR, and the proline fraction under the selected pooled orientation. The pooled mean Spearman is `+0.12184444578563125` (`reports/tables/proteingym_direction_precheck_summary.json`); three assay correlations are positive and two are negative. This is mixed evidence, and would trigger the original brief’s stop condition. At the owner’s explicit instruction to proceed, the implementation uses the pooled-mean sign as a provisional global convention: source label 1 is treated as tolerated and the project damaging label is `1 - source_binary_label`. This direction is **unresolved at the individual-assay level** and must not be read as a settled biological interpretation. No assay-specific direction was inferred from ESM.

For each selected assay, the phenotype and selection fields are those exactly present in the downloaded reference; raw scores are not pooled across assays. The audit contains the relied-on reference metadata, sequence length and digest, and rejection/numbering evidence in `reports/tables/proteingym_assay_summary.csv`.

## Resolved and unresolved checks

- Resolved: actual bundle base URL and candidate filenames were parsed from the freshly fetched bundle; candidate file sizes are recorded in the inventory.
- Resolved: 159 assays meet the mechanical length, label-column, and numbering criteria; six case-study assays are set aside; the proposal has 30 function assays from a 70-assay non-case-study pool after excluding Stability assays and deduplicating identical target sequences.
- Unresolved: individual ProteinGym assay correlations have mixed signs. The pooled mean is positive, but the assay signs disagree; the global label conversion remains provisional as explained above.
- Resolved: TLS exception scope is restricted to `marks.hms.harvard.edu`; file hashes are saved alongside the downloads.
