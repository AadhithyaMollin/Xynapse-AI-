# Label definitions

Predictions are computational hypotheses, not experimental findings.

This document is based on Phase 2 source files and saved inspection outputs; no model has been fitted when this definition is written.

## Stability target

The source is the Mega-scale Dataset 1 CSV, with reliable wild-type references drawn from matching names in `Single_DMS_list.csv`. The bioRxiv v1 full text retrieved on 2026-11-26 at `https://www.biorxiv.org/content/10.1101/2022.12.06.519132v1.full` states:

> “All ΔΔG measurements are in the original wild-type background; positive ΔΔG indicates stabilizing substitutions.”

It also states:

> “ΔΔG indicates the stability of the PROSS design (ΔG) minus the stability of the original wild-type sequence; positive ΔΔG indicates the design stabilized the domain.”

The source convention has positive values for stabilizing changes. The project convention is positive = destabilizing:

`ddg_project = deltaG_wt - deltaG_mut`

`deltaG_wt` is the median `deltaG` over reliable WT-reference rows for a domain. A WT row and a mutation row are reliable when `deltaG_95CI <= 0.5` and `-1.0 <= deltaG <= 5.0`. A mutation is eligible only when its WT reference is reliable and its numbering is verified against `Single_DMS_list.csv` at 99% or better, exactly or after a constant offset. Domains require at least 100 reliable single substitutions for the proposed sample. The source text quotes and retrieval evidence are in `docs/data_cards/megascale.md` and `reports/tables/megascale_quote_verification.txt`.

## Function target

The source assay binary column is identified from each ProteinGym assay file; the five direction-precheck assays used `DMS_score_bin`. Exactly one assay per target sequence is selected, prioritizing the largest number of unique valid single substitutions and then alphabetical assay id. Assays with `coarse_selection_type` equal to `Stability` are excluded from the function training proposal. Raw scores remain assay-specific and are not pooled.

The five-assay ESM precheck has pooled mean Spearman(label, ESM LLR) `+0.12184444578563125`, with both positive and negative per-assay correlations. At the owner’s explicit request to continue, the provisional pooled-mean direction is used: source label 1 is treated as tolerated, and the project label is `label_project = 1 - source_binary_label`, where project label 1 means potentially damaging. Individual assay direction remains unresolved; the evidence table and status are `reports/tables/proteingym_direction_precheck.csv` and `reports/tables/proteingym_direction_precheck_summary.json`. This decision is an explicit deviation from the original stop-on-mixed-evidence instruction, not a claim that the direction is settled.

Each function label table row will retain the assay id, the exact `raw_DMS_phenotype_name` value from the reference, and its raw score. `score_norm` is a within-assay percentile rank only; raw scores from distinct assays are never pooled. Mutation keys are deduplicated by `(protein_id, position, wt, mut)`.

## ClinVar labels

ClinVar labels are external-test labels only. They are not training labels and no ClinVar-derived feature enters a model. The selected variant mapping and filters are documented in `docs/data_cards/clinvar.md`; the project label is 1 for the explicitly selected pathogenic classes and 0 for the explicitly selected benign classes. Variants with other significance strings, fewer than one review star, unresolved amino-acid changes, or WT/position mismatches are excluded.
