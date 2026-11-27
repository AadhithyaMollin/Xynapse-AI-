# BIO-1 Final Report — Xynapse AI

## Project scope
BIO-1 reviewed PTEN (P60484) and TP53 (P04637) mutation datasets, ClinVar data, functional-site annotations, mechanism evidence, model disagreements, and original ML pipeline documentation.

All predictions are computational hypotheses, not experimental findings or clinical diagnoses.

## 1. Mutation coverage
- PTEN length: 403 amino acids.
- TP53 length: 393 amino acids.
- PTEN: 12,343 assay rows; 7,504 unique mutation combinations.
- TPEN theoretical coverage: 7,657 combinations; 153 missing.
- TP53: 23,392 assay rows; 7,448 unique mutation combinations.
- TP53 theoretical coverage: 7,467 combinations; 19 missing at position 72.
- No wild-type mismatches or invalid amino-acid entries were detected in the audited tables.

## 2. PTEN label-direction review
Assays reviewed:
- Matreyek 2021: 5,083 rows.
- Mighell 2018: 7,260 rows.

The project uses the provisional mapping `label_project = 1 - source_binary_label`. Individual assay label semantics remain unresolved. The observed prediction alignment with source labels rather than their complement requires caution.

No model, labels, predictions, or thresholds were changed.

## 3. ClinVar review
The BIO-1 external set contains 552 unique VariationIDs:
- TP53: 286
- PTEN: 266

Original ML documentation reports a separate 646-variant ClinVar evaluation set, with AUROC 0.869244, AUPRC 0.947390, and MCC 0.335855.

The exact variant-level overlap between the BIO-1 552 variants and original 646 variants has not been verified. Do not claim that the sets are fully disjoint.

## 4. Functional-site overlap
PTEN:
- One annotated active-site position.
- 34 assay rows mapped to it; all 34 were predicted damaging.
- This is a single-position observation, not a general conclusion about PTEN active sites.

TP53:
- DNA-binding: 191 annotated positions; 11,935 rows; 1,137 predicted damaging (9.53%).
- Binding site: four annotated positions; 253 rows; 140 predicted damaging (55.34%).

These are computational overlaps, not experimental validation. Missing annotation entries do not prove biological absence.

## 5. Mechanism engine
The BIO-1 mechanism engine v1 output contains 552 rows and 92 columns. Six recorded tests passed.

Related files:
- BIO1_mechanism_engine_v1_summary.csv
- BIO1_mechanism_engine_v1_tests.csv
- BIO1_validation_tests.csv
- mechanism_evidence_rules.csv

## 6. Regional model disagreement
The analysis compared source labels and their inverse because individual assay label direction is unresolved. Counts represent assay rows, not unique variants. Disagreement should not automatically be interpreted as model error.

## 7. Original ML pipeline review
Original documentation records a successful Phase 10 PTEN pipeline run:
- 7,657 mutation rows.
- 403 amino-acid positions.
- Maximum absolute prediction difference versus Phase 09: 0.0 for ddG and function-impact outputs.

Frozen thresholds reviewed:
- ddG damaging: 0.676349007871
- ddG stabilizing: 0.021000149137
- Function impact high: 0.587478153430
- Function impact low: unavailable (`null`).

The original frozen `.joblib` model binaries were not found in the available Drive or GitHub files. Therefore, BIO-1 could not independently rerun inference. The original run is documented as successful.

## 8. Limitations
1. Individual PTEN assay label orientation remains unresolved.
2. Exact overlap between the 552 and 646 ClinVar variant sets is unverified.
3. Frozen model binaries were unavailable for an independent local rerun.
4. Functional-site overlap depends on supplied annotation coverage.
5. PTEN active-site overlap is based on one annotated position.
6. Regional disagreement counts represent assay rows.
7. Tier-2 models were unavailable. Stability training included sequences up to 72 amino acids; longer sequences require caution.
8. Predictions require experimental validation.

## Conclusion
BIO-1 completed mutation coverage audits, PTEN label-direction review, ClinVar review, functional-site overlap analysis, mechanism engine v1 validation, regional disagreement analysis, and review of original ML documentation. All conclusions must be interpreted with the limitations above.
