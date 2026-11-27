# PTEN (P60484) Case Study

## Objective
Review PTEN mutation coverage, assay-label interpretation, functional-site overlap, and computational prediction evidence.

## Dataset
- Protein length: 403 amino acids.
- Assay rows: 12,343.
- Unique mutation combinations: 7,504.
- Expected single-substitution combinations: 7,657.
- Missing combinations: 153.
- No wild-type mismatches or invalid amino-acid entries were detected.

## Assays
- Matreyek 2021: 5,083 rows, protein stability/expression-related.
- Mighell 2018: 7,260 rows, lipid-phosphatase activity-related.

The project uses the provisional mapping `label_project = 1 - source_binary_label`. Individual assay label semantics remain unresolved, so performance results require cautious interpretation.

## Functional-site observation
One PTEN active-site position was annotated. Thirty-four assay rows mapped to this position, and all 34 were marked predicted damaging. This single-position result cannot be generalized to all PTEN active-site mutations.

## Limitations
- Assay-specific label direction needs further confirmation.
- 153 theoretical mutation combinations are absent.
- The active-site observation concerns one annotated position.
- Original frozen model binaries were unavailable for independent local inference.

## Conclusion
The PTEN analysis documents mutation coverage, assay-label uncertainty, and a limited active-site overlap observation. These are computational findings requiring biological interpretation and experimental validation.
