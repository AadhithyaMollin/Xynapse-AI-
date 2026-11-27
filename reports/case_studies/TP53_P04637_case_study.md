# TP53 (P04637) Case Study

## Objective
Review TP53 mutation coverage and overlap between computational predictions and supplied functional annotations.

## Dataset
- Protein length: 393 amino acids.
- Assay rows: 23,392.
- Unique mutation combinations: 7,448.
- Represented positions: 392.
- Position 72 was absent, corresponding to 19 missing theoretical substitutions.
- No wild-type mismatches were detected.

## Functional-site overlap
- DNA-binding annotations: 191 positions; 11,935 rows; 1,137 predicted damaging (9.53%).
- Binding-site annotations: four positions; 253 rows; 140 predicted damaging (55.34%).

These results describe overlap with supplied annotations. They do not experimentally validate predictions or establish clinical pathogenicity.

## Regional disagreement
Source labels and their inverse were compared because assay-specific label direction remains unresolved. Counts represent assay rows, not unique variants. They should not automatically be called model errors.

## Limitations
- Nineteen theoretical substitutions are missing at position 72.
- Conclusions depend on annotation coverage and definitions.
- Assay label interpretation requires caution.
- Predictions require experimental validation.

## Conclusion
The TP53 analysis documents near-complete position coverage and quantifies overlap with supplied DNA-binding and binding-site annotations. These findings form a computational case study, not an experimental or clinical conclusion.
