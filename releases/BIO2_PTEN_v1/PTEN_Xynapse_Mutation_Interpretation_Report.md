# Xynapse AI — PTEN Mutation Interpretation Report

**Protein:** PTEN
**UniProt accession:** P60484
**Protein length:** 403 amino acids
**Model:** ESM-2 12-layer, 35M parameters
**Report date:** 2026-11-28
**Dataset:** BIO-2 integrated mutation predictions

---

## 1. Executive Summary

Xynapse AI evaluated **7,657 single amino-acid substitutions** across the
403-residue PTEN protein.

The analysis combines Tier-1 ESM-2 predictions with BIO-2 structural,
evolutionary, domain, region and motif annotations.

| Metric | Result |
|---|---:|
| Total mutations evaluated | 7,657 |
| High functional-impact predictions | 1,389 |
| Predicted destabilizing mutations | 2,725 |
| Meeting both thresholds | 1,317 |

These are computational predictions, not experimentally established
functional effects or clinical classifications.

## 2. Prediction Methodology

### Stability prediction
The stability model predicts mutation-associated ΔΔG.

- Positive ΔΔG: destabilizing
- Negative ΔΔG: stabilizing
- Destabilizing threshold: 0.676349 kcal/mol

### Functional-impact prediction
The function model produces a calibrated impact probability.

- High-impact threshold: 0.587478

### BIO-2 evidence integration
The interpretation layer incorporates:

- Relative solvent accessibility
- pLDDT structural confidence
- Secondary structure
- Residue contact counts
- Evolutionary conservation
- Protein domains
- Functional regions
- Annotated motifs and sites

## 3. Predicted Mechanism Distribution

| Mechanism hypothesis | Mutation count |
|---|---:|
| Catalytic disruption | 19 |
| Folding/stability | 2,140 |
| Domain-associated impact | 2,602 |
| Interaction-associated impact | 83 |

These categories are evidence-based hypotheses. They are not independent
experimentally validated mechanism classifiers.

## 4. Biological Context Distribution

| Context | Mutation count |
|---|---:|
| phosphatase_domain | 3,268 |
| C2_domain | 2,850 |
| disordered_region | 931 |
| none | 342 |
| C2_domain;NOP53_interaction_region | 209 |
| disordered_region;PDZ_binding_motif | 57 |

## 5. Highest-Priority Mutation Candidates

The following table is an exploratory prioritization, not a clinical ranking.
Candidates are ordered by the combined presence of high functional-impact
probability and predicted destabilization, followed by a report-priority score.

| Position | Mutation | ΔΔG | Impact probability | Mechanism hypothesis |
|---:|---|---:|---:|---|
| 135 | I135P | 3.910 | 0.829 | folding_stability;domain_associated_impact |
| 135 | I135D | 3.919 | 0.817 | folding_stability;domain_associated_impact |
| 135 | I135E | 3.918 | 0.817 | folding_stability;domain_associated_impact |
| 122 | I122K | 3.864 | 0.817 | folding_stability;domain_associated_impact |
| 122 | I122D | 3.857 | 0.817 | folding_stability;domain_associated_impact |
| 122 | I122E | 3.852 | 0.817 | folding_stability;domain_associated_impact |
| 122 | I122P | 3.844 | 0.817 | folding_stability;domain_associated_impact |
| 122 | I122R | 3.824 | 0.817 | folding_stability;domain_associated_impact |
| 101 | I101R | 3.765 | 0.817 | folding_stability;domain_associated_impact |
| 101 | I101K | 3.762 | 0.817 | folding_stability;domain_associated_impact |
| 101 | I101E | 3.759 | 0.817 | folding_stability;domain_associated_impact |
| 101 | I101D | 3.755 | 0.817 | folding_stability;domain_associated_impact |
| 67 | I67K | 3.696 | 0.829 | folding_stability;domain_associated_impact |
| 67 | I67E | 3.691 | 0.829 | folding_stability;domain_associated_impact |
| 67 | I67D | 3.680 | 0.829 | folding_stability;domain_associated_impact |
| 67 | I67P | 3.656 | 0.829 | folding_stability;domain_associated_impact |
| 67 | I67R | 3.607 | 0.829 | folding_stability;domain_associated_impact |
| 135 | I135K | 3.656 | 0.817 | folding_stability;domain_associated_impact |
| 104 | F104D | 3.654 | 0.817 | folding_stability;domain_associated_impact |
| 120 | A120D | 3.638 | 0.817 | folding_stability;domain_associated_impact |

## 6. Confidence and Evidence Limitations

| Confidence measure | Result |
|---|---|
| Stability prediction confidence | Low |
| Functional prediction confidence | Medium |
| Stability model training maximum | 72 residues |
| Functional model training maximum | 465 residues |
| PTEN sequence length | 403 residues |

The stability confidence is downgraded because PTEN exceeds the stability
model's maximum training-protein length.

Structural evidence also varies across PTEN. In particular, the C-terminal
disordered region and PDZ-binding motif have low pLDDT values. Structural
interpretations in these regions should therefore be treated cautiously.

The evolutionary alignment has limited effective sequence depth, and several
functional annotations are absent from the current source data. Absence of an
annotation does not establish absence of biological function.

## 7. Important Interpretation Notes

- Domain membership provides biological context, not proof of disruption.
- Catalytic hypotheses use the supplied active-site annotation.
- Interaction-associated hypotheses use supplied region and motif annotations.
- Allosteric hypotheses remain disabled because nearest-site distance evidence
  is unavailable.
- No binding/interface disruption is inferred without supporting annotations.
- Model probabilities are not clinical pathogenicity probabilities.
- Experimental validation is required to establish biological effects.

## 8. Conclusion

This PTEN case study demonstrates how Xynapse AI combines sequence-based
mutation predictions with structural and evolutionary context to produce
interpretable mutation-level hypotheses.

The results provide a computational starting point for selecting variants
for further structural analysis and experimental validation.

---

**Xynapse AI | Computational mutation-impact interpretation**

*All reported results are computational hypotheses, not experimental findings.*
