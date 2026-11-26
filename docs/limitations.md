# Limitations

Predictions are computational hypotheses, not experimental findings.

## Scope and data

- The current dashboard prediction bundle covers only the two case-study proteins PTEN (P60484) and TP53 (P04637), with 15,124 single-substitution rows in total. Source: `reports/tables/phase_09_prediction_summary.json`.
- Stability training used 52 proteins and 45,411 rows; the maximum training sequence length is 72 aa. Function training used 26 proteins and 71,962 rows; the maximum training sequence length is 465 aa. Sources: `models/esm2_t12_35M/stability_tier1.json`, `models/esm2_t12_35M/function_tier1.json`, and `reports/tables/split_summary.csv`.
- No Tier-2 model is available. Current model predictions use Tier-1 features only. Teammate structure/function/evolution annotations may provide mechanism hints, but their completeness and sequence coverage constrain those hints.
- PTEN evolution annotations have low diversity: `cons_entropy` is zero at 360 of 403 positions, with `n_eff` values of 4 or 5 across the table. `dist_to_nearest_site` is null for all PTEN/TP53 function annotation rows. Source: `reports/tables/teammate_table_inspection.csv`.
- ddG confidence is low for both case-study proteins because both are longer than the 72-aa maximum stability-training protein. Their function confidence is medium because they do not exceed the 465-aa function-training maximum and no Tier-2 model is used. Source: `reports/gates/phase_09_gate.txt` and model metadata.
- The function label direction remains provisional: the initial ProteinGym assay direction pre-check had mixed signs. Do not interpret pooled DMS-derived labels as a universal assay phenotype. Source: `reports/tables/proteingym_direction_precheck_summary.json` and `models/esm2_t12_35M/function_tier1.json`.
- ClinVar results cover only PTEN and TP53; this is not a broad external gene panel. The one-time test set comprised 646 variants across those two proteins. Source: `reports/tables/test_clinvar_external.csv`.

## Model and structural limits

- ESM-2 35M supplies sequence-based scores; it is not an experimental measurement. Pre-training may have included sequences related to proteins used here; the pipeline does not use ClinVar labels as features.
- Stability and function labels aggregate sources with different measurement/selection phenotypes. This can introduce label noise and makes assay-level interpretations non-interchangeable.
- AlphaFold structure views, when available, are display-only predictions. The viewer colors the downloaded structure by an Xynapse per-residue prediction; it does not convert that value into structural confidence or experimental evidence.
- Mechanism flags are rule-based hints contingent on annotation availability. Missing annotations do not imply absence of a biological role. The allosteric hypothesis rule is disabled by default.
- Since the frozen function low-probability threshold is null, the current ranking policy cannot assign the `tolerated` or `potentially beneficial (hypothesis)` class; unclassified mutations remain `uncertain` unless the damaging rule fires. Source: `configs/config.yaml` and `reports/tables/rank_class_counts.csv`.

Predictions are computational hypotheses, not experimental findings.
