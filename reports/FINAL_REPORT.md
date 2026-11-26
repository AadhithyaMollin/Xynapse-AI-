# Xynapse AI — final project report

Predictions are computational hypotheses, not experimental findings.

## Release status

| Phase | Status | Branch | Recorded phase commit | Compare/PR link |
|---|---|---|---|---|
| 01 | DONE | `ml/phase-01-baseline` | `61b81f282b2e518d5ce93c76aa30fd93d6bce8dd` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-01-baseline?expand=1 |
| 02 | DONE | `ml/phase-02-datasets` | `c28cc4e65f9b42bec77abd52c8a13ff1995e8066` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-02-datasets?expand=1 |
| 03 | DONE | `ml/phase-03-labels` | `e361ac52259f789021f41d00f3ce996ab9786ad3` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-03-labels?expand=1 |
| 04 | DONE | `ml/phase-04-features` | `211718a2456f2d68a98ae4a5210c4614439acf32` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-04-features?expand=1 |
| 05 | DONE | `ml/phase-05-splits` | `2ad95cc954580a8b8ab0accb819be7ba10597bdd` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-05-splits?expand=1 |
| 06 | DONE | `ml/phase-06-stability` | `5c553bb5eecfba4f96e1027b17d63e670be7488a` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-06-stability?expand=1 |
| 07 | DONE | `ml/phase-07-function` | `3c3d85715976e544aff35e799732eeb1944bafdb` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-07-function?expand=1 |
| 08 | DONE | `ml/phase-08-freeze-test` | `ef20f6ee4d7721f2b9ab4bd5cb9a78ca3354493a` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-08-freeze-test?expand=1 |
| 09 | DONE | `ml/phase-09-ranking` | `69046043adb6f885ae78bc9d20519056927ea2fb` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-09-ranking?expand=1 |
| 10 | DONE | `ml/phase-10-pipeline` | `792ebbe6638ee401dca8533597a34f51419b8fba` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-10-pipeline?expand=1 |
| 11 | DONE | `ml/phase-11-dashboard` | `666a47ca71852f9c729c474aaa2ae07309aad65b` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-11-dashboard?expand=1 |
| 12 | DONE | `ml/phase-12-docs` | `fc79ca5aabd403f459625e9229456b790b492b25` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-12-docs?expand=1 |
| 13 | DONE | `ml/phase-13-release` | `ddb62390d50e2ebd79ccc4845ea9345ae1070f60` | https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-13-release?expand=1 |

Phase statuses, branch names, SHAs, and links are read from `reports/STATE.json`. A compare URL is supplied where a pull request was not created. The human owner merges PRs; this workflow did not merge them or push to `main`.
Release tag `v0.1` points to `ddb62390d50e2ebd79ccc4845ea9345ae1070f60`; `models-frozen` remains at `28e7506a841d953a6bea4891fe3fdd797d733883`. Remote tag/branch SHA evidence is recorded in `reports/release.json`.
## Deliverables

| Phase | Status | Deliverable paths |
|---|---|---|
| Phase 1 | DONE | `scripts/sync_to_shared.py`, `scripts/pre_push_check.py`, `scripts/zenodo_fetch.py`, `src/xynapse/schema.py`, `configs/config.yaml` |
| Phase 2 | DONE | `docs/data_cards/`, `docs/label_definitions.md`, `reports/tables/` dataset inspections and training proposals |
| Phase 3 | DONE | `src/xynapse/labels.py`, `scripts/build_labels.py`, contract label tables and prevalence/sign reports |
| Phase 4 | DONE | `scripts/build_features_tier1.py`, Tier-1 feature output and teammate request |
| Phase 5 | DONE | `src/xynapse/splits.py`, `scripts/build_splits.py`, split table and split reports |
| Phase 6 | DONE | `src/xynapse/metrics.py`, `scripts/train_stability.py`, stability model metadata and OOF/test reports |
| Phase 7 | DONE | `scripts/train_function.py`, function model/calibrator metadata and OOF/test reports |
| Phase 8 | DONE | `src/xynapse/freeze.py`, `models/esm2_t12_35M/FROZEN.json`, one-time evaluation registry and test reports |
| Phase 9 | DONE | `src/xynapse/mechanism.py`, `src/xynapse/ranking.py`, `scripts/predict_ranked_mutations.py`, predictions and ranking report |
| Phase 10 | DONE | `src/xynapse/pipeline.py`, `scripts/gates/gate_phase_10.py` |
| Phase 11 | DONE | `app/app.py`, dashboard tests and gate report |
| Phase 12 | DONE | `README.md` managed block, `docs/`, `demo_data/`, model cards and report/audit scripts |
| Phase 13 | DONE | `scripts/verify_all.py`, `scripts/gates/gate_phase_13.py`, `reports/FINAL_REPORT.md` |

## Training data and model inventory

- Stability model: Tier-1; 52 proteins, 45411 rows, 40 training clusters; longest training sequence 72 aa. Source: `reports/tables/phase_06_stability_summary.json`.
- Function model: Tier-1; 26 proteins, 71962 rows, 25 training clusters; longest training sequence 465 aa. Source: `reports/tables/phase_07_function_summary.json`.
- Both model metadata records report Tier-2 unavailable; frozen model inventory uses `facebook/esm2_t12_35M_UR50D` and short name `esm2_t12_35M`. Source: `models/esm2_t12_35M/FROZEN.json` and model metadata JSON files.
- Training inputs were selected from 60 stability domains and 30 function assays before split-level exclusions. Source: `reports/tables/phase_03_label_build_summary.json`.
- MMseqs2 split inventory: 92 proteins, 79 clusters, 14 test clusters. Source: `reports/tables/phase_05_split_summary.json`.

## Saved held-out and external results

These are the single recorded evaluations in `reports/test_eval_registry.json`; the final verification did not rerun them. They are reported as computational outputs and are not grounds for changing the frozen models.

| Evaluation | Rows / proteins | Selected model output | Source |
|---|---:|---|---|
| Stability cluster test | 7390 / 8 | Spearman 0.452339; Pearson 0.557175; RMSE 0.858766 kcal/mol | `reports/tables/phase_08_test_evaluation_summary.json` |
| Function cluster test | 4815 / 4 | AUROC 0.757075; AUPRC 0.654022; MCC 0.379725 | `reports/tables/phase_08_test_evaluation_summary.json` |
| PTEN/P60484 case-study assays | 12343 / 2 assays | Per-assay rows and metrics in saved evaluation summary | `reports/tables/phase_08_test_evaluation_summary.json` |
| TP53/P04637 case-study assays | 23392 / 4 assays | Per-assay rows and metrics in saved evaluation summary | `reports/tables/phase_08_test_evaluation_summary.json` |
| ClinVar external set | 646 / 2 proteins | AUROC 0.869244; AUPRC 0.94739; MCC 0.335855 | `reports/tables/phase_08_test_evaluation_summary.json` |

## Frozen thresholds and plan parameters

OOF-derived frozen cutoffs are recorded in `reports/tables/threshold_evidence.csv` and `configs/config.yaml`: `ddg_damaging` = 0.6763490078708566 kcal/mol; `ddg_stabilizing` = 0.02100014913691953 kcal/mol; `p_impact_high` = 0.5874781534295241; `p_impact_low` = null because the target NPV was not reached. Accordingly, the tolerated class and beneficial-hypothesis class are unavailable under the configured rules.

| Parameter | Configured value | Units / meaning |
|---|---:|---|
| `ranking.ddg_label_cutoff` | `1.0` | kcal/mol |
| `mechanism.buried_rsasa` | `0.25` | fraction |
| `confidence.plddt_low` | `70` | score |
| `confidence.n_eff_low` | `5` | count |
| `reliability.ci_max` | `0.5` | kcal/mol |
| `reliability.dg_min` | `-1.0` | kcal/mol |
| `reliability.dg_max` | `5.0` | kcal/mol |
| `reliability.min_singles_per_domain` | `100` | substitutions |
| `sampling.stability_domains` | `60` | maximum domains |
| `sampling.function_assays` | `30` | maximum assays |
| `sampling.seed` | `42` | seed |
| `function.max_assay_length` | `500` | amino acids |
| `splits.identity` | `0.3` | MMseqs2 identity |
| `splits.coverage` | `0.8` | MMseqs2 coverage |
| `splits.test_fraction_of_clusters` | `0.15` | cluster fraction |

## Acquisition and label caveats

- ClinVar source: 9356209 rows scanned, 646 retained for the mapped case-study proteins; source archive size 458570768 bytes and SHA-256 `1d910abfb5b576d92abcf042e118dd7ae1c6cf9e91382931093ae5555816206a`. Gene mapping and star mapping are in `reports/tables/clinvar_inspection_summary.json` and `docs/data_cards/clinvar.md`.
- ProteinGym direction remains unresolved at the individual-assay level: the Phase 2 evidence has mixed assay signs. The owner-authorized pooled orientation was used provisionally; see `reports/tables/proteingym_direction_precheck.csv`, `docs/data_cards/proteingym.md`, and `docs/label_definitions.md`.
- The ProteinGym TLS exception was limited to GET/HEAD requests to `marks.hms.harvard.edu`, without credentials; warnings and file hashes are documented in `docs/data_cards/proteingym.md`.
- Tsuboyama row-name classes not assigned to known wild-type/single-substitution forms remain excluded/unresolved; see `docs/data_cards/megascale.md` and the label-build summary.
- No Tier-2 model was trained. `msa_logodds` is unavailable; teammate annotation completeness and PTEN evolution-table degeneracy remain documented in `reports/teammate_request.md` and `docs/limitations.md`.

## Phase 13 verification

The final gate reports 10/10 PASS and 0 FAIL; the saved summary records 0 held-out evaluations during this phase. Details: `reports/gates/phase_13_gate.txt` and `reports/gates/phase_13_verify_all.txt`.
The gate checks full pytest, all known data/demo Parquet contracts, leakage and claims audits, prediction row counts, frozen hashes, pipeline reproduction, dashboard AppTest, pre-push policy, and a clean clone with no local data or model binaries. The pipeline reproduced P60484 predictions with maximum absolute differences of 0 for ddG and function probability; source: `reports/gates/phase_13_gate.txt`.

## Human review and release notes

- Review the provisional ProteinGym label direction before using function scores beyond the current computational prototype.
- Review limitations, source cards, model cards, threshold evidence, and the five one-time evaluation records; do not retune or rerun those named test sets.
- Review and merge the phase pull requests manually. No human-authored case-study prose or literature interpretation has been added; those remain for the owner/team.
- Large source files and model binaries are not in the repository; sharing instructions are in `docs/SHARING_ARTIFACTS.md`.
