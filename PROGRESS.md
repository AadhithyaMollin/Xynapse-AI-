# PROGRESS.md

## Decisions (date, who, why)
- ddg sign convention: positive = destabilizing (kcal/mol)
- Data contract: Parquet tables keyed by protein_id + position (+ wt, mut); see src/xynapse/schema.py
- Position tables must include `wt` (agreed with BIO-1 / BIO-2)
- ESM-2 model for all phases: `facebook/esm2_t12_35M_UR50D` (short name `esm2_t12_35M`); do not use the 650M model.

## Data cards (source URL, version/date downloaded, license, columns used, filters, n rows)

## Daily log
### Day 1
- Phase 0 (audit): DONE. pytest 31 passed; env snapshot saved.
- Phase 1 Stage A: blocked on `pos_structure` missing `wt`.
- Phase 1 re-check: owner delivered `pos_structure .parquet` with `wt`; made a
  correctly named copy (teammate file untouched); validator PASS on all tables.
- Phase 1 Stage B: DONE. Scored P04637 with ESM-2 (7,467 rows, 37.65 s), ran
  physchem, built `mut_features.parquet` (15,124 rows, 2 proteins, 29 cols).
- Tests: 40 passed in 10.74 s.

### Phase 1 V2 continuation (blocked)
- Added publishing helpers, generic URL/ZIP-member fetch mode, Phase 1 gate, tests,
  Tier-1 schema contract, D9 config entries, and model/demo ignore policy.
- Helper tests: 6 passed; saved in `reports/gates/phase_01_helper_tests.txt`.
- Full gate blocked by system Python 3.9 without pandas and missing
  `../xynapse-shared`; see `reports/gates/phase_01_gate.txt` and `reports/AGENT_LOG.md`.
- No Phase 1 V2 push; resume only after project dependencies and the shared clone
  are available. Predictions are computational hypotheses, not experimental findings.
- Resume check: Python initially launched without dependencies and both supplied
  shared-clone locations were absent. After the owner authorized safe overrides, created
  `.venv`, installed `requirements.txt`, and cloned the configured origin at the required
  sibling path. Phase 1 gate now PASSes; baseline publication is in progress.
- Phase 1 V2: DONE and pushed to `ml/phase-01-baseline`; SHA and compare URL are
  recorded in `reports/STATE.json` and `reports/gates/phase_01_push_verification.txt`.
- Phase 2 STARTED. Owner explicitly waived CP1/CP2 waits; no data acquired yet.

## Measured runtimes (never estimated)
- pytest (Phase 1): 10.74 s, 40 passed.
- ESM-2 P04637 (393 aa, 25 batches, mps): 37.65 s.
- Phase 1 join script: see `reports/tables/nan_report_mut_features.csv` inputs.

## Gate status: G0 [x] G1 [x] G2 [x] G3 [x] G4 [ ] G5 [ ]

### Phase 2 — dataset acquisition and inspection (2026-11-26)
- Data cards: `docs/data_cards/megascale.md`, `docs/data_cards/proteingym.md`, `docs/data_cards/clinvar.md`; definitions: `docs/label_definitions.md`.
- Tsuboyama: 856 eligible domains; proposed 60. Sign check: pooled X→P project ddG +1.40110918 kcal/mol across 36,712 reliable substitutions. Sources: `reports/tables/megascale_inspection_summary.txt`, `reports/tables/proposed_training_ddg_domains.csv`.
- ProteinGym: 217 reference assays; 159 mechanically eligible; 6 case-study assays; 70 non-case, non-Stability sequences after assay selection; proposed 30. Direction check: 5 assays, pooled Spearman +0.12184444578563125, signs mixed (3 positive, 2 negative). Owner explicitly authorized continuing with the pooled-mean mapping; direction remains unresolved by assay. Sources: `reports/tables/proteingym_assay_summary.csv`, `reports/tables/proteingym_direction_precheck.csv`, `reports/tables/proteingym_direction_precheck_summary.json`.
- ClinVar: 9,356,209 rows scanned; 2,659 parsed candidates for PTEN/TP53 with zero WT mismatches; 646 variants retained. Source details in `reports/tables/clinvar_inspection_summary.json` and `docs/data_cards/clinvar.md`.
- Runtime: ClinVar download 94.10 s; Tsuboyama Dataset 1 member extraction 280.73 s; ProteinGym substitution archive download 32.73 s. Sources: source `.manifest.json` files.
- Phase 2 gate: PASS; pytest 61 passed in 10.69 s locally and 10.46 s on the shared-clone tree. Raw data remain local/ignored. Branch `ml/phase-02-datasets` pushed; code commit `c28cc4e65f9b42bec77abd52c8a13ff1995e8066`; compare URL is recorded in `reports/gates/phase_02_push_verification.txt`.

### Phase 3 — labels and training proteins
- STARTED. No model fitting or test-set evaluation is performed in this phase.
- `train_proteins`: 92 proteins; ddG labels: 52,801 rows/60 proteins; function labels: 76,777 rows/30 proteins; external ClinVar: 646 rows/2 proteins. The ClinVar 2-protein count follows the source-mapping restriction and is not a training-protein shortfall. Source: `reports/tables/phase_03_label_build_summary.json`.
- Duplicate reliable Tsuboyama observations: 92 rows collapsed into 52,801 mutation keys; written X→P sign check is +1.389019036 kcal/mol over 2,620 rows. Source: `reports/tables/phase_03_label_build_summary.json` and `reports/tables/sign_check_megascale_labels.csv`.
- Gate PASS; 66 tests passed in 11.67 s; build runtime 8.58 s. Source: `reports/gates/phase_03_gate.txt` and build summary. Branch `ml/phase-03-labels` pushed; commit SHA and compare URL are recorded in `reports/gates/phase_03_push_verification.txt`.

### Phase 4 — Tier-1 features
- DONE and pushed on `ml/phase-04-features`, commit `211718a2456f2d68a98ae4a5210c4614439acf32`; compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-04-features?expand=1.
- Built Tier-1 features for 92 proteins: 216,258 rows, 10 Tier-1 columns, no missing values; model `facebook/esm2_t12_35M_UR50D`, MPS runtime 195.40 s, 7 caches reused. Full function-label direction check covered 30 assays; mean Spearman was -0.284869493. The earlier mixed assay-level direction precheck remains a caveat.
- Phase 4 gate: all 11 checks PASS; 69 tests passed locally and in the shared clone. Source: `reports/gates/phase_04_gate.txt`, `reports/tables/phase_04_feature_summary.json`.

### Phase 5 — clustering and splits
- DONE locally. Installed the project-declared Homebrew `mmseqs2` package after discovering the executable was absent from PATH. MMseqs2 `20-ae60e` clustered 92 proteins into 79 clusters (identity 0.30, coverage 0.80); 14 clusters are test-only, including all case-study/ClinVar clusters. Command and hashes: `reports/tables/phase_05_split_summary.json`; details: `docs/data_cards/splits.md`.
- ddG: train 52 proteins / 45,411 rows; test 8 / 7,390. Function: train 26 / 71,962; test 4 / 4,815. Five training-cluster CV folds; no test evaluation. Source: `reports/tables/split_summary.csv`.
- Gate 16/16 PASS; pytest 76 passed in 8.93 s. Source: `reports/gates/phase_05_gate.txt`.
- Phase 5 pushed on `ml/phase-05-splits`, commit `2ad95cc954580a8b8ab0accb819be7ba10597bdd`; compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-05-splits?expand=1.

### Phase 6 — stability model
- DONE locally. Trained `esm2_t12_35M` Tier-1 LightGBM on 52 ddG proteins / 45,411 rows from 40 training clusters; 8 test clusters excluded. No Tier-2 model (only 2 teammate-table proteins). Max stability training length 72 aa. Source: `reports/tables/phase_06_stability_summary.json` and `models/esm2_t12_35M/stability_tier1.json`.
- 20 grouped-CV hyperparameter configs completed in 50.098926 s; bounded below 10 min. OOF covers every training row once. `tier1_full` train-only OOF: Spearman 0.480236099, Pearson 0.545773782, RMSE 0.823109547 kcal/mol. See `reports/tables/stability_oof_metrics.csv`; no test scores calculated.
- All four oriented baselines have positive mean fold Spearman. Per-protein Spearman qualified all 52 proteins at the 30-variant minimum. Source: `reports/tables/stability_baseline_sign_check.json`, `reports/tables/stability_per_protein_oof.csv`.
- Gate 16/16 PASS; pytest 83 passed in 9.53 s. Source: `reports/gates/phase_06_gate.txt`.
- Phase 6 pushed on `ml/phase-06-stability`, commit `5c553bb5eecfba4f96e1027b17d63e670be7488a`; compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-06-stability?expand=1.

### Phase 7 — function classifier
- DONE locally. Tier-1 LightGBM with isotonic calibration and a logistic-regression pipeline trained on 26 DMS proteins / 71,962 training rows; 4 test clusters excluded, max training length 465 aa. No Tier-2 model.
- Calibrated grouped OOF: AUROC 0.642152509, AUPRC 0.599066818, MCC 0.276047653 at 0.587478153, Brier 0.226514778. Isotonic selected over Platt (grouped cross-fit Brier 0.228731878 vs 0.228999995). Sources: `reports/tables/function_oof_metrics.csv`, `function_calibration_method_scores.csv`.
- Gate 16/16 PASS; pytest 91 passed in 8.81 s. OOF covers all training rows; no test scores. Source: `reports/gates/phase_07_gate.txt`.
- Timing caveat: original uncached 20-config search runtime was not persisted before the 660 s terminal timeout; do not confuse the 0.001756 s cached lookup with search execution. See Phase 7 entry in `reports/AGENT_LOG.md`. The optimization reduced MCC-threshold finalization from quadratic to sorted cumulative counts.
- ProteinGym assay-direction evidence remains mixed; function-label mapping is provisional.
- Phase 7 pushed on `ml/phase-07-function`, commit `3c3d85715976e544aff35e799732eeb1944bafdb`; compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-07-function?expand=1.

### Phase 8 — thresholds and freeze
- Freeze complete locally; no test evaluations yet. OOF-derived thresholds: ddG damaging 0.676349007871, ddG stabilizing 0.0210001491369, p_impact_high 0.58747815343, p_impact_low null. There were 9,354 damaging and 440 stabilizing positives in the 45,411 stability OOF rows. No calibrated function OOF threshold met NPV >=0.90, so the tolerated class cannot be assigned under the specified rule. Source: `reports/tables/threshold_evidence.csv`.
- `FROZEN.json` fingerprints five model artifacts; model is `facebook/esm2_t12_35M_UR50D`. Phase 8 gate 10/10 PASS; pytest 96 passed in 8.47 s; freeze step 0.098527 s. No test evaluation; registry empty. Source: `reports/gates/phase_08_gate.txt`, `reports/tables/phase_08_freeze_summary.json`.
- Freeze pushed as `ml/phase-08-freeze-test` commit `28e7506a841d953a6bea4891fe3fdd797d733883`; `models-frozen` tag pushed at the same SHA. Compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-08-freeze-test?expand=1.
- Owner waived CP2. Test evaluation is now authorized and will run once per named set via `evaluate_once`.
- Phase 8 one-time evaluations completed once each; no tuning/re-evaluation. Stability held-out: 7,390 rows / 8 proteins; Tier-1 full Spearman 0.452339, Pearson 0.557175, RMSE 0.858766 kcal/mol. Function held-out: 4,815 rows / 4 proteins; calibrated model AUROC 0.757075, AUPRC 0.654022, MCC 0.379725. ClinVar: 646 variants / 2 proteins; AUROC 0.869244, AUPRC 0.947390, MCC 0.335855. Sources: `reports/tables/test_stability_cluster.csv`, `test_function_cluster.csv`, `test_clinvar_external.csv` and `reports/test_eval_registry.json`.
- Case-study outputs are per assay: P60484 12,343 rows/2 assays; P04637 23,392 rows/4 assays. Function direction remains provisional; ClinVar covers only PTEN/TP53. Gate 13/13 PASS; `pytest -q`: 98 passed in 11.06 s. Source: `reports/gates/phase_08_gate.txt`.
- Phase 8 complete and follow-up pushed to `ml/phase-08-freeze-test`, SHA `ef20f6ee4d7721f2b9ab4bd5cb9a78ca3354493a`; compare URL above. `models-frozen` remains at original freeze commit `28e7506a841d953a6bea4891fe3fdd797d733883`. No tests were rerun after evaluation.
- Phase 9 gate PASS: 10/10 checks; 105 pytest tests passed in 9.35 s. Tier-1 frozen-model predictions saved locally for PTEN/P60484 and TP53/P04637: 15,124 rows total (19 substitutions per residue), 4,102 potentially damaging and 11,022 uncertain. No potentially beneficial/tolerated predictions because frozen `p_impact_low` is null. ddG confidence low for both (stability training max 72 aa); function confidence medium (function training max 465 aa). No Tier-2 model. Sources: `reports/gates/phase_09_gate.txt`, `reports/tables/rank_class_counts.csv`, `reports/tables/phase_09_prediction_summary.json`.
- Phase 9 DONE and pushed to `ml/phase-09-ranking`, SHA `69046043adb6f885ae78bc9d20519056927ea2fb`; compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-09-ranking?expand=1.
- Phase 10 gate PASS: 6/6; pytest 108 passed in 9.91 s. Fresh P60484 CLI pipeline generated 7,657 rows and matched Phase 9 ddg_pred and p_impact exactly (max absolute delta 0.0). ESM + Tier-1 stage 18.137 s; total staged time 18.913 s. Source: `reports/gates/phase_10_gate.txt`, `reports/tables/phase_10_pipeline_summary.json`.
- Phase 10 DONE and pushed to `ml/phase-10-pipeline`, SHA `792ebbe6638ee401dca8533597a34f51419b8fba`; compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-10-pipeline?expand=1.
- Phase 11 gate PASS 6/6; AppTest zero exceptions; pytest 113 passed in 10.04 s. P60484 heatmap/mutation table and CSV each rendered 7,657 rows (19 x 403); headless server started/stopped after a 4.362 s smoke run. AlphaFold 3D is opt-in and caches only from the approved AlphaFold host. Source: `reports/gates/phase_11_gate.txt`, `reports/tables/phase_11_dashboard_summary.json`.
- Phase 11 DONE and pushed to `ml/phase-11-dashboard`, SHA `666a47ca71852f9c729c474aaa2ae07309aad65b`; compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-11-dashboard?expand=1.
- Phase 12 STARTED: reports, audits, docs, and demo bundle.
- Phase 12 local gate: 51/51 checks PASS; pytest 119 passed in 13.26 s; leakage audit 16/16 PASS; claims audit clean. Sources: `reports/gates/phase_12_gate.txt`, `reports/gates/phase_12_pytest.txt`, `reports/tables/phase_12_summary.json`.
- Demo bundle: 15,124 predictions across PTEN/P60484 (7,657) and TP53/P04637 (7,467), includes `esm_llr`; Parquet 1,036,086 bytes. Two AlphaFold structures included under the individual 2 MiB cap, with file sizes and SHA-256 in `demo_data/manifest.json` and `docs/data_cards/alphafold_demo.md`.
- `scripts/make_reports.py` regenerated 10 report/model-card files byte-for-byte across two runs. Saved source-inspection and frozen test-evaluation artifacts remain as provenance; Phase 12 did not rerun test evaluations or fit models. Tier-2 model remains unavailable. Phase 12 DONE, pushed on `ml/phase-12-docs`, SHA `fc79ca5aabd403f459625e9229456b790b492b25`; compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-12-docs?expand=1. Shared-clone pytest: 117 passed, 2 skipped in 13.56 s. Predictions are computational hypotheses, not experimental findings.

### Phase 13 — final verification and release
- STARTED from Phase 12 tip. No model fitting or held-out test reevaluation.
- Integrated gate: 10/10 PASS; local suite 123 passed; all 13 known data/demo Parquet tables passed; leakage/claims audits passed; five frozen artifact hashes matched; P60484 pipeline max absolute prediction differences were 0; dashboard tests 5/5 passed. Source: `reports/gates/phase_13_gate.txt` and `reports/tables/phase_13_summary.json`.
- Shared-clone suite with staged Phase 13 files: 121 passed, 2 skipped. The post-push clean-clone suite against the Phase 13 tree also passed: 121 passed, 2 skipped. No held-out evaluation was run.
- Phase 13 DONE and pushed on `ml/phase-13-release`; final tagged commit SHA `ddb62390d50e2ebd79ccc4845ea9345ae1070f60` matches the remote branch and `v0.1` tag. Compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-13-release?expand=1. `gh` is unavailable; human review/merge remains manual.
- README managed block expanded with project scope, label semantics/caveats, data/model methods, saved metrics, usage and limitations; other README content untouched. Claims audit PASS; local pytest 123 passed; shared pytest 121 passed, 2 skipped. See `reports/AGENT_LOG.md`.
- Built the exact three-file teammate handoff: `stability_tier1.joblib`, `function_tier1.joblib`, `function_logistic_tier1.joblib` under `models/esm2_t12_35M_handoff/team_handoff/` and as `models/esm2_t12_35M_handoff/xynapse_ai_team_model_files_esm2_t12_35M.zip`. The third file is a Platt calibrator fit from grouped OOF data; production uses selected isotonic calibration embedded in `function_tier1.joblib`. The handoff files and ZIP are now allowlisted for publication and tracked on `main`; archive SHA-256 and per-file hashes are in `reports/model_artifact_bundle.json`. No frozen binary was modified.
