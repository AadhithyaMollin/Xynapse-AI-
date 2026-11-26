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
- Phase 5 code/data-card publication is in progress on the branch from Phase 4.
