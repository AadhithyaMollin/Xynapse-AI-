# PROGRESS.md

## Decisions (date, who, why)
- ddg sign convention: positive = destabilizing (kcal/mol)
- Data contract: Parquet tables keyed by protein_id + position (+ wt, mut); see src/xynapse/schema.py
- Position tables must include `wt` (agreed with BIO-1 / BIO-2)
- ESM-2 dev model: esm2_t12_35M ; final model: esm2_t33_650M (verify memory)

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

## Gate status: G0 [x] G1 [x] G2 [x] G3 [ ] G4 [ ] G5 [ ]

### Phase 2 — dataset acquisition and inspection (2026-11-26)
- Data cards: `docs/data_cards/megascale.md`, `docs/data_cards/proteingym.md`, `docs/data_cards/clinvar.md`; definitions: `docs/label_definitions.md`.
- Tsuboyama: 856 eligible domains; proposed 60. Sign check: pooled X→P project ddG +1.40110918 kcal/mol across 36,712 reliable substitutions. Sources: `reports/tables/megascale_inspection_summary.txt`, `reports/tables/proposed_training_ddg_domains.csv`.
- ProteinGym: 217 reference assays; 159 mechanically eligible; 6 case-study assays; 70 non-case, non-Stability sequences after assay selection; proposed 30. Direction check: 5 assays, pooled Spearman +0.12184444578563125, signs mixed (3 positive, 2 negative). Owner explicitly authorized continuing with the pooled-mean mapping; direction remains unresolved by assay. Sources: `reports/tables/proteingym_assay_summary.csv`, `reports/tables/proteingym_direction_precheck.csv`, `reports/tables/proteingym_direction_precheck_summary.json`.
- ClinVar: 9,356,209 rows scanned; 2,659 parsed candidates for PTEN/TP53 with zero WT mismatches; 646 variants retained. Source details in `reports/tables/clinvar_inspection_summary.json` and `docs/data_cards/clinvar.md`.
- Runtime: ClinVar download 94.10 s; Tsuboyama Dataset 1 member extraction 280.73 s; ProteinGym substitution archive download 32.73 s. Sources: source `.manifest.json` files.
- Phase 2 gate: PASS; pytest 61 passed in 10.69 s locally and 10.46 s on the shared-clone tree. Raw data remain local/ignored. Branch `ml/phase-02-datasets` pushed; code commit `c28cc4e65f9b42bec77abd52c8a13ff1995e8066`; compare URL is recorded in `reports/gates/phase_02_push_verification.txt`.
