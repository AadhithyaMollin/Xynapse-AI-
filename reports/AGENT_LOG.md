# AGENT_LOG.md

Append-only log. Newest entries at the bottom. Numbers here are read back from the
artifacts cited next to them.

---

## Phase 0 — Audit

**Status: DONE** (started and completed this session)

### Commands run

- `find src scripts tests configs -type f -not -path '*__pycache__*' -exec wc -l {} \;`
- `find data -type f -exec stat -f "%z %N" {} \;`
- `python -m pytest -q 2>&1 | tee reports/pytest_phase0.txt`
- `python scripts/env_snapshot.py` (new) -> `reports/env_snapshot.json`

### Inventory (measured)

- `configs/config.yaml` (26 lines)
- `scripts/`: `check_esm.py` (19), `run_esm.py` (130), `run_physchem.py` (35),
  `setup_check.py` (150), `validate_tables.py` (50)
- `src/xynapse/`: `__init__.py` (0), `esm_scoring.py` (126), `mutations.py` (21),
  `physchem.py` (53), `schema.py` (56), `validate.py` (64)
- `tests/`: `__init__.py` (0), `test_esm_scoring.py` (91), `test_mutations.py` (44),
  `test_physchem.py` (70), `test_schema.py` (18), `test_validate.py` (42)
- Non-source file present: `src/.DS_Store` (not touched; logged only)

### Data present (bytes)

- `data/processed/mut_esm.parquet` 130,052
- `data/processed/pos_evolution.parquet` 7,188
- `data/processed/pos_function.parquet` 9,566
- `data/processed/pos_structure.parquet` 15,657
- `data/processed/proteins.parquet` 7,325
- `data/interim/esm/P60484_esm2_t12_35M_UR50D.parquet` 130,052
- `data/interim/mut_phys.parquet` 21,984
- `data/raw/P60484.fasta` 575

### Phase-1 expectations

- `src/xynapse/join_features.py`: MISSING
- `scripts/run_join.py`: MISSING
- `data/processed/mut_features.parquet`: MISSING

### Gate checks

- PyTorch/transformers versions match the brief's assumptions (see snapshot).
- `pytest -q`: **PASS** — `31 passed, 2 warnings in 12.44s` (from `reports/pytest_phase0.txt`).
- `reports/AGENT_LOG.md` exists: **PASS**.
- ESM model short name used by existing artifacts: `esm2_t12_35M` (from filename).

### Environment (from reports/env_snapshot.json)

- Python 3.12.13, macOS-26.6-arm64, executable `/Users/maansibr/.pyenv/versions/3.12.13/bin/python`
- torch 2.12.0: `cuda_available=false`, `mps_available=true` -> device policy selects mps
- numpy 2.4.6, pandas 2.3.3, scikit-learn 1.8.0, lightgbm 4.7.0,
  transformers 5.8.1, pyarrow 24.0.0, biopython 1.88, biotite 2.0.0

### PHASE REPORT — Phase 0

- Files created: `AGENT_BRIEF.md`, `reports/env_snapshot.json`,
  `reports/pytest_phase0.txt`, `reports/env_snapshot_stdout.txt`,
  `scripts/env_snapshot.py`, `reports/AGENT_LOG.md`.
- Files changed: none.
- Tests: 31 passed, 0 failed (12.44 s).
- Gate checks: pytest PASS; log exists PASS.
- Deviations from brief: `src/.DS_Store` present (ignored, not modified).
- Open questions: none for Phase 0.

---

## Phase 1 — Teammate tables, inspection, join

**Status: BLOCKED at Stage A** (Stage B not started)

Predictions are computational hypotheses, not experimental findings.

### Commands run

- `python scripts/inspect_tables.py` (new) -> `reports/tables/teammate_table_inspection.csv`
  (39 rows), `reports/tables/teammate_value_checks.csv` (39 rows),
  stdout saved to `reports/tables/inspect_tables_stdout.txt`; exit code 1.
- `python scripts/validate_tables.py` -> `reports/tables/validate_tables_phase1.txt`; exit code 1.

### Measured table summary (read back from teammate_table_inspection.csv)

| table | rows | proteins | contract keys | duplicate key rows | wt present | wt mismatches | rows == L |
|---|---|---|---|---|---|---|---|
| mut_esm | 7657 | 1 (P60484) | protein_id, position, wt, mut | 0 | yes | 0 | n/a (mutation) |
| pos_evolution | 403 | 1 (P60484) | protein_id, position, wt | 0 | yes | 0 | yes |
| pos_function | 796 | 2 (P04637, P60484) | protein_id, position, wt | 0 | yes | 0 | yes |
| pos_structure | 403 | 1 (P60484) | protein_id, position, wt | n/a (no wt) | **no** | n/a | yes |
| proteins | 2 | 2 (P04637, P60484) | protein_id | 0 | n/a | n/a | n/a |

- `msa_logodds` present in: none.
- Columns with nulls: `pos_function.dist_to_nearest_site` only — 796/796 null (100%), dtype object.
- Extra columns: `proteins.protein_name` (not required by the contract).
- `mut_esm` schema metadata `esm_model` = `facebook/esm2_t12_35M_UR50D` (read with `read_esm_model`).
- Range checks that passed: rsasa 0..1.328, sasa_abs 0..234, n_contacts_8A 2..17,
  plddt 25.95..98.88, gap_frac 0..0.2, n_eff 4..5, all `is_*` binary 0/1.
- Sentinel scan (-1 / 999 / -999 / empty strings): 0 hits in every numeric and object column.
- Distribution extremes: `cons_entropy` is 0.0 at 360/403 positions (n_unique=3);
  `cons_weighted` is 1.0 at 360/403 positions (n_unique=3);
  `gap_frac` is 0 at 401/403; `n_eff` is 4 or 5 only.
- `ss8` counts: `-`=108, E=108, H=81, T=40, S=38, G=21, B=7.

### Blocking problems

1. `pos_structure.parquet` is missing contract column `wt` (POS_KEYS = protein_id, position, wt).
   `validate_tables.py` output: `[FAIL] pos_structure.parquet - missing columns: ['wt']`.
   `pos_structure` cannot be left-joined on POS_KEYS, so Stage B is not allowed to start.

### Non-blocking findings requiring owner confirmation

1. `pos_function.dist_to_nearest_site` is 100% null (796/796, object/None).
2. `pos_evolution` is degenerate: cons_entropy 0 at 360/403, cons_weighted 1.0 at 360/403,
   n_eff 4-5, gap_frac 0 except 0.2 at 2 positions.
3. `msa_logodds` absent from all tables (Stage B would add it as NaN).
4. `pos_function` contains P04637 (not in mut_esm); it would be dropped by the PTEN left join.

### PHASE REPORT — Phase 1 Stage A

- Files created: `scripts/inspect_tables.py`,
  `reports/tables/teammate_table_inspection.csv`,
  `reports/tables/teammate_value_checks.csv`,
  `reports/tables/inspect_tables_stdout.txt`,
  `reports/tables/validate_tables_phase1.txt`.
- Files changed: none. Teammate files read only.
- Tests: not re-run in this phase; Phase 0 run was 31 passed.
- Gate checks: validator PASS on mut_esm, pos_evolution, pos_function, proteins;
  FAIL on pos_structure (missing `wt`). Stage B gate not reached.
- Deviations from brief: none.
- Open questions: owner corrections needed for the three items above.

---

## Phase 1 — Stage A re-check and Stage B

**Status: DONE**

### Blocking issue resolved

- Teammate delivered `data/processed/pos_structure .parquet` (stray space in the
  name) with `wt`: 403 rows, 9 columns, 0 null `wt`, positions 1..403, P60484.
- The correctly named `pos_structure.parquet` was absent. Created an identical
  copy at `data/processed/pos_structure.parquet` with `cp -p`; the teammate file
  was not modified. `validate_tables.py` then PASSed on it.
- Re-ran `python scripts/inspect_tables.py`: 49 rows in each CSV; all contract
  tables `[OK]`; only remaining warning is the all-null
  `pos_function.dist_to_nearest_site`; `msa_logodds` absent everywhere.

### Commands run

- `python scripts/run_esm.py --uniprot P04637 --model dev`
- `python scripts/run_physchem.py`
- `python scripts/run_join.py`
- `python -m pytest -q`
- `python scripts/validate_tables.py`

### Measured results

- P04637 sequence fetched to `data/raw/P04637.fasta`: 393 aa, identical to the
  sequence in `proteins.parquet` (asserted in a script).
- ESM-2 scoring of P04637: 7,467 rows, **37.65 s measured** on mps; cache
  `data/interim/esm/P04637_esm2_t12_35M_UR50D.parquet`.
- `mut_esm.parquet`: 15,124 rows, 2 proteins (P60484 7,657 = 19x403;
  P04637 7,467 = 19x393).
- `mut_phys.parquet`: 15,124 rows.
- `mut_features.parquet`: 15,124 rows, 2 proteins, 29 columns; metadata
  `esm_model=facebook/esm2_t12_35M_UR50D`.
- NaN report: TP53 structure/evolution columns 7,467 missing (tables cover only
  P60484); `dist_to_nearest_site` and `msa_logodds` 100% missing; no imputation.

### PHASE REPORT — Phase 1

- Files created: `src/xynapse/join_features.py`, `scripts/run_join.py`,
  `tests/test_join_features.py`, `data/processed/mut_features.parquet`,
  `reports/tables/nan_report_mut_features.csv`,
  `data/processed/pos_structure.parquet` (copy),
  `data/raw/P04637.fasta`, `data/interim/esm/P04637_esm2_t12_35M_UR50D.parquet`.
- Files changed: `data/processed/mut_esm.parquet`,
  `data/interim/mut_phys.parquet`, inspection CSVs under `reports/tables/`,
  `reports/AGENT_LOG.md`.
- Tests: **40 passed, 2 warnings in 10.74 s**.
- Gate checks: validator PASS on `mut_features` and all contract tables; row
  counts 19 x L PASS; pytest PASS.
- Deviations from brief: teammate file had a stray space in its name; a
  correctly named copy was created rather than renaming the teammate file.
- Open questions: none blocking. `pos_structure` and `pos_evolution` still cover
  only P60484, so TP53 has NaN Tier-2 columns by design.

---

## Phase 1 V2 — tooling and baseline publication

**Phase 1 STARTED. Status: BLOCKED; no publication performed.**

Predictions are computational hypotheses, not experimental findings.

### Changes made

- Added `scripts/sync_to_shared.py` with allowlist filtering, SHA-256 manifest, README managed-block preservation, conflict refusal, and dry-run support.
- Added `scripts/pre_push_check.py` with staged-blob path, size, and secret checks plus README managed-block enforcement.
- Extended `scripts/zenodo_fetch.py` with generic HTTPS file/ZIP-member fetching, strict Content-Length/member-size checks, Range support, a 500 MB cap, and host-scoped TLS opt-out.
- Added `scripts/gates/gate_phase_01.py`, `tests/test_phase1_helpers.py`, and the Tier-1 schema/config additions.
- Added model-metadata/demo-data ignore exceptions and an explicit reports Parquet ignore rule. Added a managed README block containing the required disclaimer.
- Created `reports/STATE.json` because no state file existed at session start.

### Validation (read from saved artifacts)

- Targeted helper suite: **6 passed, 1 warning**; source `reports/gates/phase_01_helper_tests.txt`.
- Full Phase 1 gate: **FAIL**; source `reports/gates/phase_01_gate.txt`. Full pytest collection and table validation both stop on `ModuleNotFoundError: No module named 'pandas'`; the active interpreter is system Python 3.9, not the Python 3.12 environment recorded in the earlier log.
- Pre-push gate: **FAIL**; the required `../xynapse-shared` directory does not exist from this workspace.
- `python3 -m compileall -q scripts tests`: exited successfully; no output (no numeric claim).

### Stop conditions and deviations

- G0 was not run: the specified clone is absent. No alternate clone or Git path was used for publication.
- A required `git check-ignore -v` verification was attempted in this non-Git workspace and returned `fatal: not a git repository`; per R8, no further Git commands will be run in this session. The required verification still needs to be performed in the shared clone.
- No data acquisition, Phase 2 work, branch creation, commit, push, or PR was performed. Phase 1 is not DONE because the full test/validator gate and publishing procedure remain blocked.
- Open requirements: provide the expected Python 3.12/dependency environment and the shared clone at the specified sibling path, then resume Phase 1 gate and G0. No phase branch, commit SHA, or PR/compare URL exists.

### PHASE REPORT — Phase 1 V2 tooling

- Files created: `scripts/sync_to_shared.py`, `scripts/pre_push_check.py`, `scripts/gates/gate_phase_01.py`, `tests/test_phase1_helpers.py`, `reports/gates/phase_01_gate.txt`, `reports/gates/phase_01_helper_tests.txt`, `reports/STATE.json`.
- Files changed: `scripts/zenodo_fetch.py`, `src/xynapse/schema.py`, `configs/config.yaml`, `.gitignore`, `tests/test_schema.py`, `README.md`, `reports/AGENT_LOG.md`.
- Tests: helper tests PASS (6 passed, 1 warning); full suite FAIL (blocked during collection by missing pandas).
- Gate list: helper tests PASS; syntax compilation PASS; contract validator FAIL (missing pandas); full pytest FAIL (missing pandas); staged pre-push check FAIL (shared clone missing); G0 / ignore-proof / remote SHA checks NOT RUN.
- Measured quantities: only helper-suite count/runtime from `reports/gates/phase_01_helper_tests.txt`; no model or dataset measurements were made in this continuation.
- Git branch / commit SHA / PR URL: none. Phase branch not created or pushed.

### Resume check after user environment update

- Confirmed the newly installed Python interpreter launches, but its package list contains only pip and wheel; importing pandas fails with `ModuleNotFoundError: No module named 'pandas'`. No wheel cache is available for offline installation.
- Rechecked both locations supplied for the shared clone (`../xynapse-shared` from the mounted workspace and `~/Desktop/CSE/Xynapse/xynapse-shared`); neither exists. The specified working tree under `~/Desktop/CSE/Xynapse/xynapse-ai` is also absent; this editor workspace remains `xynapse-ai` under Developer.
- No packages were installed because the project’s allowed-host list does not include an approved package index, and the Git protocol does not permit cloning a replacement repository. Awaiting the human to provide the shared clone at the agreed path and explicitly authorize the package-download hosts (or provide an offline environment/wheels).

---

## Phase 1 V2 — resumed and unblocked

**Phase 1 STARTED (resumed); gate PASS, publication pending.**

The owner explicitly authorized continuing past the brief’s checkpoints and permitting safe setup actions. The implementation still follows the no-secrets, no-data/model-binaries-in-Git, and non-malicious constraints.

### Environment and G0

- Created `.venv` with Python 3.12 and installed the packages declared in `requirements.txt`; the installation completed. The Homebrew-managed interpreter was left unchanged.
- Created the missing sibling clone from the brief’s public origin. `reports/g0_inspection.txt` records the current clone state: remote `origin/main`, baseline commit `ec67e6ecd32cb2dedb75811fd597552d7f68a492`, and no remote `ml/scaffold-esm-physchem` branch. The only local starting branch was renamed to `ml/phase-01-baseline`, preserving its tip.
- Existing Git identity is configured. The required ignore proof is recorded in `reports/gates/phase_01_ignore_proof.txt`: model metadata JSON and demo artifacts are not ignored; reports-table Parquet remains ignored.

### Gate and push preparation

- `scripts/gates/gate_phase_01.py`: **PASS**; see `reports/gates/phase_01_gate.txt`.
- Project tests: **47 passed**; gate runtime **9.94 s**; source `reports/gates/phase_01_gate.txt`.
- Shared-clone tree tests: **47 passed**; source `reports/gates/phase_01_shared_pytest.txt`.
- Helper tests: **6 passed**; source `reports/gates/phase_01_helper_tests.txt`.
- Contract validator: PASS on the canonical contract tables; the pre-existing stray-space teammate file is reported as an extra file and remains untouched.
- Staged pre-push checks: PASS for allowlist/locations, README managed block, sizes and secrets; source `reports/gates/phase_01_pre_push.txt` (the final gate reran the same checks after staging).
- Files were synchronized by the H1 helper; `.sync_manifest.json` remains local and is not a tracked output.
- Open action: finish the final synchronization/staging review, commit and push `ml/phase-01-baseline`, verify the remote SHA, and open the PR. No data has been downloaded and Phase 2 has not started.

### PHASE REPORT — Phase 1 V2

Predictions are computational hypotheses, not experimental findings.

- Files created: `scripts/sync_to_shared.py`, `scripts/pre_push_check.py`, `scripts/gates/gate_phase_01.py`, `tests/test_phase1_helpers.py`, `reports/STATE.json`, `reports/g0_inspection.txt`, and Phase 1 gate/verification artifacts under `reports/gates/`.
- Files changed: `scripts/zenodo_fetch.py`, `src/xynapse/schema.py`, `configs/config.yaml`, `.gitignore`, `tests/test_schema.py`, `README.md`, `PROGRESS.md`, and this log.
- Tests: **47 passed** in the local gate (9.94 s), **47 passed** on the shared-clone tree (10.07 s), and **6 passed** in the helper suite (1.28 s). Sources: `reports/gates/phase_01_gate.txt`, `phase_01_shared_pytest.txt`, and `phase_01_helper_tests.txt`.
- Gate results: helper tests PASS; contract validator PASS; H2 pre-push checks PASS; model metadata/demo ignore proof PASS; remote branch SHA verification PASS.
- Staged review: `git status --short` and `git diff --cached --stat` were reviewed before commit; staged paths came from the H1 sync output and H2 accepted the complete staged set.
- Deviations: setup used the declared unpinned `requirements.txt` in an isolated `.venv`; no system packages were changed. GitHub CLI is not installed, so a compare/PR URL is recorded instead of opening a PR.
- Data/model measurements: none in this phase.
- Branch: `ml/phase-01-baseline`; commit SHA and local/remote comparison: `reports/gates/phase_01_push_verification.txt`; compare URL: same file.
- Status: Phase 1 DONE and pushed. Phase 2 may now start; the owner explicitly waived the brief’s CP checkpoints.

## Phase 2 — dataset acquisition and inspection

**Phase 02 STARTED (resumed); gate PASS, publication pending.**

- Owner instruction permits continuing past CP1/CP2 and explicitly overrides blocking on the mixed ProteinGym direction check. The evidence is retained as mixed/unresolved; no direction certainty is claimed.
- Tsuboyama Dataset 1 and `Single_DMS_list.csv` were range-extracted from Zenodo record `7992926`. The archive itself was not downloaded whole. Manifests record sizes, SHA-256 values, retrieval timestamps, and measured download runtimes. The inspector read only the five required Dataset 1 columns in chunks.
- Tsuboyama saved inspection: 1,841,285 rows; 983 listed domains; 979 numbering checks at 99% or better exactly/after a constant offset; 4 numbering failures; 858 domains have a reliable WT reference; 856 satisfy the sampling eligibility filters. The pooled reliable X→P project-ddG mean is +1.40110918 kcal/mol across 36,712 substitutions. Sources: `reports/tables/megascale_inspection_summary.txt`, `reports/tables/megascale_domain_summary.csv`, `reports/tables/sign_check_megascale.csv`.
- Exact bioRxiv v1 statements and retrieval evidence are saved in `reports/tables/megascale_quote_verification.txt` and `docs/data_cards/megascale.md`. Project target is `deltaG_wt - deltaG_mut`.
- ProteinGym: inspected the fetched page and bundle; the inventory is `reports/tables/proteingym_download_inventory.csv`. The reference contains 217 assays. The inspector marks 159 mechanically eligible assays, 6 case-study assays, and 94 non-Stability assays before case-study/deduplication. The non-case one-assay-per-sequence pool has 70 candidates; a deterministic sample of 30 is saved.
- ProteinGym label precheck: 5 assays, pooled mean Spearman(label, ESM LLR) +0.12184444578563125, 3 positive and 2 negative assay correlations. The pooled-mean provisional mapping is source label 1 tolerated, project label `1 - source_label`. This conflicts with the original mixed-sign stop rule; owner instruction authorized continuation. The direction remains unresolved by assay and is documented in `docs/data_cards/proteingym.md`, `docs/label_definitions.md`, and `reports/tables/proteingym_direction_precheck_summary.json`.
- The `marks.hms.harvard.edu` TLS exception was used only for GET/HEAD requests to that host. Download sizes and hashes are in the data manifests; the exception is documented in the ProteinGym data card.
- ClinVar: downloaded only after checking the 458,570,768-byte Content-Length; the source was scanned in chunks. 9,356,209 rows were inspected for category values. UniProt JSON identified TP53/P04637 and PTEN/P60484, with exact sequence matches. Among 2,659 parsed candidate variants, WT mismatch rate is 0 for both genes. 646 variants remain after GRCh38, single-nucleotide missense, exact significance, minimum-review-star, duplicate, and sequence filters; no conflicting-label rows were dropped. Sources: `reports/tables/clinvar_inspection_summary.json`, `reports/tables/clinvar_external_counts.csv`, `reports/tables/clinvar_wt_mismatch_rates.csv`.
- Proposed training lists: 60 eligible Tsuboyama domains and 30 non-case ProteinGym assays. Neither list has been used to fit a model.
- Phase 2 tests: **61 passed in 10.69 s** in the final gate run; source `reports/gates/phase_02_pytest.txt`. The gate output is `reports/gates/phase_02_gate.txt` and all listed checks PASS, including raw-data-not-staged.
- Phase 2 data cards created: `docs/data_cards/megascale.md`, `docs/data_cards/proteingym.md`, `docs/data_cards/clinvar.md`; label definitions: `docs/label_definitions.md`.
- Deviations/open limitations: ProteinGym assay-level label direction remains mixed; owner explicitly authorized the pooled-mean convention and continuation. `coarse_selection_type=Stability` assays are excluded from function training to avoid mixing the distinct stability target; this scope decision is documented. GitHub CLI is not installed; a compare URL will be used.
- H2 initially flagged an amino-acid sequence substring as an AWS-key pattern in the assay summary. The report files were changed to retain sequence length and SHA-256 instead of raw sequences; the pre-push secret scan then passed without weakening the check.
- Shared-clone verification before commit: H2 pre-push checks PASS for 43 staged paths; clean-tree tests PASS, 61 passed in 10.47 s. Exact `git status --short` and `git diff --cached --stat` output is saved in `reports/gates/phase_02_staged_review.txt` (43 files, 6,037 insertions, 3 deletions).
- Publication status at this log entry: branch `ml/phase-02-datasets` created from the Phase 1 branch; commit/push verification still pending.

### PHASE REPORT — Phase 2

Predictions are computational hypotheses, not experimental findings.

- Files created: Phase 2 data cards and `docs/label_definitions.md`; ClinVar/ProteinGym inspectors, deterministic proposal script, Phase 2 gate, and synthetic tests; saved source-audit, selection, direction, and gate tables under `reports/tables/` and `reports/gates/`.
- Files changed: `scripts/zenodo_fetch.py`, `PROGRESS.md`, `reports/STATE.json`, and this log.
- Tests: **61 passed** locally in 10.69 s and **61 passed** in the shared-clone tree in 10.46 s. Sources: `reports/gates/phase_02_pytest.txt` and shared-clone test output documented in this log.
- Gates: **17 PASS** checks in `reports/gates/phase_02_gate.txt`; staged pre-push check PASS for 44 paths; no raw data were staged; remote branch SHA matched local code commit.
- Measured results: Tsuboyama 856 eligible domains and 60 proposed; positive pooled X→P project ddG +1.40110918 kcal/mol across 36,712 reliable rows; ProteinGym 30 proposed assays, six case-study assays, and five direction-precheck assays with pooled mean Spearman +0.12184444578563125; ClinVar 9,356,209 rows scanned, two sequence-matched project genes, zero WT mismatches among 2,659 parsed candidates, and 646 filtered unique variants. Source files are listed in the Phase 2 data cards and above.
- Deviations/open question: ProteinGym assay correlations have mixed signs. Per explicit owner instruction the pooled-mean direction is provisionally used to continue; direction remains unresolved at individual-assay level. Stability-category ProteinGym assays were excluded from function training. GitHub CLI is unavailable, so no PR was opened.
- Branch: `ml/phase-02-datasets`; phase code commit SHA `c28cc4e65f9b42bec77abd52c8a13ff1995e8066`; remote SHA equality and compare URL are in `reports/gates/phase_02_push_verification.txt`. Compare URL: `https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-02-datasets?expand=1`.
- Status: Phase 2 DONE and pushed. No model has been fitted. Phase 3 begins next.

## Phase 3 — labels and training protein table

**Phase 03 STARTED.** The work uses the saved Phase 2 proposals and source-derived label conventions. Function labels retain the owner-authorized pooled direction and its unresolved mixed-assay evidence.

- First builder run wrote contract-valid tables, then stopped at a too-broad 10-protein check on `external_clinvar`. Phase 2 permits ClinVar mapping only to the two proteins in the read-only teammate table, and ClinVar is external-only. The check was corrected to apply the 10-protein minimum to the training label tables while requiring at least 10 external variants; the run then completed. This is an explicit scope exception, not an added or inferred protein mapping.
- Final build summary: 92 rows in `train_proteins` (60 selected stability domains, 30 selected ProteinGym target sequences, and the two case-study/external proteins); 52,801 ddG rows over 60 proteins; 76,777 function rows over 30 proteins; 646 ClinVar external rows over 2 proteins. Sources: `reports/tables/phase_03_label_build_summary.json` and `reports/tables/label_prevalence.csv`.
- Duplicate Tsuboyama observations: 52,893 reliable observations collapsed to 52,801 mutation keys; 92 repeated observations averaged. Zero selected proteins were dropped. No WT mismatches remained after numbering correction. Source: `reports/tables/phase_03_label_build_summary.json`.
- Written-table project-sign check: 2,620 X→P rows, mean ddG +1.389019036 kcal/mol. All three mutation tables and the protein table passed `validate_table`; no duplicate keys. Source: `reports/tables/sign_check_megascale_labels.csv` and the build summary.
- Function conversion uses `1 - source_binary_label` under the explicitly authorized pooled-mean orientation; the audit still records `mixed_assay_signs`. Raw `DMS_score` and within-assay percentile ranks are retained per assay; no cross-assay raw-score pooling is done.
- Phase 3 test/gate output: 66 passed in 11.67 s; all 14 checks PASS in `reports/gates/phase_03_gate.txt`. Build runtime: 8.58 s, from the saved build summary.
- Phase 3 changes currently await push; no model has been trained and no test set has been evaluated.

### PHASE REPORT — Phase 3

Predictions are computational hypotheses, not experimental findings.

- Files created: `src/xynapse/labels.py`, `scripts/build_labels.py`, `scripts/gates/gate_phase_03.py`, `tests/test_labels.py`, and saved Phase 3 summary/prevalence/sign/gate reports.
- Files changed: additive `external_clinvar` schema contract in `src/xynapse/schema.py`, `tests/test_schema.py`, `PROGRESS.md`, `reports/STATE.json`, and this log.
- Tests: **66 passed** locally in 11.67 s and **66 passed** in the shared clone in 12.22 s. Sources: `reports/gates/phase_03_pytest.txt` and shared-clone result documented in the push record.
- Gate: **14 PASS** checks in `reports/gates/phase_03_gate.txt`; H2 pre-push checks PASS for 14 staged paths; remote branch SHA matched local code commit.
- Measured outputs: 92 training/case-study proteins; ddG 52,801 rows over 60 training proteins; function 76,777 rows over 30 training proteins; ClinVar external 646 rows over 2 mapped proteins; 92 reliable ddG observations collapsed; project-sign proline mean +1.389019036 kcal/mol over 2,620 X→P labels; build runtime 8.58 s. Sources: `reports/tables/phase_03_label_build_summary.json`, `reports/tables/label_prevalence.csv`, `reports/tables/sign_check_megascale_labels.csv`.
- Deviations/open limitations: ClinVar remains limited to the two verified proteins in the teammate table and is exempted from the 10-training-protein threshold because it is external-only. ProteinGym direction remains mixed and provisional. No model fitting or test evaluation occurred.
- Branch: `ml/phase-03-labels`; phase code commit `e361ac52259f789021f41d00f3ce996ab9786ad3`; remote SHA equality and compare URL are recorded in `reports/gates/phase_03_push_verification.txt`. Compare URL: `https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-03-labels?expand=1`.
- Status: Phase 3 DONE and pushed. Phase 4 begins next.

## Phase 4 — Tier-1 features

**Phase 04 STARTED.** The build will use `facebook/esm2_t12_35M_UR50D` only, with resumable per-protein caches and a single preloaded model instance. Teammate position tables remain read-only; this phase only audits their availability and emits a BIO request.

**Phase 04 DONE — PHASE REPORT.**
- Files: `scripts/build_features_tier1.py`, `scripts/gates/gate_phase_04.py`, `src/xynapse/esm_scoring.py`, `src/xynapse/features.py`, `tests/test_features.py`; report artifacts under `reports/tables/`, `reports/gates/`, plus `reports/teammate_request.md` and progress/state updates.
- Validation: 69 tests passed (9.25 s on the shared-clone tree). Gate: 11/11 checks PASS. Pre-push check: all four rules PASS.
- Measured output: 216,258 rows for 92 proteins, 10 Tier-1 columns with zero missing values; ESM model `facebook/esm2_t12_35M_UR50D`; device MPS; runtime 195.40 s; 7 cached proteins reused. Full direction check: 30 assays, mean Spearman(project label, ESM LLR) = -0.284869493. Sources: `reports/tables/phase_04_feature_summary.json`, `reports/tables/label_direction_check.csv`, `reports/gates/phase_04_gate.txt`.
- Caveat: Phase 2's ProteinGym direction precheck had mixed assay signs. The authorized pooled-mean label orientation remains provisional; the Phase 4 mean check is negative but does not resolve assay-level heterogeneity.
- Git: branch `ml/phase-04-features`; code commit `211718a2456f2d68a98ae4a5210c4614439acf32`; remote `ls-remote` SHA matched. PR/compare: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-04-features?expand=1.
- Open issues: Tier-2 modeling depends on adequate sequence-matched teammate tables; no model has been trained and no test set has been evaluated.

## Phase 05 — clustering and splits

**Phase 05 STARTED.** Required MMseqs2 was not available on this shell's PATH, so I installed the existing project-declared Homebrew package `mmseqs2` and verified version `20-ae60e`.

**Phase 05 DONE — PHASE REPORT.**
- Files: `src/xynapse/splits.py`, `scripts/build_splits.py`, `scripts/gates/gate_phase_05.py`, `tests/test_splits.py`, `docs/data_cards/splits.md`; local outputs include the FASTA, MMseqs TSV, split assignment, updated labels, and training Parquets under ignored `data/`, plus split summaries and gate reports.
- MMseqs command and measured identifiers are recorded in `reports/tables/phase_05_split_summary.json`: version `20-ae60e`; identity 0.30; coverage 0.80; covariance mode 0; input 92 proteins; 79 clusters; 14 test clusters; command runtime 2.338534 s; full build runtime 2.640350 s. Input FASTA SHA-256 `6e367e57d0ea9f9fef286974bc3e1061d3ce36b18b6ab270ca4dad6b4e446b41`; TSV SHA-256 `0db3f187d226b66d5f6aa8ff6f63ef2c7a66b6309a440aaaa766d026a76c80de`.
- Measured task split counts from `reports/tables/split_summary.csv`: ddG train 52 proteins/45,411 rows and test 8 proteins/7,390 rows; function train 26 proteins/71,962 rows and test 4 proteins/4,815 rows. Five grouped validation folds were generated within training clusters.
- Validation: Phase 5 gate 16/16 checks PASS; pytest 76 passed in 8.93 s. Both label tables validate and retain unique mutation keys; no cluster crosses train/test or validation folds; held-out case-study/ClinVar IDs are not in training.
- Deviation: MMseqs was absent from PATH despite the prior environment snapshot; installed the project-declared `mmseqs2` Homebrew package. No label/test metrics were calculated and no held-out test set was evaluated.
- Git: branch `ml/phase-05-splits`, commit `2ad95cc954580a8b8ab0accb819be7ba10597bdd`; remote `ls-remote` SHA matched. Compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-05-splits?expand=1.
- Open caveat: Phase 2 ProteinGym assay direction signs were mixed and function labels remain provisional.

## Phase 06 — stability model

**Phase 06 STARTED.** Training is restricted to the Phase 5 train split and its five cluster-grouped CV folds. The held-out split is not used for fitting, tuning, thresholds, or reporting.

**Phase 06 DONE — PHASE REPORT.**
- Files: `src/xynapse/metrics.py`, `scripts/train_stability.py`, `scripts/gates/gate_phase_06.py`, `tests/test_metrics.py`, `tests/test_stability_training.py`; reports in `reports/tables/` and `reports/gates/`; tracked metadata `models/esm2_t12_35M/stability_tier1.json`. The model binary and OOF Parquet remain ignored/local.
- Training input: 52 proteins, 40 clusters, 45,411 rows; 8 held-out test clusters excluded. Tier-1 only; Tier-2 model absent because only two teammate-table proteins are available. Max stability training protein length: 72 aa. Source: `reports/tables/phase_06_stability_summary.json`, model metadata.
- Grouped OOF covered each training row once (45,411/45,411), with no fit/validation or fit/test cluster overlap. Twenty random configurations completed in 50.098926 s (inside the 10-minute cap); the cached rerun took 0.001184 s for search lookup and 5.986520 s for full output regeneration.
- OOF summary for `tier1_full` (training only): Spearman 0.480236099, Pearson 0.545773782, RMSE 0.823109547 kcal/mol. ESM-only: Spearman 0.475046?; the saved exact values are in `reports/tables/stability_oof_metrics.csv`. Per-protein Spearman qualified all 52 training proteins (minimum 30 variants). Baselines all had positive mean fold Spearman; details in `reports/tables/stability_baseline_cv.csv` and `stability_baseline_sign_check.json`.
- Validation: Phase 6 gate 16/16 checks PASS; pytest 83 passed in 9.53 s. No held-out test scores were computed.
- Git publication is pending the Phase 6 push procedure; no test evaluation.
- Correction to the rounded ESM-only OOF value above: exact saved `stability_oof_metrics.csv` Spearman is 0.4750470126350050 (the earlier `0.475046?` text was an incomplete draft value).
- Git: branch `ml/phase-06-stability`, commit `5c553bb5eecfba4f96e1027b17d63e670be7488a`; remote SHA matched. Compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-06-stability?expand=1.

## Phase 07 — function classifier

**Phase 07 STARTED.** Only DMS-derived labels from the Phase 5 training split are eligible. ClinVar and the held-out cluster split remain unused.

**Phase 07 DONE — PHASE REPORT.**
- Files: `src/xynapse/classification_metrics.py`, `scripts/train_function.py`, `scripts/gates/gate_phase_07.py`, `tests/test_classification_metrics.py`, `tests/test_function_training.py`; tracked metadata `models/esm2_t12_35M/function_tier1.json`; metrics/calibration/assay tables and gate outputs under `reports/`. OOF Parquet and both joblib model files remain ignored/local.
- Training-only input: 26 proteins, 25 clusters, 71,962 rows; prevalence 0.431477724; four test clusters excluded; max function-training sequence length 465. DMS labels only. ClinVar and held-out labels were not evaluated.
- Selected calibrated Tier-1 LightGBM OOF (training only): AUROC 0.642152509, AUPRC 0.599066818, MCC 0.276047653 at threshold 0.587478153; Brier 0.226514778. Calibration selection: isotonic grouped cross-fit Brier 0.228731878 vs Platt 0.228999995. Sources: `reports/tables/function_oof_metrics.csv`, `function_calibration_method_scores.csv`, and model metadata.
- OOF covers 71,962/71,962 training rows; 20 model fold fits had zero train/validation or train/test cluster overlap. Per-assay AUROC covers 26 training assays. Source: `reports/tables/function_fit_audit.csv` and `function_per_assay_auroc.csv`.
- Validation: Phase 7 gate 16/16 PASS; pytest 91 passed in 8.81 s. No held-out test results were produced.
- Runtime deviation: the first training invocation exceeded the terminal's 660 s limit after all 20 search configurations, OOF predictions, and calibration artifacts had been written, but before final metrics/export. Two further invocations hit the same slow MCC-threshold scan. Diagnosed quadratic threshold scoring; replaced it with a tested sorted cumulative-count implementation. Final resumable pass reused all 20 search configs and cached folds; finalization took 1.091266 s. The original uncached search runtime was not persisted, so its exact duration is unresolved; the saved 0.001756 s is cache lookup time, not the original search duration.
- Direction caveat remains: assay-level ProteinGym signs were mixed in the Phase 2 precheck; project label orientation is provisional despite the pooled/overall Phase 4 check.
- Git: branch `ml/phase-07-function`, commit `3c3d85715976e544aff35e799732eeb1944bafdb`; remote SHA matched. Compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-07-function?expand=1.
- Search runtime resolution: exact original duration remains unavailable; the run artifacts show 20 complete configs, and saved summary distinguishes active search from the 0.001756-second cache lookup.

## Phase 08 — thresholds, freeze, and one-time test

**Phase 08 STARTED.** Thresholds must be derived from saved OOF predictions. No test split, case-study assay, or ClinVar external set has been evaluated.

**Phase 08 FREEZE — PHASE REPORT (pre-test evaluation).**
- Files: `src/xynapse/thresholds.py`, `src/xynapse/freeze.py`, `scripts/derive_thresholds.py`, `scripts/gates/gate_phase_08.py`, `tests/test_freeze_thresholds.py`; additive threshold values in `configs/config.yaml`; `models/esm2_t12_35M/FROZEN.json`; `reports/tables/threshold_evidence.csv`, `phase_08_freeze_summary.json`, and gate output.
- OOF-derived config thresholds: `ddg_damaging=0.676349007871`; `ddg_stabilizing=0.0210001491369`; `p_impact_high=0.58747815343`; `p_impact_low=null`. Evidence: 9,354 ddG-damaging positives and 440 stabilizing positives among 45,411 stability OOF rows; function threshold evidence from 71,962 calibrated OOF rows. `p_impact_low` is null because no OOF cutoff met NPV >= 0.90. Thus the specified tolerated class cannot be assigned; no alternative threshold was substituted.
- Freeze: five model artifacts inventoried and SHA-256 checked; ESM model `facebook/esm2_t12_35M_UR50D`; threshold and feature inventories embedded. Freeze derivation runtime 0.098527 s, source `reports/tables/phase_08_freeze_summary.json`.
- Validation: Phase 8 gate 10/10 PASS; pytest 96 passed in 8.47 s. Test registry is absent/empty and no `test_*.csv` exists. No test evaluation performed.
- Git: branch `ml/phase-08-freeze-test`, frozen commit `28e7506a841d953a6bea4891fe3fdd797d733883`; remote SHA matched. `models-frozen` tag points to the same SHA and is pushed. Compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-08-freeze-test?expand=1.
- CP2 waived by owner. Evaluation authorization begins only after frozen branch and tag publication. Evaluate held-out, case-study DMS, and ClinVar exactly once through `evaluate_once`; no tuning/re-evaluation.

**Phase 08 — one-time test evaluation follow-up.** The five approved evaluation names were executed once through `evaluate_once`; no evaluation was repeated and no tuning followed. Frozen model SHA-256 inventory still matches disk.

**PHASE REPORT — Phase 08 final.**
- Files changed/created for this follow-up: `scripts/gates/gate_phase_08.py`; `reports/gates/phase_08_gate.txt`; `reports/gates/phase_08_pytest.txt`; `reports/test_eval_registry.json`; `reports/tables/phase_08_test_evaluation_summary.json`; and the 12 `reports/tables/test_*.csv` outputs (stability, function, P60484/P04637 case-study assays and disagreements, and ClinVar). Raw data and model binaries remain untracked.
- Tests: `pytest -q` — 98 passed in 11.06 s (captured in `reports/gates/phase_08_pytest.txt`). Gate: 13/13 PASS; checks include exact registry inventory/statuses, all saved outputs non-empty, frozen hashes, threshold evidence, model metadata, and tests. Gate output: `reports/gates/phase_08_gate.txt`.
- Held-out stability: 7,390 rows, 8 proteins, 8 clusters; Tier-1 full model Spearman 0.452339, Pearson 0.557175, RMSE 0.858766 kcal/mol. Source: `reports/tables/test_stability_cluster.csv`.
- Held-out function: 4,815 rows, 4 proteins, 4 clusters; calibrated Tier-1 model AUROC 0.757075, AUPRC 0.654022, MCC 0.379725. Source: `reports/tables/test_function_cluster.csv`.
- Case-study DMS evaluations: P60484 12,343 total assay rows across 2 assays; P04637 23,392 rows across 4 assays. Per-assay values are retained, not pooled, in `reports/tables/test_case_study_P60484.csv` and `reports/tables/test_case_study_P04637.csv`; the P04637 Giacomelli assays each excluded 19 WT-mismatch rows as recorded in the registry.
- ClinVar external test: 646 variants, 2 proteins; AUROC 0.869244, AUPRC 0.947390, MCC 0.335855. Counts: PTEN 257 pathogenic/likely pathogenic and 10 benign/likely benign; TP53 225 and 154 respectively. Sources: `reports/tables/test_clinvar_external.csv`, `reports/tables/test_clinvar_counts_by_gene_label.csv` and registry.
- Frozen thresholds: ddG damaging 0.676349007871; ddG stabilizing 0.0210001491369 kcal/mol; function high 0.58747815343; function low null because OOF had no cutoff reaching NPV 0.90. Thus tolerated classification remains unavailable and no replacement threshold was chosen. Source: `reports/tables/threshold_evidence.csv` and `configs/config.yaml`.
- Caveats: function direction remains provisional because initial assay-level direction evidence was mixed; ClinVar represents only two case-study proteins; stability training maximum is 72 aa (Phase 6 metadata), so large-protein predictions are extrapolations. No Tier-2 model exists.
- Gate failures: none. Open follow-up: proceed with Phase 9 using frozen models/thresholds; preserve the evaluation registry and never rerun these names.
- Shared-clone publication checks: pre-push rules 4/4 PASS; `python -m pytest -q ../xynapse-shared` — 98 passed in 8.93 s. Exact staged status: `M PROGRESS.md`; `M reports/AGENT_LOG.md`; `M reports/STATE.json`; `M reports/gates/phase_08_gate.txt`; `M reports/gates/phase_08_pytest.txt`; added `reports/tables/phase_08_test_evaluation_summary.json`; added three P04637 outputs; added three P60484 outputs; added two ClinVar outputs; added two function outputs; added two stability outputs; added `reports/test_eval_registry.json`; added `scripts/evaluate_frozen_tests.py`; modified `scripts/gates/gate_phase_08.py`; added `tests/test_frozen_evaluation.py` (22 staged paths total). `git diff --cached --stat`: 37,197 insertions and 20 deletions across 22 files, including 23,393 P04637 variants and 12,344 P60484 variants.
- Git: follow-up publication pushed to `ml/phase-08-freeze-test`; local and remote SHA match `ef20f6ee4d7721f2b9ab4bd5cb9a78ca3354493a`. `models-frozen` remains at the original freeze commit `28e7506a841d953a6bea4891fe3fdd797d733883`. Compare URL: https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-08-freeze-test?expand=1.
- Predictions are computational hypotheses, not experimental findings.

## Phase 09 STARTED — mechanism rules, ranking, and predictions

**PHASE REPORT — Phase 09 implementation and gate.**
- Files created/changed: `configs/mechanism_rules.yaml`; `src/xynapse/mechanism.py`; `src/xynapse/ranking.py`; `src/xynapse/prediction.py`; `scripts/predict_ranked_mutations.py`; `scripts/gates/gate_phase_09.py`; `tests/test_ranking_mechanism.py`; `reports/gates/phase_09_gate.txt`; `reports/gates/phase_09_pytest.txt`; `reports/tables/rank_class_counts.csv`; `reports/tables/phase_09_prediction_summary.json`; local ignored `data/processed/predictions.parquet`.
- Behavior: fixed rule-based mechanism flags from present annotations only; deterministic template evidence; four classes in the specified order; frozen ESM-2 35M Tier-1 stability and calibrated function models; FROZEN inventory and thresholds verified before inference. Allosteric hint stays disabled by default. Flags were re-derived from the saved prediction table joined to source mutation annotations and matched for every row.
- Tests: `pytest -q` — 105 passed in 9.35 s, recorded at `reports/gates/phase_09_pytest.txt`. Gate: 10/10 PASS, recorded in `reports/gates/phase_09_gate.txt`; includes schema/sequence validation, row completeness, finite predictions, D10 confidence, disclaimer, model identifier, mechanisms, rank-count consistency and unchanged frozen hashes.
- Predictions: 15,124 rows across PTEN/P60484 (7,657; 403 aa) and TP53/P04637 (7,467; 393 aa), exactly 19 substitutions per residue. One model tier, Tier 1; ESM-2 `facebook/esm2_t12_35M_UR50D`. Source: `reports/tables/phase_09_prediction_summary.json`.
- Rank classes: 4,102 potentially damaging and 11,022 uncertain; no potentially beneficial or tolerated rows because frozen `p_impact_low` is null. Per protein: TP53 1,305 potentially damaging / 6,162 uncertain; PTEN 2,797 / 4,860. Source: `reports/tables/rank_class_counts.csv`.
- Confidence: ddG confidence low for both proteins because both exceed the stability training maximum of 72 aa; function confidence medium for both because both are no longer than the 465-aa function training maximum and output tier is Tier 1. Source: saved model metadata and Phase 9 gate.
- Timing: merge 0.052937 s; model inference/ranking 0.994886 s; total 1.060488 s. Source: `reports/tables/phase_09_prediction_summary.json`.
- Tier-2 model: none; no annotation features were used by the models. Existing annotations only inform optional mechanism hints and evidence text. Function-label direction remains provisional per Phase 2/7 records.
- Failures fixed during implementation: initial Tier-1 join was narrowed to the two proteins present in the teammate mutation table; synthetic expectations were aligned to the specified overlapping rule behavior; corrected a gate syntax typo before the successful run. Final gate and test run are green.
- Git/pre-push: 4/4 pre-push checks PASS; shared-clone pytest 105 passed in 8.44 s. Staged status reviewed (14 files): modified `PROGRESS.md`, `reports/AGENT_LOG.md`, `reports/STATE.json`; added `configs/mechanism_rules.yaml`, Phase 9 gate and pytest reports, prediction summary and rank counts, gate/prediction scripts, three `src/xynapse` modules, and `tests/test_ranking_mechanism.py`. Final cached diff: 751 insertions, 4 deletions across 14 files.
- Git: Phase 9 branch publication not yet run; intended branch `ml/phase-09-ranking`, based on the verified Phase 8 follow-up SHA `ef20f6ee4d7721f2b9ab4bd5cb9a78ca3354493a`; compare URL will be https://github.com/AadhithyaMollin/Xynapse-AI-/compare/ml/phase-09-ranking?expand=1.
- Predictions are computational hypotheses, not experimental findings.
