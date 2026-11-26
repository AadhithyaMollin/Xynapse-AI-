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
