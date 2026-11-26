# AGENT_BRIEF.md: Xynapse AI, ML build (ML-1 + ML-2 scope)

## 0. Binding rules (read first, apply always)

Context: you are a coding agent working in the repo root `xynapse-ai` on macOS (Apple Silicon, Python 3.12, no CUDA; torch device = mps if available else cpu). The human is Maansi, who owns ML-1 and ML-2. Two teammates (BIO-1, BIO-2) deliver biology tables. Project: predict effects of single amino-acid substitutions on protein stability (regression, ddG) and function (binary impact), then rank mutations with evidence-linked mechanism hints. All outputs are COMPUTATIONAL PREDICTIONS, never experimental findings.

R1. Never run any git command.
R2. Never fabricate. Every number in any report, doc, table, model card or dashboard text must come from code you ran in this session, saved to a file, and read back from that file. Never state accuracy, biology, literature facts, dataset properties, file names, URLs, column meanings or sign conventions from memory. If you need such a fact, get it from the actual file, README, API response or installed library; if you cannot, STOP and ask. Items marked [VERIFY] below are believed but unconfirmed. Resolve each one against real data and record the resolution in the relevant data card.
R3. Library APIs: installed versions are recent (transformers 5.8.1, torch 2.12.0, numpy 2.4.6, pandas 2.3.3, scikit-learn 1.8.0, lightgbm 4.7.0, plotly 7.2.0, streamlit 1.57.0, py3Dmol 3.0.0, biotite 2.0.0, pyarrow 24.0.0). Your memory of their APIs may be outdated. Before using any API you are not certain about, check `pip show`, `help()` or run a 3-line experiment. Do not assume parameter names.
R4. Do not modify existing files except: `configs/config.yaml` (additive keys only), `src/xynapse/schema.py` (additions only, never change existing names or values), `requirements.txt` (append only; log any addition). Teammate files in `data/processed/` (proteins.parquet, pos_*.parquet, any file you did not create) are READ-ONLY: never edit, overwrite or delete. If you find a bug in existing code (esm_scoring.py, physchem.py, mutations.py, validate.py), report it first and add a regression test with the fix.
R5. `schema.py` is the data contract. Sign convention: ddG > 0 = destabilizing, ddG < 0 = stabilizing, kcal/mol. Positions are 1-based. Mutation keys: protein_id, position, wt, mut. Position keys: protein_id, position, wt. Tables are Parquet. `scripts/validate_tables.py` must PASS on every table you write before the next phase.
R6. Tests: write tests alongside code. `pytest -q` must be fully green at the end of every phase. Phase gates must be implemented as tests or scripts that print PASS/FAIL, never as eyeballing.
R7. Failure protocol: if a check fails, diagnose, fix, rerun. At most 3 attempts per distinct failure, then STOP and report the exact error output, what you tried and your hypotheses. Never weaken, delete or skip a test or check to get green.
R8. STOP and ask the human when: a [VERIFY] item cannot be resolved from real files; any file listed in a data source is missing or the URL fails (do not guess alternate URLs); a download exceeds 500 MB (check Content-Length first); the sign or direction of any label is ambiguous; fewer than 10 training proteins survive any filter; a teammate table violates the contract; a required decision is not covered by this brief; you are about to evaluate any test set (see CP2).
R9. Logging and resumability: append (never rewrite) to `reports/AGENT_LOG.md`: phase, status (STARTED/DONE/BLOCKED), commands run, measured numbers, problems. At session start read this log and resume at the first phase not marked DONE. All long jobs are resumable (per-protein cache files; skip if the output exists). Record measured runtimes.
R10. Compute: laptop only. ESM-2 model is `facebook/esm2_t12_35M_UR50D` (config `esm.dev_model`) for everything. Do NOT run the 650M model. Parametrize all paths and outputs by the ESM model short name (for example `esm2_t12_35M`) so a later rerun with another model never mixes model sizes in one table or one model directory.
R11. Wording: never write "proven", "validated experimentally", "accurate", "discovered", "causes" in any output. Class "potentially beneficial" means only "predicted stabilizing and function-preserving, a hypothesis". Every page, report and README carries: "Predictions are computational hypotheses, not experimental findings."
R12. Determinism: use `seed` from config (42) for numpy, sklearn, lightgbm and any shuffles.
R13. No LLM or free-text generation inside the pipeline. `evidence_text` is built only from fixed templates plus values in the row.
R14. Phase report: at the end of each phase print and append to the log a PHASE REPORT with: files created or changed; tests count and result; gate checks PASS/FAIL list; measured numbers; deviations from this brief; open questions.
R15. Write code into files, never only into chat. Keep modules small. Do not print giant tables; print summaries.

## 1. Current state (verify in Phase 0, do not assume)

Believed to exist: `src/xynapse/{schema,validate,mutations,esm_scoring,physchem}.py`; possibly `join_features.py`; `scripts/{setup_check,validate_tables,run_esm,run_physchem,check_esm}.py` and possibly `run_join.py`; tests (31 passed at last report); `data/processed/mut_esm.parquet` (PTEN P60484, 7657 rows, ESM model recorded in file metadata under key b"esm_model"); `data/interim/mut_phys.parquet`; teammate files in `data/processed/`. Existing helpers `save_mut_esm` / `read_esm_model` preserve the ESM model name in Parquet schema metadata; reuse that pattern for every feature table.

## 2. Fixed design decisions

D1. Feature tiers.
- Tier 1 (always computable): esm_llr, esm_wt_logp, esm_mut_logp, blosum62, d_hydropathy, d_volume, d_charge, position-level ESM summaries computed from the protein's own 19 mutants WITHOUT labels (esm_pos_mean_llr, esm_pos_min_llr), wt and mut as categorical, and relative position (position / length).
- Tier 2 (needs teammate tables): rsasa, sasa_abs, ss3, ss8, n_contacts_8A, plddt, cons_entropy, cons_weighted, gap_frac, n_eff, msa_logodds, is_active_site, is_binding_site, is_metal, is_dna_binding, is_disulfide, is_interface, dist_to_nearest_site.
- Train a Tier-1 model on all training proteins. Train a Tier-2 model only on proteins whose Tier-2 tables are complete and whose sequence EXACTLY equals the sequence in the teammate `proteins.parquet`. Each saved model stores its feature list; apply a model only with exactly its feature list.
D2. protein_id: UniProt accession where resolvable and verified (sequence identical); otherwise the dataset's own identifier, with a `uniprot_id` column left null. Never guess accession mappings. Numbering is always relative to the sequence stored in the proteins table.
D3. Training data is single substitutions only, deduplicated on (protein_id, position, wt, mut).
D4. One function assay per protein in labels_function (key uniqueness is required by the contract). Choose deterministically: most valid single substitutions; ties broken by assay id alphabetically. Log the rejected assays.
D5. Case-study proteins (PTEN P60484, TP53 P04637, plus any third protein the human names) are NEVER used for training, validation or tuning. Their clusters are assigned to the test side.
D6. ClinVar is an external test only. No ClinVar-derived feature may enter any feature list. The function classifier is trained on DMS-derived labels only.
D7. Splits are by MMseqs2 cluster (identity 0.30, coverage 0.80, from config `splits`). Never split by position or mutation. Test = 15% of non-case-study clusters (seeded) plus all case-study clusters. Validation = 5-fold GroupKFold on training clusters (column `cv_fold`).
D8. Test sets are evaluated exactly once, after freeze (CP2).
D9. Class definitions for ranking. Conventional label cutoffs (plan parameters): damaging if ddG >= 1.0 kcal/mol, stabilizing if ddG <= -1.0 kcal/mol. Written to config under `ranking.ddg_label_cutoff: 1.0`; list this as a plan parameter in the final report.
D10. Confidence defaults (plan parameters, configurable under `confidence:` in config, listed in the final report): plddt_low: 70; n_eff_low: 5; `stability_extrapolation`: true when protein length exceeds the longest protein in the stability training set. Rules: Tier-1-only predictions are capped at "medium"; any low flag gives "low"; otherwise "high" for Tier 2 and "medium" for Tier 1.

## 3. Phases

### Phase 0: Audit
1. Read every file in src/, scripts/, tests/, configs/. List data/ with sizes. Run `pytest -q`. Save package versions to `reports/env_snapshot.json`.
2. Create `reports/AGENT_LOG.md`. Record whether `join_features.py`, `run_join.py` and `mut_features.parquet` exist.
Gate: pytest green; log exists.

### Phase 1: Teammate tables, inspection, join
Stage A (read-only). For every .parquet in data/processed/: row count, dtypes, distinct protein_id, nulls per column. Run `scripts/validate_tables.py`. For each position table check: `wt` present; 1-based; wt letters match proteins.parquet sequences; exactly L rows per protein; no duplicate POS_KEYS; missing-value encoding (look for sentinels such as -1, 999, empty strings); value ranges (rsasa about 0 to 1.5, plddt 0 to 100, gap_frac 0 to 1, n_eff positive, n_contacts_8A non-negative, is_* binary); protein_id format matches mut_esm ("P60484"); whether msa_logodds exists anywhere. Write `reports/tables/teammate_table_inspection.csv`. If a blocking problem exists (missing required columns, wrong keys, wt mismatch, duplicates, sentinel values, ID mismatch), STOP and give a numbered problem list plus a short message the human can send to the table's owner. Non-contract extra files (e.g. confidence flags) are logged and ignored.
Stage B (only if A has no blocking problem; a missing msa_logodds alone is not blocking).
- Run ESM-2 scoring (existing `scripts/run_esm.py` logic) and physchem for every protein in proteins.parquet that lacks them, so that TP53 or others delivered are covered.
- `src/xynapse/join_features.py`: `build_mut_features(...)`: merge mutation tables on MUT_KEYS; left-join each position table on POS_KEYS with `validate="many_to_one"`, only schema columns; ignore proteins not in mut_esm and report them; add msa_logodds as NaN if absent (with a warning); never fill or impute; keep dtypes; column order per `schema.TABLES["mut_features"]["required"]`; write `data/processed/mut_features.parquet` with ESM model name in metadata; `nan_report` to `reports/tables/nan_report_mut_features.csv`.
- Tests on synthetic 3-residue data: row count preserved; duplicate keys raise; wt mismatch yields NaN or error, never wrong values; NaN preserved; metadata round trip; extra proteins ignored.
Gate: validator PASS on mut_features with proteins.parquet present; per-protein rows = 19 x L; pytest green.

### Phase 2: Dataset acquisition and inspection (read-only decisions) then CP1
Download into `data/raw/<dataset>/`; record sha256, size and download date; write `docs/data_cards/<dataset>.md` (source URL, date, size, sha256, license text only if found in the source's own metadata, columns, row counts, filters). Check Content-Length first (R8).
2a. Mega-scale stability (Tsuboyama et al. 2023). Landing page: https://zenodo.org/records/7992926. List the record's files via the Zenodo API for that record or by parsing the page; read any README or documentation inside the record. Identify which file contains single-substitution ddG with reliability flags. [VERIFY file names, how ddG/dG is signed, which column marks reliable rows, how wild-type domains are named, whether sequences and structures are included]. Quote the exact sentence defining the sign. Report distinct mutation types and counts (substitutions, deletions, insertions, multi-mutants). Report domain length distribution.
2b. ProteinGym substitutions. Page: https://proteingym.org/download. Parse the page for the download links of the substitution assay files and the reference file; do not hard-code any URL from memory. [VERIFY file names, version, the assay table columns, whether a binary label column exists, the direction of that binary label, the assay phenotype column(s), numbering convention (offset) of mutant strings relative to the target sequence]. Verify numbering empirically: for each assay, check that the wt letter in the mutant string equals the target sequence at that position for at least 99% of single substitutions; if not, try a constant offset; if still not, drop the assay and log it.
2c. ClinVar: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz . Read the header. [VERIFY column names for assembly, variant type, name, gene symbol, clinical significance, review status]. Print the distinct values of review status and clinical significance. Filters: assembly GRCh38; single-nucleotide missense; protein change parsed from the Name field with regex `\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)` and converted to one-letter codes (drop Ter, Ext, frameshift, deletions); clinical significance Pathogenic / Likely pathogenic / Pathogenic/Likely pathogenic -> label 1; Benign / Likely benign / Benign/Likely benign -> label 0; drop everything else (uncertain, conflicting, other). Review status at least 1 star. Proposed star mapping to verify against observed values: 1 = "criteria provided, single submitter"; 2 = "criteria provided, multiple submitters, no conflicts"; 3 = "reviewed by expert panel"; 4 = "practice guideline"; 0 = "no assertion criteria provided" / "no assertion provided" / "no classification provided"; conflicting statuses excluded. Map only values you actually observe and print the mapping. Gene symbol to UniProt accession: only for proteins in proteins.parquet, using UniProt REST JSON (https://rest.uniprot.org/uniprotkb/{ACC}.json) for the gene name. Verify each parsed wt letter against the protein sequence; report mismatch rate per gene; if above 5% for a gene, STOP and report (likely a transcript/isoform mismatch). Collapse duplicate protein changes; drop and count those with conflicting labels.
2d. Proposed training sets (deterministic, seeded): ddG: eligible domains (at least 100 reliable single substitutions) -> seeded sample of up to 60 domains. Function: eligible assays (length <= 500, binary label available, numbering verified, one assay per protein per D4, excluding case-study proteins) -> up to 30 assays. Report the actual numbers.
2e. Label-definition note: write `docs/label_definitions.md` BEFORE any model exists: ddG sign conversion rule (stated as a formula, with the source sentence quoted), function label direction (1 = damaging/impact), phenotype per assay, dedupe rules.

CHECKPOINT CP1: print a report containing for each dataset: files used, quoted sign/direction definition, proposed conversion, [VERIFY] items resolved or unresolved, selected counts, ClinVar mapping tables, proposed training protein lists, and which ClinVar proteins the external test would cover (default: only proteins in proteins.parquet). Then STOP and wait for the human's "approved" or corrections. Do not start Phase 3 without it.

### Phase 3: Labels (after CP1 approval)
1. `src/xynapse/labels.py`: `to_project_sign(...)` implementing the approved ddG conversion in ONE function, plus tests with synthetic examples. Data-driven sign sanity check (script, saved to `reports/tables/sign_check.csv`): mean project-sign ddG for substitutions to proline must be positive; if not, STOP and report.
2. `data/processed/labels_ddg.parquet`: MUT_KEYS + ddg, source, plus `n_obs`, `domain_name` and extras as useful (columns split_id and cluster_id added in Phase 5). Duplicates handled as: average reliable observations, record `n_obs`, log how many duplicates were collapsed.
3. `data/processed/labels_function.parquet`: MUT_KEYS + label (1 = damaging), score_raw (raw assay score), plus `assay_id`, `phenotype`, `score_norm` (within-assay percentile rank). Never pool raw scores across assays. Direction check (saved to `reports/tables/label_direction_check.csv`): Spearman(label, esm_llr) per assay once ESM is computed in Phase 4; if the mean is positive, the label direction is flipped: STOP and report.
4. `data/processed/external_clinvar.parquet`: MUT_KEYS + label, review_stars, clinvar_variation_id (or the verified id column), for approved proteins.
5. Run the validator on all three.
Gate: contract PASS; no duplicate keys; label prevalence table saved (`reports/tables/label_prevalence.csv`); tests green.

### Phase 4: Features for training proteins
1. Build `data/interim/train_proteins.parquet` (PROTEIN_COLS; structure_source and mean_plddt null if unknown) for all ddG and function training proteins plus case-study and ClinVar-test proteins.
2. For each protein: generate mutations, ESM-2 scoring (resumable per-protein cache under `data/interim/esm/` as in `scripts/run_esm.py`), physchem, Tier-1 extras (D1). Sequences longer than 1022 raise an error: drop the protein and log.
3. Tier-2 availability manifest `reports/tables/training_protein_manifest.csv`: per protein: dataset, length, uniprot_id, tier-2 tables present (structure/evolution/function), sequence equals teammate sequence (yes/no/NA). Also write `reports/teammate_request.md`: the list of UniProt accessions and sequences for which BIO-1/BIO-2 should produce pos_* tables, with the exact contract columns. Do not generate Tier-2 biology features yourself.
Gate: per-protein row counts = 19 x L for every feature file; no NaN in Tier-1 columns; ESM model name identical across all files (assert); tests green.

### Phase 5: Clustering, splits, training tables
1. Write one FASTA of all proteins (training, case studies, ClinVar-test) and run `mmseqs easy-cluster` with `--min-seq-id 0.3 -c 0.8` using config values. Parse the cluster TSV (verify its format by reading real output). Assign `cluster_id` to every protein.
2. `src/xynapse/splits.py`: grouped split utility per D5/D7 with seeded shuffling; test = 15% of non-case-study clusters plus case-study clusters; 5-fold GroupKFold over training clusters -> `cv_fold`. Columns split_id (train/test) and cv_fold.
3. Write `data/processed/train_ddg.parquet` and `train_function.parquet` (features + labels + cluster_id + split_id + cv_fold). Update labels tables with cluster_id and split_id.
4. Tests: no cluster appears in two splits; no cluster in two cv folds; every row has a cluster_id; no case-study protein in train; no ClinVar-test protein shares a cluster with a training protein (if one does, move it out of training and log).
Gate: all the tests above; report n clusters, n proteins and n rows per split.

### Phase 6: Stability model (regression, ddG)
1. `src/xynapse/metrics.py`: Spearman, Pearson, RMSE, per-protein Spearman (min 30 variants; report how many proteins qualify), handling NaN and constant arrays. `reports/tables/*.csv` are always script-generated.
2. Baselines (score oriented so higher = more destabilizing): negative blosum62, abs(d_hydropathy), abs(d_volume), negative esm_llr. Evaluate with grouped CV (cv_fold) on training rows only. If any baseline's correlation sign contradicts expectation across the board, flag a possible sign problem in the report.
3. LightGBM regressors (Tier-1 feature set; and Tier-2 on the subset where available). Preprocessing fit on training folds only (use sklearn Pipelines or explicit fold-wise fitting). Categorical encodings with fixed category lists derived from training data; unseen -> NaN.
4. Ablations: ESM only; + physchem; + Tier-1 extras; (+ structure, + evolution, + function where Tier-2 data exists). Tables with n_proteins and n_rows.
5. Hyper-parameter search: random search of 20 configs (seeded), inner grouped CV, capped at 10 minutes wall time. Search space: num_leaves {7,15,31,63}; learning_rate {0.02,0.05,0.1,0.2}; n_estimators {200,400,800}; min_child_samples {20,50,100}; subsample {0.7,1.0}; colsample_bytree {0.6,0.8,1.0}; reg_lambda {0,1,10}. Choose by mean grouped-CV Spearman; record all configs.
6. Save OOF predictions (`reports/tables/oof_ddg.parquet`), final fit on all training rows (train split only; the test split stays untouched), model with joblib to `models/<esm_short>/stability_<tier>.joblib` and metadata JSON (feature list, library versions, n_proteins, n_rows, config hash, ESM model name).
Gate: OOF covers every training row exactly once; no cluster in both fit and predict folds (assert); metrics tables written; tests green.

### Phase 7: Function classifier
1. Baselines: negative esm_llr, negative blosum62, conservation only (Tier 2, if available). Classifier: LightGBM (and a logistic regression with median imputation + scaling inside a Pipeline). Metrics: AUROC, AUPRC, MCC (at the threshold maximizing MCC on OOF), calibration curve data, plus prevalence. Per-assay AUROC distribution.
2. The classifier does NOT use predicted ddG as a feature in v1. Optional ablation: if added, use ONLY out-of-fold ddG predictions for training rows (assert).
3. Same ablations and the same hyper-parameter protocol as Phase 6 (adapt search to the classifier metric: AUPRC).
4. Calibration: calibrate OOF probabilities with isotonic or Platt; choose by grouped cross-validated Brier score on the OOF predictions; apply to final fit. Save calibrator with the model.
5. Save models + metadata like Phase 6; OOF predictions to `reports/tables/oof_function.parquet`.
Gate: same assertions as Phase 6; calibration curve CSV written; tests green.

### Phase 8: Thresholds, freeze, test (CP2)
1. Derive thresholds from OOF only and write them to `configs/config.yaml` under `thresholds` together with the evidence table `reports/tables/threshold_evidence.csv`:
   - ddg_damaging: predicted-ddG cutoff maximizing MCC against (true ddG >= ranking.ddg_label_cutoff).
   - ddg_stabilizing: predicted-ddG cutoff maximizing MCC against (true ddG <= -ranking.ddg_label_cutoff). If too few positives (< 50), set null and log; "potentially beneficial" is then disabled.
   - p_impact_high: calibrated probability maximizing MCC on OOF.
   - p_impact_low: the largest calibrated probability for which NPV on OOF is >= 0.90; if unreachable, set null and log (tolerated then cannot be assigned confidently: everything falls to "uncertain").
2. `src/xynapse/freeze.py`: `freeze_models()` writes `models/<esm_short>/FROZEN.json` (sha256 of every model file, thresholds, feature lists, timestamp). `evaluate_once(name, ...)` records the evaluation in `reports/test_eval_registry.json` and REFUSES a second evaluation of the same name unless env `XYNAPSE_ALLOW_REEVAL=1` is set (in which case log the reason). Tests for both.
3. CHECKPOINT CP2: print the list of everything to be evaluated (stability held-out cluster test, classifier held-out cluster test, ClinVar external test), the frozen hashes, and the thresholds with evidence. STOP and wait for the human's "approved". Never evaluate a test set before this.
4. After approval, evaluate each test set exactly once through `evaluate_once`. Metrics: stability Spearman, Pearson, RMSE, per-protein Spearman; classifier AUROC, AUPRC, MCC, calibration; ClinVar: AUROC, AUPRC, MCC, n variants per label per protein. Always next to the baselines. Report n_proteins and n_rows with every number. Save `reports/tables/test_*.csv`. Report results whatever they are. If external performance is far below grouped-CV, say so in the report; do not retune.
Gate: registry has one entry per evaluation; tests green.

### Phase 9: Mechanism rules, ranking, predictions
1. `configs/mechanism_rules.yaml` + `src/xynapse/mechanism.py`: rule table (flags stored as a semicolon-separated string, empty string if none). Use only columns that exist:
   - catalytic_site: is_active_site true and p_impact >= p_impact_high.
   - binding_site: any of is_binding_site, is_metal, is_dna_binding true and p_impact >= p_impact_high.
   - folding_stability: ddg_pred >= ddg_damaging and (rsasa <= 0.25 if available else no condition).
   - interface: is_interface true and p_impact >= p_impact_high.
   - disulfide: is_disulfide true and (p_impact >= p_impact_high or ddg_pred >= ddg_damaging).
   - allosteric_hypothesis: DISABLED by default (config flag `mechanism.allosteric: false`); if enabled: p_impact >= p_impact_high, not in any annotated site, dist_to_nearest_site above a configured value, labeled "hypothesis".
   Unit tests for every rule with synthetic rows, including missing columns (flag simply not assigned).
2. `src/xynapse/ranking.py` with four classes using frozen thresholds: "potentially damaging" if p_impact >= p_impact_high or ddg_pred >= ddg_damaging; "potentially beneficial (hypothesis)" if thresholds exist and ddg_pred <= ddg_stabilizing and p_impact <= p_impact_low and confidence != low; "tolerated" if p_impact_low exists and p_impact <= p_impact_low and ddg_pred < ddg_damaging; "uncertain" otherwise. Confidence per D10. Missing models or features -> "uncertain".
3. `evidence_text` template, assembled only from row values: predicted ddG (kcal/mol), p_impact, ESM llr, flags, rsasa/plddt/conservation if present, annotation evidence code only if present in pos_function, tier of the model, confidence reason, and the fixed sentence "Computational prediction; not experimental evidence." No other free text.
4. Write `data/processed/predictions.parquet` (MUT_KEYS + PRED_COLS + `tier`, `esm_model`) for every protein with mut_features rows (PTEN, plus TP53 or others if present). Use the Tier-2 models where all their features exist for that protein, otherwise the Tier-1 models, recorded in `tier`.
Gate: validator PASS; rows = 19 x L per protein; rank_class counts per protein saved; no NaN in rank_class; every row has evidence_text containing the fixed sentence; tests green.

### Phase 10: Pipeline CLI
`python -m xynapse.pipeline --uniprot P60484 [--esm-model dev] [--out data/processed]`: fetch the UniProt sequence (cached), generate mutations, ESM score (resumable), physchem, Tier-1 extras, join position tables if present in `data/processed/` (otherwise run Tier-1 only and label the output as such), predict (stability + classifier, calibrated), apply frozen thresholds, mechanism flags, ranking, write predictions. Print stage timings (measured). Clear error if a required model file is missing or the frozen hash does not match. Document in the module docstring where BIO retrieval/evolution/structure modules plug in; do not implement them.
Gate: run the CLI for P60484 from scratch in a temp output dir (delete only that temp dir) and assert the result equals Phase 9's predictions for ddg_pred and p_impact within 1e-6; tests green.

### Phase 11: Dashboard (Streamlit, local, reads cached Parquet only)
`app/app.py`: top banner on every page: "Predictions are computational hypotheses, not experimental findings." Pages/tabs: protein summary (n mutations, class counts, tier, ESM model name); heatmap (Plotly; position x mutant amino acid; selectable metric: ddg_pred, p_impact, esm_llr); mutation table with filters (class, confidence, mechanism flag, position range) and CSV download; mechanism evidence panel (evidence_text); 3D view with py3Dmol colored by a per-residue value using the AlphaFold structure file if present under `data/raw/` (if no structure file exists, hide the 3D tab with an explanatory message; never fabricate); limitations tab built from `docs/limitations.md`. Any placeholder data must be labeled "DEMO - NOT REAL"; prefer real data only.
Tests with `streamlit.testing.v1.AppTest`: app runs without exception; banner text present; heatmap and table render for P60484; CSV download data has 19 x L rows. Also verify the headless server starts (`streamlit run app/app.py --server.headless true` for a few seconds, then stop it) and report the result.
Gate: AppTest passes; server starts.

### Phase 12: Reports, audits, docs
1. `scripts/make_reports.py`: regenerates every table and figure under `reports/` from saved outputs (no hand-typed numbers).
2. Model cards `docs/model_cards/*.md` generated from model metadata: training data (n proteins, n rows, datasets, versions/hashes), features, tier, CV metrics, test metrics, known limits.
3. `docs/limitations.md` generated or checked against facts: shallow MSAs possible; AlphaFold confidence varies; stability training domains are small so predictions for larger proteins are extrapolation (state the max training length from data); label noise; assay phenotypes differ; ESM-2 pre-trained on sequences that may include the test proteins (labels never); only the training proteins' number and tiers are reported.
4. `scripts/leakage_audit.py`: assert and print PASS/FAIL for: cluster disjointness across splits; no case-study protein in training; no duplicate keys in any label table; no ClinVar-derived column in any feature list; preprocessing fit inside folds (static check that no `fit` is called on a full table in the training modules); test registry has single evaluations; frozen hashes match; ESM model name consistent across every feature table and model metadata; OOF coverage exactly once.
5. `scripts/claims_audit.py`: scan README, docs and app text for the forbidden words in R11 and require the disclaimer sentence; fail otherwise.
6. Case-study tables (data only; humans write the prose): per case-study protein: predictions vs DMS values where available (Spearman, n), and the list of worst disagreements. Show failures, not only successes. Literature statements are left as `[HUMAN TO FILL]`.
7. `README.md` (setup, usage, reproduce every table and figure, limitations link) and `PROGRESS.md` entries appended (measured runtimes, data cards list, gates).

### Phase 13: Final verification
`scripts/verify_all.py` runs and prints PASS/FAIL: pytest; validator on every table; leakage_audit; claims_audit; row counts; frozen hash match; pipeline reproduction check; AppTest. Final report: a table of every deliverable with path and status, all measured numbers with file sources, the plan parameters used (D9, D10), unresolved [VERIFY] items, the training protein counts, and a list of what the human should review. Do not claim anything the files do not show.

## 4. Definition of done
All phases DONE in the log; `scripts/verify_all.py` exits 0; `pytest -q` green; no git commands were run; no teammate file modified; every number in docs traceable to a file in `reports/`.
