# Sharing large artifacts

Predictions are computational hypotheses, not experimental findings.

The GitHub source tree excludes raw datasets and full processed research tables. The compact three-file teammate handoff and the complete frozen model bundle are committed under `models/esm2_t12_35M_handoff/` so teammates can fetch them with the repository. The bundles do not include training data or ESM-2 weights. Other large artifacts should be shared through an owner-managed Drive location or a GitHub Release, not staged into the repository.

For the complete production model package, include the matching `FROZEN.json`, model metadata JSON, a checksum manifest, the ESM model short name (`esm2_t12_35M` for this build), and a note that files must match the frozen hashes. The separately named three-file teammate handoff below has its own manifest and records the source frozen-inventory hash; its `function_logistic_tier1.joblib` intentionally has a different role and must not be overlaid on the canonical frozen model directory. Keep raw data out of public releases unless its source terms permit redistribution. Do not include access tokens or credential-bearing links in documentation, manifests, or commit history.

The tracked `demo_data/` bundle contains case-study predictions and optional small display structures; it does not include model weights and is marked `DEMO - NOT REAL` by the dashboard. It is not a substitute for the full model/data package.

## Build frozen artifacts and the three-file teammate handoff

From the repository root, run:

```sh
python scripts/package_model_artifacts.py
```

The script verifies the canonical frozen files against `FROZEN.json`, then creates the complete-production archive, the teammate archive, and the checksum/size manifest `reports/model_artifact_bundle.json`. The complete-production archive, exact teammate archive, and three extracted teammate model files are narrowly allowlisted for Git; all other model binaries remain ignored and blocked by the pre-push policy.

- `models/esm2_t12_35M_handoff/xynapse_ai_esm2_t12_35M_model_bundle.zip` contains the complete frozen inventory, metadata, `FROZEN.json`, and a correctly named standalone copy of the selected isotonic calibrator (`function_calibrator_tier1.joblib`). This verified 505 KiB archive is committed to the repository; it does not include ESM-2 weights or training data.
- `models/esm2_t12_35M_handoff/xynapse_ai_team_model_files_esm2_t12_35M.zip` contains the exact three requested top-level filenames: `stability_tier1.joblib`, `function_tier1.joblib`, and `function_logistic_tier1.joblib`. The ZIP and unpacked copies under `models/esm2_t12_35M_handoff/team_handoff/` are committed to the repository, outside the canonical frozen model directory. Its manifest is embedded as `TEAM_HANDOFF_MANIFEST.json` and also recorded in the sidecar JSON.

**Important same-name distinction:** in the canonical frozen model directory, `function_logistic_tier1.joblib` is the auxiliary logistic-regression feature baseline and is not a calibrator. In the separate `team_handoff/` directory and teammate archive, that requested filename is a Platt/logistic probability calibrator fitted to the saved grouped OOF scores. It is an alternative calibrator; the selected production calibrator is isotonic and remains embedded in the frozen `function_tier1.joblib` bundle. The manifest records this distinction, the OOF source-table checksum, and the artifact hashes. Do not extract the teammate handoff over `models/esm2_t12_35M/`, because that would replace the frozen logistic baseline with a calibrator of a different role.

The handoff Platt model is fit from the training OOF table only; held-out cluster, case-study, and ClinVar test labels are not used. The calibration comparison and selected-method evidence are in `models/esm2_t12_35M/function_tier1.json`. Teammates can retrieve the archive and individual files directly from the repository. Before loading serialized files, verify the archive SHA-256 from `reports/model_artifact_bundle.json`; then verify every file against the relevant embedded manifest. The committed complete frozen archive is the one to use for reproducing production inference. Verify its SHA-256 from `reports/model_artifact_bundle.json` before extracting it into `models/`.

Predictions are computational hypotheses, not experimental findings.
