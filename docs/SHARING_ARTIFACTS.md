# Sharing large artifacts

Predictions are computational hypotheses, not experimental findings.

The GitHub source tree intentionally excludes raw datasets, full processed research tables, and model binaries. Share those larger artifacts through an owner-managed Drive location or a GitHub Release, not by staging them into the repository.

For every shared model package, include the matching `FROZEN.json`, model metadata JSON, a checksum manifest, the ESM model short name (`esm2_t12_35M` for this build), and a note that model files must match the frozen hashes. Keep raw data out of public releases unless its source terms permit redistribution. Do not include access tokens or credential-bearing links in documentation, manifests, or commit history.

The tracked `demo_data/` bundle contains case-study predictions and optional small display structures; it does not include model weights and is marked `DEMO - NOT REAL` by the dashboard. It is not a substitute for the full model/data package.

Predictions are computational hypotheses, not experimental findings.
