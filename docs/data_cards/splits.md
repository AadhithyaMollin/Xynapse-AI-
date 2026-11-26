# Sequence clustering and split data card

Predictions are computational hypotheses, not experimental findings.

## Inputs and reproducibility

- MMseqs2 version: `20-ae60e`.
- Exact command: `/opt/homebrew/bin/mmseqs easy-cluster /Users/parthsrivastava/Developer/xynapse-ai/data/interim/train_proteins.fasta /Users/parthsrivastava/Developer/xynapse-ai/data/interim/mmseqs_train_proteins /Users/parthsrivastava/Developer/xynapse-ai/data/interim/mmseqs_tmp --min-seq-id 0.3 -c 0.8 --cov-mode 0`.
- Input FASTA: `data/interim/train_proteins.fasta`; SHA-256 `6e367e57d0ea9f9fef286974bc3e1061d3ce36b18b6ab270ca4dad6b4e446b41`.
- Observed MMseqs cluster TSV: `data/interim/mmseqs_train_proteins_cluster.tsv`; SHA-256 `0db3f187d226b66d5f6aa8ff6f63ef2c7a66b6309a440aaaa766d026a76c80de`.
- Retrieval/build date (UTC): 2026-11-26.
- TSV inspection: two tab-separated fields per row (representative, member); every FASTA ID occurs exactly once as a member.
- Measured proteins: 92; clusters: 79; test clusters: 14.
- Split policy: complete case-study/external-test clusters plus a seeded 15% sample of other clusters are test; remaining clusters are training. Training clusters are assigned to five shuffled GroupKFold validation folds with seed 42.
- All training artifacts remain local/ignored. See `reports/tables/split_summary.csv` for counts by task and split.
