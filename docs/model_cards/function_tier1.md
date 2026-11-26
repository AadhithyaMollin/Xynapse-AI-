# Function model card

Predictions are computational hypotheses, not experimental findings.

## Model

- Task: `function_binary_classification`
- Tier: `tier1`
- ESM model: `facebook/esm2_t12_35M_UR50D` (`esm2_t12_35M`)
- Training proteins: 26
- Training rows: 71962
- Maximum training protein length: 465 aa
- Training-label SHA-256: `c09f6d98281b4ce585a9673e4f3884bcf8eb4dbeb7968518ff135a94dd84392b`
- Feature-table SHA-256: `35304ff669593fde16983dcc62cec11e86bfaa844346f1f366184c945f7b3fb2`

## Features

- `esm_llr`
- `esm_wt_logp`
- `esm_mut_logp`
- `blosum62`
- `d_hydropathy`
- `d_volume`
- `d_charge`
- `esm_pos_mean_llr`
- `esm_pos_min_llr`
- `rel_position`
- `wt`
- `mut`

## Grouped out-of-fold metrics

- `auprc`: 0.5990668182855758
- `auroc`: 0.6421525085537665
- `brier`: 0.22651477761299615
- `mcc`: 0.27604765310840523
- `n_negative`: 40912
- `n_positive`: 31050
- `n_proteins`: 26
- `n_rows`: 71962
- `prevalence`: 0.4314777243545204
- `threshold`: 0.5874781534295241
- `tier`: tier1

## Held-out cluster metrics

- `auprc`: 0.6540219217863887
- `auroc`: 0.7570751729638872
- `brier`: 0.18865405688148276
- `mcc`: 0.37972523917436884
- `model`: tier1_full_calibrated
- `n_negative`: 3141
- `n_positive`: 1674
- `n_proteins`: 4
- `n_rows`: 4815
- `prevalence`: 0.34766355140186916
- `threshold`: 0.5874781534295241
- `tier`: tier1

## Known limits

Tier-2 model unavailable. Results are computational hypotheses; training labels and assays have heterogeneous sources. The stability model was trained on short domains; predictions beyond the recorded maximum length are extrapolations.

See [`docs/limitations.md`](../limitations.md).

Predictions are computational hypotheses, not experimental findings.
