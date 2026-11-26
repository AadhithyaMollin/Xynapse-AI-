# Stability model card

Predictions are computational hypotheses, not experimental findings.

## Model

- Task: `stability_regression`
- Tier: `tier1`
- ESM model: `facebook/esm2_t12_35M_UR50D` (`esm2_t12_35M`)
- Training proteins: 52
- Training rows: 45411
- Maximum training protein length: 72 aa
- Training-label SHA-256: `19c7dfe3c2ac0d608209e4ca5e9c70936dc448cd3cf724288f0d0748d84af65f`
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

- `model`: tier1_full
- `n`: 45411
- `n_proteins`: 52
- `n_rows`: 45411
- `pearson`: 0.5457737817448227
- `rmse`: 0.8231095474399082
- `spearman`: 0.4802360990327169
- `tier`: tier1

## Held-out cluster metrics

- `model`: tier1_full
- `n`: 7390
- `n_proteins`: 8
- `n_rows`: 7390
- `pearson`: 0.5571748386680233
- `rmse`: 0.8587660488399698
- `spearman`: 0.4523390593252031
- `tier`: tier1

## Known limits

Tier-2 model unavailable. Results are computational hypotheses; training labels and assays have heterogeneous sources. The stability model was trained on short domains; predictions beyond the recorded maximum length are extrapolations.

See [`docs/limitations.md`](../limitations.md).

Predictions are computational hypotheses, not experimental findings.
