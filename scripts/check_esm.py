import pandas as pd
import pyarrow.parquet as pq

path = "data/processed/mut_esm.parquet"
df = pd.read_parquet(path)
print("rows:", len(df), "| model:", pq.read_schema(path).metadata.get(b"esm_model"))

# 1. The wt log-prob is read from one masked pass per position,
#    so all 19 mutants at a position must share it.
spread = df.groupby("position")["esm_wt_logp"].agg(lambda s: s.max() - s.min())
print("max wt_logp spread within a position (expect ~0):", spread.max())

# 2. Direction check: the real letter is usually the model's favourite,
#    so most llr values should be negative.
print("fraction of llr < 0 (expect clearly above 0.5):", (df.esm_llr < 0).mean().round(3))

# 3. Per-position sensitivity: mean llr over the 19 substitutions.
pos = df.groupby(["position", "wt"])["esm_llr"].mean().sort_values()
print("\n10 most sensitive positions:\n", pos.head(10))
print("\n10 most tolerant positions:\n", pos.tail(10))