"""ESM-2 masked-marginal scoring of single amino-acid substitutions.

For each residue the residue is replaced with <mask> and the model is run once;
the log-softmax at that token gives the log-probability of all 20 amino acids.
`esm_llr = esm_mut_logp - esm_wt_logp` (higher = mutant more plausible to the
model). This is a model score, not a ddG and not an experimental measurement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from .mutations import generate_mutations
from .schema import AA, MUT_ESM_COLS, MUT_KEYS

# configs/config.yaml -> esm.max_len. The tokenizer adds <cls> + <eos>,
# so this is the longest residue sequence that fits ESM-2's 1024 positions.
MAX_LEN = 1022


def resolve_device(device: str | None = None) -> str:
    """Explicit device wins; otherwise cuda, else mps, else cpu."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_esm(model_name: str, device: str | None = None):
    """Return (tokenizer, model) for an ESM-2 masked-LM, in eval mode."""
    from transformers import AutoTokenizer, EsmForMaskedLM

    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).eval().to(device)
    return tokenizer, model


def score_protein(
    protein_id: str,
    sequence: str,
    model_name: str,
    batch_size: int = 16,
    device: str | None = None,
) -> pd.DataFrame:
    """Score every single substitution of `sequence` (19 * L rows, 1-based).

    One masked forward pass per position, batched `batch_size` positions at a
    time. Columns: MUT_KEYS + MUT_ESM_COLS.
    """
    sequence = sequence.strip().upper()
    if len(sequence) > MAX_LEN:
        raise ValueError(
            f"{protein_id}: sequence length {len(sequence)} > max_len {MAX_LEN}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    tokenizer, model = load_esm(model_name, device)
    model_device = next(model.parameters()).device

    # token index p == residue position p (the tokenizer puts <cls> at 0)
    input_ids = tokenizer(sequence, return_tensors="pt")["input_ids"].to(model_device)
    mask_id = tokenizer.mask_token_id
    aa_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(a) for a in AA], device=model_device
    )
    L = len(sequence)

    logp = np.empty((L, len(AA)), dtype=np.float64)
    positions = list(range(1, L + 1))

    with torch.no_grad():
        for start in tqdm(
            range(0, L, batch_size),
            desc=f"ESM {protein_id} ({L} positions)",
            unit="batch",
        ):
            batch_pos = positions[start:start + batch_size]
            ids = input_ids.repeat(len(batch_pos), 1)
            for row, pos in enumerate(batch_pos):
                ids[row, pos] = mask_id

            logits = model(input_ids=ids).logits  # [B, L+2, vocab]
            rows = torch.arange(len(batch_pos), device=model_device)
            cols = torch.tensor(batch_pos, device=model_device)
            lp = torch.log_softmax(logits[rows, cols, :].float(), dim=-1)
            logp[np.asarray(batch_pos) - 1] = lp[:, aa_ids].cpu().numpy()

    df = generate_mutations(protein_id, sequence)
    aa_index = {aa: i for i, aa in enumerate(AA)}
    pos_idx = df["position"].to_numpy() - 1
    df["esm_wt_logp"] = logp[pos_idx, df["wt"].map(aa_index).to_numpy()]
    df["esm_mut_logp"] = logp[pos_idx, df["mut"].map(aa_index).to_numpy()]
    df["esm_llr"] = df["esm_mut_logp"] - df["esm_wt_logp"]
    return df[MUT_KEYS + MUT_ESM_COLS]


def save_mut_esm(df: pd.DataFrame, path: str | Path, model_name: str) -> None:
    """Write a mut_esm table to Parquet, recording the model in schema metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        df[MUT_KEYS + MUT_ESM_COLS], preserve_index=False
    )
    metadata = dict(table.schema.metadata or {})
    metadata[b"esm_model"] = model_name.encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


def read_esm_model(path: str | Path) -> str:
    """Return the ESM model name stored in a Parquet file's schema metadata."""
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(b"esm_model")
    if raw is None:
        raise KeyError(f"{path}: no b'esm_model' entry in schema metadata")
    return raw.decode("utf-8")
