#!/usr/bin/env python
"""Score all single mutations of a UniProt protein with ESM-2.

Usage: python scripts/run_esm.py --uniprot P60484 --model dev [--batch-size 16]
Writes data/interim/esm/{ACC}_{model_short}.parquet (cache) and
data/processed/mut_esm.parquet (one model only).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xynapse.esm_scoring import (  # noqa: E402
    read_esm_model,
    resolve_device,
    save_mut_esm,
    score_protein,
)
from xynapse.schema import MUT_ESM_COLS, MUT_KEYS  # noqa: E402

MODEL_ALIASES = {"dev": "dev_model", "final": "final_model"}
COLUMNS = MUT_KEYS + MUT_ESM_COLS


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def resolve_model(spec: str, config: dict) -> str:
    key = MODEL_ALIASES.get(spec)
    if key:
        return config["esm"][key]
    return spec


def short_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def fetch_sequence(acc: str) -> str:
    fasta = ROOT / "data" / "raw" / f"{acc}.fasta"
    if fasta.exists():
        text = fasta.read_text()
    else:
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
        fasta.parent.mkdir(parents=True, exist_ok=True)
        fasta.write_text(text)
    sequence = "".join(
        line.strip() for line in text.splitlines() if not line.startswith(">")
    )
    return sequence.upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniprot", required=True, help="UniProt accession, e.g. P60484")
    ap.add_argument("--model", default="dev",
                    help="dev | final | a Hugging Face model name")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    model_name = resolve_model(args.model, load_config())
    short = short_name(model_name)
    device = resolve_device()
    sequence = fetch_sequence(args.uniprot)

    print(f"uniprot: {args.uniprot}")
    print(f"model:   {model_name}")
    print(f"device:  {device}")
    print(f"length:  {len(sequence)}")

    cache_dir = ROOT / "data" / "interim" / "esm"
    cache = cache_dir / f"{args.uniprot}_{short}.parquet"

    started = time.perf_counter()
    if cache.exists():
        df = pd.read_parquet(cache)
        runtime = None
        print(f"cache:   {cache.relative_to(ROOT)} (already scored, skipped)")
    else:
        df = score_protein(args.uniprot, sequence, model_name,
                           batch_size=args.batch_size, device=device)
        runtime = time.perf_counter() - started
        save_mut_esm(df, cache, model_name)
        print(f"cache:   wrote {cache.relative_to(ROOT)}")
    print(f"rows:    {len(df)}")

    processed = ROOT / "data" / "processed" / "mut_esm.parquet"
    frames = []
    if processed.exists():
        existing_model = read_esm_model(processed)
        if existing_model != model_name:
            sys.exit(f"ERROR: {processed} holds {existing_model}; refusing to mix "
                     f"models. Move it aside or rerun with that model.")
        frames.append(pd.read_parquet(processed)[COLUMNS])

    for path in sorted(cache_dir.glob(f"*_{short}.parquet")):
        if read_esm_model(path) != model_name:
            print(f"skip:    {path.name} (different model)")
            continue
        frames.append(pd.read_parquet(path)[COLUMNS])

    merged = pd.concat(frames, ignore_index=True)
    merged = (merged.drop_duplicates(MUT_KEYS, keep="last")
                    .sort_values(MUT_KEYS)
                    .reset_index(drop=True))
    save_mut_esm(merged, processed, model_name)

    print(f"processed: {processed.relative_to(ROOT)}")
    print(f"proteins:  {merged['protein_id'].nunique()}")
    print(f"total rows: {len(merged)}")
    if runtime is None:
        print("runtime: 0.00 s (cached; not measured this run)")
    else:
        print(f"runtime: {runtime:.2f} s (measured)")


if __name__ == "__main__":
    main()
