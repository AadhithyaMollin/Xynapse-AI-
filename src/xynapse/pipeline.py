"""Single-accession prediction CLI. BIO retrieval/evolution/structure modules plug in at the position-table join below."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Keep Apple-Silicon CPU inference bounded and avoid OpenMP over-subscription
# when PyTorch/Transformers and LightGBM are used in the same process.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd

from .esm_scoring import resolve_device, save_mut_esm, score_protein
from .features import add_tier1_features, model_short_name
from .prediction import ESM_MODEL, load_prediction_context, predict_mutations
from .schema import MUT_KEYS, POS_KEYS

ROOT = Path(__file__).resolve().parents[2]


def _parse_fasta(payload: str) -> tuple[str, str]:
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError("UniProt FASTA response did not contain a FASTA header")
    sequence = "".join(line for line in lines[1:] if not line.startswith(">"))
    if not sequence:
        raise ValueError("UniProt FASTA response contained an empty sequence")
    return lines[0][1:], sequence.upper()


def _fetch_fasta(accession: str, cache_dir: Path, project_root: Path) -> tuple[str, str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{accession}.fasta"
    if cache_path.is_file():
        header, sequence = _parse_fasta(cache_path.read_text())
        return accession, sequence, "fasta_cache"

    # Reuse a locally verified sequence record when this accession is already in project data.
    for table_path in (project_root / "data/interim/train_proteins.parquet",
                       project_root / "data/processed/proteins.parquet"):
        if not table_path.is_file():
            continue
        proteins = pd.read_parquet(table_path)
        candidates = proteins.loc[
            proteins.protein_id.astype(str).eq(accession)
            | proteins.uniprot_id.fillna("").astype(str).eq(accession)
        ]
        if not candidates.empty:
            sequence = str(candidates.iloc[0].sequence).strip().upper()
            header = f"{accession} locally cached verified sequence"
            cache_path.write_text(f">{header}\n{sequence}\n")
            return accession, sequence, f"local_sequence:{table_path.relative_to(project_root)}"

    if not re.fullmatch(r"[A-Z0-9]{6,10}", accession):
        raise ValueError("--uniprot must be a UniProt accession using uppercase letters and digits")
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    request = urllib.request.Request(url, headers={"User-Agent": "Xynapse-AI/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"UniProt returned HTTP {response.status} for {accession}")
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"could not retrieve {accession} from rest.uniprot.org: {exc}") from exc
    header, sequence = _parse_fasta(payload)
    cache_path.write_text(f">{header}\n{sequence}\n")
    return accession, sequence, "rest.uniprot.org"


def _join_position_tables(frame: pd.DataFrame, protein_id: str, processed_dir: Path) -> tuple[pd.DataFrame, str]:
    joined = frame.copy()
    joined_any = False
    for table_name in ("pos_structure", "pos_evolution", "pos_function"):
        table_path = processed_dir / f"{table_name}.parquet"
        if not table_path.is_file():
            continue
        table = pd.read_parquet(table_path)
        table = table.loc[table.protein_id.astype(str).eq(protein_id)].copy()
        if table.empty:
            continue
        missing = [key for key in POS_KEYS if key not in table.columns]
        if missing:
            raise ValueError(f"{table_name} violates the position-table contract: missing {missing}")
        duplicates = table.duplicated(POS_KEYS).sum()
        if duplicates:
            raise ValueError(f"{table_name} has {duplicates} duplicate position keys for {protein_id}")
        value_columns = [column for column in table.columns if column not in POS_KEYS]
        joined = joined.merge(table[POS_KEYS + value_columns], on=POS_KEYS, how="left", validate="many_to_one")
        joined_any = True
    return joined, "tier1_plus_available_position_tables" if joined_any else "tier1_only"


def run_pipeline(
    uniprot_accession: str,
    *,
    esm_model: str = "dev",
    out: str | Path = "data/processed",
    project_root: str | Path = ROOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    project_root = Path(project_root).resolve()
    output_dir = Path(out).expanduser().resolve()
    if esm_model not in {"dev", ESM_MODEL}:
        raise ValueError(f"unsupported ESM model {esm_model!r}; this project uses only {ESM_MODEL}")
    model_name = ESM_MODEL
    model_short = model_short_name(model_name)
    stage_times: dict[str, float] = {}

    started = time.monotonic()
    protein_id, sequence, sequence_source = _fetch_fasta(
        str(uniprot_accession).strip().upper(), project_root / "data/raw/uniprot_cache", project_root
    )
    if len(sequence) > 1022:
        raise ValueError(f"{protein_id}: sequence length {len(sequence)} exceeds configured ESM max_len 1022")
    bad = set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
    if bad:
        raise ValueError(f"{protein_id}: sequence contains non-canonical residues {sorted(bad)}")
    stage_times["sequence_fetch_cache_seconds"] = time.monotonic() - started

    started = time.monotonic()
    cache_dir = output_dir / "interim/esm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9]", "_", protein_id)
    cache_path = cache_dir / f"{safe_id}_{model_short}_UR50D.parquet"
    if cache_path.is_file():
        esm = pd.read_parquet(cache_path)
        if set(esm.protein_id.astype(str)) != {protein_id} or len(esm) != 19 * len(sequence):
            raise ValueError(f"ESM cache does not match accession/sequence length: {cache_path}")
    else:
        esm = score_protein(protein_id, sequence, model_name, device=resolve_device())
        save_mut_esm(esm, cache_path, model_name)
    tier1 = add_tier1_features(esm, sequence)
    stage_times["esm_and_tier1_seconds"] = time.monotonic() - started

    started = time.monotonic()
    frame, data_tier = _join_position_tables(tier1, protein_id, project_root / "data/processed")
    frame["protein_length"] = len(sequence)
    stage_times["position_join_seconds"] = time.monotonic() - started

    started = time.monotonic()
    features_path = project_root / "data/processed/features_tier1.parquet"
    model_dir = project_root / "models" / model_short
    context = load_prediction_context(model_dir, project_root / "configs/config.yaml", features_path)
    predictions = predict_mutations(frame, context, protein_length=len(sequence))
    stage_times["frozen_model_prediction_seconds"] = time.monotonic() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    tier1.to_parquet(output_dir / "features_tier1.parquet", index=False)
    summary = {
        "protein_id": protein_id,
        "sequence_length": len(sequence),
        "sequence_source": sequence_source,
        "n_mutations": int(len(predictions)),
        "data_tier": data_tier,
        "esm_model": model_name,
        "esm_model_short": model_short,
        "stage_seconds": stage_times,
        "total_seconds": float(sum(stage_times.values())),
        "prediction_output": str(output_dir / "predictions.parquet"),
        "disclaimer": "Predictions are computational hypotheses, not experimental findings.",
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return predictions, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict effects of all single substitutions for a UniProt accession.")
    parser.add_argument("--uniprot", required=True, help="verified UniProt accession")
    parser.add_argument("--esm-model", default="dev", help="only dev / ESM-2 35M is supported")
    parser.add_argument("--out", default="data/processed", help="prediction output directory")
    args = parser.parse_args(argv)
    _, summary = run_pipeline(args.uniprot, esm_model=args.esm_model, out=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
