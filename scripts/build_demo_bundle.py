#!/usr/bin/env python3
"""Build a small, checksum-manifested demo bundle from saved predictions and AlphaFold DB."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
ALLOWED_HOST = "alphafold.ebi.ac.uk"
MAX_STRUCTURE_BYTES = 2 * 1024 * 1024
PROTEINS = ("P60484", "P04637")
MUT_KEYS = ["protein_id", "position", "wt", "mut"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(url: str, *, max_bytes: int) -> tuple[bytes, dict[str, str]]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
        raise ValueError(f"Refusing URL outside the approved host: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Xynapse-AI/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise ValueError(f"Response exceeds {max_bytes} byte limit: {length}")
        payload = bytearray()
        while True:
            chunk = response.read(min(64 * 1024, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise ValueError(f"Response exceeded {max_bytes} byte limit while streaming")
        return bytes(payload), dict(response.headers.items())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)


def _fetch_structure(accession: str, structure_dir: Path) -> dict[str, Any]:
    api_url = f"https://{ALLOWED_HOST}/api/prediction/{urllib.parse.quote(accession)}"
    api_bytes, _ = _request(api_url, max_bytes=2 * 1024 * 1024)
    payload = json.loads(api_bytes.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected AlphaFold API shape for {accession}")
    record = next((row for row in payload if isinstance(row, dict)
                   and row.get("uniprotAccession") == accession
                   and row.get("uniprotStart") == 1), None)
    if record is None or not isinstance(record.get("pdbUrl"), str):
        raise ValueError(f"No full-length PDB record in AlphaFold API response for {accession}")
    pdb_url = record["pdbUrl"]
    pdb_bytes, headers = _request(pdb_url, max_bytes=MAX_STRUCTURE_BYTES)
    if not pdb_bytes.startswith(b"HEADER") or b"ATOM  " not in pdb_bytes:
        raise ValueError(f"PDB response for {accession} lacks expected PDB records")
    path = structure_dir / f"{accession}.pdb"
    _atomic_write(path, pdb_bytes)
    return {
        "accession": accession,
        "api_url": api_url,
        "pdb_url": pdb_url,
        "model_entity_id": record.get("modelEntityId"),
        "latest_version": record.get("latestVersion"),
        "sequence_start": record.get("uniprotStart"),
        "sequence_end": record.get("uniprotEnd"),
        "bytes": len(pdb_bytes),
        "sha256": hashlib.sha256(pdb_bytes).hexdigest(),
        "path": path.relative_to(ROOT).as_posix(),
        "content_type": headers.get("Content-Type"),
    }


def build(root: Path = ROOT) -> dict[str, Any]:
    processed = root / "data/processed"
    demo = root / "demo_data"
    predictions = pd.read_parquet(processed / "predictions.parquet")
    features = pd.read_parquet(processed / "features_tier1.parquet")[MUT_KEYS + ["esm_llr"]]
    if features.duplicated(MUT_KEYS).any():
        raise ValueError("Tier-1 feature table has duplicate mutation keys")
    predictions = predictions.merge(features, on=MUT_KEYS, how="left", validate="one_to_one")
    if predictions.esm_llr.isna().any():
        raise ValueError("ESM LLR is missing for one or more demo predictions")
    predictions = predictions.loc[predictions.protein_id.astype(str).isin(PROTEINS)].copy()
    if set(predictions.protein_id.astype(str).unique()) != set(PROTEINS):
        raise ValueError("Demo predictions must contain both designated case-study accessions")
    if predictions.duplicated(MUT_KEYS).any():
        raise ValueError("Demo predictions have duplicate mutation keys")
    if predictions.esm_model.astype(str).nunique() != 1:
        raise ValueError("Demo predictions contain mixed ESM model names")
    for accession, frame in predictions.groupby("protein_id"):
        positions = int(frame.position.max())
        if len(frame) != 19 * positions or not frame.evidence_text.astype(str).str.contains(
                "Computational prediction; not experimental evidence.", regex=False).all():
            raise ValueError(f"Unexpected mutation count or evidence text for {accession}")
    demo.mkdir(parents=True, exist_ok=True)
    prediction_path = demo / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    if prediction_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Demo prediction Parquet exceeds the 8 MiB tracked-file cap")

    structure_records = []
    structure_dir = demo / "structures"
    for accession in PROTEINS:
        structure_records.append(_fetch_structure(accession, structure_dir))
    if sum(item["bytes"] for item in structure_records) > 20 * 1024 * 1024:
        raise ValueError("Demo structures exceed the combined 20 MiB tracked-file cap")

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generated_at_utc": generated_at,
        "disclaimer": DISCLAIMER,
        "esm_model": str(predictions.esm_model.iloc[0]),
        "prediction_rows": int(len(predictions)),
        "proteins": {str(key): int(len(value)) for key, value in predictions.groupby("protein_id")},
        "files": {},
        "structures": structure_records,
    }
    bundle_files = [prediction_path, *(structure_dir / f"{accession}.pdb" for accession in PROTEINS)]
    manifest["files"] = {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in bundle_files
    }
    manifest_path = demo / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    card_lines = [
        "# AlphaFold demo structures data card", "", DISCLAIMER, "",
        "## Retrieval", "",
        f"- Retrieval date (UTC): {generated_at}",
        f"- Source API host: `{ALLOWED_HOST}`; GET-only requests; no credentials.",
        "- The API response was inspected for accession, full-length start, model ID, version, and PDB URL fields; the returned PDB URL host was checked before retrieval.",
        "- Structures are display-only model data; their presence does not change prediction features or model outputs.", "",
        "## Files", "",
    ]
    for item in structure_records:
        card_lines.extend([
            f"### `{item['accession']}`", "",
            f"- URL: {item['pdb_url']}",
            f"- AlphaFold model entity: `{item['model_entity_id']}`; version field observed: `{item['latest_version']}`.",
            f"- Sequence span reported by API: {item['sequence_start']}–{item['sequence_end']}.",
            f"- Retrieved size: {item['bytes']} bytes; SHA-256: `{item['sha256']}`.",
            f"- Local bundle path: `{item['path']}`.", "",
        ])
    card_lines.append(DISCLAIMER)
    (root / "docs/data_cards/alphafold_demo.md").write_text("\n".join(card_lines) + "\n", encoding="utf-8")
    print(f"Demo rows={len(predictions)}; Parquet bytes={prediction_path.stat().st_size}; ESM={manifest['esm_model']}")
    for record in structure_records:
        print(f"Structure {record['accession']}: bytes={record['bytes']}; sha256={record['sha256']}")
    print(f"Manifest: {manifest_path.relative_to(root)}")
    return manifest


if __name__ == "__main__":
    try:
        build()
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Demo bundle build failed: {exc}") from exc
