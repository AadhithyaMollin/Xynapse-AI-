"""Immutable model inventory and one-time test evaluation registry."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + "\n")
    temp.replace(path)


def freeze_models(model_dir: str | Path, thresholds: dict, feature_lists: dict[str, list[str]],
                  *, esm_model: str, output_path: str | Path | None = None) -> dict:
    """Hash every model artifact (excluding FROZEN.json) and persist freeze metadata."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {directory}")
    model_files = sorted(path for path in directory.rglob("*")
                         if path.is_file() and path.name != "FROZEN.json" and not path.name.endswith(".tmp"))
    if not model_files:
        raise ValueError("cannot freeze an empty model directory")
    if not esm_model.endswith("esm2_t12_35M_UR50D"):
        raise ValueError(f"unsupported project ESM model in freeze: {esm_model}")
    inventory = {str(path.relative_to(directory)): sha256(path) for path in model_files}
    payload = {
        "esm_model": esm_model,
        "esm_model_short": "esm2_t12_35M",
        "model_files_sha256": inventory,
        "thresholds": thresholds,
        "feature_lists": feature_lists,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = Path(output_path) if output_path else directory / "FROZEN.json"
    _write_json(target, payload)
    return payload


def verify_frozen(frozen_path: str | Path, model_dir: str | Path) -> tuple[bool, list[str]]:
    frozen_path = Path(frozen_path)
    directory = Path(model_dir)
    if not frozen_path.is_file():
        return False, [f"missing freeze file: {frozen_path}"]
    frozen = json.loads(frozen_path.read_text())
    current_files = {str(path.relative_to(directory)): sha256(path)
                     for path in sorted(directory.rglob("*"))
                     if path.is_file() and path.name != "FROZEN.json" and not path.name.endswith(".tmp")}
    expected = frozen.get("model_files_sha256", {})
    problems = []
    if set(current_files) != set(expected):
        problems.append("model file inventory changed after freeze")
    for name in sorted(set(current_files) & set(expected)):
        if current_files[name] != expected[name]:
            problems.append(f"SHA-256 mismatch: {name}")
    return not problems, problems


def evaluate_once(name: str, evaluator: Callable[[], Any], registry_path: str | Path) -> Any:
    """Reserve and execute a named evaluation once; re-evaluation requires explicit override and reason."""
    if not name or not name.strip():
        raise ValueError("evaluation name must be non-empty")
    path = Path(registry_path)
    registry = json.loads(path.read_text()) if path.is_file() else {"evaluations": {}}
    evaluations = registry.setdefault("evaluations", {})
    override = os.environ.get("XYNAPSE_ALLOW_REEVAL") == "1"
    reason = os.environ.get("XYNAPSE_REEVAL_REASON", "").strip()
    previous = evaluations.get(name)
    if previous is not None and not override:
        raise RuntimeError(f"evaluation {name!r} already recorded with status {previous.get('status')}")
    if previous is not None and override and not reason:
        raise RuntimeError("XYNAPSE_REEVAL_REASON is required when XYNAPSE_ALLOW_REEVAL=1")
    entry = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    if previous is not None:
        entry["re_evaluation_reason"] = reason
        entry["previous_entry"] = previous
    evaluations[name] = entry
    _write_json(path, registry)
    try:
        result = evaluator()
    except Exception as exc:
        entry.update({"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                      "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(path, registry)
        raise
    entry.update({"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "result": result})
    _write_json(path, registry)
    return result
