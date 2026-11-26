#!/usr/bin/env python3
"""Copy only approved project artifacts into the human-maintained shared clone."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = "reports/.sync_manifest.json"
DIRECTORY_ROOTS = {"src", "scripts", "tests", "configs", "docs", "app", "reports", "demo_data"}
ROOT_FILES = {
    "README.md", "requirements.txt", "requirements.lock", "pytest.ini", ".gitignore", ".python-version",
    "PROGRESS.md", "AGENT_BRIEF.md", "AGENT_BRIEF_V2.md", "colab_setup.sh", "local_setup.sh",
}
PLACEHOLDER_DIRS = {"data/raw", "data/interim", "data/external", "data/processed", "models", "logs", "notebooks"}
REPORT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".png"}
DEMO_EXTENSIONS = REPORT_EXTENSIONS | {".parquet", ".pdb", ".cif", ".mmcif"}


class SyncConflict(RuntimeError):
    pass


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_allowed(rel: str, size: int | None = None) -> bool:
    path = PurePosixPath(rel)
    parts = rel.split("/")
    root = parts[0] if parts else ""
    if not parts or path.is_absolute() or ".." in parts or "" in parts:
        return False
    if rel in ROOT_FILES:
        return True
    if root == "models" and len(parts) >= 3 and path.suffix.lower() == ".json":
        return size is None or size <= 5 * 1024 * 1024
    if root in PLACEHOLDER_DIRS:
        return len(parts) == 2 and path.name == ".gitkeep"
    if root == "reports":
        if rel == MANIFEST_REL:
            return False
        if path.suffix.lower() not in REPORT_EXTENSIONS:
            return False
        if path.suffix.lower() == ".parquet":
            return False
        return size is None or size <= 5 * 1024 * 1024
    if root == "demo_data":
        if path.suffix.lower() not in DEMO_EXTENSIONS:
            return False
        if path.suffix.lower() in {".pdb", ".cif", ".mmcif"} and not rel.startswith("demo_data/structures/"):
            return False
        cap = 2 * 1024 * 1024 if rel.startswith("demo_data/structures/") else 8 * 1024 * 1024
        return size is None or size <= cap
    if root in DIRECTORY_ROOTS:
        if path.name == ".DS_Store" or "__pycache__" in parts or ".pytest_cache" in parts:
            return False
        if path.suffix.lower() in {".parquet", ".joblib", ".pkl", ".pt", ".bin", ".zip", ".gz", ".tar"}:
            return False
        return size is None or size <= 5 * 1024 * 1024
    return False


def _read_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    files = raw.get("files", raw)
    if not isinstance(files, dict):
        raise ValueError(f"Invalid sync manifest format: {path}")
    return {str(name): str(value) for name, value in files.items()}


def _managed_block(text: str) -> tuple[int, int, str]:
    begin, end = "<!-- XYNAPSE-ML:BEGIN -->", "<!-- XYNAPSE-ML:END -->"
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < start:
        raise ValueError("README.md must contain the managed XYNAPSE-ML markers")
    return start, stop + len(end), text[start:stop + len(end)]


def _readme_output(source: str, existing: str | None) -> str:
    _, _, source_block = _managed_block(source)
    if existing is None:
        return source
    try:
        start, stop, _ = _managed_block(existing)
    except ValueError:
        return existing.rstrip() + "\n\n" + source_block + "\n"
    return existing[:start] + source_block + existing[stop:]


def collect_source_files(root: Path = ROOT) -> list[Path]:
    found: list[Path] = []
    for name in sorted(ROOT_FILES):
        candidate = root / name
        if candidate.is_file() and is_allowed(name, candidate.stat().st_size):
            found.append(candidate)
    for dirname in sorted(DIRECTORY_ROOTS | PLACEHOLDER_DIRS):
        base = root / dirname
        if not base.is_dir():
            continue
        for candidate in sorted(base.rglob("*")):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            rel = candidate.relative_to(root).as_posix()
            if is_allowed(rel, candidate.stat().st_size):
                found.append(candidate)
    return sorted(found, key=lambda item: item.relative_to(root).as_posix())


def sync(shared: Path, dry_run: bool = False, root: Path = ROOT) -> list[str]:
    shared = shared.expanduser().resolve()
    if not shared.is_dir():
        raise FileNotFoundError(f"Shared clone directory does not exist: {shared}")
    manifest_path = root / MANIFEST_REL
    previous = _read_manifest(manifest_path)
    plan: list[tuple[str, Path, bytes, str]] = []
    conflicts: list[str] = []

    for source in collect_source_files(root):
        rel = source.relative_to(root).as_posix()
        target = shared / rel
        source_bytes = source.read_bytes()
        if rel == "README.md" and target.exists():
            desired = _readme_output(source_bytes.decode("utf-8"), target.read_text(encoding="utf-8")).encode("utf-8")
        else:
            desired = source_bytes
        desired_hash = sha256_bytes(desired)
        if target.exists():
            current_hash = sha256_file(target)
            can_initialize_readme_block = False
            if rel == "README.md":
                try:
                    _managed_block(target.read_text(encoding="utf-8"))
                except ValueError:
                    can_initialize_readme_block = True
            if current_hash != desired_hash and current_hash != previous.get(rel) and not can_initialize_readme_block:
                conflicts.append(rel)
                continue
        if not target.exists() or sha256_file(target) != desired_hash:
            plan.append((rel, target, desired, desired_hash))

    if conflicts:
        raise SyncConflict("Conflicting shared-clone files (not overwritten): " + ", ".join(conflicts))

    print("Sync plan:")
    if not plan:
        print("  no file copies needed")
    for rel, _, _, digest in plan:
        print(f"  {'WOULD COPY' if dry_run else 'COPY'} {rel} sha256={digest}")
    if dry_run:
        return [rel for rel, *_ in plan]

    synced_hashes = dict(previous)
    for rel, target, content, digest in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        synced_hashes[rel] = digest
    # Refresh entries for all files that already match source/desired content.
    for source in collect_source_files(root):
        rel = source.relative_to(root).as_posix()
        target = shared / rel
        if target.exists():
            synced_hashes[rel] = sha256_file(target)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"files": dict(sorted(synced_hashes.items()))}, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest updated: {manifest_path}")
    return [rel for rel, *_ in plan]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        sync(args.shared, dry_run=args.dry_run)
    except (OSError, ValueError, SyncConflict, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
