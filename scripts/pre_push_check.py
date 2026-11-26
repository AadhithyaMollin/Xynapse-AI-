#!/usr/bin/env python3
"""Reject staged files that violate the shared-repository publication policy."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_FILE = 5 * 1024 * 1024
DEMO_FILE = 8 * 1024 * 1024
DEMO_TOTAL = 20 * 1024 * 1024
STRUCTURE_FILE = 2 * 1024 * 1024
ROOT_FILES = {
    "README.md", "requirements.txt", "requirements.lock", "pytest.ini", ".gitignore",
    ".python-version", "PROGRESS.md", "AGENT_BRIEF.md", "AGENT_BRIEF_V2.md",
    "colab_setup.sh", "local_setup.sh",
}
DIR_ROOTS = {"src", "scripts", "tests", "configs", "docs", "app", "reports", "demo_data"}
PLACEHOLDER_ROOTS = {"data/raw", "data/interim", "data/external", "data/processed", "models", "logs", "notebooks"}
TEXT_REPORT_EXT = {".md", ".txt", ".csv", ".json", ".png"}
BLOCKED_EXT = {".joblib", ".pkl", ".pt", ".bin", ".zip", ".gz", ".tar", ".tgz"}
SECRET_PATTERNS = [
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"BEGIN\s+PRIVATE\s+KEY"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?i)\b(?:api[_-]?key|secret|password)\b\s*[:=]\s*['\"]?[^\s'\"]{3,}"),
]


def allowed_path(name: str) -> tuple[bool, str]:
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or ".." in parts or not parts:
        return False, "unsafe path"
    if path.name == ".DS_Store" or "__pycache__" in parts or ".pytest_cache" in parts:
        return False, "generated/system file"
    if path.name == ".env" or path.name.startswith(".env."):
        return False, "environment file"
    if path.suffix.lower() in BLOCKED_EXT or (path.suffix.lower() == ".parquet" and not name.startswith("demo_data/")):
        return False, "forbidden extension or Parquet location"
    if name in ROOT_FILES:
        return True, "allowlisted root file"
    if parts[0] in PLACEHOLDER_ROOTS:
        if parts[0] == "models" and len(parts) >= 3 and path.suffix.lower() == ".json":
            return True, "model metadata JSON"
        if len(parts) == 2 and parts[1] == ".gitkeep":
            return True, "placeholder"
        return False, "forbidden data/model/log path"
    if parts[0] == "reports":
        if path.suffix.lower() not in TEXT_REPORT_EXT:
            return False, "reports extension not allowed"
        if path.suffix.lower() == ".parquet":
            return False, "Parquet outside demo_data"
        return True, "reports text artifact"
    if parts[0] == "demo_data":
        if path.suffix.lower() not in TEXT_REPORT_EXT | {".parquet", ".pdb", ".cif", ".mmcif"}:
            return False, "demo extension not allowed"
        if path.suffix.lower() in {".pdb", ".cif", ".mmcif"} and not name.startswith("demo_data/structures/"):
            return False, "structure outside demo_data/structures"
        return True, "demo artifact"
    if parts[0] in DIR_ROOTS:
        return True, "allowlisted directory"
    return False, "outside allowlist"


def staged_paths(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def staged_blob(repo: Path, name: str) -> bytes:
    result = subprocess.run(["git", "show", f":{name}"], cwd=repo, check=True, capture_output=True)
    return result.stdout


def _outside_readme_block(text: str) -> str:
    begin, end = "<!-- XYNAPSE-ML:BEGIN -->", "<!-- XYNAPSE-ML:END -->"
    start, stop = text.find(begin), text.find(end)
    if start < 0 or stop < start:
        return text
    return (text[:start] + text[stop + len(end):]).rstrip()


def check(repo: Path) -> tuple[list[tuple[str, bool, str]], list[str]]:
    paths = staged_paths(repo)
    outcomes: list[tuple[str, bool, str]] = []
    failures: list[str] = []

    path_errors = []
    for name in paths:
        ok, reason = allowed_path(name)
        if not ok:
            path_errors.append(f"{name}: {reason}")
    outcomes.append(("allowlist and forbidden locations/extensions", not path_errors, "; ".join(path_errors) or f"{len(paths)} staged path(s) checked"))

    readme_ok, readme_detail = True, "README not staged"
    if "README.md" in paths:
        try:
            staged_readme = staged_blob(repo, "README.md").decode("utf-8")
            begin, end = "<!-- XYNAPSE-ML:BEGIN -->", "<!-- XYNAPSE-ML:END -->"
            if begin not in staged_readme or end not in staged_readme or staged_readme.find(end) < staged_readme.find(begin):
                readme_ok, readme_detail = False, "managed block markers missing or out of order"
            else:
                base = subprocess.run(["git", "show", "HEAD:README.md"], cwd=repo, capture_output=True)
                if base.returncode == 0:
                    base_text = base.stdout.decode("utf-8")
                    readme_ok = _outside_readme_block(staged_readme) == _outside_readme_block(base_text)
                    readme_detail = "only managed block changed" if readme_ok else "content outside managed block changed"
                else:
                    readme_detail = "managed block present; no prior README in HEAD"
        except (UnicodeDecodeError, subprocess.CalledProcessError) as exc:
            readme_ok, readme_detail = False, f"could not inspect staged README ({exc})"
    outcomes.append(("README managed-block rule", readme_ok, readme_detail))

    size_errors = []
    demo_total = 0
    staged_sizes: dict[str, int] = {}
    for name in paths:
        try:
            size = len(staged_blob(repo, name))
        except subprocess.CalledProcessError:
            size_errors.append(f"{name}: staged blob unavailable")
            continue
        staged_sizes[name] = size
        if name.startswith("demo_data/"):
            demo_total += size
            cap = STRUCTURE_FILE if name.startswith("demo_data/structures/") else DEMO_FILE
            if size > cap:
                size_errors.append(f"{name}: {size} bytes exceeds {cap}")
        elif size > MAX_FILE:
            size_errors.append(f"{name}: {size} bytes exceeds {MAX_FILE}")
    if demo_total > DEMO_TOTAL:
        size_errors.append(f"demo_data total {demo_total} bytes exceeds {DEMO_TOTAL}")
    outcomes.append(("file-size caps", not size_errors, "; ".join(size_errors) or "staged files are within caps"))

    secret_errors = []
    for name in paths:
        try:
            content = staged_blob(repo, name)
        except subprocess.CalledProcessError as exc:
            secret_errors.append(f"{name}: cannot read staged blob ({exc})")
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                secret_errors.append(f"{name}: secret-pattern match ({pattern.pattern.decode('ascii', errors='replace')})")
                break
    outcomes.append(("secret scan", not secret_errors, "; ".join(secret_errors) or "no secret patterns found"))

    for rule, passed, detail in outcomes:
        print(f"{'PASS' if passed else 'FAIL'}: {rule} — {detail}")
        if not passed:
            failures.append(rule)
    return outcomes, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    if not (repo / ".git").exists():
        print(f"FAIL: not a Git working tree: {repo}")
        return 1
    try:
        _, failures = check(repo)
    except subprocess.CalledProcessError as exc:
        print(f"FAIL: git diff --cached failed with exit {exc.returncode}: {exc.stderr.strip()}")
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
