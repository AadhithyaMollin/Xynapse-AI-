#!/usr/bin/env python3
"""Audit user-facing documentation for required disclaimer and restricted wording."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
FORBIDDEN = re.compile(r"\b(?:proven|validated\s+experimentally|accurate|discovered|causes)\b", re.IGNORECASE)


def _managed_readme(text: str) -> str:
    begin = "<!-- XYNAPSE-ML:BEGIN -->"
    end = "<!-- XYNAPSE-ML:END -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("README must contain exactly one managed-block boundary pair")
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    if stop <= start:
        raise ValueError("README managed-block boundaries are out of order")
    return text[start:stop]


def audit(root: Path = ROOT) -> list[tuple[str, bool, str]]:
    readme = root / "README.md"
    managed = _managed_readme(readme.read_text(encoding="utf-8"))
    targets = [("README.md managed block", managed)]
    for directory in (root / "docs", root / "app"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".py", ".txt"}:
                targets.append((path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
    checks: list[tuple[str, bool, str]] = []
    for name, text in targets:
        hits = [match.group(0) for match in FORBIDDEN.finditer(text)]
        checks.append((f"{name} has no restricted claims wording", not hits, f"matches={hits}"))
        if name.endswith(".md") or name == "README.md managed block" or name.startswith("app/"):
            present = DISCLAIMER in text
            checks.append((f"{name} contains the required disclaimer", present,
                           f"disclaimer_present={present}"))
    return checks


def main() -> int:
    checks = audit()
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name} | {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
