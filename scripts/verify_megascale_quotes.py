#!/usr/bin/env python3
"""Verify requested sign and reliability quotations against the fetched bioRxiv HTML."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/tsuboyama/biorxiv_v1.full.html"
OUT = ROOT / "reports/tables/megascale_quote_verification.txt"
URL = "https://www.biorxiv.org/content/10.1101/2022.12.06.519132v1.full"
QUOTES = {
    "source_ddg_direction": "All ΔΔG measurements are in the original wild-type background; positive ΔΔG indicates stabilizing substitutions.",
    "pross_ddg_direction": "ΔΔG indicates the stability of the PROSS design (ΔG) minus the stability of the original wild-type sequence; positive ΔΔG indicates the design stabilized the domain.",
    "reliability_ci": "black corner slashes indicate lower confidence ΔG estimates, (95% confidence interval > 0.5 kcal/mol), including ΔG estimates near the edges of the dynamic range.",
    "reliability_range": "All folding stability data were clipped between from -1 to 5 (kcal/mol) because the folding stability outside the dynamic range is not reliable, and then the average of the stability for 20 amino acids for each site was subtracted from the data.",
}
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def verify(text: str) -> dict[str, bool]:
    decoded = unescape(text)
    return {key: quote in decoded for key, quote in QUOTES.items()}


def main() -> int:
    raw = SOURCE.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    result = verify(text)
    lines = [DISCLAIMER, f"Source URL: {URL}", f"Retrieved date: {datetime.now(timezone.utc).date().isoformat()}",
             f"Source bytes: {len(raw)}", f"Source SHA-256: {hashlib.sha256(raw).hexdigest()}",
             "Resolution: the supplied range wording is a clause in the longer exact sentence quoted below."]
    for key, quote in QUOTES.items():
        lines.append(f"{'PASS' if result[key] else 'FAIL'}: {key}")
        lines.append(f'  "{quote}"')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"), end="")
    return 0 if all(result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
