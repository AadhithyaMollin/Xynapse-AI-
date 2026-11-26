#!/usr/bin/env python3
"""Parse ProteinGym's current download bundle and HEAD every discovered file."""
from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PAGE_URL = "https://proteingym.org/download"
ALLOWED_BASE_HOST = "marks.hms.harvard.edu"
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


class ScriptSrcParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            src = dict(attrs).get("src")
            if src:
                self.sources.append(src)


def inspect_downloads(page_path: Path, js_path: Path, out_csv: Path) -> tuple[pd.DataFrame, str, str]:
    page_bytes = page_path.read_bytes()
    parser = ScriptSrcParser()
    parser.feed(page_bytes.decode("utf-8"))
    if not parser.sources:
        raise ValueError("No script bundle source found in the fetched ProteinGym download page")
    js_url = urljoin(PAGE_URL, parser.sources[0])
    if urlparse(js_url).hostname != "proteingym.org":
        raise ValueError(f"Bundle host is not the approved ProteinGym host: {js_url}")
    response = requests.get(js_url, timeout=90)
    response.raise_for_status()
    js_path.parent.mkdir(parents=True, exist_ok=True)
    js_path.write_bytes(response.content)
    bundle = response.text
    base_match = re.search(r'const\s+\w+="(https://marks\.hms\.harvard\.edu/proteingym/[^\"]+)"', bundle)
    if not base_match:
        raise ValueError("ProteinGym base download URL not found in the fetched bundle")
    base = base_match.group(1)
    file_names = sorted(set(re.findall(r"\$\{\w+\}/([A-Za-z0-9_.-]+\.(?:csv|zip))", bundle)))
    if not file_names:
        raise ValueError("No candidate CSV/ZIP file names found in the fetched bundle")

    print(f"WARNING: TLS certificate verification disabled for HEAD requests to {ALLOWED_BASE_HOST} only")
    rows = []
    for file_name in file_names:
        url = f"{base}/{file_name}"
        if urlparse(url).hostname != ALLOWED_BASE_HOST:
            raise ValueError(f"Refusing HEAD outside approved TLS exception host: {url}")
        head = requests.head(url, timeout=90, allow_redirects=True, verify=False)
        for hop in [*head.history, head]:
            if urlparse(hop.url).hostname != ALLOWED_BASE_HOST:
                raise ValueError(f"Redirect left approved host for {file_name}: {hop.url}")
        head.raise_for_status()
        if "Content-Length" not in head.headers:
            raise ValueError(f"HEAD did not report Content-Length for {url}")
        rows.append({"file_name": file_name, "url": url, "status": head.status_code,
                     "content_length": int(head.headers["Content-Length"])})
    frame = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    print(DISCLAIMER)
    print(f"Page URL: {PAGE_URL}")
    print(f"Page SHA-256: {hashlib.sha256(page_bytes).hexdigest()}")
    print(f"Bundle URL: {js_url}")
    print(f"Bundle SHA-256: {hashlib.sha256(response.content).hexdigest()}")
    print(f"Base URL parsed from bundle: {base}")
    print(f"Inventory written: {out_csv}")
    for row in rows:
        print(f"{row['status']} {row['content_length']} {row['file_name']}")
    return frame, base, js_url


def main() -> int:
    try:
        inspect_downloads(
            ROOT / "data/raw/proteingym/download.html",
            ROOT / "data/raw/proteingym/proteingym_bundle.js",
            ROOT / "reports/tables/proteingym_download_inventory.csv",
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
