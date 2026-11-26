#!/usr/bin/env python
"""Check every known table in data/processed against the data contract.
Usage: python scripts/validate_tables.py [--dir data/processed]
Exit code 0 = all known tables passed."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from xynapse.schema import TABLES  # noqa: E402
from xynapse.validate import validate_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed")
    a = ap.parse_args()
    folder = Path(a.dir)
    files = sorted(folder.glob("*.parquet"))
    if not files:
        print(f"No .parquet files in {folder}")
        sys.exit(1)

    proteins = None
    if (folder / "proteins.parquet").exists():
        proteins = pd.read_parquet(folder / "proteins.parquet")
    else:
        print("NOTE: proteins.parquet not found, so wt letters and row counts cannot be checked.\n")

    failed = 0
    for f in files:
        name = f.stem
        if name not in TABLES:
            print(f"[SKIP] {f.name}: not in the data contract (extra file, fine)")
            continue
        problems = validate_table(pd.read_parquet(f), name, proteins)
        if problems:
            failed += 1
            print(f"[FAIL] {f.name}")
            for p in problems:
                print(f"       - {p}")
        else:
            print(f"[PASS] {f.name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
