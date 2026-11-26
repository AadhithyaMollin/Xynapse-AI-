#!/usr/bin/env python
"""Attach physico-chemical mutation features to the ESM-2 mutation table.

Reads data/processed/mut_esm.parquet (keys only) and writes
data/interim/mut_phys.parquet.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xynapse.physchem import add_physchem  # noqa: E402
from xynapse.schema import MUT_KEYS  # noqa: E402


def main() -> None:
    source = ROOT / "data" / "processed" / "mut_esm.parquet"
    target = ROOT / "data" / "interim" / "mut_phys.parquet"

    df = pd.read_parquet(source)[MUT_KEYS]
    out = add_physchem(df)

    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)

    print(f"input:   {source.relative_to(ROOT)} ({len(df)} rows)")
    print(f"output:  {target.relative_to(ROOT)} ({len(out)} rows)")
    print(f"columns: {list(out.columns)}")


if __name__ == "__main__":
    main()
