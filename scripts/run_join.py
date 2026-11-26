#!/usr/bin/env python
"""Join mutation and position feature tables into data/processed/mut_features.parquet.

Reads (all read-only):
- data/processed/mut_esm.parquet
- data/interim/mut_phys.parquet
- data/processed/pos_{structure,evolution,function}.parquet when present

Writes:
- data/processed/mut_features.parquet (ESM model name in schema metadata)
- reports/tables/nan_report_mut_features.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xynapse.esm_scoring import read_esm_model  # noqa: E402
from xynapse.join_features import (  # noqa: E402
    build_mut_features,
    save_mut_features,
)
from xynapse.schema import TABLES  # noqa: E402
from xynapse.validate import validate_table  # noqa: E402


def main() -> None:
    processed = ROOT / "data" / "processed"
    esm_path = processed / "mut_esm.parquet"
    if not esm_path.exists():
        sys.exit(f"ERROR: {esm_path} not found")

    esm_model = read_esm_model(esm_path)
    mut_esm = pd.read_parquet(esm_path)

    phys_path = ROOT / "data" / "interim" / "mut_phys.parquet"
    mut_phys = pd.read_parquet(phys_path) if phys_path.exists() else None
    if mut_phys is None:
        print(f"WARNING: {phys_path.relative_to(ROOT)} not found; physchem stays NaN")

    pos_tables = {}
    for name in ("pos_structure", "pos_evolution", "pos_function"):
        path = processed / f"{name}.parquet"
        if path.exists():
            pos_tables[name] = pd.read_parquet(path)
        else:
            print(f"WARNING: {path.name} not found; its columns stay NaN")

    if "pos_evolution" not in pos_tables:
        print("WARNING: msa_logodds is absent; writing NaN (no imputation)")

    features, report = build_mut_features(
        mut_esm,
        mut_phys=mut_phys,
        pos_tables=pos_tables,
        esm_model=esm_model,
    )

    out_path = processed / "mut_features.parquet"
    save_mut_features(features, out_path, esm_model)

    nan_path = ROOT / "reports" / "tables" / "nan_report_mut_features.csv"
    nan_path.parent.mkdir(parents=True, exist_ok=True)
    report["nan_report"].to_csv(nan_path, index=False)

    print(f"esm model: {esm_model}")
    print(f"rows:      {len(features)}")
    print(f"proteins:  {features['protein_id'].nunique()}")
    print(f"columns:   {len(features.columns)}")
    print(f"output:    {out_path.relative_to(ROOT)}")
    print(f"nan report:{nan_path.relative_to(ROOT)}")
    for name, ids in report["ignored_proteins"].items():
        print(f"ignored {name} proteins not in mut_esm: {ids}")
    if not report["ignored_proteins"]:
        print("ignored proteins: none")

    proteins = pd.read_parquet(processed / "proteins.parquet")
    problems = validate_table(features, "mut_features", proteins)
    if problems:
        print("VALIDATION: FAIL")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("VALIDATION: PASS (mut_features)")


if __name__ == "__main__":
    main()
