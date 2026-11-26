#!/usr/bin/env python
"""Phase 1 Stage A: read-only inspection of data/processed/*.parquet.

Writes:
  reports/tables/teammate_table_inspection.csv  (one row per column)
  reports/tables/teammate_value_checks.csv      (one row per column, value stats)

Never writes to data/. Exit code 1 if any blocking contract problem is found.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xynapse.schema import TABLES  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "reports" / "tables"
SENTINELS = (-1, 999, -999)
SENTINEL_STRINGS = {"", "NA", "N/A", "nan", "NaN", "None", "null"}

RANGES = {  # inclusive bounds, checked on non-null numeric values
    "rsasa": (0.0, 1.5),
    "sasa_abs": (0.0, None),
    "plddt": (0.0, 100.0),
    "gap_frac": (0.0, 1.0),
    "n_contacts_8A": (0.0, None),
}
POSITIVE = {"n_eff"}  # strictly positive, handled before RANGES


def load_proteins():
    path = PROCESSED / "proteins.parquet"
    if not path.exists():
        return None
    proteins = pd.read_parquet(path)
    if not {"protein_id", "sequence"} <= set(proteins.columns):
        return None
    proteins["length_from_sequence"] = proteins["sequence"].str.len()
    return proteins.set_index("protein_id")


def numeric_checks(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "min": None if values.empty else float(values.min()),
        "max": None if values.empty else float(values.max()),
        "n_unique": int(values.nunique()),
        "n_zero": int((values == 0).sum()),
        "n_at_min": int((values == values.min()).sum()),
        "n_at_max": int((values == values.max()).sum()),
        "n_sentinel_minus1": int((values == -1).sum()),
        "n_sentinel_999": int((values == 999).sum()),
        "n_sentinel_neg999": int((values == -999).sum()),
    }


def range_check(column: str, series: pd.Series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None, "all null"
    if column.startswith("is_"):
        ok = bool(values.isin([0, 1]).all())
        return ok, "binary (0/1)" if ok else f"values outside 0/1: {sorted(values.unique())[:5]}"
    if column in POSITIVE:
        ok = bool((values > 0).all())
        return ok, f"observed {values.min():.4g}..{values.max():.4g}, expected > 0"
    if column in RANGES:
        low, high = RANGES[column]
        ok = True
        if low is not None:
            ok &= bool((values >= low).all())
        if high is not None:
            ok &= bool((values <= high).all())
        return ok, f"observed {values.min():.4g}..{values.max():.4g}, expected {low}..{high}"
    return None, "no range rule"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proteins = load_proteins()
    seqs = proteins["sequence"] if proteins is not None else None

    mut_esm_path = PROCESSED / "mut_esm.parquet"
    mut_esm_ids = set()
    if mut_esm_path.exists():
        mut_esm_ids = set(pd.read_parquet(mut_esm_path, columns=["protein_id"])["protein_id"])

    inspection_rows, value_rows = [], []
    blocking = {}
    msa_logodds_tables = []

    for path in sorted(PROCESSED.glob("*.parquet")):
        name = path.stem
        df = pd.read_parquet(path)
        in_contract = name in TABLES
        spec = TABLES.get(name, {})
        level = spec.get("level", "n/a")
        required = list(spec.get("required", []))
        extra = [c for c in df.columns if c not in required] if in_contract else []

        if "msa_logodds" in df.columns:
            msa_logodds_tables.append(name)

        ids = sorted(df["protein_id"].dropna().unique().tolist()) if "protein_id" in df.columns else []
        n_proteins = len(ids)

        problems = []
        if in_contract:
            missing = [c for c in required if c not in df.columns]
            if missing:
                problems.append(f"missing required columns {missing}")
        else:
            missing = []
            print(f"  NOTE: '{name}' is not a contract table (extra file, ignored)")

        keys = list(spec.get("keys", []))
        dup_keys = None
        if keys and set(keys) <= set(df.columns):
            dup_keys = int(df.duplicated(keys).sum())
            if dup_keys:
                problems.append(f"{dup_keys} duplicate rows for keys {keys}")

        wt_present = "wt" in df.columns
        wt_mismatches = None
        if wt_present and seqs is not None:
            wt_mismatches = 0
            for pid, group in df.groupby("protein_id"):
                if pid not in seqs.index:
                    problems.append(f"protein_id {pid} not in proteins.parquet")
                    continue
                sequence = seqs[pid]
                expected = [
                    sequence[p - 1] if 1 <= p <= len(sequence) else None
                    for p in group["position"]
                ]
                wt_mismatches += int(
                    (group["wt"].astype(str).to_numpy() != pd.Series(expected).astype(str).to_numpy()).sum()
                )
            if wt_mismatches:
                problems.append(f"{wt_mismatches} wt letters do not match proteins.parquet")

        position_min = int(df["position"].min()) if "position" in df.columns else None
        position_max = int(df["position"].max()) if "position" in df.columns else None
        if "position" in df.columns and (df["position"] < 1).any():
            problems.append(f"{int((df['position'] < 1).sum())} positions < 1")

        rows_per_protein_ok = None
        if level == "position" and seqs is not None:
            rows_per_protein_ok = True
            for pid, group in df.groupby("protein_id"):
                if pid not in seqs.index:
                    rows_per_protein_ok = False
                    continue
                expected_rows = len(seqs[pid])
                if len(group) != expected_rows:
                    rows_per_protein_ok = False
                    problems.append(
                        f"{pid}: {len(group)} rows, expected L={expected_rows}"
                    )

        if "protein_id" in df.columns:
            unknown = sorted(set(ids) - set(seqs.index)) if seqs is not None else []
            if unknown:
                problems.append(f"protein_id not in proteins.parquet: {unknown}")

        all_null = []
        for column in df.columns:
            series = df[column]
            null_count = int(series.isna().sum())
            if null_count == len(df):
                all_null.append(column)
            is_numeric = pd.api.types.is_numeric_dtype(series)
            stats = numeric_checks(series) if is_numeric else {}
            range_ok, range_note = range_check(column, series) if is_numeric else (None, "non-numeric")
            sentinel_hits = (
                stats.get("n_sentinel_minus1", 0)
                + stats.get("n_sentinel_999", 0)
                + stats.get("n_sentinel_neg999", 0)
                if is_numeric else 0
            )
            empty_strings = 0
            if not is_numeric:
                non_null = series.dropna().astype(str).str.strip()
                empty_strings = int(non_null.isin(SENTINEL_STRINGS).sum())
            if sentinel_hits:
                problems.append(f"{column}: {sentinel_hits} sentinel values (-1/999/-999)")
            if empty_strings:
                problems.append(f"{column}: {empty_strings} empty/placeholder strings")

            inspection_rows.append({
                "file": path.name, "table": name, "in_contract": in_contract,
                "level": level, "n_rows": len(df), "n_proteins": n_proteins,
                "protein_ids": "|".join(ids), "column": column,
                "dtype": str(series.dtype), "null_count": null_count,
                "null_pct": round(100.0 * null_count / len(df), 4) if len(df) else None,
                "required": column in required, "extra": column in extra,
                "all_null": null_count == len(df),
                "position_min": position_min, "position_max": position_max,
                "key_columns": "|".join(keys), "dup_key_rows": dup_keys,
                "wt_present": wt_present,
                "wt_mismatches": wt_mismatches,
                "rows_per_protein_ok": rows_per_protein_ok,
                "blocking_reasons": " ; ".join(problems),
            })
            value_rows.append({
                "file": path.name, "table": name, "column": column,
                "dtype": str(series.dtype), "n_rows": len(df),
                "n_null": null_count, **stats,
                "n_empty_strings": empty_strings,
                "range_ok": range_ok, "range_note": range_note,
            })

        blocking[name] = problems

    pd.DataFrame(inspection_rows).to_csv(OUT_DIR / "teammate_table_inspection.csv", index=False)
    pd.DataFrame(value_rows).to_csv(OUT_DIR / "teammate_value_checks.csv", index=False)

    print(f"wrote {OUT_DIR / 'teammate_table_inspection.csv'} ({len(inspection_rows)} rows)")
    print(f"wrote {OUT_DIR / 'teammate_value_checks.csv'} ({len(value_rows)} rows)")
    print(f"msa_logodds present in: {msa_logodds_tables or 'none'}")

    any_blocking = False
    print("\nPer-table verdicts:")
    for name, problems in blocking.items():
        status = "BLOCK" if problems else "OK"
        if problems:
            any_blocking = True
        print(f"  [{status}] {name}: " + ("; ".join(problems) if problems else "no blocking problem"))
        for column_row in [r for r in inspection_rows if r["table"] == name and r["all_null"]]:
            print(f"           WARNING all-null column: {column_row['column']}")

    print("\nprotein_id sets:")
    for name in blocking:
        ids = next((r["protein_ids"] for r in inspection_rows if r["table"] == name), "")
        print(f"  {name}: {ids}")
    print(f"mut_esm protein_id set: {'|'.join(sorted(mut_esm_ids))}")
    print(f"ids in mut_esm only: {sorted(mut_esm_ids - set().union(*[set(r['protein_ids'].split('|')) for r in inspection_rows])) if inspection_rows else []}")

    sys.exit(1 if any_blocking else 0)


if __name__ == "__main__":
    main()
