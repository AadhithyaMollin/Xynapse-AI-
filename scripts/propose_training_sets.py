#!/usr/bin/env python3
"""Deterministically propose stability domains from the saved Phase 2 audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def propose_stability_domains(summary_path: Path, output_path: Path, maximum: int = 60, seed: int = 42) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    required = {"domain", "length", "n_singles", "n_reliable", "numbering_match_rate", "n_wt",
                "DMS_group", "appears_in_dataset1", "wt_reference_reliable", "wt_deltaG_reference",
                "numbering_verified"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Domain summary missing required columns: {sorted(missing)}")
    eligible = summary[
        summary["appears_in_dataset1"].astype(bool)
        & summary["numbering_verified"].astype(bool)
        & summary["wt_reference_reliable"].astype(bool)
        & summary["n_reliable"].ge(100)
    ].copy()
    if eligible["domain"].duplicated().any():
        raise ValueError("Eligible domain identifiers are not unique")
    if len(eligible) < 10:
        raise ValueError(f"Fewer than 10 eligible stability domains: {len(eligible)}")
    selected = eligible.sample(n=min(maximum, len(eligible)), random_state=seed)
    numbering_path = summary_path.with_name("megascale_numbering_check.csv")
    numbering = pd.read_csv(numbering_path)
    selected = selected.merge(numbering[["domain", "constant_offset", "offset_match_rate"]],
                              on="domain", how="left", validate="one_to_one")
    selected = selected.sort_values("domain").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/tables/megascale_domain_summary.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/tables/proposed_training_ddg_domains.csv")
    parser.add_argument("--maximum", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selected = propose_stability_domains(args.summary, args.out, args.maximum, args.seed)
    print(f"Eligible stability domains: {len(pd.read_csv(args.summary).query('appears_in_dataset1 == True and numbering_verified == True and wt_reference_reliable == True and n_reliable >= 100'))}")
    print(f"Proposed stability training domains: {len(selected)}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
