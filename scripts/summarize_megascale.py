#!/usr/bin/env python3
"""Create a compact, source-backed summary from saved Mega-scale inspection tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def main() -> None:
    table_dir = ROOT / "reports/tables"
    census = pd.read_csv(table_dir / "megascale_name_census.csv")
    numbering = pd.read_csv(table_dir / "megascale_numbering_check.csv")
    summary = pd.read_csv(table_dir / "megascale_domain_summary.csv")
    sign = pd.read_csv(table_dir / "sign_check_megascale.csv")
    eligible = summary[
        summary["appears_in_dataset1"]
        & summary["numbering_verified"]
        & summary["wt_reference_reliable"]
        & (summary["n_reliable"] >= 100)
    ]
    eligible_sign = sign[sign["domain"].isin(set(eligible["domain"]))]
    n_proline = int(eligible_sign["n_x_to_proline"].sum()) if not eligible_sign.empty else 0
    mean_project = (
        float((eligible_sign["mean_project_ddg"] * eligible_sign["n_x_to_proline"]).sum() / n_proline)
        if n_proline else float("nan")
    )
    failed = numbering.loc[~numbering["numbering_verified"], "domain"].astype(str).tolist()
    lines = [
        DISCLAIMER,
        f"Dataset rows classified: {int(census['count'].sum())}",
        "Name census:",
    ]
    lines.extend(f"  {row['name_class']}: {int(row['count'])}; examples={row['examples']}" for row in census.to_dict("records"))
    lines.extend([
        f"Single_DMS domains listed: {len(summary)}",
        f"Domains appearing in Dataset 1: {int(summary['appears_in_dataset1'].sum())}",
        f"Domains with reliable WT reference: {int(summary['wt_reference_reliable'].sum())}",
        f"Domains with verified numbering (exact or constant offset): {int(numbering['numbering_verified'].sum())}",
        f"Domains failing numbering: {len(failed)}; examples={' | '.join(failed[:5])}",
        f"Eligible domains with at least 100 reliable singles: {len(eligible)}",
        f"Reliable X->P substitutions in eligible domains: {n_proline}",
        f"Pooled mean project ddG for X->P: {mean_project:.8f} kcal/mol",
    ])
    out = table_dir / "megascale_inspection_summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
