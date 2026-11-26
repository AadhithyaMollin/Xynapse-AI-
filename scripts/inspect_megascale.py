#!/usr/bin/env python3
"""Chunked inspection and label-precheck for the Tsuboyama Dataset 1 CSV."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
USECOLS = ["name", "deltaG", "deltaG_95CI", "deltaG_95CI_high", "deltaG_95CI_low"]
SINGLE_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
WT_TOKENS = {"wt", "wildtype", "wild_type", "wild-type"}
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def mutation_token(name: str) -> tuple[str, str, int, str] | None:
    """Parse a substitution from the right-most underscore-delimited token."""
    domain, separator, token = name.rpartition("_")
    if not separator:
        return None
    match = SINGLE_RE.fullmatch(token)
    if not match:
        return None
    return domain, match.group(1), int(match.group(2)), match.group(3)


def name_class(name: str, known_domains: set[str]) -> str:
    if name in known_domains:
        return "wild_type_like"
    parsed = mutation_token(name)
    if parsed is not None:
        return "single_substitution"
    domain, separator, token = name.rpartition("_")
    if separator and domain in known_domains and token.lower() in WT_TOKENS:
        return "wild_type_like"
    return "other"


def is_reliable(delta_g: float, ci: float, ci_max: float = 0.5,
                dg_min: float = -1.0, dg_max: float = 5.0) -> bool:
    return bool(pd.notna(delta_g) and pd.notna(ci) and ci <= ci_max and dg_min <= delta_g <= dg_max)


def numbering_rates(records: list[tuple[str, int, str]], sequence: str) -> tuple[float, int | None]:
    """Return exact-numbering rate and the best constant offset (reported - actual)."""
    if not records:
        return 0.0, None
    exact = sum(1 for wt, position, _ in records if 1 <= position <= len(sequence) and sequence[position - 1] == wt)
    exact_rate = exact / len(records)
    offset_counts: Counter[int] = Counter()
    possibilities: list[set[int]] = []
    for wt, position, _ in records:
        offsets = {position - (index + 1) for index, aa in enumerate(sequence) if aa == wt}
        possibilities.append(offsets)
        offset_counts.update(offsets)
    best_offset = None
    best_count = 0
    for offset, _ in offset_counts.most_common():
        count = sum(offset in offsets for offsets in possibilities)
        if count > best_count:
            best_count, best_offset = count, offset
    return exact_rate, best_offset


def _classify_name(name: str, domain_names: set[str]) -> str:
    return name_class(name, domain_names)


def inspect(dataset_path: Path, single_path: Path, output_dir: Path,
            chunksize: int = 100_000) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    singles = pd.read_csv(single_path, usecols=["name", "aa_seq", "DMS_group"], dtype={"name": str, "aa_seq": str})
    if singles["name"].duplicated().any():
        raise ValueError("Single_DMS_list.csv has duplicate domain names")
    domains = {row.name: {"sequence": row.aa_seq, "DMS_group": row.DMS_group} for row in singles.itertuples(index=False)}
    domain_names = set(domains)
    census: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    seen_domains: set[str] = set()
    numbering_records: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    reliable_counts: Counter[str] = Counter()
    total_single_counts: Counter[str] = Counter()
    wt_rows: dict[str, list[float]] = defaultdict(list)
    single_records: dict[str, list[tuple[str, int, str, float, float]]] = defaultdict(list)

    chunks = pd.read_csv(dataset_path, usecols=USECOLS, chunksize=chunksize, low_memory=False)
    for frame in chunks:
        for row in frame.itertuples(index=False):
            name = str(row.name)
            category = _classify_name(name, domain_names)
            census[category] += 1
            if len(examples[category]) < 3 and name not in examples[category]:
                examples[category].append(name)
            if name in domain_names:
                seen_domains.add(name)
                if is_reliable(row.deltaG, row.deltaG_95CI):
                    wt_rows[name].append(float(row.deltaG))
                continue
            parsed = mutation_token(name)
            if parsed is None:
                prefix, separator, token = name.rpartition("_")
                if separator and prefix in domain_names and token.lower() in WT_TOKENS:
                    seen_domains.add(prefix)
                    if is_reliable(row.deltaG, row.deltaG_95CI):
                        wt_rows[prefix].append(float(row.deltaG))
                continue
            domain, wt, position, mut = parsed
            if domain not in domain_names:
                continue
            seen_domains.add(domain)
            numbering_records[domain].append((wt, position, mut))
            total_single_counts[domain] += 1

            mutation_delta = float(row.deltaG) if pd.notna(row.deltaG) else float("nan")
            mutation_ci = float(row.deltaG_95CI) if pd.notna(row.deltaG_95CI) else float("nan")
            if is_reliable(mutation_delta, mutation_ci):
                reliable_counts[domain] += 1
            single_records[domain].append((wt, position, mut, mutation_delta, mutation_ci))

    numbering_rows = []
    summary_rows = []
    sign_rows = []
    sign_values: list[float] = []
    for domain in sorted(domain_names):
        seq = domains[domain]["sequence"]
        records = numbering_records.get(domain, [])
        exact_rate, offset = numbering_rates(records, seq)
        shifted_rate = 0.0
        if records and offset is not None:
            shifted_rate = sum(
                1 for wt, position, _ in records
                if 1 <= position - offset <= len(seq) and seq[position - offset - 1] == wt
            ) / len(records)
        verified = exact_rate >= 0.99
        corrected_offset = None
        if not verified and shifted_rate >= 0.99:
            verified = True
            corrected_offset = offset
        numbering_rows.append({
            "domain": domain, "n_singles": len(records), "numbering_match_rate": exact_rate,
            "constant_offset": corrected_offset, "offset_match_rate": shifted_rate,
            "numbering_verified": verified,
        })
        wt_values = wt_rows.get(domain, [])
        n_wt = len(wt_values)
        wt_reference = float(pd.Series(wt_values, dtype="float64").median()) if n_wt else float("nan")
        summary_rows.append({
            "domain": domain, "length": len(seq), "n_singles": total_single_counts[domain],
            "n_reliable": reliable_counts[domain] if n_wt else 0,
            "numbering_match_rate": exact_rate, "n_wt": n_wt,
            "DMS_group": domains[domain]["DMS_group"], "appears_in_dataset1": domain in seen_domains,
            "wt_reference_reliable": n_wt > 0, "wt_deltaG_reference": wt_reference,
            "numbering_verified": verified,
        })
        if n_wt and verified:
            proline_values = [
                wt_reference - delta_g
                for wt, position, mut, delta_g, ci in single_records.get(domain, [])
                if mut == "P" and is_reliable(delta_g, ci)
            ]
            if proline_values:
                sign_values.extend(proline_values)
                sign_rows.append({"domain": domain, "n_x_to_proline": len(proline_values),
                                  "mean_project_ddg": sum(proline_values) / len(proline_values)})

    census_rows = [{"name_class": key, "count": census[key], "examples": " | ".join(examples[key])}
                   for key in ("wild_type_like", "single_substitution", "other")]
    census_path = output_dir / "megascale_name_census.csv"
    numbering_path = output_dir / "megascale_numbering_check.csv"
    summary_path = output_dir / "megascale_domain_summary.csv"
    sign_path = output_dir / "sign_check_megascale.csv"
    pd.DataFrame(census_rows).to_csv(census_path, index=False)
    pd.DataFrame(numbering_rows).to_csv(numbering_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(sign_rows).to_csv(sign_path, index=False)

    eligible_domain_ids = sorted(domain_names & seen_domains)
    pd.DataFrame({"domain": eligible_domain_ids}).to_csv(output_dir / "megascale_domains_in_single_dms.csv", index=False)
    if not sign_values or sum(sign_values) / len(sign_values) <= 0:
        raise ValueError("Proline sign check is not positive in project convention")
    return {"census": census_path, "numbering": numbering_path, "summary": summary_path, "sign": sign_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/raw/tsuboyama/Tsuboyama2023_Dataset1_20230416.csv")
    parser.add_argument("--single-dms", type=Path, default=ROOT / "data/raw/tsuboyama/Single_DMS_list.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/tables")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    try:
        outputs = inspect(args.dataset, args.single_dms, args.out, args.chunksize)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(DISCLAIMER)
    for name, path in outputs.items():
        print(f"PASS: {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
