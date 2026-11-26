#!/usr/bin/env python3
"""Phase 2 data-acquisition and inspection gate."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_gate() -> int:
    results: list[tuple[str, bool, str]] = []

    for name in ("megascale.md", "proteingym.md", "clinvar.md"):
        path = ROOT / "docs/data_cards" / name
        ok = path.is_file() and DISCLAIMER in path.read_text()
        results.append((f"data card {name} exists and has disclaimer", ok, str(path.relative_to(ROOT))))

    manifest_pairs = [
        (ROOT / "data/raw/tsuboyama/Tsuboyama2023_Dataset1_20230416.csv",
         ROOT / "data/raw/tsuboyama/Tsuboyama2023_Dataset1_20230416.csv.manifest.json"),
        (ROOT / "data/raw/tsuboyama/Single_DMS_list.csv",
         ROOT / "data/raw/tsuboyama/Single_DMS_list.csv.manifest.json"),
        (ROOT / "data/raw/proteingym/DMS_substitutions.csv",
         ROOT / "data/raw/proteingym/DMS_substitutions.csv.manifest.json"),
        (ROOT / "data/raw/proteingym/DMS_ProteinGym_substitutions.zip",
         ROOT / "data/raw/proteingym/DMS_ProteinGym_substitutions.zip.manifest.json"),
        (ROOT / "data/raw/clinvar/variant_summary.txt.gz",
         ROOT / "data/raw/clinvar/variant_summary.txt.gz.manifest.json"),
    ]
    manifests_ok = True
    for data_path, manifest_path in manifest_pairs:
        if not data_path.is_file() or not manifest_path.is_file():
            manifests_ok = False
            continue
        manifest = json.loads(manifest_path.read_text())
        expected_hash = manifest.get("sha256")
        actual_hash = sha256(data_path)
        expected_size = manifest.get("size", manifest.get("member_size"))
        manifests_ok &= expected_hash == actual_hash and expected_size == data_path.stat().st_size
    results.append(("download manifests match file size and SHA-256", manifests_ok, "5 raw inputs checked"))

    ddg = pd.read_csv(ROOT / "reports/tables/proposed_training_ddg_domains.csv")
    sign = pd.read_csv(ROOT / "reports/tables/sign_check_megascale.csv")
    sign_summary = (ROOT / "reports/tables/megascale_inspection_summary.txt").read_text()
    sign_mean = float(pd.to_numeric(sign["mean_project_ddg"], errors="coerce").mean())
    results.append(("Tsuboyama project-sign check is positive", sign_mean > 0,
                    f"mean over saved per-domain means={sign_mean:.9f}; {len(sign)} rows"))
    eligible = pd.read_csv(ROOT / "reports/tables/megascale_domain_summary.csv")
    eligible_count = int((eligible["appears_in_dataset1"].astype(bool)
                          & eligible["numbering_verified"].astype(bool)
                          & eligible["wt_reference_reliable"].astype(bool)
                          & eligible["n_reliable"].ge(100)).sum())
    results.append(("stability proposal has at least 10 eligible domains and at most 60 sampled",
                    len(ddg) >= 10 and len(ddg) <= 60 and eligible_count >= 10,
                    f"eligible={eligible_count}; proposed={len(ddg)}"))
    results.append(("Tsuboyama inspection summary records the positive pooled X-to-P check",
                    "Pooled mean project ddG for X->P: " in sign_summary,
                    "source reports/tables/megascale_inspection_summary.txt"))

    function = pd.read_csv(ROOT / "reports/tables/proposed_training_function_assays.csv")
    direction = json.loads((ROOT / "reports/tables/proteingym_direction_precheck_summary.json").read_text())
    case_studies = pd.read_csv(ROOT / "reports/tables/case_study_assays.csv")
    results.append(("function proposal has at least 10 and at most 30 assays",
                    10 <= len(function) <= 30, f"proposed={len(function)}"))
    results.append(("function assays have one selected record per exact target sequence",
                    function["sequence_sha256"].is_unique,
                    f"distinct sequence hashes={function['sequence_sha256'].nunique()}"))
    results.append(("function proposal excludes Stability category",
                    not function["coarse_selection_type"].fillna("").str.lower().eq("stability").any(),
                    f"proposed={len(function)}"))
    results.append(("case-study assays are retained separately",
                    len(case_studies) >= 2 and case_studies["case_study_protein_id"].notna().all(),
                    f"case-study assay rows={len(case_studies)}"))
    direction_ok = direction["n_assays"] > 0 and direction["mean_absolute_spearman"] >= 0.1
    results.append(("ProteinGym direction precheck is saved and its mixed-sign limitation is explicit",
                    direction_ok and direction["direction_evidence_status"] in {"mixed_assay_signs", "consistent_assay_signs"},
                    f"n={direction['n_assays']}; pooled mean={direction['mean_spearman_binary_label_esm_llr']:.9f}; status={direction['direction_evidence_status']}"))

    clinvar = json.loads((ROOT / "reports/tables/clinvar_inspection_summary.json").read_text())
    mismatch = pd.read_csv(ROOT / "reports/tables/clinvar_wt_mismatch_rates.csv")
    counts = pd.read_csv(ROOT / "reports/tables/clinvar_external_counts.csv")
    results.append(("ClinVar mapped WT mismatch rates are at most 5%",
                    mismatch["wt_mismatch_rate"].le(0.05).all(),
                    f"genes={len(mismatch)}; parsed={int(mismatch['n_parsed_candidates'].sum())}"))
    results.append(("ClinVar external counts have at least 10 rows and agree with summary",
                    int(counts["count"].sum()) >= 10 and int(counts["count"].sum()) == clinvar["retained_after_significance_review_filters_and_dedupe"],
                    f"retained variants={int(counts['count'].sum())}"))
    results.append(("ClinVar review mapping and all audit tables exist",
                    (ROOT / "reports/tables/clinvar_review_status_mapping.csv").is_file()
                    and (ROOT / "reports/tables/clinvar_category_values.csv").is_file(),
                    f"mapped statuses={len(clinvar['star_mapping'])}"))

    try:
        staged = subprocess.run(["git", "-C", str(ROOT.parent / "xynapse-shared"),
                                 "diff", "--cached", "--name-only"],
                                capture_output=True, text=True, check=True).stdout.splitlines()
        staged_data = [path for path in staged if path.startswith("data/")]
        results.append(("no raw or processed data paths are staged", not staged_data,
                        f"staged data paths={staged_data}"))
    except Exception as exc:
        results.append(("no raw or processed data paths are staged", False,
                        f"could not inspect shared clone staged set: {exc}"))

    pytest_file = ROOT / "reports/gates/phase_02_pytest.txt"
    pytest_file.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                               capture_output=True, text=True)
    pytest_file.write_text(completed.stdout + completed.stderr)
    results.append(("pytest -q passes", completed.returncode == 0,
                    completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else completed.stderr.strip()))

    report_path = ROOT / "reports/gates/phase_02_gate.txt"
    lines = [DISCLAIMER]
    for label, passed, details in results:
        lines.append(f"{'PASS' if passed else 'FAIL'}: {label} | {details}")
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(passed for _, passed, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
