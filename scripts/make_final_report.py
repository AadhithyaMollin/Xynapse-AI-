#!/usr/bin/env python3
"""Render the release report from saved run artifacts; does not recompute models."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render() -> str:
    state = _json("reports/STATE.json")
    stability = _json("reports/tables/phase_06_stability_summary.json")
    function = _json("reports/tables/phase_07_function_summary.json")
    evaluation = _json("reports/tables/phase_08_test_evaluation_summary.json")
    split = _json("reports/tables/phase_05_split_summary.json")
    clinvar = _json("reports/tables/clinvar_inspection_summary.json")
    labels = _json("reports/tables/phase_03_label_build_summary.json")
    verification = _json("reports/tables/phase_13_summary.json")
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    frozen = _json("models/esm2_t12_35M/FROZEN.json")

    phases = []
    for number, record in state["phases"].items():
        if number == "00":
            continue
        phases.append(
            f"| {number} | {record['status']} | `{record.get('branch') or '—'}` | "
            f"`{record.get('commit_sha') or 'pending'}` | {record.get('pr_url') or '—'} |"
        )

    deliverables = [
        ("Phase 1", "`scripts/sync_to_shared.py`, `scripts/pre_push_check.py`, `scripts/zenodo_fetch.py`, `src/xynapse/schema.py`, `configs/config.yaml`"),
        ("Phase 2", "`docs/data_cards/`, `docs/label_definitions.md`, `reports/tables/` dataset inspections and training proposals"),
        ("Phase 3", "`src/xynapse/labels.py`, `scripts/build_labels.py`, contract label tables and prevalence/sign reports"),
        ("Phase 4", "`scripts/build_features_tier1.py`, Tier-1 feature output and teammate request"),
        ("Phase 5", "`src/xynapse/splits.py`, `scripts/build_splits.py`, split table and split reports"),
        ("Phase 6", "`src/xynapse/metrics.py`, `scripts/train_stability.py`, stability model metadata and OOF/test reports"),
        ("Phase 7", "`scripts/train_function.py`, function model/calibrator metadata and OOF/test reports"),
        ("Phase 8", "`src/xynapse/freeze.py`, `models/esm2_t12_35M/FROZEN.json`, one-time evaluation registry and test reports"),
        ("Phase 9", "`src/xynapse/mechanism.py`, `src/xynapse/ranking.py`, `scripts/predict_ranked_mutations.py`, predictions and ranking report"),
        ("Phase 10", "`src/xynapse/pipeline.py`, `scripts/gates/gate_phase_10.py`"),
        ("Phase 11", "`app/app.py`, dashboard tests and gate report"),
        ("Phase 12", "`README.md` managed block, `docs/`, `demo_data/`, model cards and report/audit scripts"),
        ("Phase 13", "`scripts/verify_all.py`, `scripts/gates/gate_phase_13.py`, `reports/FINAL_REPORT.md`"),
    ]
    deliverable_lines = [
        f"| {phase} | {state['phases'][phase.split()[-1].zfill(2)]['status']} | {paths} |"
        for phase, paths in deliverables
    ]

    ddg_test = evaluation["evaluations"]["stability_cluster_test"]
    ddg_model = next(row for row in ddg_test["metrics"] if row["model"] == "tier1_full")
    function_test = evaluation["evaluations"]["function_cluster_test"]
    function_model = next(row for row in function_test["metrics"] if row["model"] == "tier1_full_calibrated")
    clinvar_test = evaluation["evaluations"]["clinvar_external_test"]["overall_metrics"]
    case_pten = evaluation["evaluations"]["case_study_dms_P60484"]
    case_tp53 = evaluation["evaluations"]["case_study_dms_P04637"]

    params = [
        ("`ranking.ddg_label_cutoff`", config["ranking"]["ddg_label_cutoff"], "kcal/mol"),
        ("`mechanism.buried_rsasa`", config["mechanism"]["buried_rsasa"], "fraction"),
        ("`confidence.plddt_low`", config["confidence"]["plddt_low"], "score"),
        ("`confidence.n_eff_low`", config["confidence"]["n_eff_low"], "count"),
        ("`reliability.ci_max`", config["reliability"]["ci_max"], "kcal/mol"),
        ("`reliability.dg_min`", config["reliability"]["dg_min"], "kcal/mol"),
        ("`reliability.dg_max`", config["reliability"]["dg_max"], "kcal/mol"),
        ("`reliability.min_singles_per_domain`", config["reliability"]["min_singles_per_domain"], "substitutions"),
        ("`sampling.stability_domains`", config["sampling"]["stability_domains"], "maximum domains"),
        ("`sampling.function_assays`", config["sampling"]["function_assays"], "maximum assays"),
        ("`sampling.seed`", config["sampling"]["seed"], "seed"),
        ("`function.max_assay_length`", config["function"]["max_assay_length"], "amino acids"),
        ("`splits.identity`", config["splits"]["identity"], "MMseqs2 identity"),
        ("`splits.coverage`", config["splits"]["coverage"], "MMseqs2 coverage"),
        ("`splits.test_fraction_of_clusters`", config["splits"]["test_fraction_of_clusters"], "cluster fraction"),
    ]
    parameter_lines = [f"| {name} | `{value}` | {unit} |" for name, value, unit in params]

    lines = [
        "# Xynapse AI — final project report",
        "",
        DISCLAIMER,
        "",
        "## Release status",
        "",
        "| Phase | Status | Branch | Recorded phase commit | Compare/PR link |",
        "|---|---|---|---|---|",
        *phases,
        "",
        "Phase statuses, branch names, SHAs, and links are read from `reports/STATE.json`. A compare URL is supplied where a pull request was not created. The human owner merges PRs; this workflow did not merge them or push to `main`.",
        "",
        "## Deliverables",
        "",
        "| Phase | Status | Deliverable paths |",
        "|---|---|---|",
        *deliverable_lines,
        "",
        "## Training data and model inventory",
        "",
        f"- Stability model: Tier-1; {stability['n_training_proteins']} proteins, {stability['n_training_rows']} rows, {stability['n_training_clusters']} training clusters; longest training sequence {stability['max_training_protein_length']} aa. Source: `reports/tables/phase_06_stability_summary.json`.",
        f"- Function model: Tier-1; {function['n_training_proteins']} proteins, {function['n_training_rows']} rows, {function['n_training_clusters']} training clusters; longest training sequence {function['max_training_protein_length']} aa. Source: `reports/tables/phase_07_function_summary.json`.",
        f"- Both model metadata records report Tier-2 unavailable; frozen model inventory uses `{frozen['esm_model']}` and short name `{frozen['esm_model_short']}`. Source: `models/esm2_t12_35M/FROZEN.json` and model metadata JSON files.",
        f"- Training inputs were selected from {labels['n_selected_stability_domains']} stability domains and {labels['n_selected_function_assays']} function assays before split-level exclusions. Source: `reports/tables/phase_03_label_build_summary.json`.",
        f"- MMseqs2 split inventory: {split['n_proteins']} proteins, {split['n_clusters']} clusters, {split['n_test_clusters']} test clusters. Source: `reports/tables/phase_05_split_summary.json`.",
        "",
        "## Saved held-out and external results",
        "",
        "These are the single recorded evaluations in `reports/test_eval_registry.json`; the final verification did not rerun them. They are reported as computational outputs and are not grounds for changing the frozen models.",
        "",
        "| Evaluation | Rows / proteins | Selected model output | Source |",
        "|---|---:|---|---|",
        f"| Stability cluster test | {ddg_test['n_rows']} / {ddg_test['n_proteins']} | Spearman {_fmt(ddg_model['spearman'])}; Pearson {_fmt(ddg_model['pearson'])}; RMSE {_fmt(ddg_model['rmse'])} kcal/mol | `reports/tables/phase_08_test_evaluation_summary.json` |",
        f"| Function cluster test | {function_test['n_rows']} / {function_test['n_proteins']} | AUROC {_fmt(function_model['auroc'])}; AUPRC {_fmt(function_model['auprc'])}; MCC {_fmt(function_model['mcc'])} | `reports/tables/phase_08_test_evaluation_summary.json` |",
        f"| PTEN/P60484 case-study assays | {case_pten['n_rows']} / {case_pten['n_assays']} assays | Per-assay rows and metrics in saved evaluation summary | `reports/tables/phase_08_test_evaluation_summary.json` |",
        f"| TP53/P04637 case-study assays | {case_tp53['n_rows']} / {case_tp53['n_assays']} assays | Per-assay rows and metrics in saved evaluation summary | `reports/tables/phase_08_test_evaluation_summary.json` |",
        f"| ClinVar external set | {clinvar_test['n_rows']} / {evaluation['evaluations']['clinvar_external_test']['n_proteins']} proteins | AUROC {_fmt(clinvar_test['auroc'])}; AUPRC {_fmt(clinvar_test['auprc'])}; MCC {_fmt(clinvar_test['mcc'])} | `reports/tables/phase_08_test_evaluation_summary.json` |",
        "",
        "## Frozen thresholds and plan parameters",
        "",
        "OOF-derived frozen cutoffs are recorded in `reports/tables/threshold_evidence.csv` and `configs/config.yaml`: `ddg_damaging` = 0.6763490078708566 kcal/mol; `ddg_stabilizing` = 0.02100014913691953 kcal/mol; `p_impact_high` = 0.5874781534295241; `p_impact_low` = null because the target NPV was not reached. Accordingly, the tolerated class and beneficial-hypothesis class are unavailable under the configured rules.",
        "",
        "| Parameter | Configured value | Units / meaning |",
        "|---|---:|---|",
        *parameter_lines,
        "",
        "## Acquisition and label caveats",
        "",
        f"- ClinVar source: {clinvar['total_rows_scanned']} rows scanned, {clinvar['retained_after_significance_review_filters_and_dedupe']} retained for the mapped case-study proteins; source archive size {clinvar['source_size_bytes']} bytes and SHA-256 `{clinvar['source_sha256']}`. Gene mapping and star mapping are in `reports/tables/clinvar_inspection_summary.json` and `docs/data_cards/clinvar.md`.",
        "- ProteinGym direction remains unresolved at the individual-assay level: the Phase 2 evidence has mixed assay signs. The owner-authorized pooled orientation was used provisionally; see `reports/tables/proteingym_direction_precheck.csv`, `docs/data_cards/proteingym.md`, and `docs/label_definitions.md`.",
        "- The ProteinGym TLS exception was limited to GET/HEAD requests to `marks.hms.harvard.edu`, without credentials; warnings and file hashes are documented in `docs/data_cards/proteingym.md`.",
        "- Tsuboyama row-name classes not assigned to known wild-type/single-substitution forms remain excluded/unresolved; see `docs/data_cards/megascale.md` and the label-build summary.",
        "- No Tier-2 model was trained. `msa_logodds` is unavailable; teammate annotation completeness and PTEN evolution-table degeneracy remain documented in `reports/teammate_request.md` and `docs/limitations.md`.",
        "",
        "## Phase 13 verification",
        "",
        f"The final gate reports {verification['passed']}/{verification['checks']} PASS and {verification['failed']} FAIL; the saved summary records {verification['held_out_test_evaluations_run']} held-out evaluations during this phase. Details: `reports/gates/phase_13_gate.txt` and `reports/gates/phase_13_verify_all.txt`.",
        "The gate checks full pytest, all known data/demo Parquet contracts, leakage and claims audits, prediction row counts, frozen hashes, pipeline reproduction, dashboard AppTest, pre-push policy, and a clean clone with no local data or model binaries. The pipeline reproduced P60484 predictions with maximum absolute differences of 0 for ddG and function probability; source: `reports/gates/phase_13_gate.txt`.",
        "",
        "## Human review and release notes",
        "",
        "- Review the provisional ProteinGym label direction before using function scores beyond the current computational prototype.",
        "- Review limitations, source cards, model cards, threshold evidence, and the five one-time evaluation records; do not retune or rerun those named test sets.",
        "- Review and merge the phase pull requests manually. No human-authored case-study prose or literature interpretation has been added; those remain for the owner/team.",
        "- Large source files and model binaries are not in the repository; sharing instructions are in `docs/SHARING_ARTIFACTS.md`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    target = ROOT / "reports/FINAL_REPORT.md"
    target.write_text(render(), encoding="utf-8")
    print(f"Wrote {target.relative_to(ROOT)} from saved run artifacts")


if __name__ == "__main__":
    main()
