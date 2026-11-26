import zipfile

import pandas as pd

from scripts.inspect_proteingym import (
    assess_numbering,
    find_binary_label_column,
    identify_case_study,
    inspect,
    parse_single_mutation,
)


def test_parse_single_mutation_is_strict():
    assert parse_single_mutation("A12V") == ("A", 12, "V")
    assert parse_single_mutation("A0V") == ("A", 0, "V")
    assert parse_single_mutation("A12del") is None
    assert parse_single_mutation("A12VV") is None


def test_numbering_exact_and_constant_offset():
    frame = pd.DataFrame({"mutant": ["A1V", "C2Y", "D3A"], "DMS_score_bin": [0, 1, 0]})
    assert assess_numbering(frame, "ACDE", "DMS_score_bin")[:3] == (1.0, 0, 1.0)
    offset_frame = pd.DataFrame({"mutant": ["A2V", "C3Y", "D4A"], "DMS_score_bin": [0, 1, 0]})
    exact, offset, rate, _ = assess_numbering(offset_frame, "ACDE", "DMS_score_bin")
    assert exact == 0.0
    assert offset == 1
    assert rate == 1.0


def test_case_study_detection_uses_verified_reference_names():
    assert identify_case_study(
        {"DMS_id": "P53_HUMAN_Giacomelli_2018", "DMS_filename": "", "UniProt_ID": "P53_HUMAN"},
        "ACDE",
        {"P04637": "M" * 393, "P60484": "K" * 403},
    ) == "P04637"
    assert identify_case_study(
        {"DMS_id": "other", "DMS_filename": "", "UniProt_ID": "other"},
        "M" * 393,
        {"P04637": "M" * 393},
    ) == "P04637"


def test_binary_label_detection_prefers_documented_binary_column():
    frame = pd.DataFrame({"DMS_score_bin": [0, 1], "other": [0, 1], "score": [-1.2, 0.4]})
    assert find_binary_label_column(frame) == "DMS_score_bin"
    assert find_binary_label_column(pd.DataFrame({"only": [0, 1], "score": [2.3, 3.4]})) == "only"
    assert find_binary_label_column(pd.DataFrame({"a": [0, 1], "b": [0, 1]})) is None


def test_inspect_writes_summary_and_deterministic_proposal(tmp_path):
    reference = pd.DataFrame([{
        "DMS_id": "SYNTHETIC_ASSAY", "DMS_filename": "synthetic.csv", "UniProt_ID": "SYNTHETIC",
        "target_seq": "ACDE", "seq_len": 4, "selection_assay": "synthetic assay",
        "selection_type": "Growth", "coarse_selection_type": "OrganismalFitness",
        "raw_DMS_phenotype_name": "fitness", "raw_DMS_directionality": 1,
        "raw_DMS_mutant_column": "mutant", "DMS_binarization_cutoff": 0.5,
        "DMS_binarization_method": "median", "raw_mut_offset": 0,
    }])
    ref_path = tmp_path / "reference.csv"
    reference.to_csv(ref_path, index=False)
    proteins_path = tmp_path / "proteins.parquet"
    pd.DataFrame([
        {"protein_id": "P60484", "sequence": "M" * 403},
        {"protein_id": "P04637", "sequence": "M" * 393},
    ]).to_parquet(proteins_path, index=False)
    archive_path = tmp_path / "assays.zip"
    assay = pd.DataFrame({
        "mutant": ["A1V", "A1C", "C2A", "C2D", "D3A", "D3C"],
        "DMS_score_bin": [0, 1, 0, 1, 0, 1],
    })
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("DMS_ProteinGym_substitutions/synthetic.csv", assay.to_csv(index=False))
    out = tmp_path / "out"
    paths = inspect(ref_path, archive_path, proteins_path, out, precheck=False)
    summary = pd.read_csv(paths["summary"])
    proposal = pd.read_csv(paths["proposed"])
    assert len(summary) == 1
    assert "target_seq" not in summary.columns
    assert "sequence_sha256" in summary.columns
    assert "target_seq" not in proposal.columns
    assert bool(summary.loc[0, "numbering_verified"])
    assert summary.loc[0, "n_valid_single_rows"] == 6
    assert len(proposal) == 1
    assert proposal.loc[0, "DMS_id"] == "SYNTHETIC_ASSAY"
    assert paths["precheck"].exists()
