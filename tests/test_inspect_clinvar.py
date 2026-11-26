import pytest

from scripts.inspect_clinvar import parse_protein_change, star_mapping


def test_parse_missense_three_letter_changes_only():
    assert parse_protein_change("NM_000000.1:c.1A>G (p.Ala1Val)") == ("A", 1, "V")
    assert parse_protein_change("(p.Arg175His)") == ("R", 175, "H")
    assert parse_protein_change("(p.Arg175Ter)") is None
    assert parse_protein_change("(p.Arg175=)") is None
    assert parse_protein_change("(p.Arg175Hisfs)") is None


def test_review_mapping_requires_observed_statuses_to_be_mapped():
    mapped = star_mapping({"criteria provided, single submitter", "practice guideline"})
    assert mapped == {"criteria provided, single submitter": 1, "practice guideline": 4}
    with pytest.raises(ValueError, match="Unmapped observed"):
        star_mapping({"unseen status"})
