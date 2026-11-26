from __future__ import annotations

import pandas as pd

from scripts.inspect_megascale import inspect, is_reliable, mutation_token, name_class, numbering_rates


def test_mutation_parser_uses_rightmost_token():
    assert mutation_token("EA|run2_0325_0005.pdb_A45D") == ("EA|run2_0325_0005.pdb", "A", 45, "D")
    assert mutation_token("domain_with_under_scores_R7P") == ("domain_with_under_scores", "R", 7, "P")
    assert mutation_token("domain_with_under_scores") is None


def test_name_census_classification_uses_verified_domains():
    domains = {"domain_with_under_scores"}
    assert name_class("domain_with_under_scores", domains) == "wild_type_like"
    assert name_class("domain_with_under_scores_WT", domains) == "wild_type_like"
    assert name_class("domain_with_under_scores_R7P", domains) == "single_substitution"
    assert name_class("unmapped_thing", domains) == "other"


def test_reliability_bounds_are_inclusive():
    assert is_reliable(-1.0, 0.5)
    assert is_reliable(5.0, 0.0)
    assert not is_reliable(5.01, 0.1)
    assert not is_reliable(2.0, 0.5001)
    assert not is_reliable(float("nan"), 0.1)


def test_numbering_checks_exact_and_constant_offset():
    sequence = "ACDE"
    exact = [("A", 1, "P"), ("C", 2, "D")]
    assert numbering_rates(exact, sequence) == (1.0, 0)
    shifted = [("A", 2, "P"), ("C", 3, "D")]
    exact_rate, offset = numbering_rates(shifted, sequence)
    assert exact_rate == 0.0
    assert offset == 1


def test_inspection_computes_project_sign_and_writes_tables(tmp_path):
    single_path = tmp_path / "single.csv"
    dataset_path = tmp_path / "dataset.csv"
    out = tmp_path / "out"
    pd.DataFrame([
        {"name": "D_1", "aa_seq": "A", "DMS_group": "G0"},
    ]).to_csv(single_path, index=False)
    pd.DataFrame([
        {"name": "D_1", "deltaG": 3.0, "deltaG_95CI": 0.1, "deltaG_95CI_high": 3.1,
         "deltaG_95CI_low": 2.9, "dna_seq": "excluded"},
        {"name": "D_1_A1P", "deltaG": 1.0, "deltaG_95CI": 0.2, "deltaG_95CI_high": 1.1,
         "deltaG_95CI_low": 0.9, "dna_seq": "excluded"},
        {"name": "D_1_A1C", "deltaG": 4.0, "deltaG_95CI": 0.2, "deltaG_95CI_high": 4.1,
         "deltaG_95CI_low": 3.9, "dna_seq": "excluded"},
    ]).to_csv(dataset_path, index=False)

    outputs = inspect(dataset_path, single_path, out, chunksize=1)
    assert set(outputs) == {"census", "numbering", "summary", "sign"}
    summary = pd.read_csv(outputs["summary"]).iloc[0]
    sign = pd.read_csv(outputs["sign"]).iloc[0]
    assert summary["numbering_verified"]
    assert summary["n_wt"] == 1
    assert summary["n_reliable"] == 2
    assert sign["mean_project_ddg"] == 2.0
