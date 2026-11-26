import pandas as pd

from scripts.propose_training_sets import propose_stability_domains


def test_proposal_filters_and_is_deterministic(tmp_path):
    summary_path = tmp_path / "megascale_domain_summary.csv"
    numbering_path = tmp_path / "megascale_numbering_check.csv"
    rows = []
    numbering = []
    for i in range(12):
        rows.append({
            "domain": f"domain_{i}", "length": 50 + i, "n_singles": 200,
            "n_reliable": 120 if i < 11 else 99, "numbering_match_rate": 1.0,
            "n_wt": 1, "DMS_group": "G0", "appears_in_dataset1": True,
            "wt_reference_reliable": True, "wt_deltaG_reference": 2.0,
            "numbering_verified": True,
        })
        numbering.append({"domain": f"domain_{i}", "constant_offset": 0,
                          "offset_match_rate": 1.0})
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(numbering).to_csv(numbering_path, index=False)
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    first = propose_stability_domains(summary_path, out_a, maximum=5, seed=42)
    second = propose_stability_domains(summary_path, out_b, maximum=5, seed=42)
    assert first["domain"].tolist() == second["domain"].tolist()
    assert len(first) == 5
    assert first["n_reliable"].ge(100).all()
