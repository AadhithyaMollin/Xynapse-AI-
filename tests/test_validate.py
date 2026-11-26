import pandas as pd
from xynapse.schema import AA
from xynapse.validate import validate_table

SEQ = "MKV"
PROTEINS = pd.DataFrame({
    "protein_id": ["TEST"], "uniprot_id": ["TEST"], "sequence": [SEQ],
    "length": [3], "structure_source": ["AF"], "mean_plddt": [90.0],
})


def make_mut_esm():
    rows = [("TEST", i + 1, wt, m, 0.0, 0.0, 0.0)
            for i, wt in enumerate(SEQ) for m in AA if m != wt]
    return pd.DataFrame(rows, columns=["protein_id", "position", "wt", "mut",
                                       "esm_llr", "esm_wt_logp", "esm_mut_logp"])


def test_good_table_passes():
    assert validate_table(make_mut_esm(), "mut_esm", PROTEINS) == []


def test_wrong_wt_is_caught():
    df = make_mut_esm()
    df.loc[df.position == 1, "wt"] = "A"
    assert any("wt" in p for p in validate_table(df, "mut_esm", PROTEINS))


def test_zero_based_position_is_caught():
    df = make_mut_esm()
    df["position"] -= 1
    assert any("1-based" in p for p in validate_table(df, "mut_esm", PROTEINS))


def test_missing_rows_is_caught():
    df = make_mut_esm().iloc[:-1]
    assert any("expected" in p for p in validate_table(df, "mut_esm", PROTEINS))


def test_missing_column_is_caught():
    df = make_mut_esm().drop(columns=["esm_llr"])
    assert any("missing columns" in p for p in validate_table(df, "mut_esm", PROTEINS))
