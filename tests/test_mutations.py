import pytest
from xynapse.mutations import generate_mutations
from xynapse.schema import AA, MUT_KEYS

SEQ = "MKV"


def test_columns():
    assert list(generate_mutations("T", SEQ).columns) == MUT_KEYS


def test_row_count_is_19_times_length():
    assert len(generate_mutations("T", SEQ)) == 19 * len(SEQ)  # 57


def test_no_self_mutations():
    df = generate_mutations("T", SEQ)
    assert (df["wt"] != df["mut"]).all()


def test_positions_are_one_based():
    df = generate_mutations("T", SEQ)
    assert df["position"].min() == 1 and df["position"].max() == len(SEQ)


def test_wt_matches_sequence():
    df = generate_mutations("T", SEQ)
    assert all(SEQ[p - 1] == w for p, w in zip(df["position"], df["wt"]))


def test_each_position_has_all_19_alternatives():
    df = generate_mutations("T", SEQ)
    for _, g in df.groupby("position"):
        assert len(g) == 19 and set(g["mut"]) == set(AA) - set(g["wt"])


def test_no_duplicates():
    df = generate_mutations("T", SEQ)
    assert not df.duplicated(MUT_KEYS).any()


def test_non_canonical_letter_raises():
    with pytest.raises(ValueError):
        generate_mutations("T", "MKX")
