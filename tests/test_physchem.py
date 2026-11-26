import numpy as np
import pandas as pd
import pytest

from xynapse.physchem import add_physchem
from xynapse.schema import AA, MUT_PHYS_COLS

D_COLS = ["d_hydropathy", "d_volume", "d_charge"]


def features(wt, mut):
    df = pd.DataFrame({"protein_id": ["T"], "position": [1], "wt": [wt], "mut": [mut]})
    return add_physchem(df).iloc[0]


def all_pairs():
    rows = [("T", 1, a, b) for a in AA for b in AA if a != b]
    return pd.DataFrame(rows, columns=["protein_id", "position", "wt", "mut"])


def test_blosum62_known_values():
    assert features("L", "I")["blosum62"] == 2
    assert features("D", "E")["blosum62"] == 2
    assert features("A", "R")["blosum62"] == -1
    assert features("W", "G")["blosum62"] == -2


def test_hydropathy_known_value():
    assert features("I", "D")["d_hydropathy"] == pytest.approx(-8.0)


def test_volume_known_value():
    assert features("G", "W")["d_volume"] == pytest.approx(167.7, abs=0.05)


def test_charge_known_value():
    assert features("D", "K")["d_charge"] == 2


def test_antisymmetry_and_symmetry():
    table = add_physchem(all_pairs()).set_index(["wt", "mut"])
    for wt in AA:
        for mut in AA:
            if wt == mut:
                continue
            for column in D_COLS:
                assert table.loc[(wt, mut), column] == pytest.approx(
                    -table.loc[(mut, wt), column]
                )
            assert table.loc[(wt, mut), "blosum62"] == pytest.approx(
                table.loc[(mut, wt), "blosum62"]
            )


def test_no_nans_for_all_pairs():
    out = add_physchem(all_pairs())
    assert len(out) == 20 * 19
    assert list(out.columns)[-4:] == MUT_PHYS_COLS
    assert not out[MUT_PHYS_COLS].isna().any().any()
    assert np.isfinite(out[MUT_PHYS_COLS].to_numpy()).all()


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        add_physchem(pd.DataFrame({"wt": ["A"]}))


def test_non_canonical_letter_raises():
    with pytest.raises(ValueError, match="non-canonical"):
        add_physchem(pd.DataFrame({"wt": ["A"], "mut": ["X"]}))
