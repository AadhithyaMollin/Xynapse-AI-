import numpy as np
import pandas as pd
import pytest

from xynapse.esm_scoring import read_esm_model
from xynapse.join_features import build_mut_features, save_mut_features
from xynapse.schema import (
    AA,
    MUT_ESM_COLS,
    MUT_KEYS,
    MUT_PHYS_COLS,
    POS_EVO_COLS,
    POS_FUNC_COLS,
    POS_STRUCT_COLS,
    TABLES,
)

SEQ = "MKV"
PID = "TEST"


def make_mut_esm():
    rows = [
        (PID, i + 1, wt, mut, float(i), float(i), float(i))
        for i, wt in enumerate(SEQ)
        for mut in AA
        if mut != wt
    ]
    return pd.DataFrame(rows, columns=MUT_KEYS + MUT_ESM_COLS)


def make_mut_phys():
    df = make_mut_esm()[MUT_KEYS].copy()
    for j, column in enumerate(MUT_PHYS_COLS):
        df[column] = float(j)
    return df


def make_pos_structure():
    return pd.DataFrame({
        "protein_id": [PID] * 3,
        "position": [1, 2, 3],
        "wt": list(SEQ),
        "rsasa": [0.1, 0.2, 0.3],
        "sasa_abs": [10.0, 20.0, 30.0],
        "ss8": ["H", "E", "-"],
        "ss3": ["H", "E", "C"],
        "n_contacts_8A": [4, 5, 6],
        "plddt": [80.0, 90.0, 70.0],
    })


def make_pos_evolution():
    return pd.DataFrame({
        "protein_id": [PID] * 3,
        "position": [1, 2, 3],
        "wt": list(SEQ),
        **{column: [0.1, 0.2, 0.3] for column in POS_EVO_COLS},
    })


def make_pos_function():
    df = pd.DataFrame({
        "protein_id": [PID] * 3,
        "position": [1, 2, 3],
        "wt": list(SEQ),
        "dist_to_nearest_site": [np.nan, np.nan, np.nan],
    })
    for column in POS_FUNC_COLS:
        if column not in df.columns:
            df[column] = [0, 1, 0]
    return df[["protein_id", "position", "wt"] + POS_FUNC_COLS]


def build(**overrides):
    tables = {
        "pos_structure": make_pos_structure(),
        "pos_evolution": make_pos_evolution(),
        "pos_function": make_pos_function(),
    }
    tables.update(overrides)
    return build_mut_features(
        make_mut_esm(), mut_phys=make_mut_phys(), pos_tables=tables,
        esm_model="facebook/esm2_t12_35M_UR50D",
    )


def test_row_count_and_column_order():
    out, report = build()
    assert len(out) == 19 * len(SEQ)
    assert list(out.columns) == TABLES["mut_features"]["required"]
    assert report["ignored_proteins"] == {}


def test_duplicate_position_keys_raise():
    pos = pd.concat([make_pos_structure(), make_pos_structure().iloc[[0]]],
                    ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build(pos_structure=pos)


def test_duplicate_mutation_keys_raise():
    mut = pd.concat([make_mut_esm(), make_mut_esm().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_mut_features(mut, pos_tables={})


def test_wt_mismatch_yields_nan_never_wrong_values():
    pos = make_pos_structure()
    pos.loc[pos["position"] == 2, "wt"] = "A"  # wrong wt -> no join match
    out, _ = build(pos_structure=pos)

    at2 = out[out["position"] == 2]
    assert at2["rsasa"].isna().all()
    assert at2["plddt"].isna().all()
    at1 = out[out["position"] == 1]
    assert at1["rsasa"].notna().all()
    assert set(at1["rsasa"]) == {0.1}


def test_nan_is_preserved():
    pos = make_pos_structure()
    pos.loc[pos["position"] == 1, "rsasa"] = np.nan
    out, _ = build(pos_structure=pos)
    assert out.loc[out["position"] == 1, "rsasa"].isna().all()
    assert out.loc[out["position"] == 2, "rsasa"].notna().all()


def test_msa_logodds_added_as_nan():
    out, _ = build()
    assert out["msa_logodds"].isna().all()


def test_extra_proteins_are_ignored_and_reported():
    pos = pd.concat([make_pos_structure(), pd.DataFrame({
        "protein_id": ["OTHER"], "position": [1], "wt": ["M"],
        "rsasa": [0.5], "sasa_abs": [1.0], "ss8": ["H"], "ss3": ["H"],
        "n_contacts_8A": [1], "plddt": [50.0],
    })], ignore_index=True)
    out, report = build(pos_structure=pos)
    assert "OTHER" not in set(out["protein_id"])
    assert report["ignored_proteins"] == {"pos_structure": ["OTHER"]}
    assert len(out) == 19 * len(SEQ)


def test_missing_wt_column_raises():
    pos = make_pos_structure().drop(columns=["wt"])
    with pytest.raises(ValueError, match="missing columns"):
        build(pos_structure=pos)


def test_metadata_round_trip(tmp_path):
    out, _ = build()
    path = tmp_path / "mut_features.parquet"
    save_mut_features(out, path, "facebook/esm2_t12_35M_UR50D")
    assert read_esm_model(path) == "facebook/esm2_t12_35M_UR50D"
    assert len(pd.read_parquet(path)) == len(out)
