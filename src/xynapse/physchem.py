"""Physico-chemical mutation features from standard reference tables.

All features are mutation-level: `blosum62` is the substitution score for the
(wt, mut) pair, and every `d_*` column is feature(mut) - feature(wt).
"""
import pandas as pd
from Bio.Align import substitution_matrices
from Bio.SeqUtils.ProtParamData import kd

from .schema import AA, MUT_PHYS_COLS

BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Kyte & Doolittle (1982) hydropathy index, taken from Biopython.
HYDROPATHY = {aa: float(kd[aa]) for aa in AA}

# Zamyatnin (1972) side-chain volumes, cubic angstroms.
# [VERIFY against the original table]
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Formal side-chain charge at ~pH 7: K, R = +1; D, E = -1; all others 0.
# Histidine (side-chain pKa ~6) is simplified to 0.
CHARGE = {aa: 0 for aa in AA}
CHARGE.update({"K": 1, "R": 1, "D": -1, "E": -1})


def add_physchem(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` (needs `wt` and `mut`) with MUT_PHYS_COLS added."""
    missing = [c for c in ("wt", "mut") if c not in df.columns]
    if missing:
        raise ValueError(f"add_physchem: missing columns {missing}")

    letters = set(df["wt"]) | set(df["mut"])
    bad = letters - set(AA)
    if bad:
        raise ValueError(f"add_physchem: non-canonical residues {sorted(bad)}")

    out = df.copy()
    pairs = list(zip(out["wt"], out["mut"]))
    values = {
        "blosum62": [float(BLOSUM62[wt, mut]) for wt, mut in pairs],
        "d_hydropathy": [HYDROPATHY[mut] - HYDROPATHY[wt] for wt, mut in pairs],
        "d_volume": [VOLUME[mut] - VOLUME[wt] for wt, mut in pairs],
        "d_charge": [CHARGE[mut] - CHARGE[wt] for wt, mut in pairs],
    }
    for column in MUT_PHYS_COLS:
        out[column] = values[column]
    return out
