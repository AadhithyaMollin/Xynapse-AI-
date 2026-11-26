"""Mutation generator: every possible single amino-acid substitution of a protein."""
import pandas as pd

from .schema import AA, MUT_KEYS


def generate_mutations(protein_id: str, sequence: str) -> pd.DataFrame:
    """Return one row per (position, mutant letter): 19 * len(sequence) rows.
    Positions are 1-based, matching UniProt."""
    sequence = sequence.strip().upper()
    bad = set(sequence) - set(AA)
    if bad:
        raise ValueError(f"{protein_id}: non-canonical residues {sorted(bad)}")

    rows = []
    for position, wt in enumerate(sequence, start=1):
        for mut in AA:
            if mut == wt:
                continue
            rows.append((protein_id, position, wt, mut))
    return pd.DataFrame(rows, columns=MUT_KEYS)
