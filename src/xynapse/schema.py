"""Data contract shared by all modules. Change only via team agreement (and update tests)."""
AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical amino acids

# One row per (protein, position, mutant aa) -> 19 * L rows per protein
MUT_KEYS = ["protein_id", "position", "wt", "mut"]  # position is 1-based
# One row per (protein, position)
POS_KEYS = ["protein_id", "position", "wt"]

PROTEIN_COLS = ["protein_id", "uniprot_id", "sequence", "length", "structure_source", "mean_plddt"]
POS_STRUCT_COLS = ["rsasa", "sasa_abs", "ss8", "ss3", "n_contacts_8A", "plddt"]
POS_EVO_COLS = ["cons_entropy", "cons_weighted", "gap_frac", "n_eff"]
POS_FUNC_COLS = ["is_active_site", "is_binding_site", "is_metal", "is_dna_binding",
                 "is_disulfide", "is_interface", "dist_to_nearest_site"]
MUT_ESM_COLS = ["esm_llr", "esm_wt_logp", "esm_mut_logp"]
MUT_EVO_COLS = ["msa_logodds"]
MUT_PHYS_COLS = ["blosum62", "d_hydropathy", "d_volume", "d_charge"]
TIER1_EXTRA_COLS = ["esm_pos_mean_llr", "esm_pos_min_llr", "rel_position"]
PRED_COLS = ["ddg_pred", "p_impact", "rank_class", "mechanism_flags", "evidence_text", "confidence"]
LABEL_DDG_COLS = ["ddg", "source", "split_id", "cluster_id"]
LABEL_FUNC_COLS = ["label", "score_raw", "source", "split_id", "cluster_id"]

# level: protein / position / mutation
# complete=True: the table must hold every position (L rows) or every
# mutation (19*L rows) for each protein it contains.
TABLES = {
    "proteins": {"level": "protein", "keys": ["protein_id"],
                 "required": PROTEIN_COLS, "complete": False},
    "pos_structure": {"level": "position", "keys": POS_KEYS,
                      "required": POS_KEYS + POS_STRUCT_COLS, "complete": True},
    "pos_evolution": {"level": "position", "keys": POS_KEYS,
                      "required": POS_KEYS + POS_EVO_COLS, "complete": True},
    "pos_function": {"level": "position", "keys": POS_KEYS,
                     "required": POS_KEYS + POS_FUNC_COLS, "complete": True},
    "mut_esm": {"level": "mutation", "keys": MUT_KEYS,
                "required": MUT_KEYS + MUT_ESM_COLS, "complete": True},
    "mut_features": {"level": "mutation", "keys": MUT_KEYS,
                     "required": MUT_KEYS + POS_STRUCT_COLS + POS_EVO_COLS + POS_FUNC_COLS
                                 + MUT_ESM_COLS + MUT_EVO_COLS + MUT_PHYS_COLS,
                     "complete": True},
    "features_tier1": {"level": "mutation", "keys": MUT_KEYS,
                        "required": MUT_KEYS + MUT_ESM_COLS + MUT_PHYS_COLS + TIER1_EXTRA_COLS,
                        "complete": True},
    "predictions": {"level": "mutation", "keys": MUT_KEYS,
                    "required": MUT_KEYS + PRED_COLS, "complete": True},
    "labels_ddg": {"level": "mutation", "keys": MUT_KEYS,
                   "required": MUT_KEYS + LABEL_DDG_COLS, "complete": False},
    "labels_function": {"level": "mutation", "keys": MUT_KEYS,
                        "required": MUT_KEYS + LABEL_FUNC_COLS, "complete": False},
    "external_clinvar": {"level": "mutation", "keys": MUT_KEYS,
                          "required": MUT_KEYS + ["label", "review_stars", "clinvar_variation_id", "gene"],
                          "complete": False},
}


def check_columns(df, required, name="table"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    return df


def mutation_label(wt: str, position: int, mut: str) -> str:
    return f"{wt}{position}{mut}"  # e.g. R175H
