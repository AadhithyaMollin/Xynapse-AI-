
import pandas as pd

def _valid(x):
    return pd.notna(x) and str(x).strip() != ""

def _num(row, key):
    try:
        x = float(row[key])
        return x if pd.notna(x) else None
    except (KeyError, TypeError, ValueError):
        return None

def _has(text, term):
    return term.lower() in str(text).lower()

def annotate_bio2(df, config):
    out = df.copy()

    thresholds = config["thresholds"]
    mechanism = config["mechanism"]
    confidence = config["confidence"]

    ddg_cut = float(thresholds["ddg_damaging"])
    impact_cut = float(thresholds["p_impact_high"])
    buried_cut = float(mechanism["buried_rsasa"])
    plddt_cut = float(confidence["plddt_low"])

    contexts = []
    hypotheses = []
    quality = []
    notes = []

    for _, row in out.iterrows():
        context = []
        mechanisms = []
        flags = []
        evidence_notes = []

        ddg = _num(row, "ddg_pred")
        impact = _num(row, "p_impact")
        rsasa = _num(row, "rsasa")
        plddt = _num(row, "plddt")

        high_impact = impact is not None and impact >= impact_cut
        damaging = ddg is not None and ddg >= ddg_cut

        domain = str(row.get("domain_annotations", "") or "")
        region = str(row.get("region_annotations", "") or "")
        motif = str(row.get("motif_annotations", "") or "")

        # Biological context
        if _has(domain, "Phosphatase tensin-type"):
            context.append("phosphatase_domain")

        if _has(domain, "C2 tensin-type"):
            context.append("C2_domain")

        if _has(region, "Disordered"):
            context.append("disordered_region")

        if _has(region, "NOP53"):
            context.append("NOP53_interaction_region")

        if _has(motif, "PDZ"):
            context.append("PDZ_binding_motif")

        # Catalytic hypothesis: directly annotated active site
        if bool(row.get("is_active_site", False)) and high_impact:
            mechanisms.append("catalytic_disruption")
            evidence_notes.append("Active-site annotation with high predicted impact")

        # Folding/stability: damaging prediction at a buried residue
        if damaging and rsasa is not None and rsasa <= buried_cut:
            mechanisms.append("folding_stability")
            evidence_notes.append("Destabilizing ddG prediction at a buried residue")

        # Domain-associated impact: context, not proof of domain dysfunction
        if (high_impact or damaging) and (
            "phosphatase_domain" in context or "C2_domain" in context
        ):
            mechanisms.append("domain_associated_impact")
            evidence_notes.append("Predicted impact within an annotated domain")

        # Interaction-associated impact: region/motif annotation plus prediction
        if (high_impact or damaging) and (
            "NOP53_interaction_region" in context
            or "PDZ_binding_motif" in context
        ):
            mechanisms.append("interaction_associated_impact")
            evidence_notes.append("Predicted impact within an annotated interaction region or motif")

        # Evidence-quality qualifiers
        if plddt is not None and plddt < plddt_cut:
            flags.append("low_structural_confidence")

        if str(row.get("ss8", "")) == "-":
            flags.append("ss8_unassigned")

        if bool(row.get("low_n_eff", False)):
            flags.append("limited_evolutionary_support")

        if plddt is None:
            flags.append("missing_plddt")

        if rsasa is None:
            flags.append("missing_rsasa")

        if not mechanisms:
            evidence_notes.append("No BIO-2 mechanism hypothesis met the configured evidence rules")

        contexts.append(";".join(context))
        hypotheses.append(";".join(dict.fromkeys(mechanisms)))
        quality.append(";".join(dict.fromkeys(flags)) if flags else "no_flagged_quality_issue")
        notes.append("; ".join(evidence_notes))

    out["bio2_context"] = contexts
    out["bio2_mechanism_hypotheses"] = hypotheses
    out["bio2_evidence_quality"] = quality
    out["bio2_mechanism_evidence"] = notes

    return out
