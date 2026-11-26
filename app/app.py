"""Streamlit dashboard for computational protein mutation hypotheses."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import cast

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html as st_html

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."
ALPHAFOLD_HOST = "alphafold.ebi.ac.uk"


def resolve_data_dir(project_root: Path = ROOT) -> Path:
    requested = Path(os.environ.get("XYNAPSE_DATA_DIR", project_root / "data/processed")).expanduser()
    if (requested / "predictions.parquet").is_file():
        return requested
    demo = project_root / "demo_data"
    if (demo / "predictions.parquet").is_file():
        return demo
    return requested


@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def load_alpha_fold_pdb(accession: str, cache_dir: Path) -> tuple[str | None, str | None]:
    """Fetch/cached AlphaFold PDB after checking the observed API response fields."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = cache_dir / f"{accession}.pdb"
    if pdb_path.is_file():
        return pdb_path.read_text(), None
    if not accession.isalnum() or not accession.isupper():
        return None, "The selected protein identifier is not a supported UniProt accession."
    api_url = f"https://{ALPHAFOLD_HOST}/api/prediction/{urllib.parse.quote(accession)}"
    request = urllib.request.Request(api_url, headers={"User-Agent": "Xynapse-AI/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"AlphaFold DB could not be reached: {exc}"
    if not isinstance(payload, list):
        return None, "AlphaFold DB returned an unexpected response shape."
    record = next((item for item in payload if isinstance(item, dict)
                   and item.get("uniprotAccession") == accession
                   and item.get("uniprotStart") == 1), None)
    if record is None or not isinstance(record.get("pdbUrl"), str):
        return None, "No full-length AlphaFold PDB record was available for this accession."
    pdb_url = urllib.parse.urlparse(record["pdbUrl"])
    if pdb_url.scheme != "https" or pdb_url.hostname != ALPHAFOLD_HOST:
        return None, "AlphaFold DB returned a structure URL outside the approved host."
    try:
        request = urllib.request.Request(record["pdbUrl"], headers={"User-Agent": "Xynapse-AI/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            size = response.headers.get("Content-Length")
            if size is not None and int(size) > 20 * 1024 * 1024:
                return None, "AlphaFold structure exceeds the dashboard download limit."
            content = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError, ValueError) as exc:
        return None, f"AlphaFold structure download failed: {exc}"
    if not content.startswith("HEADER") and "ATOM  " not in content:
        return None, "AlphaFold DB response did not contain PDB atom records."
    pdb_path.write_text(content)
    return content, None


def color_pdb_by_residue(pdb_text: str, predictions: pd.DataFrame, metric: str) -> tuple[str, float, float]:
    """Set PDB B-factor display values to per-residue means of the selected prediction."""
    residue_means = cast(pd.Series, predictions.groupby("position")[metric].mean())
    residue_values = {int(position): float(value) for position, value in residue_means.items()}
    numeric_values = [float(value) for value in residue_values.values() if pd.notna(value)]
    if not numeric_values:
        raise ValueError(f"No per-residue values are available for {metric}")
    colored_lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
            try:
                position = int(line[22:26].strip())
            except ValueError:
                colored_lines.append(line)
                continue
            value = residue_values.get(position)
            if value is not None and pd.notna(value):
                line = line[:60] + f"{float(value):6.2f}" + line[66:]
        colored_lines.append(line)
    return "\n".join(colored_lines) + "\n", min(numeric_values), max(numeric_values)


def _render_structure(pdb_text: str, frame: pd.DataFrame, metric: str) -> None:
    import py3Dmol

    colored, minimum, maximum = color_pdb_by_residue(pdb_text, frame, metric)
    view = py3Dmol.view(width=900, height=520)
    view.addModel(colored, "pdb")
    view.setStyle({"cartoon": {"colorscheme": {
        "prop": "b", "gradient": "roygb", "min": minimum, "max": maximum,
    }}})
    view.zoomTo()
    st_html(view._make_html(), height=540, scrolling=False)
    st.caption(f"Color: per-residue mean {metric}; PDB B-factor field changed for display only. Values: {minimum:.3f} to {maximum:.3f}.")


def _load_limitations(project_root: Path = ROOT) -> str:
    path = project_root / "docs/limitations.md"
    if not path.is_file():
        return "Limitations documentation is not available in this checkout."
    return path.read_text(encoding="utf-8")


def main() -> None:
    st.set_page_config(page_title="Xynapse AI", layout="wide")
    st.title("Xynapse AI — mutation-effect predictions")
    st.warning(DISCLAIMER)
    data_dir = resolve_data_dir()
    if data_dir.resolve() == (ROOT / "demo_data").resolve():
        st.error("DEMO - NOT REAL")
    predictions_path = data_dir / "predictions.parquet"
    if not predictions_path.is_file():
        st.error(f"No predictions.parquet found in {data_dir}. Create predictions or select a demo bundle.")
        st.caption(DISCLAIMER)
        return
    try:
        all_predictions = load_predictions(str(predictions_path))
        if "esm_llr" not in all_predictions.columns:
            feature_candidates = (data_dir / "features_tier1.parquet",
                                  ROOT / "data/processed/features_tier1.parquet")
            for feature_path in feature_candidates:
                if feature_path.is_file():
                    esm_features = pd.read_parquet(feature_path)[
                        ["protein_id", "position", "wt", "mut", "esm_llr"]
                    ]
                    all_predictions = all_predictions.merge(
                        esm_features, on=["protein_id", "position", "wt", "mut"],
                        how="left", validate="one_to_one",
                    )
                    break
    except Exception as exc:
        st.error(f"Could not load prediction Parquet: {exc}")
        st.caption(DISCLAIMER)
        return
    required = {"protein_id", "position", "wt", "mut", "ddg_pred", "p_impact", "rank_class",
                "mechanism_flags", "evidence_text", "confidence", "ddg_confidence", "tier", "esm_model"}
    missing = sorted(required - set(all_predictions.columns))
    if missing:
        st.error(f"Prediction table is missing required columns: {missing}")
        st.caption(DISCLAIMER)
        return
    protein_ids = sorted(all_predictions.protein_id.astype(str).unique().tolist())
    default_index = protein_ids.index("P60484") if "P60484" in protein_ids else 0
    protein_id = str(st.selectbox("Protein", protein_ids, index=default_index))
    frame = all_predictions.loc[all_predictions.protein_id.astype(str).eq(protein_id)].copy()
    esm_names = sorted(frame.esm_model.dropna().astype(str).unique().tolist())
    tiers = sorted(frame.tier.dropna().astype(str).unique().tolist())
    st.caption(f"Data: `{data_dir}` · {len(frame):,} mutations · Tier {', '.join(tiers)} · ESM model: {', '.join(esm_names)}")

    load_structure = st.checkbox("Load AlphaFold 3D view (optional network request)", value=False)
    pdb_text: str | None = None
    structure_error: str | None = None
    if load_structure:
        pdb_text, structure_error = load_alpha_fold_pdb(protein_id, ROOT / "data/raw/alphafold")

    tab_names = ["Protein summary", "Heatmap", "Mutation table", "Mechanism evidence"]
    if pdb_text is not None:
        tab_names.append("3D view")
    tab_names.append("Limitations")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.caption(DISCLAIMER)
        st.subheader(f"Protein summary: {protein_id}")
        st.metric("Single substitutions", f"{len(frame):,}")
        st.dataframe(frame.rank_class.value_counts().rename_axis("rank_class").reset_index(name="n_mutations"),
                     width="stretch", hide_index=True)
        st.write({"tiers": tiers, "ESM model": esm_names,
                  "ddG confidence": sorted(frame.ddg_confidence.dropna().astype(str).unique().tolist()),
                  "function confidence": sorted(frame.confidence.dropna().astype(str).unique().tolist())})

    with tabs[1]:
        st.caption(DISCLAIMER)
        metric = st.selectbox("Heatmap metric", ["ddg_pred", "p_impact", "esm_llr"], key="heatmap_metric")
        if metric not in frame:
            st.warning(f"Heatmap metric `{metric}` is unavailable in this prediction table.")
        else:
            matrix = frame.pivot(index="mut", columns="position", values=metric).sort_index()
            fig = px.imshow(matrix, aspect="auto", origin="lower", labels={
                "x": "Position", "y": "Mutant amino acid", "color": metric,
            }, color_continuous_scale="RdBu_r")
            fig.update_layout(title=f"{protein_id}: {metric} by position and mutant")
            st.plotly_chart(fig, width="stretch")

    with tabs[2]:
        st.caption(DISCLAIMER)
        classes = sorted(frame.rank_class.dropna().astype(str).unique().tolist())
        confidences = sorted(frame.confidence.dropna().astype(str).unique().tolist())
        ddg_confidences = sorted(frame.ddg_confidence.dropna().astype(str).unique().tolist())
        flags = sorted({flag for value in frame.mechanism_flags.fillna("").astype(str)
                        for flag in value.split(";") if flag})
        selected_classes = st.multiselect("Rank class", classes, default=[], key="rank_filter")
        selected_confidence = st.multiselect("Function confidence", confidences, default=[], key="confidence_filter")
        selected_ddg_confidence = st.multiselect("ddG confidence", ddg_confidences, default=[], key="ddg_confidence_filter")
        selected_flags = st.multiselect("Mechanism flag", flags, default=[], key="mechanism_filter")
        low, high = int(frame.position.min()), int(frame.position.max())
        position_range = st.slider("Position range", min_value=low, max_value=high, value=(low, high))
        filtered = frame.loc[frame.position.between(position_range[0], position_range[1])].copy()
        if selected_classes:
            filtered = filtered.loc[filtered.rank_class.isin(selected_classes)]
        if selected_confidence:
            filtered = filtered.loc[filtered.confidence.isin(selected_confidence)]
        if selected_ddg_confidence:
            filtered = filtered.loc[filtered.ddg_confidence.isin(selected_ddg_confidence)]
        if selected_flags:
            filtered = filtered.loc[filtered.mechanism_flags.fillna("").map(
                lambda value: any(flag in str(value).split(";") for flag in selected_flags))]
        st.dataframe(filtered, width="stretch", hide_index=True)
        st.download_button("Download filtered mutations CSV", data=csv_bytes(filtered),
                           file_name=f"{protein_id}_predictions.csv", mime="text/csv")
        st.caption(f"Showing {len(filtered):,} of {len(frame):,} substitutions.")

    with tabs[3]:
        st.caption(DISCLAIMER)
        evidence_rows = frame.copy()
        evidence_rows["mutation"] = evidence_rows.wt.astype(str) + evidence_rows.position.astype(str) + evidence_rows.mut.astype(str)
        selected_mutation = st.selectbox("Mutation evidence", evidence_rows.mutation.tolist(), key="evidence_mutation")
        row = evidence_rows.loc[evidence_rows.mutation.eq(selected_mutation)].iloc[0]
        st.write(row.evidence_text)
        evidence_fields = pd.DataFrame({"field": row.index.astype(str), "value": row.astype(str).to_numpy()})
        st.dataframe(evidence_fields, width="stretch", hide_index=True)

    tab_index = 4
    if pdb_text is not None:
        with tabs[tab_index]:
            st.caption(DISCLAIMER)
            metric = st.selectbox("Color structure by per-residue mean", ["p_impact", "ddg_pred", "esm_llr"], key="structure_metric")
            if metric not in frame:
                st.info(f"The selected per-residue metric `{metric}` is unavailable.")
            else:
                _render_structure(pdb_text, frame, metric)
        tab_index += 1
    elif structure_error is not None:
        st.info(f"3D view hidden: {structure_error}")

    with tabs[tab_index]:
        st.caption(DISCLAIMER)
        st.markdown(_load_limitations())


if __name__ == "__main__":
    main()
