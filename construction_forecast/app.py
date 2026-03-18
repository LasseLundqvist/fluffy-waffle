"""
Construction Market Forecasting PoC
Prognosecenteret — Nordic Construction Market Analysis

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — ensure local modules are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Construction Market Forecaster | Prognosecenteret",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local imports (after path setup)
# ---------------------------------------------------------------------------
from agent_narrative import generate_narrative
from agent_research import generate_assumptions
from docx_generator import generate_narrative_docx
from ingestion import ingest_pdf
from schemas import AssumptionSet, NarrativeResult
from vector_store import add_chunks, get_document_stats

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "assumption_sets" not in st.session_state:
    st.session_state.assumption_sets: list[AssumptionSet] = []

if "narratives" not in st.session_state:
    st.session_state.narratives: list[NarrativeResult] = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.shields.io/badge/Prognosecenteret-Construction%20AI-1E5FA6?style=for-the-badge",
        use_container_width=True,
    )
    st.title("🏗️ Nordic Construction\nMarket Forecaster")
    st.divider()

    # Country selector
    country = st.selectbox(
        "Country",
        options=["DK", "NO", "SE"],
        format_func=lambda c: {"DK": "🇩🇰 Denmark", "NO": "🇳🇴 Norway", "SE": "🇸🇪 Sweden"}[c],
        help="Select the target country for analysis and document filtering.",
    )
    st.session_state["country"] = country

    st.divider()

    # API key
    api_key_env = os.getenv("ANTHROPIC_API_KEY", "")
    api_key_secret = ""
    try:
        api_key_secret = st.secrets.get("ANTHROPIC_API_KEY", "")  # type: ignore[attr-defined]
    except Exception:
        pass

    api_key_input = st.text_input(
        "Anthropic API Key",
        value=api_key_secret or api_key_env,
        type="password",
        help="Enter your API key or set ANTHROPIC_API_KEY in .env / st.secrets",
    )
    api_key = api_key_input or api_key_secret or api_key_env
    st.session_state["api_key"] = api_key

    if api_key:
        st.success("API key configured ✓", icon="🔑")
    else:
        st.warning("No API key set", icon="⚠️")

    st.divider()

    # ChromaDB stats
    st.subheader("📚 Document Store")
    try:
        stats = get_document_stats()
        total_chunks = stats.get("total_chunks", 0)
        total_docs = len(stats.get("documents", {}))
        col1, col2 = st.columns(2)
        col1.metric("Documents", total_docs)
        col2.metric("Chunks", total_chunks)
    except Exception as e:
        st.caption(f"Store not available: {e}")

    st.divider()
    st.caption("Prognosecenteret © 2025")
    st.caption("Powered by Claude claude-sonnet-4-6-20250514")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Document Ingestion", "🔍 Research & Assumptions", "📊 Table Narratives", "📁 Report Archive"]
)


# ============================================================
# TAB 1: Document Ingestion
# ============================================================

with tab1:
    st.header("📄 Document Ingestion")
    st.markdown(
        "Upload PDF market reports, policy documents, and statistical publications. "
        "They will be parsed, chunked, embedded, and stored in the local ChromaDB vector store."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to ingest.",
    )

    ingest_btn = st.button(
        "⚙️ Ingest Documents",
        type="primary",
        disabled=not uploaded_files,
    )

    if ingest_btn and uploaded_files:
        total_chunks = 0
        progress = st.progress(0, text="Starting ingestion…")

        results_log: list[dict] = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                (i) / len(uploaded_files),
                text=f"Processing {uploaded_file.name}…",
            )
            try:
                pdf_bytes = uploaded_file.read()
                chunks = ingest_pdf(
                    pdf_bytes,
                    filename=uploaded_file.name,
                    country=country,
                )
                added = add_chunks(chunks)
                total_chunks += added
                results_log.append(
                    {"file": uploaded_file.name, "chunks": added, "status": "✅ OK"}
                )
            except Exception as exc:
                results_log.append(
                    {"file": uploaded_file.name, "chunks": 0, "status": f"❌ {exc}"}
                )

        progress.progress(1.0, text="Done!")

        st.success(f"Ingested **{total_chunks}** chunks from **{len(uploaded_files)}** file(s).")

        results_df = pd.DataFrame(results_log)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Force sidebar stats refresh
        st.rerun()

    # Document list
    st.divider()
    st.subheader("Ingested Documents")

    try:
        stats = get_document_stats()
        doc_map = stats.get("documents", {})
        if not doc_map:
            st.info("No documents ingested yet. Upload PDFs above to get started.")
        else:
            rows = [
                {
                    "Filename": fname,
                    "Country": info.get("country", ""),
                    "Chunks": info.get("chunk_count", 0),
                    "Uploaded": info.get("upload_date", "")[:10],
                }
                for fname, info in doc_map.items()
            ]
            st.dataframe(
                pd.DataFrame(rows).sort_values("Uploaded", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as e:
        st.warning(f"Could not load document stats: {e}")


# ============================================================
# TAB 2: Research & Assumptions
# ============================================================

with tab2:
    st.header("🔍 Research & Assumptions Generator")
    st.markdown(
        "Describe the market drivers you want to analyse. The agent will retrieve relevant "
        "context from ingested documents via RAG and generate structured 3-year assumptions."
    )

    default_query = {
        "DK": "Generer antagelser for danske boligbyggerier, byggeomkostninger og reelle renter for 2026-2028",
        "NO": "Generer antagelser for norske boligstarter, byggekostnader og realrenter for 2026-2028",
        "SE": "Generera antaganden för svenska bostadsstarter, byggkostnader och realräntor för 2026-2028",
    }

    query = st.text_area(
        "Market drivers to analyse",
        value=default_query.get(country, ""),
        height=120,
        help="Describe the indicators you want forecasts for. Be as specific as possible.",
        placeholder="E.g. Generate assumptions for Danish housing starts, construction costs, and real interest rates 2026-2028",
    )

    gen_btn = st.button(
        "🚀 Generate Assumptions",
        type="primary",
        disabled=not query.strip() or not api_key,
    )

    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar.")

    if gen_btn and query.strip() and api_key:
        with st.spinner("Retrieving context and generating assumptions via Claude…"):
            try:
                assumption_set = generate_assumptions(
                    query=query,
                    country=country,
                    api_key=api_key,
                )
                st.session_state.assumption_sets.append(assumption_set)

                # Persist to output directory
                output_path = OUTPUT_DIR / f"assumptions_{assumption_set.id}_{assumption_set.timestamp[:10]}.json"
                output_path.write_text(assumption_set.model_dump_json(indent=2), encoding="utf-8")

                st.success(
                    f"Generated **{len(assumption_set.assumptions)}** assumptions "
                    f"using **{assumption_set.rag_chunks_used}** RAG chunks."
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.stop()

    # Display latest assumption set
    if st.session_state.assumption_sets:
        latest = st.session_state.assumption_sets[-1]

        st.divider()
        col_h1, col_h2 = st.columns([3, 1])
        col_h1.subheader(f"Latest: Set {latest.id} — {latest.country} ({latest.timestamp[:10]})")

        json_bytes = latest.model_dump_json(indent=2).encode()
        col_h2.download_button(
            "⬇️ Download JSON",
            data=json_bytes,
            file_name=f"assumptions_{latest.id}.json",
            mime="application/json",
        )

        for assumption in latest.assumptions:
            with st.expander(
                f"**{assumption.metric}** — Confidence: {assumption.confidence.upper()}",
                expanded=True,
            ):
                cols = st.columns([1, 1, 1, 1])
                cols[0].metric("Current Value", assumption.current_value[:40])
                cols[1].metric("2026", assumption.forecast_2026[:40])
                cols[2].metric("2027", assumption.forecast_2027[:40])
                cols[3].metric("2028", assumption.forecast_2028[:40])

                st.markdown("**Key Drivers:**")
                for driver in assumption.key_drivers:
                    st.markdown(f"- {driver}")

                if assumption.source_documents:
                    st.caption("Sources: " + ", ".join(assumption.source_documents))

    # History panel
    if len(st.session_state.assumption_sets) > 1:
        st.divider()
        st.subheader("📋 Session History")
        for aset in reversed(st.session_state.assumption_sets[:-1]):
            with st.expander(
                f"Set {aset.id} — {aset.country} | {aset.timestamp[:16]} | {len(aset.assumptions)} assumptions"
            ):
                st.caption(f"Query: {aset.query}")
                for a in aset.assumptions:
                    st.markdown(f"- **{a.metric}** ({a.confidence}): {a.forecast_2026[:60]}…")


# ============================================================
# TAB 3: Table Narratives
# ============================================================

with tab3:
    st.header("📊 Table Narrative Writer")
    st.markdown(
        "Upload a forecast data table (CSV or Excel), select relevant assumption sets, "
        "and generate editorial commentary in The Economist style."
    )

    # Table upload
    table_file = st.file_uploader(
        "Upload forecast table",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel file containing the forecast data to be described.",
    )

    df: pd.DataFrame | None = None

    if table_file:
        try:
            if table_file.name.endswith(".csv"):
                df = pd.read_csv(table_file)
            else:
                df = pd.read_excel(table_file)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read file: {e}")

    # Assumption set selector
    st.divider()
    if not st.session_state.assumption_sets:
        st.info("No assumption sets in session. Generate assumptions in the Research tab first.")
        selected_assumption_ids: list[str] = []
    else:
        options = {
            f"Set {aset.id} — {aset.country} ({aset.timestamp[:10]}) — {len(aset.assumptions)} assumptions": aset.id
            for aset in st.session_state.assumption_sets
        }
        selected_labels = st.multiselect(
            "Select assumption sets to reference",
            options=list(options.keys()),
            default=list(options.keys())[:1],
        )
        selected_assumption_ids = [options[label] for label in selected_labels]

    selected_assumption_sets = [
        aset
        for aset in st.session_state.assumption_sets
        if aset.id in selected_assumption_ids
    ]

    # Generate button
    narr_btn = st.button(
        "✍️ Generate Narrative",
        type="primary",
        disabled=(df is None or not api_key),
    )

    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar.")

    if narr_btn and df is not None and api_key:
        table_markdown = df.to_markdown(index=False)
        if table_markdown is None:
            table_markdown = df.to_string(index=False)

        with st.spinner("Writing editorial narrative via Claude…"):
            try:
                narrative = generate_narrative(
                    table_markdown=table_markdown,
                    assumption_sets=selected_assumption_sets,
                    country=country,
                    api_key=api_key,
                )
                st.session_state.narratives.append(narrative)

                # Persist JSON
                output_path = OUTPUT_DIR / f"narrative_{narrative.id}_{narrative.timestamp[:10]}.json"
                output_path.write_text(narrative.model_dump_json(indent=2), encoding="utf-8")

                st.success("Narrative generated!")
            except Exception as exc:
                st.error(f"Narrative generation failed: {exc}")
                st.stop()

    # Display latest narrative
    if st.session_state.narratives:
        latest_narr = st.session_state.narratives[-1]

        st.divider()
        st.subheader(latest_narr.title)
        st.caption(f"Generated: {latest_narr.timestamp[:16]} | Country: {latest_narr.country}")

        st.markdown(latest_narr.narrative_text)

        # Download Word
        if st.button("📥 Prepare Word Download"):
            with st.spinner("Generating .docx…"):
                try:
                    ref_sets = [
                        aset
                        for aset in st.session_state.assumption_sets
                        if aset.id in latest_narr.referenced_assumption_ids
                    ]
                    docx_bytes = generate_narrative_docx(latest_narr, ref_sets)

                    st.download_button(
                        label="⬇️ Download as Word (.docx)",
                        data=docx_bytes,
                        file_name=f"narrative_{latest_narr.id}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                except Exception as exc:
                    st.error(f"Word generation failed: {exc}")


# ============================================================
# TAB 4: Report Archive
# ============================================================

with tab4:
    st.header("📁 Report Archive")
    st.markdown("All generated reports and assumption sets saved in `./output/`.")

    refresh_btn = st.button("🔄 Refresh Archive")

    output_files = sorted(OUTPUT_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    if not output_files:
        st.info("No archived reports yet. Generate assumptions or narratives to populate the archive.")
    else:
        for fpath in output_files:
            try:
                raw = json.loads(fpath.read_text(encoding="utf-8"))
                report_type = "assumptions" if "assumptions" in fpath.name else "narrative"
                date_str = raw.get("timestamp", "")[:10]
                country_code = raw.get("country", "?")

                icon = "🔍" if report_type == "assumptions" else "📝"
                label = f"{icon} {fpath.stem} — {country_code} — {date_str}"

                with st.expander(label):
                    col1, col2 = st.columns([3, 1])

                    if report_type == "assumptions":
                        assumptions_list = raw.get("assumptions", [])
                        col1.markdown(f"**Query:** {raw.get('query', '')[:120]}")
                        col1.markdown(f"**Assumptions:** {len(assumptions_list)} metrics")
                        for a in assumptions_list[:3]:
                            col1.markdown(f"- {a.get('metric', '')} ({a.get('confidence', '')})")
                        if len(assumptions_list) > 3:
                            col1.caption(f"…and {len(assumptions_list) - 3} more")
                    else:
                        preview = raw.get("narrative_text", "")[:300]
                        col1.markdown(preview + "…")

                    col2.download_button(
                        "⬇️ JSON",
                        data=fpath.read_bytes(),
                        file_name=fpath.name,
                        mime="application/json",
                        key=f"dl_{fpath.stem}",
                    )

                    # For narratives: also offer Word download
                    if report_type == "narrative":
                        try:
                            narr_obj = NarrativeResult.model_validate(raw)
                            ref_sets = [
                                aset
                                for aset in st.session_state.assumption_sets
                                if aset.id in narr_obj.referenced_assumption_ids
                            ]
                            docx_bytes = generate_narrative_docx(narr_obj, ref_sets)
                            col2.download_button(
                                "⬇️ DOCX",
                                data=docx_bytes,
                                file_name=fpath.stem + ".docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"docx_{fpath.stem}",
                            )
                        except Exception:
                            pass

            except Exception as exc:
                st.warning(f"Could not read {fpath.name}: {exc}")
