from __future__ import annotations

from pathlib import Path

import streamlit as st

from agents.web_search import SiteSearchAgent
from app.agent_settings import AgentFeatureSettingsStore
from app.models import AgentFeatureSettings, SUPPORTED_EMBEDDING_MODELS, SUPPORTED_GENERATION_MODELS
from ingestion.pdf_ingestion import PdfIngestionService
from ingestion.uploads import UploadRegistry
from ingestion.vector_store import LocalVectorStore


def render_admin_page(
    upload_registry: UploadRegistry,
    pdf_ingestion: PdfIngestionService,
    vector_store: LocalVectorStore,
    search_agent: SiteSearchAgent,
    agent_settings_store: AgentFeatureSettingsStore,
    default_generation_model: str,
    default_embedding_model: str,
) -> None:
    st.subheader("Admin")
    st.caption("Manage ingestion, indexed documents, website cache, and agent behavior.")
    section = st.radio(
        "Admin section",
        options=["Ingestion", "Agent parameters"],
        key="admin_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section == "Agent parameters":
        _render_agent_parameters_section(
            agent_settings_store,
            default_generation_model=default_generation_model,
            default_embedding_model=default_embedding_model,
        )
        return

    _render_ingestion_section(upload_registry, pdf_ingestion, vector_store, search_agent)


def _render_ingestion_section(
    upload_registry: UploadRegistry,
    pdf_ingestion: PdfIngestionService,
    vector_store: LocalVectorStore,
    search_agent: SiteSearchAgent,
) -> None:
    st.markdown("### Ingestion")
    st.caption("Upload PDFs, review indexed documents, and manage the ESILV website cache.")
    _render_upload_section(upload_registry, pdf_ingestion, vector_store)
    st.divider()
    _render_cache_section(search_agent)


def _render_agent_parameters_section(
    agent_settings_store: AgentFeatureSettingsStore,
    default_generation_model: str,
    default_embedding_model: str,
) -> None:
    settings = agent_settings_store.load()
    st.markdown("### Agent parameters")
    st.caption(
        "Choose which retrieval tools the assistant is allowed to use for all conversations, "
        "whether iterative Super Agent orchestration is active, and which Gemini models are used "
        "for generation and embeddings."
    )
    st.info(
        _behavior_summary(
            settings,
            default_generation_model=default_generation_model,
            default_embedding_model=default_embedding_model,
        )
    )

    selected_generation_model = _selected_option(
        current_value=settings.generation_model,
        default_value=default_generation_model,
        allowed_values=SUPPORTED_GENERATION_MODELS,
    )
    selected_embedding_model = _selected_option(
        current_value=settings.embedding_model,
        default_value=default_embedding_model,
        allowed_values=SUPPORTED_EMBEDDING_MODELS,
    )

    with st.form("agent-parameters-form"):
        rag_enabled = st.checkbox(
            "Enable RAG on uploaded PDFs",
            value=settings.rag_enabled,
            help="Allows the assistant to retrieve from ingested PDF documents and attach document citations.",
        )
        web_search_enabled = st.checkbox(
            "Enable ESILV site search",
            value=settings.web_search_enabled,
            help="Allows the assistant to search the cached ESILV website and attach web citations.",
        )
        super_agent_enabled = st.checkbox(
            "Enable Super Agent iterative orchestration",
            value=settings.super_agent_enabled,
            help=(
                "Runs up to 3 grounded retrieve/search iterations, evaluates answer completeness "
                "against the original question, and stores a visible trace for each turn."
            ),
        )
        generation_model = st.selectbox(
            "Generation model",
            options=list(SUPPORTED_GENERATION_MODELS),
            index=SUPPORTED_GENERATION_MODELS.index(selected_generation_model),
            help="Used by the orchestrator and every Gemini-based generation step across the agents.",
        )
        embedding_model = st.selectbox(
            "Embedding model",
            options=list(SUPPORTED_EMBEDDING_MODELS),
            index=SUPPORTED_EMBEDDING_MODELS.index(selected_embedding_model),
            help="Used for PDF ingestion and retrieval embeddings across the backend.",
        )
        submitted = st.form_submit_button("Save parameters")

    if not submitted:
        return

    updated_settings = AgentFeatureSettings(
        rag_enabled=rag_enabled,
        web_search_enabled=web_search_enabled,
        super_agent_enabled=super_agent_enabled,
        generation_model=generation_model,
        embedding_model=embedding_model,
    )
    agent_settings_store.save(updated_settings)
    st.success("Agent parameters saved.")
    st.rerun()


def _behavior_summary(
    settings: AgentFeatureSettings,
    default_generation_model: str,
    default_embedding_model: str,
) -> str:
    mode_label = "Super Agent mode" if settings.super_agent_enabled else "Standard mode"
    if settings.rag_enabled and settings.web_search_enabled:
        tool_summary = "direct answers, PDF RAG, and ESILV site search are all enabled."
    elif settings.rag_enabled:
        tool_summary = "direct answers and PDF RAG only. ESILV site search is disabled."
    elif settings.web_search_enabled:
        tool_summary = "direct answers and ESILV site search only. PDF RAG is disabled."
    else:
        tool_summary = "direct answers only. No PDF or ESILV site retrieval will be used."

    if settings.super_agent_enabled and (settings.rag_enabled or settings.web_search_enabled):
        return (
            f"Current mode: {mode_label}. {tool_summary} "
            "Non-direct retrieval turns may run up to 3 retrieve/search iterations. "
            f"Generation model: {_selected_option(settings.generation_model, default_generation_model, SUPPORTED_GENERATION_MODELS)}. "
            f"Embedding model: {_selected_option(settings.embedding_model, default_embedding_model, SUPPORTED_EMBEDDING_MODELS)}."
        )
    return (
        f"Current mode: {mode_label}. {tool_summary} "
        f"Generation model: {_selected_option(settings.generation_model, default_generation_model, SUPPORTED_GENERATION_MODELS)}. "
        f"Embedding model: {_selected_option(settings.embedding_model, default_embedding_model, SUPPORTED_EMBEDDING_MODELS)}."
    )


def _selected_option(
    current_value: str | None,
    default_value: str,
    allowed_values: tuple[str, ...],
) -> str:
    if current_value in allowed_values:
        return current_value
    if default_value in allowed_values:
        return default_value
    return allowed_values[0]


def _render_upload_section(
    upload_registry: UploadRegistry,
    pdf_ingestion: PdfIngestionService,
    vector_store: LocalVectorStore,
) -> None:
    st.markdown("### PDF ingestion")
    with st.form("upload-form"):
        uploaded_files = st.file_uploader(
            "Upload PDF files to parse, chunk, embed, and store in the local vector index",
            accept_multiple_files=True,
            type=["pdf"],
        )
        submitted = st.form_submit_button("Ingest PDFs")

    if submitted and uploaded_files:
        indexed_count = 0
        failed_files: list[str] = []
        for uploaded_file in uploaded_files:
            try:
                pdf_ingestion.ingest(uploaded_file)
                indexed_count += 1
            except Exception as exc:
                failed_files.append(f"{uploaded_file.name}: {exc}")

        if indexed_count:
            st.success(f"Ingested {indexed_count} PDF document(s).")
        if failed_files:
            st.error("Some PDFs could not be ingested:\n" + "\n".join(failed_files))

    records = upload_registry.list()
    vector_stats = vector_store.stats()
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Indexed documents", vector_stats.document_count)
    metric_col2.metric("Stored chunks", vector_stats.chunk_count)
    metric_col3.metric("Tracked PDFs", len(records))

    st.markdown("### Indexed documents")
    if not records:
        st.caption("No PDFs have been ingested yet.")
        return

    for record in records:
        info_col, action_col = st.columns([6, 1])
        with info_col:
            st.markdown(
                f"**{record.filename}**  \n"
                f"Status: `{record.status}` | Pages: `{record.page_count}` | Chunks: `{record.chunk_count}`  \n"
                f"Size: `{record.size}` bytes  \n"
                f"Stored at: `{Path(record.stored_path).as_posix()}`"
            )
            if record.indexed_at:
                st.caption(f"Indexed at: {record.indexed_at}")
            if record.error_message:
                st.error(record.error_message)
        with action_col:
            if st.button("Delete", key=f"delete-upload-{record.id}", use_container_width=True):
                pdf_ingestion.delete_document(record.id)
                st.rerun()


def _render_cache_section(search_agent: SiteSearchAgent) -> None:
    st.markdown("### Website cache")
    if not getattr(search_agent, "configured", True):
        st.warning(
            "TAVILY_API_KEY is not configured. Live ESILV web search and cache refresh are disabled."
        )
    action_col, clear_col = st.columns(2)
    with action_col:
        if st.button("Refresh cache", use_container_width=True):
            count = search_agent.refresh_cache()
            st.success(f"Fetched or refreshed {count} page(s).")
            st.rerun()
    with clear_col:
        if st.button("Clear cache", use_container_width=True):
            search_agent.clear_cache()
            st.success("Cache cleared.")
            st.rerun()

    stats = search_agent.cache_stats()
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Cached pages", stats.page_count)
    metric_col2.metric("Stale pages", stats.stale_count)
    if stats.last_refresh:
        st.caption(f"Last refresh: {stats.last_refresh}")

    if stats.urls:
        st.markdown("**Recent cached URLs**")
        for url in stats.urls:
            st.markdown(f"- {url}")
