from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402


def main() -> None:
    from app.runtime import build_services
    from ui.admin_page import render_admin_page
    from ui.chat_page import render_chat_page
    from ui.components import ensure_session_state, inject_css, render_header, render_sidebar

    st.set_page_config(
        page_title="ESILV Smart Assistant",
        page_icon=":speech_balloon:",
        layout="wide",
    )
    inject_css()

    services = build_services()
    ensure_session_state(services.conversation_store)

    render_sidebar(services.conversation_store)
    render_header()

    selected_conversation_id = st.session_state.selected_conversation_id
    if st.session_state.current_view == "admin":
        render_admin_page(
            upload_registry=services.upload_registry,
            pdf_ingestion=services.pdf_ingestion,
            vector_store=services.vector_store,
            search_agent=services.search_agent,
            agent_settings_store=services.agent_settings_store,
            default_generation_model=services.settings.gemini_model,
            default_embedding_model=services.settings.gemini_embedding_model,
        )
        return

    render_chat_page(
        conversation_store=services.conversation_store,
        orchestrator=services.orchestrator,
        selected_conversation_id=selected_conversation_id,
        gemini_configured=services.orchestrator.llm_client.configured,
    )


if __name__ == "__main__":
    main()
