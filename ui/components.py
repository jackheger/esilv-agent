from __future__ import annotations

import streamlit as st

from app.conversation_store import ConversationStore


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 4rem;
            padding-bottom: 2rem;
        }
        .header-admin-button-spacer {
            height: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_session_state(conversation_store: ConversationStore) -> None:
    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"
    if "admin_section" not in st.session_state:
        st.session_state.admin_section = "Ingestion"
    if "selected_conversation_id" not in st.session_state:
        conversations = conversation_store.list()
        if conversations:
            st.session_state.selected_conversation_id = conversations[0].id
        else:
            st.session_state.selected_conversation_id = conversation_store.create().id
    elif not conversation_store.exists(st.session_state.selected_conversation_id):
        conversations = conversation_store.list()
        st.session_state.selected_conversation_id = (
            conversations[0].id if conversations else conversation_store.create().id
        )


def render_header() -> None:
    left_column, right_column = st.columns([5, 1])
    with left_column:
        st.title("ESILV Smart Assistant")
        st.caption("Local demo with Gemini orchestration and ESILV site search.")
    with right_column:
        st.markdown('<div class="header-admin-button-spacer"></div>', unsafe_allow_html=True)
        label = "Admin" if st.session_state.current_view == "chat" else "Back to chat"
        if st.button(label, use_container_width=True):
            st.session_state.current_view = "admin" if st.session_state.current_view == "chat" else "chat"
            st.rerun()


def render_sidebar(conversation_store: ConversationStore) -> None:
    with st.sidebar:
        st.subheader("Conversations")
        if st.button("New chat", use_container_width=True):
            st.session_state.selected_conversation_id = conversation_store.create().id
            st.session_state.current_view = "chat"
            st.rerun()

        conversations = conversation_store.list()
        if not conversations:
            st.caption("No conversations yet.")
            return

        for record in conversations:
            select_col, delete_col = st.columns([6, 1])
            label = record.title or "New conversation"
            button_type = "primary" if record.id == st.session_state.selected_conversation_id else "secondary"
            with select_col:
                if st.button(
                    label,
                    key=f"conversation-select-{record.id}",
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state.selected_conversation_id = record.id
                    st.session_state.current_view = "chat"
                    st.rerun()
            with delete_col:
                if st.button("X", key=f"conversation-delete-{record.id}", use_container_width=True):
                    conversation_store.delete(record.id)
                    remaining = conversation_store.list()
                    st.session_state.selected_conversation_id = (
                        remaining[0].id if remaining else conversation_store.create().id
                    )
                    st.rerun()
