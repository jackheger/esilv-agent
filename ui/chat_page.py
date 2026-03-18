from __future__ import annotations

import streamlit as st

from agents.orchestrator import OrchestratorAgent
from app.conversation_store import ConversationStore
from app.models import SuperAgentTraceRecord


def render_chat_page(
    conversation_store: ConversationStore,
    orchestrator: OrchestratorAgent,
    selected_conversation_id: str,
    gemini_configured: bool,
) -> None:
    conversation = conversation_store.load(selected_conversation_id)
    st.subheader(conversation.title)

    if not gemini_configured:
        st.warning("GEMINI_API_KEY is not configured. The chat UI works, but responses are disabled.")

    for message in conversation.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.citations:
                st.markdown("**Sources**")
                for citation in message.citations:
                    if citation.kind == "document":
                        label = citation.title
                        if citation.page_number is not None:
                            label = f"{label}, page {citation.page_number}"
                        st.markdown(f"- {label}")
                    elif citation.url:
                        st.markdown(f"- [{citation.title}]({citation.url})")
                    else:
                        st.markdown(f"- {citation.title}")
            if message.super_agent_trace:
                with st.expander("Super Agent trace", expanded=False):
                    if message.super_agent_stop_reason:
                        st.caption(f"Final stop reason: {message.super_agent_stop_reason}")
                    for item in message.super_agent_trace:
                        _render_super_agent_trace_item(item)

    user_text = st.chat_input("Ask about ESILV or about the uploaded PDF documents...")
    if user_text:
        with st.spinner("Thinking..."):
            orchestrator.handle_turn(selected_conversation_id, user_text)
        st.rerun()


def _render_super_agent_trace_item(item: SuperAgentTraceRecord) -> None:
    st.markdown(
        f"**Iteration {item.iteration_number}**  \n"
        f"Strategy: `{item.selected_strategy}`  \n"
        f"Executed query: `{item.executed_query}`"
    )
    if item.retrieval_outcome is not None:
        st.markdown(
            "Retrieval outcome: "
            f"hits=`{item.retrieval_outcome.hit_count}`, "
            f"weak=`{item.retrieval_outcome.weak}`, "
            f"top_score=`{item.retrieval_outcome.top_score}`, "
            f"top_overlap=`{item.retrieval_outcome.top_lexical_overlap}`"
        )
    if item.search_outcome is not None:
        st.markdown(
            "Web search outcome: "
            f"hits=`{item.search_outcome.hit_count}`, "
            f"weak=`{item.search_outcome.weak}`, "
            f"top_score=`{item.search_outcome.top_score}`, "
            f"top_overlap=`{item.search_outcome.top_lexical_overlap}`, "
            f"top_expanded_overlap=`{item.search_outcome.top_expanded_overlap}`"
        )
    st.markdown(f"Draft answer:\n\n{item.draft_answer}")
    st.markdown(
        "Evaluator result: "
        f"sufficient=`{item.evaluator_result.is_sufficient}`, "
        f"grounded=`{item.evaluator_result.is_grounded}`, "
        f"missing=`{', '.join(item.evaluator_result.missing_information) or 'none'}`, "
        f"unsupported=`{', '.join(item.evaluator_result.unsupported_claims) or 'none'}`, "
        f"next_strategy=`{item.evaluator_result.next_strategy or 'none'}`"
    )
    st.markdown(f"Rewritten query: `{item.rewritten_query or 'none'}`")
    st.markdown(f"Stop reason: `{item.stop_reason or 'continue'}`")
    st.divider()
