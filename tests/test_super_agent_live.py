from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import time

import pytest

from agents.orchestrator import GoogleGeminiClient, OrchestratorAgent
from agents.retrieval import RetrievalAgent
from agents.web_search import SiteSearchAgent
from app.conversation_store import ConversationStore
from app.models import AgentFeatureSettings
from app.settings import AppSettings
from ingestion.pdf_ingestion import GoogleEmbeddingClient
from ingestion.vector_store import LocalVectorStore

SETTINGS = AppSettings()
PLM_PRESENT = Path("data/uploads/registry.json").exists() and "PLM.pdf" in Path(
    "data/uploads/registry.json"
).read_text("utf-8")
LIVE_READY = bool(SETTINGS.gemini_api_key and SETTINGS.tavily_api_key and PLM_PRESENT)


class InMemoryFeatureSettingsStore:
    def __init__(self, settings: AgentFeatureSettings) -> None:
        self.settings = settings

    def load(self) -> AgentFeatureSettings:
        return self.settings


def build_live_orchestrator(
    tmp_dir: str,
    feature_settings: AgentFeatureSettings,
) -> tuple[ConversationStore, OrchestratorAgent]:
    conversation_store = ConversationStore(Path(tmp_dir) / "conversations")
    vector_store = LocalVectorStore(Path("data/vector_store"))
    retrieval_agent = RetrievalAgent(
        vector_store=vector_store,
        embedding_client=GoogleEmbeddingClient(
            api_key=SETTINGS.gemini_api_key,
            model=SETTINGS.gemini_embedding_model,
        ),
    )
    search_agent = SiteSearchAgent(
        cache_dir=Path(tmp_dir) / "site_cache",
        allowed_domains=SETTINGS.allowed_domains,
        api_key=SETTINGS.tavily_api_key,
        ttl_hours=SETTINGS.site_cache_ttl_hours,
    )
    orchestrator = OrchestratorAgent(
        conversation_store=conversation_store,
        search_agent=search_agent,
        retrieval_agent=retrieval_agent,
        llm_client=GoogleGeminiClient(
            api_key=SETTINGS.gemini_api_key,
            model=SETTINGS.gemini_model,
        ),
        feature_settings_store=InMemoryFeatureSettingsStore(feature_settings),
        max_search_hits=SETTINGS.max_search_hits,
    )
    return conversation_store, orchestrator


def run_live_turn(
    orchestrator: OrchestratorAgent,
    conversation_id: str,
    user_text: str,
    attempts: int = 3,
) -> object:
    last_response = None
    for attempt in range(attempts):
        last_response = orchestrator.handle_turn(conversation_id, user_text)
        if "temporarily unavailable" not in last_response.content.lower():
            return last_response
        if attempt < attempts - 1:
            time.sleep(1)
    return last_response


@pytest.mark.skipif(not LIVE_READY, reason="Live Gemini/Tavily keys or PLM data are not available")
def test_live_rag_only_plm_query_returns_document_citations():
    with TemporaryDirectory() as tmp_dir:
        conversation_store, orchestrator = build_live_orchestrator(
            tmp_dir,
            AgentFeatureSettings(rag_enabled=True, web_search_enabled=False, super_agent_enabled=False),
        )
        conversation = conversation_store.create()

        response = run_live_turn(
            orchestrator,
            conversation.id,
            "According to the PLM document, what are the main functionalities of a PLM system?",
        )

    assert response.citations
    assert any(citation.kind == "document" and citation.title == "PLM.pdf" for citation in response.citations)
    lowered = response.content.lower()
    assert any(
        marker in lowered
        for marker in (
            "product structure management",
            "document management",
            "process",
            "workflow",
        )
    )


@pytest.mark.skipif(not LIVE_READY, reason="Live Gemini/Tavily keys or PLM data are not available")
def test_live_search_only_admissions_query_returns_esilv_citations():
    with TemporaryDirectory() as tmp_dir:
        conversation_store, orchestrator = build_live_orchestrator(
            tmp_dir,
            AgentFeatureSettings(rag_enabled=False, web_search_enabled=True, super_agent_enabled=False),
        )
        conversation = conversation_store.create()

        response = run_live_turn(orchestrator, conversation.id, "How do ESILV admissions work?")

    assert response.citations
    assert all(citation.kind == "web" and citation.url and "esilv.fr" in citation.url for citation in response.citations)
    lowered = response.content.lower()
    assert any(marker in lowered for marker in ("admission", "admissions", "concours avenir", "parcoursup"))


@pytest.mark.skipif(not LIVE_READY, reason="Live Gemini/Tavily keys or PLM data are not available")
def test_live_super_agent_combines_plm_and_esilv_sources():
    with TemporaryDirectory() as tmp_dir:
        conversation_store, orchestrator = build_live_orchestrator(
            tmp_dir,
            AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=True),
        )
        conversation = conversation_store.create()

        response = run_live_turn(
            orchestrator,
            conversation.id,
            "According to the PLM document, what are the main functionalities of a PLM system, and how do ESILV admissions work?",
        )

    assert response.orchestration_mode == "super"
    assert response.super_agent_trace
    assert any(citation.kind == "document" and citation.title == "PLM.pdf" for citation in response.citations)
    assert any(
        (
            citation.kind == "web"
            and citation.url
            and "esilv.fr" in citation.url
        )
        for citation in response.citations
    ) or any(
        item.search_outcome is not None
        and item.search_outcome.hit_count > 0
        and any("esilv.fr" in citation for citation in item.search_outcome.top_citations)
        for item in response.super_agent_trace
    )
    strategies = [item.selected_strategy for item in response.super_agent_trace]
    assert strategies[0] == "hybrid"
