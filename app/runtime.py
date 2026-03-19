from __future__ import annotations

from dataclasses import dataclass

from agents.orchestrator import GoogleGeminiClient, OrchestratorAgent
from agents.retrieval import RetrievalAgent
from agents.web_search import SiteSearchAgent
from app.agent_settings import AgentFeatureSettingsStore
from app.conversation_store import ConversationStore
from app.registration_store import RegistrationStore
from app.settings import AppSettings
from ingestion.pdf_ingestion import GoogleEmbeddingClient, PdfIngestionService
from ingestion.uploads import UploadRegistry
from ingestion.vector_store import LocalVectorStore


@dataclass
class AppServices:
    settings: AppSettings
    conversation_store: ConversationStore
    registration_store: RegistrationStore
    agent_settings_store: AgentFeatureSettingsStore
    upload_registry: UploadRegistry
    vector_store: LocalVectorStore
    pdf_ingestion: PdfIngestionService
    retrieval_agent: RetrievalAgent
    search_agent: SiteSearchAgent
    orchestrator: OrchestratorAgent


def build_services(settings: AppSettings | None = None) -> AppServices:
    settings = settings or AppSettings()
    settings.ensure_directories()

    conversation_store = ConversationStore(settings.conversations_dir)
    registration_store = RegistrationStore(
        settings.registration_sessions_dir,
        settings.registration_submissions_dir,
    )
    agent_settings_store = AgentFeatureSettingsStore(settings.agent_settings_path)
    agent_runtime_settings = agent_settings_store.load()
    upload_registry = UploadRegistry(settings.upload_registry_path, settings.upload_files_dir)
    vector_store = LocalVectorStore(settings.vector_store_dir)
    generation_model = agent_runtime_settings.generation_model or settings.gemini_model
    embedding_model = agent_runtime_settings.embedding_model or settings.gemini_embedding_model
    embedding_client = GoogleEmbeddingClient(
        api_key=settings.gemini_api_key,
        model=embedding_model,
    )
    pdf_ingestion = PdfIngestionService(
        upload_registry=upload_registry,
        vector_store=vector_store,
        embedding_client=embedding_client,
        docling_artifacts_dir=settings.docling_artifacts_dir,
    )
    retrieval_agent = RetrievalAgent(
        vector_store=vector_store,
        embedding_client=embedding_client,
    )
    search_agent = SiteSearchAgent(
        cache_dir=settings.site_cache_dir,
        allowed_domains=settings.allowed_domains,
        api_key=settings.tavily_api_key,
        ttl_hours=settings.site_cache_ttl_hours,
    )
    llm_client = GoogleGeminiClient(api_key=settings.gemini_api_key, model=generation_model)
    orchestrator = OrchestratorAgent(
        conversation_store=conversation_store,
        registration_store=registration_store,
        search_agent=search_agent,
        retrieval_agent=retrieval_agent,
        llm_client=llm_client,
        feature_settings_store=agent_settings_store,
        max_search_hits=settings.max_search_hits,
    )
    return AppServices(
        settings=settings,
        conversation_store=conversation_store,
        registration_store=registration_store,
        agent_settings_store=agent_settings_store,
        upload_registry=upload_registry,
        vector_store=vector_store,
        pdf_ingestion=pdf_ingestion,
        retrieval_agent=retrieval_agent,
        search_agent=search_agent,
        orchestrator=orchestrator,
    )
