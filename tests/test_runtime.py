from app.agent_settings import AgentFeatureSettingsStore
from app.models import AgentFeatureSettings
from app.runtime import build_services
from app.settings import AppSettings


def test_build_services_uses_admin_selected_models(tmp_path):
    settings = AppSettings(
        app_data_dir=tmp_path,
        gemini_model="gemini-2.5-flash-lite",
        gemini_embedding_model="gemini-embedding-001",
    )
    settings.ensure_directories()
    AgentFeatureSettingsStore(settings.agent_settings_path).save(
        AgentFeatureSettings(
            rag_enabled=True,
            web_search_enabled=True,
            super_agent_enabled=False,
            generation_model="gemini-2.5-pro",
            embedding_model="gemini-embedding-2-preview",
        )
    )

    services = build_services(settings)

    assert services.orchestrator.llm_client.model == "gemini-2.5-pro"
    assert services.pdf_ingestion.embedding_client.model == "gemini-embedding-2-preview"
    assert services.retrieval_agent.embedding_client.model == "gemini-embedding-2-preview"


def test_build_services_falls_back_to_env_models_when_admin_selection_missing(tmp_path):
    settings = AppSettings(
        app_data_dir=tmp_path,
        gemini_model="gemini-2.5-flash-lite",
        gemini_embedding_model="gemini-embedding-2-preview",
    )
    settings.ensure_directories()
    AgentFeatureSettingsStore(settings.agent_settings_path).save(
        AgentFeatureSettings(
            rag_enabled=True,
            web_search_enabled=True,
            super_agent_enabled=False,
        )
    )

    services = build_services(settings)

    assert services.orchestrator.llm_client.model == "gemini-2.5-flash-lite"
    assert services.pdf_ingestion.embedding_client.model == "gemini-embedding-2-preview"
