from app.agent_settings import AgentFeatureSettingsStore
from app.models import AgentFeatureSettings


def test_agent_settings_defaults_when_file_is_missing(tmp_path):
    store = AgentFeatureSettingsStore(tmp_path / "agent_settings.json")

    settings = store.load()

    assert settings == AgentFeatureSettings(
        rag_enabled=True,
        web_search_enabled=True,
        super_agent_enabled=False,
        generation_model=None,
        embedding_model=None,
    )


def test_agent_settings_persist_after_save_and_reload(tmp_path):
    path = tmp_path / "agent_settings.json"
    store = AgentFeatureSettingsStore(path)
    store.save(
        AgentFeatureSettings(
            rag_enabled=False,
            web_search_enabled=True,
            super_agent_enabled=True,
            generation_model="gemini-2.5-pro",
            embedding_model="gemini-embedding-2-preview",
        )
    )

    reloaded = AgentFeatureSettingsStore(path).load()

    assert reloaded == AgentFeatureSettings(
        rag_enabled=False,
        web_search_enabled=True,
        super_agent_enabled=True,
        generation_model="gemini-2.5-pro",
        embedding_model="gemini-embedding-2-preview",
    )


def test_agent_settings_fall_back_to_defaults_on_malformed_json(tmp_path):
    path = tmp_path / "agent_settings.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = AgentFeatureSettingsStore(path)

    settings = store.load()

    assert settings == AgentFeatureSettings(
        rag_enabled=True,
        web_search_enabled=True,
        super_agent_enabled=False,
        generation_model=None,
        embedding_model=None,
    )
