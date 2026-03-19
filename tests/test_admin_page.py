from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def build_admin_page_script(root: Path, seed_submission: bool = False) -> str:
    return f"""
from pathlib import Path

from app.agent_settings import AgentFeatureSettingsStore
from app.models import CacheStats, RegistrationAnswersRecord, RegistrationRecommendationRecord, RegistrationSubmissionRecord
from app.registration_store import RegistrationStore
from ingestion.uploads import UploadRegistry
from ingestion.vector_store import LocalVectorStore
from ui.admin_page import render_admin_page


class DummyPdfIngestion:
    def ingest(self, uploaded_file):
        raise AssertionError("ingest should not be called")

    def delete_document(self, document_id):
        raise AssertionError("delete should not be called")


class DummySearchAgent:
    def refresh_cache(self):
        return 0

    def clear_cache(self):
        return None

    def cache_stats(self):
        return CacheStats(
            page_count=2,
            stale_count=1,
            last_refresh="2026-03-18T00:00:00+00:00",
            urls=["https://www.esilv.fr/admissions/"],
        )


upload_registry = UploadRegistry(
    Path(r"{(root / "uploads" / "registry.json").as_posix()}"),
    Path(r"{(root / "uploads" / "files").as_posix()}"),
)
vector_store = LocalVectorStore(Path(r"{(root / "vector_store").as_posix()}"))
settings_store = AgentFeatureSettingsStore(Path(r"{(root / "agent_settings.json").as_posix()}"))
registration_store = RegistrationStore(
    Path(r"{(root / "registrations" / "sessions").as_posix()}"),
    Path(r"{(root / "registrations" / "submissions").as_posix()}"),
)
pdf_ingestion = DummyPdfIngestion()
search_agent = DummySearchAgent()

if {seed_submission!r}:
    registration_store.save_submission(
        RegistrationSubmissionRecord(
            id="submission-1",
            conversation_id="conversation-1",
            answers=RegistrationAnswersRecord(
                full_name="Ada Lovelace",
                email="ada@example.com",
                location="Paris",
                program_interest="AI",
                discovery_source="LinkedIn",
                degree_level="master",
                desired_start_date="September 2026",
            ),
            recommendation=RegistrationRecommendationRecord(
                program_name="Data Engineering & AI",
                message="Data Engineering & AI is the closest fit.",
                source_mode="rules",
            ),
        )
    )

render_admin_page(
    upload_registry=upload_registry,
    pdf_ingestion=pdf_ingestion,
    vector_store=vector_store,
    search_agent=search_agent,
    registration_store=registration_store,
    agent_settings_store=settings_store,
    default_generation_model="gemini-2.5-flash",
    default_embedding_model="gemini-embedding-001",
)
"""


def test_admin_page_shows_ingestion_controls(tmp_path):
    at = AppTest.from_string(build_admin_page_script(tmp_path))

    at.run()

    assert at.radio[0].options == ["Ingestion", "Agent parameters", "Application Forms"]
    assert [button.label for button in at.button] == ["Ingest PDFs", "Refresh cache", "Clear cache"]
    assert [metric.label for metric in at.metric] == [
        "Indexed documents",
        "Stored chunks",
        "Tracked PDFs",
        "Cached pages",
        "Stale pages",
    ]
    assert at.markdown[0].value == "### Ingestion"


def test_admin_page_persists_agent_parameters_across_reruns(tmp_path):
    script = build_admin_page_script(tmp_path)
    at = AppTest.from_string(script)

    at.run()
    at.radio[0].set_value("Agent parameters")
    at.run()
    at.checkbox[0].set_value(False)
    at.checkbox[2].set_value(True)
    at.selectbox[0].set_value("gemini-2.5-pro")
    at.selectbox[1].set_value("gemini-embedding-2-preview")
    at.button[0].click()
    at.run()

    assert [checkbox.value for checkbox in at.checkbox] == [False, True, True]
    assert [selectbox.value for selectbox in at.selectbox] == [
        "gemini-2.5-pro",
        "gemini-embedding-2-preview",
    ]
    assert [message.value for message in at.success] == ["Agent parameters saved."]

    reloaded = AppTest.from_string(script)
    reloaded.run()
    reloaded.radio[0].set_value("Agent parameters")
    reloaded.run()

    assert [checkbox.label for checkbox in reloaded.checkbox] == [
        "Enable RAG on uploaded PDFs",
        "Enable ESILV site search",
        "Enable Super Agent iterative orchestration",
    ]
    assert [checkbox.value for checkbox in reloaded.checkbox] == [False, True, True]
    assert [selectbox.label for selectbox in reloaded.selectbox] == [
        "Generation model",
        "Embedding model",
    ]
    assert [selectbox.value for selectbox in reloaded.selectbox] == [
        "gemini-2.5-pro",
        "gemini-embedding-2-preview",
    ]


def test_admin_page_lists_application_forms(tmp_path):
    at = AppTest.from_string(build_admin_page_script(tmp_path, seed_submission=True))

    at.run()
    at.radio[0].set_value("Application Forms")
    at.run()

    assert at.markdown[0].value == "### Application Forms"
    assert at.metric[0].label == "Submitted forms"
    assert at.metric[0].value == "1"
    assert "Ada Lovelace" in at.expander[0].label
