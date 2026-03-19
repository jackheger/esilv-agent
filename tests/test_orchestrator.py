import pytest

from app.models import AgentFeatureSettings
from agents.orchestrator import OrchestratorAgent
from app.conversation_store import ConversationStore
from app.models import MessageRecord, RetrievalHit, SearchHit
from app.registration_store import RegistrationStore


class FakeLlmClient:
    def __init__(self, structured_responses, text_responses, configured: bool = True) -> None:
        self.structured_responses = list(structured_responses)
        self.text_responses = list(text_responses)
        self._configured = configured
        self.text_calls = []
        self.structured_calls = []

    @property
    def configured(self) -> bool:
        return self._configured

    def generate_structured(self, prompt, system_instruction, schema, temperature=0.0):
        self.structured_calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "schema": schema,
                "temperature": temperature,
            }
        )
        return self.structured_responses.pop(0)

    def generate_text(self, prompt, system_instruction, temperature=0.2):
        self.text_calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "temperature": temperature,
            }
        )
        return self.text_responses.pop(0)


class ExplodingLlmClient(FakeLlmClient):
    def generate_text(self, prompt, system_instruction, temperature=0.2):
        raise RuntimeError("quota exceeded")


class FakeSearchAgent:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        return list(self.results)


class FakeRetrievalAgent:
    def __init__(self, results, weak: bool = False):
        self.results = list(results)
        self.weak = weak
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        return list(self.results)

    def is_weak(self, hits):
        return self.weak


class InMemoryFeatureSettingsStore:
    def __init__(self, settings: AgentFeatureSettings | None = None) -> None:
        self.settings = settings or AgentFeatureSettings()

    def load(self) -> AgentFeatureSettings:
        return self.settings


class DummySuperAgent:
    def __init__(self, response: MessageRecord) -> None:
        self.response = response
        self.calls = []

    def run(self, messages, user_text, initial_action, initial_query, feature_settings):
        self.calls.append(
            {
                "messages": messages,
                "user_text": user_text,
                "initial_action": initial_action,
                "initial_query": initial_query,
                "feature_settings": feature_settings,
            }
        )
        return self.response


def make_orchestrator(
    store,
    search,
    retrieval,
    llm,
    settings: AgentFeatureSettings | None = None,
):
    registration_store = RegistrationStore(
        store.conversations_dir.parent / "registrations" / "sessions",
        store.conversations_dir.parent / "registrations" / "submissions",
    )
    return OrchestratorAgent(
        conversation_store=store,
        registration_store=registration_store,
        search_agent=search,
        retrieval_agent=retrieval,
        llm_client=llm,
        feature_settings_store=InMemoryFeatureSettingsStore(settings),
        max_search_hits=5,
    )


def test_orchestrator_routes_to_search_and_attaches_citations(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "search_query": "ESILV admissions"}],
        text_responses=["Admissions answer from sources"],
    )
    search = FakeSearchAgent(
        [
            SearchHit(
                url="https://www.esilv.fr/admissions/",
                title="Admissions | ESILV",
                snippet="Admissions requirements and application steps.",
                score=9.0,
                lexical_overlap=0.5,
                expanded_overlap=0.5,
                fetched_at="2026-03-18T12:00:00+00:00",
            )
        ]
    )
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True),
    )

    response = orchestrator.handle_turn(conversation.id, "How do ESILV admissions work?")

    assert search.queries == [("ESILV admissions", 5)]
    assert retrieval.queries == []
    assert "Admissions answer from sources" in response.content
    assert response.citations[0].kind == "web"
    assert response.citations[0].url == "https://www.esilv.fr/admissions/"
    assert response.pending_action == "registration"
    assert len(store.load(conversation.id).messages) == 2


def test_orchestrator_answers_directly_without_search(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "direct"}],
        text_responses=["Hello from ESILV assistant"],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(store, search, retrieval, llm)

    response = orchestrator.handle_turn(conversation.id, "Hello")

    assert search.queries == []
    assert retrieval.queries == []
    assert response.content == "Hello from ESILV assistant"


def test_orchestrator_overrides_direct_route_for_document_anchored_requests(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "direct"}],
        text_responses=["The uploaded PDF says the admissions office is open from 9am to 5pm."],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent(
        [
            RetrievalHit(
                document_id="doc-1",
                filename="admissions.pdf",
                page_number=2,
                snippet="Admissions office hours: Monday to Friday, 9am to 5pm.",
                score=0.94,
                cosine_score=0.9,
                lexical_overlap=0.5,
            )
        ]
    )
    orchestrator = make_orchestrator(store, search, retrieval, llm)

    response = orchestrator.handle_turn(conversation.id, "According to the uploaded PDF, what are the admissions office hours?")

    assert retrieval.queries == [("According to the uploaded PDF, what are the admissions office hours?", 5)]
    assert search.queries == [("According to the uploaded PDF, what are the admissions office hours?", 5)]
    assert response.citations[0].kind == "document"


def test_orchestrator_routes_to_retrieval_and_attaches_document_citations(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "retrieve", "tool_query": "according to the uploaded pdf admissions hours"}],
        text_responses=["The uploaded PDF says the admissions office is open from 9am to 5pm."],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent(
        [
            RetrievalHit(
                document_id="doc-1",
                filename="admissions.pdf",
                page_number=2,
                snippet="Admissions office hours: Monday to Friday, 9am to 5pm.",
                score=0.94,
                cosine_score=0.9,
                lexical_overlap=0.5,
            )
        ]
    )
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=False),
    )

    response = orchestrator.handle_turn(conversation.id, "According to the uploaded PDF, what are the admissions office hours?")

    assert retrieval.queries == [("according to the uploaded pdf admissions hours", 5)]
    assert search.queries == []
    assert response.content == "The uploaded PDF says the admissions office is open from 9am to 5pm."
    assert response.citations[0].kind == "document"
    assert response.citations[0].title == "admissions.pdf"
    assert response.citations[0].page_number == 2
    assert response.citations[0].url is None


def test_orchestrator_returns_uncertain_message_on_weak_retrieval_without_search(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "retrieve", "tool_query": "uploaded report storage"}],
        text_responses=[],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([], weak=True)
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=False),
    )

    response = orchestrator.handle_turn(conversation.id, "According to the uploaded report, what storage setup was used?")

    assert retrieval.queries == [("uploaded report storage", 5)]
    assert search.queries == []
    assert response.citations == []
    assert "PDF passage close enough" in response.content


def test_orchestrator_returns_interactive_clarification_on_weak_search(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "tool_query": "Does ESILV propose SQL courses?"}],
        text_responses=[],
    )
    search = FakeSearchAgent(
        [
            SearchHit(
                url="https://www.esilv.fr/formations/",
                title="Programs | ESILV",
                snippet="Overview of programs and majors.",
                score=9.0,
                lexical_overlap=0.0,
                expanded_overlap=0.05,
                fetched_at="2026-03-18T12:00:00+00:00",
            )
        ]
    )
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True),
    )

    response = orchestrator.handle_turn(conversation.id, "Does ESILV propose SQL courses?")

    assert search.queries == [("Does ESILV propose SQL courses?", 5)]
    assert response.pending_action == "search"
    assert response.pending_query is not None
    assert "database-related courses" in response.content


def test_orchestrator_reuses_pending_query_after_yes_follow_up(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "tool_query": "Does ESILV propose SQL courses?"}],
        text_responses=["Yes, ESILV proposes database-oriented courses."],
    )
    search = FakeSearchAgent(
        [
            SearchHit(
                url="https://www.esilv.fr/formations/",
                title="Programs | ESILV",
                snippet="Overview of programs and majors.",
                score=9.0,
                lexical_overlap=0.0,
                expanded_overlap=0.05,
                fetched_at="2026-03-18T12:00:00+00:00",
            )
        ]
    )
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True),
    )

    first_response = orchestrator.handle_turn(conversation.id, "Does ESILV propose SQL courses?")
    search.results = [
        SearchHit(
            url="https://www.esilv.fr/course-database/",
            title="Database courses | ESILV",
            snippet="Database systems and relational modeling.",
            score=12.0,
            lexical_overlap=0.3,
            expanded_overlap=0.4,
            fetched_at="2026-03-18T12:10:00+00:00",
        )
    ]

    second_response = orchestrator.handle_turn(conversation.id, "yes")

    assert first_response.pending_query is not None
    assert len(search.queries) == 2
    assert search.queries[1] == (first_response.pending_query, 5)
    assert second_response.content == "Yes, ESILV proposes database-oriented courses."
    assert second_response.citations[0].kind == "web"


def test_orchestrator_returns_temporary_failure_message_when_generation_errors(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = ExplodingLlmClient(
        structured_responses=[{"action": "direct"}],
        text_responses=[],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(store, search, retrieval, llm)

    response = orchestrator.handle_turn(conversation.id, "Hello")

    assert response.citations == []
    assert "temporarily unavailable" in response.content


def test_orchestrator_falls_back_to_direct_when_rag_is_disabled(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "retrieve", "tool_query": "uploaded pdf office hours"}],
        text_responses=["I cannot verify that from the uploaded PDF right now."],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True),
    )

    response = orchestrator.handle_turn(conversation.id, "According to the uploaded PDF, what are the office hours?")

    assert retrieval.queries == []
    assert search.queries == []
    assert response.content == "I cannot verify that from the uploaded PDF right now."
    assert response.citations == []


def test_orchestrator_falls_back_to_retrieval_when_search_is_disabled(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "search_query": "ESILV admissions"}],
        text_responses=["The uploaded PDF says the admissions office is open from 9am to 5pm."],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent(
        [
            RetrievalHit(
                document_id="doc-1",
                filename="admissions.pdf",
                page_number=2,
                snippet="Admissions office hours: Monday to Friday, 9am to 5pm.",
                score=0.94,
                cosine_score=0.9,
                lexical_overlap=0.5,
            )
        ]
    )
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=False),
    )

    response = orchestrator.handle_turn(conversation.id, "According to the uploaded PDF, what are the admissions office hours?")

    assert search.queries == []
    assert retrieval.queries == [("According to the uploaded PDF, what are the admissions office hours?", 5)]
    assert response.citations[0].kind == "document"
    assert response.citations[0].title == "admissions.pdf"


def test_orchestrator_runs_hybrid_single_pass_when_both_tools_are_enabled(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "search_query": "ESILV admissions"}],
        text_responses=["Hybrid answer from both sources."],
    )
    search = FakeSearchAgent(
        [
            SearchHit(
                url="https://www.esilv.fr/admissions/",
                title="Admissions | ESILV",
                snippet="Admissions requirements and application steps.",
                score=9.0,
                lexical_overlap=0.5,
                expanded_overlap=0.5,
                fetched_at="2026-03-18T12:00:00+00:00",
            )
        ]
    )
    retrieval = FakeRetrievalAgent(
        [
            RetrievalHit(
                document_id="doc-1",
                filename="admissions.pdf",
                page_number=3,
                snippet="Admissions office details from the uploaded PDF.",
                score=0.93,
                cosine_score=0.9,
                lexical_overlap=0.55,
            )
        ],
        weak=False,
    )
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=False),
    )

    response = orchestrator.handle_turn(conversation.id, "How do ESILV admissions work?")

    assert retrieval.queries == [("How do ESILV admissions work?", 5)]
    assert search.queries == [("How do ESILV admissions work?", 5)]
    assert "Hybrid answer from both sources." in response.content
    assert {citation.kind for citation in response.citations} == {"document", "web"}
    assert response.pending_action == "registration"


def test_orchestrator_falls_back_to_web_search_when_retrieval_is_weak(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "search_query": "ESILV admissions"}],
        text_responses=["Admissions answer from sources"],
    )
    search = FakeSearchAgent(
        [
            SearchHit(
                url="https://www.esilv.fr/admissions/",
                title="Admissions | ESILV",
                snippet="Admissions requirements and application steps.",
                score=9.0,
                lexical_overlap=0.5,
                expanded_overlap=0.5,
                fetched_at="2026-03-18T12:00:00+00:00",
            )
        ]
    )
    retrieval = FakeRetrievalAgent([], weak=True)
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=False),
    )

    response = orchestrator.handle_turn(conversation.id, "How do ESILV admissions work?")

    assert retrieval.queries == [("How do ESILV admissions work?", 5)]
    assert search.queries == [("How do ESILV admissions work?", 5)]
    assert "Admissions answer from sources" in response.content
    assert response.citations[0].kind == "web"
    assert response.citations[0].url == "https://www.esilv.fr/admissions/"
    assert response.pending_action == "registration"


def test_orchestrator_forces_direct_answer_when_all_tools_are_disabled(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[],
        text_responses=["Direct answer only."],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=False),
    )

    response = orchestrator.handle_turn(conversation.id, "How do ESILV admissions work?")

    assert search.queries == []
    assert retrieval.queries == []
    assert "Direct answer only." in response.content
    assert response.citations == []
    assert response.pending_action == "registration"


def test_orchestrator_delegates_to_super_agent_for_retrieval_or_search_turns(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "retrieve", "tool_query": "plm functions"}],
        text_responses=[],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=False, super_agent_enabled=True),
    )
    dummy_super_agent = DummySuperAgent(
        MessageRecord(
            role="assistant",
            content="Super agent answer",
            orchestration_mode="super",
            super_agent_stop_reason="answer_sufficient",
        )
    )
    orchestrator.super_agent = dummy_super_agent

    response = orchestrator.handle_turn(conversation.id, "According to the PLM document, what are the main functionalities?")

    assert response.content == "Super agent answer"
    assert len(dummy_super_agent.calls) == 1
    assert dummy_super_agent.calls[0]["initial_action"] == "retrieve"
    assert dummy_super_agent.calls[0]["initial_query"] == "plm functions"
    assert retrieval.queries == []
    assert search.queries == []


def test_orchestrator_does_not_delegate_direct_turns_to_super_agent(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "direct"}],
        text_responses=["Direct answer still used."],
    )
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=True),
    )
    dummy_super_agent = DummySuperAgent(MessageRecord(role="assistant", content="Should not be used"))
    orchestrator.super_agent = dummy_super_agent

    response = orchestrator.handle_turn(conversation.id, "Hello")

    assert response.content == "Direct answer still used."
    assert dummy_super_agent.calls == []


def test_orchestrator_starts_registration_immediately_for_contact_requests(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(structured_responses=[], text_responses=[], configured=False)
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(store, search, retrieval, llm)

    response = orchestrator.handle_turn(conversation.id, "Please contact me about ESILV")

    assert "full name" in response.content.lower()
    assert search.queries == []
    assert retrieval.queries == []
    assert orchestrator.registration_agent.registration_store.load_session(conversation.id) is not None


def test_orchestrator_starts_registration_after_affirmative_cta_follow_up(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "search_query": "ESILV admissions"}],
        text_responses=["Admissions answer from sources"],
    )
    search = FakeSearchAgent(
        [
            SearchHit(
                url="https://www.esilv.fr/admissions/",
                title="Admissions | ESILV",
                snippet="Admissions requirements and application steps.",
                score=9.0,
                lexical_overlap=0.5,
                expanded_overlap=0.5,
                fetched_at="2026-03-19T12:00:00+00:00",
            )
        ]
    )
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True),
    )

    first_response = orchestrator.handle_turn(conversation.id, "How do ESILV admissions work?")
    second_response = orchestrator.handle_turn(conversation.id, "yes")

    assert first_response.pending_action == "registration"
    assert "full name" in second_response.content.lower()
    assert orchestrator.registration_agent.registration_store.load_session(conversation.id) is not None


def test_orchestrator_routes_active_registration_turns_before_normal_routing(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(structured_responses=[], text_responses=[], configured=False)
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(store, search, retrieval, llm)

    orchestrator.handle_turn(conversation.id, "Please contact me about ESILV")
    response = orchestrator.handle_turn(conversation.id, "Ada Lovelace")

    assert "email" in response.content.lower()
    assert search.queries == []
    assert retrieval.queries == []


@pytest.mark.parametrize(
    "prompt",
    [
        "I am interested in joining ESILV",
        "I am interested in ESILV courses",
        "I want to join ESILV",
        "Help me choose courses at ESILV",
    ],
)
def test_orchestrator_immediately_starts_registration_for_lead_intent_phrases(tmp_path, prompt):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(structured_responses=[], text_responses=[], configured=False)
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(store, search, retrieval, llm)

    response = orchestrator.handle_turn(conversation.id, prompt)

    assert "full name" in response.content.lower()
    assert orchestrator.registration_agent.registration_store.load_session(conversation.id) is not None
    assert search.queries == []
    assert retrieval.queries == []


def test_orchestrator_adds_registration_cta_for_course_discovery_questions(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "search_query": "ESILV courses"}],
        text_responses=["Here are the main ESILV course families."],
    )
    search = FakeSearchAgent(
        [
            SearchHit(
                url="https://www.esilv.fr/en/programmes/",
                title="Programs | ESILV",
                snippet="Overview of programs and course families.",
                score=9.0,
                lexical_overlap=0.4,
                expanded_overlap=0.4,
                fetched_at="2026-03-19T12:00:00+00:00",
            )
        ]
    )
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True),
    )

    response = orchestrator.handle_turn(conversation.id, "Can you tell me about ESILV courses?")

    assert "Here are the main ESILV course families." in response.content
    assert response.pending_action == "registration"


def test_orchestrator_pins_english_language_for_grounded_answers(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    llm = FakeLlmClient(
        structured_responses=[{"action": "search", "search_query": "ESILV courses"}],
        text_responses=["English grounded answer."],
    )
    search = FakeSearchAgent(
        [
            SearchHit(
                url="https://www.esilv.fr/formations/",
                title="Formations | ESILV",
                snippet="Les programmes d'ingenierie et les cours sont presentes ici.",
                score=9.0,
                lexical_overlap=0.5,
                expanded_overlap=0.5,
                fetched_at="2026-03-19T12:00:00+00:00",
            )
        ]
    )
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(
        store,
        search,
        retrieval,
        llm,
        settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True),
    )

    response = orchestrator.handle_turn(conversation.id, "What courses does ESILV offer?")

    assert response.content.startswith("English grounded answer.")
    assert llm.text_calls
    assert "Required answer language: English." in llm.text_calls[-1]["prompt"]
    assert "Required response language: English." in llm.text_calls[-1]["system_instruction"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Can you explain ESILV admissions in English?", False),
        ("Pouvez-vous expliquer les admissions a l'ESILV ?", True),
    ],
)
def test_orchestrator_language_detection_matches_user_query_language(tmp_path, query, expected):
    store = ConversationStore(tmp_path / "conversations")
    llm = FakeLlmClient(structured_responses=[], text_responses=[], configured=False)
    search = FakeSearchAgent([])
    retrieval = FakeRetrievalAgent([])
    orchestrator = make_orchestrator(store, search, retrieval, llm)

    assert orchestrator._looks_french(query) is expected
