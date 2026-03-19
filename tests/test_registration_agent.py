from app.models import AgentFeatureSettings, CitationRecord, MessageRecord, RegistrationAnswersRecord, RetrievalHit, SearchHit
from app.registration_store import RegistrationStore
from agents.registration import RegistrationAgent


class FakeLlmClient:
    def __init__(self, configured: bool, *, text_responses=None, structured_responses=None) -> None:
        self._configured = configured
        self.text_responses = list(text_responses or [])
        self.structured_responses = list(structured_responses or [])

    @property
    def configured(self) -> bool:
        return self._configured

    def generate_text(self, prompt, system_instruction, temperature=0.2):
        if self.text_responses:
            return self.text_responses.pop(0)
        marker = "Fallback question:"
        if marker in prompt:
            return prompt.split(marker, 1)[1].strip()
        return "What is your full name?"

    def generate_structured(self, prompt, system_instruction, schema, temperature=0.0):
        if self.structured_responses:
            return self.structured_responses.pop(0)
        return schema.model_validate(RegistrationAnswersRecord())


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


class FakeSuperAgent:
    def __init__(self, response: MessageRecord) -> None:
        self.response = response
        self.calls = []

    def run(self, messages, user_text, initial_action, initial_query, feature_settings):
        self.calls.append((messages, user_text, initial_action, initial_query, feature_settings))
        return self.response


def make_agent(
    tmp_path,
    *,
    llm_configured=True,
    llm_text_responses=None,
    llm_structured_responses=None,
    search_results=None,
    retrieval_results=None,
    retrieval_weak=False,
    super_response=None,
    search_message="MSc Computer Science & Data Science is the best fit.",
    retrieval_message="Master in Engineering is the strongest fit from the uploaded material.",
):
    registration_store = RegistrationStore(
        tmp_path / "registrations" / "sessions",
        tmp_path / "registrations" / "submissions",
    )
    search_agent = FakeSearchAgent(search_results or [])
    retrieval_agent = FakeRetrievalAgent(retrieval_results or [], weak=retrieval_weak)
    super_agent = FakeSuperAgent(super_response or MessageRecord(role="assistant", content="No grounded answer"))
    return RegistrationAgent(
        registration_store=registration_store,
        search_agent=search_agent,
        retrieval_agent=retrieval_agent,
        llm_client=FakeLlmClient(
            llm_configured,
            text_responses=llm_text_responses,
            structured_responses=llm_structured_responses,
        ),
        super_agent=super_agent,
        max_search_hits=5,
        looks_french=lambda text: "bonjour" in text.lower() or "oui" in text.lower(),
        search_is_weak=lambda query, hits: not hits,
        citations_from_hits=lambda hits: [CitationRecord(kind="web", title=hits[0].title, url=hits[0].url)] if hits else [],
        citations_from_retrieval_hits=(
            lambda hits: [CitationRecord(kind="document", title=hits[0].filename, page_number=hits[0].page_number)]
            if hits
            else []
        ),
        retrieval_answer_builder=lambda messages, query, hits: retrieval_message,
        search_answer_builder=lambda messages, query, hits: search_message,
    )


def complete_registration(agent: RegistrationAgent, conversation_id: str, feature_settings: AgentFeatureSettings):
    messages = [MessageRecord(role="user", content="I need help choosing a program")]
    agent.start_session(conversation_id, "I need help choosing a program")
    prompts = [
        "Ada Lovelace",
        "ada@example.com",
        "Paris",
        "AI and data science",
        "LinkedIn",
        "master",
        "September 2026",
    ]
    response = None
    for answer in prompts:
        response = agent.continue_session(conversation_id, messages, answer, feature_settings)
        assert response is not None
    return response


def test_registration_agent_progresses_and_persists_submission_with_rules_fallback(tmp_path):
    agent = make_agent(tmp_path, llm_configured=False)
    response = complete_registration(
        agent,
        "conversation-1",
        AgentFeatureSettings(rag_enabled=False, web_search_enabled=False, super_agent_enabled=False),
    )

    assert response is not None
    assert "Recommended program" in response.content
    submissions = agent.registration_store.list_submissions()
    assert len(submissions) == 1
    assert submissions[0].recommendation is not None
    assert submissions[0].recommendation.source_mode == "rules"
    assert submissions[0].recommendation.program_name == "Data Engineering & AI"
    assert agent.registration_store.load_session("conversation-1") is None


def test_registration_agent_retries_invalid_email_without_advancing(tmp_path):
    agent = make_agent(tmp_path, llm_configured=False)
    agent.start_session("conversation-1", "contact me")

    first_response = agent.continue_session(
        "conversation-1",
        [MessageRecord(role="user", content="contact me")],
        "Ada Lovelace",
        AgentFeatureSettings(),
    )
    second_response = agent.continue_session(
        "conversation-1",
        [MessageRecord(role="user", content="contact me")],
        "not-an-email",
        AgentFeatureSettings(),
    )

    assert first_response is not None
    assert "email" in first_response.content.lower()
    assert second_response is not None
    assert "valid email" in second_response.content.lower()
    session = agent.registration_store.load_session("conversation-1")
    assert session is not None
    assert session.current_field == "email"


def test_registration_agent_accepts_uncertain_degree_level_and_moves_forward(tmp_path):
    agent = make_agent(tmp_path, llm_configured=True)
    feature_settings = AgentFeatureSettings(rag_enabled=False, web_search_enabled=False, super_agent_enabled=False)
    messages = [MessageRecord(role="user", content="I need help choosing a program")]

    agent.start_session("conversation-1", "I need help choosing a program")
    agent.continue_session("conversation-1", messages, "Ada Lovelace", feature_settings)
    agent.continue_session("conversation-1", messages, "ada@example.com", feature_settings)
    agent.continue_session("conversation-1", messages, "Paris", feature_settings)
    agent.continue_session("conversation-1", messages, "AI and data science", feature_settings)
    degree_prompt = agent.continue_session("conversation-1", messages, "LinkedIn", feature_settings)

    assert degree_prompt is not None
    assert "bachelor" in degree_prompt.content.lower() or "master" in degree_prompt.content.lower()

    start_date_prompt = agent.continue_session("conversation-1", messages, "I don't know yet", feature_settings)

    assert start_date_prompt is not None
    assert "start" in start_date_prompt.content.lower()
    session = agent.registration_store.load_session("conversation-1")
    assert session is not None
    assert session.current_field == "desired_start_date"
    assert session.answers.degree_level == "I don't know yet"

    completion = agent.continue_session("conversation-1", messages, "September 2026", feature_settings)

    assert completion is not None
    submission = agent.registration_store.list_submissions()[0]
    assert submission.answers.degree_level == "I don't know yet"


def test_registration_agent_uses_search_recommendation_when_available(tmp_path):
    agent = make_agent(
        tmp_path,
        llm_configured=True,
        search_results=[
            SearchHit(
                url="https://www.esilv.fr/formations/msc-computer-science-data-science/",
                title="MSc Computer Science & Data Science - ESILV",
                snippet="A master's focused on computer science and data science.",
                score=10.0,
                lexical_overlap=0.5,
                expanded_overlap=0.5,
                fetched_at="2026-03-19T00:00:00+00:00",
            )
        ],
    )

    complete_registration(
        agent,
        "conversation-1",
        AgentFeatureSettings(rag_enabled=False, web_search_enabled=True, super_agent_enabled=False),
    )

    recommendation = agent.registration_store.list_submissions()[0].recommendation
    assert recommendation is not None
    assert recommendation.source_mode == "search"
    assert recommendation.program_name == "MSc Computer Science & Data Science"
    assert agent.search_agent.queries


def test_registration_agent_uses_retrieval_recommendation_when_search_is_disabled(tmp_path):
    agent = make_agent(
        tmp_path,
        llm_configured=True,
        retrieval_results=[
            RetrievalHit(
                document_id="doc-1",
                filename="esilv-programs.pdf",
                page_number=2,
                snippet="Master in Engineering is the broad engineering path.",
                score=0.95,
                cosine_score=0.9,
                lexical_overlap=0.4,
            )
        ],
    )

    complete_registration(
        agent,
        "conversation-1",
        AgentFeatureSettings(rag_enabled=True, web_search_enabled=False, super_agent_enabled=False),
    )

    recommendation = agent.registration_store.list_submissions()[0].recommendation
    assert recommendation is not None
    assert recommendation.source_mode == "retrieve"
    assert recommendation.program_name == "Master in Engineering"
    assert agent.retrieval_agent.queries


def test_registration_agent_uses_super_agent_when_enabled(tmp_path):
    agent = make_agent(
        tmp_path,
        llm_configured=True,
        super_response=MessageRecord(
            role="assistant",
            content="Data Engineering & AI is the best grounded fit.",
            citations=[CitationRecord(kind="web", title="AI program", url="https://www.esilv.fr/programs/ai")],
        ),
    )

    complete_registration(
        agent,
        "conversation-1",
        AgentFeatureSettings(rag_enabled=False, web_search_enabled=True, super_agent_enabled=True),
    )

    recommendation = agent.registration_store.list_submissions()[0].recommendation
    assert recommendation is not None
    assert recommendation.source_mode == "super"
    assert recommendation.program_name == "Data Engineering & AI"
    assert len(agent.super_agent.calls) == 1
