from app.models import (
    AgentFeatureSettings,
    CitationRecord,
    MessageRecord,
    RetrievalHit,
    SearchHit,
)
from agents.super_agent import SuperAgent


class FakeEvaluationLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_structured(self, prompt, system_instruction, schema, temperature=0.0):
        self.prompts.append(prompt)
        return schema.model_validate(self.responses.pop(0))


class FakeSearchAgent:
    def __init__(self, results_by_query):
        self.results_by_query = dict(results_by_query)
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        return list(self.results_by_query.get(query, []))


class FakeRetrievalAgent:
    def __init__(self, results_by_query, weak_by_query):
        self.results_by_query = dict(results_by_query)
        self.weak_by_query = dict(weak_by_query)
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        return list(self.results_by_query.get(query, []))

    def is_weak(self, hits):
        if not self.queries:
            return True
        return self.weak_by_query.get(self.queries[-1][0], True)


def build_super_agent(llm, search_agent, retrieval_agent):
    return SuperAgent(
        search_agent=search_agent,
        retrieval_agent=retrieval_agent,
        llm_client=llm,
        max_search_hits=5,
        format_messages=lambda messages: "\n".join(f"{message.role}: {message.content}" for message in messages),
        search_is_weak=lambda query, hits: not hits or hits[0].score < 5,
        citations_from_hits=lambda hits: [
            CitationRecord(kind="web", title=hit.title, url=hit.url) for hit in hits[:3]
        ],
        citations_from_retrieval_hits=lambda hits: [
            CitationRecord(kind="document", title=hit.filename, page_number=hit.page_number)
            for hit in hits[:3]
        ],
        retrieval_answer_builder=lambda messages, user_text, hits: f"retrieval answer for {hits[0].filename}",
        search_answer_builder=lambda messages, user_text, hits: f"search answer for {hits[0].title}",
        hybrid_answer_builder=lambda messages, user_text, retrieval_hits, search_hits: (
            f"hybrid answer using {retrieval_hits[0].filename} and {search_hits[0].title}"
        ),
        looks_french=lambda text: False,
        is_retrieval_intent=lambda lowered: any(token in lowered for token in ("pdf", "document", "uploaded")),
        is_search_intent=lambda lowered: any(token in lowered for token in ("esilv", "admissions")),
        has_independent_search_intent=lambda lowered: "esilv" in lowered and "document" not in lowered,
    )


def test_super_agent_retries_rag_only_with_rewritten_query():
    retrieval_hits = {
        "original query": [
            RetrievalHit(
                document_id="doc-1",
                filename="PLM.pdf",
                page_number=4,
                snippet="Weak hit",
                score=0.41,
                cosine_score=0.4,
                lexical_overlap=0.1,
            )
        ],
        "refined query": [
            RetrievalHit(
                document_id="doc-1",
                filename="PLM.pdf",
                page_number=17,
                snippet="Strong hit",
                score=0.82,
                cosine_score=0.8,
                lexical_overlap=0.5,
            )
        ],
    }
    retrieval_agent = FakeRetrievalAgent(
        results_by_query=retrieval_hits,
        weak_by_query={"original query": True, "refined query": False},
    )
    search_agent = FakeSearchAgent({})
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": False,
                "is_grounded": False,
                "missing_information": ["main PLM functionalities"],
                "unsupported_claims": [],
                "next_strategy": "retrieve",
                "rewritten_query": "refined query",
                "stop_reason": None,
            },
            {
                "is_sufficient": True,
                "is_grounded": True,
                "missing_information": [],
                "unsupported_claims": [],
                "next_strategy": None,
                "rewritten_query": None,
                "stop_reason": "answer_sufficient",
            },
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)

    message = agent.run(
        messages=[MessageRecord(role="user", content="original question")],
        user_text="original question",
        initial_action="retrieve",
        initial_query="original query",
        feature_settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=False, super_agent_enabled=True),
    )

    assert retrieval_agent.queries == [("original query", 5), ("refined query", 5)]
    assert search_agent.queries == []
    assert message.content == "retrieval answer for PLM.pdf"
    assert message.super_agent_stop_reason == "answer_sufficient"
    assert len(message.super_agent_trace) == 2
    assert message.super_agent_trace[0].rewritten_query == "refined query"


def test_super_agent_retries_web_only_with_rewritten_query():
    search_hits = {
        "admissions initial": [
            SearchHit(
                url="https://www.esilv.fr/weak",
                title="Weak result",
                snippet="Weak snippet",
                score=3.0,
                lexical_overlap=0.1,
                expanded_overlap=0.1,
                fetched_at="2026-03-18T12:00:00+00:00",
            )
        ],
        "admissions refined": [
            SearchHit(
                url="https://www.esilv.fr/admissions/",
                title="Admissions | ESILV",
                snippet="Admissions details",
                score=9.0,
                lexical_overlap=0.5,
                expanded_overlap=0.6,
                fetched_at="2026-03-18T12:05:00+00:00",
            )
        ],
    }
    retrieval_agent = FakeRetrievalAgent({}, {})
    search_agent = FakeSearchAgent(search_hits)
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": False,
                "is_grounded": False,
                "missing_information": ["admission steps"],
                "unsupported_claims": [],
                "next_strategy": "search",
                "rewritten_query": "admissions refined",
                "stop_reason": None,
            },
            {
                "is_sufficient": True,
                "is_grounded": True,
                "missing_information": [],
                "unsupported_claims": [],
                "next_strategy": None,
                "rewritten_query": None,
                "stop_reason": "answer_sufficient",
            },
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)

    message = agent.run(
        messages=[MessageRecord(role="user", content="How do ESILV admissions work?")],
        user_text="How do ESILV admissions work?",
        initial_action="search",
        initial_query="admissions initial",
        feature_settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True, super_agent_enabled=True),
    )

    assert search_agent.queries == [("admissions initial", 5), ("admissions refined", 5)]
    assert retrieval_agent.queries == []
    assert message.content == "search answer for Admissions | ESILV"
    assert message.super_agent_stop_reason == "answer_sufficient"
    assert len(message.super_agent_trace) == 2


def test_super_agent_runs_hybrid_when_both_tools_are_enabled():
    user_text = "According to the document, what are the ESILV admissions details?"
    retrieval_agent = FakeRetrievalAgent(
        results_by_query={
            user_text: [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=2,
                    snippet="Weak document hit",
                    score=0.4,
                    cosine_score=0.35,
                    lexical_overlap=0.1,
                )
            ]
        },
        weak_by_query={user_text: True},
    )
    search_agent = FakeSearchAgent(
        {
            user_text: [
                SearchHit(
                    url="https://www.esilv.fr/admissions/",
                    title="Admissions | ESILV",
                    snippet="Admissions steps",
                    score=8.2,
                    lexical_overlap=0.5,
                    expanded_overlap=0.5,
                    fetched_at="2026-03-18T12:00:00+00:00",
                )
            ]
        }
    )
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": True,
                "is_grounded": True,
                "missing_information": [],
                "unsupported_claims": [],
                "next_strategy": None,
                "rewritten_query": None,
                "stop_reason": "answer_sufficient",
            },
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)

    message = agent.run(
        messages=[MessageRecord(role="user", content=user_text)],
        user_text=user_text,
        initial_action="retrieve",
        initial_query="mixed query",
        feature_settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=True),
    )

    assert retrieval_agent.queries == [(user_text, 5)]
    assert search_agent.queries == [(user_text, 5)]
    assert [item.selected_strategy for item in message.super_agent_trace] == ["hybrid"]
    assert message.content == "hybrid answer using PLM.pdf and Admissions | ESILV"


def test_super_agent_starts_with_hybrid_for_mixed_question():
    retrieval_agent = FakeRetrievalAgent(
        results_by_query={
            "According to the uploaded document and how do ESILV admissions work?": [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=17,
                    snippet="PLM functionality",
                    score=0.82,
                    cosine_score=0.8,
                    lexical_overlap=0.5,
                )
            ]
        },
        weak_by_query={"According to the uploaded document and how do ESILV admissions work?": False},
    )
    search_agent = FakeSearchAgent(
        {
            "According to the uploaded document and how do ESILV admissions work?": [
                SearchHit(
                    url="https://www.esilv.fr/admissions/",
                    title="Admissions | ESILV",
                    snippet="Admissions details",
                    score=8.5,
                    lexical_overlap=0.5,
                    expanded_overlap=0.6,
                    fetched_at="2026-03-18T12:00:00+00:00",
                )
            ]
        }
    )
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": True,
                "is_grounded": True,
                "missing_information": [],
                "unsupported_claims": [],
                "next_strategy": None,
                "rewritten_query": None,
                "stop_reason": "answer_sufficient",
            }
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)
    query = "According to the uploaded document and how do ESILV admissions work?"

    message = agent.run(
        messages=[MessageRecord(role="user", content=query)],
        user_text=query,
        initial_action="retrieve",
        initial_query=query,
        feature_settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=True),
    )

    assert [item.selected_strategy for item in message.super_agent_trace] == ["hybrid"]
    assert "hybrid answer using PLM.pdf and Admissions | ESILV" == message.content
    assert {citation.kind for citation in message.citations} == {"document", "web"}


def test_super_agent_uses_original_user_query_for_hybrid_when_router_started_with_search():
    user_text = "How do ESILV admissions work?"
    retrieval_agent = FakeRetrievalAgent(
        results_by_query={
            user_text: [
                RetrievalHit(
                    document_id="doc-1",
                    filename="admissions.pdf",
                    page_number=2,
                    snippet="Weak document hit",
                    score=0.35,
                    cosine_score=0.3,
                    lexical_overlap=0.1,
                )
            ]
        },
        weak_by_query={user_text: True},
    )
    search_agent = FakeSearchAgent(
        {
            user_text: [
                SearchHit(
                    url="https://www.esilv.fr/admissions/",
                    title="Admissions | ESILV",
                    snippet="Admissions details",
                    score=9.0,
                    lexical_overlap=0.5,
                    expanded_overlap=0.6,
                    fetched_at="2026-03-18T12:00:00+00:00",
                )
            ]
        }
    )
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": True,
                "is_grounded": True,
                "missing_information": [],
                "unsupported_claims": [],
                "next_strategy": None,
                "rewritten_query": None,
                "stop_reason": "answer_sufficient",
            },
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)

    message = agent.run(
        messages=[MessageRecord(role="user", content=user_text)],
        user_text=user_text,
        initial_action="search",
        initial_query="ESILV admissions",
        feature_settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=True),
    )

    assert retrieval_agent.queries == [(user_text, 5)]
    assert search_agent.queries == [(user_text, 5)]
    assert [item.selected_strategy for item in message.super_agent_trace] == ["hybrid"]


def test_super_agent_retries_hybrid_with_rewritten_query_when_both_tools_are_enabled():
    query = "initial hybrid query"
    refined_query = "refined hybrid query"
    retrieval_agent = FakeRetrievalAgent(
        results_by_query={
            query: [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=17,
                    snippet="PLM functionality",
                    score=0.82,
                    cosine_score=0.8,
                    lexical_overlap=0.5,
                )
            ],
            refined_query: [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=18,
                    snippet="Refined PLM functionality",
                    score=0.84,
                    cosine_score=0.82,
                    lexical_overlap=0.52,
                )
            ],
        },
        weak_by_query={query: False, refined_query: False},
    )
    search_agent = FakeSearchAgent(
        {
            query: [
                SearchHit(
                    url="https://www.esilv.fr/admissions/",
                    title="Admissions | ESILV",
                    snippet="Admissions details",
                    score=8.5,
                    lexical_overlap=0.5,
                    expanded_overlap=0.6,
                    fetched_at="2026-03-18T12:00:00+00:00",
                )
            ],
            refined_query: [
                SearchHit(
                    url="https://www.esilv.fr/admissions/",
                    title="Admissions | ESILV",
                    snippet="Refined admissions details",
                    score=8.8,
                    lexical_overlap=0.55,
                    expanded_overlap=0.65,
                    fetched_at="2026-03-18T12:05:00+00:00",
                )
            ],
        }
    )
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": False,
                "is_grounded": True,
                "missing_information": ["more precise details"],
                "unsupported_claims": [],
                "next_strategy": "hybrid",
                "rewritten_query": refined_query,
                "stop_reason": None,
            },
            {
                "is_sufficient": True,
                "is_grounded": True,
                "missing_information": [],
                "unsupported_claims": [],
                "next_strategy": None,
                "rewritten_query": None,
                "stop_reason": "answer_sufficient",
            },
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)

    message = agent.run(
        messages=[MessageRecord(role="user", content=query)],
        user_text=query,
        initial_action="retrieve",
        initial_query=query,
        feature_settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=True),
    )

    assert retrieval_agent.queries == [(query, 5), (refined_query, 5)]
    assert search_agent.queries == [(query, 5), (refined_query, 5)]
    assert [item.selected_strategy for item in message.super_agent_trace] == ["hybrid", "hybrid"]
    assert message.content == "hybrid answer using PLM.pdf and Admissions | ESILV"
    assert {citation.kind for citation in message.citations} == {"document", "web"}


def test_super_agent_accumulates_previous_iteration_results_in_later_generation():
    initial_query = "initial hybrid query"
    refined_query = "refined hybrid query"
    retrieval_agent = FakeRetrievalAgent(
        results_by_query={
            initial_query: [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=17,
                    snippet="Doc A",
                    score=0.82,
                    cosine_score=0.8,
                    lexical_overlap=0.5,
                )
            ],
            refined_query: [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=18,
                    snippet="Doc B",
                    score=0.84,
                    cosine_score=0.82,
                    lexical_overlap=0.52,
                )
            ],
        },
        weak_by_query={initial_query: False, refined_query: False},
    )
    search_agent = FakeSearchAgent(
        {
            initial_query: [
                SearchHit(
                    url="https://www.esilv.fr/a",
                    title="Web A",
                    snippet="Admissions A",
                    score=8.5,
                    lexical_overlap=0.5,
                    expanded_overlap=0.6,
                    fetched_at="2026-03-18T12:00:00+00:00",
                )
            ],
            refined_query: [
                SearchHit(
                    url="https://www.esilv.fr/b",
                    title="Web B",
                    snippet="Admissions B",
                    score=8.8,
                    lexical_overlap=0.55,
                    expanded_overlap=0.65,
                    fetched_at="2026-03-18T12:05:00+00:00",
                )
            ],
        }
    )
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": False,
                "is_grounded": True,
                "missing_information": ["more evidence"],
                "unsupported_claims": [],
                "next_strategy": "hybrid",
                "rewritten_query": refined_query,
                "stop_reason": None,
            },
            {
                "is_sufficient": True,
                "is_grounded": True,
                "missing_information": [],
                "unsupported_claims": [],
                "next_strategy": None,
                "rewritten_query": None,
                "stop_reason": "answer_sufficient",
            },
        ]
    )
    agent = SuperAgent(
        search_agent=search_agent,
        retrieval_agent=retrieval_agent,
        llm_client=llm,
        max_search_hits=5,
        format_messages=lambda messages: "\n".join(f"{message.role}: {message.content}" for message in messages),
        search_is_weak=lambda query, hits: not hits or hits[0].score < 5,
        citations_from_hits=lambda hits: [
            CitationRecord(kind="web", title=hit.title, url=hit.url) for hit in hits[:3]
        ],
        citations_from_retrieval_hits=lambda hits: [
            CitationRecord(kind="document", title=hit.filename, page_number=hit.page_number)
            for hit in hits[:3]
        ],
        retrieval_answer_builder=lambda messages, user_text, hits: "retrieval-only",
        search_answer_builder=lambda messages, user_text, hits: "search-only",
        hybrid_answer_builder=lambda messages, user_text, retrieval_hits, search_hits: (
            "docs="
            + ",".join(hit.snippet for hit in retrieval_hits)
            + "|web="
            + ",".join(hit.title for hit in search_hits)
        ),
        looks_french=lambda text: False,
        is_retrieval_intent=lambda lowered: "query" in lowered,
        is_search_intent=lambda lowered: "query" in lowered,
        has_independent_search_intent=lambda lowered: True,
    )

    message = agent.run(
        messages=[MessageRecord(role="user", content=initial_query)],
        user_text=initial_query,
        initial_action="retrieve",
        initial_query=initial_query,
        feature_settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=True, super_agent_enabled=True),
    )

    assert all(marker in message.super_agent_trace[1].draft_answer for marker in ("Doc A", "Doc B", "Web A", "Web B"))
    assert all(marker in message.content for marker in ("Doc A", "Doc B", "Web A", "Web B"))


def test_super_agent_stops_when_next_step_is_redundant():
    retrieval_agent = FakeRetrievalAgent({}, {})
    search_agent = FakeSearchAgent(
        {
            "same query": [
                SearchHit(
                    url="https://www.esilv.fr/admissions/",
                    title="Admissions | ESILV",
                    snippet="Admissions details",
                    score=8.0,
                    lexical_overlap=0.5,
                    expanded_overlap=0.6,
                    fetched_at="2026-03-18T12:00:00+00:00",
                )
            ]
        }
    )
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": False,
                "is_grounded": True,
                "missing_information": ["specific deadlines"],
                "unsupported_claims": [],
                "next_strategy": "search",
                "rewritten_query": "same query",
                "stop_reason": None,
            }
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)

    message = agent.run(
        messages=[MessageRecord(role="user", content="How do ESILV admissions work?")],
        user_text="How do ESILV admissions work?",
        initial_action="search",
        initial_query="same query",
        feature_settings=AgentFeatureSettings(rag_enabled=False, web_search_enabled=True, super_agent_enabled=True),
    )

    assert search_agent.queries == [("same query", 5)]
    assert message.super_agent_stop_reason == "no_novel_next_step"
    assert len(message.super_agent_trace) == 1


def test_super_agent_returns_best_candidate_with_limitation_when_budget_exhausts():
    retrieval_agent = FakeRetrievalAgent(
        results_by_query={
            "query one": [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=5,
                    snippet="Basic context",
                    score=0.6,
                    cosine_score=0.58,
                    lexical_overlap=0.2,
                )
            ],
            "query two": [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=17,
                    snippet="Main functionality",
                    score=0.82,
                    cosine_score=0.8,
                    lexical_overlap=0.5,
                )
            ],
            "query three": [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=18,
                    snippet="Another partial hit",
                    score=0.8,
                    cosine_score=0.78,
                    lexical_overlap=0.45,
                )
            ],
        },
        weak_by_query={"query one": True, "query two": False, "query three": False},
    )
    search_agent = FakeSearchAgent({})
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": False,
                "is_grounded": False,
                "missing_information": ["functional modules"],
                "unsupported_claims": [],
                "next_strategy": "retrieve",
                "rewritten_query": "query two",
                "stop_reason": None,
            },
            {
                "is_sufficient": False,
                "is_grounded": True,
                "missing_information": ["version management"],
                "unsupported_claims": [],
                "next_strategy": "retrieve",
                "rewritten_query": "query three",
                "stop_reason": None,
            },
            {
                "is_sufficient": False,
                "is_grounded": True,
                "missing_information": ["workflow details", "user management"],
                "unsupported_claims": [],
                "next_strategy": "retrieve",
                "rewritten_query": "query three",
                "stop_reason": None,
            },
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)

    message = agent.run(
        messages=[MessageRecord(role="user", content="What does the PLM document say?")],
        user_text="What does the PLM document say?",
        initial_action="retrieve",
        initial_query="query one",
        feature_settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=False, super_agent_enabled=True),
    )

    assert retrieval_agent.queries == [("query one", 5), ("query two", 5), ("query three", 5)]
    assert message.super_agent_stop_reason == "max_iterations_reached"
    assert "Partial grounded answer only" in message.content
    assert "retrieval answer for PLM.pdf" in message.content
    assert len(message.super_agent_trace) == 3


def test_super_agent_persists_trace_details_on_message():
    retrieval_agent = FakeRetrievalAgent(
        results_by_query={
            "plm": [
                RetrievalHit(
                    document_id="doc-1",
                    filename="PLM.pdf",
                    page_number=17,
                    snippet="Main functionality",
                    score=0.82,
                    cosine_score=0.8,
                    lexical_overlap=0.5,
                )
            ]
        },
        weak_by_query={"plm": False},
    )
    search_agent = FakeSearchAgent({})
    llm = FakeEvaluationLlm(
        [
            {
                "is_sufficient": True,
                "is_grounded": True,
                "missing_information": [],
                "unsupported_claims": [],
                "next_strategy": None,
                "rewritten_query": None,
                "stop_reason": "answer_sufficient",
            }
        ]
    )
    agent = build_super_agent(llm, search_agent, retrieval_agent)

    message = agent.run(
        messages=[MessageRecord(role="user", content="According to the PLM document, what are the main functionalities?")],
        user_text="According to the PLM document, what are the main functionalities?",
        initial_action="retrieve",
        initial_query="plm",
        feature_settings=AgentFeatureSettings(rag_enabled=True, web_search_enabled=False, super_agent_enabled=True),
    )

    assert message.orchestration_mode == "super"
    assert message.super_agent_trace[0].retrieval_outcome is not None
    assert message.super_agent_trace[0].retrieval_outcome.hit_count == 1
    assert message.super_agent_trace[0].evaluator_result.is_sufficient is True
