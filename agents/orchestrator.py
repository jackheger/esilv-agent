from __future__ import annotations

from typing import Literal, Protocol

from google import genai
from google.genai import types
from pydantic import BaseModel

from agents.registration import RegistrationAgent
from agents.super_agent import SuperAgent
from app.agent_settings import AgentFeatureSettingsStore
from app.conversation_store import ConversationStore
from app.models import AgentFeatureSettings, CitationRecord, MessageRecord, RetrievalHit, SearchHit
from app.registration_store import RegistrationStore

SYSTEM_PROMPT = """You are the ESILV Smart Assistant for a local demo.
Reply in the same language as the user's latest message.
Be concise, helpful, and explicit when information comes from ESILV sources.
Do not invent ESILV facts. If the context is weak or missing, say so clearly.
You must answer in the same language as the user's last question.
"""
FRENCH_LANGUAGE_MARKERS = {
    "bonjour",
    "merci",
    "je",
    "j",
    "tu",
    "vous",
    "pouvez",
    "pouvez-vous",
    "votre",
    "vos",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "comment",
    "pourquoi",
    "expliquer",
    "expliquez",
    "est",
    "suis",
    "avec",
    "dans",
    "pour",
    "les",
    "des",
    "une",
    "un",
    "inscription",
    "candidature",
    "rejoindre",
    "etudier",
    "ecole",
}
ENGLISH_LANGUAGE_MARKERS = {
    "hello",
    "hi",
    "thanks",
    "thank",
    "please",
    "what",
    "which",
    "when",
    "where",
    "how",
    "why",
    "can",
    "could",
    "would",
    "should",
    "join",
    "study",
    "apply",
    "school",
}
DIRECT_STARTERS = ("hello", "hi", "hey", "thanks", "thank you", "bonjour", "salut", "merci")
RETRIEVE_KEYWORDS = (
    "pdf",
    "document",
    "documents",
    "report",
    "uploaded",
    "internal",
    "according to",
    "in the document",
    "dans le document",
    "dans le pdf",
    "rapport",
    "fichier",
)
DOCUMENT_ANCHORED_PHRASES = (
    "according to the uploaded",
    "according to the pdf",
    "according to the document",
    "in the uploaded pdf",
    "in the uploaded document",
    "in the document",
    "in the pdf",
    "uploaded pdf",
    "uploaded document",
    "uploaded report",
    "dans le document",
    "dans le pdf",
)
SEARCH_KEYWORDS = (
    "esilv",
    "admission",
    "admissions",
    "programme",
    "program",
    "formations",
    "course",
    "courses",
    "campus",
    "tuition",
    "deadline",
    "application",
    "inscription",
    "master",
    "bachelor",
    "engineer",
    "cours",
    "website",
    "site",
)
AFFIRMATIVE_REPLIES = (
    "yes",
    "yes.",
    "yes!",
    "yes please",
    "yeah",
    "yep",
    "sure",
    "exactly",
    "oui",
    "oui.",
    "oui!",
    "oui exact",
    "oui exactement",
    "exactement",
)
CLARIFICATION_SUGGESTIONS = {
    "sql": {
        "en": "database-related courses",
        "fr": "des cours lies aux bases de donnees",
        "query": "database courses sql relational database bases de donnees data management",
        "label": "SQL",
    },
    "database": {
        "en": "database-related courses",
        "fr": "des cours lies aux bases de donnees",
        "query": "database courses sql relational database bases de donnees data management",
        "label": "database",
    },
    "databases": {
        "en": "database-related courses",
        "fr": "des cours lies aux bases de donnees",
        "query": "database courses sql relational database bases de donnees data management",
        "label": "databases",
    },
    "cloud": {
        "en": "cloud or infrastructure courses",
        "fr": "des cours lies au cloud ou a l'infrastructure",
        "query": "cloud courses infrastructure cloud computing platform engineering",
        "label": "cloud",
    },
    "cybersecurity": {
        "en": "cybersecurity-related courses",
        "fr": "des cours lies a la cybersecurite",
        "query": "cybersecurity courses security cybersecurite securite informatique",
        "label": "cybersecurity",
    },
    "cybersecurite": {
        "en": "cybersecurity-related courses",
        "fr": "des cours lies a la cybersecurite",
        "query": "cybersecurity courses security cybersecurite securite informatique",
        "label": "cybersecurite",
    },
    "ai": {
        "en": "AI or machine-learning courses",
        "fr": "des cours lies a l'IA ou au machine learning",
        "query": "AI courses artificial intelligence machine learning intelligence artificielle",
        "label": "AI",
    },
    "ia": {
        "en": "AI or machine-learning courses",
        "fr": "des cours lies a l'IA ou au machine learning",
        "query": "AI courses artificial intelligence machine learning intelligence artificielle",
        "label": "IA",
    },
}


class RoutingDecision(BaseModel):
    action: Literal["direct", "retrieve", "search"]
    tool_query: str | None = None
    search_query: str | None = None
    rationale: str | None = None


class SearchAgentProtocol(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]: ...


class RetrievalAgentProtocol(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]: ...

    def is_weak(self, hits: list[RetrievalHit]) -> bool: ...


class LlmClientProtocol(Protocol):
    @property
    def configured(self) -> bool: ...

    def generate_text(self, prompt: str, system_instruction: str, temperature: float = 0.2) -> str: ...

    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel: ...


class GoogleGeminiClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = genai.Client(api_key=api_key) if api_key else None

    @property
    def configured(self) -> bool:
        return self._client is not None

    def generate_text(self, prompt: str, system_instruction: str, temperature: float = 0.2) -> str:
        if self._client is None:
            raise RuntimeError("Gemini API key is not configured")

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                systemInstruction=system_instruction,
                temperature=temperature,
            ),
        )
        return (response.text or "").strip()

    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        if self._client is None:
            raise RuntimeError("Gemini API key is not configured")

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                systemInstruction=system_instruction,
                responseMimeType="application/json",
                responseSchema=schema,
                temperature=temperature,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return schema.model_validate(parsed)
        return schema.model_validate_json(response.text or "{}")


class OrchestratorAgent:
    def __init__(
        self,
        conversation_store: ConversationStore,
        registration_store: RegistrationStore,
        search_agent: SearchAgentProtocol,
        retrieval_agent: RetrievalAgentProtocol,
        llm_client: LlmClientProtocol,
        feature_settings_store: AgentFeatureSettingsStore,
        max_search_hits: int = 5,
    ) -> None:
        self.conversation_store = conversation_store
        self.search_agent = search_agent
        self.retrieval_agent = retrieval_agent
        self.llm_client = llm_client
        self.feature_settings_store = feature_settings_store
        self.max_search_hits = max_search_hits
        self.super_agent = SuperAgent(
            search_agent=search_agent,
            retrieval_agent=retrieval_agent,
            llm_client=llm_client,
            max_search_hits=max_search_hits,
            format_messages=self._format_messages,
            search_is_weak=self._search_is_weak,
            citations_from_hits=self._citations_from_hits,
            citations_from_retrieval_hits=self._citations_from_retrieval_hits,
            retrieval_answer_builder=self._retrieval_answer,
            search_answer_builder=self._grounded_answer,
            hybrid_answer_builder=self._hybrid_answer,
            looks_french=self._looks_french,
            is_retrieval_intent=self._is_retrieval_intent,
            is_search_intent=self._is_search_intent,
            has_independent_search_intent=self._has_independent_search_intent,
        )
        self.registration_agent = RegistrationAgent(
            registration_store=registration_store,
            search_agent=search_agent,
            retrieval_agent=retrieval_agent,
            llm_client=llm_client,
            super_agent=self.super_agent,
            max_search_hits=max_search_hits,
            looks_french=self._looks_french,
            search_is_weak=self._search_is_weak,
            citations_from_hits=self._citations_from_hits,
            citations_from_retrieval_hits=self._citations_from_retrieval_hits,
            retrieval_answer_builder=self._retrieval_answer,
            search_answer_builder=self._grounded_answer,
        )

    def handle_turn(self, conversation_id: str, user_text: str) -> MessageRecord:
        user_message = MessageRecord(role="user", content=user_text)
        self.conversation_store.append_message(conversation_id, user_message)
        conversation = self.conversation_store.load(conversation_id)

        feature_settings = self.feature_settings_store.load()
        registration_message = self.registration_agent.continue_session(
            conversation_id=conversation_id,
            messages=conversation.messages,
            user_text=user_text,
            feature_settings=feature_settings,
        )
        if registration_message is not None:
            self.conversation_store.append_message(conversation_id, registration_message)
            return registration_message

        if self.registration_agent.should_start_from_follow_up(conversation.messages, user_text):
            assistant_message = self.registration_agent.start_session(conversation_id, user_text)
            self.conversation_store.append_message(conversation_id, assistant_message)
            return assistant_message

        if self.registration_agent.should_start_immediately(user_text):
            assistant_message = self.registration_agent.start_session(conversation_id, user_text)
            self.conversation_store.append_message(conversation_id, assistant_message)
            return assistant_message

        if not self.llm_client.configured:
            assistant_message = MessageRecord(
                role="assistant",
                content=self._config_missing_message(user_text),
            )
            if self.registration_agent.should_offer_registration(user_text, assistant_message):
                assistant_message = self.registration_agent.append_registration_cta(assistant_message, user_text)
            self.conversation_store.append_message(conversation_id, assistant_message)
            return assistant_message

        decision = self._follow_up_decision(conversation.messages, user_text, feature_settings) or self._route(
            conversation.messages,
            user_text,
            feature_settings,
        )
        try:
            if (
                feature_settings.super_agent_enabled
                and decision.action in {"retrieve", "search"}
                and (feature_settings.rag_enabled or feature_settings.web_search_enabled)
            ):
                query = decision.tool_query or decision.search_query or user_text
                assistant_message = self.super_agent.run(
                    messages=conversation.messages,
                    user_text=user_text,
                    initial_action=decision.action,
                    initial_query=query,
                    feature_settings=feature_settings,
                )
            else:
                assistant_message = self._execute_single_pass(
                    decision=decision,
                    messages=conversation.messages,
                    user_text=user_text,
                    feature_settings=feature_settings,
                )
        except Exception:
            assistant_message = MessageRecord(
                role="assistant",
                content=self._temporary_failure_message(user_text),
            )

        if self.registration_agent.should_offer_registration(user_text, assistant_message):
            assistant_message = self.registration_agent.append_registration_cta(assistant_message, user_text)

        self.conversation_store.append_message(conversation_id, assistant_message)
        return assistant_message

    def _route(
        self,
        messages: list[MessageRecord],
        user_text: str,
        feature_settings: AgentFeatureSettings,
    ) -> RoutingDecision:
        if not feature_settings.rag_enabled and not feature_settings.web_search_enabled:
            return RoutingDecision(action="direct")

        allowed_actions = self._allowed_actions(feature_settings)
        route_guidance = [
            "Use 'direct' for greetings, thanks, meta questions, or conversation that can be handled without external verification.",
        ]
        if feature_settings.rag_enabled:
            route_guidance.append(
                "Use 'retrieve' for uploaded/internal documents, PDF reports, or questions explicitly grounded in a document or report."
            )
        else:
            route_guidance.append(
                "Uploaded-document retrieval is unavailable for this turn. For document-grounded requests, prefer 'direct' unless the user also clearly asks for ESILV website facts."
            )
        if feature_settings.web_search_enabled:
            route_guidance.append(
                "Use 'search' for factual questions about ESILV programs, admissions, courses, campus information, tuition, deadlines, registration, or school services."
            )
        else:
            route_guidance.append("ESILV website search is unavailable for this turn. Do not return 'search'.")

        try:
            prompt = (
                "Decide how the assistant should respond.\n"
                f"Allowed actions for this turn: {', '.join(repr(action) for action in allowed_actions)}.\n"
                "Choose only one of the allowed actions.\n"
                f"{'\n'.join(route_guidance)}\n\n"
                f"Conversation:\n{self._format_messages(messages[-8:])}\n\n"
                f"Latest user message:\n{user_text}\n"
            )
            decision = self.llm_client.generate_structured(
                prompt=prompt,
                system_instruction="Return only a routing decision.",
                schema=RoutingDecision,
                temperature=0.0,
            )
            return self._normalize_decision(
                RoutingDecision.model_validate(decision),
                user_text=user_text,
                feature_settings=feature_settings,
            )
        except Exception:
            return self._heuristic_route(messages, user_text, feature_settings)

    def _direct_answer(self, messages: list[MessageRecord], user_text: str) -> str:
        required_language = self._response_language(user_text)
        prompt = (
            f"Conversation:\n{self._format_messages(messages[-8:])}\n\n"
            f"Latest user message:\n{user_text}\n\n"
            f"Required answer language: {required_language}.\n"
            "Answer directly from the conversation and general knowledge only. Do not claim that "
            "you checked the ESILV website or uploaded documents unless source snippets were "
            "provided. If you are unsure, say so clearly and ask for clarification if needed."
        )
        return self.llm_client.generate_text(
            prompt=prompt,
            system_instruction=self._answer_system_prompt(user_text),
        )

    def _grounded_answer(self, messages: list[MessageRecord], user_text: str, hits: list[SearchHit]) -> str:
        required_language = self._response_language(user_text)
        sources = "\n\n".join(
            f"Source {index + 1}\nTitle: {hit.title}\nURL: {hit.url}\nSnippet: {hit.snippet}"
            for index, hit in enumerate(hits[: self.max_search_hits])
        )
        prompt = (
            f"Conversation:\n{self._format_messages(messages[-8:])}\n\n"
            f"Latest user message:\n{user_text}\n\n"
            f"Required answer language: {required_language}.\n"
            f"If the source snippets are in another language, translate or summarize them into {required_language}.\n"
            "Use only the source snippets below to answer. If they do not fully answer the "
            "question, state the limitation and do not invent details.\n\n"
            f"{sources}"
        )
        return self.llm_client.generate_text(
            prompt=prompt,
            system_instruction=self._answer_system_prompt(user_text),
        )

    def _hybrid_answer(
        self,
        messages: list[MessageRecord],
        user_text: str,
        retrieval_hits: list[RetrievalHit],
        search_hits: list[SearchHit],
    ) -> str:
        required_language = self._response_language(user_text)
        retrieval_sources = "\n\n".join(
            (
                f"Document source {index + 1}\n"
                f"Document: {hit.filename}\n"
                f"Page: {hit.page_number}\n"
                f"Snippet: {hit.snippet}"
            )
            for index, hit in enumerate(retrieval_hits[: self.max_search_hits])
        )
        search_sources = "\n\n".join(
            f"Web source {index + 1}\nTitle: {hit.title}\nURL: {hit.url}\nSnippet: {hit.snippet}"
            for index, hit in enumerate(search_hits[: self.max_search_hits])
        )
        prompt = (
            f"Conversation:\n{self._format_messages(messages[-8:])}\n\n"
            f"Latest user message:\n{user_text}\n\n"
            f"Required answer language: {required_language}.\n"
            f"If the source snippets are in another language, translate or summarize them into {required_language}.\n"
            "Use only the document and web source snippets below to answer. Compare the current draft "
            "against the entire user question. If any part remains unsupported or missing, say so clearly "
            "and do not invent details.\n\n"
            f"Uploaded-document snippets:\n{retrieval_sources or '(none)'}\n\n"
            f"ESILV website snippets:\n{search_sources or '(none)'}"
        )
        return self.llm_client.generate_text(
            prompt=prompt,
            system_instruction=self._answer_system_prompt(user_text),
        )

    def _retrieval_answer(
        self,
        messages: list[MessageRecord],
        user_text: str,
        hits: list[RetrievalHit],
    ) -> str:
        required_language = self._response_language(user_text)
        sources = "\n\n".join(
            (
                f"Source {index + 1}\n"
                f"Document: {hit.filename}\n"
                f"Page: {hit.page_number}\n"
                f"Snippet: {hit.snippet}"
            )
            for index, hit in enumerate(hits[: self.max_search_hits])
        )
        prompt = (
            f"Conversation:\n{self._format_messages(messages[-8:])}\n\n"
            f"Latest user message:\n{user_text}\n\n"
            f"Required answer language: {required_language}.\n"
            f"If the source snippets are in another language, translate or summarize them into {required_language}.\n"
            "Use only the uploaded-document snippets below to answer. If they do not fully answer "
            "the question, state the limitation and do not invent details.\n\n"
            f"{sources}"
        )
        return self.llm_client.generate_text(
            prompt=prompt,
            system_instruction=self._answer_system_prompt(user_text),
        )

    def _execute_single_pass(
        self,
        decision: RoutingDecision,
        messages: list[MessageRecord],
        user_text: str,
        feature_settings: AgentFeatureSettings,
    ) -> MessageRecord:
        if (
            not feature_settings.super_agent_enabled
            and feature_settings.rag_enabled
            and feature_settings.web_search_enabled
            and decision.action in {"retrieve", "search"}
        ):
            return self._execute_hybrid_single_pass(
                messages=messages,
                user_text=user_text,
            )

        if decision.action == "retrieve" and feature_settings.rag_enabled:
            query = decision.tool_query or decision.search_query or user_text
            hits = self.retrieval_agent.search(query, top_k=self.max_search_hits)
            if self.retrieval_agent.is_weak(hits):
                return self._clarification_message(
                    action="retrieve",
                    user_text=user_text,
                    query=query,
                )
            return MessageRecord(
                role="assistant",
                content=self._retrieval_answer(messages, user_text, hits),
                citations=self._citations_from_retrieval_hits(hits),
            )

        if decision.action == "search" and feature_settings.web_search_enabled:
            query = decision.tool_query or decision.search_query or user_text
            hits = self.search_agent.search(query, top_k=self.max_search_hits)
            if self._search_is_weak(query, hits):
                return self._clarification_message(
                    action="search",
                    user_text=user_text,
                    query=query,
                )
            return MessageRecord(
                role="assistant",
                content=self._grounded_answer(messages, user_text, hits),
                citations=self._citations_from_hits(hits),
            )

        return MessageRecord(
            role="assistant",
            content=self._direct_answer(messages, user_text),
        )

    def _execute_hybrid_single_pass(
        self,
        messages: list[MessageRecord],
        user_text: str,
    ) -> MessageRecord:
        retrieval_hits = self.retrieval_agent.search(user_text, top_k=self.max_search_hits)
        search_hits = self.search_agent.search(user_text, top_k=self.max_search_hits)

        retrieval_weak = self.retrieval_agent.is_weak(retrieval_hits)
        search_weak = self._search_is_weak(user_text, search_hits)

        strong_retrieval_hits = [] if retrieval_weak else retrieval_hits
        strong_search_hits = [] if search_weak else search_hits

        if strong_retrieval_hits and strong_search_hits:
            return MessageRecord(
                role="assistant",
                content=self._hybrid_answer(messages, user_text, strong_retrieval_hits, strong_search_hits),
                citations=self._merge_citations(
                    self._citations_from_retrieval_hits(strong_retrieval_hits),
                    self._citations_from_hits(strong_search_hits),
                ),
            )

        if strong_retrieval_hits:
            return MessageRecord(
                role="assistant",
                content=self._retrieval_answer(messages, user_text, strong_retrieval_hits),
                citations=self._citations_from_retrieval_hits(strong_retrieval_hits),
            )

        if strong_search_hits:
            return MessageRecord(
                role="assistant",
                content=self._grounded_answer(messages, user_text, strong_search_hits),
                citations=self._citations_from_hits(strong_search_hits),
            )

        return self._clarification_message(
            action="search",
            user_text=user_text,
            query=user_text,
        )

    @staticmethod
    def _format_messages(messages: list[MessageRecord]) -> str:
        if not messages:
            return "(empty)"
        return "\n".join(f"{message.role.upper()}: {message.content}" for message in messages)

    def _search_is_weak(self, query: str, hits: list[SearchHit]) -> bool:
        if not hits:
            return True
        top_hit = hits[0]
        if top_hit.score < 4 or top_hit.expanded_overlap < 0.18:
            return True
        return self._suggest_refinement(query) is not None and top_hit.lexical_overlap < 0.2

    @staticmethod
    def _citations_from_hits(hits: list[SearchHit]) -> list[CitationRecord]:
        seen: set[str] = set()
        citations: list[CitationRecord] = []
        for hit in hits:
            if hit.url in seen:
                continue
            seen.add(hit.url)
            citations.append(CitationRecord(kind="web", title=hit.title, url=hit.url))
        return citations[:3]

    @staticmethod
    def _citations_from_retrieval_hits(hits: list[RetrievalHit]) -> list[CitationRecord]:
        seen: set[tuple[str, int]] = set()
        citations: list[CitationRecord] = []
        for hit in hits:
            key = (hit.filename, hit.page_number)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                CitationRecord(
                    kind="document",
                    title=hit.filename,
                    page_number=hit.page_number,
                )
            )
        return citations[:3]

    @staticmethod
    def _merge_citations(
        retrieval_citations: list[CitationRecord],
        search_citations: list[CitationRecord],
    ) -> list[CitationRecord]:
        merged: list[CitationRecord] = []
        seen_docs: set[tuple[str, int | None]] = set()
        seen_urls: set[str] = set()
        for citation in retrieval_citations + search_citations:
            if citation.kind == "document":
                key = (citation.title, citation.page_number)
                if key in seen_docs:
                    continue
                seen_docs.add(key)
            elif citation.url:
                if citation.url in seen_urls:
                    continue
                seen_urls.add(citation.url)
            merged.append(citation)
        return merged[:6]

    @staticmethod
    def _allowed_actions(feature_settings: AgentFeatureSettings) -> tuple[str, ...]:
        actions = ["direct"]
        if feature_settings.rag_enabled:
            actions.append("retrieve")
        if feature_settings.web_search_enabled:
            actions.append("search")
        return tuple(actions)

    def _follow_up_decision(
        self,
        messages: list[MessageRecord],
        user_text: str,
        feature_settings: AgentFeatureSettings,
    ) -> RoutingDecision | None:
        if not self._is_affirmative_reply(user_text):
            return None
        if len(messages) < 2:
            return None

        previous_message = messages[-2]
        if previous_message.role != "assistant":
            return None
        if not previous_message.pending_action or not previous_message.pending_query:
            return None
        if previous_message.pending_action not in self._allowed_actions(feature_settings):
            return None

        return RoutingDecision(
            action=previous_message.pending_action,
            tool_query=previous_message.pending_query,
        )

    def _normalize_decision(
        self,
        decision: RoutingDecision,
        user_text: str,
        feature_settings: AgentFeatureSettings,
    ) -> RoutingDecision:
        if decision.action == "direct":
            lowered = user_text.lower()
            if feature_settings.rag_enabled and self._is_document_anchored_request(lowered):
                return RoutingDecision(action="retrieve", tool_query=user_text)
            if feature_settings.web_search_enabled and self._is_search_intent(lowered):
                return RoutingDecision(action="search", tool_query=user_text)
        if decision.action in self._allowed_actions(feature_settings):
            return decision
        return self._fallback_for_disabled_action(decision.action, user_text, feature_settings)

    def _fallback_for_disabled_action(
        self,
        disabled_action: str,
        user_text: str,
        feature_settings: AgentFeatureSettings,
    ) -> RoutingDecision:
        lowered = user_text.lower()
        if disabled_action == "retrieve":
            if feature_settings.web_search_enabled and self._has_independent_search_intent(lowered):
                return RoutingDecision(action="search", tool_query=user_text)
            return RoutingDecision(action="direct")
        if disabled_action == "search":
            if feature_settings.rag_enabled and self._is_retrieval_intent(lowered):
                return RoutingDecision(action="retrieve", tool_query=user_text)
            return RoutingDecision(action="direct")
        return RoutingDecision(action="direct")

    def _heuristic_route(
        self,
        messages: list[MessageRecord],
        user_text: str,
        feature_settings: AgentFeatureSettings,
    ) -> RoutingDecision:
        lowered = user_text.lower()
        follow_up = self._follow_up_decision(messages, user_text, feature_settings)
        if follow_up is not None:
            return follow_up
        if lowered.strip().startswith(DIRECT_STARTERS) and "?" not in lowered:
            return RoutingDecision(action="direct")
        if feature_settings.rag_enabled and self._is_retrieval_intent(lowered):
            return RoutingDecision(action="retrieve", tool_query=user_text)
        if feature_settings.web_search_enabled and self._is_search_intent(lowered):
            return RoutingDecision(action="search", tool_query=user_text)
        if self._is_retrieval_intent(lowered):
            return self._fallback_for_disabled_action("retrieve", user_text, feature_settings)
        if self._is_search_intent(lowered):
            return self._fallback_for_disabled_action("search", user_text, feature_settings)
        return RoutingDecision(action="direct")

    @staticmethod
    def _is_retrieval_intent(lowered: str) -> bool:
        return any(keyword in lowered for keyword in RETRIEVE_KEYWORDS)

    @staticmethod
    def _is_search_intent(lowered: str) -> bool:
        return any(keyword in lowered for keyword in SEARCH_KEYWORDS)

    @staticmethod
    def _is_document_anchored_request(lowered: str) -> bool:
        return any(phrase in lowered for phrase in DOCUMENT_ANCHORED_PHRASES)

    def _has_independent_search_intent(self, lowered: str) -> bool:
        return self._is_search_intent(lowered) and not self._is_document_anchored_request(lowered)

    @staticmethod
    def _looks_french(text: str) -> bool:
        lowered = text.lower()
        if any(character in lowered for character in "àâçéèêëîïôùûüÿœæ"):
            return True
        tokens = {
            token.strip(".,!?;:()[]{}'\"")
            for token in lowered.split()
            if token.strip(".,!?;:()[]{}'\"")
        }
        french_hits = len(tokens & FRENCH_LANGUAGE_MARKERS)
        english_hits = len(tokens & ENGLISH_LANGUAGE_MARKERS)
        if french_hits == english_hits == 0:
            return False
        return french_hits > english_hits

    def _response_language(self, user_text: str) -> str:
        return "French" if self._looks_french(user_text) else "English"

    def _answer_system_prompt(self, user_text: str) -> str:
        return f"{SYSTEM_PROMPT}\nRequired response language: {self._response_language(user_text)}."

    @staticmethod
    def _is_affirmative_reply(text: str) -> bool:
        lowered = " ".join(text.lower().split())
        return any(lowered == marker or lowered.startswith(f"{marker} ") for marker in AFFIRMATIVE_REPLIES)

    def _clarification_message(self, action: Literal["search", "retrieve"], user_text: str, query: str) -> MessageRecord:
        suggestion = self._suggest_refinement(query)
        if suggestion is None:
            return MessageRecord(
                role="assistant",
                content=self._generic_clarification_message(action, user_text),
            )

        refined_query = f"{query} {suggestion['query']}".strip()
        if self._looks_french(user_text):
            if action == "search":
                content = (
                    f"Tel quel, je n'ai pas trouve de resultat ESILV assez proche pour {suggestion['label']}. "
                    f"Voulez-vous dire {suggestion['fr']} ?"
                )
            else:
                content = (
                    f"Tel quel, je n'ai pas trouve de passage assez proche dans les PDF pour {suggestion['label']}. "
                    f"Voulez-vous dire {suggestion['fr']} ?"
                )
        else:
            if action == "search":
                content = (
                    f"Directly like this, I did not find an ESILV result close enough for {suggestion['label']}. "
                    f"Do you mean {suggestion['en']}?"
                )
            else:
                content = (
                    f"Directly like this, I did not find a PDF passage close enough for {suggestion['label']}. "
                    f"Do you mean {suggestion['en']}?"
                )

        return MessageRecord(
            role="assistant",
            content=content,
            pending_action=action,
            pending_query=refined_query,
        )

    @staticmethod
    def _suggest_refinement(query: str) -> dict[str, str] | None:
        lowered = query.lower()
        for token, suggestion in CLARIFICATION_SUGGESTIONS.items():
            if token in lowered:
                return suggestion
        return None

    def _generic_clarification_message(self, action: Literal["search", "retrieve"], user_text: str) -> str:
        if self._looks_french(user_text):
            if action == "search":
                return (
                    "Je n'ai pas trouve de resultat ESILV assez proche pour repondre proprement. "
                    "Pouvez-vous preciser la matiere, le programme ou la majeure que vous visez ?"
                )
            return (
                "Je n'ai pas trouve de passage assez proche dans les PDF pour repondre proprement. "
                "Pouvez-vous preciser le document, la matiere ou le sujet que vous visez ?"
            )
        if action == "search":
            return (
                "I did not find an ESILV result close enough to answer confidently. "
                "Can you narrow it down to a subject area, program, or major?"
            )
        return (
            "I did not find a PDF passage close enough to answer confidently. "
            "Can you clarify the document, subject area, or topic you mean?"
        )

    def _uncertain_message(self, user_text: str) -> str:
        if self._looks_french(user_text):
            return (
                "Je n'ai pas pu verifier cette information sur le site ESILV pour le moment. "
                "Pouvez-vous preciser le programme, le campus ou l'annee visee ?"
            )
        return (
            "I could not verify that information on the ESILV website yet. "
            "Can you clarify the program, campus, or year you mean?"
        )

    def _retrieval_uncertain_message(self, user_text: str) -> str:
        if self._looks_french(user_text):
            return (
                "Les documents televerses ne permettent pas de verifier clairement cette information. "
                "Pouvez-vous preciser le document ou reformuler la question ?"
            )
        return (
            "The uploaded documents do not clearly verify that information yet. "
            "Can you specify the document or rephrase the question?"
        )

    def _config_missing_message(self, user_text: str) -> str:
        if self._looks_french(user_text):
            return (
                "La cle Gemini n'est pas configuree. Ajoutez GEMINI_API_KEY dans le fichier .env "
                "pour activer les reponses du chatbot."
            )
        return (
            "The Gemini API key is not configured. Add GEMINI_API_KEY to the .env file "
            "to enable chatbot responses."
        )

    def _temporary_failure_message(self, user_text: str) -> str:
        if self._looks_french(user_text):
            return (
                "Le service de generation est temporairement indisponible. "
                "Veuillez reessayer dans un instant."
            )
        return (
            "The generation service is temporarily unavailable. "
            "Please try again in a moment."
        )
