from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol
from uuid import uuid4

from pydantic import BaseModel

from app.models import (
    AgentFeatureSettings,
    CitationRecord,
    MessageRecord,
    RegistrationAnswersRecord,
    RegistrationFieldName,
    RegistrationRecommendationRecord,
    RegistrationSessionRecord,
    RegistrationSubmissionRecord,
    RetrievalHit,
    SearchHit,
)
from app.registration_store import RegistrationStore

FIELD_ORDER: tuple[RegistrationFieldName, ...] = (
    "full_name",
    "email",
    "location",
    "program_interest",
    "discovery_source",
    "degree_level",
    "desired_start_date",
)
AFFIRMATIVE_REPLIES = (
    "yes",
    "yes.",
    "yes!",
    "yes please",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "oui",
    "oui.",
    "oui!",
    "d'accord",
    "ok merci",
)
IMMEDIATE_TRIGGER_PHRASES = (
    "contact me",
    "be contacted",
    "call me",
    "email me",
    "reach out to me",
    "help me choose a program",
    "help me choose a programme",
    "which program should i choose",
    "which programme should i choose",
    "recommend a program",
    "recommend a programme",
    "i want to be contacted",
    "i would like to be contacted",
    "can someone contact me",
    "j'aimerais etre contacte",
    "je voudrais etre contacte",
    "peut-on me contacter",
    "peut on me contacter",
    "aidez-moi a choisir",
    "aide moi a choisir",
    "quel programme choisir",
    "quel programme me conseillez-vous",
    "quel programme me conseillez vous",
    "i want to join esilv",
    "i want to study at esilv",
    "i am interested in joining esilv",
    "i am interested in esilv courses",
    "i'm interested in joining esilv",
    "i'm interested in esilv courses",
    "help me choose courses",
    "help me choose courses at esilv",
    "help me choose a course",
    "help me choose a course at esilv",
    "je veux rejoindre l'esilv",
    "je veux rejoindre esilv",
    "je suis interesse par l'esilv",
    "je suis interesse par les cours de l'esilv",
)
FIRST_PERSON_INTEREST_MARKERS = (
    "i am interested",
    "i'm interested",
    "interested in",
    "i want",
    "i would like",
    "i'd like",
    "help me",
    "can you help me",
    "je suis interesse",
    "je veux",
    "je voudrais",
    "j'aimerais",
    "aidez-moi",
    "aide moi",
)
LEAD_INTENT_KEYWORDS = (
    "join",
    "joining",
    "study",
    "admission",
    "admissions",
    "apply",
    "application",
    "register",
    "registration",
    "program",
    "programme",
    "course",
    "courses",
    "cours",
    "formation",
    "formations",
    "esilv",
    "inscription",
    "candidature",
    "rejoindre",
)
OFFER_TRIGGER_KEYWORDS = (
    "admission",
    "admissions",
    "apply",
    "application",
    "deadline",
    "inscription",
    "concours",
    "candidature",
    "program",
    "programme",
    "course",
    "courses",
    "cours",
    "join",
    "joining",
    "study",
    "formation",
    "formations",
    "bachelor",
    "master",
    "msc",
)
NAME_UNKNOWN_MARKERS = (
    "not sure",
    "i don't know",
    "i do not know",
    "unknown",
    "prefer not to say",
    "pas sur",
    "je ne sais pas",
)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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


class SuperAgentProtocol(Protocol):
    def run(
        self,
        messages: list[MessageRecord],
        user_text: str,
        initial_action: str,
        initial_query: str,
        feature_settings: AgentFeatureSettings,
    ) -> MessageRecord: ...


@dataclass(frozen=True)
class ProgramRule:
    program_name: str
    degree_level: str
    keywords: tuple[str, ...]
    summary: str


PROGRAM_RULES: tuple[ProgramRule, ...] = (
    ProgramRule(
        program_name="Bachelor Informatique & Cybersecurite",
        degree_level="bachelor",
        keywords=("cyber", "cybersecurity", "security", "cloud", "informatique", "computer"),
        summary="It matches a bachelor-level profile focused on computing, connected systems, and cybersecurity.",
    ),
    ProgramRule(
        program_name="Bachelor Technologie & Management",
        degree_level="bachelor",
        keywords=("management", "business", "marketing", "startup", "entrepreneur", "technology"),
        summary="It fits a bachelor-level profile looking for a hybrid technology and management path.",
    ),
    ProgramRule(
        program_name="MSc Computer Science & Data Science",
        degree_level="master",
        keywords=("data science", "computer science", "software", "programming", "developer"),
        summary="It fits a master's profile focused on software, programming, and data-driven engineering.",
    ),
    ProgramRule(
        program_name="Data Engineering & AI",
        degree_level="master",
        keywords=("ai", "artificial intelligence", "machine learning", "data", "analytics"),
        summary="It aligns with a master's profile targeting AI, machine learning, and data engineering.",
    ),
    ProgramRule(
        program_name="MSc Cyber Resilience & Crisis Leadership",
        degree_level="master",
        keywords=("cyber", "cybersecurity", "resilience", "security"),
        summary="It suits a master's profile interested in cybersecurity, resilience, and crisis management.",
    ),
    ProgramRule(
        program_name="MSc Financial Engineering",
        degree_level="master",
        keywords=("finance", "financial", "quant", "trading", "risk"),
        summary="It is the closest master's fit for finance-oriented quantitative and risk topics.",
    ),
    ProgramRule(
        program_name="MSc Fintech & Digital Assets",
        degree_level="master",
        keywords=("fintech", "blockchain", "crypto", "digital assets"),
        summary="It fits a master's profile focused on fintech, blockchain, and digital-asset topics.",
    ),
    ProgramRule(
        program_name="MSc Aeronautical & Aerospatial Engineering",
        degree_level="master",
        keywords=("aerospace", "aeronautical", "aerospatial", "space", "avionics"),
        summary="It matches a master's profile targeting aerospace, aeronautics, and space systems.",
    ),
    ProgramRule(
        program_name="MSc Innovation & Creative Technology",
        degree_level="master",
        keywords=("creative", "design", "innovation", "ux", "product"),
        summary="It suits a master's profile combining innovation, design, and creative technology.",
    ),
    ProgramRule(
        program_name="MSc Game Programming",
        degree_level="master",
        keywords=("game", "gaming", "graphics", "real-time"),
        summary="It is the closest master's fit for game programming and interactive-technology interests.",
    ),
    ProgramRule(
        program_name="Industry & Robotics",
        degree_level="master",
        keywords=("robotics", "industry", "manufacturing", "automation", "industrial"),
        summary="It aligns with a master's profile interested in robotics, automation, and industrial systems.",
    ),
    ProgramRule(
        program_name="Aerospace & Defence",
        degree_level="master",
        keywords=("defence", "aerospace", "embedded", "systems"),
        summary="It fits a master's profile focused on aerospace, defence, and systems engineering.",
    ),
    ProgramRule(
        program_name="Master in Engineering",
        degree_level="master",
        keywords=("engineering", "generalist", "ingenieur"),
        summary="It is the best default fit for a master's-level prospect seeking the broader ESILV engineering track.",
    ),
)


class RegistrationAgent:
    def __init__(
        self,
        registration_store: RegistrationStore,
        search_agent: SearchAgentProtocol,
        retrieval_agent: RetrievalAgentProtocol,
        llm_client: LlmClientProtocol,
        super_agent: SuperAgentProtocol,
        max_search_hits: int,
        looks_french: Callable[[str], bool],
        search_is_weak: Callable[[str, list[SearchHit]], bool],
        citations_from_hits: Callable[[list[SearchHit]], list[CitationRecord]],
        citations_from_retrieval_hits: Callable[[list[RetrievalHit]], list[CitationRecord]],
        retrieval_answer_builder: Callable[[list[MessageRecord], str, list[RetrievalHit]], str],
        search_answer_builder: Callable[[list[MessageRecord], str, list[SearchHit]], str],
    ) -> None:
        self.registration_store = registration_store
        self.search_agent = search_agent
        self.retrieval_agent = retrieval_agent
        self.llm_client = llm_client
        self.super_agent = super_agent
        self.max_search_hits = max_search_hits
        self.looks_french = looks_french
        self.search_is_weak = search_is_weak
        self.citations_from_hits = citations_from_hits
        self.citations_from_retrieval_hits = citations_from_retrieval_hits
        self.retrieval_answer_builder = retrieval_answer_builder
        self.search_answer_builder = search_answer_builder

    def has_active_session(self, conversation_id: str) -> bool:
        return self.registration_store.load_session(conversation_id) is not None

    def should_start_from_follow_up(self, messages: list[MessageRecord], user_text: str) -> bool:
        if not self._is_affirmative_reply(user_text) or len(messages) < 2:
            return False
        previous_message = messages[-2]
        return previous_message.role == "assistant" and previous_message.pending_action == "registration"

    def should_start_immediately(self, user_text: str) -> bool:
        lowered = self._normalize(user_text)
        if any(phrase in lowered for phrase in IMMEDIATE_TRIGGER_PHRASES):
            return True
        return self._looks_like_explicit_lead_intent(lowered)

    def should_offer_registration(self, user_text: str, assistant_message: MessageRecord) -> bool:
        if assistant_message.pending_action is not None:
            return False
        lowered = self._normalize(user_text)
        if self.should_start_immediately(user_text):
            return False
        if any(marker in lowered for marker in ("pdf", "document", "uploaded", "report", "fichier")):
            return False
        return any(keyword in lowered for keyword in OFFER_TRIGGER_KEYWORDS) or self._looks_like_explicit_lead_intent(lowered)

    def start_session(self, conversation_id: str, user_text: str) -> MessageRecord:
        session = RegistrationSessionRecord(
            conversation_id=conversation_id,
            current_field=FIELD_ORDER[0],
            language="fr" if self.looks_french(user_text) else "en",
        )
        self.registration_store.save_session(session)
        return MessageRecord(role="assistant", content=self._start_message(session.language))

    def continue_session(
        self,
        conversation_id: str,
        messages: list[MessageRecord],
        user_text: str,
        feature_settings: AgentFeatureSettings,
    ) -> MessageRecord | None:
        session = self.registration_store.load_session(conversation_id)
        if session is None:
            return None

        extracted_answers = self._extract_answers(session, messages, user_text)
        updated_answers = self._merge_answers(session.answers, extracted_answers)
        extracted_fields = self._filled_fields(extracted_answers)

        current_value, error_message = self._resolve_current_field_value(
            field_name=session.current_field,
            answers=updated_answers,
            user_text=user_text,
            language=session.language,
            extracted_fields=extracted_fields,
        )
        if current_value is not None:
            updated_answers = updated_answers.model_copy(update={session.current_field: current_value})

        if error_message is not None:
            updated_session = session.model_copy(update={"answers": updated_answers})
            self.registration_store.save_session(updated_session)
            return MessageRecord(role="assistant", content=error_message)

        next_field = self._first_unanswered_field(updated_answers)
        if next_field is not None:
            updated_session = session.model_copy(
                update={
                    "answers": updated_answers,
                    "current_field": next_field,
                }
            )
            self.registration_store.save_session(updated_session)
            return MessageRecord(
                role="assistant",
                content=self._next_question_message(
                    language=session.language,
                    next_field=next_field,
                    answers=updated_answers,
                    messages=messages,
                    user_text=user_text,
                ),
            )

        try:
            recommendation = self._recommend_program(
                messages=messages,
                answers=updated_answers,
                language=session.language,
                feature_settings=feature_settings,
            )
        except Exception:
            seeded_rule = self._match_program_rule(updated_answers)
            recommendation = RegistrationRecommendationRecord(
                program_name=seeded_rule.program_name,
                message=self._rule_based_message(seeded_rule, session.language),
                source_mode="rules",
                citations=[],
            )

        submission = RegistrationSubmissionRecord(
            id=uuid4().hex,
            conversation_id=conversation_id,
            answers=updated_answers,
            recommendation=recommendation,
        )
        self.registration_store.save_submission(submission)
        self.registration_store.delete_session(conversation_id)
        return MessageRecord(
            role="assistant",
            content=self._completion_message(
                language=session.language,
                recommendation=recommendation,
            ),
            citations=recommendation.citations,
        )

    def append_registration_cta(self, assistant_message: MessageRecord, user_text: str) -> MessageRecord:
        content = f"{assistant_message.content}\n\n{self._cta_message('fr' if self.looks_french(user_text) else 'en')}"
        return assistant_message.model_copy(
            update={
                "content": content,
                "pending_action": "registration",
                "pending_query": None,
            }
        )

    def _extract_answers(
        self,
        session: RegistrationSessionRecord,
        messages: list[MessageRecord],
        user_text: str,
    ) -> RegistrationAnswersRecord:
        if not self.llm_client.configured:
            return RegistrationAnswersRecord()

        prompt = (
            "Extract only the registration details that are explicitly stated in the latest user reply.\n"
            f"Language: {session.language}\n"
            f"Current field being asked: {session.current_field}\n"
            f"Latest user reply: {user_text}\n\n"
            "Already collected answers:\n"
            f"{self._answers_for_prompt(session.answers)}\n\n"
            "Recent conversation:\n"
            f"{self._messages_for_prompt(messages[-6:])}\n\n"
            "Rules:\n"
            "- Return JSON only.\n"
            "- Extract only information clearly present in the latest user reply.\n"
            "- Leave absent fields as null.\n"
            "- Keep the user's wording concise and do not invent details.\n"
            "- If the user says they are unsure about bachelor or master, keep that uncertainty as text instead of forcing a choice.\n"
            "- Do not rewrite uncertainty into bachelor or master.\n"
        )
        try:
            extracted = self.llm_client.generate_structured(
                prompt=prompt,
                system_instruction=(
                    "You extract structured facts for an ESILV registration flow. "
                    "Be conservative and only capture explicitly stated information."
                ),
                schema=RegistrationAnswersRecord,
                temperature=0.0,
            )
            return RegistrationAnswersRecord.model_validate(extracted)
        except Exception:
            return RegistrationAnswersRecord()

    def _merge_answers(
        self,
        existing: RegistrationAnswersRecord,
        extracted: RegistrationAnswersRecord,
    ) -> RegistrationAnswersRecord:
        updates: dict[str, str] = {}
        for field_name in FIELD_ORDER:
            candidate = self._normalize_extracted_value(field_name, getattr(extracted, field_name))
            if candidate is not None:
                updates[field_name] = candidate
        if not updates:
            return existing
        return existing.model_copy(update=updates)

    def _resolve_current_field_value(
        self,
        field_name: RegistrationFieldName,
        answers: RegistrationAnswersRecord,
        user_text: str,
        language: str,
        extracted_fields: set[RegistrationFieldName],
    ) -> tuple[str | None, str | None]:
        existing_value = self._normalize_extracted_value(field_name, getattr(answers, field_name))
        raw_value = " ".join(user_text.split())

        if field_name == "email":
            if existing_value is not None:
                return existing_value, None
            raw_email = raw_value.lower()
            if raw_email and EMAIL_PATTERN.match(raw_email):
                return raw_email, None
            return None, self._retry_message(field_name, language, reason="invalid_email")

        if field_name == "full_name":
            if existing_value is not None:
                return existing_value, None
            if raw_value and not self._looks_like_unknown_name(raw_value) and not EMAIL_PATTERN.match(raw_value):
                return raw_value, None
            return None, self._retry_message(field_name, language, reason="missing_name")

        if existing_value is not None:
            return existing_value, None
        if extracted_fields:
            return None, self._retry_message(field_name, language, reason="missing_field")
        if raw_value:
            return raw_value, None
        return None, self._retry_message(field_name, language, reason="missing_field")

    def _recommend_program(
        self,
        messages: list[MessageRecord],
        answers: RegistrationAnswersRecord,
        language: str,
        feature_settings: AgentFeatureSettings,
    ) -> RegistrationRecommendationRecord:
        seeded_rule = self._match_program_rule(answers)
        query = self._recommendation_query(answers, seeded_rule.program_name, language)

        if self.llm_client.configured:
            if feature_settings.super_agent_enabled and (feature_settings.rag_enabled or feature_settings.web_search_enabled):
                super_response = self.super_agent.run(
                    messages=messages,
                    user_text=query,
                    initial_action="search" if feature_settings.web_search_enabled else "retrieve",
                    initial_query=query,
                    feature_settings=feature_settings,
                )
                if super_response.citations:
                    program_name = self._detect_program_name(super_response.content, seeded_rule.program_name)
                    return RegistrationRecommendationRecord(
                        program_name=program_name,
                        message=super_response.content,
                        source_mode="super",
                        citations=super_response.citations,
                    )

            if feature_settings.web_search_enabled:
                search_hits = self.search_agent.search(query, top_k=self.max_search_hits)
                if not self.search_is_weak(query, search_hits):
                    message = self.search_answer_builder(messages, query, search_hits)
                    return RegistrationRecommendationRecord(
                        program_name=self._detect_program_name(message, seeded_rule.program_name),
                        message=message,
                        source_mode="search",
                        citations=self.citations_from_hits(search_hits),
                    )

            if feature_settings.rag_enabled:
                retrieval_hits = self.retrieval_agent.search(query, top_k=self.max_search_hits)
                if not self.retrieval_agent.is_weak(retrieval_hits):
                    message = self.retrieval_answer_builder(messages, query, retrieval_hits)
                    return RegistrationRecommendationRecord(
                        program_name=self._detect_program_name(message, seeded_rule.program_name),
                        message=message,
                        source_mode="retrieve",
                        citations=self.citations_from_retrieval_hits(retrieval_hits),
                    )

        return RegistrationRecommendationRecord(
            program_name=seeded_rule.program_name,
            message=self._rule_based_message(seeded_rule, language),
            source_mode="rules",
            citations=[],
        )

    @staticmethod
    def _normalize_degree_level(value: str) -> str | None:
        lowered = RegistrationAgent._normalize(value)
        bachelor_markers = ("bachelor", "licence", "undergraduate", "bac+3")
        master_markers = ("master", "msc", "ms", "graduate", "bac+5", "ingenieur")
        if any(marker in lowered for marker in bachelor_markers):
            return "bachelor"
        if any(marker in lowered for marker in master_markers):
            return "master"
        return None

    def _match_program_rule(self, answers: RegistrationAnswersRecord) -> ProgramRule:
        degree_level = self._normalize_degree_level(answers.degree_level or "") or "master"
        interest_text = self._normalize(answers.program_interest or "")

        matching_rules = [rule for rule in PROGRAM_RULES if rule.degree_level == degree_level]
        ranked_rules = sorted(
            (
                (
                    sum(1 for keyword in rule.keywords if keyword in interest_text),
                    len(rule.keywords),
                    rule,
                )
                for rule in matching_rules
            ),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        if ranked_rules and ranked_rules[0][0] > 0:
            return ranked_rules[0][2]

        fallback_rule = next((rule for rule in matching_rules if rule.program_name == "Master in Engineering"), None)
        if fallback_rule is not None:
            return fallback_rule
        return matching_rules[0]

    def _detect_program_name(self, content: str, default_program_name: str) -> str:
        lowered = self._normalize(content)
        for rule in PROGRAM_RULES:
            if self._normalize(rule.program_name) in lowered:
                return rule.program_name
        return default_program_name

    def _recommendation_query(
        self,
        answers: RegistrationAnswersRecord,
        seeded_program_name: str,
        language: str,
    ) -> str:
        degree = self._clamp(answers.degree_level or "not specified", 20)
        interest = self._clamp(answers.program_interest or "not specified", 60)
        location = self._clamp(answers.location or "not specified", 40)
        start_date = self._clamp(answers.desired_start_date or "not specified", 30)
        discovery = self._clamp(answers.discovery_source or "not specified", 30)
        seeded_program = self._clamp(seeded_program_name, 50)
        if language == "fr":
            return (
                "Recommande exactement un programme ESILV. "
                f"Niveau: {degree}. Interet: {interest}. Lieu: {location}. "
                f"Rentree: {start_date}. Source: {discovery}. "
                f"Si les sources le permettent, privilegie {seeded_program}. "
                "Donne le nom du programme puis une justification breve."
            )
        return (
            "Recommend exactly one ESILV program. "
            f"Degree: {degree}. Interest: {interest}. Location: {location}. "
            f"Start: {start_date}. Source: {discovery}. "
            f"If the evidence supports it, prioritize {seeded_program}. "
            "Return the program name first, then a brief explanation."
        )

    def _rule_based_message(self, rule: ProgramRule, language: str) -> str:
        if language == "fr":
            return f"Le programme ESILV le plus adapte semble etre {rule.program_name}. {rule.summary}"
        return f"The most suitable ESILV program looks to be {rule.program_name}. {rule.summary}"

    def _completion_message(
        self,
        language: str,
        recommendation: RegistrationRecommendationRecord,
    ) -> str:
        if language == "fr":
            return (
                "Merci, votre formulaire a bien ete enregistre.\n\n"
                f"Programme recommande: {recommendation.program_name}\n\n"
                f"{recommendation.message}"
            )
        return (
            "Thank you, your registration form has been saved.\n\n"
            f"Recommended program: {recommendation.program_name}\n\n"
            f"{recommendation.message}"
        )

    def _start_message(self, language: str) -> str:
        fallback = (
            "Je peux m'en charger. Je vais vous guider pas a pas pour le formulaire de contact ESILV.\n\n"
            f"{self._question_for('full_name', language)}"
            if language == "fr"
            else "I can help with that. I will guide you through the ESILV contact form step by step.\n\n"
            f"{self._question_for('full_name', language)}"
        )
        if not self.llm_client.configured:
            return fallback

        prompt = (
            "You are starting an ESILV registration conversation.\n"
            f"Language: {language}\n"
            "Instructions:\n"
            "- Briefly say you can help with the registration/contact request.\n"
            "- Ask only the first question.\n"
            "- Sound like a human admissions assistant, not a rigid form.\n"
            "- Keep it to at most two short sentences.\n"
            f"Fallback question: {fallback}\n"
        )
        return self._generate_guided_text(prompt=prompt, fallback=fallback)

    def _next_question_message(
        self,
        language: str,
        next_field: RegistrationFieldName,
        answers: RegistrationAnswersRecord,
        messages: list[MessageRecord],
        user_text: str,
    ) -> str:
        fallback = self._question_for(next_field, language)
        if not self.llm_client.configured:
            return fallback

        prompt = (
            "Write the next assistant message for an ESILV registration flow.\n"
            f"Language: {language}\n"
            f"Latest user reply: {user_text}\n"
            f"Next field to ask: {next_field}\n"
            f"Collected answers:\n{self._answers_for_prompt(answers)}\n\n"
            f"Recent conversation:\n{self._messages_for_prompt(messages[-6:])}\n\n"
            "Instructions:\n"
            "- Ask exactly one next question.\n"
            "- Briefly acknowledge the latest user reply when useful.\n"
            "- If the user is unsure, accept the uncertainty and move on naturally.\n"
            "- Do not force the user to choose bachelor or master if they are undecided.\n"
            "- Do not ask about fields that are already collected.\n"
            "- Keep it under two sentences.\n"
            f"Fallback question: {fallback}\n"
        )
        return self._generate_guided_text(prompt=prompt, fallback=fallback)

    def _retry_message(
        self,
        field_name: RegistrationFieldName,
        language: str,
        reason: str,
    ) -> str:
        fallback = self._fallback_retry_message(field_name, language, reason)
        if not self.llm_client.configured:
            return fallback

        prompt = (
            "Write a short follow-up message for an ESILV registration flow.\n"
            f"Language: {language}\n"
            f"Field that still needs to be collected: {field_name}\n"
            f"Reason: {reason}\n"
            "Instructions:\n"
            "- Ask for the same field again in a natural way.\n"
            "- Be polite and concise.\n"
            "- Ask exactly one question.\n"
            "- Keep it under two sentences.\n"
            f"Fallback question: {fallback}\n"
        )
        return self._generate_guided_text(prompt=prompt, fallback=fallback)

    def _generate_guided_text(self, prompt: str, fallback: str) -> str:
        try:
            content = self.llm_client.generate_text(
                prompt=prompt,
                system_instruction=(
                    "You are the ESILV Registration Agent. "
                    "Guide the user conversationally while collecting one missing field at a time."
                ),
                temperature=0.3,
            ).strip()
        except Exception:
            return fallback
        return content or fallback

    @staticmethod
    def _question_for(field_name: RegistrationFieldName, language: str) -> str:
        prompts = {
            "en": {
                "full_name": "What is your full name?",
                "email": "What email address should we use to contact you?",
                "location": "Where do you currently live?",
                "program_interest": "Which ESILV program or subject area interests you the most right now?",
                "discovery_source": "How did you hear about ESILV?",
                "degree_level": "Are you currently leaning more toward a bachelor or a master, or are you still deciding?",
                "desired_start_date": "When would you ideally like to start?",
            },
            "fr": {
                "full_name": "Quel est votre nom complet ?",
                "email": "Quelle adresse email devons-nous utiliser pour vous contacter ?",
                "location": "Ou habitez-vous actuellement ?",
                "program_interest": "Quel programme ESILV ou quel domaine vous interesse le plus pour le moment ?",
                "discovery_source": "Comment avez-vous connu l'ESILV ?",
                "degree_level": "Vous orientez-vous plutot vers un bachelor, un master, ou etes-vous encore en reflexion ?",
                "desired_start_date": "Quand aimeriez-vous idealement commencer ?",
            },
        }
        return prompts[language][field_name]

    def _cta_message(self, language: str) -> str:
        if language == "fr":
            return (
                "Si vous voulez, je peux aussi lancer un formulaire de contact ESILV et recueillir vos informations "
                "pas a pas pour qu'un suivi soit possible. Repondez simplement \"oui\"."
            )
        return (
            "If you want, I can also start an ESILV contact form and collect your details step by step "
            "so an admissions follow-up is possible. Just reply with \"yes\"."
        )

    def _fallback_retry_message(
        self,
        field_name: RegistrationFieldName,
        language: str,
        reason: str,
    ) -> str:
        if field_name == "email" and reason == "invalid_email":
            if language == "fr":
                return "Je n'ai pas reconnu une adresse email valide. Quelle adresse email devons-nous utiliser ?"
            return "I did not catch a valid email address. What email should we use to contact you?"
        if field_name == "full_name":
            if language == "fr":
                return "J'ai encore besoin de votre nom complet pour continuer. Quel est votre nom complet ?"
            return "I still need your full name to continue. What is your full name?"
        if language == "fr":
            return f"J'ai encore besoin de cette information pour continuer. {self._question_for(field_name, language)}"
        return f"I still need that detail to continue. {self._question_for(field_name, language)}"

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _clamp(text: str, limit: int) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= limit:
            return collapsed
        return f"{collapsed[: limit - 3].rstrip()}..."

    @staticmethod
    def _is_affirmative_reply(text: str) -> bool:
        lowered = RegistrationAgent._normalize(text)
        return any(lowered == marker or lowered.startswith(f"{marker} ") for marker in AFFIRMATIVE_REPLIES)

    @staticmethod
    def _looks_like_explicit_lead_intent(lowered: str) -> bool:
        return any(marker in lowered for marker in FIRST_PERSON_INTEREST_MARKERS) and any(
            keyword in lowered for keyword in LEAD_INTENT_KEYWORDS
        )

    @staticmethod
    def _messages_for_prompt(messages: list[MessageRecord]) -> str:
        if not messages:
            return "(none)"
        return "\n".join(f"{message.role}: {message.content}" for message in messages)

    @staticmethod
    def _answers_for_prompt(answers: RegistrationAnswersRecord) -> str:
        lines = []
        for field_name in FIELD_ORDER:
            value = getattr(answers, field_name)
            lines.append(f"- {field_name}: {value or 'null'}")
        return "\n".join(lines)

    @staticmethod
    def _filled_fields(answers: RegistrationAnswersRecord) -> set[RegistrationFieldName]:
        return {
            field_name
            for field_name in FIELD_ORDER
            if isinstance(getattr(answers, field_name), str) and getattr(answers, field_name).strip()
        }

    def _normalize_extracted_value(
        self,
        field_name: RegistrationFieldName,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split())
        if not normalized:
            return None
        if field_name == "email":
            lowered = normalized.lower()
            return lowered if EMAIL_PATTERN.match(lowered) else None
        if field_name == "full_name":
            if self._looks_like_unknown_name(normalized) or EMAIL_PATTERN.match(normalized):
                return None
        return normalized

    def _looks_like_unknown_name(self, value: str) -> bool:
        lowered = self._normalize(value)
        return any(marker in lowered for marker in NAME_UNKNOWN_MARKERS)

    @staticmethod
    def _first_unanswered_field(answers: RegistrationAnswersRecord) -> RegistrationFieldName | None:
        for field_name in FIELD_ORDER:
            value = getattr(answers, field_name)
            if value is None or not str(value).strip():
                return field_name
        return None
