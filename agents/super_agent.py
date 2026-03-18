from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from app.models import (
    AgentFeatureSettings,
    CitationRecord,
    MessageRecord,
    RetrievalHit,
    RetrievalOutcomeSummary,
    SearchHit,
    SearchOutcomeSummary,
    SuperAgentEvaluationRecord,
    SuperAgentTraceRecord,
)

logger = logging.getLogger(__name__)

Strategy = Literal["retrieve", "search", "hybrid"]


class SearchAgentProtocol(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]: ...


class RetrievalAgentProtocol(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]: ...

    def is_weak(self, hits: list[RetrievalHit]) -> bool: ...


class LlmClientProtocol(Protocol):
    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        schema: type[SuperAgentEvaluationRecord],
        temperature: float = 0.0,
    ) -> SuperAgentEvaluationRecord: ...


@dataclass
class CandidateRecord:
    content: str
    citations: list[CitationRecord]
    evaluation: SuperAgentEvaluationRecord
    has_strong_sources: bool
    meets_source_coverage: bool
    strong_source_count: int
    iteration_number: int

    @property
    def rank(self) -> tuple[bool, bool, int, int, int]:
        effective_grounded = self.evaluation.is_grounded and not self.evaluation.unsupported_claims
        effective_sufficient = (
            self.evaluation.is_sufficient
            and effective_grounded
            and self.has_strong_sources
            and self.meets_source_coverage
        )
        return (
            effective_sufficient,
            effective_grounded and self.has_strong_sources and self.meets_source_coverage,
            self.strong_source_count,
            -len(self.evaluation.missing_information),
            self.iteration_number,
        )


@dataclass
class NextStepDecision:
    stop_reason: str | None
    next_strategy: Strategy | None = None
    next_query: str | None = None


class SuperAgent:
    def __init__(
        self,
        search_agent: SearchAgentProtocol,
        retrieval_agent: RetrievalAgentProtocol,
        llm_client: LlmClientProtocol,
        max_search_hits: int,
        format_messages: Callable[[list[MessageRecord]], str],
        search_is_weak: Callable[[str, list[SearchHit]], bool],
        citations_from_hits: Callable[[list[SearchHit]], list[CitationRecord]],
        citations_from_retrieval_hits: Callable[[list[RetrievalHit]], list[CitationRecord]],
        retrieval_answer_builder: Callable[[list[MessageRecord], str, list[RetrievalHit]], str],
        search_answer_builder: Callable[[list[MessageRecord], str, list[SearchHit]], str],
        hybrid_answer_builder: Callable[
            [list[MessageRecord], str, list[RetrievalHit], list[SearchHit]],
            str,
        ],
        looks_french: Callable[[str], bool],
        is_retrieval_intent: Callable[[str], bool],
        is_search_intent: Callable[[str], bool],
        has_independent_search_intent: Callable[[str], bool],
    ) -> None:
        self.search_agent = search_agent
        self.retrieval_agent = retrieval_agent
        self.llm_client = llm_client
        self.max_search_hits = max_search_hits
        self.format_messages = format_messages
        self.search_is_weak = search_is_weak
        self.citations_from_hits = citations_from_hits
        self.citations_from_retrieval_hits = citations_from_retrieval_hits
        self.retrieval_answer_builder = retrieval_answer_builder
        self.search_answer_builder = search_answer_builder
        self.hybrid_answer_builder = hybrid_answer_builder
        self.looks_french = looks_french
        self.is_retrieval_intent = is_retrieval_intent
        self.is_search_intent = is_search_intent
        self.has_independent_search_intent = has_independent_search_intent

    def run(
        self,
        messages: list[MessageRecord],
        user_text: str,
        initial_action: Literal["retrieve", "search"],
        initial_query: str,
        feature_settings: AgentFeatureSettings,
    ) -> MessageRecord:
        current_strategy = self._initial_strategy(
            initial_action=initial_action,
            feature_settings=feature_settings,
        )
        current_query = self._initial_query(
            user_text=user_text,
            initial_action=initial_action,
            initial_query=initial_query,
            current_strategy=current_strategy,
        )
        requires_combined_sources = self._requires_combined_sources(user_text, feature_settings)
        tried_pairs: set[tuple[str, str]] = set()
        trace: list[SuperAgentTraceRecord] = []
        best_candidate: CandidateRecord | None = None
        accumulated_retrieval_hits: list[RetrievalHit] = []
        accumulated_search_hits: list[SearchHit] = []
        accumulated_retrieval_has_strong = False
        accumulated_search_has_strong = False
        best_retrieval_hits: list[RetrievalHit] | None = None
        best_search_hits: list[SearchHit] | None = None
        final_stop_reason: str | None = None

        for iteration_number in range(1, 4):
            normalized_pair = (current_strategy, self._normalize_query(current_query))
            if normalized_pair in tried_pairs:
                final_stop_reason = "redundant_query_blocked"
                break
            tried_pairs.add(normalized_pair)

            retrieval_hits, retrieval_summary = self._run_retrieval_if_needed(
                strategy=current_strategy,
                query=current_query,
                feature_settings=feature_settings,
            )
            search_hits, search_summary = self._run_search_if_needed(
                strategy=current_strategy,
                query=current_query,
                feature_settings=feature_settings,
            )
            accumulated_retrieval_hits = self._merge_retrieval_hit_lists(accumulated_retrieval_hits, retrieval_hits)
            accumulated_search_hits = self._merge_search_hit_lists(accumulated_search_hits, search_hits)
            if self._is_strong_outcome(retrieval_summary):
                accumulated_retrieval_has_strong = True
            if self._is_strong_outcome(search_summary):
                accumulated_search_has_strong = True
            draft_answer = self._build_draft_answer(
                strategy=current_strategy,
                messages=messages,
                user_text=user_text,
                retrieval_hits=accumulated_retrieval_hits,
                search_hits=accumulated_search_hits,
            )
            evaluation = self._evaluate_iteration(
                messages=messages,
                user_text=user_text,
                strategy=current_strategy,
                query=current_query,
                retrieval_hits=accumulated_retrieval_hits,
                search_hits=accumulated_search_hits,
                draft_answer=draft_answer,
                trace=trace,
                feature_settings=feature_settings,
            )
            citations = self._merge_citations(
                self.citations_from_retrieval_hits(accumulated_retrieval_hits),
                self.citations_from_hits(accumulated_search_hits),
            )
            if accumulated_retrieval_hits:
                best_retrieval_hits = list(accumulated_retrieval_hits)
            if accumulated_search_hits:
                best_search_hits = list(accumulated_search_hits)
            has_strong_sources = accumulated_retrieval_has_strong or accumulated_search_has_strong
            meets_source_coverage = self._meets_source_coverage_from_flags(
                requires_combined_sources=requires_combined_sources,
                retrieval_has_strong=accumulated_retrieval_has_strong,
                search_has_strong=accumulated_search_has_strong,
            )
            strong_source_count = (
                len(accumulated_retrieval_hits) if accumulated_retrieval_has_strong else 0
            ) + (
                len(accumulated_search_hits) if accumulated_search_has_strong else 0
            )
            candidate = CandidateRecord(
                content=draft_answer,
                citations=citations,
                evaluation=evaluation,
                has_strong_sources=has_strong_sources,
                meets_source_coverage=meets_source_coverage,
                strong_source_count=strong_source_count,
                iteration_number=iteration_number,
            )
            if best_candidate is None or candidate.rank > best_candidate.rank:
                best_candidate = candidate

            next_step = self._determine_next_step(
                iteration_number=iteration_number,
                user_text=user_text,
                current_strategy=current_strategy,
                current_query=current_query,
                evaluation=evaluation,
                retrieval_summary=retrieval_summary,
                search_summary=search_summary,
                feature_settings=feature_settings,
                requires_combined_sources=requires_combined_sources,
                accumulated_retrieval_has_strong=accumulated_retrieval_has_strong,
                accumulated_search_has_strong=accumulated_search_has_strong,
                tried_pairs=tried_pairs,
            )
            trace_record = SuperAgentTraceRecord(
                iteration_number=iteration_number,
                selected_strategy=current_strategy,
                executed_query=current_query,
                retrieval_outcome=retrieval_summary,
                search_outcome=search_summary,
                draft_answer=draft_answer,
                evaluator_result=evaluation,
                rewritten_query=evaluation.rewritten_query,
                stop_reason=next_step.stop_reason,
            )
            trace.append(trace_record)
            logger.info(
                "SuperAgent iteration=%s strategy=%s query=%r retrieval_hits=%s search_hits=%s stop_reason=%s",
                iteration_number,
                current_strategy,
                current_query,
                retrieval_summary.hit_count if retrieval_summary else 0,
                search_summary.hit_count if search_summary else 0,
                next_step.stop_reason,
            )

            if next_step.stop_reason is not None:
                final_stop_reason = next_step.stop_reason
                break

            current_strategy = next_step.next_strategy or current_strategy
            current_query = next_step.next_query or current_query

        if final_stop_reason is None:
            final_stop_reason = "max_iterations_reached"
            if trace:
                trace[-1].stop_reason = final_stop_reason

        combined_content: str | None = None
        combined_citations: list[CitationRecord] = []
        if requires_combined_sources and best_retrieval_hits and best_search_hits:
            combined_content = self.hybrid_answer_builder(
                messages,
                user_text,
                best_retrieval_hits,
                best_search_hits,
            )
            combined_citations = self._merge_citations(
                self.citations_from_retrieval_hits(best_retrieval_hits),
                self.citations_from_hits(best_search_hits),
            )

        if (
            best_candidate is not None
            and best_candidate.evaluation.is_sufficient
            and best_candidate.evaluation.is_grounded
            and not best_candidate.evaluation.unsupported_claims
            and best_candidate.has_strong_sources
            and best_candidate.meets_source_coverage
        ):
            content = best_candidate.content
            citations = best_candidate.citations
        elif combined_content is not None and best_candidate is not None:
            if best_candidate.evaluation.is_sufficient and best_candidate.evaluation.is_grounded:
                content = combined_content
            else:
                content = self._force_limitation_notice(
                    user_text=user_text,
                    content=combined_content,
                    evaluation=best_candidate.evaluation,
                )
            citations = combined_citations
        elif best_candidate is not None and best_candidate.citations:
            content = self._force_limitation_notice(
                user_text=user_text,
                content=best_candidate.content,
                evaluation=best_candidate.evaluation,
            )
            citations = best_candidate.citations
        else:
            content = self._no_grounded_answer_message(user_text, final_stop_reason)
            citations = []

        return MessageRecord(
            role="assistant",
            content=content,
            citations=citations,
            orchestration_mode="super",
            super_agent_stop_reason=final_stop_reason,
            super_agent_trace=trace,
        )

    def _initial_strategy(
        self,
        initial_action: Literal["retrieve", "search"],
        feature_settings: AgentFeatureSettings,
    ) -> Strategy:
        if feature_settings.rag_enabled and feature_settings.web_search_enabled:
            return "hybrid"
        if feature_settings.rag_enabled:
            return "retrieve"
        if feature_settings.web_search_enabled:
            return "search"
        return initial_action

    @staticmethod
    def _initial_query(
        user_text: str,
        initial_action: Literal["retrieve", "search"],
        initial_query: str,
        current_strategy: Strategy,
    ) -> str:
        if current_strategy == "hybrid":
            return user_text
        if current_strategy == "retrieve" and initial_action == "search":
            return user_text
        return initial_query or user_text

    def _run_retrieval_if_needed(
        self,
        strategy: Strategy,
        query: str,
        feature_settings: AgentFeatureSettings,
    ) -> tuple[list[RetrievalHit], RetrievalOutcomeSummary | None]:
        if strategy not in {"retrieve", "hybrid"} or not feature_settings.rag_enabled:
            return [], None

        hits = self.retrieval_agent.search(query, top_k=self.max_search_hits)
        weak = self.retrieval_agent.is_weak(hits)
        top_hit = hits[0] if hits else None
        summary = RetrievalOutcomeSummary(
            hit_count=len(hits),
            weak=weak,
            top_score=top_hit.score if top_hit else None,
            top_lexical_overlap=top_hit.lexical_overlap if top_hit else None,
            top_citations=[
                self._format_document_citation(hit)
                for hit in hits[: min(3, len(hits))]
            ],
        )
        return hits, summary

    def _run_search_if_needed(
        self,
        strategy: Strategy,
        query: str,
        feature_settings: AgentFeatureSettings,
    ) -> tuple[list[SearchHit], SearchOutcomeSummary | None]:
        if strategy not in {"search", "hybrid"} or not feature_settings.web_search_enabled:
            return [], None

        hits = self.search_agent.search(query, top_k=self.max_search_hits)
        weak = self.search_is_weak(query, hits)
        top_hit = hits[0] if hits else None
        summary = SearchOutcomeSummary(
            hit_count=len(hits),
            weak=weak,
            top_score=top_hit.score if top_hit else None,
            top_lexical_overlap=top_hit.lexical_overlap if top_hit else None,
            top_expanded_overlap=top_hit.expanded_overlap if top_hit else None,
            top_citations=[hit.url for hit in hits[: min(3, len(hits))]],
        )
        return hits, summary

    def _build_draft_answer(
        self,
        strategy: Strategy,
        messages: list[MessageRecord],
        user_text: str,
        retrieval_hits: list[RetrievalHit],
        search_hits: list[SearchHit],
    ) -> str:
        if retrieval_hits and search_hits:
            return self.hybrid_answer_builder(messages, user_text, retrieval_hits, search_hits)
        if retrieval_hits:
            return self.retrieval_answer_builder(messages, user_text, retrieval_hits)
        if search_hits:
            return self.search_answer_builder(messages, user_text, search_hits)
        return self._empty_source_message(user_text, strategy)

    def _evaluate_iteration(
        self,
        messages: list[MessageRecord],
        user_text: str,
        strategy: Strategy,
        query: str,
        retrieval_hits: list[RetrievalHit],
        search_hits: list[SearchHit],
        draft_answer: str,
        trace: list[SuperAgentTraceRecord],
        feature_settings: AgentFeatureSettings,
    ) -> SuperAgentEvaluationRecord:
        available = ", ".join(self._available_strategies(feature_settings))
        prompt = (
            "Evaluate whether the grounded draft answer fully answers the ORIGINAL user question.\n"
            "Judge the draft against the full original question, not only the latest rewritten query.\n"
            "If any material part is unanswered, unclear, or unsupported by the evidence, mark the answer insufficient.\n"
            "List unsupported claims explicitly when the draft says more than the evidence supports.\n"
            f"Available next strategies: {available}.\n\n"
            f"Conversation:\n{self.format_messages(messages[-8:])}\n\n"
            f"Original user question:\n{user_text}\n\n"
            f"Current strategy: {strategy}\n"
            f"Executed query: {query}\n\n"
            f"Draft answer:\n{draft_answer}\n\n"
            f"Retrieved document evidence:\n{self._format_retrieval_hits(retrieval_hits)}\n\n"
            f"Web search evidence:\n{self._format_search_hits(search_hits)}\n\n"
            f"Iteration history:\n{self._format_trace_history(trace)}\n"
        )
        result = self.llm_client.generate_structured(
            prompt=prompt,
            system_instruction="Return only the evaluation JSON.",
            schema=SuperAgentEvaluationRecord,
            temperature=0.0,
        )
        return SuperAgentEvaluationRecord.model_validate(result)

    def _determine_next_step(
        self,
        iteration_number: int,
        user_text: str,
        current_strategy: Strategy,
        current_query: str,
        evaluation: SuperAgentEvaluationRecord,
        retrieval_summary: RetrievalOutcomeSummary | None,
        search_summary: SearchOutcomeSummary | None,
        feature_settings: AgentFeatureSettings,
        requires_combined_sources: bool,
        accumulated_retrieval_has_strong: bool,
        accumulated_search_has_strong: bool,
        tried_pairs: set[tuple[str, str]],
    ) -> NextStepDecision:
        has_strong_sources = accumulated_retrieval_has_strong or accumulated_search_has_strong
        meets_source_coverage = self._meets_source_coverage_from_flags(
            requires_combined_sources=requires_combined_sources,
            retrieval_has_strong=accumulated_retrieval_has_strong,
            search_has_strong=accumulated_search_has_strong,
        )
        if (
            evaluation.is_sufficient
            and evaluation.is_grounded
            and not evaluation.unsupported_claims
            and has_strong_sources
            and meets_source_coverage
        ):
            return NextStepDecision(stop_reason=evaluation.stop_reason or "answer_sufficient")

        if iteration_number >= 3:
            return NextStepDecision(stop_reason="max_iterations_reached")

        rewritten_query = (evaluation.rewritten_query or current_query).strip() or current_query
        if feature_settings.rag_enabled and feature_settings.web_search_enabled:
            hybrid_pair = ("hybrid", self._normalize_query(rewritten_query))
            if hybrid_pair not in tried_pairs:
                return NextStepDecision(
                    stop_reason=None,
                    next_strategy="hybrid",
                    next_query=rewritten_query,
                )
            return NextStepDecision(stop_reason=evaluation.stop_reason or "no_novel_next_step")

        if requires_combined_sources and current_strategy != "hybrid" and "hybrid" in self._available_strategies(feature_settings):
            hybrid_pair = ("hybrid", self._normalize_query(user_text))
            if hybrid_pair not in tried_pairs:
                return NextStepDecision(
                    stop_reason=None,
                    next_strategy="hybrid",
                    next_query=user_text,
                )

        next_strategy = self._coerce_strategy(
            evaluation.next_strategy or current_strategy,
            feature_settings,
            fallback=current_strategy,
        )
        alternate = self._alternate_strategy(current_strategy, feature_settings)
        current_is_weak = self._current_strategy_is_weak(
            current_strategy=current_strategy,
            retrieval_summary=retrieval_summary,
            search_summary=search_summary,
        )
        if current_is_weak and alternate is not None:
            alternate_pair = (alternate, self._normalize_query(rewritten_query))
            if alternate_pair not in tried_pairs:
                return NextStepDecision(
                    stop_reason=None,
                    next_strategy=alternate,
                    next_query=rewritten_query,
                )

        next_pair = (next_strategy, self._normalize_query(rewritten_query))
        if next_pair not in tried_pairs:
            return NextStepDecision(stop_reason=None, next_strategy=next_strategy, next_query=rewritten_query)

        for fallback_strategy in self._available_strategies(feature_settings):
            fallback_pair = (fallback_strategy, self._normalize_query(rewritten_query))
            if fallback_pair not in tried_pairs:
                return NextStepDecision(
                    stop_reason=None,
                    next_strategy=fallback_strategy,
                    next_query=rewritten_query,
                )

        return NextStepDecision(stop_reason=evaluation.stop_reason or "no_novel_next_step")

    def _available_strategies(self, feature_settings: AgentFeatureSettings) -> tuple[Strategy, ...]:
        strategies: list[Strategy] = []
        if feature_settings.rag_enabled:
            strategies.append("retrieve")
        if feature_settings.web_search_enabled:
            strategies.append("search")
        if feature_settings.rag_enabled and feature_settings.web_search_enabled:
            strategies.append("hybrid")
        return tuple(strategies)

    def _coerce_strategy(
        self,
        preferred: str,
        feature_settings: AgentFeatureSettings,
        fallback: Strategy,
    ) -> Strategy:
        available = self._available_strategies(feature_settings)
        if preferred in available:
            return preferred  # type: ignore[return-value]
        return fallback if fallback in available else available[0]

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query).strip().lower()

    def _requires_combined_sources(
        self,
        user_text: str,
        feature_settings: AgentFeatureSettings,
    ) -> bool:
        if not (feature_settings.rag_enabled and feature_settings.web_search_enabled):
            return False

        lowered = user_text.lower()
        return (
            self.is_retrieval_intent(lowered)
            and self.is_search_intent(lowered)
            and (
                self.has_independent_search_intent(lowered)
                or any(marker in lowered for marker in (" and ", " et ", " also ", " plus ", " as well as "))
            )
        )

    @staticmethod
    def _meets_source_coverage(
        requires_combined_sources: bool,
        retrieval_summary: RetrievalOutcomeSummary | None,
        search_summary: SearchOutcomeSummary | None,
    ) -> bool:
        return SuperAgent._meets_source_coverage_from_flags(
            requires_combined_sources=requires_combined_sources,
            retrieval_has_strong=SuperAgent._is_strong_outcome(retrieval_summary),
            search_has_strong=SuperAgent._is_strong_outcome(search_summary),
        )

    @staticmethod
    def _meets_source_coverage_from_flags(
        requires_combined_sources: bool,
        retrieval_has_strong: bool,
        search_has_strong: bool,
    ) -> bool:
        if not requires_combined_sources:
            return True

        return retrieval_has_strong and search_has_strong

    @staticmethod
    def _format_document_citation(hit: RetrievalHit) -> str:
        return f"{hit.filename}, page {hit.page_number}"

    @staticmethod
    def _format_retrieval_hits(hits: list[RetrievalHit]) -> str:
        if not hits:
            return "(none)"
        return "\n\n".join(
            (
                f"Document source {index + 1}\n"
                f"Filename: {hit.filename}\n"
                f"Page: {hit.page_number}\n"
                f"Score: {hit.score}\n"
                f"Lexical overlap: {hit.lexical_overlap}\n"
                f"Snippet: {hit.snippet}"
            )
            for index, hit in enumerate(hits)
        )

    @staticmethod
    def _format_search_hits(hits: list[SearchHit]) -> str:
        if not hits:
            return "(none)"
        return "\n\n".join(
            (
                f"Web source {index + 1}\n"
                f"Title: {hit.title}\n"
                f"URL: {hit.url}\n"
                f"Score: {hit.score}\n"
                f"Lexical overlap: {hit.lexical_overlap}\n"
                f"Expanded overlap: {hit.expanded_overlap}\n"
                f"Snippet: {hit.snippet}"
            )
            for index, hit in enumerate(hits)
        )

    @staticmethod
    def _format_trace_history(trace: list[SuperAgentTraceRecord]) -> str:
        if not trace:
            return "(none yet)"
        return "\n".join(
            (
                f"Iteration {item.iteration_number}: strategy={item.selected_strategy}, "
                f"query={item.executed_query!r}, "
                f"retrieval_hits={item.retrieval_outcome.hit_count if item.retrieval_outcome else 0}, "
                f"search_hits={item.search_outcome.hit_count if item.search_outcome else 0}, "
                f"missing={item.evaluator_result.missing_information}, "
                f"unsupported={item.evaluator_result.unsupported_claims}, "
                f"stop_reason={item.stop_reason}"
            )
            for item in trace
        )

    @staticmethod
    def _has_strong_sources(
        retrieval_summary: RetrievalOutcomeSummary | None,
        search_summary: SearchOutcomeSummary | None,
    ) -> bool:
        retrieval_strong = retrieval_summary is not None and retrieval_summary.hit_count > 0 and not retrieval_summary.weak
        search_strong = search_summary is not None and search_summary.hit_count > 0 and not search_summary.weak
        return retrieval_strong or search_strong

    @staticmethod
    def _is_strong_outcome(
        summary: RetrievalOutcomeSummary | SearchOutcomeSummary | None,
    ) -> bool:
        return summary is not None and summary.hit_count > 0 and not summary.weak

    @staticmethod
    def _merge_retrieval_hit_lists(
        existing_hits: list[RetrievalHit],
        new_hits: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        merged: list[RetrievalHit] = []
        seen: set[tuple[str, int, str]] = set()
        for hit in new_hits + existing_hits:
            key = (hit.document_id, hit.page_number, hit.snippet)
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
        return merged

    @staticmethod
    def _merge_search_hit_lists(
        existing_hits: list[SearchHit],
        new_hits: list[SearchHit],
    ) -> list[SearchHit]:
        merged: list[SearchHit] = []
        seen: set[tuple[str, str, str]] = set()
        for hit in new_hits + existing_hits:
            key = (hit.url, hit.title, hit.snippet)
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
        return merged

    @staticmethod
    def _strong_source_count(
        retrieval_summary: RetrievalOutcomeSummary | None,
        search_summary: SearchOutcomeSummary | None,
    ) -> int:
        strong_count = 0
        if retrieval_summary is not None and retrieval_summary.hit_count > 0 and not retrieval_summary.weak:
            strong_count += retrieval_summary.hit_count
        if search_summary is not None and search_summary.hit_count > 0 and not search_summary.weak:
            strong_count += search_summary.hit_count
        return strong_count

    @staticmethod
    def _current_strategy_is_weak(
        current_strategy: Strategy,
        retrieval_summary: RetrievalOutcomeSummary | None,
        search_summary: SearchOutcomeSummary | None,
    ) -> bool:
        if current_strategy == "retrieve":
            return retrieval_summary is None or retrieval_summary.weak
        if current_strategy == "search":
            return search_summary is None or search_summary.weak
        retrieval_weak = retrieval_summary is None or retrieval_summary.weak
        search_weak = search_summary is None or search_summary.weak
        return retrieval_weak and search_weak

    @staticmethod
    def _alternate_strategy(
        current_strategy: Strategy,
        feature_settings: AgentFeatureSettings,
    ) -> Strategy | None:
        if current_strategy == "retrieve" and feature_settings.web_search_enabled:
            return "search"
        if current_strategy == "search" and feature_settings.rag_enabled:
            return "retrieve"
        return None

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

    def _empty_source_message(self, user_text: str, strategy: Strategy) -> str:
        if self.looks_french(user_text):
            if strategy == "retrieve":
                return "Je n'ai pas trouve de passage pertinent dans les documents televerses pour cette tentative."
            if strategy == "search":
                return "Je n'ai pas trouve de resultat ESILV suffisamment pertinent pour cette tentative."
            return "Je n'ai pas trouve de sources documentaires ou web suffisamment pertinentes pour cette tentative."
        if strategy == "retrieve":
            return "I did not find a relevant passage in the uploaded documents for this attempt."
        if strategy == "search":
            return "I did not find an ESILV result that was relevant enough for this attempt."
        return "I did not find document or ESILV web sources that were relevant enough for this attempt."

    def _force_limitation_notice(
        self,
        user_text: str,
        content: str,
        evaluation: SuperAgentEvaluationRecord,
    ) -> str:
        missing = "; ".join(evaluation.missing_information[:3])
        if self.looks_french(user_text):
            limitation = "Reponse partielle uniquement"
            if missing:
                limitation += f" : il manque encore {missing}."
            else:
                limitation += " : certaines informations restent non verifiees."
            return f"{content}\n\n{limitation}"
        limitation = "Partial grounded answer only"
        if missing:
            limitation += f": still missing {missing}."
        else:
            limitation += ": some information remains unverified."
        return f"{content}\n\n{limitation}"

    def _no_grounded_answer_message(self, user_text: str, stop_reason: str) -> str:
        if self.looks_french(user_text):
            return (
                "Je n'ai pas pu produire de reponse completement verifiee a partir des PDF televerses et du site ESILV. "
                f"Arret de l'orchestration: {stop_reason}."
            )
        return (
            "I could not produce a fully grounded answer from the uploaded PDFs and the ESILV website. "
            f"Orchestration stopped because: {stop_reason}."
        )
