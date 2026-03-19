from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CitationRecord(BaseModel):
    kind: Literal["web", "document"] = "web"
    title: str
    url: str | None = None
    page_number: int | None = None


class RetrievalOutcomeSummary(BaseModel):
    hit_count: int = 0
    weak: bool = True
    top_score: float | None = None
    top_lexical_overlap: float | None = None
    top_citations: list[str] = Field(default_factory=list)


class SearchOutcomeSummary(BaseModel):
    hit_count: int = 0
    weak: bool = True
    top_score: float | None = None
    top_lexical_overlap: float | None = None
    top_expanded_overlap: float | None = None
    top_citations: list[str] = Field(default_factory=list)


class SuperAgentEvaluationRecord(BaseModel):
    is_sufficient: bool = False
    is_grounded: bool = False
    missing_information: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    next_strategy: Literal["retrieve", "search", "hybrid"] | None = None
    rewritten_query: str | None = None
    stop_reason: str | None = None


class SuperAgentTraceRecord(BaseModel):
    iteration_number: int
    selected_strategy: Literal["retrieve", "search", "hybrid"]
    executed_query: str
    retrieval_outcome: RetrievalOutcomeSummary | None = None
    search_outcome: SearchOutcomeSummary | None = None
    draft_answer: str
    evaluator_result: SuperAgentEvaluationRecord
    rewritten_query: str | None = None
    stop_reason: str | None = None


class MessageRecord(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(default_factory=utc_now_iso)
    citations: list[CitationRecord] = Field(default_factory=list)
    pending_action: Literal["search", "retrieve", "registration"] | None = None
    pending_query: str | None = None
    orchestration_mode: Literal["standard", "super"] = "standard"
    super_agent_stop_reason: str | None = None
    super_agent_trace: list[SuperAgentTraceRecord] = Field(default_factory=list)


class ConversationRecord(BaseModel):
    id: str
    title: str = "New conversation"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    messages: list[MessageRecord] = Field(default_factory=list)


class UploadedDocumentRecord(BaseModel):
    id: str
    filename: str
    stored_path: str
    mime_type: str
    size: int
    uploaded_at: str = Field(default_factory=utc_now_iso)
    status: Literal["uploaded", "indexed", "failed"] = "uploaded"
    page_count: int = 0
    chunk_count: int = 0
    indexed_at: str | None = None
    error_message: str | None = None


class ParsedPdfChunk(BaseModel):
    page_number: int
    chunk_index: int
    text: str


class VectorChunkRecord(BaseModel):
    id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    embedding: list[float]


class IndexedDocumentRecord(BaseModel):
    document_id: str
    filename: str
    indexed_at: str = Field(default_factory=utc_now_iso)
    page_count: int
    chunk_count: int
    chunks: list[VectorChunkRecord] = Field(default_factory=list)


class VectorStoreStats(BaseModel):
    document_count: int = 0
    chunk_count: int = 0


class RetrievalHit(BaseModel):
    document_id: str
    filename: str
    page_number: int
    snippet: str
    score: float
    cosine_score: float = 0.0
    lexical_overlap: float = 0.0


class SearchHit(BaseModel):
    url: str
    title: str
    snippet: str
    score: float
    lexical_overlap: float = 0.0
    expanded_overlap: float = 0.0
    fetched_at: str


class CachedChunkRecord(BaseModel):
    heading: str
    text: str


class CachedPageRecord(BaseModel):
    url: str
    title: str
    fetched_at: str = Field(default_factory=utc_now_iso)
    chunks: list[CachedChunkRecord] = Field(default_factory=list)


class CacheStats(BaseModel):
    page_count: int = 0
    stale_count: int = 0
    last_refresh: str | None = None
    urls: list[str] = Field(default_factory=list)


class SearchCacheRecord(BaseModel):
    query: str
    normalized_query: str
    expanded_query: str
    fetched_at: str = Field(default_factory=utc_now_iso)
    results: list[SearchHit] = Field(default_factory=list)


RegistrationFieldName = Literal[
    "full_name",
    "email",
    "location",
    "program_interest",
    "discovery_source",
    "degree_level",
    "desired_start_date",
]


class RegistrationAnswersRecord(BaseModel):
    full_name: str | None = None
    email: str | None = None
    location: str | None = None
    program_interest: str | None = None
    discovery_source: str | None = None
    degree_level: str | None = None
    desired_start_date: str | None = None


class RegistrationRecommendationRecord(BaseModel):
    program_name: str
    message: str
    source_mode: Literal["super", "search", "retrieve", "rules"]
    citations: list[CitationRecord] = Field(default_factory=list)


class RegistrationSessionRecord(BaseModel):
    conversation_id: str
    current_field: RegistrationFieldName = "full_name"
    language: Literal["en", "fr"] = "en"
    answers: RegistrationAnswersRecord = Field(default_factory=RegistrationAnswersRecord)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class RegistrationSubmissionRecord(BaseModel):
    id: str
    conversation_id: str
    answers: RegistrationAnswersRecord
    recommendation: RegistrationRecommendationRecord | None = None
    submitted_at: str = Field(default_factory=utc_now_iso)


SUPPORTED_GENERATION_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
)
SUPPORTED_EMBEDDING_MODELS = (
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
)
GenerationModelName = Literal[
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]
EmbeddingModelName = Literal[
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
]


class AgentFeatureSettings(BaseModel):
    rag_enabled: bool = True
    web_search_enabled: bool = True
    super_agent_enabled: bool = False
    generation_model: GenerationModelName | None = None
    embedding_model: EmbeddingModelName | None = None
