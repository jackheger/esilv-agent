from __future__ import annotations

import math
import re
from typing import Iterable

from app.models import IndexedDocumentRecord, RetrievalHit, VectorChunkRecord
from ingestion.pdf_ingestion import EmbeddingClientProtocol
from ingestion.vector_store import LocalVectorStore

STOPWORDS = {
    "a",
    "about",
    "according",
    "an",
    "and",
    "are",
    "as",
    "au",
    "aux",
    "can",
    "ce",
    "ces",
    "comment",
    "dans",
    "de",
    "des",
    "document",
    "do",
    "does",
    "du",
    "en",
    "et",
    "for",
    "from",
    "how",
    "i",
    "in",
    "internal",
    "is",
    "la",
    "le",
    "les",
    "me",
    "mon",
    "my",
    "of",
    "on",
    "or",
    "pdf",
    "please",
    "pour",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "report",
    "show",
    "sur",
    "the",
    "this",
    "to",
    "uploaded",
    "what",
    "where",
    "which",
    "with",
}
QUERY_EXPANSIONS = {
    "admission": ("admissions", "application", "concours avenir"),
    "admissions": ("admission", "application", "concours avenir"),
    "course": ("courses", "module", "modules", "class", "classes"),
    "courses": ("course", "module", "modules", "class", "classes"),
    "cours": ("module", "modules", "course", "courses"),
    "sql": ("database", "databases", "bases de donnees", "relational database", "data management"),
    "database": ("databases", "bases de donnees", "sql", "relational database", "data management"),
    "databases": ("database", "bases de donnees", "sql", "relational database", "data management"),
    "cloud": ("cloud computing", "infrastructure", "platform engineering"),
    "cybersecurity": ("cybersecurite", "security", "securite informatique"),
    "cybersecurite": ("cybersecurity", "security", "securite informatique"),
    "ai": ("artificial intelligence", "intelligence artificielle", "machine learning", "data"),
    "ia": ("intelligence artificielle", "artificial intelligence", "machine learning", "data"),
    "finance": ("financial engineering", "ingenierie financiere"),
}


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[0-9A-Za-zÀ-ÿ]{2,}", text.lower())
    return [token for token in raw_tokens if token not in STOPWORDS]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def snippet_for(text: str, query: str, limit: int = 280) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= limit:
        return cleaned

    lower = cleaned.lower()
    for token in tokenize(query):
        position = lower.find(token)
        if position >= 0:
            start = max(0, position - 80)
            end = min(len(cleaned), position + limit - 80)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(cleaned) else ""
            return f"{prefix}{cleaned[start:end].strip()}{suffix}"

    return f"{cleaned[:limit].rstrip()}..."


class RetrievalAgent:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        embedding_client: EmbeddingClientProtocol,
        weak_score_threshold: float = 0.58,
        weak_overlap_threshold: float = 0.15,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.weak_score_threshold = weak_score_threshold
        self.weak_overlap_threshold = weak_overlap_threshold

    def expand_query(self, query: str) -> str:
        normalized_query = normalize_whitespace(query)
        if not normalized_query:
            return ""

        expansions: list[str] = [normalized_query]
        for token in tokenize(normalized_query):
            for expanded in QUERY_EXPANSIONS.get(token, ()):
                expansions.append(expanded)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in expansions:
            compact = normalize_whitespace(item)
            key = compact.lower()
            if not compact or key in seen:
                continue
            seen.add(key)
            deduped.append(compact)
        return " ".join(deduped)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        chunks = list(self._all_chunks())
        if not chunks or not self.embedding_client.configured:
            return []

        expanded_query = self.expand_query(query)
        query_embedding = self.embedding_client.embed_texts(
            [expanded_query or query],
            task_type="RETRIEVAL_QUERY",
        )[0]
        query_tokens = set(tokenize(expanded_query or query))

        hits: list[RetrievalHit] = []
        for document, chunk in chunks:
            cosine_score = self._cosine_similarity(query_embedding, chunk.embedding)
            overlap_ratio = self._lexical_overlap_ratio(query_tokens, chunk.text)
            final_score = cosine_score + (0.08 * overlap_ratio)

            if final_score <= 0:
                continue

            hits.append(
                RetrievalHit(
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=chunk.page_number,
                    snippet=snippet_for(chunk.text, query),
                    score=round(final_score, 4),
                    cosine_score=round(cosine_score, 4),
                    lexical_overlap=round(overlap_ratio, 4),
                )
            )

        hits.sort(key=lambda hit: (hit.score, hit.lexical_overlap, hit.cosine_score), reverse=True)
        return hits[:top_k]

    def is_weak(self, hits: list[RetrievalHit]) -> bool:
        if not hits:
            return True
        top_hit = hits[0]
        return (
            top_hit.score < self.weak_score_threshold
            and top_hit.lexical_overlap < self.weak_overlap_threshold
        )

    def _all_chunks(self) -> Iterable[tuple[IndexedDocumentRecord, VectorChunkRecord]]:
        for document in self.vector_store.list_documents():
            for chunk in document.chunks:
                yield document, chunk

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _lexical_overlap_ratio(query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = set(tokenize(text))
        if not text_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens)
        return overlap / max(1, len(query_tokens))
