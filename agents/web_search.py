from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from tavily import TavilyClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.models import CacheStats, SearchCacheRecord, SearchHit, utc_now_iso

USER_AGENT = "ESILVSmartAssistant/0.2 (+local-demo)"
DEFAULT_SEARCH_DEPTH = "advanced"
DEFAULT_COUNTRY = "france"
DEFAULT_WARMUP_QUERIES = (
    "ESILV admissions concours avenir",
    "ESILV campus Paris La Defense",
    "ESILV cycle ingenieur majeures",
    "ESILV ingenierie financiere",
    "ESILV bachelor informatique",
    "ESILV alternance",
)
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "avec",
    "comment",
    "dans",
    "de",
    "des",
    "do",
    "does",
    "du",
    "en",
    "esilv",
    "est",
    "et",
    "for",
    "how",
    "la",
    "le",
    "les",
    "of",
    "on",
    "ou",
    "pour",
    "proposez",
    "proposez-vous",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "site",
    "sur",
    "the",
    "une",
    "vous",
    "website",
}
QUERY_EXPANSIONS = {
    "finance": ("financial engineering", "ingenierie financiere"),
    "sql": ("database", "databases", "bases de donnees", "relational database", "data management"),
    "database": ("databases", "bases de donnees", "sql", "relational database", "data management"),
    "databases": ("database", "bases de donnees", "sql", "relational database", "data management"),
    "majeur": ("majeure", "majeures", "major", "majors", "cycle ingenieur"),
    "majeure": ("majeur", "majeures", "major", "majors", "cycle ingenieur"),
    "major": ("majeure", "majeures", "majors", "cycle ingenieur"),
    "majors": ("major", "majeure", "majeures", "cycle ingenieur"),
    "campus": ("paris la defense", "la defense campus"),
    "admission": ("admissions", "concours avenir"),
    "admissions": ("admission", "concours avenir"),
    "ingenieur": ("cycle ingenieur", "engineering"),
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def tokenize(text: str) -> list[str]:
    normalized = strip_accents(text).lower()
    raw_tokens = re.findall(r"[0-9a-z]{2,}", normalized)
    return [token for token in raw_tokens if token not in STOPWORDS]


def snippet_for(text: str, query: str, limit: int = 320) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= limit:
        return cleaned

    lower = strip_accents(cleaned).lower()
    for token in tokenize(query):
        position = lower.find(token)
        if position >= 0:
            start = max(0, position - 90)
            end = min(len(cleaned), position + limit - 90)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(cleaned) else ""
            return f"{prefix}{cleaned[start:end].strip()}{suffix}"

    return f"{cleaned[:limit].rstrip()}..."


class SiteSearchAgent:
    def __init__(
        self,
        cache_dir: Path,
        allowed_domains: Iterable[str],
        api_key: str | None = None,
        ttl_hours: int = 24,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.allowed_domains = self._normalize_allowed_domains(allowed_domains)
        self.api_key = api_key
        self.ttl_hours = ttl_hours
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.client = TavilyClient(api_key=api_key, session=self.session) if api_key else None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.query_cache_dir = self.cache_dir / "queries"
        self.query_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return self._search(query=query, top_k=top_k, force_refresh=False)

    def refresh_cache(self, limit: int = 25) -> int:
        if not self.configured:
            return 0

        refreshed = 0
        for query in DEFAULT_WARMUP_QUERIES[:limit]:
            hits = self._search(query=query, top_k=5, force_refresh=True)
            if hits:
                refreshed += 1
        return refreshed

    def clear_cache(self) -> None:
        for path in self.query_cache_dir.glob("query_*.json"):
            path.unlink(missing_ok=True)
        for path in self.cache_dir.glob("page_*.json"):
            path.unlink(missing_ok=True)
        (self.cache_dir / "sitemap_urls.json").unlink(missing_ok=True)

    def cache_stats(self) -> CacheStats:
        records = self._load_cache_records()
        records.sort(key=lambda item: item.fetched_at, reverse=True)
        stale_count = sum(1 for record in records if self._is_stale(record.fetched_at))
        unique_urls: list[str] = []
        seen: set[str] = set()
        for record in records:
            for hit in record.results:
                if hit.url in seen:
                    continue
                seen.add(hit.url)
                unique_urls.append(hit.url)

        return CacheStats(
            page_count=len(unique_urls),
            stale_count=stale_count,
            last_refresh=records[0].fetched_at if records else None,
            urls=unique_urls[:10],
        )

    def _search(self, query: str, top_k: int, force_refresh: bool) -> list[SearchHit]:
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        cached = self._load_cache_record(normalized_query)
        if cached is not None and not force_refresh and not self._is_stale(cached.fetched_at):
            return cached.results[:top_k]

        if not self.configured:
            return cached.results[:top_k] if cached is not None else []

        expanded_query = self._expand_query(query)
        payload = {
            "query": expanded_query,
            "search_depth": DEFAULT_SEARCH_DEPTH,
            "chunks_per_source": 3,
            "topic": "general",
            "country": DEFAULT_COUNTRY,
            "max_results": min(max(top_k * 3, 8), 12),
            "include_answer": False,
            "include_raw_content": False,
            "include_domains": list(self.allowed_domains),
        }
        response = self._search_tavily(payload)
        hits = self._build_hits(query=query, fetched_at=utc_now_iso(), payload=response)
        record = SearchCacheRecord(
            query=query,
            normalized_query=normalized_query,
            expanded_query=expanded_query,
            results=hits,
        )
        self._write_cache_record(record)
        return hits[:top_k]

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _search_tavily(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            return {}
        data = self.client.search(timeout=20, **payload)
        if not isinstance(data, dict):
            return {}
        return data

    def _build_hits(self, query: str, fetched_at: str, payload: dict[str, Any]) -> list[SearchHit]:
        query_tokens = tokenize(query)
        expanded_tokens = tokenize(self._expand_query(query))
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            return []

        best_by_url: dict[str, SearchHit] = {}
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            url = str(raw_result.get("url", "")).strip()
            if not self._is_allowed_url(url):
                continue

            title = normalize_whitespace(str(raw_result.get("title", "")).strip()) or url
            content = normalize_whitespace(
                str(raw_result.get("content") or raw_result.get("raw_content") or "").strip()
            )
            snippet = snippet_for(content or title, query)
            haystack = " ".join(part for part in (title, content, url) if part)
            score = self._rank_result(
                url=url,
                title=title,
                content=content,
                base_score=float(raw_result.get("score") or 0.0),
                query_tokens=query_tokens,
                expanded_tokens=expanded_tokens,
            )
            hit = SearchHit(
                url=url,
                title=title,
                snippet=snippet,
                score=score,
                lexical_overlap=round(self._lexical_overlap_ratio(query_tokens, haystack), 4),
                expanded_overlap=round(self._lexical_overlap_ratio(expanded_tokens, haystack), 4),
                fetched_at=fetched_at,
            )
            existing = best_by_url.get(url)
            if existing is None or hit.score > existing.score:
                best_by_url[url] = hit

        hits = list(best_by_url.values())
        hits.sort(key=lambda hit: (hit.score, hit.fetched_at), reverse=True)
        return hits

    def _rank_result(
        self,
        url: str,
        title: str,
        content: str,
        base_score: float,
        query_tokens: list[str],
        expanded_tokens: list[str],
    ) -> float:
        title_text = strip_accents(title).lower()
        content_text = strip_accents(content).lower()
        url_text = strip_accents(url).lower()

        lexical_score = 0.0
        lexical_score += 2.5 * sum(title_text.count(token) for token in query_tokens)
        lexical_score += 1.5 * sum(content_text.count(token) for token in query_tokens)
        lexical_score += 1.0 * sum(url_text.count(token) for token in query_tokens)
        lexical_score += 1.2 * sum(title_text.count(token) for token in expanded_tokens)
        lexical_score += 0.8 * sum(content_text.count(token) for token in expanded_tokens)
        lexical_score += 0.8 * sum(url_text.count(token) for token in expanded_tokens)

        finance_major_bonus = 0.0
        if "finance" in query_tokens and any(
            token in query_tokens for token in ("majeur", "majeure", "major", "majors")
        ):
            if "financial engineering" in title_text or "financial engineering" in content_text:
                finance_major_bonus += 4.0
            if "ingenierie financiere" in title_text or "ingenierie financiere" in content_text:
                finance_major_bonus += 4.0
            if "ingenierie-financiere" in url_text or "financial-engineering" in url_text:
                finance_major_bonus += 4.0

        return round((base_score * 10.0) + lexical_score + finance_major_bonus, 4)

    def _expand_query(self, query: str) -> str:
        normalized = strip_accents(query).lower()
        expansions: list[str] = ["ESILV", normalize_whitespace(query)]

        for token in tokenize(normalized):
            for expanded in QUERY_EXPANSIONS.get(token, ()):
                expansions.append(expanded)

        if "finance" in normalized:
            expansions.extend(["finance major", "major finance", "financial engineering"])

        deduped: list[str] = []
        seen: set[str] = set()
        for item in expansions:
            compact = normalize_whitespace(item)
            key = strip_accents(compact).lower()
            if not compact or key in seen:
                continue
            seen.add(key)
            deduped.append(compact)
        return " ".join(deduped)

    @staticmethod
    def _lexical_overlap_ratio(query_tokens: list[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = set(tokenize(text))
        if not text_tokens:
            return 0.0
        overlap = len(set(query_tokens) & text_tokens)
        return overlap / max(1, len(set(query_tokens)))

    def _load_cache_records(self) -> list[SearchCacheRecord]:
        records: list[SearchCacheRecord] = []
        for path in self.query_cache_dir.glob("query_*.json"):
            try:
                records.append(SearchCacheRecord.model_validate_json(path.read_text("utf-8")))
            except (OSError, ValueError):
                continue
        return records

    def _load_cache_record(self, normalized_query: str) -> SearchCacheRecord | None:
        path = self._cache_path(normalized_query)
        if not path.exists():
            return None
        try:
            return SearchCacheRecord.model_validate_json(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None

    def _write_cache_record(self, record: SearchCacheRecord) -> None:
        self._cache_path(record.normalized_query).write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _cache_path(self, normalized_query: str) -> Path:
        digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
        return self.query_cache_dir / f"query_{digest}.json"

    @staticmethod
    def _normalize_query(query: str) -> str:
        return normalize_whitespace(strip_accents(query).lower())

    @staticmethod
    def _normalize_allowed_domains(allowed_domains: Iterable[str]) -> tuple[str, ...]:
        filtered = [
            domain.strip().lower()
            for domain in allowed_domains
            if domain and "esilv.fr" in domain.strip().lower()
        ]
        if not filtered:
            filtered = ["esilv.fr", "www.esilv.fr"]
        ordered: list[str] = []
        seen: set[str] = set()
        for domain in filtered:
            if domain in seen:
                continue
            seen.add(domain)
            ordered.append(domain)
        return tuple(ordered)

    def _is_allowed_url(self, url: str) -> bool:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return bool(
            hostname
            and any(
                hostname == allowed or hostname.endswith(f".{allowed}") for allowed in self.allowed_domains
            )
        )

    def _is_stale(self, fetched_at: str) -> bool:
        try:
            fetched = datetime.fromisoformat(fetched_at)
        except ValueError:
            return True
        return datetime.now(timezone.utc) - fetched > timedelta(hours=self.ttl_hours)
