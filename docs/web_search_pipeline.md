# Tavily ESILV Web Search Pipeline

## Overview

The web-search agent now uses Tavily as the only live search backend for ESILV website questions.
The rest of the application still talks to the same `SiteSearchAgent` interface, so the orchestrator,
admin page, and chat flow do not need separate branching logic for Tavily.

## Request Flow

1. The orchestrator routes an ESILV website question to `SiteSearchAgent.search(...)`.
2. The search agent expands the user query with ESILV-specific terms.
   - Example: finance-major questions add terms such as `financial engineering` and `ingenierie financiere`.
3. The agent sends one Tavily Search request through the Tavily Python SDK (`TavilyClient.search(...)`).
4. The request is constrained with Tavily `include_domains` so results come only from ESILV domains.
5. Returned results are filtered again locally, then re-ranked with:
   - Tavily relevance score
   - lexical overlap with the original query
   - ESILV-specific boosts for important program names such as `Financial Engineering`
6. Final `SearchHit` objects are returned to the orchestrator, which uses them for answer generation and citations.

## ESILV-Only Guardrails

- Runtime domain filtering keeps only domains containing `esilv.fr`.
- Tavily `include_domains` is always sent with the ESILV domain list.
- Returned URLs are validated again before they become citations.

This means external search results are rejected even if Tavily were to return them unexpectedly.

## Cache Behavior

The old page-scrape cache has been replaced by a query-result cache under `data/site_cache/queries/`.

- Each normalized query is cached as a JSON file.
- Cached results are reused until `SITE_CACHE_TTL_HOURS` expires.
- The admin cache actions now work on Tavily search-result cache files.
- `Refresh cache` warms a small set of ESILV seed queries such as admissions, campuses, and majors.

## Configuration

Required environment variables:

```env
GEMINI_API_KEY=...
TAVILY_API_KEY=...
```

Relevant existing settings:

- `ESILV_ALLOWED_DOMAINS`
- `SITE_CACHE_TTL_HOURS`
- `MAX_SEARCH_HITS`

Only `esilv.fr` domains are used for Tavily search, even if broader domains are present elsewhere in the app config.

## Testing

Automated coverage includes:

- mocked Tavily response tests for ESILV-only filtering and finance-major ranking
- cache reuse tests when a cached Tavily result already exists
- an optional live Tavily test for the French query `Proposez vous une majeur de Finance?`

Expected live behavior for that query:

- top results should point to ESILV finance-major pages
- at least one result should clearly mention `Financial Engineering` or `Ingenierie financiere`
