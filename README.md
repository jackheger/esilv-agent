# ESILV Smart Assistant

Local Streamlit app for the ESILV Smart Assistant project. This version includes:

- A single Streamlit app with chat and admin views
- Persistent multi-conversation history stored as JSON
- A Gemini-backed orchestrator agent
- A dedicated Registration Agent for conversational lead capture and form persistence
- An optional Super Agent that can iterate across PDF RAG, ESILV web search, or both
- An ESILV-only website search agent powered by Tavily with a local query cache
- PDF ingestion with Docling parsing/chunking, Gemini embeddings, and a local on-disk vector store
- An admin page for document ingestion, indexed-document inventory, website-cache management, and submitted application forms
- Admin controls to choose the Gemini generation model used by all answering agents and the Gemini embedding model used for ingestion/retrieval

## Requirements

- Python 3.12 managed by `uv`
- A Gemini API key
- A Tavily API key for web search

## Setup

```bash
uv sync
copy .env.example .env
```

Set `GEMINI_API_KEY` and `TAVILY_API_KEY` in `.env`, then run:

```bash
uv run streamlit run app/main.py
```

## Project Structure

- `app/`: settings, models, runtime wiring, and the main Streamlit entrypoint
- `agents/`: orchestrator, registration, and ESILV site-search logic
- `docs/`: pipeline notes, including focused Super Agent and registration flow documentation
- `ui/`: chat and admin views
- `ingestion/`: PDF ingestion pipeline, upload registry, and local vector store
- `notebooks/`: reserved for exploration notebooks
- `data/`: runtime conversation, upload, and cache data
- `tests/`: unit tests

## Notes

- This is a local demo only. There is no authentication or deployment setup.
- Only PDF ingestion is enabled in the admin page.
- Ingested PDFs are parsed and chunked with `docling`, then embedded with Gemini and stored in a local JSON-backed vector index under `data/vector_store/`.
- Docling model artifacts are downloaded on first ingestion under `data/docling_artifacts/` so later ingests can reuse them locally.
- ESILV factual answers come from a Tavily-backed search flow restricted to `esilv.fr` domains, with local caching under `data/site_cache/queries/`.
- Conversational registration state is stored under `data/registrations/sessions/`, and completed application forms are stored under `data/registrations/submissions/`.
- When both RAG and web search are enabled and `super_agent_enabled` is off, every non-direct turn runs both retrieval and web search in the same pass.
- When `super_agent_enabled` is turned on in the admin page and both tools are enabled, each non-direct iteration also runs both retrieval and web search, with a visible trace in chat.
- The admin "Agent parameters" page also persists the generation model and embedding model; if no explicit choice was saved yet, the app falls back to `GEMINI_MODEL` and `GEMINI_EMBEDDING_MODEL` from `.env`.
- The admin page also includes an `Application Forms` section that reads submitted registrations directly from the JSON-backed registration store.
