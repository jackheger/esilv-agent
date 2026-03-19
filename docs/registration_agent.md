# Registration Agent

This document describes the dedicated ESILV `RegistrationAgent` that handles conversational lead capture and application-form persistence inside the existing Streamlit app.

## Where It Lives

- Agent logic: `agents/registration.py`
- File-backed persistence: `app/registration_store.py`
- Shared schemas: `app/models.py`
- Orchestrator integration: `agents/orchestrator.py`
- Admin rendering: `ui/admin_page.py`

The registration flow is not a separate service. It runs inside the same single-process Streamlit app and is invoked from `OrchestratorAgent.handle_turn(...)`.

## Triggering

The agent uses deterministic keyword rules to enter the flow instead of LLM-only routing.

- Immediate start:
  - explicit contact requests such as `contact me`, `be contacted`, `can someone contact me`
  - explicit joining/study-interest requests such as `I am interested in joining ESILV`, `I want to join ESILV`
  - explicit program/course-choice requests such as `help me choose a program`, `help me choose courses`, `recommend a program`
- Opt-in CTA:
  - admissions and program-discovery conversations still use the normal orchestrator answer path first
  - after the answer, the assistant can append a registration CTA and mark the message with `pending_action="registration"`
  - a follow-up `yes` / `oui` starts the form

Document-grounded requests like uploaded-PDF questions do not get the registration CTA.

## State Management

The registration flow keeps a deterministic field-completion order, but the turn-by-turn conversation is LLM-guided when Gemini is configured.

The backend still tracks completion in this fixed order:

1. `full_name`
2. `email`
3. `location`
4. `program_interest`
5. `discovery_source`
6. `degree_level`
7. `desired_start_date`

State is persisted per conversation, so the flow survives Streamlit reruns:

- active session file: `data/registrations/sessions/<conversation_id>.json`
- completion deletes the active session file and creates a submission file

Each session stores:

- `conversation_id`
- `current_field`
- `language`
- partial `answers`
- timestamps

Collection behavior is split between LLM guidance and deterministic state updates:

- Gemini extracts explicitly stated fields from the latest user reply when available
- the next missing field is still computed deterministically in Python
- `email` must still match a simple email pattern
- `full_name` still has to be explicit enough to use as a name
- other fields, including `degree_level`, accept non-empty free text and preserve uncertainty such as `I don't know yet`
- when Gemini is unavailable or fails, the flow falls back to deterministic prompt strings without losing persistence

## Recommendation Behavior

After the last answer, the agent stores the form and returns a thank-you message plus one recommended ESILV program.

Recommendation order respects the current admin tool toggles:

1. `super` if `super_agent_enabled` is on and at least one retrieval tool is enabled
2. `search` if ESILV web search is enabled and strong enough
3. `retrieve` if PDF RAG is enabled and strong enough
4. `rules` fallback when enabled tools are unavailable, weak, or Gemini is not configured

The rules fallback uses a deterministic ESILV program table seeded from the program set already present in the local site cache.

Each submission stores:

- collected answers
- recommended `program_name`
- recommendation `message`
- recommendation `source_mode`
- any saved citations

## Data Storage

Completed submissions are written as pretty-printed JSON under:

- `data/registrations/submissions/<submission_id>.json`

This makes forms easy to inspect and debug without a database or API server.

## Admin Interface

The existing admin radio switcher now includes a third section:

- `Application Forms`

That section reads `RegistrationStore.list_submissions()` directly and renders submissions newest-first as expanders. Each expander shows:

- name
- email
- location
- program interest
- discovery source
- degree level
- desired start date
- saved recommendation, when present

No separate backend endpoint is used; the Streamlit admin page reads the same JSON-backed store that the registration agent writes.
