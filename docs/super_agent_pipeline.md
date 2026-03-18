# Super Agent Iterative Orchestration

## Architecture

The Super Agent is implemented in `agents/super_agent.py` and is invoked from `OrchestratorAgent.handle_turn(...)`.
It does not replace the existing router, `RetrievalAgent`, or `SiteSearchAgent`.

Runtime behavior:

1. `OrchestratorAgent` still persists the user message, loads `AgentFeatureSettings`, and runs the existing route selection.
2. If `super_agent_enabled` is `false`, the current single-pass `direct` / `retrieve` / `search` flow runs unchanged.
3. If `super_agent_enabled` is `true` and the selected route is `retrieve` or `search`, the orchestrator delegates to `SuperAgent.run(...)`.

The Super Agent reuses:

- `RetrievalAgent.search(...)`
- `RetrievalAgent.is_weak(...)`
- `SiteSearchAgent.search(...)`
- `OrchestratorAgent._search_is_weak(...)`
- existing retrieval-only and search-only answer synthesis prompts
- one additional hybrid synthesis prompt in `OrchestratorAgent._hybrid_answer(...)`

## Execution Flow

Each Super Agent run uses the original user question as the evaluation target for the full loop.

1. Pick an initial strategy:
   - `hybrid` if both tools are enabled and the question clearly mixes document and ESILV web intent.
   - otherwise reuse the orchestrator route (`retrieve` or `search`).
2. Execute the chosen strategy with the current query.
3. Build deterministic retrieval/search summaries:
   - hit count
   - weak/non-weak status
   - top score and overlap values
   - top citations
4. Generate a grounded draft answer from the retrieved evidence.
   - From iteration 2 onward, the draft is generated from the accumulated retrieval and web results gathered across previous iterations, not only from the latest query.
5. Run a Gemini structured evaluation against:
   - the original user question
   - the current draft answer
   - the exact sources used
   - the iteration history so far
6. Decide whether to stop or continue.
7. If continuing, choose a rewritten query and next strategy.
8. Repeat up to 3 iterations.

The final assistant message stores:

- `orchestration_mode="super"`
- `super_agent_stop_reason`
- `super_agent_trace[]`

The chat UI renders this trace inside a collapsed `Super Agent trace` expander.

## Iteration Logic

Supported strategies:

- `retrieve`
- `search`
- `hybrid`

The loop is intentionally small and deterministic:

- hard cap of 3 iterations
- normalized `(strategy, query)` deduplication
- explicit pivot to the alternate tool when the current tool is weak and the alternate tool has not been tried yet
- final best-candidate selection even if the loop does not fully satisfy the question

Best-candidate ranking uses:

1. evaluator sufficiency
2. groundedness plus strong sources
3. strong source count
4. fewer missing-information items
5. later iteration number

## Stopping Criteria

The loop stops when one of the following happens:

- evaluator says the answer is sufficient, grounded, and there are strong sources
- 3 iterations have been used
- the next `(strategy, query)` would repeat a prior attempt
- there is no novel next step left

Guardrails override evaluator optimism:

- never accept success if all used tools were weak or empty
- never accept success if the evaluator reports unsupported claims

## Query Rewriting Strategy

Query rewriting is evaluator-assisted but execution remains deterministic.

- Gemini proposes `rewritten_query` and `next_strategy` in structured JSON.
- The controller normalizes that query, rejects duplicates, and checks tool availability.
- If the current tool is weak and an alternate enabled tool has not been tried yet, the controller pivots to the alternate tool before retrying the same tool family.

The rewritten query is therefore a proposal, not an unconditional command.

## Fallback Behavior

- If the loop finds a sufficient grounded answer, that answer is returned with citations.
- If the loop does not fully answer the question but found a grounded partial answer, the best grounded candidate is returned with an explicit limitation notice.
- If no grounded candidate exists, the assistant returns a failure/limitation message instead of claiming success.

This keeps the behavior inspectable and reduces false-positive “success” states.
