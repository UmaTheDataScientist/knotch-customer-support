# Multi-Tool Customer Support Agent

A multi-turn support agent built as an explicit LangGraph state machine, with a
separate Compliance Agent guardrail, self-verification, idempotent embedding
ingestion, and a homegrown structured tracer for observability.

**Start here as a reviewer:**
1. `app/agents/graph.py` — the Plan → Act → Observe → Verify loop as a LangGraph.
2. `app/agents/compliance.py` — the guardrail agent and the "runs first" design decision.
3. `tests/integration/test_conversations.py` — replicates the four example
   interactions from the assignment doc, and one extra (security escalation).
4. `eval/run_eval.py` — automated eval harness (bonus #5), 100% pass rate offline.

## Quickstart (no API key required)

The app defaults to `LLM_PROVIDER=fake`, a deterministic, network-free client
(rule-based planning/compliance/verification + a hash-based embedder). This is
enough to run the full API, the test suite, and the eval harness with zero
setup — useful for reviewing the *architecture* without burning the rate-limited
OpenAI key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the test suite (29 tests, all offline)
pytest -q

# run the eval harness (13 scripted scenarios, all offline)
python eval/run_eval.py

# start the API
uvicorn app.main:app --reload --port 8000
```

Example curl (matches "Example 1" in the assignment doc):

```bash
curl -s -X POST http://127.0.0.1:8000/conversations/abc-123/messages \
  -H 'Content-Type: application/json' \
  -d '{"message": "How do I restore my account to its default settings?"}' | python -m json.tool

curl -s http://127.0.0.1:8000/conversations/abc-123/trace | python -m json.tool
```

## Running with a real LLM

```bash
cp .env.example .env
# edit .env: LLM_PROVIDER=openai, OPENAI_API_KEY=..., model names as provided
uvicorn app.main:app --reload --port 8000
```

Swapping to Anthropic instead of OpenAI is a one-line env change
(`LLM_PROVIDER=anthropic`) — no code in `app/agents/*` or `app/tools/*` changes,
by design (see **Abstractions** below). Anthropic doesn't offer an embeddings
API, so `AnthropicLLMClient.embed()` transparently falls back to OpenAI
embeddings if `OPENAI_API_KEY` is also set.

## Architecture

```
                     ┌────────────────────┐
  POST /messages ──▶ │  Compliance Agent   │──UNSAFE──▶ refusal (source=compliance)
                     │ (own LLM call, own  │
                     │  system prompt)     │
                     └─────────┬──────────┘
                              SAFE
                               │
                     ┌─────────▼──────────┐
                     │   plan_step        │◀──────────────┐
                     │ classify intent,   │                │ replan
                     │ choose ONE tool    │                │ (bounded: max 2
                     └─────────┬──────────┘                │  verification
                               │                            │  retries)
                     ┌─────────▼──────────┐                │
                     │   act_step         │                │
                     │ dispatch to tool   │                │
                     └─────────┬──────────┘                │
                               │                            │
                     ┌─────────▼──────────┐                │
                     │  observe_step      │                │
                     │ synthesize draft   │                │
                     │ response           │                │
                     └─────────┬──────────┘                │
                               │                            │
                     ┌─────────▼──────────┐   fail  ────────┘
                     │  verify_step       │──────────┐
                     │ addresses_question?│          │
                     │ grounded? leaks?   │        pass or
                     └─────────┬──────────┘        retries exhausted
                               │
                     ┌─────────▼──────────┐
                     │  finalize_step     │──▶ ChatMessageOut + trace
                     └────────────────────┘
```

State (`AgentState`, a `TypedDict`) flows through the graph and is discarded
per-turn; durable conversation history lives separately in `ConversationState`
(`app/core/state.py`), which the graph reads from and writes back to via the
orchestrator (`app/agents/orchestrator.py`).

### Why the Compliance Agent runs *before* the main agent, not parallel/after
Documented in `app/agents/compliance.py`, in short:
- **Cost/latency**: a blocked message never reaches FAQ search, embeddings, or
  the planning LLM call — no wasted tool calls.
- **Safety**: nothing the main agent produces (which could itself be steered by
  an injection) is ever shown to the user before a policy check happens.
- **Simplicity**: "runs first, can veto" is a far easier invariant to test and
  audit than "runs concurrently and wins races."

A regex fast-path catches the most blatant injection phrases before spending an
LLM call, but the authoritative decision is always the model's — the regex path
logs identical `category`/`reasoning` fields so both paths are indistinguishable
in the audit trail.

### Plan → Act → Observe → Reflect
Implemented as literal graph nodes rather than a single function with a
while-loop, specifically so the two independent bounds are visible as edges:
- `max_agent_iterations` (default 6): total plan/act/observe cycles. If
  exhausted, the planner is forced to `escalate_to_human` rather than looping
  forever or returning nothing.
- `max_verification_retries` (default 2): how many times a *verified-failing*
  draft answer triggers a replan before we give up and return the best attempt
  with `verified: false` in the response.

### Multi-turn state & context window management
`ConversationState` keeps the last `max_turns_in_context` raw turns verbatim.
Anything older is collapsed into a single running summary string via one short
LLM call — regenerated only when a turn actually ages out, not on every
message. This avoids both unbounded context growth and per-turn summarization
overhead.

### Self-verification
A separate LLM call (`VERIFY_SYSTEM_PROMPT`) checks three things: does the
answer address the question, is it grounded in retrieved KB content (vs.
invented), and does it leak tool names/system-prompt details. Direct-response
tools (`ask_user_clarification`, `refuse`, `escalate_to_human`) skip
verification since their output is structurally fixed, not generated —
nothing to hallucinate-check.

### Tools beyond the assignment's minimum list
The assignment explicitly allows adding tools ("welcome to add more tools if
it makes the agent stronger"). Two were added, both mocked the same way
`escalate_to_human` already is (deterministic stub, not a real backend call):

- **`check_system_status(component=None)`**: for "is the site down" /
  "why is it slow" questions. The KB's static FAQ answer for this
  ("check the status page...") is a workaround for *not having* this
  capability — a real support bot should be able to check live status
  itself rather than tell the user to go check a page manually. Reports
  `operational` for everything except a `payments` component, which is
  deliberately deterministic (not random) so it's testable.
- **`lookup_account_status(account_id)`**: for "is my account locked" style
  questions where the user has given an identifier. Deterministic
  hash-based stub (same `account_id` always returns the same status),
  standing in for a real accounts-service call.

Both are wired all the way through: the planner prompt lists them as real
options, `observe_step` turns their output into a user-facing response, and
they're covered by unit tests (`tests/unit/test_tools.py`), integration tests
that go through the full planner (`tests/integration/test_conversations.py`),
and two scenarios in the eval harness.

### Knowledge base cleaning decisions
Documented in `app/retrieval/knowledge_base.py`:
- `"x"` is excluded from the *searchable* index — its "answer" is itself an
  instruction to ask for clarification, not a fact. Indexing it would make it
  falsely match almost any short query. The `ask_user_clarification` tool
  handles this dynamically instead.
- `"help!!! my account is locked"` is a real case, just noisily formatted —
  kept, with a normalized version used only for the embedding text (so
  formatting noise doesn't hurt retrieval), while the original is still shown
  to users/logs.
- Everything else indexed as-is.

### Idempotent embedding ingestion (bonus #9)
`EmbeddingIndex.build()` hashes each item's embedding text (`sha256`) and
caches `{hash, vector}` in `data/embedding_cache.json`. Re-running ingestion
only calls the embedder for rows whose hash changed. Verified in
`tests/unit/test_retrieval.py::test_embedding_cache_is_idempotent`.

### Abstractions
- **LLM provider**: `app/core/llm_client.py` defines an `LLMClient` ABC with
  `OpenAILLMClient`, `AnthropicLLMClient`, and `FakeLLMClient` (offline dev/CI).
  Agent/tool code only ever calls `LLMClient.chat`/`.embed`.
- **Prompts**: externalized and tagged with `PROMPT_VERSION` in
  `app/agents/prompts.py`, not inlined in graph nodes.
- **Tools**: `Tool` dataclasses in `app/tools/definitions.py` pair a Pydantic
  args schema with a plain callable and expose `.tool_spec()` (OpenAI/Anthropic
  function-calling JSON schema) — reusable outside LangGraph if the framework
  changes.

### Observability
A homegrown JSON tracer (`app/observability/tracer.py`) records every step
(`compliance_check`, `plan`, `tool_call`, `observe`, `verify`, `replan`,
`final_response`) with latency, token counts, and estimated cost. Retrieved
via `GET /conversations/{id}/trace`. Chosen over LangSmith/OTel purely to keep
the submission dependency-free and runnable offline; swapping in an OTel
exporter later means implementing the same `Tracer.step()` interface, nothing
upstream changes.

## Project layout

```
app/
  main.py                 FastAPI app, exception handlers
  config.py                Settings (env-driven)
  models/schemas.py        Pydantic request/response + trace models
  core/
    llm_client.py          Provider abstraction (OpenAI/Anthropic/Fake)
    state.py                Conversation state store + context window mgmt
    dependencies.py         Singleton wiring for FastAPI Depends
  agents/
    graph.py                LangGraph state machine (the main agent)
    compliance.py            Compliance Agent guardrail
    orchestrator.py          Wires compliance + graph per turn
    prompts.py               Externalized, versioned prompts
  tools/definitions.py       Tool schemas + implementations
  retrieval/
    knowledge_base.py        FAQ loading + cleaning rules
    embeddings.py             Idempotent embedding index + cosine search
  observability/tracer.py    Structured JSON trace logger
  api/routes/conversations.py  POST /messages, GET /trace
data/faq_kb.json             Source FAQ dataset
tests/unit/                  Retrieval + tool tests
tests/integration/            Full-conversation + FastAPI TestClient tests
eval/                         Eval harness (dataset + scoring script)
```

## Production evaluation plan

**Subjective quality (helpfulness, tone, hallucination rate):** LLM-as-judge
on a held-out sample of real conversations, scored against a rubric (does it
resolve the issue, is tone on-brand, are any claims unsupported by KB/tool
output). Pair with periodic human spot-checks — LLM-judge and human agreement
rate itself becomes a metric to watch for judge drift.

**Objective metrics:**
- *Retrieval*: precision@k against a labeled query→expected-FAQ set;
  track score distribution of accepted vs. rejected (below `faq_min_score`)
  matches to tune the threshold over time.
- *Tool-call accuracy*: does the planner pick the tool a human reviewer would
  have picked, on a sampled/labeled set of real traces.
- *Latency*: per-step (`TraceStep.latency_ms`) and end-to-end, p50/p95/p99,
  broken out by which tool was called (search_faq should be fast; escalation
  involves no LLM synthesis and should be near-instant).
- *Cost*: `ConversationTrace.total_cost_usd`, aggregated by category/day; alert
  on conversations that hit `max_verification_retries` since those are
  paying for extra LLM calls without resolving.

**Failure modes to watch for in production:**
- *Verification false-negatives*: the verify step itself hallucinating "grounded:
  true" for an ungrounded answer. Mitigate with periodic human audits of a
  sample of `verified: true` responses.
- *Compliance false-positives/negatives*: legitimate questions refused, or
  injections that slip through creative phrasing. Track refusal rate over time
  and sample refused messages for review.
- *Retrieval drift*: KB content changes but cached embeddings don't (should be
  caught by the hash-based cache, but worth a canary test that re-embeds and
  diffs top-k results periodically).
- *Loop non-convergence*: agent regularly hitting `max_agent_iterations` and
  force-escalating — a leading indicator that the tool set or prompts need
  attention.
- *Context window truncation losing critical details*: watch for a spike in
  clarification requests on turn 3+ of long conversations, which would suggest
  the running summary is dropping something the user already told the agent.

**With two more weeks:** Postgres + pgvector instead of the in-memory/JSON-file
store (real persistence + concurrent access), parallel tool execution for
independent lookups, LLM-as-judge wired into CI against a larger labeled eval
set (currently rule-based only), and a human-in-the-loop review queue for
`escalate_to_human` cases with an approval endpoint.

## What I'd flag if this were a real handoff
- `data/embedding_cache.json` is git-ignored; first run always embeds fresh.
- The `FakeLLMClient`/`LLM_PROVIDER=fake` path is intentionally more capable
  than a typical "always returns the same string" stub, specifically so the
  test suite and eval harness are meaningful without needing the (rate-limited)
  OpenAI key. It is not a substitute for testing against the real model before
  shipping — see `eval/run_eval.py` usage with `LLM_PROVIDER=openai`.
- In-memory conversation store means state doesn't survive a process restart;
  fine for this submission, flagged as the first thing to swap for anything
  beyond a demo.
