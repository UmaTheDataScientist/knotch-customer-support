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

# run the test suite (46 tests, all offline)
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

### Keeping the planner's prompt in sync with the actual KB
Early on, I hardcoded a list of "common support categories" directly into the
planner's system prompt, to nudge it toward `search_faq` over
`general_knowledge_lookup`. When I checked it against the real data, that
hand-typed list was already missing 4 of the KB's 12 actual categories
(`account_lifecycle`, `data_recovery`, `developer`, `security_incident`) —
a classic single-source-of-truth bug: two copies of the same information
(the KB's real categories, and my guess at them in a prompt string), with
nothing forcing them to stay in sync.

I replaced the static prompt string with `build_plan_system_prompt()` in
`app/agents/prompts.py`, which takes the actual category list — derived from
`data/faq_kb.json` at startup via `SupportOrchestrator` — and formats it
directly into the prompt. Editing the KB (adding, removing, renaming a
category) now updates the planner's guidance automatically, with zero prompt
changes required. `tests/integration/test_conversations.py` has two tests
locking this in: one confirms every real category actually appears in the
generated prompt, the other simulates a brand-new category being added and
confirms the prompt picks it up without any code change.

### Plain-text output, defended two ways
The `response` field is a raw string, displayed as-is (the assignment's own
example outputs are all plain text, no Markdown). I noticed a real model
occasionally writes `**Settings**`-style Markdown by default, which shows up
as literal asterisks in a plain-text display rather than bold text. I fixed
this two ways, not just one:
1. Both prompts that generate free text (`SYNTHESIZE_SYSTEM_PROMPT` and the
   `general_knowledge_lookup` tool's own system prompt) now explicitly ask
   for plain text, no Markdown.
2. Since a prompt instruction is not a guarantee, `app/core/text_utils.py`
   also strips common Markdown syntax (bold, headers, inline code, links,
   blockquotes) from every final response server-side, regardless of which
   tool produced it. It's a no-op on already-plain text, so it's safe to
   apply uniformly rather than only to LLM-generated paths.

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
  `app/agents/prompts.py`, not inlined in graph nodes. The planner's prompt
  specifically is data-driven rather than static (see above), which I think
  matters more than the version tag for keeping prompts honest over time.
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

## Beyond the assignment: what wasn't explicitly asked for

The assignment names 6 required tools, a defined set of bonuses, and specific
deliverables. Everything below was added on top of that, either because it
made the required system stronger or because building/testing it surfaced a
real gap I wanted to close rather than leave for a reviewer to find.

**Two extra tools** (the doc explicitly allows this, but doesn't ask for it):
`check_system_status` and `lookup_account_status` (`app/tools/definitions.py`).
Both replace a static FAQ workaround ("go check the status page yourself")
with something an actual support bot should be able to do itself. Both are
mocked the same way the required `escalate_to_human` already is, and both are
wired all the way through -- planner prompt, tool implementation, `observe`
step, unit tests, and eval harness scenarios. Details in "Tools beyond the
assignment's minimum list" above.

**`FakeLLMClient`, a full offline stand-in for the LLM provider**
(`app/core/llm_client.py`). Not just a stub that returns a fixed string --
it's rule-based enough to route intents, classify compliance verdicts, and
produce plausible answers, entirely without network calls. This is what lets
all 46 tests and the entire eval harness run in under 2 seconds with zero API
cost, which mattered a lot given the real key is rate-limited (50 req/min)
and rotates weekly -- I didn't want every test run to compete with actual
development for that budget.

**A local dev chat UI** (`devtools/chat_ui.html`), a single self-contained
HTML file with no build step: a chat window plus a live, step-by-step trace
panel that updates after every message. Not part of the graded system (it's
outside `app/`, calls the API over plain HTTP) -- built purely so I could
watch the plan/act/observe/verify cycle happen in real time instead of
reading raw trace JSON in a terminal. Required adding CORS middleware to
`app/main.py` so a browser-served page could call the API at all.

**Two small diagnostic scripts** (`scripts/`): `smoke_test_real_key.py` makes
exactly 2 API calls (one chat, one embedding) to confirm the real key and
model names work before spending any real budget on the full eval harness;
`check_env_loading.py` inspects what actually got loaded from `.env` (length,
stray whitespace, quote characters, BOM bytes) without ever printing the
secret itself -- came out of a real debugging session where a working key
still failed silently on Windows.

**Retry logic that doesn't retry the wrong things** (`app/core/llm_client.py`,
`_llm_retry`). The default naive approach retries every failed API call
2-3 times with backoff -- fine for a rate limit or a timeout, actively wasteful
for a bad key or a malformed request, which will never succeed no matter how
many times you retry them. `_llm_retry()` explicitly excludes
`AuthenticationError`, `BadRequestError`, `NotFoundError`, and
`PermissionDeniedError` (across both the OpenAI and Anthropic SDKs) from the
retry policy, so those fail in one attempt instead of three.

**A server-side Markdown-stripping safety net** (`app/core/text_utils.py`).
Prompts ask the model for plain text, but that's an instruction, not a
guarantee -- a real model still occasionally wrote `**bold**`-style Markdown,
which showed up as literal asterisks in a plain-text response field. Rather
than rely on the prompt alone, `strip_markdown()` runs on every final
response server-side regardless of source, with its own test suite (13 unit
tests) plus an integration test I specifically proved has teeth by disabling
the fix and confirming it fails before trusting it.

**Deriving the planner's prompt from the real KB instead of hand-typing it**
(`app/agents/prompts.py::build_plan_system_prompt`). An earlier version of
this prompt hardcoded a list of "common support categories" from memory, and
that list was already missing 4 of the KB's 12 real categories the moment it
was written. The category list is now computed from `data/faq_kb.json` at
startup, so editing the KB can never make the prompt stale again -- there's
no second copy of the category list anywhere to forget to update.

**`GET /health`**, a trivial but conventional readiness endpoint most API
frameworks ship with by default, not asked for in the spec.

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
