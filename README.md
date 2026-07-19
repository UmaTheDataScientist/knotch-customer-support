# Knotch AI Customer Support

A customer support app that handles multi-turn conversations, figures out which
tool to use for each request, and checks its own answers before replying. Built
with FastAPI and LangGraph.

## Quick start (any laptop, Docker only)

Requires Docker Desktop and git installed.

```bash
git clone https://github.com/UmaTheDataScientist/knotch-customer-support.git
cd knotch-customer-support
docker compose up --build
```

- API: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5500/chat_ui.html`

No Python install or virtual environment needed. Runs in an offline mode by
default (`LLM_PROVIDER=fake`), so there's nothing else to configure to try it
out.

To use a real OpenAI (or Anthropic) key instead, export these before running
the command above:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
docker compose up --build
```

Don't have Docker? See "Manual setup" below for the plain Python path.

## Manual setup (no Docker)

Requires Python 3.11 or newer installed.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run the test suite (46 tests, all offline)
pytest -q

# run the eval harness (13 scenarios, all offline)
python eval/run_eval.py

# start the API
uvicorn app.main:app --reload --port 8000
```

Example request:
```bash
curl -X POST http://127.0.0.1:8000/conversations/abc-123/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?"}'
```

To use a real OpenAI (or Anthropic) key instead of the offline default, copy
`.env.example` to `.env` and fill in `LLM_PROVIDER` plus your API key.

To also run the frontend manually, in a second terminal alongside the API
above:
```bash
cd devtools
python -m http.server 5500
```
Then open `http://127.0.0.1:5500/chat_ui.html`. It's a simple two-panel chat
interface: plain chat on the left, a live trace of each plan/act/observe/
verify step on the right. Click Browse to see every conversation currently
held by the server and click into one, or type a conversation id directly and
click Load. Both work as long as the server hasn't restarted, since
conversation state is in-memory only (see "Extras" below). If nothing happens
when you send a message, check the "API base" field at the bottom of the page
matches the port your server is running on.

**Start here as a reviewer:**
- `app/agents/graph.py`, the Plan → Act → Observe → Verify loop as a LangGraph
  state machine (the core of the system).
- `app/agents/compliance.py`, the guardrail agent, and why it runs before the
  main agent rather than in parallel or after.
- `tests/integration/test_conversations.py`, covers the four example
  interactions from the assignment doc, plus a few regression tests for real
  bugs found by testing against a live model, not just the offline suite.
- `eval/run_eval.py`, automated eval harness, 13 scenarios, 100% pass rate.

## Architecture

Every message goes through two agents:

1. **Compliance Agent** checks the message first. Is it a jailbreak attempt,
   off-topic, or a real support question? Unsafe messages are refused
   immediately, before the main agent ever sees them.
2. **Main agent**, if the message is safe, runs a bounded loop:
   - **Plan**: classify every distinct request in the message (a message can
     contain more than one, e.g. "edit my avatar and cancel my subscription")
     and pick exactly one tool per request.
   - **Act**: call each tool, in order.
   - **Observe**: turn each tool's result into an answer, then combine them
     into one coherent reply if there was more than one request.
   - **Verify**: check the combined draft actually answers everything asked,
     is grounded in real content, and doesn't leak internal details. If it
     fails, loop back to Plan (capped at 2 retries) instead of looping
     forever.

```
User message
     |
     v
Compliance Agent --unsafe--> refuse (source = compliance)
     |
    safe
     |
     v
 LangGraph main-agent loop (per conversation, checkpointed)
+------------------------------------------------+
|Plan (1+ sub-requests, one tool each)           |
|  |                                             |
|  v                                             |
|Act (run each sub-request's tool, in order;     |
|     a clarification/refuse/escalation short-   |
|     circuits the rest)                         |
|  |                                             |
|  v                                             |
|Observe (turn each result into an answer,       |
|         combine if more than one sub-request)  |
|  |                                             |
|  v                                             |
|Verify (addresses question? grounded?           |
|        no leaked internals?)                   |
|  |                                             |
|  +--- fails (<= 2 retries) ---> back to Plan   |
|  |                                             |
| passes                                         |
|  |                                             |
|  v                                             |
|Finalize (strip markdown, mark done)            |
+------------------------------------------------+
     |
     v
ChatMessageOut  (full step-by-step trace at GET /conversations/{id}/trace)
```

If any sub-request in Act resolves to `ask_user_clarification`, `refuse`, or
`escalate_to_human`, that one short-circuits the rest of the loop for this
turn rather than getting blended with an unrelated answer.

This is built as an explicit LangGraph state machine. Each step above is a
node, with a real conditional edge for the retry loop, rather than a single
function with a hidden while-loop. That makes the two safety bounds (max
iterations, max verification retries) visible and testable instead of buried
in control flow, and every step logs its own trace entry, so a bad
conversation can be debugged by reading `GET /conversations/{id}/trace`
instead of guessing. Each conversation also gets its own LangGraph
checkpointer (`app/core/state.py`), so every node transition is a real
persisted snapshot, inspectable via `get_state_history()`, not just an
in-memory pass-through.

The compliance check running before the main agent, not in parallel or after,
is deliberate. It avoids wasting a planning/retrieval call on a message
that's about to be refused anyway, and it means nothing the main agent
produces is ever shown to a user before a policy check has happened.

Multi-request messages are handled by decomposing the plan step's output into
a list of sub-requests instead of a single tool call, then executing each one
and combining the answers (`app/agents/graph.py`). If any sub-request resolves
to a clarification question, a refusal, or an escalation, that one takes over
the whole turn instead of being silently blended with an unrelated answer,
since mixing "I need more detail" with a confident answer in the same reply
would be confusing and is arguably worse than just asking.

Six required tools plus two extra ones (`check_system_status`,
`lookup_account_status`) live in `app/tools/definitions.py`. The FAQ knowledge
base is embedded and cached (`app/retrieval/`), so re-running ingestion only
re-embeds rows that actually changed. Conversation history is kept in memory
per `conversation_id`, with older turns compressed into a running summary
instead of growing the context forever.

The provided FAQ dataset is intentionally messy. Two entries needed a judgment
call, documented in `app/retrieval/knowledge_base.py`: `"x"` is excluded from
the *searchable* index since its answer is itself an instruction to ask for
clarification, not a fact, indexing it would make it falsely match almost any
short query, and `ask_user_clarification` already handles that case
dynamically. `"help!!! my account is locked"` is a real case just noisily
formatted, so it's kept, with a normalized version used only for the
embedding text so the formatting noise doesn't hurt retrieval.

## Bonus points implemented

The assignment lists 11 optional bonuses and says to pick a few rather than
attempt all of them. Six are done here, matching the assignment's own
numbering:

- **#1, LangGraph state machine with cycles and checkpointing.** Explicit
  nodes/edges/conditional transitions (`app/agents/graph.py`), a real replan
  cycle (verify fails -> back to plan, capped at 2 retries), and genuine
  LangGraph checkpointing (a `MemorySaver` per conversation) -- proven with a
  test that inspects `get_state_history()` directly and fails if the
  checkpointer is removed (`tests/integration/test_checkpointing.py`).
- **#2, parallel tool execution.** A multi-intent message's independent
  sub-requests (e.g. two separate `search_faq` calls) run concurrently via a
  `ThreadPoolExecutor` instead of one after another. Proven with a timing
  test: three artificially delayed tool calls finish in ~1x the delay, not
  ~3x (`test_independent_sub_requests_run_in_parallel_not_sequentially`). A
  single escalation sub-request still short-circuits the rest of the plan
  *before* anything runs, preserving the cost-saving behavior that predates
  parallelism.
- **#5, eval harness.** `eval/dataset.jsonl` (13 scenarios: happy path,
  ambiguous, off-topic, malicious, multi-turn, escalation, status checks)
  scored by `eval/run_eval.py` against source accuracy, tool-use accuracy,
  and guardrail success rate. Currently 100% across all three, offline.
- **#6, human-in-the-loop interrupt.** Any plan that includes
  `escalate_to_human` pauses instead of executing immediately
  (`app/core/review_queue.py`), the turn returns a "waiting on human
  sign-off" response with a `pending_review_id`, and `GET /reviews`,
  `POST /reviews/{id}/approve`, `POST /reviews/{id}/reject` let a (mocked)
  reviewer inspect the queued plan and actually resume or reject it. The gate
  is opt-in (only active when a `ReviewQueue` is wired in), so a caller with
  no reviewer configured gets the old immediate-execution behavior
  unchanged.
- **#9, idempotent embedding management.** `EmbeddingIndex.build()` hashes
  each item's embedding text (`sha256`) and only re-embeds rows whose hash
  actually changed, verified by building twice and checking `reused` vs
  `embedded` counts.
- **#11, Dockerfile + docker-compose for the whole system.** Two services,
  API and frontend, one `docker compose up --build` command for both.

Not implemented, for the record: #3 long-term memory, #4 cost-aware routing,
#7 auth via `Depends`, #8 Postgres + pgvector, #10 async embedding ingestion
via Celery.

## Extras beyond what the assignment asked for

Things added that aren't on the assignment's bonus list at all, but seemed
worth building along the way:

- **Two extra tools**: `check_system_status` and `lookup_account_status`,
  replacing a static "go check the status page" FAQ answer with something the
  bot can actually check itself.
- **A fully offline LLM stand-in** (`FakeLLMClient`) that routes intents,
  classifies compliance verdicts, and answers questions with zero network
  calls. This is what lets the whole test suite and eval harness run free and
  in under 2 seconds, which mattered since the real API key is rate-limited
  and rotates weekly.
- **A local dev chat UI** (`devtools/chat_ui.html`), a self-contained HTML
  file with no build step or dependencies. A chat window on one side, a live
  trace of every plan/act/observe/verify step on the other, updating after
  each message. Not part of the graded API itself, built so the agent's
  reasoning could be watched happen in real time instead of reading raw trace
  JSON in a terminal.
- **`POST /conversations`, `GET /conversations`, and `GET /conversations/{id}/messages`**
  -- create a new conversation with a server-guaranteed-unique id, list every
  conversation currently in memory (with a preview), and fetch a
  conversation's full message history, not just its trace. The create
  endpoint matters more than it sounds: the dev UI originally generated its
  own conversation ids client-side with `Math.random()`, which has no real
  uniqueness guarantee -- a real client shouldn't be trusted to generate
  collision-free ids, the server should. `GET /conversations` and
  `GET /conversations/{id}/messages` back the dev UI's Browse/Load buttons;
  without them there was no way to discover or resume an existing
  conversation by id.
- **Two small diagnostic scripts** in `scripts/`. One confirms a real API key
  works with exactly 2 calls, the other inspects what actually got loaded from
  `.env` without ever printing the secret.
- **Retry logic that skips retrying errors that can never succeed** (bad key,
  malformed request) instead of wasting 3 attempts on something guaranteed to
  fail every time.
- **A server-side Markdown-stripping safety net.** Prompts ask the model for
  plain text, but that's not a guarantee, so responses are also sanitized
  server-side regardless of source.
- **The planner's FAQ category list is derived from the real knowledge base
  at runtime**, not hand-typed, so editing the KB can never make the prompt
  silently go stale.
- **A precondition-mismatch check on retrieved FAQ content.** The provided
  dataset's only password-reset entry says "enter your current password,"
  which doesn't fit someone who says they forgot it, and there's no separate
  account-recovery entry in the data at all. Rather than handing over steps
  the user has already said they can't perform, synthesis and verification
  both check for this kind of mismatch and redirect to support instead of
  presenting a bad-fit answer as if it were correct.
- **The replan loop now actually tells the planner what failed and why.**
  Found via live testing: "why is the site slow today?" (which the KB
  answers directly under `troubleshooting`) got routed to
  `check_system_status` instead, whose generic "systems operational"
  response correctly failed verification -- but the retry produced the
  *identical* plan every time, since `verification_reasoning` was computed
  and even logged in the trace but never fed back into the next plan
  attempt. Six identical retries later, it force-escalated to a human for
  something the FAQ could have answered. The plan node now includes the
  previous attempt's tool and failure reason as explicit context on replan,
  so a second attempt has a real chance to try something different instead
  of repeating the same mistake until the iteration budget runs out.

## How I'd evaluate this in production

**Subjective quality** (helpfulness, tone, hallucination rate): LLM-as-judge
against a rubric on a sample of real conversations, paired with periodic human
spot-checks. The agreement rate between judge and human becomes its own
metric to watch for judge drift.

**Objective metrics**: retrieval precision@k against a labeled query set;
tool-call accuracy (did the planner pick the tool a human reviewer would
have); per-step and end-to-end latency (`TraceStep.latency_ms`), broken out
by tool; cost per conversation (`ConversationTrace.total_cost_usd`), with
alerts on conversations that hit `max_verification_retries`, since those pay
for extra LLM calls without necessarily resolving anything.

**Failure modes to watch for**: the verify step itself hallucinating
"grounded: true" for an ungrounded answer (needs human audits of a sample of
`verified: true` responses, not just trusting the flag); the Compliance
Agent producing false positives/negatives as phrasing gets more creative;
retrieval drift if the KB changes but the embedding cache doesn't catch it;
the planner regularly hitting the iteration cap and force-escalating, which
would signal the tool set or prompts need attention.

**With two more weeks**: Postgres and pgvector instead of the in-memory store
for real persistence and concurrent access; a larger labeled eval set with
LLM-as-judge wired into CI instead of the current rule-based scoring; a
"Pending Reviews" panel in the dev UI so the human-in-the-loop
approve/reject flow is demoable end-to-end from one screen instead of
needing curl or `/docs` (the backend for this already exists --
`GET /reviews`, `POST /reviews/{id}/approve`, `POST /reviews/{id}/reject` --
it's only the UI side that's missing).
