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
verify step on the right. If nothing happens when you send a message, check
the "API base" field at the bottom of the page matches the port your server
is running on.

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
   - **Plan**: classify intent, pick exactly one tool (search the FAQ, ask a
     clarifying question, escalate to a human, etc).
   - **Act**: call that tool.
   - **Observe**: turn the tool's result into a draft reply.
   - **Verify**: check the draft actually answers the question, is grounded
     in real content, and doesn't leak internal details. If it fails, loop
     back to Plan (capped at 2 retries) instead of looping forever.

This is built as an explicit LangGraph state machine. Each step above is a
node, with a real conditional edge for the retry loop, rather than a single
function with a hidden while-loop. That makes the two safety bounds (max
iterations, max verification retries) visible and testable instead of buried
in control flow, and every step logs its own trace entry, so a bad
conversation can be debugged by reading `GET /conversations/{id}/trace`
instead of guessing.

The compliance check running before the main agent, not in parallel or after,
is deliberate. It avoids wasting a planning/retrieval call on a message
that's about to be refused anyway, and it means nothing the main agent
produces is ever shown to a user before a policy check has happened.

Six required tools plus two extra ones (`check_system_status`,
`lookup_account_status`) live in `app/tools/definitions.py`. The FAQ knowledge
base is embedded and cached (`app/retrieval/`), so re-running ingestion only
re-embeds rows that actually changed. Conversation history is kept in memory
per `conversation_id`, with older turns compressed into a running summary
instead of growing the context forever.

## Extras beyond what the assignment asked for

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
- **Docker Compose covers the whole system**, not just the API. A second
  service serves the frontend too, so `docker compose up` is a single command
  for both.

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
for real persistence and concurrent access; parallel tool execution for
independent lookups; a larger labeled eval set with LLM-as-judge wired into
CI instead of the current rule-based scoring; a human-in-the-loop review
queue for `escalate_to_human` cases.
