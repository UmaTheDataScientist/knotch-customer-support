# Knotch AI Customer Support

A customer support app that handles multi-turn conversations, figures out which
tool to use for each request, and checks its own answers before replying. Built
with FastAPI and LangGraph.

## How to run it

No API key needed to try it out — it defaults to an offline mode for
development and testing.

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
`.env.example` to `.env` and fill in `LLM_PROVIDER` + your API key.

## Architecture

Every message goes through two agents:

1. **Compliance Agent** checks the message first — is it a jailbreak attempt,
   off-topic, or a real support question? Unsafe messages are refused
   immediately, before the main agent ever sees them.
2. **Main agent**, if the message is safe, runs a bounded loop:
   - **Plan** — classify intent, pick exactly one tool (search the FAQ, ask a
     clarifying question, escalate to a human, etc).
   - **Act** — call that tool.
   - **Observe** — turn the tool's result into a draft reply.
   - **Verify** — check the draft actually answers the question, is grounded
     in real content, and doesn't leak internal details. If it fails, loop
     back to Plan (capped at 2 retries) instead of looping forever.

This is built as an explicit LangGraph state machine — each step above is a
node, with a real conditional edge for the retry loop — rather than a single
function with a hidden while-loop. That makes the two safety bounds (max
iterations, max verification retries) visible and testable instead of buried
in control flow, and every step logs its own trace entry, so a bad
conversation can be debugged by reading `GET /conversations/{id}/trace`
instead of guessing.

The compliance check running *before* the main agent (not in parallel or
after) is deliberate: it avoids wasting a planning/retrieval call on a message
that's about to be refused anyway, and it means nothing the main agent
produces is ever shown to a user before a policy check has happened.

Six required tools plus two extra ones (`check_system_status`,
`lookup_account_status`) live in `app/tools/definitions.py`. The FAQ knowledge
base is embedded and cached (`app/retrieval/`), so re-running ingestion only
re-embeds rows that actually changed. Conversation history is kept in memory
per `conversation_id`, with older turns compressed into a running summary
instead of growing the context forever.

## Extras beyond what the assignment asked for

- **Two extra tools** — `check_system_status` and `lookup_account_status` —
  replacing a static "go check the status page" FAQ answer with something the
  bot can actually check itself.
- **A fully offline LLM stand-in** (`FakeLLMClient`) that routes intents,
  classifies compliance verdicts, and answers questions with zero network
  calls — this is what lets the whole test suite and eval harness run free
  and in under 2 seconds, which mattered since the real API key is
  rate-limited and rotates weekly.
- **A local dev chat UI** (see below) with a live, step-by-step trace panel.
- **Two small diagnostic scripts** in `scripts/` — one confirms a real API key
  works with exactly 2 calls, the other inspects what actually got loaded from
  `.env` without ever printing the secret.
- **Retry logic that skips retrying errors that can never succeed** (bad key,
  malformed request) instead of wasting 3 attempts on something guaranteed to
  fail every time.
- **A server-side Markdown-stripping safety net** — prompts ask the model for
  plain text, but that's not a guarantee, so responses are also sanitized
  server-side regardless of source.
- **The planner's FAQ category list is derived from the real knowledge base
  at runtime**, not hand-typed — so editing the KB can never make the prompt
  silently go stale.

## Frontend (local dev chat UI)

A simple two-panel chat interface — plain chat on the left, a live trace of
each plan/act/observe/verify step on the right. Not part of the API itself;
just a way to interact with it visually instead of using curl.

**1. Start the API** (one terminal):
```bash
uvicorn app.main:app --reload --port 8000
```

**2. Serve the UI folder** (a second terminal):
```bash
cd devtools
python -m http.server 5500
```

**3. Open in your browser:**
```
http://127.0.0.1:5500/chat_ui.html
```

If nothing happens when you send a message, check the "API base" field at the
bottom of the page matches the port your server is running on.
