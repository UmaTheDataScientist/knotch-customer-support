"""FAQ coverage check.

For every indexed question in the knowledge base, sends that exact question
text through the real agent (a fresh conversation per question) and checks:
  1. Did it route through search_faq (source == "faq"), rather than getting
     misrouted to escalate_to_human / check_system_status / clarification /
     general_knowledge_lookup?
  2. Did the item's OWN question appear among the retrieved matches?
  3. Is it the TOP match, or did something else outrank it?

This is the test that would have caught the "site slow" and "account
locked" misrouting bugs across the WHOLE dataset, not just the one or two
cases we happened to test by hand. Every KB question is, by construction,
the easiest possible input for that same entry to retrieve -- if asking a
question verbatim doesn't reliably surface its own answer, that's a real
routing or retrieval problem worth knowing about before a user hits it.

IMPORTANT -- cost/rate-limit note: this makes multiple real LLM calls per
question (compliance + plan + synthesis + verify), so for 32 questions that
is roughly 100+ calls. The assignment's key is rate-limited to 50 req/min.
Don't run this back-to-back with other real-key testing; run it once,
deliberately, and read the results.

Usage:
    LLM_PROVIDER=openai python eval/faq_coverage_check.py
    LLM_PROVIDER=fake python eval/faq_coverage_check.py   # structural smoke test only,
                                                            # does not validate real routing
                                                            # (FakeLLMClient is rule-based,
                                                            # not a semantic judge)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import SupportOrchestrator
from app.config import get_settings
from app.core.llm_client import build_llm_client
from app.core.state import ConversationStore
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.knowledge_base import load_faq_items

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "faq_kb.json"
CACHE_PATH = Path(__file__).resolve().parent / "faq_coverage_embedding_cache.json"


def run() -> None:
    settings = get_settings()
    llm = build_llm_client(settings)
    faq_items = load_faq_items(KB_PATH)
    indexed_items = [i for i in faq_items if i.indexed]

    index = EmbeddingIndex(llm, cache_path=CACHE_PATH)
    index.build(faq_items)
    orchestrator = SupportOrchestrator(llm=llm, settings=settings, index=index, faq_items=faq_items)
    store = ConversationStore()

    print(f"Provider: {settings.llm_provider}")
    print(f"Testing {len(indexed_items)} indexed FAQ questions verbatim...\n")

    results = []
    for i, item in enumerate(indexed_items):
        conv = store.get_or_create(f"faq-coverage-{item.id}")
        try:
            out = orchestrator.handle_message(conv, item.question)
        except Exception as exc:  # noqa: BLE001 - keep going even if one question errors
            results.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "ok": False,
                    "reason": f"exception: {exc}",
                    "source": None,
                    "tools_used": [],
                    "top_match": None,
                }
            )
            continue

        source_ok = out.source.value == "faq"
        self_retrieved = item.question in out.matched_questions
        top_match = out.matched_questions[0] if out.matched_questions else None
        is_top = top_match == item.question
        ok = source_ok and self_retrieved

        reason = ""
        if not source_ok:
            reason = f"routed to source={out.source.value}, tools_used={out.tools_used}"
        elif not self_retrieved:
            reason = f"own question not in matches: {out.matched_questions}"
        elif not is_top:
            reason = f"retrieved but not top match (top was: {top_match!r})"

        results.append(
            {
                "id": item.id,
                "question": item.question,
                "ok": ok,
                "reason": reason,
                "source": out.source.value,
                "tools_used": out.tools_used,
                "top_match": top_match,
                "is_top": is_top,
            }
        )
        print(
            f"[{i + 1}/{len(indexed_items)}] {'OK  ' if ok else 'FAIL'} {item.question!r}"
            + (f"  -- {reason}" if reason else "")
        )

        # Small pacing gap to be gentle on a rate-limited key across ~4
        # calls/question; harmless no-op for the offline fake client.
        if settings.llm_provider != "fake":
            time.sleep(0.2)

    total = len(results)
    passed = sum(r["ok"] for r in results)
    top_match_count = sum(r.get("is_top", False) for r in results if r["ok"])

    print("\n" + "=" * 70)
    print(f"Routed to search_faq AND self-retrieved: {passed}/{total} ({passed / total:.1%})")
    print(f"Of those, ranked as the TOP match:        {top_match_count}/{passed}")

    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"\n{len(failures)} failing question(s):")
        for r in failures:
            print(f"  - {r['question']!r}: {r['reason']}")
        sys.exit(1)


if __name__ == "__main__":
    run()
