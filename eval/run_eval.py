"""Automated eval harness.

Runs eval/dataset.jsonl through the real orchestrator (compliance agent +
LangGraph agent) and scores:
  - tool_use_accuracy: did the agent call at least one of the expected tools?
  - source_accuracy: did the response come from the expected source?
  - guardrail_success_rate: for off_topic/malicious cases specifically, was
    the request correctly blocked by the Compliance Agent?

This is deliberately rule-based (checking source/tools_used against the
dataset's `expect_*` fields) rather than LLM-as-judge, since these cases
have an objectively correct routing decision -- LLM-as-judge is better
saved for grading subjective answer *quality*, which the README's
production-evaluation section covers separately.

Usage:
    LLM_PROVIDER=fake python eval/run_eval.py
    LLM_PROVIDER=openai OPENAI_API_KEY=... python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.llm_client import build_llm_client
from app.core.state import ConversationStore
from app.agents.orchestrator import SupportOrchestrator
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.knowledge_base import load_faq_items

DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
KB_PATH = Path(__file__).resolve().parent.parent / "data" / "faq_kb.json"


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def run() -> None:
    settings = get_settings()
    llm = build_llm_client(settings)
    faq_items = load_faq_items(KB_PATH)
    index = EmbeddingIndex(llm, cache_path=Path(__file__).resolve().parent / "eval_embedding_cache.json")
    index.build(faq_items)
    orchestrator = SupportOrchestrator(llm=llm, settings=settings, index=index, faq_items=faq_items)
    store = ConversationStore()

    cases = load_dataset()
    results = []
    for case in cases:
        conv = store.get_or_create(f"eval-{case['id']}")
        out = None
        for turn in case["turns"]:
            out = orchestrator.handle_message(conv, turn)

        source_ok = out.source.value == case["expect_source"]
        tool_ok = any(t in out.tools_used for t in case["expect_tools_any"])
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "source_ok": source_ok,
                "tool_ok": tool_ok,
                "got_source": out.source.value,
                "got_tools": out.tools_used,
                "response_preview": out.response[:80],
            }
        )

    total = len(results)
    source_acc = sum(r["source_ok"] for r in results) / total
    tool_acc = sum(r["tool_ok"] for r in results) / total

    guardrail_cases = [r for r in results if r["category"] in ("off_topic", "malicious")]
    guardrail_rate = (
        sum(r["source_ok"] and r["tool_ok"] for r in guardrail_cases) / len(guardrail_cases)
        if guardrail_cases
        else float("nan")
    )

    print(f"{'ID':<14} {'CATEGORY':<12} {'SOURCE_OK':<10} {'TOOL_OK':<8} GOT_SOURCE / GOT_TOOLS")
    for r in results:
        print(
            f"{r['id']:<14} {r['category']:<12} {str(r['source_ok']):<10} {str(r['tool_ok']):<8} "
            f"{r['got_source']} / {r['got_tools']}"
        )

    print()
    print(f"source_accuracy:        {source_acc:.2%}")
    print(f"tool_use_accuracy:      {tool_acc:.2%}")
    print(f"guardrail_success_rate: {guardrail_rate:.2%}")

    failures = [r for r in results if not (r["source_ok"] and r["tool_ok"])]
    if failures:
        print(f"\n{len(failures)} failing case(s): {[f['id'] for f in failures]}")
        sys.exit(1)


if __name__ == "__main__":
    run()
