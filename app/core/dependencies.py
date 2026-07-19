"""Application-wide singletons, built once and shared via FastAPI Depends.

Kept in one place so tests can monkeypatch `get_orchestrator`/`get_store`
to inject a FakeLLMClient without touching route code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.agents.orchestrator import SupportOrchestrator
from app.config import get_settings
from app.core.llm_client import build_llm_client
from app.core.review_queue import ReviewQueue
from app.core.state import ConversationStore
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.knowledge_base import load_faq_items

_KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "faq_kb.json"
_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "embedding_cache.json"


@lru_cache
def get_conversation_store() -> ConversationStore:
    return ConversationStore()


@lru_cache
def get_review_queue() -> ReviewQueue:
    return ReviewQueue()


@lru_cache
def get_orchestrator() -> SupportOrchestrator:
    settings = get_settings()
    llm = build_llm_client(settings)
    faq_items = load_faq_items(_KB_PATH)
    index = EmbeddingIndex(llm, cache_path=_CACHE_PATH)
    index.build(faq_items)
    return SupportOrchestrator(
        llm=llm, settings=settings, index=index, faq_items=faq_items, review_queue=get_review_queue()
    )


def reset_singletons() -> None:
    """Used by tests to force fresh singletons under a different LLM_PROVIDER."""
    get_conversation_store.cache_clear()
    get_review_queue.cache_clear()
    get_orchestrator.cache_clear()
