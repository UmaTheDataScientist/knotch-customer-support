from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import SupportOrchestrator
from app.config import Settings
from app.core.llm_client import FakeLLMClient
from app.core.state import ConversationStore
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.knowledge_base import load_faq_items

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "faq_kb.json"


@pytest.fixture
def faq_items():
    return load_faq_items(KB_PATH)


@pytest.fixture
def scripted_llm():
    return FakeLLMClient()


@pytest.fixture
def test_settings():
    return Settings(llm_provider="fake", max_agent_iterations=4, max_verification_retries=2, faq_top_k=3, faq_min_score=0.05)


@pytest.fixture
def embedding_index(scripted_llm, faq_items, tmp_path):
    idx = EmbeddingIndex(scripted_llm, cache_path=tmp_path / "cache.json")
    idx.build(faq_items)
    return idx


@pytest.fixture
def orchestrator(scripted_llm, test_settings, embedding_index, faq_items):
    return SupportOrchestrator(llm=scripted_llm, settings=test_settings, index=embedding_index, faq_items=faq_items)


@pytest.fixture
def conv_store():
    return ConversationStore()
