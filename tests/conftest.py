from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import SupportOrchestrator
from app.config import Settings
from app.core.llm_client import FakeLLMClient, LLMResult
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


class RecordingLLMClient(FakeLLMClient):
    """Wraps FakeLLMClient but records the exact `messages` list passed to
    every chat() call, so tests can assert on message *shape* (e.g. no
    duplicate consecutive turns) rather than only on final output."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, json_mode: bool = False, temperature: float = 0.2) -> LLMResult:
        self.calls.append(messages)
        return super().chat(messages, json_mode=json_mode, temperature=temperature)

    def plan_calls(self) -> list[list[dict]]:
        return [m for m in self.calls if any("planning step" in msg["content"] for msg in m if msg["role"] == "system")]


@pytest.fixture
def recording_llm():
    return RecordingLLMClient()


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
def recording_orchestrator(recording_llm, test_settings, faq_items, tmp_path):
    idx = EmbeddingIndex(recording_llm, cache_path=tmp_path / "recording_cache.json")
    idx.build(faq_items)
    return SupportOrchestrator(llm=recording_llm, settings=test_settings, index=idx, faq_items=faq_items)


@pytest.fixture
def conv_store():
    return ConversationStore()
