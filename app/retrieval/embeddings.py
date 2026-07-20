"""Embedding index over the FAQ knowledge base.

Bonus #9 (idempotent embedding management): each item's embedding is
cached on disk keyed by a content hash (question+category text). Re-running
`build()` only calls the embedder for rows whose hash changed, so a 30-row
KB doesn't re-embed on every process start, and editing one FAQ doesn't
burn API calls re-embedding the other 29.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.llm_client import LLMClient
from app.retrieval.knowledge_base import FAQItem


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class ScoredItem:
    item: FAQItem
    score: float


class EmbeddingIndex:
    def __init__(self, llm_client: LLMClient, cache_path: str | Path = "data/embedding_cache.json"):
        self._llm = llm_client
        self._cache_path = Path(cache_path)
        self._cache: dict[str, dict] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._items: dict[str, FAQItem] = {}
        if self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text())

    def build(self, items: list[FAQItem]) -> dict[str, int]:
        """Embed only items whose content changed since last run. Returns
        counts of {'reused': n, 'embedded': n} for observability/tests."""
        indexable = [i for i in items if i.indexed]
        to_embed: list[FAQItem] = []
        hashes: dict[str, str] = {}

        embed_model_id = self._llm.embed_model_id
        for item in indexable:
            text = item.embedding_text()
            h = _content_hash(text)
            hashes[item.id] = h
            cached = self._cache.get(item.id)
            if cached and cached.get("hash") == h and cached.get("embed_model") == embed_model_id:
                self._vectors[item.id] = np.array(cached["vector"], dtype=np.float32)
            else:
                to_embed.append(item)
            self._items[item.id] = item

        reused = len(indexable) - len(to_embed)
        if to_embed:
            vectors = self._llm.embed([i.embedding_text() for i in to_embed])
            for item, vec in zip(to_embed, vectors):
                self._vectors[item.id] = np.array(vec, dtype=np.float32)
                self._cache[item.id] = {"hash": hashes[item.id], "vector": vec, "embed_model": embed_model_id}
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache))

        return {"reused": reused, "embedded": len(to_embed)}

    def search(
        self, query: str, top_k: int = 3, category_filter: str | None = None, min_score: float = 0.0
    ) -> list[ScoredItem]:
        if not self._vectors:
            return []
        query_vec = np.array(self._llm.embed([query])[0], dtype=np.float32)
        results: list[ScoredItem] = []
        for item_id, vec in self._vectors.items():
            item = self._items[item_id]
            if category_filter and item.category != category_filter:
                continue
            score = self._cosine(query_vec, vec)
            if score >= min_score:
                results.append(ScoredItem(item=item, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
