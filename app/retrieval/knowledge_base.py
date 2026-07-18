"""Loads and cleans the FAQ dataset.

Cleaning decisions (documented here, not just in the README, so the
reasoning travels with the code):

1. Entries with a `flagged` field in the source data are structurally
   different from real FAQ pairs -- they're either noise ("x") or
   emotionally-charged one-offs ("help!!! my account is locked") rather
   than stable knowledge. We don't delete them (data loss is worse than
   noise), but we exclude them from the *embedded, searchable* index.
2. "x" -> its "answer" is itself an instruction to ask for clarification,
   not a fact. If indexed, it would falsely match on literally any short
   query. This is exactly the `ask_user_clarification` tool's job at
   runtime, so we let the agent handle it dynamically instead of baking
   a fake KB entry for it.
3. "help!!! my account is locked" -> this *is* a real support case, just
   noisily formatted. We normalize (strip exclamation spam, lowercase
   sentiment) for the embedding text while preserving the original
   question/answer for display, so retrieval quality doesn't degrade due
   to formatting noise.
4. Everything else is well-formed and indexed as-is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FAQItem:
    id: str
    question: str
    answer: str
    category: str
    indexed: bool = True
    normalized_question: str = ""

    def embedding_text(self) -> str:
        base = self.normalized_question or self.question
        return f"{base} {self.category}"


_NOISE_MARKERS = ("!!!", "!!")


def _normalize(question: str) -> str:
    q = question
    for marker in _NOISE_MARKERS:
        q = q.replace(marker, ".")
    return " ".join(q.split()).strip()


def load_faq_items(path: str | Path) -> list[FAQItem]:
    raw = json.loads(Path(path).read_text())
    items: list[FAQItem] = []
    for row in raw["knowledge_base_items"]:
        flagged = row.get("flagged")
        indexed = flagged != "excluded_from_index_ambiguous_meta_entry"
        normalized = _normalize(row["question"]) if flagged else ""
        items.append(
            FAQItem(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                category=row["category"],
                indexed=indexed,
                normalized_question=normalized,
            )
        )
    return items
