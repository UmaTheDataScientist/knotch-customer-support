from __future__ import annotations

from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.knowledge_base import load_faq_items


def test_ambiguous_x_entry_excluded_from_index(faq_items):
    x_item = next(i for i in faq_items if i.question == "x")
    assert x_item.indexed is False


def test_noisy_entry_still_indexed_and_normalized(faq_items):
    noisy = next(i for i in faq_items if "help!!!" in i.question)
    assert noisy.indexed is True
    assert "!!!" not in noisy.normalized_question


def test_locked_account_entry_has_real_guidance_not_the_original_broken_answer(faq_items):
    """Regression test: the source data's answer for this entry was
    'pls help me unlock it ASAP!!!' -- another frustrated statement in the
    same voice as the question, not actual guidance. Confirmed via live
    testing that it could be retrieved and handed to a user verbatim as
    the "answer" to their own question. This checks the corrected content
    directly, independent of embedding/ranking behavior."""
    locked = next(i for i in faq_items if "locked" in i.question.lower())
    assert "pls help me unlock it asap" not in locked.answer.lower()
    assert "asap!!!" not in locked.answer.lower()
    # It should read as real guidance, not another complaint.
    assert any(word in locked.answer.lower() for word in ("wait", "contact", "support", "login"))


def test_all_other_items_indexed(faq_items):
    non_flagged = [i for i in faq_items if i.question not in {"x"}]
    assert all(i.indexed for i in non_flagged)


def test_search_returns_relevant_password_faq(embedding_index):
    results = embedding_index.search("I forgot my password, how do I reset it?", top_k=3)
    assert results, "expected at least one match"
    top_questions = [r.item.question for r in results]
    assert any("password" in q.lower() for q in top_questions)


def test_search_respects_category_filter(embedding_index):
    results = embedding_index.search("change something", top_k=5, category_filter="billing")
    assert all(r.item.category == "billing" for r in results)


def test_embedding_cache_is_idempotent(scripted_llm, faq_items, tmp_path):
    cache_path = tmp_path / "cache.json"
    idx1 = EmbeddingIndex(scripted_llm, cache_path=cache_path)
    stats1 = idx1.build(faq_items)
    assert stats1["embedded"] > 0
    assert stats1["reused"] == 0

    # Re-running against the same unchanged data should reuse everything.
    idx2 = EmbeddingIndex(scripted_llm, cache_path=cache_path)
    stats2 = idx2.build(faq_items)
    assert stats2["embedded"] == 0
    assert stats2["reused"] == stats1["embedded"]
