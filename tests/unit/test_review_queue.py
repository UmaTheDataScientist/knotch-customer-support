from __future__ import annotations

from app.core.review_queue import ReviewQueue, ReviewStatus


def test_enqueue_creates_a_pending_review():
    queue = ReviewQueue()
    review = queue.enqueue("conv-1", "escalate this please", [{"tool": "escalate_to_human"}])

    assert review.status == ReviewStatus.PENDING
    assert review.conversation_id == "conv-1"
    assert review.resolution is None
    assert len(review.review_id) > 0


def test_list_pending_only_shows_unresolved_reviews():
    queue = ReviewQueue()
    r1 = queue.enqueue("conv-1", "msg1", [])
    r2 = queue.enqueue("conv-2", "msg2", [])

    assert {r.review_id for r in queue.list_pending()} == {r1.review_id, r2.review_id}

    queue.resolve(r1.review_id, ReviewStatus.APPROVED, resolution="done")
    pending_ids = {r.review_id for r in queue.list_pending()}
    assert pending_ids == {r2.review_id}


def test_resolve_unknown_review_id_returns_none():
    queue = ReviewQueue()
    assert queue.resolve("does-not-exist", ReviewStatus.APPROVED) is None


def test_get_returns_the_review_with_its_resolution_after_resolving():
    queue = ReviewQueue()
    review = queue.enqueue("conv-1", "msg", [])
    queue.resolve(review.review_id, ReviewStatus.REJECTED, resolution="not approved")

    fetched = queue.get(review.review_id)
    assert fetched.status == ReviewStatus.REJECTED
    assert fetched.resolution == "not approved"
