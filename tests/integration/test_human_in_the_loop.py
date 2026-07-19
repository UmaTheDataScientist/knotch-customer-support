from __future__ import annotations

from app.core.review_queue import ReviewStatus
from app.models.schemas import ResponseSource


def test_escalation_pauses_for_review_instead_of_executing_immediately(orchestrator_with_review, conv_store, review_queue):
    conv = conv_store.get_or_create("hitl-1")
    out = orchestrator_with_review.handle_message(conv, "my account has been compromised and hacked, help")

    # Nothing executed yet -- escalate_to_human should NOT be in tools_used,
    # unlike the no-review-queue path (covered in test_conversations.py).
    assert out.tools_used == []
    assert out.pending_review_id is not None
    assert out.source == ResponseSource.ESCALATION

    pending = review_queue.list_pending()
    assert len(pending) == 1
    assert pending[0].review_id == out.pending_review_id
    assert pending[0].status == ReviewStatus.PENDING


def test_approving_a_review_actually_executes_the_escalation(orchestrator_with_review, conv_store, review_queue):
    conv = conv_store.get_or_create("hitl-2")
    paused = orchestrator_with_review.handle_message(conv, "my account has been compromised and hacked, help")
    review_id = paused.pending_review_id

    result = orchestrator_with_review.resume_review(conv, review_id)

    assert result is not None
    assert "escalate_to_human" in result.tools_used
    assert result.source == ResponseSource.ESCALATION
    assert "ticket" in result.response.lower()

    resolved = review_queue.get(review_id)
    assert resolved.status == ReviewStatus.APPROVED
    assert resolved.resolution == result.response


def test_rejecting_a_review_does_not_execute_anything(orchestrator_with_review, conv_store, review_queue):
    conv = conv_store.get_or_create("hitl-3")
    paused = orchestrator_with_review.handle_message(conv, "my account has been compromised and hacked, help")
    review_id = paused.pending_review_id

    result = orchestrator_with_review.reject_review(conv, review_id)

    assert result is not None
    assert result.tools_used == []
    assert "not able to process" in result.response.lower()

    resolved = review_queue.get(review_id)
    assert resolved.status == ReviewStatus.REJECTED


def test_approving_an_already_resolved_review_returns_none(orchestrator_with_review, conv_store):
    conv = conv_store.get_or_create("hitl-4")
    paused = orchestrator_with_review.handle_message(conv, "my account has been compromised and hacked, help")
    review_id = paused.pending_review_id

    first = orchestrator_with_review.resume_review(conv, review_id)
    assert first is not None

    second = orchestrator_with_review.resume_review(conv, review_id)
    assert second is None


def test_without_a_review_queue_escalation_still_executes_immediately(orchestrator, conv_store):
    """The gate is opt-in: an orchestrator with no review_queue configured
    (the default) behaves exactly as before -- no pause, immediate
    execution. This is what the pre-existing escalation tests in
    test_conversations.py rely on."""
    conv = conv_store.get_or_create("hitl-no-queue")
    out = orchestrator.handle_message(conv, "my account has been compromised and hacked, help")

    assert out.pending_review_id is None
    assert "escalate_to_human" in out.tools_used
