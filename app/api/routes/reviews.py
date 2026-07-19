from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agents.orchestrator import SupportOrchestrator
from app.core.dependencies import get_conversation_store, get_orchestrator, get_review_queue
from app.core.review_queue import ReviewQueue
from app.core.state import ConversationStore
from app.models.schemas import ChatMessageOut, PendingReviewOut, ReviewListOut

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("", response_model=ReviewListOut)
async def list_reviews(
    queue: ReviewQueue = Depends(get_review_queue),
) -> ReviewListOut:
    """Lists every pending human-in-the-loop review -- the mock 'reviewer
    dashboard' the assignment asks for. A real reviewer (or, in this mock,
    anyone calling this endpoint) inspects the queued plan before it's
    allowed to execute."""
    pending = queue.list_pending()
    return ReviewListOut(
        reviews=[
            PendingReviewOut(
                review_id=r.review_id,
                conversation_id=r.conversation_id,
                user_message=r.user_message,
                sub_plans=r.sub_plans,
                status=r.status.value,
                created_at=r.created_at,
            )
            for r in pending
        ]
    )


@router.post("/{review_id}/approve", response_model=ChatMessageOut)
async def approve_review(
    review_id: str,
    queue: ReviewQueue = Depends(get_review_queue),
    store: ConversationStore = Depends(get_conversation_store),
    orchestrator: SupportOrchestrator = Depends(get_orchestrator),
) -> ChatMessageOut:
    """Approves a paused plan and actually resumes execution -- the agent
    was waiting on exactly this before it could act."""
    review = queue.get(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    conv = store.get(review.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation for this review no longer exists")

    result = orchestrator.resume_review(conv, review_id)
    if result is None:
        raise HTTPException(status_code=409, detail="Review is not pending (already resolved or unknown)")
    return result


@router.post("/{review_id}/reject", response_model=ChatMessageOut)
async def reject_review(
    review_id: str,
    queue: ReviewQueue = Depends(get_review_queue),
    store: ConversationStore = Depends(get_conversation_store),
    orchestrator: SupportOrchestrator = Depends(get_orchestrator),
) -> ChatMessageOut:
    """Rejects a paused plan -- nothing it queued gets executed."""
    review = queue.get(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    conv = store.get(review.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation for this review no longer exists")

    result = orchestrator.reject_review(conv, review_id)
    if result is None:
        raise HTTPException(status_code=409, detail="Review is not pending (already resolved or unknown)")
    return result
