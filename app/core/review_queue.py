"""In-memory human-in-the-loop review queue (bonus #6: human-in-the-loop
interrupt).

Mocks the "reviewer" side of a pause/approve/resume flow, exactly as the
assignment describes: "the agent can pause, surface its plan to a human
reviewer, and resume after approval." When the planner produces a
sub-request that resolves to `escalate_to_human`, the graph pauses instead
of executing that tool immediately -- see `_route_after_plan` in
app/agents/graph.py. A human (or, in this mock, anyone calling the
/reviews endpoints) can inspect the paused plan and approve or reject it;
approving actually resumes execution.

This is process-wide, not per-conversation, since a human reviewer works
across every paused conversation, not one at a time -- same reasoning as
why ConversationStore is a single shared instance.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class PendingReview:
    review_id: str
    conversation_id: str
    user_message: str
    sub_plans: list[dict[str, Any]]
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolution: Optional[str] = None  # the final response text, once resumed/rejected


class ReviewQueue:
    def __init__(self):
        self._reviews: dict[str, PendingReview] = {}
        self._lock = threading.Lock()

    def enqueue(self, conversation_id: str, user_message: str, sub_plans: list[dict[str, Any]]) -> PendingReview:
        review = PendingReview(
            review_id=uuid.uuid4().hex[:8],
            conversation_id=conversation_id,
            user_message=user_message,
            sub_plans=sub_plans,
        )
        with self._lock:
            self._reviews[review.review_id] = review
        return review

    def get(self, review_id: str) -> Optional[PendingReview]:
        return self._reviews.get(review_id)

    def list_pending(self) -> list[PendingReview]:
        with self._lock:
            return [r for r in self._reviews.values() if r.status == ReviewStatus.PENDING]

    def resolve(self, review_id: str, status: ReviewStatus, resolution: Optional[str] = None) -> Optional[PendingReview]:
        with self._lock:
            review = self._reviews.get(review_id)
            if review is None:
                return None
            review.status = status
            review.resolution = resolution
            return review
