"""Pydantic schemas for API I/O and internal trace records.

Kept separate from ORM/state classes (app.core.state) on purpose: API
contracts and internal working state evolve at different rates, and
conflating them is how you end up leaking internal fields to clients.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ResponseSource(str, Enum):
    FAQ = "faq"
    AGENT = "agent"
    COMPLIANCE = "compliance"
    ESCALATION = "escalation"
    GENERAL_KNOWLEDGE = "general_knowledge"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ConversationCreateOut(BaseModel):
    conversation_id: str


class ChatMessageOut(BaseModel):
    conversation_id: str
    response: str
    source: ResponseSource
    matched_questions: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    verified: bool
    iterations: int = 1


class ConversationTurnOut(BaseModel):
    role: MessageRole
    content: str


class ConversationHistoryOut(BaseModel):
    conversation_id: str
    turns: list[ConversationTurnOut]


class ConversationSummaryOut(BaseModel):
    conversation_id: str
    turn_count: int
    first_message_preview: str = ""


class ConversationListOut(BaseModel):
    conversations: list[ConversationSummaryOut]


class TraceStepType(str, Enum):
    COMPLIANCE_CHECK = "compliance_check"
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    OBSERVE = "observe"
    VERIFY = "verify"
    REPLAN = "replan"
    FINAL_RESPONSE = "final_response"


class TraceStep(BaseModel):
    step_type: TraceStepType
    timestamp: datetime = Field(default_factory=_utcnow)
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ConversationTrace(BaseModel):
    conversation_id: str
    steps: list[TraceStep] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
