"""Homegrown structured trace logger.

Chosen over LangSmith/OTel for this submission because it has zero external
dependencies (no account, no collector to stand up) and the requirement is
just "pick one and use it consistently." Swapping this for an OTel exporter
later is a matter of implementing the same `record()` interface -- nothing
upstream needs to change since agent code only calls `Tracer.record`.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from app.models.schemas import ConversationTrace, TraceStep, TraceStepType


class Tracer:
    """One Tracer per conversation. Held by the in-memory conversation store
    and returned wholesale by GET /conversations/{id}/trace."""

    def __init__(self, conversation_id: str):
        self.trace = ConversationTrace(conversation_id=conversation_id)

    @contextmanager
    def step(self, step_type: TraceStepType, **detail: Any) -> Iterator[TraceStep]:
        start = time.perf_counter()
        record = TraceStep(step_type=step_type, detail=detail)
        try:
            yield record
        finally:
            record.latency_ms = (time.perf_counter() - start) * 1000
            self.trace.steps.append(record)
            self.trace.total_tokens += record.prompt_tokens + record.completion_tokens
            self.trace.total_cost_usd += record.estimated_cost_usd

    def record_llm_usage(self, step: TraceStep, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        step.prompt_tokens = prompt_tokens
        step.completion_tokens = completion_tokens
        step.estimated_cost_usd = cost_usd
        # Re-sum since these are mutated after being added inside the `step` context in some paths.
        self.trace.total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in self.trace.steps)
        self.trace.total_cost_usd = sum(s.estimated_cost_usd for s in self.trace.steps)
