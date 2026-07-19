"""Ties the Compliance Agent and the main SupportAgentGraph together per turn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.agents.compliance import ComplianceAgent
from app.agents.graph import SupportAgentGraph
from app.agents.prompts import PROMPT_VERSION
from app.config import Settings
from app.core.llm_client import LLMClient
from app.core.review_queue import ReviewQueue, ReviewStatus
from app.core.state import ConversationState
from app.models.schemas import ChatMessageOut, MessageRole, ResponseSource, TraceStepType
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.knowledge_base import FAQItem
from app.tools.definitions import build_tools


class SupportOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        index: EmbeddingIndex,
        faq_items: list[FAQItem],
        review_queue: Optional[ReviewQueue] = None,
    ):
        self._llm = llm
        self._settings = settings
        self._compliance = ComplianceAgent(llm)
        self._tools = build_tools(index, llm, faq_items, settings.faq_top_k, settings.faq_min_score)
        self._review_queue = review_queue
        # Derived from the real KB, not hand-typed -- see
        # app.agents.prompts.build_plan_system_prompt for why.
        self._faq_categories = sorted({item.category for item in faq_items})

    def handle_message(self, conv: ConversationState, user_message: str) -> ChatMessageOut:
        conv.add_turn(MessageRole.USER, user_message)

        with conv.tracer.step(TraceStepType.COMPLIANCE_CHECK) as trace:
            verdict = self._compliance.check(user_message)
            trace.detail["verdict"] = {
                "safe": verdict.safe,
                "category": verdict.category,
                "reasoning": verdict.reasoning,
            }
            trace.detail["prompt_version"] = PROMPT_VERSION
            if verdict.llm_result:
                conv.tracer.record_llm_usage(
                    trace,
                    verdict.llm_result.prompt_tokens,
                    verdict.llm_result.completion_tokens,
                    verdict.llm_result.estimated_cost_usd,
                )

        if not verdict.safe:
            conv.add_turn(MessageRole.ASSISTANT, verdict.refusal_message)
            return ChatMessageOut(
                conversation_id=conv.conversation_id,
                response=verdict.refusal_message,
                source=ResponseSource.COMPLIANCE,
                tools_used=["refuse"],
                verified=True,
                iterations=0,
            )

        graph = SupportAgentGraph(
            self._llm, self._tools, self._settings, conv, self._faq_categories, self._review_queue
        )
        result = graph.run(user_message)

        response_text = result.get("final_response", "")
        conv.add_turn(MessageRole.ASSISTANT, response_text)
        conv.pending_clarification = result.get("short_circuit_tool") == "ask_user_clarification"

        return ChatMessageOut(
            conversation_id=conv.conversation_id,
            response=response_text,
            source=ResponseSource(result.get("source", ResponseSource.AGENT.value)),
            matched_questions=result.get("matched_questions", []),
            tools_used=result.get("tools_used", []),
            verified=bool(result.get("verified", False)),
            iterations=result.get("iteration", 1),
            pending_review_id=result.get("pending_review_id"),
        )

    def resume_review(self, conv: ConversationState, review_id: str) -> Optional[ChatMessageOut]:
        """Approves a paused review and actually executes what was queued
        (bonus: human-in-the-loop interrupt -- "resume after approval").
        Runs the same act/observe/verify node logic the normal turn would
        have used, directly, rather than replaying the whole plan step
        (the plan was already approved as-is; re-planning could produce a
        different plan than what the reviewer actually saw and signed off
        on)."""
        if self._review_queue is None:
            return None
        review = self._review_queue.get(review_id)
        if review is None or review.status != ReviewStatus.PENDING:
            return None

        graph = SupportAgentGraph(
            self._llm, self._tools, self._settings, conv, self._faq_categories, self._review_queue
        )
        state: dict = {
            "conversation_id": conv.conversation_id,
            "user_message": review.user_message,
            "sub_plans": review.sub_plans,
            "tools_used": [],
            "matched_questions": [],
            "iteration": 1,
            "verification_retries": 0,
            "done": False,
        }
        state = graph._act_node(state)  # noqa: SLF001 - intentional direct node call, see docstring
        state = graph._observe_node(state)
        state = graph._verify_node(state)
        # Bounded, same budget as a normal turn's replan loop -- if the
        # approved plan's answer somehow fails verification, retry a couple
        # of times rather than looping forever, then finalize regardless.
        retries = 0
        while not state.get("verified", True) and retries < self._settings.max_verification_retries:
            retries += 1
            state = graph._act_node(state)
            state = graph._observe_node(state)
            state = graph._verify_node(state)
        state = graph._finalize_node(state)

        response_text = state.get("final_response", "")
        self._review_queue.resolve(review_id, ReviewStatus.APPROVED, resolution=response_text)
        conv.add_turn(MessageRole.ASSISTANT, response_text)

        return ChatMessageOut(
            conversation_id=conv.conversation_id,
            response=response_text,
            source=ResponseSource(state.get("source", ResponseSource.ESCALATION.value)),
            matched_questions=state.get("matched_questions", []),
            tools_used=state.get("tools_used", []),
            verified=bool(state.get("verified", False)),
            iterations=1,
        )

    def reject_review(self, conv: ConversationState, review_id: str) -> Optional[ChatMessageOut]:
        """Rejects a paused review without executing anything it queued."""
        if self._review_queue is None:
            return None
        review = self._review_queue.get(review_id)
        if review is None or review.status != ReviewStatus.PENDING:
            return None

        response_text = (
            "After review, we're not able to process this request through this channel. "
            "Please contact support directly for further help."
        )
        self._review_queue.resolve(review_id, ReviewStatus.REJECTED, resolution=response_text)
        conv.add_turn(MessageRole.ASSISTANT, response_text)

        return ChatMessageOut(
            conversation_id=conv.conversation_id,
            response=response_text,
            source=ResponseSource.AGENT,
            verified=True,
            iterations=1,
        )
