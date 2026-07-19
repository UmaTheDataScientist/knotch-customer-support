"""Ties the Compliance Agent and the main SupportAgentGraph together per turn."""
from __future__ import annotations

from dataclasses import dataclass

from app.agents.compliance import ComplianceAgent
from app.agents.graph import SupportAgentGraph
from app.agents.prompts import PROMPT_VERSION
from app.config import Settings
from app.core.llm_client import LLMClient
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
    ):
        self._llm = llm
        self._settings = settings
        self._compliance = ComplianceAgent(llm)
        self._tools = build_tools(index, llm, faq_items, settings.faq_top_k, settings.faq_min_score)
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

        graph = SupportAgentGraph(self._llm, self._tools, self._settings, conv, self._faq_categories)
        result = graph.run(user_message)

        response_text = result.get("final_response", "")
        conv.add_turn(MessageRole.ASSISTANT, response_text)
        conv.pending_clarification = result.get("tool_name") == "ask_user_clarification"

        return ChatMessageOut(
            conversation_id=conv.conversation_id,
            response=response_text,
            source=ResponseSource(result.get("source", ResponseSource.AGENT.value)),
            matched_questions=result.get("matched_questions", []),
            tools_used=result.get("tools_used", []),
            verified=bool(result.get("verified", False)),
            iterations=result.get("iteration", 1),
        )
