"""Tool schemas + implementations.

Each tool is a plain Python callable plus a Pydantic schema describing its
arguments. The schema is what the LLM sees (as a JSON-schema tool spec via
`.tool_spec()`); the callable is what the graph actually invokes. Keeping
these paired but separate from any single agent framework's tool class
means the same tools are reusable if we ever swap LangGraph for a plain
ReAct loop (bonus criterion: "tool definitions reusable across different
agent implementations").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field

from app.core.llm_client import LLMClient
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.knowledge_base import FAQItem


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------


class SearchFaqArgs(BaseModel):
    query: str = Field(..., description="The user's question, rephrased if needed for search.")
    category_filter: Optional[str] = Field(
        None, description="Restrict to a category if the user's intent is already known."
    )


class GetFaqByCategoryArgs(BaseModel):
    category: str = Field(..., description="Category name, e.g. 'billing', 'security'.")


class AskUserClarificationArgs(BaseModel):
    question: str = Field(..., description="A specific follow-up question to ask the user.")


class GeneralKnowledgeLookupArgs(BaseModel):
    query: str = Field(..., description="A general question outside the FAQ knowledge base.")


class EscalateToHumanArgs(BaseModel):
    reason: str = Field(..., description="Why this needs a human.")
    transcript: str = Field(..., description="Relevant conversation transcript excerpt.")


class RefuseArgs(BaseModel):
    reason: str = Field(..., description="Why the request is being refused.")


@dataclass
class ToolResult:
    tool_name: str
    output: dict[str, Any]


@dataclass
class Tool:
    name: str
    description: str
    args_schema: Type[BaseModel]
    func: Callable[..., ToolResult]

    def tool_spec(self) -> dict[str, Any]:
        """OpenAI/Anthropic-style function-calling JSON schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }

    def run(self, **kwargs) -> ToolResult:
        validated = self.args_schema(**kwargs)
        return self.func(validated)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def build_tools(
    index: EmbeddingIndex, llm: LLMClient, all_items: list[FAQItem], faq_top_k: int, faq_min_score: float
) -> dict[str, Tool]:
    def _search_faq(args: SearchFaqArgs) -> ToolResult:
        results = index.search(
            args.query, top_k=faq_top_k, category_filter=args.category_filter, min_score=faq_min_score
        )
        return ToolResult(
            tool_name="search_faq",
            output={
                "matches": [
                    {"question": r.item.question, "answer": r.item.answer, "category": r.item.category, "score": round(r.score, 4)}
                    for r in results
                ]
            },
        )

    def _ask_user_clarification(args: AskUserClarificationArgs) -> ToolResult:
        return ToolResult(tool_name="ask_user_clarification", output={"question": args.question})

    def _general_knowledge_lookup(args: GeneralKnowledgeLookupArgs) -> ToolResult:
        result = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a support assistant answering a general question that is not "
                        "covered by the FAQ knowledge base. Be concise, factual, and clearly note "
                        "this is general knowledge, not official policy. Respond in plain text only "
                        "-- no Markdown formatting (no **bold**, no bullet/numbered lists with "
                        "Markdown syntax, no headers). This text is displayed as a raw string, not "
                        "rendered through a Markdown renderer."
                    ),
                },
                {"role": "user", "content": args.query},
            ]
        )
        return ToolResult(
            tool_name="general_knowledge_lookup",
            output={"answer": result.content, "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens},
        )

    def _escalate_to_human(args: EscalateToHumanArgs) -> ToolResult:
        # Mock: log and return a stub. A real implementation would push to a
        # queue (e.g. Zendesk/PagerDuty) and return a ticket id.
        return ToolResult(
            tool_name="escalate_to_human",
            output={"status": "escalated", "reason": args.reason, "ticket_id": "TICKET-STUB-0001"},
        )

    def _refuse(args: RefuseArgs) -> ToolResult:
        return ToolResult(tool_name="refuse", output={"reason": args.reason})

    def _get_faq_by_category(args: GetFaqByCategoryArgs) -> ToolResult:
        matches = [i for i in all_items if i.category == args.category]
        return ToolResult(
            tool_name="get_faq_by_category",
            output={"questions": [{"question": i.question, "answer": i.answer} for i in matches]},
        )

    tools = {
        "search_faq": Tool(
            name="search_faq",
            description="Semantic search over the FAQ knowledge base for a user question.",
            args_schema=SearchFaqArgs,
            func=_search_faq,
        ),
        "get_faq_by_category": Tool(
            name="get_faq_by_category",
            description="List all FAQ questions in a given category.",
            args_schema=GetFaqByCategoryArgs,
            func=_get_faq_by_category,
        ),
        "ask_user_clarification": Tool(
            name="ask_user_clarification",
            description="Ask the user a follow-up question when their request is ambiguous.",
            args_schema=AskUserClarificationArgs,
            func=_ask_user_clarification,
        ),
        "general_knowledge_lookup": Tool(
            name="general_knowledge_lookup",
            description="Answer a general question not covered by the FAQ, via a plain LLM call.",
            args_schema=GeneralKnowledgeLookupArgs,
            func=_general_knowledge_lookup,
        ),
        "escalate_to_human": Tool(
            name="escalate_to_human",
            description="Escalate to a human agent for sensitive or out-of-scope issues.",
            args_schema=EscalateToHumanArgs,
            func=_escalate_to_human,
        ),
        "refuse": Tool(
            name="refuse",
            description="Refuse to answer due to policy violation, jailbreak attempt, or off-topic request.",
            args_schema=RefuseArgs,
            func=_refuse,
        ),
    }
    return tools
