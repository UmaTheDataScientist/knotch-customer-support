from __future__ import annotations

import json

from app.agents.orchestrator import SupportOrchestrator
from app.config import Settings
from app.core.llm_client import LLMResult
from app.retrieval.embeddings import EmbeddingIndex
from tests.conftest import RecordingLLMClient


class _FailFirstVerifyClient(RecordingLLMClient):
    """Records every call like RecordingLLMClient, forces the first PLAN
    call to return a specific, scripted tool choice (rather than relying on
    FakeLLMClient's default routing, so this test doesn't depend on which
    tools happen to exist), forces the FIRST verification to fail with a
    specific reason, then behaves normally from the second attempt on.

    This is what lets us prove, offline, that the planner's second attempt
    actually receives and reacts to the failure -- something the real bug
    report showed wasn't happening (the same failing plan was repeated 6
    times)."""

    FAILURE_REASON = "The answer does not address what the user actually asked."
    # Deliberately NOT general_knowledge_lookup: _force_search_faq_first
    # (app/agents/graph.py) rewrites that tool to search_faq on the first
    # attempt, which would make this test assert on the override's tool
    # name instead of the replan-feedback behavior it actually targets.
    FIRST_TOOL = "get_faq_by_category"

    def __init__(self):
        super().__init__()
        self._verify_call_count = 0
        self._plan_call_count = 0

    def chat(self, messages, *, json_mode: bool = False, temperature: float = 0.2) -> LLMResult:
        self.calls.append(messages)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")

        if "planning step" in system:
            self._plan_call_count += 1
            if self._plan_call_count == 1:
                return LLMResult(
                    content=json.dumps(
                        {
                            "sub_requests": [
                                {
                                    "intent": "scripted_first_attempt",
                                    "tool": self.FIRST_TOOL,
                                    "tool_args": {"category": "billing"},
                                    "reasoning": "scripted first attempt for this test",
                                }
                            ]
                        }
                    ),
                    model="fake",
                    prompt_tokens=10,
                    completion_tokens=10,
                )
            # Second and later plan calls: fall through to default routing
            # (a plain support question), which will resolve fine now that
            # verification is set to pass on later attempts below.
            return super(RecordingLLMClient, self).chat(messages, json_mode=json_mode, temperature=temperature)

        if "verification step" in system:
            self._verify_call_count += 1
            if self._verify_call_count == 1:
                return LLMResult(
                    content=json.dumps(
                        {
                            "addresses_question": False,
                            "grounded": True,
                            "leaks_internals": False,
                            "reasoning": self.FAILURE_REASON,
                        }
                    ),
                    model="fake",
                    prompt_tokens=10,
                    completion_tokens=10,
                )
        # Fall through to the normal FakeLLMClient behavior for everything
        # else (including later verify calls, which will pass).
        return super(RecordingLLMClient, self).chat(messages, json_mode=json_mode, temperature=temperature)


def test_replan_includes_the_previous_failure_reason_and_tool(faq_items, tmp_path):
    """Regression test for a real bug found via live testing: the replan
    loop computed verification_reasoning but never passed it back into the
    next plan attempt, so a failing plan was retried identically (same
    tool, same reasoning) until the iteration budget ran out and forced an
    unnecessary human escalation. This test proves the second plan call's
    messages actually reference the first attempt's tool and failure
    reason."""
    llm = _FailFirstVerifyClient()
    index = EmbeddingIndex(llm, cache_path=tmp_path / "cache.json")
    index.build(faq_items)
    settings = Settings(llm_provider="fake", max_agent_iterations=4, max_verification_retries=2, faq_top_k=3, faq_min_score=0.05)
    orchestrator = SupportOrchestrator(llm=llm, settings=settings, index=index, faq_items=faq_items)

    from app.core.state import ConversationStore

    conv = ConversationStore().get_or_create("replan-feedback-1")
    orchestrator.handle_message(conv, "a plain support question")

    plan_calls = llm.plan_calls()
    assert len(plan_calls) >= 2, "expected at least 2 plan calls (initial + at least one replan)"

    second_call_messages = plan_calls[1]
    feedback_messages = [
        m["content"] for m in second_call_messages if m["role"] == "system" and "already tried this" in m["content"]
    ]
    assert feedback_messages, "second plan call should include feedback about the first attempt's failure"
    assert _FailFirstVerifyClient.FAILURE_REASON in feedback_messages[0]
    assert _FailFirstVerifyClient.FIRST_TOOL in feedback_messages[0]
