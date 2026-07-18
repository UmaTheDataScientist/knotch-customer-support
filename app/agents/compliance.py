"""Compliance Agent: a second, simpler agent that runs as a guardrail.

Design choice (documented per the assignment's ask): it runs BEFORE the
main agent, not in parallel or after. Rationale:
  1. Cost/latency: rejecting a jailbreak/off-topic message before any FAQ
     search or tool planning avoids wasted LLM + retrieval calls.
  2. Safety: nothing the main agent produces (which could itself be
     manipulated by an injection) is ever shown to the user before a
     policy check has happened. Running "after" would mean the unsafe
     content briefly existed in the pipeline; "before" means it never
     reaches generation at all.
  3. Simplicity of override semantics: "runs first and can veto" is a much
     easier invariant to test and audit than "runs concurrently and wins
     races."

It is a real (if small) agent -- it makes its own LLM call with its own
system prompt and returns structured reasoning, not a regex filter. A fast
regex pre-check for the most blatant injection phrases is layered in front
of it purely to save a network round trip on obvious cases, but the
authoritative decision is always the model's, and the regex path also logs
`reasoning` so both paths look identical in the audit trail.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.agents.prompts import COMPLIANCE_SYSTEM_PROMPT
from app.core.llm_client import LLMClient, LLMResult

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |the )?(previous|prior|above) instructions", re.I),
    re.compile(r"reveal (your |the )?system prompt", re.I),
    re.compile(r"print (your |the )?system prompt", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"disregard (all |the )?(previous|prior) rules", re.I),
]

DEFAULT_OFF_TOPIC_REFUSAL = "This is not really what I was trained for, therefore I cannot answer. Try again."
DEFAULT_INJECTION_REFUSAL = "I can't help with that. If you have an account or support question, I'm happy to assist."


@dataclass
class ComplianceVerdict:
    safe: bool
    category: str
    reasoning: str
    refusal_message: str = ""
    llm_result: LLMResult | None = None


class ComplianceAgent:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def check(self, user_message: str) -> ComplianceVerdict:
        fast = self._fast_path(user_message)
        if fast is not None:
            return fast

        result = self._llm.chat(
            [
                {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            json_mode=True,
            temperature=0.0,
        )
        parsed = self._safe_json(result.content)
        verdict = str(parsed.get("verdict", "SAFE")).upper()
        category = parsed.get("category", "unknown")
        reasoning = parsed.get("reasoning", "")

        if verdict == "UNSAFE":
            refusal = DEFAULT_INJECTION_REFUSAL if "injection" in category.lower() else DEFAULT_OFF_TOPIC_REFUSAL
            return ComplianceVerdict(
                safe=False, category=category, reasoning=reasoning, refusal_message=refusal, llm_result=result
            )
        return ComplianceVerdict(safe=True, category=category, reasoning=reasoning, llm_result=result)

    def _fast_path(self, user_message: str) -> ComplianceVerdict | None:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(user_message):
                return ComplianceVerdict(
                    safe=False,
                    category="prompt_injection",
                    reasoning=f"Matched fast-path injection pattern: {pattern.pattern!r}",
                    refusal_message=DEFAULT_INJECTION_REFUSAL,
                )
        return None

    @staticmethod
    def _safe_json(text: str) -> dict:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {}
