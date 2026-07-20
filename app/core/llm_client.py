"""Provider-agnostic LLM client.

Agent/graph code only ever talks to `LLMClient`. Swapping OpenAI for
Anthropic (or plugging in the deterministic `FakeLLMClient` for tests)
is a config change (`LLM_PROVIDER`), never an agent-logic change.

Each call returns an `LLMResult` carrying token counts so observability
can attribute cost per step without every call site re-deriving it.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings

# Client-side errors that will NEVER succeed on retry (wrong key, malformed
# request, etc). Retrying these just burns real API calls for no benefit --
# important when working against a rate-limited key. Anything else (rate
# limits, timeouts, transient 5xx) is assumed retryable, which is tenacity's
# default behavior when nothing is excluded.
_NON_RETRYABLE_ERRORS: tuple[type[Exception], ...] = ()
try:
    from openai import AuthenticationError as _OpenAIAuthError
    from openai import BadRequestError as _OpenAIBadRequestError
    from openai import NotFoundError as _OpenAINotFoundError
    from openai import PermissionDeniedError as _OpenAIPermissionError

    _NON_RETRYABLE_ERRORS += (_OpenAIAuthError, _OpenAIBadRequestError, _OpenAINotFoundError, _OpenAIPermissionError)
except ImportError:
    pass
try:
    from anthropic import AuthenticationError as _AnthropicAuthError
    from anthropic import BadRequestError as _AnthropicBadRequestError
    from anthropic import NotFoundError as _AnthropicNotFoundError
    from anthropic import PermissionDeniedError as _AnthropicPermissionError

    _NON_RETRYABLE_ERRORS += (
        _AnthropicAuthError,
        _AnthropicBadRequestError,
        _AnthropicNotFoundError,
        _AnthropicPermissionError,
    )
except ImportError:
    pass


def _llm_retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_not_exception_type(_NON_RETRYABLE_ERRORS),
        reraise=True,
    )

# Rough $/1K token estimates for cost tracking. Not billing-accurate;
# good enough for relative cost-awareness in traces/eval.
_COST_PER_1K = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-sonnet-4-6": (0.003, 0.015),
    "fake-model": (0.0, 0.0),
}


@dataclass
class LLMResult:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_json: Optional[dict[str, Any]] = None

    @property
    def estimated_cost_usd(self) -> float:
        in_rate, out_rate = _COST_PER_1K.get(self.model, (0.0005, 0.0015))
        return (self.prompt_tokens / 1000) * in_rate + (self.completion_tokens / 1000) * out_rate


class LLMClient(ABC):
    """Minimal surface every provider must implement."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResult:
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def embed_model_id(self) -> str:
        """Identifies which embedding source/model produced a vector.

        Used by EmbeddingIndex to invalidate cached vectors when the
        embedding source changes (e.g. switching LLM_PROVIDER between
        `fake` and `openai`), since vectors from different models/dimensions
        are not comparable and mixing them corrupts cosine similarity or
        crashes on a shape mismatch.
        """
        ...


class OpenAILLMClient(LLMClient):
    def __init__(self, settings: Settings):
        from openai import OpenAI  # local import: keeps `openai` optional for fake/test runs

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._embed_model = settings.openai_embed_model

    @_llm_retry()
    def chat(self, messages, *, json_mode: bool = False, temperature: float = 0.2) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResult(
            content=choice,
            model=self._chat_model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    @_llm_retry()
    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._embed_model, input=texts)
        return [d.embedding for d in resp.data]

    @property
    def embed_model_id(self) -> str:
        return f"openai:{self._embed_model}"


class AnthropicLLMClient(LLMClient):
    """Chat via Anthropic. Embeddings are not offered by Anthropic, so this
    client delegates embedding calls to OpenAI if a key is present -- this
    is exactly the kind of seam the abstraction is meant to expose cleanly
    rather than hide."""

    def __init__(self, settings: Settings):
        import anthropic  # local import, optional dependency

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model
        self._settings = settings
        self._embed_fallback: Optional[OpenAILLMClient] = (
            OpenAILLMClient(settings) if settings.openai_api_key else None
        )

    @_llm_retry()
    def chat(self, messages, *, json_mode: bool = False, temperature: float = 0.2) -> LLMResult:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self._model,
            system=system or None,
            messages=turns,
            max_tokens=1000,
            temperature=temperature,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResult(
            content=text,
            model=self._model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._embed_fallback:
            raise RuntimeError(
                "Anthropic has no embeddings API; set OPENAI_API_KEY as an embedding fallback."
            )
        return self._embed_fallback.embed(texts)

    @property
    def embed_model_id(self) -> str:
        if not self._embed_fallback:
            return "anthropic:no-embed-fallback"
        return self._embed_fallback.embed_model_id


class FakeLLMClient(LLMClient):
    """Deterministic, network-free client used in unit/integration tests and
    local dev without an API key. Chat responses are simple rule-based
    heuristics; embeddings are a stable hash-based projection so cosine
    similarity behaves sanely (same text -> same vector, similar text ->
    closer vectors isn't guaranteed, but exact/near-duplicate matching is,
    which is what our tests assert on)."""

    def chat(self, messages, *, json_mode: bool = False, temperature: float = 0.2) -> LLMResult:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        content = self._route(system, last_user, json_mode)
        return LLMResult(
            content=content,
            model="fake-model",
            prompt_tokens=sum(len(m["content"].split()) for m in messages),
            completion_tokens=len(content.split()),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    @property
    def embed_model_id(self) -> str:
        return "fake:hash-4096"

    _STOPWORDS = {
        "a", "an", "the", "is", "it", "to", "do", "i", "my", "how", "can", "are",
        "there", "for", "of", "in", "on", "what", "if", "and", "or", "be", "you",
        "your", "please", "help", "me", "any",
    }

    @classmethod
    def _hash_embed(cls, text: str, dim: int = 4096) -> list[float]:
        # High dimensionality keeps hash collisions rare enough that cosine
        # similarity tracks actual shared-token overlap -- good enough for a
        # deterministic, network-free stand-in used in dev/tests. A real
        # embedding model is used in production via OPENAI_EMBED_MODEL.
        normalized = text.lower().strip()
        vec = [0.0] * dim
        tokens = [t for t in normalized.replace("?", " ").replace("!", " ").split() if t not in cls._STOPWORDS]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    # -- rule-based routing, mirrors what a real model would decide -------
    _INJECTION_HINTS = ("ignore previous", "ignore all previous", "disregard all prior", "system prompt", "you are now")
    _OFF_TOPIC_HINTS = ("poem", "pirate", "weather", "joke", "sing a song", "trivia")
    _CATEGORY_OVERVIEW_HINTS = ("options do you support", "overview of")
    _KNOWN_CATEGORIES = (
        "billing", "security", "profile", "privacy", "subscription", "notifications",
        "troubleshooting", "developer", "settings", "account_lifecycle", "data_recovery",
        "security_incident",
    )
    _GENERAL_KNOWLEDGE_HINTS = ("rule of thumb", "conceptually", "in general,")

    @classmethod
    def _route(cls, system: str, user_text: str, json_mode: bool) -> str:
        text_low = user_text.lower().strip()

        if system.startswith("You are a Compliance Agent"):
            if any(h in text_low for h in cls._INJECTION_HINTS):
                return json.dumps({"verdict": "UNSAFE", "category": "prompt_injection", "reasoning": "injection phrasing detected"})
            if any(h in text_low for h in cls._OFF_TOPIC_HINTS):
                return json.dumps({"verdict": "UNSAFE", "category": "off_topic", "reasoning": "unrelated to account/product support"})
            return json.dumps({"verdict": "SAFE", "category": "support_question", "reasoning": "appears to be an on-topic support query"})

        if "planning step" in system:
            if len(text_low) <= 3 or text_low in {"help", "help!", "help!!!"}:
                return json.dumps(
                    {
                        "sub_requests": [
                            {
                                "intent": "ambiguous",
                                "tool": "ask_user_clarification",
                                "tool_args": {
                                    "question": "Could you tell me a bit more about what you're trying to do?"
                                },
                                "reasoning": "message too short to act on",
                            }
                        ]
                    }
                )
            if any(p in text_low for p in ("speak to a manager", "legal action", "lawsuit", "file a complaint against")):
                return json.dumps(
                    {
                        "sub_requests": [
                            {
                                "intent": "requires_human_authority",
                                "tool": "escalate_to_human",
                                "tool_args": {"reason": "requires human authority beyond FAQ scope", "transcript": user_text},
                                "reasoning": "no FAQ entry covers this; genuinely needs a human",
                            }
                        ]
                    }
                )
            if any(h in text_low for h in cls._CATEGORY_OVERVIEW_HINTS):
                category = next((c for c in cls._KNOWN_CATEGORIES if c in text_low), "billing")
                return json.dumps(
                    {
                        "sub_requests": [
                            {
                                "intent": "category_overview",
                                "tool": "get_faq_by_category",
                                "tool_args": {"category": category},
                                "reasoning": "user wants an overview of a topic area, not one specific question",
                            }
                        ]
                    }
                )
            if any(h in text_low for h in cls._GENERAL_KNOWLEDGE_HINTS):
                return json.dumps(
                    {
                        "sub_requests": [
                            {
                                "intent": "general_knowledge",
                                "tool": "general_knowledge_lookup",
                                "tool_args": {"query": user_text},
                                "reasoning": "support-adjacent general knowledge question with no plausible FAQ category coverage",
                            }
                        ]
                    }
                )
            # Naive multi-intent simulation: split on " and " and treat each
            # part as its own sub-request, so offline tests can genuinely
            # exercise decomposition without needing a real model. This is
            # deliberately simple (word-based, not true intent parsing) --
            # real decomposition quality depends on the actual LLM.
            if " and " in user_text.lower():
                raw_parts = [p.strip() for p in user_text.split(" and ") if p.strip()]
                if len(raw_parts) >= 2:
                    return json.dumps(
                        {
                            "sub_requests": [
                                {
                                    "intent": "faq_lookup",
                                    "tool": "search_faq",
                                    "tool_args": {"query": part},
                                    "reasoning": f"distinct sub-request identified: {part!r}",
                                }
                                for part in raw_parts
                            ]
                        }
                    )
            return json.dumps(
                {
                    "sub_requests": [
                        {
                            "intent": "faq_lookup",
                            "tool": "search_faq",
                            "tool_args": {"query": user_text},
                            "reasoning": "looks like a concrete support question worth checking the FAQ",
                        }
                    ]
                }
            )

        if "verification step" in system:
            return json.dumps({"addresses_question": True, "grounded": True, "leaks_internals": False, "reasoning": "heuristic pass"})

        if system.startswith("You are a customer support agent writing"):
            try:
                payload = json.loads(user_text)
            except (json.JSONDecodeError, TypeError):
                payload = {}
            matches = payload.get("kb_matches") or []
            question = (payload.get("user_question") or "").lower()
            if matches:
                top_answer = matches[0].get("answer", "")
                # Simulate the precondition-mismatch check: an answer that
                # requires the current password doesn't fit someone who
                # says they forgot it.
                if "forgot" in question and "current password" in top_answer.lower():
                    return (
                        "It looks like the steps we have on file for changing your password require "
                        "entering your current password first, which won't work if you've forgotten it. "
                        "Since I don't have a dedicated account-recovery process in the FAQ for this, "
                        "I'd recommend reaching out to support directly so they can help you regain access."
                    )
                return top_answer
            return "I couldn't find a specific match, but here's some general guidance."

        if "general question that is not" in system:
            return "This is general knowledge, not official policy: please check the relevant documentation for specifics."

        if "Summarize the key facts" in user_text:
            return "Earlier: user was troubleshooting an account-related issue."

        return "This is a fake LLM response for offline testing."


def build_llm_client(settings: Optional[Settings] = None) -> LLMClient:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAILLMClient(settings)
    if provider == "anthropic":
        return AnthropicLLMClient(settings)
    if provider == "fake":
        return FakeLLMClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
