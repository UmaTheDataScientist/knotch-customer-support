"""Conversation state: turns, pending clarification, and a rolling summary.

Context window management: rather than concatenating every turn forever,
we keep the last `max_turns_in_context` raw turns verbatim and collapse
anything older into a single running summary string. The summary is
regenerated (cheaply, via a short LLM call) only when a turn ages out --
not on every single message.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver

from app.core.llm_client import LLMClient
from app.models.schemas import MessageRole
from app.observability.tracer import Tracer


@dataclass
class Turn:
    role: MessageRole
    content: str


@dataclass
class ConversationState:
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    pending_clarification: bool = False
    tracer: Tracer = field(default=None)  # type: ignore[assignment]
    checkpointer: Any = field(default=None)

    def __post_init__(self):
        if self.tracer is None:
            self.tracer = Tracer(self.conversation_id)
        if self.checkpointer is None:
            # One checkpointer per conversation, reused across every turn's
            # graph instance (a fresh SupportAgentGraph is built per message,
            # but they all share this same checkpointer object) so LangGraph
            # actually persists state snapshots under this conversation's
            # thread_id, rather than each turn starting a checkpointer with
            # no history of the previous one.
            self.checkpointer = MemorySaver()

    def add_turn(self, role: MessageRole, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    def context_messages(self, max_turns: int, llm: Optional[LLMClient] = None) -> list[dict[str, str]]:
        """Return chat-format messages: an optional summary system message
        followed by the most recent `max_turns` raw turns."""
        if len(self.turns) > max_turns:
            aged_out = self.turns[: -max_turns]
            recent = self.turns[-max_turns:]
            if llm is not None and aged_out:
                self.summary = self._summarize(aged_out, llm)
        else:
            recent = self.turns

        messages: list[dict[str, str]] = []
        if self.summary:
            messages.append({"role": "system", "content": f"Summary of earlier conversation: {self.summary}"})
        messages.extend({"role": t.role.value, "content": t.content} for t in recent)
        return messages

    def _summarize(self, turns: list[Turn], llm: LLMClient) -> str:
        transcript = "\n".join(f"{t.role.value}: {t.content}" for t in turns)
        prompt = (
            "Summarize the key facts and unresolved questions from this support "
            f"conversation excerpt in 2-3 sentences:\n\n{transcript}"
        )
        result = llm.chat([{"role": "user", "content": prompt}])
        return result.content


class ConversationStore:
    """Thread-safe in-memory store. Swap for Redis/Postgres by implementing
    the same get_or_create/get interface."""

    def __init__(self):
        self._states: dict[str, ConversationState] = {}
        self._lock = threading.Lock()

    def get_or_create(self, conversation_id: str) -> ConversationState:
        with self._lock:
            if conversation_id not in self._states:
                self._states[conversation_id] = ConversationState(conversation_id=conversation_id)
            return self._states[conversation_id]

    def create(self) -> ConversationState:
        """Generates a fresh, server-guaranteed-unique conversation id,
        rather than trusting a caller to supply a collision-free one
        (get_or_create is fine when the caller genuinely wants to resume a
        specific known id, but a client picking its own id for a *new*
        conversation -- e.g. Math.random() in a browser -- has no real
        uniqueness guarantee)."""
        with self._lock:
            while True:
                candidate = uuid.uuid4().hex[:12]
                if candidate not in self._states:
                    self._states[candidate] = ConversationState(conversation_id=candidate)
                    return self._states[candidate]

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        return self._states.get(conversation_id)

    def list_ids(self) -> list[str]:
        """Returns every conversation id currently held in memory. There is
        no persistence, ordering, or pagination here on purpose -- this is a
        thin debugging/discovery aid, not a production listing API (which
        would need created/updated timestamps and a real backing store)."""
        with self._lock:
            return list(self._states.keys())
