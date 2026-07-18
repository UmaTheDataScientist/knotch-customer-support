from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agents.orchestrator import SupportOrchestrator
from app.core.dependencies import get_conversation_store, get_orchestrator
from app.core.state import ConversationStore
from app.models.schemas import ChatMessageIn, ChatMessageOut, ConversationTrace

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/{conversation_id}/messages", response_model=ChatMessageOut)
async def post_message(
    conversation_id: str,
    body: ChatMessageIn,
    store: ConversationStore = Depends(get_conversation_store),
    orchestrator: SupportOrchestrator = Depends(get_orchestrator),
) -> ChatMessageOut:
    conv = store.get_or_create(conversation_id)
    try:
        return orchestrator.handle_message(conv, body.message)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent failed to process message: {exc}") from exc


@router.get("/{conversation_id}/trace", response_model=ConversationTrace)
async def get_trace(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationTrace:
    conv = store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv.tracer.trace
