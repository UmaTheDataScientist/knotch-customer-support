from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agents.orchestrator import SupportOrchestrator
from app.core.dependencies import get_conversation_store, get_orchestrator
from app.core.state import ConversationStore
from app.models.schemas import (
    ChatMessageIn,
    ChatMessageOut,
    ConversationCreateOut,
    ConversationHistoryOut,
    ConversationListOut,
    ConversationSummaryOut,
    ConversationTrace,
    ConversationTurnOut,
    MessageRole,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationCreateOut)
async def create_conversation(
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationCreateOut:
    """Creates a new conversation with a server-generated, guaranteed-unique
    id. Use this when starting a fresh conversation; use
    POST /{conversation_id}/messages directly (with your own chosen id) only
    when you specifically want to resume or address a known id."""
    conv = store.create()
    return ConversationCreateOut(conversation_id=conv.conversation_id)


@router.get("", response_model=ConversationListOut)
async def list_conversations(
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationListOut:
    """Lists every conversation currently held in memory. A debugging/
    discovery aid, not a production listing API -- there's no persistence,
    pagination, or ordering by recency here (see ConversationStore.list_ids)."""
    summaries = []
    for conv_id in store.list_ids():
        conv = store.get(conv_id)
        if conv is None:
            continue
        first_user_turn = next((t.content for t in conv.turns if t.role == MessageRole.USER), "")
        summaries.append(
            ConversationSummaryOut(
                conversation_id=conv_id,
                turn_count=len(conv.turns),
                first_message_preview=first_user_turn[:80],
            )
        )
    return ConversationListOut(conversations=summaries)


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


@router.get("/{conversation_id}/messages", response_model=ConversationHistoryOut)
async def get_messages(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationHistoryOut:
    """Returns the raw message history for a conversation, so a client can
    resume/redisplay an existing conversation by id rather than only seeing
    its trace. Separate from /trace since trace entries don't store the raw
    user/assistant text as a labeled field -- this is the source of truth
    for "what was actually said," /trace is "what the agent did."""
    conv = store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryOut(
        conversation_id=conversation_id,
        turns=[ConversationTurnOut(role=t.role, content=t.content) for t in conv.turns],
    )
