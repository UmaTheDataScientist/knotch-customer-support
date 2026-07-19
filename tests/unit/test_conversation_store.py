from __future__ import annotations

from app.core.state import ConversationStore


def test_create_generates_a_new_conversation_each_time():
    store = ConversationStore()
    first = store.create()
    second = store.create()

    assert first.conversation_id != second.conversation_id
    assert store.get(first.conversation_id) is first
    assert store.get(second.conversation_id) is second


def test_create_ids_never_collide_across_many_calls():
    store = ConversationStore()
    ids = {store.create().conversation_id for _ in range(500)}
    assert len(ids) == 500
