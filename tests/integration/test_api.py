from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.dependencies import get_conversation_store, get_orchestrator
from app.main import app


def _override_deps(app_, orchestrator, conv_store):
    app_.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app_.dependency_overrides[get_conversation_store] = lambda: conv_store


def test_post_message_and_get_trace(orchestrator, conv_store):
    _override_deps(app, orchestrator, conv_store)
    client = TestClient(app)

    resp = client.post(
        "/conversations/api-test-1/messages",
        json={"message": "How do I restore my account to its default settings?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "api-test-1"
    assert body["source"] == "faq"

    trace_resp = client.get("/conversations/api-test-1/trace")
    assert trace_resp.status_code == 200
    trace_body = trace_resp.json()
    assert trace_body["conversation_id"] == "api-test-1"
    assert len(trace_body["steps"]) > 0

    app.dependency_overrides.clear()


def test_get_trace_for_unknown_conversation_is_404(orchestrator, conv_store):
    _override_deps(app, orchestrator, conv_store)
    client = TestClient(app)

    resp = client.get("/conversations/does-not-exist/trace")
    assert resp.status_code == 404

    app.dependency_overrides.clear()


def test_empty_message_is_rejected(orchestrator, conv_store):
    _override_deps(app, orchestrator, conv_store)
    client = TestClient(app)

    resp = client.post("/conversations/api-test-2/messages", json={"message": ""})
    assert resp.status_code == 422  # pydantic min_length validation

    app.dependency_overrides.clear()


def test_health_check():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_messages_returns_full_turn_history(orchestrator, conv_store):
    _override_deps(app, orchestrator, conv_store)
    client = TestClient(app)

    client.post("/conversations/history-test/messages", json={"message": "x"})
    client.post("/conversations/history-test/messages", json={"message": "i forgot my password"})

    resp = client.get("/conversations/history-test/messages")
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "history-test"
    assert len(body["turns"]) == 4  # user, assistant, user, assistant
    assert body["turns"][0]["role"] == "user"
    assert body["turns"][0]["content"] == "x"
    assert body["turns"][1]["role"] == "assistant"

    app.dependency_overrides.clear()


def test_get_messages_for_unknown_conversation_is_404(orchestrator, conv_store):
    _override_deps(app, orchestrator, conv_store)
    client = TestClient(app)

    resp = client.get("/conversations/does-not-exist/messages")
    assert resp.status_code == 404

    app.dependency_overrides.clear()


def test_list_conversations_is_empty_initially(orchestrator, conv_store):
    _override_deps(app, orchestrator, conv_store)
    client = TestClient(app)

    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert resp.json() == {"conversations": []}

    app.dependency_overrides.clear()


def test_list_conversations_shows_active_ones_with_preview(orchestrator, conv_store):
    _override_deps(app, orchestrator, conv_store)
    client = TestClient(app)

    client.post("/conversations/list-test-1/messages", json={"message": "How do I reset my password?"})
    client.post("/conversations/list-test-2/messages", json={"message": "x"})

    resp = client.get("/conversations")
    assert resp.status_code == 200
    body = resp.json()
    ids = {c["conversation_id"] for c in body["conversations"]}
    assert ids == {"list-test-1", "list-test-2"}

    conv1 = next(c for c in body["conversations"] if c["conversation_id"] == "list-test-1")
    assert conv1["turn_count"] == 2  # user + assistant
    assert conv1["first_message_preview"] == "How do I reset my password?"

    app.dependency_overrides.clear()
