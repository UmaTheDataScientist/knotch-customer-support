from __future__ import annotations

from app.agents.graph import SupportAgentGraph


def test_checkpointer_persists_real_state_snapshots(orchestrator, conv_store):
    """Bonus requirement check: 'implement the agent as an explicit graph
    with nodes, edges, and conditional transitions. Show cycles and
    checkpointing.' Cycles were already covered by the replan-loop tests;
    this test specifically proves checkpointing is functioning, not just
    that compile(checkpointer=...) was called without erroring. It asserts
    on the actual number of persisted state snapshots after a real run."""
    conv = conv_store.get_or_create("checkpoint-test-1")

    result = orchestrator.handle_message(conv, "How do I reset my password?")
    assert result.response

    graph = SupportAgentGraph(
        orchestrator._llm, orchestrator._tools, orchestrator._settings, conv, orchestrator._faq_categories
    )
    thread_id = f"{conv.conversation_id}:1"
    config = {"configurable": {"thread_id": thread_id}}

    history = list(graph._graph.get_state_history(config))  # noqa: SLF001 - intentionally inspecting internals

    # One snapshot per node transition through plan -> act -> observe ->
    # verify -> finalize, plus the initial __start__ snapshot.
    assert len(history) >= 6, f"expected at least 6 checkpointed snapshots, got {len(history)}"

    node_names = set()
    for snap in history:
        if isinstance(snap.metadata.get("writes"), dict):
            node_names.update(snap.metadata["writes"].keys())
    expected_nodes = {"plan_step", "act_step", "observe_step", "verify_step", "finalize_step"}
    assert expected_nodes.issubset(node_names), f"missing nodes in checkpoint history: {expected_nodes - node_names}"


def test_each_conversation_gets_its_own_checkpointer(conv_store):
    """Two different conversations should not share checkpoint history."""
    conv_a = conv_store.get_or_create("checkpoint-conv-a")
    conv_b = conv_store.get_or_create("checkpoint-conv-b")

    assert conv_a.checkpointer is not conv_b.checkpointer
