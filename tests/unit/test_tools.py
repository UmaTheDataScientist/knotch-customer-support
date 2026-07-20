from __future__ import annotations

from app.tools.definitions import build_tools


def _tools(scripted_llm, embedding_index, faq_items):
    return build_tools(embedding_index, scripted_llm, faq_items, faq_top_k=3, faq_min_score=0.05)


def test_search_faq_tool_returns_matches(scripted_llm, embedding_index, faq_items):
    tools = _tools(scripted_llm, embedding_index, faq_items)
    result = tools["search_faq"].run(query="how do I reset my password")
    assert result.tool_name == "search_faq"
    assert isinstance(result.output["matches"], list)


def test_get_faq_by_category_tool(scripted_llm, embedding_index, faq_items):
    tools = _tools(scripted_llm, embedding_index, faq_items)
    result = tools["get_faq_by_category"].run(category="billing")
    questions = [q["question"] for q in result.output["questions"]]
    assert "Can I get a refund?" in questions


def test_ask_user_clarification_tool(scripted_llm, embedding_index, faq_items):
    tools = _tools(scripted_llm, embedding_index, faq_items)
    result = tools["ask_user_clarification"].run(question="What do you mean?")
    assert result.output["question"] == "What do you mean?"


def test_refuse_tool(scripted_llm, embedding_index, faq_items):
    tools = _tools(scripted_llm, embedding_index, faq_items)
    result = tools["refuse"].run(reason="off topic")
    assert result.output["reason"] == "off topic"


def test_escalate_to_human_tool_returns_stub_ticket(scripted_llm, embedding_index, faq_items):
    tools = _tools(scripted_llm, embedding_index, faq_items)
    result = tools["escalate_to_human"].run(reason="account compromised", transcript="user: help my account is hacked")
    assert result.output["status"] == "escalated"
    assert "ticket_id" in result.output


def test_tool_args_are_validated(scripted_llm, embedding_index, faq_items):
    tools = _tools(scripted_llm, embedding_index, faq_items)
    try:
        tools["get_faq_by_category"].run()  # missing required 'category'
        assert False, "expected validation error"
    except Exception:
        pass
