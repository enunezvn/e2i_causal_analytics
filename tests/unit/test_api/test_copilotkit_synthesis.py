"""Unit coverage for the synthesize-node prompt builder.

``build_synthesis_prompt`` frames the user's question + the assistant's tool-call
args (brand/window) + the tool results so the synthesizer answers the ACTUAL
question and never re-asks for a brand/period the user already provided.

It also receives the prior conversation turns (``history``): the synthesizer is
a separate LLM call from chat_node, so without an explicit transcript it cannot
resolve follow-up references like "is that above baseline?" and answers
"I'm missing the preceding conversation" even though chat_node saw everything.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.api.routes.copilotkit import _extract_synthesis_history, build_synthesis_prompt


@pytest.mark.unit
def test_includes_question_and_args():
    p = build_synthesis_prompt(
        "NRx for Kisqali past 3 months",
        [
            {
                "name": "kpi_calculate_tool",
                "args": {"kpi_name": "NRx", "brand": "Kisqali", "window": "past 3 months"},
            }
        ],
        [{"tool": "kpi_calculate_tool", "result": '{"value": 3394, "window_status": "applied"}'}],
    )
    assert "Kisqali" in p and "past 3 months" in p and "3394" in p
    assert "User question" in p


@pytest.mark.unit
def test_handles_empty():
    p = build_synthesis_prompt("", [], [])
    assert isinstance(p, str) and "User question" in p


@pytest.mark.unit
def test_history_appears_before_question():
    """Follow-up questions must see the prior turns' established values."""
    p = build_synthesis_prompt(
        "is that above or below baseline?",
        [{"name": "e2i_data_query_tool", "args": {"brand": "Fabhalta", "kpi_name": "TRx"}}],
        [{"tool": "e2i_data_query_tool", "result": '{"count": 0}'}],
        history=[
            {"role": "user", "content": "what is the Fabhalta Trx in the past 30 days?"},
            {"role": "assistant", "content": "Fabhalta TRx last 30 days: **15,239**"},
        ],
    )
    assert "15,239" in p
    assert "Conversation so far" in p
    # History must precede the question so the model reads context first
    assert p.index("15,239") < p.index("is that above or below baseline?")


@pytest.mark.unit
def test_no_history_section_when_history_absent():
    p_none = build_synthesis_prompt("q", [], [], history=None)
    p_empty = build_synthesis_prompt("q", [], [], history=[])
    assert "Conversation so far" not in p_none
    assert "Conversation so far" not in p_empty


@pytest.mark.unit
def test_extract_history_prior_text_turns_only():
    """Everything before the last human message, text turns only: no tool
    messages, no tool-call stub AIMessages (empty content), no system."""
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="what is the Fabhalta Trx in the past 30 days?"),
        AIMessage(content="Fabhalta TRx last 30 days: 15,239"),
        HumanMessage(content="is that above or below baseline?"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content='{"count": 0}', tool_call_id="1"),
    ]
    history = _extract_synthesis_history(messages)
    assert history == [
        {"role": "user", "content": "what is the Fabhalta Trx in the past 30 days?"},
        {"role": "assistant", "content": "Fabhalta TRx last 30 days: 15,239"},
    ]


@pytest.mark.unit
def test_extract_history_first_turn_is_empty():
    messages = [
        HumanMessage(content="what is the Fabhalta Trx in the past 30 days?"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content='{"value": 15239}', tool_call_id="1"),
    ]
    assert _extract_synthesis_history(messages) == []


@pytest.mark.unit
def test_extract_history_handles_dict_messages():
    """chat_node tolerates raw dict messages in state; the extractor must too."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert _extract_synthesis_history(messages) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


@pytest.mark.unit
def test_extract_history_caps_length():
    """Long conversations are capped to the most recent turns and long
    messages truncated, so the synthesis prompt cannot grow unbounded."""
    messages = []
    for i in range(30):
        messages.append(HumanMessage(content=f"question {i} " + "x" * 3000))
        messages.append(AIMessage(content=f"answer {i}"))
    messages.append(HumanMessage(content="current question"))
    history = _extract_synthesis_history(messages)
    assert len(history) <= 12
    # Most recent prior turns are kept
    assert history[-1]["content"] == "answer 29"
    assert all(len(h["content"]) <= 2000 for h in history)


@pytest.mark.unit
def test_prompt_forbids_overlapping_window_baselines():
    """2026-07-07 session review: the synthesizer compared last-30d TRx against
    a last-90d figure that CONTAINS those same 30 days and called the -3.4%
    delta a 'softening'. The prompt must forbid overlapping-window baselines
    and require prior non-overlapping periods instead."""
    p = build_synthesis_prompt(
        "is that above or below baseline?",
        [{"name": "kpi_calculate_tool", "args": {"kpi_name": "TRx", "window": "last 90 days"}}],
        [{"tool": "kpi_calculate_tool", "result": '{"value": 15767}'}],
    )
    assert "overlap" in p.lower()
    assert "non-overlapping" in p.lower()


@pytest.mark.unit
def test_prompt_requires_surfacing_coverage_warning():
    """When a tool result carries a coverage_warning, the synthesizer must repeat
    it and refuse trend conclusions from that figure."""
    # Results deliberately do NOT contain the literal string — the INSTRUCTION
    # block itself must tell the synthesizer what to do with a coverage_warning.
    p = build_synthesis_prompt(
        "is that above or below baseline?",
        [{"name": "kpi_calculate_tool", "args": {"kpi_name": "TRx"}}],
        [{"tool": "kpi_calculate_tool", "result": '{"value": 15767}'}],
    )
    assert "coverage_warning" in p
