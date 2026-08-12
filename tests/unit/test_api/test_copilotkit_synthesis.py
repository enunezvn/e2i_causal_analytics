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
def test_prompt_requires_prose_to_match_own_table():
    """#1550 (2026-08-11 eval): the synthesizer's comparative prose contradicted
    the correctly-retrieved table in the same answer — a wrong driver rank (1.5),
    two different 'largest' cohorts two lines apart (1.7), a sign flip on both an
    effect and its dollar impact (+0.092 described as '-0.09' / -$30.6K) (2.4),
    and a drifted average (5.1). The instruction block must require every
    comparative claim (rank, superlative, sign, dollar impact, average) to be
    re-derived from the printed values, with the table winning any conflict."""
    # Results deliberately generic — the INSTRUCTION block itself must carry the
    # self-consistency requirement regardless of payload content.
    p = build_synthesis_prompt(
        "what drives TRx?",
        [{"name": "causal_analysis_tool", "args": {"brand": "Kisqali"}}],
        [{"tool": "causal_analysis_tool", "result": '{"drivers": []}'}],
    )
    assert "PROSE MUST MATCH YOUR OWN TABLE" in p
    # Sign/direction discipline (2.4's +0.092 -> "-0.09" flip)
    assert "sign and direction" in p
    assert "+0.092" in p
    # Ranking discipline (1.5's wrong rank)
    assert "rank" in p.lower()
    # Superlative discipline (1.7's two 'largest' cohorts)
    assert "exactly ONE" in p
    # Average discipline (5.1's 0.845 vs 0.8543)
    assert "average" in p.lower()
    # Conflict resolution: the table wins
    assert "TABLE is correct" in p


@pytest.mark.unit
def test_prompt_requires_period_grain_discipline():
    """#1552 (2026-08-11 eval, 6.5): the synthesizer merged two source rows
    into an invented 'Jun/Jul 2026' two-month bucket, rendered it in the same
    table as an 'Aug 2026' row, and called the resulting gap 'an unexplained
    scale discontinuity' — converting a period-labeling gap into user-facing
    doubt about the data (the substrate is uniformly monthly-grain; measured).
    The instruction block must require: one source period per row (no merged
    period labels), no mixed period widths in one table unlabeled, partial-
    period claims only when the tool results state them, and period-width
    differences explained as such — never as unexplained discontinuities."""
    # Results deliberately generic — the INSTRUCTION block itself must carry
    # the grain discipline regardless of payload content.
    p = build_synthesis_prompt(
        "Forecast Kisqali TRx volume for the next two quarters",
        [{"name": "e2i_data_query_tool", "args": {"query_type": "predictions"}}],
        [{"tool": "e2i_data_query_tool", "result": '{"count": 0}'}],
    )
    assert "PERIOD GRAIN" in p
    # No merged period labels (the invented 'Jun/Jul 2026' bucket)
    assert "Jun/Jul" in p
    # No mixed widths unlabeled in a single table
    assert "width" in p.lower()
    # Partial-period (MTD) claims only from the tool results, never invented
    assert "month-to-date" in p.lower() or "MTD" in p
    # Period-width artifacts are explained, never 'unexplained discontinuities'
    assert "unexplained" in p
    # No trend inference across unequal periods
    assert "trend" in p.lower()


@pytest.mark.unit
def test_prompt_forbids_orphan_footnote_markers():
    """5.1 (same eval): a table cell read '3 (0)*' with no footnote anywhere
    in the answer — the marker was LLM-authored (the tool payload contains no
    asterisk; verified against the raw event stream). The instruction block
    must forbid footnote markers without their footnote text."""
    p = build_synthesis_prompt(
        "What is the current system health score?",
        [{"name": "e2i_data_query_tool", "args": {"query_type": "agent_analysis"}}],
        [{"tool": "e2i_data_query_tool", "result": '{"count": 0}'}],
    )
    assert "FOOTNOTES" in p
    assert "footnote marker" in p.lower()


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
