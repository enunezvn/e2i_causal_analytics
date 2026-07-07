"""Unit coverage for chat_node's streamed tool-call accumulation.

Root cause (2026-07-07 session review): Anthropic streams a message as content
blocks, so when the model emits leading text before its tool calls, the text is
block 0 and the first ``tool_use`` block is index 1. ``tool_call_chunks`` carry
that provider index, but the parallel ``chunk.tool_calls`` entries carry NO
index — the old accumulator invented one from list position (0), the args-merge
pass then couldn't find the entry at index 1, created a second one, and the
original shipped to ToolNode as a ghost call with ``{}`` args. The resulting
Pydantic validation error rendered raw in the chat UI ("Error invoking tool
'kpi_calculate_tool' with kwargs {} …").

Contract under test:
- ``_accumulate_tool_call_event``: ``tool_call_chunks`` is the authoritative
  channel; ``chunk.tool_calls`` is only a fallback (merged by id) for chunks
  that carry no tool_call_chunks.
- ``_finalize_tool_calls``: parses streamed ``args_str`` into args, drops
  nameless entries, and collapses duplicate ids preferring the entry that
  actually received args (defense-in-depth against any future ghost).
"""

from typing import Any

import pytest

from src.api.routes.copilotkit import _accumulate_tool_call_event, _finalize_tool_calls


class FakeChunk:
    """Minimal stand-in for an AIMessageChunk (only the fields the accumulator reads)."""

    def __init__(self, tool_calls: Any = None, tool_call_chunks: Any = None):
        self.tool_calls = tool_calls or []
        self.tool_call_chunks = tool_call_chunks or []


def _run(chunks: list) -> list:
    accumulated: list[dict] = []
    for c in chunks:
        _accumulate_tool_call_event(accumulated, c)
    return _finalize_tool_calls(accumulated)


@pytest.mark.unit
def test_anthropic_leading_text_no_ghost_call():
    """Leading text shifts the tool_use block to index 1 — must yield ONE call."""
    chunks = [
        # content_block_start for the tool_use block: langchain emits BOTH a
        # parsed tool_calls entry (no index, empty args) and a tool_call_chunk
        # carrying the provider's block index.
        FakeChunk(
            tool_calls=[
                {"name": "kpi_calculate_tool", "id": "toolu_1", "args": {}, "type": "tool_call"}
            ],
            tool_call_chunks=[
                {"index": 1, "id": "toolu_1", "name": "kpi_calculate_tool", "args": ""}
            ],
        ),
        FakeChunk(
            tool_call_chunks=[{"index": 1, "id": None, "name": None, "args": '{"kpi_name": "TRx"'}]
        ),
        FakeChunk(
            tool_call_chunks=[
                {
                    "index": 1,
                    "id": None,
                    "name": None,
                    "args": ', "brand": "Fabhalta", "window": "last 90 days"}',
                }
            ]
        ),
    ]
    calls = _run(chunks)
    assert len(calls) == 1, f"ghost call minted: {calls}"
    assert calls[0]["id"] == "toolu_1"
    assert calls[0]["args"] == {
        "kpi_name": "TRx",
        "brand": "Fabhalta",
        "window": "last 90 days",
    }


@pytest.mark.unit
def test_two_calls_after_leading_text_no_ghost():
    """The live turn-2 pattern: text block 0, tool blocks 1 and 2 → exactly two calls."""
    chunks = [
        FakeChunk(
            tool_calls=[
                {"name": "kpi_calculate_tool", "id": "toolu_1", "args": {}, "type": "tool_call"}
            ],
            tool_call_chunks=[
                {"index": 1, "id": "toolu_1", "name": "kpi_calculate_tool", "args": ""}
            ],
        ),
        FakeChunk(
            tool_call_chunks=[{"index": 1, "id": None, "name": None, "args": '{"kpi_name": "TRx"}'}]
        ),
        FakeChunk(
            tool_calls=[
                {"name": "e2i_data_query_tool", "id": "toolu_2", "args": {}, "type": "tool_call"}
            ],
            tool_call_chunks=[
                {"index": 2, "id": "toolu_2", "name": "e2i_data_query_tool", "args": ""}
            ],
        ),
        FakeChunk(
            tool_call_chunks=[
                {"index": 2, "id": None, "name": None, "args": '{"query_type": "kpi"}'}
            ]
        ),
    ]
    calls = _run(chunks)
    assert [c["name"] for c in calls] == ["kpi_calculate_tool", "e2i_data_query_tool"]
    assert all(c["args"] for c in calls), f"a call shipped with empty args: {calls}"


@pytest.mark.unit
def test_no_leading_text_single_call():
    """No leading text → block index 0; the pre-fix happy path must keep working."""
    chunks = [
        FakeChunk(
            tool_calls=[
                {"name": "kpi_calculate_tool", "id": "toolu_1", "args": {}, "type": "tool_call"}
            ],
            tool_call_chunks=[
                {"index": 0, "id": "toolu_1", "name": "kpi_calculate_tool", "args": ""}
            ],
        ),
        FakeChunk(
            tool_call_chunks=[{"index": 0, "id": None, "name": None, "args": '{"kpi_name": "TRx"}'}]
        ),
    ]
    calls = _run(chunks)
    assert len(calls) == 1
    assert calls[0]["args"] == {"kpi_name": "TRx"}


@pytest.mark.unit
def test_openai_style_indexed_chunks():
    """OpenAI-style streams (tool_call_chunks only, tool-call-ordinal indices)."""
    chunks = [
        FakeChunk(tool_call_chunks=[{"index": 0, "id": "call_a", "name": "tool_a", "args": ""}]),
        FakeChunk(tool_call_chunks=[{"index": 0, "id": None, "name": None, "args": '{"x": 1}'}]),
        FakeChunk(
            tool_call_chunks=[{"index": 1, "id": "call_b", "name": "tool_b", "args": '{"y": 2}'}]
        ),
    ]
    calls = _run(chunks)
    assert [(c["name"], c["args"]) for c in calls] == [
        ("tool_a", {"x": 1}),
        ("tool_b", {"y": 2}),
    ]


@pytest.mark.unit
def test_fallback_complete_tool_calls_without_chunks():
    """Providers that emit only parsed tool_calls (no chunks) must still work."""
    chunks = [
        FakeChunk(tool_calls=[{"name": "tool_a", "id": "call_a", "args": {"x": 1}}]),
    ]
    calls = _run(chunks)
    assert calls == [{"id": "call_a", "name": "tool_a", "args": {"x": 1}}]


@pytest.mark.unit
def test_fallback_reemission_merges_by_id():
    """A re-emitted parsed tool_calls entry for the same id must NOT mint a duplicate."""
    chunks = [
        FakeChunk(tool_calls=[{"name": "tool_a", "id": "call_a", "args": {}}]),
        FakeChunk(tool_calls=[{"name": "tool_a", "id": "call_a", "args": {"x": 1}}]),
    ]
    calls = _run(chunks)
    assert len(calls) == 1
    assert calls[0]["args"] == {"x": 1}


@pytest.mark.unit
def test_finalize_collapses_duplicate_ids_preferring_args():
    """Defense-in-depth: a ghost duplicate (same id, no args) must never survive."""
    accumulated = [
        {"index": 0, "id": "toolu_1", "name": "kpi_calculate_tool", "args": {}, "args_str": ""},
        {
            "index": 1,
            "id": "toolu_1",
            "name": "kpi_calculate_tool",
            "args": {},
            "args_str": '{"kpi_name": "TRx"}',
        },
    ]
    calls = _finalize_tool_calls(accumulated)
    assert len(calls) == 1
    assert calls[0]["args"] == {"kpi_name": "TRx"}


@pytest.mark.unit
def test_finalize_drops_nameless_entries():
    accumulated = [
        {"index": 3, "id": None, "name": "", "args": {}, "args_str": '{"orphan": true}'},
        {"index": 0, "id": "call_a", "name": "tool_a", "args": {}, "args_str": '{"x": 1}'},
    ]
    calls = _finalize_tool_calls(accumulated)
    assert [c["name"] for c in calls] == ["tool_a"]


@pytest.mark.unit
def test_finalize_wraps_partial_json_args():
    """args_str that started mid-JSON keeps the existing brace-repair behavior."""
    accumulated = [
        {"index": 0, "id": "call_a", "name": "tool_a", "args": {}, "args_str": '"x": 1}'},
    ]
    calls = _finalize_tool_calls(accumulated)
    assert calls[0]["args"] == {"x": 1}


@pytest.mark.unit
def test_distinct_calls_with_empty_args_are_kept():
    """Two DIFFERENT calls (distinct ids) where one has genuinely no args must both survive
    — dedup is strictly by id, never by name."""
    accumulated = [
        {"index": 0, "id": "call_a", "name": "tool_a", "args": {}, "args_str": ""},
        {"index": 1, "id": "call_b", "name": "tool_a", "args": {}, "args_str": '{"x": 1}'},
    ]
    calls = _finalize_tool_calls(accumulated)
    assert len(calls) == 2
