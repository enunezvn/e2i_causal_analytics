"""Client-fidelity grading for the AG-UI runner (2026-08-19, session no90vkf).

The defect class this closes: the browser's ``@ag-ui/core`` Zod schema aborts
the ENTIRE CopilotKit run at the first invalid event — measured live on
``TEXT_MESSAGE_CONTENT`` with ``delta: ""`` ("Delta must not be an empty
string") — while this runner's lenient parser keeps reading and grades the
turn green. The 44/7/0 full eval could not see the P0 that killed every
browser tool+synthesis turn, because NO server-side artifact carries the
failure (HTTP 200, RUN_FINISHED, answer persisted, error_occurred=f).

Contract: ``client_fatal_events`` mirrors the client rule (TEXT_MESSAGE_CONTENT
requires a non-empty string delta) and ``_grade_stream_health`` fails any turn
whose stream contains such a frame — even when the runner itself extracted a
perfectly good ``response_text`` from the rest of the stream.
"""

from typing import Any, Dict, List

import pytest

from scripts.demos.copilot_agui_runner import _grade_stream_health, client_fatal_events

pytestmark = pytest.mark.unit


def _content(delta: Any, mid: str = "m1") -> Dict[str, Any]:
    ev: Dict[str, Any] = {"type": "TEXT_MESSAGE_CONTENT", "messageId": mid, "t_ms": 10.0}
    if delta is not None:
        ev["delta"] = delta
    return ev


def _healthy_stream() -> List[Dict[str, Any]]:
    return [
        {"type": "RUN_STARTED", "t_ms": 1.0},
        {"type": "TEXT_MESSAGE_START", "messageId": "m1", "t_ms": 5.0},
        _content("West-region TRx shortfall: "),
        _content("Remibrutinib access barriers."),
        {"type": "TEXT_MESSAGE_END", "messageId": "m1", "t_ms": 20.0},
        {"type": "RUN_FINISHED", "t_ms": 25.0},
    ]


def _record(events: List[Dict[str, Any]], response_text: str = "an answer") -> Dict[str, Any]:
    return {
        "events": events,
        "response_text": response_text,
        "stream_frames": len(events),
        "http_status": 200,
        "error": None,
    }


class TestClientFatalEvents:
    def test_empty_delta_is_fatal(self):
        """The exact live frame shape (no90vkf frames 197/200)."""
        events = _healthy_stream()
        events.insert(2, _content(""))
        fatal = client_fatal_events(events)
        assert len(fatal) == 1 and fatal[0].get("delta") == ""

    def test_missing_delta_is_fatal(self):
        events = _healthy_stream()
        events.insert(2, _content(None))
        assert len(client_fatal_events(events)) == 1

    def test_non_string_delta_is_fatal(self):
        """The TS schema types delta as string; a list/object delta is just as
        dead on arrival."""
        events = _healthy_stream()
        events.insert(2, _content([{"text": "x", "type": "text"}]))
        assert len(client_fatal_events(events)) == 1

    def test_healthy_stream_has_no_fatal_events(self):
        assert client_fatal_events(_healthy_stream()) == []

    def test_tool_call_args_empty_delta_not_fatal(self):
        """Zod constrains ONLY TextMessageContent's delta — TOOL_CALL_ARGS is
        unconstrained (measured in @ag-ui/core dist); don't over-fail."""
        events = _healthy_stream()
        events.insert(2, {"type": "TOOL_CALL_ARGS", "toolCallId": "t1", "delta": "", "t_ms": 8.0})
        assert client_fatal_events(events) == []


class TestGradeStreamHealthZod:
    def test_turn_with_empty_delta_fails_even_with_answer_text(self):
        """The blind-spot shape: good response_text, dead browser."""
        events = _healthy_stream()
        events.insert(2, _content(""))
        record = _record(events)
        _grade_stream_health(record)
        assert record["error"], "Zod-fatal frame graded green — the browser blind spot survives"
        assert "delta" in record["error"].lower() or "zod" in record["error"].lower()

    def test_healthy_turn_stays_green(self):
        record = _record(_healthy_stream())
        _grade_stream_health(record)
        assert record["error"] is None

    def test_existing_error_not_overwritten(self):
        events = _healthy_stream()
        events.insert(2, _content(""))
        record = _record(events)
        record["error"] = "HTTP 500: upstream"
        _grade_stream_health(record)
        assert record["error"] == "HTTP 500: upstream"
