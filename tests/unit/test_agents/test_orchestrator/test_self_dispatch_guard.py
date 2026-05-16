"""Unit tests for the centralized self-dispatch guard (Issue #251 F1).

The orchestrator routes to OTHER agents and must never appear in either
``dispatch_plan`` or ``agents_dispatched``. These tests pin the
invariants exposed by ``_self_dispatch_guard`` and the consumer sites in
``agent._build_output``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.agents.orchestrator._self_dispatch_guard import (
    SELF_AGENT_NAME,
    SELF_DEGRADED_MARKER,
    is_self_dispatch,
    strip_self_dispatch,
)
from src.agents.orchestrator.agent import OrchestratorAgent


class TestStripSelfDispatchHelper:
    """The helper must remove only ``orchestrator`` entries, preserve
    order of the rest, and emit a WARNING log when stripping fires."""

    def test_removes_orchestrator_entries(self) -> None:
        cleaned = strip_self_dispatch(
            ["causal_impact", SELF_AGENT_NAME, "explainer"],
            context="test",
        )
        assert cleaned == ["causal_impact", "explainer"]
        assert SELF_AGENT_NAME not in cleaned

    def test_preserves_order_and_duplicates_of_non_self(self) -> None:
        cleaned = strip_self_dispatch(
            ["a", "b", SELF_AGENT_NAME, "a", "c"],
            context="test",
        )
        assert cleaned == ["a", "b", "a", "c"]

    def test_returns_empty_when_only_orchestrator(self) -> None:
        cleaned = strip_self_dispatch(
            [SELF_AGENT_NAME, SELF_AGENT_NAME],
            context="test",
        )
        assert cleaned == []

    def test_emits_warning_when_stripping(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            strip_self_dispatch(
                ["explainer", SELF_AGENT_NAME],
                context="unit.test_emits_warning_when_stripping",
            )
        msgs = [r.message for r in caplog.records if "F1 invariant" in r.message]
        assert msgs, caplog.text

    def test_no_warning_when_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            strip_self_dispatch(["causal_impact", "explainer"], context="clean")
        msgs = [r.message for r in caplog.records if "F1 invariant" in r.message]
        assert not msgs, caplog.text

    def test_is_self_dispatch_predicate(self) -> None:
        assert is_self_dispatch(SELF_AGENT_NAME) is True
        assert is_self_dispatch("explainer") is False
        assert is_self_dispatch(None) is False

    def test_degraded_marker_is_distinct(self) -> None:
        # The F2 degraded marker must never accidentally equal the F1
        # self literal — the whole point of MED-2's fix is that callers
        # can return the marker instead of the literal.
        assert SELF_DEGRADED_MARKER != SELF_AGENT_NAME


class TestBuildOutputStripsSelfDispatch:
    """``_build_output`` rebuilds ``agents_dispatched`` from
    ``agent_results.keys()`` *after* the router has already run
    ``_strip_self_dispatch`` on the dispatch plan. If a worker agent
    result somehow names ``orchestrator`` (a reflection loop, a buggy
    upstream registry, a test stub) it would re-introduce the F1
    violation at the API boundary. The MED-1 fix re-applies the strip in
    ``_build_output`` itself."""

    def _orchestrator(self) -> OrchestratorAgent:
        # Construct with empty registry; we call ``_build_output``
        # directly, so the graph never runs.
        return OrchestratorAgent(agent_registry={}, enable_checkpointing=False, enable_opik=False)

    def test_build_output_drops_orchestrator_named_agent_result(self) -> None:
        orchestrator = self._orchestrator()
        agent_results: List[Dict[str, Any]] = [
            {"agent_name": "causal_impact", "success": True, "latency_ms": 10},
            {"agent_name": SELF_AGENT_NAME, "success": True, "latency_ms": 5},
            {"agent_name": "explainer", "success": True, "latency_ms": 7},
        ]
        state: Dict[str, Any] = {"agent_results": agent_results, "status": "completed"}

        output = orchestrator._build_output(state)  # type: ignore[arg-type]

        assert SELF_AGENT_NAME not in output["agents_dispatched"], output
        assert SELF_AGENT_NAME not in output["successful_agents"], output
        assert SELF_AGENT_NAME not in output["failed_agents"], output
        # Other agents preserved
        assert output["agents_dispatched"] == ["causal_impact", "explainer"]

    def test_build_output_drops_orchestrator_when_failed(self) -> None:
        orchestrator = self._orchestrator()
        agent_results: List[Dict[str, Any]] = [
            {"agent_name": SELF_AGENT_NAME, "success": False, "error": "self-loop"},
            {"agent_name": "explainer", "success": True, "latency_ms": 7},
        ]
        state: Dict[str, Any] = {"agent_results": agent_results, "status": "completed"}

        output = orchestrator._build_output(state)  # type: ignore[arg-type]

        assert SELF_AGENT_NAME not in output["agents_dispatched"], output
        assert SELF_AGENT_NAME not in output["failed_agents"], output

    def test_build_output_logs_warning_on_self_strip(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        orchestrator = self._orchestrator()
        agent_results: List[Dict[str, Any]] = [
            {"agent_name": SELF_AGENT_NAME, "success": True},
        ]
        state: Dict[str, Any] = {"agent_results": agent_results}
        with caplog.at_level("WARNING"):
            orchestrator._build_output(state)  # type: ignore[arg-type]
        assert any("F1 invariant" in r.message for r in caplog.records), caplog.text

    def test_build_output_clean_path_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        orchestrator = self._orchestrator()
        agent_results: List[Dict[str, Any]] = [
            {"agent_name": "causal_impact", "success": True},
        ]
        state: Dict[str, Any] = {"agent_results": agent_results}
        with caplog.at_level("WARNING"):
            orchestrator._build_output(state)  # type: ignore[arg-type]
        assert not any("F1 invariant" in r.message for r in caplog.records), caplog.text


class TestDownstreamSerializersCallGuard:
    """codex MED-1 follow-up LOW-1: the unit tests must non-vacuously
    cover that the serializer call sites in
    ``chatbot_graph.py`` / ``chatbot_tools.py`` / ``copilotkit.py`` /
    ``cognitive.py`` actually call ``strip_self_dispatch`` on the raw
    ``agents_dispatched`` payload. We assert by AST-scanning the source
    rather than running the FastAPI app (which requires the full
    LangGraph stack). Reverting any one strip call would trip the
    corresponding assertion."""

    def _src(self, rel: str) -> str:
        from pathlib import Path

        return Path(__file__).resolve().parents[4].joinpath(rel).read_text()

    def test_chatbot_graph_serializer_uses_helper(self) -> None:
        src = self._src("src/api/routes/chatbot_graph.py")
        assert "strip_self_dispatch(" in src, (
            "chatbot_graph.py must call strip_self_dispatch on the "
            "orchestrator_result agents_dispatched payload"
        )
        # The raw orchestrator field must not be assigned without the strip.
        assert 'agents_dispatched = orchestrator_result.get("agents_dispatched", [])' not in src

    def test_chatbot_tools_serializer_uses_helper(self) -> None:
        src = self._src("src/api/routes/chatbot_tools.py")
        assert "strip_self_dispatch(" in src
        assert 'agents_dispatched = orchestrator_result.get("agents_dispatched", [])' not in src

    def test_copilotkit_serializers_use_helper(self) -> None:
        src = self._src("src/api/routes/copilotkit.py")
        # Three sites: non-streaming result, streaming SSE dispatch_info,
        # run_causal_analysis agents_used.
        assert src.count("strip_self_dispatch(") >= 3, (
            "copilotkit.py must call strip_self_dispatch at all three "
            "serializer sites (non-streaming, streaming, run_causal_analysis)"
        )
        # Raw assignment must be gone from the non-streaming path.
        assert 'agents_dispatched = result.get("agents_dispatched", [])' not in src
        # Raw forward to agents_used must be gone too.
        assert '"agents_used": result.get("agents_dispatched", []),' not in src

    def test_cognitive_serializer_uses_helper(self) -> None:
        src = self._src("src/api/routes/cognitive.py")
        assert "strip_self_dispatch(" in src
        # The inline `if a != "orchestrator"` list-comp must be gone.
        assert "[a for a in agents_dispatched if a !=" not in src
