"""#1447 — the health summary must not narrate an UNMEASURED state as a
measured catastrophe, and must name the scope it actually checked.

Context (READ BEFORE CHANGING): the 0.0 / Grade-"F" structured payload for a
zero-measured check is the DELIBERATE F1 anti-fabrication guard in
``ScoreComposerNode.execute`` ("with nothing measured we must NOT claim a
healthy grade-A/100 system"). These tests PIN that payload — the defect is the
NARRATION seam, not the score.

The observed production string (request aaec638e, 2026-08-03, check_scope
"models", no metrics_store wired) was::

    System health is critical (Grade: F, Score: 0.0/100). 1 critical issue(s) detected.

which is byte-identical to a genuinely-measured grade-F catastrophe and drops
the explanatory critical issue the node already builds.

Guards here:
  (a) nothing measured  -> prose says UNKNOWN/unmeasured, never asserts a
      critical system, while overall_health_score stays 0.0 and health_grade
      stays "F" in the structured payload;
  (b) a MEASURED full-scope summary is byte-identical to the historical
      rendering (the optimizable ``summary_template`` documents this contract);
  (c) the scope is named in the prose (a models-scoped check must not be
      narrated as a whole-system verdict).

Offline only: pure node computation, ``emit_recipient_signal`` patched.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.health_score.nodes.score_composer import ScoreComposerNode

EMIT_TARGET = "src.agents.health_score.nodes.score_composer.emit_recipient_signal"

# The exact leading critical issue the node already builds for a zero-measured
# check (score_composer.execute). Requirement (2): the prose must surface this
# TEXT, not merely its count.
UNMEASURED_ISSUE = (
    "No health dimensions could be measured - no real health "
    "backends are wired (component/model/pipeline/agent). Health "
    "status is UNKNOWN, not healthy."
)


def _unmeasured_state(scope: str = "models") -> Dict[str, Any]:
    """A check where NO dimension was measured (the #1447 production shape)."""
    return {
        "query": "What is the ROC-AUC and calibration of the current Kisqali model?",
        "check_scope": scope,
        "component_statuses": [],
        "model_metrics": [],
        "pipeline_statuses": [],
        "agent_statuses": [],
        "total_latency_ms": 0,
        "errors": [],
        "status": "checking",
    }


def _measured_state(**overrides: Any) -> Dict[str, Any]:
    """A fully-MEASURED full-scope check (all four dimensions)."""
    state: Dict[str, Any] = {
        "query": "",
        "check_scope": "full",
        "component_statuses": [],
        "component_health_score": 1.0,
        "component_health_measured": True,
        "model_metrics": [],
        "model_health_score": 1.0,
        "model_health_measured": True,
        "pipeline_statuses": [],
        "pipeline_health_score": 1.0,
        "pipeline_health_measured": True,
        "agent_statuses": [],
        "agent_health_score": 1.0,
        "agent_health_measured": True,
        "total_latency_ms": 0,
        "errors": [],
        "status": "checking",
    }
    state.update(overrides)
    return state


class TestUnmeasuredNarration:
    """(a) UNKNOWN must not be narrated as a measured critical failure."""

    @pytest.mark.asyncio
    async def test_unmeasured_summary_says_unknown_not_critical(self):
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(_unmeasured_state())

        summary = result["health_summary"]

        # The exact production defect string must be gone.
        assert "System health is critical" not in summary
        assert "critical issue(s) detected" not in summary
        # No assertion of a critical system anywhere in the prose.
        assert "critical" not in summary.lower(), summary
        # It must say what actually happened.
        assert "UNKNOWN" in summary
        assert "measured" in summary.lower()

    @pytest.mark.asyncio
    async def test_unmeasured_payload_semantics_unchanged(self):
        """The F1 anti-fabrication guard is DELIBERATE — pin it.

        Only the narration changes; the structured payload must still refuse to
        claim a healthy system.
        """
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(_unmeasured_state())

        assert result["overall_health_score"] == 0.0
        assert result["health_grade"] == "F"
        assert result["data_provenance"] == "unknown"
        assert result["critical_issues"][0] == UNMEASURED_ISSUE

    @pytest.mark.asyncio
    async def test_unmeasured_summary_surfaces_issue_text_not_just_count(self):
        """(2) the leading critical-issue TEXT is surfaced, not its count."""
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(_unmeasured_state())

        assert UNMEASURED_ISSUE in result["health_summary"]

    @pytest.mark.asyncio
    async def test_unmeasured_summary_explains_the_zero_placeholder(self):
        """The 0.0/F payload is reconciled in prose so a reader seeing the
        widget's 0.0/F cannot mistake it for a measured failure."""
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(_unmeasured_state())

        summary = result["health_summary"]
        assert "0.0/100" in summary
        assert "Grade-F" in summary

    @pytest.mark.asyncio
    async def test_no_training_signal_emitted_when_nothing_measured(self):
        """The reward heuristic penalises the word "unknown" in a summary; the
        UNKNOWN narration must never reach it.

        Guarded structurally: the emit contract requires a populated
        ``component_scores``, which is empty when no dimension was measured.
        """
        node = ScoreComposerNode()
        emit = AsyncMock()
        with patch(EMIT_TARGET, new=emit):
            await node.execute(_unmeasured_state())

        assert emit.await_count == 0


class TestMeasuredRenderingByteIdentical:
    """(b) regression guard: the MEASURED full-scope rendering is unchanged.

    ``dspy_integration.HealthReportPrompts.summary_template`` is OPTIMIZABLE and
    documents that its default renders byte-identically to the node's historical
    inline construction. Changing the unmeasured branch must not disturb it.
    """

    @pytest.mark.asyncio
    async def test_healthy_full_scope_summary_byte_identical(self):
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(_measured_state())

        assert result["health_summary"] == (
            "System health is excellent (Grade: A, Score: 100.0/100). All systems operational."
        )

    @pytest.mark.asyncio
    async def test_graded_full_scope_summary_byte_identical(self):
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(
                _measured_state(
                    component_health_score=0.9,
                    model_health_score=0.8,
                    pipeline_health_score=0.85,
                    agent_health_score=0.95,
                )
            )

        assert result["health_summary"] == (
            "System health is good (Grade: B, Score: 86.5/100). All systems operational."
        )

    @pytest.mark.asyncio
    async def test_measured_critical_full_scope_summary_byte_identical(self):
        """A genuinely-measured grade-F catastrophe still reads as critical."""
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(
                _measured_state(
                    component_health_score=0.3,
                    model_health_score=0.3,
                    pipeline_health_score=0.3,
                    agent_health_score=0.3,
                    component_statuses=[{"component_name": "db", "status": "unhealthy"}],
                )
            )

        # The diagnostic block is appended after a blank line; the
        # template-rendered summary is the first paragraph.
        first_paragraph = result["health_summary"].split("\n\n")[0]
        assert first_paragraph == (
            "System health is critical (Grade: F, Score: 30.0/100). 1 critical issue(s) detected."
        )

    def test_default_template_getter_is_backward_compatible(self):
        """Existing 5-kwarg callers of the optimizable getter still render the
        historical string (no required new parameters)."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration

        integration = HealthScoreDSPyIntegration()
        rendered = integration.get_summary_prompt(
            grade="A",
            score=92.5,
            components="database, cache, api",
            critical_count=0,
            warning_count=2,
        )
        assert rendered == (
            "System health is excellent (Grade: A, Score: 92.5/100). "
            "All systems operational. Components: database, cache, api."
        )


class TestScopeIsNamed:
    """(c) a scoped check must not be narrated as a whole-system verdict.

    ``_record_full_check`` (src/api/routes/health_score.py) already states the
    repo's position: only a FULL check is a faithful overall measurement; a
    quick check measures components only and single-dimension scopes are not an
    overall measurement either.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scope,label",
        [
            ("models", "Model"),
            ("pipelines", "Pipeline"),
            ("agents", "Agent"),
            ("quick", "Component"),
            ("full", "System"),
        ],
    )
    async def test_scope_named_in_unmeasured_prose(self, scope: str, label: str):
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(_unmeasured_state(scope=scope))

        assert result["health_summary"].startswith(f"{label} health status is UNKNOWN")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scope,label",
        [
            ("models", "Model"),
            ("pipelines", "Pipeline"),
            ("agents", "Agent"),
            ("quick", "Component"),
            ("full", "System"),
        ],
    )
    async def test_scope_named_in_measured_prose(self, scope: str, label: str):
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(_measured_state(check_scope=scope))

        assert result["health_summary"].startswith(f"{label} health is excellent")

    @pytest.mark.asyncio
    async def test_absent_scope_defaults_to_system(self):
        """A state with no check_scope keeps the historical "System" label."""
        state = _measured_state()
        state.pop("check_scope")

        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock()):
            result = await node.execute(state)

        assert result["health_summary"].startswith("System health is excellent")
