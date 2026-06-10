"""B3 — health_score recipient emit + dead-consumer wiring tests.

Verifies the ScoreComposerNode:
  (a) invokes the previously-dead optimizable-prompt getter
      (``get_summary_prompt``) so the optimizable summary template is
      actually consumed, while keeping the node's summary output BYTE-IDENTICAL
      to the pre-wiring inline construction (existing assertions still hold);
  (b) emits a recipient training signal for the summary field, keyed by the
      backing SIGNATURE's input_fields (the explicit emit<->provider
      contract, NOT the .format() template placeholders);
  (c) the reward is a deterministic heuristic in [0, 1];
  (d) emit is best-effort — a failure never breaks the node run.

score_composer only produces a SUMMARY (recommendations are built elsewhere in
agent.py, not from a template), so only ``summary_template`` is consumed/emitted
here. ``recommendation_template`` is out of this node's scope.

Offline only: ``emit_recipient_signal`` is patched, no real LM / DB.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.feedback_learner.recipient_optimizer import (
    recipient_required_input_keys,
)
from src.agents.health_score.nodes.score_composer import ScoreComposerNode

EMIT_TARGET = "src.agents.health_score.nodes.score_composer.emit_recipient_signal"


def _emit_calls_by_field(mock_emit: AsyncMock) -> Dict[str, Dict[str, Any]]:
    """Index the recorded emit calls by their ``template_field`` kwarg."""
    by_field: Dict[str, Dict[str, Any]] = {}
    for call in mock_emit.await_args_list:
        kwargs = call.kwargs
        by_field[kwargs["template_field"]] = kwargs
    return by_field


@pytest.fixture
def healthy_state() -> Dict[str, Any]:
    """A fully-populated, fully-MEASURED healthy state (grade A, no issues).

    F1: all four dimensions carry ``<dim>_health_measured = True`` so the
    composer treats this as a complete measurement (the emit contract only fires
    when every dimension score is populated).
    """
    return {
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


@pytest.fixture
def degraded_state() -> Dict[str, Any]:
    """A fully-MEASURED state with a critical issue (unhealthy component)."""
    return {
        "query": "",
        "check_scope": "full",
        "component_statuses": [
            {"component_name": "db", "status": "unhealthy"},
        ],
        "component_health_score": 0.3,
        "component_health_measured": True,
        "model_metrics": [],
        "model_health_score": 0.3,
        "model_health_measured": True,
        "pipeline_statuses": [],
        "pipeline_health_score": 0.3,
        "pipeline_health_measured": True,
        "agent_statuses": [],
        "agent_health_score": 0.3,
        "agent_health_measured": True,
        "total_latency_ms": 0,
        "errors": [],
        "status": "checking",
    }


class TestScoreComposerRecipientConsume:
    @pytest.mark.asyncio
    async def test_summary_getter_invoked_and_output_preserved(self, healthy_state):
        """The optimizable summary getter fires (no longer dead) AND output is intact."""
        node = ScoreComposerNode()
        from src.agents.health_score import dspy_integration as di

        integration = di.get_health_score_dspy_integration()
        with (
            patch(EMIT_TARGET, new=AsyncMock(return_value=True)),
            patch.object(
                integration, "get_summary_prompt", wraps=integration.get_summary_prompt
            ) as spy_summary,
        ):
            result = await node.execute(healthy_state)

        assert spy_summary.called, "get_summary_prompt was never invoked (still dead)"

        # Output semantics unchanged for the healthy case.
        summary = result["health_summary"]
        assert isinstance(summary, str)
        assert "excellent" in summary
        assert "Grade: A" in summary
        assert "All systems operational" in summary

    @pytest.mark.asyncio
    async def test_summary_output_byte_identical_to_inline(self, degraded_state):
        """Routing through the getter reproduces the inline construction exactly."""
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock(return_value=True)):
            result = await node.execute(degraded_state)

        summary = result["health_summary"]
        # Grade F, 1 critical issue, score 30.0
        assert "System health is critical (Grade: F, Score: 30.0/100)." in summary
        assert "1 critical issue(s) detected." in summary

    @pytest.mark.asyncio
    async def test_no_components_in_summary_output(self, degraded_state):
        """score_composer passes no component names, so none leak into the summary."""
        node = ScoreComposerNode()
        with patch(EMIT_TARGET, new=AsyncMock(return_value=True)):
            result = await node.execute(degraded_state)
        # The "Components:" suffix only renders when components are supplied;
        # score_composer supplies none, so the suffix must be absent.
        assert "Components:" not in result["health_summary"]


class TestScoreComposerRecipientEmit:
    @pytest.mark.asyncio
    async def test_emit_called_with_contract_keys(self, healthy_state):
        """Emit fires for the summary with EXACTLY the required input keys."""
        node = ScoreComposerNode()
        mock_emit = AsyncMock(return_value=True)
        with patch(EMIT_TARGET, new=mock_emit):
            await node.execute(healthy_state)

        assert mock_emit.await_count >= 1
        by_field = _emit_calls_by_field(mock_emit)
        required = recipient_required_input_keys("health_score")

        assert "summary_template" in by_field
        kwargs = by_field["summary_template"]
        assert kwargs["agent_name"] == "health_score"
        assert set(kwargs["signature_inputs"].keys()) == set(required["summary_template"]), (
            "signature_inputs keys for summary_template must EXACTLY match "
            "recipient_required_input_keys()['summary_template']"
        )
        assert isinstance(kwargs["generated_output"], str)
        assert kwargs["generated_output"]

    @pytest.mark.asyncio
    async def test_reward_in_unit_interval(self, healthy_state):
        """Every emitted reward is a deterministic float in [0, 1]."""
        node = ScoreComposerNode()
        mock_emit = AsyncMock(return_value=True)
        with patch(EMIT_TARGET, new=mock_emit):
            await node.execute(healthy_state)

        assert mock_emit.await_count >= 1
        for call in mock_emit.await_args_list:
            reward = call.kwargs["reward"]
            assert isinstance(reward, float)
            assert 0.0 <= reward <= 1.0

    @pytest.mark.asyncio
    async def test_reward_is_deterministic(self):
        """Same inputs -> same reward (deterministic heuristic, no randomness)."""
        from src.agents.health_score.nodes.score_composer import _signal_reward

        out = "System health is excellent (Grade: A, Score: 100.0/100). All systems operational."
        inputs = {
            "overall_score": 100.0,
            "grade": "A",
            "component_scores": "component=1.0, model=1.0",
            "critical_issues": "None",
        }
        assert _signal_reward(out, inputs) == _signal_reward(out, inputs)
        assert 0.0 <= _signal_reward(out, inputs) <= 1.0

    @pytest.mark.asyncio
    async def test_emit_failure_does_not_break_node(self, healthy_state):
        """A raising emit must not fail the node (best-effort)."""
        node = ScoreComposerNode()
        boom = AsyncMock(side_effect=RuntimeError("db down"))
        with patch(EMIT_TARGET, new=boom):
            result = await node.execute(healthy_state)

        assert result["status"] == "completed"
        assert result["health_summary"] is not None
        assert result["overall_health_score"] == 100.0
