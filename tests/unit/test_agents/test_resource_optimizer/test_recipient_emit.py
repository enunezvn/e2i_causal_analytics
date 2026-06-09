"""B4 — resource_optimizer recipient emit + dead-consumer wiring tests.

Verifies the impact_projector node:
  (a) invokes the previously-dead optimizable-prompt getters
      (``get_summary_prompt`` / ``get_recommendation_prompt``) so the
      optimizable template is actually consumed, while keeping the node's
      output shape unchanged (existing assertions still hold);
  (b) emits a recipient training signal per produced field, keyed by the
      backing SIGNATURE's input_fields (the explicit emit<->provider
      contract, NOT the .format() template placeholders);
  (c) the reward is a deterministic heuristic in [0, 1];
  (d) emit is best-effort — a failure never breaks the node run.

Offline only: ``emit_recipient_signal`` is patched, no real LM / DB.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.feedback_learner.recipient_optimizer import (
    recipient_required_input_keys,
)
from src.agents.resource_optimizer.nodes.impact_projector import ImpactProjectorNode

EMIT_TARGET = "src.agents.resource_optimizer.nodes.impact_projector.emit_recipient_signal"


def _emit_calls_by_field(mock_emit: AsyncMock) -> Dict[str, Dict[str, Any]]:
    """Index the recorded emit calls by their ``template_field`` kwarg."""
    by_field: Dict[str, Dict[str, Any]] = {}
    for call in mock_emit.await_args_list:
        kwargs = call.kwargs
        by_field[kwargs["template_field"]] = kwargs
    return by_field


class TestImpactProjectorRecipientEmit:
    @pytest.mark.asyncio
    async def test_getters_invoked_and_output_shape_preserved(self, optimized_state):
        """The optimizable getters fire (no longer dead) AND output shape is intact."""
        node = ImpactProjectorNode()
        from src.agents.resource_optimizer import dspy_integration as di

        integration = di.get_resource_optimizer_dspy_integration()
        with (
            patch(EMIT_TARGET, new=AsyncMock(return_value=True)),
            patch.object(
                integration, "get_summary_prompt", wraps=integration.get_summary_prompt
            ) as spy_summary,
            patch.object(
                integration,
                "get_recommendation_prompt",
                wraps=integration.get_recommendation_prompt,
            ) as spy_rec,
        ):
            result = await node.execute(optimized_state)

        # Getters were actually consumed.
        assert spy_summary.called, "get_summary_prompt was never invoked (still dead)"
        assert spy_rec.called, "get_recommendation_prompt was never invoked (still dead)"

        # Output shape unchanged: existing semantics preserved.
        assert result["status"] == "completed"
        summary = result["optimization_summary"]
        assert isinstance(summary, str)
        assert "Optimization complete" in summary
        assert "ROI" in summary
        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)
        assert any("Increase" in r for r in recommendations)
        assert any("Reduce" in r for r in recommendations)

    @pytest.mark.asyncio
    async def test_emit_called_with_contract_keys(self, optimized_state):
        """Emit fires for summary + recommendation with EXACTLY the required input keys."""
        node = ImpactProjectorNode()
        mock_emit = AsyncMock(return_value=True)
        with patch(EMIT_TARGET, new=mock_emit):
            await node.execute(optimized_state)

        assert mock_emit.await_count >= 2
        by_field = _emit_calls_by_field(mock_emit)
        required = recipient_required_input_keys("resource_optimizer")

        assert "summary_template" in by_field
        assert "recommendation_template" in by_field

        for field in ("summary_template", "recommendation_template"):
            kwargs = by_field[field]
            assert kwargs["agent_name"] == "resource_optimizer"
            assert set(kwargs["signature_inputs"].keys()) == set(required[field]), (
                f"signature_inputs keys for {field} must EXACTLY match "
                f"recipient_required_input_keys()[{field}]"
            )
            assert isinstance(kwargs["generated_output"], str)
            assert kwargs["generated_output"]

    @pytest.mark.asyncio
    async def test_reward_in_unit_interval(self, optimized_state):
        """Every emitted reward is a deterministic float in [0, 1]."""
        node = ImpactProjectorNode()
        mock_emit = AsyncMock(return_value=True)
        with patch(EMIT_TARGET, new=mock_emit):
            await node.execute(optimized_state)

        assert mock_emit.await_count >= 1
        for call in mock_emit.await_args_list:
            reward = call.kwargs["reward"]
            assert isinstance(reward, float)
            assert 0.0 <= reward <= 1.0

    @pytest.mark.asyncio
    async def test_reward_is_deterministic(self, optimized_state):
        """Same inputs -> same reward (deterministic heuristic, no randomness)."""
        from src.agents.resource_optimizer.nodes.impact_projector import _signal_reward

        out = "Optimization complete. Projected outcome: 410000 (ROI: 1.23)."
        inputs = {"optimization_results": "x", "objective_value": 410000.0}
        assert _signal_reward(out, inputs) == _signal_reward(out, inputs)
        assert 0.0 <= _signal_reward(out, inputs) <= 1.0

    @pytest.mark.asyncio
    async def test_emit_failure_does_not_break_node(self, optimized_state):
        """A raising emit must not fail the node (best-effort)."""
        node = ImpactProjectorNode()
        boom = AsyncMock(side_effect=RuntimeError("db down"))
        with patch(EMIT_TARGET, new=boom):
            result = await node.execute(optimized_state)

        assert result["status"] == "completed"
        assert result["optimization_summary"] is not None
        assert result["recommendations"] is not None

    @pytest.mark.asyncio
    async def test_no_emit_when_no_allocations(self, base_state):
        """A failed projection (no allocations) emits nothing."""
        base_state["status"] = "projecting"
        base_state["optimal_allocations"] = []
        node = ImpactProjectorNode()
        mock_emit = AsyncMock(return_value=True)
        with patch(EMIT_TARGET, new=mock_emit):
            result = await node.execute(base_state)

        assert result["status"] == "failed"
        mock_emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returned_text_comes_from_getter(self, optimized_state):
        """M1: the optimized getter output must REACH the node output (not be discarded).

        Patch the integration getters to return sentinels; assert the sentinels
        appear in the node's optimization_summary / recommendations. This proves
        the DSPy-optimized template flows to the user (the loop is not hollow).
        """
        node = ImpactProjectorNode()
        from src.agents.resource_optimizer import dspy_integration as di

        integration = di.get_resource_optimizer_dspy_integration()
        summary_sentinel = "SENTINEL_SUMMARY_FROM_GETTER ROI"
        rec_sentinel = "SENTINEL_REC_FROM_GETTER Increase Reduce"
        with (
            patch(EMIT_TARGET, new=AsyncMock(return_value=True)),
            patch.object(integration, "get_summary_prompt", return_value=summary_sentinel),
            patch.object(integration, "get_recommendation_prompt", return_value=rec_sentinel),
        ):
            result = await node.execute(optimized_state)

        assert result["status"] == "completed"
        assert result["optimization_summary"] == summary_sentinel
        assert all(r == rec_sentinel for r in result["recommendations"])
        assert len(result["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_getter_failure_falls_back_to_inline(self, optimized_state):
        """A getter that raises falls back to the inline canonical text (no break)."""
        node = ImpactProjectorNode()
        from src.agents.resource_optimizer import dspy_integration as di

        integration = di.get_resource_optimizer_dspy_integration()
        with (
            patch(EMIT_TARGET, new=AsyncMock(return_value=True)),
            patch.object(integration, "get_summary_prompt", side_effect=KeyError("boom")),
            patch.object(integration, "get_recommendation_prompt", side_effect=KeyError("boom")),
        ):
            result = await node.execute(optimized_state)

        assert result["status"] == "completed"
        assert "Optimization complete" in result["optimization_summary"]
        recs = result["recommendations"]
        assert any("Increase allocation to" in r for r in recs)
        assert any("Reduce allocation from" in r for r in recs)

    @pytest.mark.asyncio
    async def test_unconstrained_run_still_emits_both(self, optimized_state):
        """I2: an unconstrained run (constraints=[]) must still emit BOTH signals.

        An empty constraints string is a valid signature value, not a sentinel —
        it must not zero out the training data for unconstrained optimizations.
        """
        optimized_state["constraints"] = []
        node = ImpactProjectorNode()
        mock_emit = AsyncMock(return_value=True)
        with patch(EMIT_TARGET, new=mock_emit):
            result = await node.execute(optimized_state)

        assert result["status"] == "completed"
        by_field = _emit_calls_by_field(mock_emit)
        required = recipient_required_input_keys("resource_optimizer")

        assert "summary_template" in by_field, "unconstrained run dropped the summary signal"
        assert "recommendation_template" in by_field, (
            "unconstrained run dropped the recommendation signal"
        )
        for field in ("summary_template", "recommendation_template"):
            kwargs = by_field[field]
            assert set(kwargs["signature_inputs"].keys()) == set(required[field])
            # The constraints field is present (per contract) but legitimately empty.
            constraint_key = "constraints_used" if field == "summary_template" else "constraints"
            assert kwargs["signature_inputs"][constraint_key] == ""
