"""B2: explainer recipient self-emission + optimizable-template consume wiring.

These tests prove that the NarrativeGeneratorNode:
  (a) CONSUMES the explainer's optimizable prompt getters (so the
      feedback_learner-optimized templates are actually used to source the
      final text), with the node's output SHAPE unchanged, and
  (b) EMITS recipient training signals via
      ``feedback_learner.recipient_emit.emit_recipient_signal`` keyed by the
      SIGNATURE input_fields (the B1-B4 contract, discoverable via
      ``recipient_required_input_keys('explainer')``), with reward in [0, 1],
  (c) is robust: an emit failure does NOT break narrative generation.

Fully offline: no real LM, no DB. ``emit_recipient_signal`` is patched with a
recording stub; the dspy getters run against the in-process default templates.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from src.agents.explainer.nodes.narrative_generator import NarrativeGeneratorNode
from src.agents.feedback_learner.recipient_optimizer import recipient_required_input_keys

# The recipient-emit symbol AS IMPORTED INTO the node module (patch target).
EMIT_TARGET = "src.agents.explainer.nodes.narrative_generator.emit_recipient_signal"


def _reasoned_state() -> Dict[str, Any]:
    """A minimal post-reasoning explainer state with insights to narrate."""
    return {
        "query": "What is driving sales performance?",
        "analysis_results": [{"agent": "causal_impact", "analysis_type": "effect_estimation"}],
        "user_expertise": "analyst",
        "output_format": "narrative",
        "focus_areas": ["sales"],
        "extracted_insights": [
            {
                "insight_id": "1",
                "category": "finding",
                "statement": "Marketing campaign shows 23% sales uplift with high confidence",
                "supporting_evidence": ["Source: causal_impact"],
                "confidence": 0.89,
                "priority": 1,
                "actionability": "immediate",
            },
            {
                "insight_id": "2",
                "category": "recommendation",
                "statement": "Scale the campaign to adjacent regions",
                "supporting_evidence": ["Source: gap_analyzer"],
                "confidence": 0.80,
                "priority": 2,
                "actionability": "short_term",
            },
        ],
        "key_themes": ["sales_uplift", "regional_opportunity"],
        "status": "generating",
        "errors": [],
        "warnings": [],
    }


class _Recorder:
    """Awaitable recording stub for emit_recipient_signal."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> bool:
        self.calls.append(kwargs)
        return True


@pytest.mark.asyncio
async def test_consume_getter_is_invoked_and_shape_preserved(monkeypatch):
    """The optimizable getter is consumed AND output shape is unchanged."""
    rec = _Recorder()
    monkeypatch.setattr(EMIT_TARGET, rec)

    # Spy on the executive-summary getter to prove it is consumed.
    import src.agents.explainer.dspy_integration as di

    integration = di.get_explainer_dspy_integration()
    seen: Dict[str, int] = {"exec": 0}
    orig = integration.get_executive_summary_prompt

    def _spy(*args, **kwargs):
        seen["exec"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(integration, "get_executive_summary_prompt", _spy)

    node = NarrativeGeneratorNode(use_llm=False)
    result = await node.execute(_reasoned_state())

    # Getter consumed (optimizable template actually used).
    assert seen["exec"] >= 1

    # Output shape preserved: same required string keys, completed status.
    assert result["status"] == "completed"
    assert isinstance(result["executive_summary"], str) and result["executive_summary"]
    assert isinstance(result["detailed_explanation"], str) and result["detailed_explanation"]
    assert isinstance(result["narrative_sections"], list) and result["narrative_sections"]


@pytest.mark.asyncio
async def test_emit_contract_keys_and_reward(monkeypatch):
    """emit called with agent_name='explainer', valid template_field, exact keys, reward in [0,1]."""
    rec = _Recorder()
    monkeypatch.setattr(EMIT_TARGET, rec)

    node = NarrativeGeneratorNode(use_llm=False)
    await node.execute(_reasoned_state())

    assert rec.calls, "expected at least one recipient signal emission"

    required = recipient_required_input_keys("explainer")
    for call in rec.calls:
        assert call["agent_name"] == "explainer"
        field = call["template_field"]
        assert field in required, f"unexpected template_field {field}"
        # signature_inputs keys EXACTLY match the signature's input_fields.
        assert set(call["signature_inputs"].keys()) == set(required[field]), (
            f"{field}: keys {sorted(call['signature_inputs'])} != "
            f"required {sorted(required[field])}"
        )
        # generated_output is the actual produced text.
        assert isinstance(call["generated_output"], str) and call["generated_output"]
        # reward is a deterministic heuristic in [0, 1].
        assert 0.0 <= float(call["reward"]) <= 1.0


@pytest.mark.asyncio
async def test_emit_failure_does_not_break_node(monkeypatch):
    """A raising emit must not break narrative generation (best-effort)."""

    async def _boom(**_kwargs: Any) -> bool:
        raise RuntimeError("simulated emit outage")

    monkeypatch.setattr(EMIT_TARGET, _boom)

    node = NarrativeGeneratorNode(use_llm=False)
    result = await node.execute(_reasoned_state())

    assert result["status"] == "completed"
    assert result["executive_summary"]
    assert result["detailed_explanation"]


def test_signal_reward_heuristic_bounds():
    """_signal_reward is deterministic and bounded to [0, 1]."""
    from src.agents.explainer.nodes.narrative_generator import _signal_reward

    inputs = {"analysis_results": "uplift", "user_expertise": "analyst"}
    good = _signal_reward(
        "A thorough analyst-facing summary about the uplift finding with detail.", inputs
    )
    empty = _signal_reward("", inputs)
    assert 0.0 <= good <= 1.0
    assert 0.0 <= empty <= 1.0
    assert good > empty
    # Deterministic.
    assert good == _signal_reward(
        "A thorough analyst-facing summary about the uplift finding with detail.", inputs
    )


@pytest.mark.asyncio
async def test_emit_skipped_when_no_insights(monkeypatch):
    """Failed/empty paths must not crash; node still returns a state."""
    rec = _Recorder()
    monkeypatch.setattr(EMIT_TARGET, rec)

    state = _reasoned_state()
    state["status"] = "failed"  # already-failed short-circuit
    node = NarrativeGeneratorNode(use_llm=False)
    result = await node.execute(state)
    assert result["status"] == "failed"
    # No emission on the short-circuit path.
    assert not rec.calls


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(test_consume_getter_is_invoked_and_shape_preserved(pytest))  # type: ignore[arg-type]
