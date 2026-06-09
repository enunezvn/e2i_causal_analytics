"""F6 (HIGH) — tool_composer must fail CLOSED when 0/N tools succeed.

Audit finding F6: when every executed tool fails, the BUILD-RESULT block still
mapped to ``status=PARTIAL`` → ``success=True`` and synthesized a ~0.8-confidence
answer over zero successful tool outputs (fabrication, fails OPEN). The composer
must instead fail closed: ``status=FAILED``, ``success=False``, confidence 0.0, and
NOT invoke the LLM synthesizer to fabricate an answer from nothing.

Red-first per the TDD protocol.
"""

from unittest.mock import AsyncMock

import pytest

from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.models.composition_models import (
    ComposedResponse,
    CompositionStatus,
    DecompositionResult,
    ExecutionPlan,
    ExecutionTrace,
)


def _wire(composer, *, tools_executed, tools_succeeded, synth_confidence):
    """Mock the four phase handlers so execution yields the given success ratio."""
    decomp = DecompositionResult(original_query="q", sub_questions=[], decomposition_reasoning="r")
    plan = ExecutionPlan(decomposition=decomp, steps=[], tool_mappings=[], planning_reasoning="r")
    trace = ExecutionTrace(
        plan_id=plan.plan_id,
        tools_executed=tools_executed,
        tools_succeeded=tools_succeeded,
        tools_failed=tools_executed - tools_succeeded,
    )
    resp = ComposedResponse(answer="synthesized", confidence=synth_confidence)
    composer.decomposer.decompose = AsyncMock(return_value=decomp)
    composer.planner.plan = AsyncMock(return_value=plan)
    composer.executor.execute = AsyncMock(return_value=trace)
    composer.synthesizer.synthesize = AsyncMock(return_value=resp)
    return composer


@pytest.mark.asyncio
async def test_zero_tools_succeeded_fails_closed_without_fabrication():
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    _wire(composer, tools_executed=2, tools_succeeded=0, synth_confidence=0.8)

    result = await composer.compose("compare causal impact of X and predict Y")

    assert result.status == CompositionStatus.FAILED, (
        f"0/N tools succeeded must be FAILED, got {result.status}"
    )
    assert result.success is False, "0/N tools succeeded must not report success=True"
    assert result.response is not None
    assert result.response.confidence == 0.0, (
        "must not present a confident answer synthesized over zero tool outputs"
    )
    # Anti-fabrication: the LLM synthesizer must NOT be asked to invent an answer
    # when there is nothing to synthesize.
    composer.synthesizer.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_success_still_synthesizes_and_reports_partial():
    """Regression guard: when SOME tools succeed, synthesis runs and status is PARTIAL."""
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    _wire(composer, tools_executed=2, tools_succeeded=1, synth_confidence=0.6)

    result = await composer.compose("compare causal impact of X and predict Y")

    assert result.status == CompositionStatus.PARTIAL
    assert result.success is True
    composer.synthesizer.synthesize.assert_awaited()
