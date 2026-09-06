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


class _RecordingAuditService:
    """Faithful stand-in for AuditChainService: mints a workflow id the way
    ``start_workflow`` does and records every ``add_entry`` kwargs."""

    def __init__(self):
        from uuid import uuid4

        self.workflow_id = uuid4()
        self.entries = []

    def start_workflow(self, **kwargs):
        class _Entry:
            workflow_id = self.workflow_id

        return _Entry()

    def add_entry(self, **kwargs):
        self.entries.append(kwargs)
        return object()


@pytest.fixture
def recording_audit():
    from src.agents.base.audit_chain_mixin import set_audit_chain_service

    svc = _RecordingAuditService()
    set_audit_chain_service(svc)
    yield svc
    set_audit_chain_service(None)


@pytest.mark.asyncio
async def test_zero_tools_succeeded_writes_execute_error_audit_row(recording_audit):
    """The F6 fail-closed return is a FAILED run; the audit chain must say so
    with an ``execute_error`` row — the one execution-failure marker the
    /system-health and /analytics readers count (2026-09-06). The ``execute``
    row's validation_passed=False alone is a tool verdict, not a run outcome."""
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    _wire(composer, tools_executed=2, tools_succeeded=0, synth_confidence=0.8)

    result = await composer.compose("compare causal impact of X and predict Y")

    assert result.status == CompositionStatus.FAILED
    actions = [e["action_type"] for e in recording_audit.entries]
    assert "execute_error" in actions
    error_row = next(e for e in recording_audit.entries if e["action_type"] == "execute_error")
    assert error_row["validation_passed"] is False
    assert error_row["workflow_id"] == recording_audit.workflow_id


@pytest.mark.asyncio
async def test_partial_success_writes_no_error_audit_row(recording_audit):
    """A PARTIAL composition completed (synthesis ran over the tools that
    succeeded): the tool verdict stays on the ``execute`` row, no error row."""
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    _wire(composer, tools_executed=2, tools_succeeded=1, synth_confidence=0.6)

    result = await composer.compose("compare causal impact of X and predict Y")

    assert result.status == CompositionStatus.PARTIAL
    actions = [e["action_type"] for e in recording_audit.entries]
    assert not any(a.endswith("_error") for a in actions), actions
    execute_row = next(e for e in recording_audit.entries if e["action_type"] == "execute")
    assert execute_row["validation_passed"] is False


@pytest.mark.asyncio
async def test_phase_exception_writes_phase_error_audit_row(recording_audit):
    """A phase exception returns a FAILED result via ``_create_error_result``;
    the audit chain must carry a ``<phase>_error`` row for it too (codex
    iter-2), or the run reads as successful on /system-health and /analytics."""
    from src.agents.tool_composer.decomposer import DecompositionError

    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    _wire(composer, tools_executed=1, tools_succeeded=1, synth_confidence=0.6)
    composer.decomposer.decompose = AsyncMock(side_effect=DecompositionError("llm down"))

    result = await composer.compose("compare causal impact of X and predict Y")

    assert result.status == CompositionStatus.FAILED
    error_rows = [e for e in recording_audit.entries if e["action_type"].endswith("_error")]
    assert [e["action_type"] for e in error_rows] == ["decompose_error"]
    assert error_rows[0]["validation_passed"] is False
    assert error_rows[0]["workflow_id"] == recording_audit.workflow_id


@pytest.mark.asyncio
async def test_unexpected_exception_writes_compose_error_audit_row(recording_audit):
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    _wire(composer, tools_executed=1, tools_succeeded=1, synth_confidence=0.6)
    composer.planner.plan = AsyncMock(side_effect=RuntimeError("unexpected"))

    result = await composer.compose("compare causal impact of X and predict Y")

    assert result.status == CompositionStatus.FAILED
    error_rows = [e for e in recording_audit.entries if e["action_type"].endswith("_error")]
    assert [e["action_type"] for e in error_rows] == ["compose_error"]
