"""Tier-0 orchestrator audit_workflow_id propagation tests (sub-shard D1.1).

These tests pin the audit-chain integrity contract surfaced by the Phase-1 D1
investigation: the orchestrator at ``src/agents/tier_0/pipeline.py`` must thread
a single workflow-level ``audit_workflow_id`` into every per-agent input dict.

Pre-D1.1 (regression target): the orchestrator minted a UUID via
``audit_service.start_workflow()`` (``pipeline.py:408``) but never threaded it
into any of the 6 ``*_input`` dicts at lines 520-998. Each per-agent State
independently minted a fresh UUID via ``Field(default_factory=uuid4)``. Joining
``audit_chain`` table records (orchestrator UUID) to per-agent output artifacts
(per-agent UUIDs) produced a fractured 1:N relationship that should be 1:1.

Post-D1.1: every ``*_input`` dict carries ``"audit_workflow_id":
audit_workflow_id`` referencing the same local variable; the orchestrator also
mints a fresh UUID when ``audit_service`` is unavailable so downstream agents
always receive a non-None UUID.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

PIPELINE_PATH = Path(__file__).parent.parent.parent / "src" / "agents" / "tier_0" / "pipeline.py"


def test_pipeline_threads_audit_workflow_id_into_all_six_input_dicts() -> None:
    """Static check: every ``*_input`` dict literal includes ``audit_workflow_id``.

    This is a regression guard — if a future refactor adds a new agent input
    or moves dict construction without threading the workflow UUID, this test
    fails loudly. Pre-D1.1 this test would have failed on all 6 dicts.
    """
    src = PIPELINE_PATH.read_text()

    expected_dict_names = [
        "scope_input",
        "data_prep_input",
        "selector_input",
        "trainer_input",
        "analyzer_input",
        "deployer_input",
    ]

    for dict_name in expected_dict_names:
        # Match `dict_name = {` through the matching closing `}`.
        # Non-greedy match across multiple lines.
        pattern = re.compile(
            rf"{re.escape(dict_name)}\s*=\s*\{{(.*?)^\s*\}}",
            re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(src)
        assert match, (
            f"Could not find ``{dict_name}`` dict literal in pipeline.py — "
            f"refactor moved or renamed it; update this test."
        )
        body = match.group(1)
        assert '"audit_workflow_id"' in body, (
            f"Dict ``{dict_name}`` does NOT thread audit_workflow_id "
            f"(D1.1 regression). Body:\n{body}"
        )


def test_pipeline_mints_audit_workflow_id_unconditionally() -> None:
    """Static check: orchestrator mints UUID even when audit_service is None.

    Pre-D1.1 the orchestrator left ``audit_workflow_id`` as ``None`` when
    audit_service initialization failed (line 401-411 try/except). Post-D1.1,
    a follow-up unconditional mint at line ~415 ensures the variable is
    always a UUID before ``PipelineResult`` is constructed.
    """
    src = PIPELINE_PATH.read_text()
    assert "if audit_workflow_id is None:" in src, (
        "Pipeline missing unconditional audit_workflow_id mint after the "
        "audit_service block (D1.1 regression). Look for the ``if "
        "audit_workflow_id is None:`` branch."
    )
    assert "audit_workflow_id = uuid.uuid4()" in src, (
        "Pipeline missing ``audit_workflow_id = uuid.uuid4()`` fallback when "
        "audit_service is unavailable (D1.1 regression)."
    )


@pytest.mark.asyncio
async def test_orchestrator_threads_audit_workflow_id_to_scope_definer() -> None:
    """Runtime check: scope_definer (first agent in the pipeline) receives
    a UUID-typed ``audit_workflow_id`` in its input_data.

    Strategy: patch ``_get_agent`` to return a mock whose ``.run`` captures
    the input dict and raises an intentional error to short-circuit the
    pipeline. Assert the captured input has a non-None UUID.
    """
    from src.agents.tier_0.pipeline import (
        MLFoundationPipeline,
        PipelineConfig,
    )

    captured: list[dict] = []

    async def capture_and_fail(input_data):
        captured.append(dict(input_data))
        raise RuntimeError("intentional stop after audit_workflow_id capture")

    mock_scope = MagicMock()
    mock_scope.run = AsyncMock(side_effect=capture_and_fail)

    config = PipelineConfig(skip_mlflow=True, skip_benchmarks=True)
    pipeline = MLFoundationPipeline(config=config)

    with patch.object(pipeline, "_get_agent", return_value=mock_scope):
        # The orchestrator catches RuntimeError from agent.run and records
        # it; it doesn't re-raise. So we do NOT use pytest.raises here.
        try:
            await pipeline.run(
                input_data={
                    "problem_description": "test",
                    "business_objective": "test",
                    "target_outcome": "test",
                    "data_source": "test",
                }
            )
        except RuntimeError:
            pass  # acceptable — orchestrator may or may not re-raise

    assert len(captured) >= 1, "scope_definer.run was never called"
    first_input = captured[0]
    assert "audit_workflow_id" in first_input, (
        "scope_input is missing audit_workflow_id (D1.1 regression)"
    )
    assert first_input["audit_workflow_id"] is not None, (
        "audit_workflow_id was None — orchestrator did not mint one when "
        "audit_service was unavailable (D1.1 regression)"
    )
    assert isinstance(first_input["audit_workflow_id"], UUID), (
        f"audit_workflow_id should be a UUID instance, got "
        f"{type(first_input['audit_workflow_id']).__name__}: "
        f"{first_input['audit_workflow_id']!r}"
    )


@pytest.mark.asyncio
async def test_orchestrator_does_not_crash_when_audit_service_none() -> None:
    """Runtime check: orchestrator initializes successfully even when
    audit_service is None and threads a freshly-minted UUID.

    Pre-D1.1, audit_workflow_id stayed ``None`` when audit_service was
    unavailable, and any downstream code reading ``result.audit_workflow_id``
    would have to handle Optional. Post-D1.1, the unconditional mint
    guarantees a UUID is set on PipelineResult before stage 1 begins.
    """
    from src.agents.tier_0.pipeline import (
        MLFoundationPipeline,
        PipelineConfig,
    )

    captured: list[dict] = []

    async def capture_and_fail(input_data):
        captured.append(dict(input_data))
        raise RuntimeError("stop after capture")

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=capture_and_fail)

    config = PipelineConfig(skip_mlflow=True, skip_benchmarks=True)
    pipeline = MLFoundationPipeline(config=config)

    # Force audit_service to None via patch.
    with (
        patch.object(pipeline, "_get_agent", return_value=mock_agent),
        patch.object(pipeline, "_get_audit_service", return_value=None),
    ):
        try:
            await pipeline.run(
                input_data={
                    "problem_description": "test",
                    "business_objective": "test",
                    "target_outcome": "test",
                    "data_source": "test",
                }
            )
        except RuntimeError:
            pass

    assert len(captured) >= 1
    audit_id = captured[0].get("audit_workflow_id")
    assert audit_id is not None, (
        "Orchestrator did not mint audit_workflow_id when audit_service was None (D1.1 regression)"
    )
    assert isinstance(audit_id, UUID)
