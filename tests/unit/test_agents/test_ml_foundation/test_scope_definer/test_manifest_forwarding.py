"""Phase B: scope_definer forwards ``feature_manifest_source`` onto scope_spec.

scope_definer is the canonical producer of ``scope_spec``. The cohort-identity
resolution happens upstream (in the pipeline, which has ``data_source``), and
scope_definer threads the resolved value through so the scope_spec it emits is
self-consistent — every consumer (data_preparer Layer-1/3, model_deployer's
regulatory manifest) reads it from the same place.

Opt-in safety: when no source is provided the field stays absent/None so
synthetic / research regimes never get cross-cohort false positives.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.ml_foundation.scope_definer.agent import ScopeDefinerAgent
from src.agents.ml_foundation.scope_definer.nodes.scope_builder import build_scope_spec

_BASE_STATE = {
    "business_objective": "Predict biologic initiation",
    "target_outcome": "initiated_biologic_180d",
    "inferred_problem_type": "binary_classification",
    "inferred_target_variable": "initiated_biologic_180d",
    "brand": "competitor",
    "region": "all",
    "use_case": "commercial_targeting",
}


@pytest.mark.asyncio
async def test_scope_builder_forwards_manifest_source_onto_scope_spec() -> None:
    state = {**_BASE_STATE, "feature_manifest_source": "optum"}
    out = await build_scope_spec(state)
    assert out["scope_spec"].get("feature_manifest_source") == "optum"


@pytest.mark.asyncio
async def test_scope_builder_leaves_manifest_unset_by_default() -> None:
    out = await build_scope_spec(dict(_BASE_STATE))
    # Absent or None — never a stray value that would engage the wrong manifest.
    assert out["scope_spec"].get("feature_manifest_source") is None


class _StopAfterCapture(Exception):
    """Stop before the real graph runs once the initial_state is captured."""


@pytest.mark.asyncio
async def test_agent_run_threads_manifest_source_into_initial_state() -> None:
    agent = ScopeDefinerAgent()
    captured: dict = {}

    async def _capture_ainvoke(initial_state):
        captured["initial_state"] = initial_state
        raise _StopAfterCapture()

    agent.graph = MagicMock()
    agent.graph.ainvoke = AsyncMock(side_effect=_capture_ainvoke)

    try:
        await agent.run(
            {
                "problem_description": "p",
                "business_objective": "b",
                "target_outcome": "t",
                "feature_manifest_source": "optum",
                "audit_workflow_id": "00000000-0000-0000-0000-000000000001",
            }
        )
    except _StopAfterCapture:
        pass

    assert captured["initial_state"].get("feature_manifest_source") == "optum"


@pytest.mark.asyncio
async def test_agent_run_manifest_source_absent_is_none() -> None:
    agent = ScopeDefinerAgent()
    captured: dict = {}

    async def _capture_ainvoke(initial_state):
        captured["initial_state"] = initial_state
        raise _StopAfterCapture()

    agent.graph = MagicMock()
    agent.graph.ainvoke = AsyncMock(side_effect=_capture_ainvoke)

    try:
        await agent.run(
            {
                "problem_description": "p",
                "business_objective": "b",
                "target_outcome": "t",
                "audit_workflow_id": "00000000-0000-0000-0000-000000000001",
            }
        )
    except _StopAfterCapture:
        pass

    assert captured["initial_state"].get("feature_manifest_source") is None
