"""#2175: scope_definer's writers must read the keys the REAL ScopeSpec carries.

``scope_builder.build_scope_spec`` names the target ``prediction_target`` and the
features ``required_features``. The agent's writers read ``target_variable`` and
``features`` instead — keys that only exist in hand-written test fixtures — so on
every real run the knowledge graph linked each Experiment to one nameless
``var:`` Variable (90/90 live) and ``ml_experiments.prediction_target`` was
written empty (692 of 1,068 rows on 2026-09-18).

The spec here is produced by the real classifier + builder nodes, so the fixture
cannot drift from the production shape again. Only the persistence seams
(repository, memory-hook class) are recorded.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.agents.ml_foundation.scope_definer.agent import ScopeDefinerAgent
from src.agents.ml_foundation.scope_definer.nodes.problem_classifier import classify_problem
from src.agents.ml_foundation.scope_definer.nodes.scope_builder import build_scope_spec
from src.repositories.ml_experiment import MLExperiment

pytestmark = pytest.mark.unit


def _real_output() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "problem_description": "Predict which HCPs will prescribe Kisqali",
        "business_objective": "Grow new prescribers",
        "target_outcome": "hcp prescribes kisqali",
        "brand": "Kisqali",
        "region": "all",
        "use_case": "commercial_targeting",
        "performance_requirements": {},
    }

    async def _build() -> Dict[str, Any]:
        state.update(await classify_problem(state))
        state.update(await build_scope_spec(state))
        return state

    final = asyncio.run(_build())
    return {
        "scope_spec": final["scope_spec"],
        "success_criteria": {"minimum_auc": 0.7},
        "experiment_id": final["scope_spec"]["experiment_id"],
        "experiment_name": final["scope_spec"]["experiment_name"],
    }


@pytest.fixture
def real_output() -> Dict[str, Any]:
    # Sync fixture: built outside the async tests' event loop.
    return _real_output()


def test_the_real_spec_has_no_target_variable_key():
    spec = _real_output()["scope_spec"]
    assert spec["prediction_target"]
    assert spec["required_features"]
    assert "target_variable" not in spec


def test_semantic_writer_receives_the_real_target_and_features():
    output = _real_output()
    with patch("src.agents.ml_foundation.scope_definer.agent.ScopeDefinerMemoryHooks") as HookCls:
        hook = HookCls.return_value
        hook.store_experiment_pattern = AsyncMock(return_value=True)
        asyncio.run(ScopeDefinerAgent()._update_semantic_memory(output))

    kwargs = hook.store_experiment_pattern.await_args.kwargs
    assert kwargs["target_variable"] == output["scope_spec"]["prediction_target"]
    assert kwargs["features"] == output["scope_spec"]["required_features"]


@pytest.mark.asyncio
async def test_new_experiment_row_gets_the_real_prediction_target(real_output):
    output = real_output
    repo = AsyncMock()
    repo.get_by_name.return_value = None
    with patch(
        "src.agents.ml_foundation.scope_definer.agent._get_experiment_repository",
        new=AsyncMock(return_value=repo),
    ):
        await ScopeDefinerAgent()._persist_scope_spec(
            output, problem_description="Predict which HCPs will prescribe Kisqali"
        )

    kwargs = repo.create_experiment.await_args.kwargs
    assert kwargs["prediction_target"] == output["scope_spec"]["prediction_target"]
    assert kwargs["description"] == "Predict which HCPs will prescribe Kisqali"


@pytest.mark.asyncio
async def test_procedural_pattern_records_the_real_target(real_output):
    memory = AsyncMock()
    with patch(
        "src.agents.ml_foundation.scope_definer.agent._get_procedural_memory",
        return_value=memory,
    ):
        await ScopeDefinerAgent()._update_procedural_memory(real_output)

    pattern = memory.store_pattern.await_args.kwargs["pattern_data"]
    assert pattern["target_variable"] == real_output["scope_spec"]["prediction_target"]


@pytest.mark.asyncio
async def test_refresh_repairs_an_empty_prediction_target(real_output):
    output = real_output
    existing = MLExperiment(
        id=uuid4(),
        experiment_name=output["experiment_name"],
        prediction_target="",
        created_by="scope_definer",
        status="completed",
    )
    repo = AsyncMock()
    repo.get_by_name.return_value = existing
    with patch(
        "src.agents.ml_foundation.scope_definer.agent._get_experiment_repository",
        new=AsyncMock(return_value=repo),
    ):
        await ScopeDefinerAgent()._persist_scope_spec(output)

    _row_id, updates = repo.update.await_args.args
    assert updates["prediction_target"] == output["scope_spec"]["prediction_target"]


def test_hand_built_spec_keeps_its_features():
    """A caller-built spec using the input names keeps both target and features."""
    output = {
        "experiment_id": "exp-legacy",
        "scope_spec": {
            "problem_type": "regression",
            "target_variable": "trx",
            "features": ["a", "b"],
        },
    }
    with patch("src.agents.ml_foundation.scope_definer.agent.ScopeDefinerMemoryHooks") as HookCls:
        hook = HookCls.return_value
        hook.store_experiment_pattern = AsyncMock(return_value=True)
        asyncio.run(ScopeDefinerAgent()._update_semantic_memory(output))

    kwargs = hook.store_experiment_pattern.await_args.kwargs
    assert kwargs["target_variable"] == "trx"
    assert kwargs["features"] == ["a", "b"]
