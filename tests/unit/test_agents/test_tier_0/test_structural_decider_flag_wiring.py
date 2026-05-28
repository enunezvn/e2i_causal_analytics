"""D5.0 wiring tests (Track-2B-v3 Stream A).

``adaptive_structural_decider_enabled`` must propagate
config → data_prep_input → DataPreparerState so the structural-decider read at
``adaptive_validity_check.py`` (``state.get("adaptive_structural_decider_enabled",
False)``) is cohort-scoped by the per-run ``PipelineConfig``.

Before D5.0 the flag had NO production surface: the node only read it with a
``.get(..., False)`` default and the value was set ONLY in a node-level test.
These tests pin the three touch-points so the dark decider can be activated
per-cohort without flipping a global read-default.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.ml_foundation.data_preparer import DataPreparerAgent
from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)


class _StopAfterCapture(Exception):
    """Raised by a mocked downstream call once its input has been captured,
    so the test never runs the real graph / data_preparer."""


def test_pipeline_config_structural_decider_flag_defaults_false() -> None:
    # Touch-point 1: the flag exists on PipelineConfig and is dark by default.
    assert PipelineConfig().adaptive_structural_decider_enabled is False


def test_pipeline_config_structural_decider_flag_accepts_true() -> None:
    cfg = PipelineConfig(adaptive_structural_decider_enabled=True)
    assert cfg.adaptive_structural_decider_enabled is True


@pytest.mark.asyncio
async def test_run_data_preparation_threads_flag_from_config_into_data_prep_input() -> None:
    # Touch-point 2: _run_data_preparation copies config.<flag> into the
    # data_prep_input dict handed to data_preparer.run (mirrors skip_leakage_check).
    config = PipelineConfig(adaptive_structural_decider_enabled=True, enable_feast=False)
    pipeline = MLFoundationPipeline(config=config)

    result = PipelineResult(
        pipeline_run_id="pipe_test",
        status="running",
        current_stage=PipelineStage.SCOPE_DEFINITION,
        experiment_id="exp_test",
    )
    result.scope_spec = {"experiment_id": "exp_test"}

    captured: dict = {}

    async def _capture_run(data_prep_input):
        captured.update(data_prep_input)
        raise _StopAfterCapture()

    mock_dp = MagicMock()
    mock_dp.run = _capture_run
    pipeline._get_agent = MagicMock(return_value=mock_dp)

    try:
        await pipeline._run_data_preparation({"data_source": "patient_journeys"}, result, {})
    except _StopAfterCapture:
        pass

    assert captured.get("adaptive_structural_decider_enabled") is True


@pytest.mark.asyncio
async def test_agent_run_copies_flag_from_input_into_initial_state() -> None:
    # Touch-point 3: DataPreparerAgent.run copies input_data[<flag>] into the
    # initial_state DataPreparerState that graph.ainvoke (and the node) read.
    agent = DataPreparerAgent()

    captured: dict = {}

    async def _capture_ainvoke(initial_state):
        captured["initial_state"] = initial_state
        raise _StopAfterCapture()

    agent.graph = MagicMock()
    agent.graph.ainvoke = AsyncMock(side_effect=_capture_ainvoke)

    try:
        await agent.run(
            {
                "scope_spec": {"experiment_id": "exp_test", "prediction_target": "target"},
                "data_source": "patient_journeys",
                "adaptive_structural_decider_enabled": True,
            }
        )
    except Exception:
        pass

    assert captured["initial_state"].get("adaptive_structural_decider_enabled") is True


@pytest.mark.asyncio
async def test_agent_run_flag_defaults_false_when_absent() -> None:
    # Absent in input_data → False in state (dark default), never missing.
    agent = DataPreparerAgent()

    captured: dict = {}

    async def _capture_ainvoke(initial_state):
        captured["initial_state"] = initial_state
        raise _StopAfterCapture()

    agent.graph = MagicMock()
    agent.graph.ainvoke = AsyncMock(side_effect=_capture_ainvoke)

    try:
        await agent.run(
            {
                "scope_spec": {"experiment_id": "exp_test"},
                "data_source": "patient_journeys",
            }
        )
    except Exception:
        pass

    assert captured["initial_state"].get("adaptive_structural_decider_enabled") is False
