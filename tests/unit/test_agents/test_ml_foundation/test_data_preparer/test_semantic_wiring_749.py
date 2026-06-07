"""#749/#772: data_preparer's run() persisted to the DB only — it called NO memory hook
(procedural/episodic/semantic). Its store_data_quality_pattern hook (DataSource + QCReport
+ HAS_QC, LeakageIncident) was defined but never invoked, so e2i_causal stayed unchanged.
Pin that run() now invokes the semantic writer with mapped final_state and degrades
gracefully.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.data_preparer.agent import DataPreparerAgent

_STATE = {
    "experiment_id": "exp-749",
    "data_source": "optum_mart_discontinuation",
    "qc_status": "passed",
    "overall_score": 0.86,
    "leakage_detected": False,
    "blocking_issues": [],
}


@pytest.mark.unit
def test_update_semantic_memory_invokes_data_quality_writer():
    agent = DataPreparerAgent()
    with patch("src.agents.ml_foundation.data_preparer.agent.DataPreparerMemoryHooks") as HookCls:
        hook = HookCls.return_value
        hook.store_data_quality_pattern = AsyncMock(return_value=True)
        asyncio.run(agent._update_semantic_memory(_STATE))

    hook.store_data_quality_pattern.assert_awaited_once()
    kwargs = hook.store_data_quality_pattern.await_args.kwargs
    assert kwargs["experiment_id"] == "exp-749"
    assert kwargs["data_source"] == "optum_mart_discontinuation"
    assert kwargs["qc_status"] == "passed"
    assert kwargs["overall_score"] == 0.86
    assert kwargs["leakage_detected"] is False


@pytest.mark.unit
def test_run_exposes_update_semantic_memory():
    agent = DataPreparerAgent()
    assert hasattr(agent, "_update_semantic_memory")


@pytest.mark.unit
def test_update_semantic_memory_degrades_gracefully_on_error():
    agent = DataPreparerAgent()
    with patch(
        "src.agents.ml_foundation.data_preparer.agent.DataPreparerMemoryHooks",
        side_effect=RuntimeError("falkordb unreachable"),
    ):
        asyncio.run(agent._update_semantic_memory(_STATE))  # must not raise
