"""Shard 08 T6 — faithful resource_optimizer reallocation.

Proves the assumption that high-CATE -> higher expected_response coefficient ->
the optimal solution shifts budget there, end-to-end through the real PuLP solver
(returns solver_status='optimal' when feasible). Gated E2I_DB_INTEGRATION=1, -n0,
OOM-safe; runs the agent in-process (no DB needed).

STALE-PLAN CORRECTION (verified against state.py + agent.py, do not guess):
``ResourceOptimizerOutput.optimal_allocations`` is a List[AllocationResult] where
AllocationResult is a TypedDict (subscript, not attribute) whose allocation field
is ``optimized_allocation`` — NOT ``recommended_allocation`` (that name is only on
the route's AllocationResult, not the agent Output). entity_id is also a dict key.
"""

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful integration: set E2I_DB_INTEGRATION=1",
)


@pytest.mark.asyncio
async def test_optimal_status_reallocates_toward_high_cate():
    from src.agents.resource_optimizer.agent import ResourceOptimizerAgent
    from src.ml.synthetic.artifacts.allocation_builder import targets_from_cate_frame

    cate = pd.DataFrame(
        {
            "hcp_id": ["h_high", "h_low", "h_mid"],
            "cate_estimate": [1.5, -0.2, 0.4],
            "current_spend": [10000.0, 10000.0, 10000.0],  # equal start
            "is_synthetic": [True, True, True],
        }
    )
    targets, budget = targets_from_cate_frame(cate)
    constraints = [{"constraint_type": "budget", "value": budget, "scope": "global"}]
    agent = ResourceOptimizerAgent(enable_opik=False, enable_memory=False)
    out = await agent.optimize(
        allocation_targets=targets,
        constraints=constraints,
        resource_type="budget",
        objective="maximize_outcome",
    )
    assert out.solver_status == "optimal", out.solver_status
    alloc = {a["entity_id"]: a["optimized_allocation"] for a in (out.optimal_allocations or [])}
    assert alloc["h_high"] > alloc["h_low"], "budget must shift to the high-CATE HCP"
