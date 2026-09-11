"""#2007 follow-up — the DISCOVERY path must fetch the negative-control outcome too.

Live cert 2026-09-11 (main ``903b7addc``, Remibrutinib discovery job 457b345f):
``psp_enrolled -> persistent_180d`` persisted its ``negative_control_outcome``
row as SKIPPED ``negative_control_column_missing`` although the registry
declares ``treatment_initiated`` for ``psp_enrolled``. Cause: the submit path
(``run_causal_agent_analysis``) passes the declared control to the loader as a
``passthrough_columns`` entry, but ``_run_discover_effects_task`` — the path
that produces every live discovery row — loaded its frame WITHOUT it, so the
split in ``_run_agent_analysis_task`` (``negative_control in df.columns``)
never fired and the node had no control frame.

Pinned here: for each candidate question the discovery task asks the loader for
exactly the registry's control (``None`` when undeclared or when the control IS
the outcome under test), hands the agent task a frame that carries that column,
and never lets it into the adjustment set.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.routes import causal as causal_routes
from src.api.schemas.causal import AgentCausalAnalysisResponse
from tests.unit.test_api.test_causal_discover_effects_select_cancel import (
    _completed_agent_response,
    _memory_store,
)

CANDIDATES = [
    # declared control (registry: copay_support -> treatment_initiated)
    causal_routes._CandidateQuestion(
        "copay_support", "adherent_180d", "Remibrutinib", ["disease_severity"]
    ),
    # declared control (psp_enrolled -> treatment_initiated) — the live miss
    causal_routes._CandidateQuestion(
        "psp_enrolled", "persistent_180d", "Remibrutinib", ["disease_severity"]
    ),
    # the control IS the outcome under test -> fail-closed None
    causal_routes._CandidateQuestion(
        "psp_enrolled", "treatment_initiated", "Remibrutinib", ["disease_severity"]
    ),
    # undeclared treatment -> None
    causal_routes._CandidateQuestion(
        "treatment_arm", "persistent_180d", "Remibrutinib", ["disease_severity"]
    ),
]

EXPECTED_PASSTHROUGH = {
    ("copay_support", "adherent_180d"): ["treatment_initiated"],
    ("psp_enrolled", "persistent_180d"): ["treatment_initiated"],
    ("psp_enrolled", "treatment_initiated"): None,
    ("treatment_arm", "persistent_180d"): None,
}


@pytest.fixture
def task_env(monkeypatch):
    store = _memory_store("test:discover-nc")
    agent_store = DurableJobStore(
        "test:agent-nc", AgentCausalAnalysisResponse, redis_factory=store._redis_factory
    )
    monkeypatch.setattr(causal_routes, "_discover_effects_store", store)
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", agent_store)

    async def identity_prerank(dataset, questions):
        return list(questions)

    monkeypatch.setattr(causal_routes, "_prerank_questions", identity_prerank)
    monkeypatch.setattr(causal_routes, "_attach_clinical_context", AsyncMock())

    loads: List[Dict[str, Any]] = []
    agent_calls: List[Dict[str, Any]] = []

    async def fake_load(**kw):
        loads.append(kw)
        cols = {kw["treatment_var"]: [0, 1, 0, 1], kw["outcome_var"]: [0, 1, 1, 0]}
        for c in kw["covariates"]:
            cols[c] = [1.0, 2.0, 3.0, 4.0]
        # The real loader appends passthrough columns to the frame but NEVER to
        # the returned expanded column list (see _load_agent_estimation_frame).
        for c in kw.get("passthrough_columns") or []:
            cols[c] = [1, 0, 1, 0]
        df = pd.DataFrame(cols)
        return df, [kw["treatment_var"], kw["outcome_var"], *kw["covariates"]]

    async def fake_agent(aid, req, df, cov, data_source):
        agent_calls.append(
            {"pair": (req.treatment_var, req.outcome_var), "columns": list(df.columns), "cov": cov}
        )
        await agent_store.set(
            aid, _completed_agent_response(aid, req.treatment_var, req.outcome_var)
        )

    monkeypatch.setattr(causal_routes, "_load_agent_estimation_frame", fake_load)
    monkeypatch.setattr(causal_routes, "_run_agent_analysis_task", fake_agent)
    return {"store": store, "loads": loads, "agent_calls": agent_calls}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_discovery_task_fetches_the_declared_control_as_a_passthrough_column(task_env):
    await causal_routes._run_discover_effects_task(
        "job-nc", "patient_journeys", list(CANDIDATES), "synthetic", "Remibrutinib"
    )
    job = await task_env["store"].get("job-nc")
    assert job is not None and job.status == "completed" and job.completed == 4

    by_pair = {(kw["treatment_var"], kw["outcome_var"]): kw for kw in task_env["loads"]}
    assert set(by_pair) == set(EXPECTED_PASSTHROUGH)
    for pair, expected in EXPECTED_PASSTHROUGH.items():
        # RED on main 903b7addc: the discovery loader call has no passthrough_columns kwarg.
        assert by_pair[pair].get("passthrough_columns") == expected, pair

    agent_by_pair = {c["pair"]: c for c in task_env["agent_calls"]}
    for pair, expected in EXPECTED_PASSTHROUGH.items():
        cols = agent_by_pair[pair]["columns"]
        if expected:
            assert expected[0] in cols, (pair, cols)  # the node can split it off
        else:
            assert "treatment_initiated" not in cols or pair[1] == "treatment_initiated"
        # The control never enters the adjustment set.
        assert agent_by_pair[pair]["cov"] == ["disease_severity"], pair
