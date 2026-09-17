"""#2155: refutation refuses to critique a model the estimate did not fit.

The node reconstructs on ``state["confounders"]`` (falling back to the estimate's
``covariates_adjusted``), while the estimation node fits on the graph builder's
adjustment set and records exactly that list as ``covariates_adjusted``. The
graph builder can drop a declared confounder (e.g. one the DAG shows as a
descendant of the treatment), so the two lists can differ. Measured 2026-09-16:
they never differed on a live run (237/237 discovered DAGs, 55/55 manual-DAG
replays on real cohort frames) -- this is a fail-closed guard for a latent gap,
not a fix for an observed one.

The guard compares the reconstruction's columns with the columns the reported
estimator used: ``covariates_adjusted`` passed through the SAME #1188 rule the
reconstruction uses (an efficiency run conditions on its baselines). A mismatch,
or a result that never recorded ``covariates_adjusted`` (the estimation node has
always written it), is a ``RefutationError`` before any reconstruction runs, so
it surfaces like every other refutation refusal: ``refutation_error``,
``status="failed"``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.repositories.provenance import PROVENANCE_DROP_COLS
from tests.unit.test_agents.test_causal_impact.test_refutation_negative_control_2007 import (
    COVS,
    _frames,
    _node_state,
    _wire,
)


@pytest.fixture(scope="module")
def frames():
    return _frames()


def _estimate(state: dict, **fields) -> dict:
    state = dict(state)
    state["estimation_result"] = {**state["estimation_result"], **fields}
    return state


async def _run(monkeypatch, state):
    node = RefutationNode()
    seen = _wire(node, monkeypatch, (None, None))
    result = await node.execute(state)
    return result, seen


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refuses_when_the_graph_dropped_a_declared_confounder(frames, monkeypatch):
    """The mismatch built by the REAL graph builder: a curated-schema edge makes
    ``hcp_engagement_level`` a descendant of ``marketing_spend``, so the backdoor set
    the estimate fits on drops it while ``confounders`` still lists it."""
    frame, _ = frames
    treatment, outcome = "marketing_spend", "prescription_volume"
    declared = ["hcp_engagement_level", "region_score"]
    rng = np.random.default_rng(2155)
    dag_frame = pd.DataFrame(
        {
            treatment: rng.binomial(1, 0.5, len(frame)),
            outcome: rng.binomial(1, 0.5, len(frame)),
            "hcp_engagement_level": rng.normal(size=len(frame)),
            "region_score": rng.normal(size=len(frame)),
        }
    )
    logging.disable(logging.CRITICAL)
    try:
        graph = await GraphBuilderNode().execute(
            {
                "query": "",
                "treatment_var": treatment,
                "outcome_var": outcome,
                "confounders": declared,
                "modeled_confounders": declared,
                "anchored_confounders": [],
                "auto_discover": False,
                "data_cache": {},
            }
        )
    finally:
        logging.disable(logging.NOTSET)
    # estimation.py: covariates_adjusted = adjustment_sets[0] minus provenance columns
    fitted = [
        c for c in graph["causal_graph"]["adjustment_sets"][0] if c not in PROVENANCE_DROP_COLS
    ]
    assert fitted == ["region_score"], fitted  # the precondition this test is about

    state = _node_state(
        dag_frame,
        treatment_var=treatment,
        outcome_var=outcome,
        confounders=declared,
        estimation_data=dag_frame,
    )
    state = _estimate(state, covariates_adjusted=fitted)

    result, seen = await _run(monkeypatch, state)

    assert "recon" not in seen and "runner" not in seen
    assert result["status"] == "failed"
    details = result["refutation_error_details"]
    assert details["reason"] == "reconstruction_columns_differ_from_estimate"
    assert details["missing_from_estimate"] == ["hcp_engagement_level"]
    assert details["missing_from_reconstruction"] == []
    assert "hcp_engagement_level" in result["refutation_error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refuses_when_the_estimate_adjusted_for_more(frames, monkeypatch):
    frame, _ = frames
    state = _estimate(_node_state(frame, confounders=[COVS[0]]), covariates_adjusted=list(COVS))

    result, seen = await _run(monkeypatch, state)

    assert "recon" not in seen
    details = result["refutation_error_details"]
    assert details["reason"] == "reconstruction_columns_differ_from_estimate"
    assert details["missing_from_reconstruction"] == [COVS[1]]
    assert details["missing_from_estimate"] == []
    assert COVS[1] in result["refutation_error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refuses_a_result_that_never_recorded_its_columns(frames, monkeypatch):
    frame, _ = frames
    state = _node_state(frame)
    del state["estimation_result"]["covariates_adjusted"]

    result, seen = await _run(monkeypatch, state)

    assert "recon" not in seen
    assert result["status"] == "failed"
    assert result["refutation_error_details"]["reason"] == "estimate_columns_unrecorded"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_equal_sets_in_a_different_order_still_refute(frames, monkeypatch):
    """Positive control: the guard compares SETS (order is #2084's pin)."""
    frame, _ = frames
    state = _estimate(
        _node_state(frame, confounders=list(COVS)), covariates_adjusted=list(COVS)[::-1]
    )

    result, seen = await _run(monkeypatch, state)

    assert "refutation_error" not in result
    assert seen["recon"]["common_causes"] == list(COVS)
    assert "runner" in seen


@pytest.mark.unit
@pytest.mark.asyncio
async def test_efficiency_run_refutes_on_its_baselines(frames, monkeypatch):
    """Positive control, #1188: an RCT efficiency run records an empty backdoor
    and conditions on its baselines; both sides resolve to the baselines."""
    frame, _ = frames
    state = _estimate(
        _node_state(frame, confounders=[]),
        covariates_adjusted=[],
        adjustment_type="efficiency",
        baseline_covariates_adjusted=list(COVS),
    )

    result, seen = await _run(monkeypatch, state)

    assert "refutation_error" not in result
    assert seen["recon"]["common_causes"] == list(COVS)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validated_empty_backdoor_refutes_unadjusted(frames, monkeypatch):
    """Positive control: a randomized question with no confounders and no
    baselines reconstructs on no columns, exactly as the estimate did."""
    frame, _ = frames
    state = _estimate(_node_state(frame, confounders=[]), covariates_adjusted=[])

    result, seen = await _run(monkeypatch, state)

    assert "refutation_error" not in result
    assert seen["recon"]["common_causes"] == []
