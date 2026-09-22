"""#2207: the live energy-score path records its per-estimator evaluations.

Census finding (2026-09-22): ``estimator_evaluations`` had a writer
(``EnergyScoreMLflowTracker``) that nothing on the live path instantiated. The live
energy-score path is ``EstimationNode._select_estimator_with_energy_score`` (the
causal_impact agent — chat + ``/api/causal`` — evaluating the estimator registry and
picking the best by energy score). Owner decision (a): instantiate the tracker there.

Contract proven here, through the REAL node on the explicit synthetic opt-in frame
(real econml/sklearn estimators, no mocked math):

* after every selection the node records the SelectionResult with the query's identity
  (query_id / session_id), the variables, brand and the frame's data_source;
* the recording is a side channel — a failing tracker never fails estimation;
* the node's tracker is DB-only (``enable_mlflow=False``): no MLflow run is opened on a
  chat request.

The causal-impact unit conftest neutralises the recording by default (the same
hermeticity reason as #788's memory-contribution no-op); the wiring tests below opt
back in with ``real_estimator_evaluation_recording``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.causal_impact.nodes.estimation import EstimationNode
from src.causal_engine.energy_score.estimator_selector import SelectionResult
from src.causal_engine.energy_score.mlflow_tracker import EnergyScoreMLflowTracker


@pytest.fixture()
def base_state():
    return {
        "query": "What is the impact of engagement on conversion?",
        "query_id": "q-2207",
        "session_id": "sess-2207",
        "brand": "Kisqali",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region", "hcp_specialty"],
        # explicit synthetic opt-in (anti-fabrication guard, #416/#417): real
        # estimators run against seeded synthetic data.
        "data_source": "synthetic",
        "causal_graph": {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("hcp_engagement_level", "patient_conversion_rate"),
                ("geographic_region", "hcp_engagement_level"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "digraph {}",
            "confidence": 0.85,
        },
        "errors": [],
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_selection_is_recorded_with_the_query_context(
    base_state, real_estimator_evaluation_recording
):
    tracker = MagicMock(spec=EnergyScoreMLflowTracker)
    tracker.record_evaluations.return_value = "run-2207"
    node = EstimationNode()
    with patch.object(EstimationNode, "_get_evaluation_tracker", return_value=tracker):
        out = await node.execute(base_state)

    assert out.get("estimation_result") is not None, out.get("estimation_error")
    tracker.record_evaluations.assert_called_once()
    (result,), kwargs = tracker.record_evaluations.call_args
    assert isinstance(result, SelectionResult)
    assert len(result.all_results) >= 1
    assert kwargs["query_id"] == "q-2207"
    assert kwargs["session_id"] == "sess-2207"
    assert kwargs["treatment"] == "hcp_engagement_level"
    assert kwargs["outcome"] == "patient_conversion_rate"
    assert kwargs["brand"] == "Kisqali"
    assert kwargs["data_source"] == "synthetic"


@pytest.mark.asyncio
async def test_a_failing_tracker_never_fails_estimation(
    base_state, real_estimator_evaluation_recording
):
    tracker = MagicMock(spec=EnergyScoreMLflowTracker)
    tracker.record_evaluations.side_effect = RuntimeError("db down")
    node = EstimationNode()
    with patch.object(EstimationNode, "_get_evaluation_tracker", return_value=tracker):
        out = await node.execute(base_state)
    assert out.get("estimation_result") is not None, out.get("estimation_error")
    tracker.record_evaluations.assert_called_once()


@pytest.mark.asyncio
async def test_an_unconstructible_tracker_never_fails_estimation(
    base_state, real_estimator_evaluation_recording
):
    node = EstimationNode()
    with patch.object(
        EstimationNode, "_get_evaluation_tracker", side_effect=RuntimeError("no psycopg2")
    ):
        out = await node.execute(base_state)
    assert out.get("estimation_result") is not None, out.get("estimation_error")


def test_the_nodes_tracker_is_db_only_and_never_opens_an_mlflow_run():
    node = EstimationNode()
    with patch(
        "src.causal_engine.energy_score.mlflow_tracker.EnergyScoreMLflowTracker._check_mlflow",
        side_effect=AssertionError("must not probe mlflow"),
    ):
        tracker = node._get_evaluation_tracker()
    assert isinstance(tracker, EnergyScoreMLflowTracker)
    assert tracker.enable_db_logging is True
    assert tracker._mlflow_available is False
    assert node._get_evaluation_tracker() is tracker  # built once per node
