"""Issue #1744: the drift_monitor graph state must never carry the tier0 frame.

Sibling of #1734 (heterogeneous_optimizer, fixed in PR #1742): langgraph's
top-level ``on_chain_start`` event streams the caller's raw ``ainvoke`` input
dict BEFORE state-schema filtering, and every nested node's
``on_chain_start``/``on_chain_end`` re-serializes the full state — so a pandas
DataFrame in graph state rides every streamed event (measured on het: one
377.6 MB chat turn, eval 4.4). drift_monitor carries the same structural seam
(``DriftMonitorInput.tier0_data`` flowed straight into graph state); today only
the tier0 harness exercises it, so this is preventive parity, not a live-bloat
fix.

Fix under test: the frame is stashed in the process-local registry
(``src.utils.frame_registry``) and only a small string handle
(``tier0_frame_ref``) rides graph state; the data/model/concept drift nodes
resolve the SAME frame through the handle, so drift results are unchanged.

These tests deliberately import NOTHING from the registry module so they
collect (and fail RED for the right reason — the leak) on the unfixed code.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.agents.drift_monitor.agent import DriftMonitorAgent, DriftMonitorInput
from src.agents.drift_monitor.nodes.data_drift import DataDriftNode
from src.agents.drift_monitor.state import DriftMonitorState

# ---------------------------------------------------------------------------
# Real synthetic patient frame (no mocks at the seam under test)
# ---------------------------------------------------------------------------

_FEATURES = ["engagement_score", "rx_count", "days_since_last_visit"]


def _make_patient_frame(n: int = 240, seed: int = 1744) -> pd.DataFrame:
    """A real patient-level frame sized to clear every node's tier0 guard.

    n=240 >> data drift's ``_min_samples``=30 per half, model drift's
    ``_min_samples * 2``=60, and concept drift's ``_min_samples * 2``=100 —
    so the tier0 branches under test actually execute (a frame small enough
    to be skipped would make the leak probe vacuous). ``discontinuation_flag``
    is the outcome column concept drift's tier0 path looks for.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "patient_journey_id": [f"pj-{i:05d}" for i in range(n)],
            "engagement_score": rng.normal(0.0, 1.0, n),
            "rx_count": rng.poisson(4.0, n).astype(float),
            "days_since_last_visit": rng.normal(30.0, 10.0, n),
            "discontinuation_flag": rng.integers(0, 2, n).astype(float),
        }
    )


def _make_input(df: pd.DataFrame) -> DriftMonitorInput:
    # model_id set so the model-drift and concept-drift tier0 branches run
    # (both nodes skip outright without one); the query string deliberately
    # avoids the leak markers so the event scan below cannot be satisfied by
    # its own probe text.
    return DriftMonitorInput(
        query="Check drift in monitored engagement features",
        features_to_monitor=list(_FEATURES),
        model_id="model-1744",
        time_window="7d",
        tier0_data=df,
    )


@pytest.fixture(autouse=True)
def _release_stashed_frames():
    """Hygiene only (soft import so the RED run on unfixed code still
    collects): drop any registry entries a test left behind."""
    yield
    try:
        from src.utils.frame_registry import _clear_all_for_tests

        _clear_all_for_tests()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Schema pin — the DataFrame channel must not exist at all
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_drift_state_schema_declares_no_dataframe_channel():
    """The state schema must not declare ``tier0_data``: LangGraph materializes
    a channel per declared key, and a declared frame channel re-serializes into
    every nested on_chain_* event when this graph runs under a streaming
    callback context (#1734 mechanism). The lightweight ``tier0_frame_ref``
    handle replaces it."""
    annotations = DriftMonitorState.__annotations__
    assert "tier0_data" not in annotations, (
        "DriftMonitorState must not declare a raw-frame channel; the tier0 "
        "frame rides the process-local frame registry (#1744, sibling of #1734)"
    )
    assert "tier0_frame_ref" in annotations


# ---------------------------------------------------------------------------
# Initial-state pin — the caller boundary must carry the handle, not the frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initial_state_carries_no_dataframe_and_no_patient_rows():
    """The top-level ``on_chain_start`` for a nested ``graph.ainvoke`` carries
    the CALLER's raw input dict before schema filtering (verified against the
    installed langgraph on #1734), so the initial state built by the agent must
    already be frame-free."""
    df = _make_patient_frame()
    agent = DriftMonitorAgent(enable_mlflow=False)

    state = agent._create_initial_state(_make_input(df))

    frame_keys = [k for k, v in state.items() if isinstance(v, pd.DataFrame)]
    assert frame_keys == [], f"initial state must carry no DataFrame, found {frame_keys}"
    serialized = json.dumps(state, default=str)
    assert "patient_journey_id" not in serialized
    assert "tier0_data" not in serialized


# ---------------------------------------------------------------------------
# THE acceptance pin — real graph, real frame, streamed events
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_stream_carries_no_patient_rows_and_still_serves_real_drift():
    """Run the REAL compiled drift_monitor graph over a real 240-row frame
    through ``astream_events`` — the exact channel the AG-UI/CoAgent bridge
    consumes — and pin BOTH halves of the #1744 acceptance across ALL streamed
    event types:

    1. no emitted event carries ``patient_journey_id`` or a ``tier0_data``
       payload (nor any frame), and
    2. the drift results are still REAL values computed from the full frame
       via the tier0 passthrough (not the skipped/empty branch).
    """
    n = 240
    df = _make_patient_frame(n=n)
    agent = DriftMonitorAgent(enable_mlflow=False)
    state = agent._create_initial_state(_make_input(df))

    leaks = []
    final_output = None
    async for ev in agent.graph.astream_events(state, version="v2"):
        payload = json.dumps(ev.get("data", {}), default=str)
        if "patient_journey_id" in payload or "tier0_data" in payload:
            leaks.append((ev.get("event"), ev.get("name")))
        if ev.get("event") == "on_chain_end" and ev.get("name") == "LangGraph":
            final_output = ev["data"]["output"]

    # (1) the frontend channel: not one event may carry patient-level rows.
    assert leaks == [], (
        f"{len(leaks)} streamed events carried patient-level rows / the tier0 "
        f"frame (first 5: {leaks[:5]}) — the #1734 event-bloat seam, drift_monitor "
        f"sibling (#1744)"
    )

    # The emitted final state is frame-free under EVERY key.
    assert final_output is not None, "expected the top-level on_chain_end event"
    frame_keys = [k for k, v in final_output.items() if isinstance(v, pd.DataFrame)]
    assert frame_keys == []
    assert "tier0_data" not in final_output

    # (2) drift quality unchanged: REAL statistics computed from the REAL frame.
    # The tier0 branches actually executed — one data-drift result PER monitored
    # feature (the tier0 splitter only drops features missing from the frame).
    assert final_output["status"] == "completed"
    assert final_output["features_checked"] == len(_FEATURES)
    data_results = final_output["data_drift_results"]
    assert {r["feature"] for r in data_results} == set(_FEATURES)
    for r in data_results:
        assert math.isfinite(r["p_value"]) and 0.0 <= r["p_value"] <= 1.0
        assert math.isfinite(r["test_statistic"]) and r["test_statistic"] >= 0.0
    # model_id was supplied and n=240 clears the model-drift guard, so the
    # pseudo-score KS check produced at least the score-drift result.
    assert len(final_output["model_drift_results"]) >= 1
    assert isinstance(final_output["concept_drift_results"], list)
    score = final_output["overall_drift_score"]
    assert isinstance(score, float) and 0.0 <= score <= 1.0
    assert final_output["drift_summary"]


# ---------------------------------------------------------------------------
# Direct node invocation keeps accepting an in-dict frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_direct_node_invocation_still_accepts_in_dict_frame():
    """Library/unit-test callers hand ``node._fetch_data`` a plain dict with
    the frame in it — a function argument, never graph state (the schema does
    not declare the channel, so LangGraph drops it at every compiled-graph
    boundary; verified against the installed langgraph on #1734). That
    contract stays."""
    df = _make_patient_frame(n=120)
    node = DataDriftNode(connector=object())  # sentinel: must never be queried
    state = {
        "features_to_monitor": list(_FEATURES),
        "time_window": "7d",
        "tier0_data": df,
    }
    baseline_data, current_data = await node._fetch_data(state)
    for feature in _FEATURES:
        # Real per-feature splits derived from the in-dict frame (120 rows ->
        # 60/60 halves), not the empty arrays of the frame-missing branch.
        assert len(baseline_data[feature]) == 60
        assert len(current_data[feature]) == 60
