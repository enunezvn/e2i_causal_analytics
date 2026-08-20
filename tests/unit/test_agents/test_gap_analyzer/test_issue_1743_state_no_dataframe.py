"""Issue #1743: the gap_analyzer graph state must never carry the tier0 frame.

Sibling of #1734 (heterogeneous_optimizer, fixed in PR #1742). Same structural
seam, preventive parity — no live chat caller injects a frame into gap_analyzer
today, but the mechanism is identical: langgraph's top-level ``on_chain_start``
event streams the caller's RAW ``ainvoke`` input dict BEFORE state-schema
filtering, and a declared frame channel re-serializes the full patient-level
cohort into every nested ``on_chain_*`` event on the chat path (measured on het:
one 377.6 MB SSE chat turn, eval 4.4 — ~11.6 MB of serialized ``tier0_data`` in
each of 112 events). It also violates the aggregates-only frontend contract and
makes state unserializable for any checkpointer (#1351 class).

Fix under test: the frame is stashed in the process-local registry
(``src.utils.frame_registry``) and only a small string handle
(``tier0_frame_ref``) rides graph state; gap_detector / instrument_analyzer
resolve the SAME frame through the handle, so answer quality is unchanged.

These tests deliberately import NOTHING from the registry module at module
scope so they collect (and fail RED for the right reason — the leak) on the
unfixed code.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.gap_analyzer.agent import GapAnalyzerAgent
from src.agents.gap_analyzer.nodes.gap_detector import GapDetectorNode
from src.agents.gap_analyzer.state import GapAnalyzerState

# ---------------------------------------------------------------------------
# Real synthetic patient cohort (no mocks at the seam under test)
# ---------------------------------------------------------------------------

_N_NORTHEAST = 40
_N_WEST = 20
_N = _N_NORTHEAST + _N_WEST  # 60 rows >= 50 so gap_detector takes the tier0 path


def _make_patient_frame(seed: int = 1743) -> pd.DataFrame:
    """A real patient-level frame with a PLANTED regional imbalance.

    Northeast carries 40 patients and West 20, so the derived ``trx`` metric
    (patient count per region) is exactly 40.0 / 20.0 and the vs_target
    benchmark (current * 1.20) yields two REAL gaps of 16.67% each — the
    aggregate assertions below pin computed values, not ``is not None``.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "patient_journey_id": [f"pj-{i:05d}" for i in range(_N)],
            "region": ["Northeast"] * _N_NORTHEAST + ["West"] * _N_WEST,
            "discontinuation_flag": rng.integers(0, 2, size=_N),
            "tenure_months": rng.integers(1, 48, size=_N),
        }
    )


def _make_input(df: pd.DataFrame) -> dict:
    return {
        "query": "Where are the biggest trx gaps by region?",
        "metrics": ["trx"],
        "segments": ["region"],
        "brand": "kisqali",
        "time_period": "current_quarter",
        "gap_type": "vs_target",
        "min_gap_threshold": 5.0,
        "max_opportunities": 10,
        "tier0_data": df,
    }


def _make_agent() -> GapAnalyzerAgent:
    # Trackers off; the tier0 passthrough is priority-1 in gap_detector, so the
    # (real-mode) connectors constructed at graph build time are never queried.
    return GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)


@pytest.fixture(autouse=True)
def _release_stashed_frames():
    """Hygiene only (soft import so the RED run on unfixed code still collects):
    drop any registry entries a test left behind."""
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
def test_gap_state_schema_declares_no_dataframe_channel():
    """The state schema must not declare ``tier0_data``: LangGraph materializes
    a channel per declared key, and a declared frame channel is re-serialized
    into every nested on_chain_* event on the chat path (#1734/#1743). The
    lightweight ``tier0_frame_ref`` handle replaces it."""
    annotations = GapAnalyzerState.__annotations__
    assert "tier0_data" not in annotations, (
        "GapAnalyzerState must not declare a raw-frame channel; the tier0 "
        "frame rides the process-local frame registry (#1743)"
    )
    assert "tier0_frame_ref" in annotations


# ---------------------------------------------------------------------------
# Initial-state pin — the caller boundary must carry the handle, not the frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initialize_state_carries_no_dataframe_and_no_patient_rows():
    """The top-level ``on_chain_start`` for a nested ``graph.ainvoke`` carries
    the CALLER's raw input dict before schema filtering (verified against the
    installed langgraph on #1734), so the initial state built by the agent must
    already be frame-free."""
    df = _make_patient_frame()
    agent = _make_agent()

    state = agent._initialize_state(_make_input(df))

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
async def test_graph_stream_carries_no_patient_rows_and_still_serves_real_gaps():
    """Run the REAL compiled gap_analyzer graph over a real 60-row cohort frame
    through ``astream_events`` — the exact channel the AG-UI/CoAgent bridge
    consumes — and pin BOTH halves of the #1743 acceptance across ALL streamed
    event types (chain events included):

    1. no emitted event carries ``patient_journey_id`` or a ``tier0_data``
       payload (nor any frame), and
    2. the gap aggregates are still REAL values computed from the full frame.
    """
    df = _make_patient_frame()
    agent = _make_agent()
    state = agent._initialize_state(_make_input(df))

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
        f"frame (first 5: {leaks[:5]}) — the #1734 event-bloat mechanism (#1743)"
    )

    # The emitted final state is frame-free under EVERY key.
    assert final_output is not None, "expected the top-level on_chain_end event"
    frame_keys = [k for k, v in final_output.items() if isinstance(v, pd.DataFrame)]
    assert frame_keys == []
    assert "tier0_data" not in final_output

    # (2) answer quality unchanged: REAL aggregates computed from the REAL frame.
    assert final_output["status"] == "completed"
    gaps = final_output["gaps_detected"]
    assert {(g["segment_value"], g["current_value"], g["target_value"]) for g in gaps} == {
        ("Northeast", 40.0, 48.0),
        ("West", 20.0, 24.0),
    }
    assert all(g["gap_percentage"] == pytest.approx(100.0 * 8 / 48) for g in gaps)
    assert final_output["total_gap_value"] == pytest.approx(12.0)
    assert len(final_output["roi_estimates"]) == 2
    assert final_output["executive_summary"]


# ---------------------------------------------------------------------------
# Direct node invocation keeps accepting an in-dict frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_direct_node_invocation_still_accepts_in_dict_frame():
    """Library/unit-test callers hand ``node.execute`` a plain dict with the
    frame in it — a function argument, never graph state (the schema does not
    declare the channel, so LangGraph drops it at every graph boundary;
    verified against the installed langgraph on #1734). That contract stays."""
    df = _make_patient_frame()
    node = GapDetectorNode(use_mock=True)
    state = {
        "query": "identify trx gaps",
        "metrics": ["trx"],
        "segments": ["region"],
        "brand": "kisqali",
        "time_period": "current_quarter",
        "filters": None,
        "gap_type": "vs_target",
        "min_gap_threshold": 5.0,
        "tier0_data": df,
    }
    out = await node.execute(state)  # type: ignore[arg-type]
    assert not out.get("errors")
    gaps = out["gaps_detected"]
    assert {(g["segment_value"], g["current_value"]) for g in gaps} == {
        ("Northeast", 40.0),
        ("West", 20.0),
    }
