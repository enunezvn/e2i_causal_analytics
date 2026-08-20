"""Issue #1734: the het graph state must never carry the patient-level tier0 frame.

Measured defect (post1730 full eval, raw record question_id 4.4, 377.6 MB turn):
the heterogeneous_optimizer graph runs NESTED under the checkpointed chatbot
graph on the chat path (checkpoint_ns ``tools:...|dispatch:...|<node>:...``), so
``astream_events`` ``on_chain_start``/``on_chain_end`` for EVERY het node carries
the full het state — including the serialized 8,638-row ``tier0_data`` cohort
frame (~11.6 MB of each ~11.9 MB event) — through the AG-UI bridge to the
browser as RAW (112 events, 294.6 MB) and STATE_SNAPSHOT (23 events, 82.9 MB)
events. Patient-level rows (``patient_journey_id``) must never leave the
backend: chat clients receive aggregates only.

Fix under test: the frame is stashed in a process-local registry
(``src.utils.frame_registry``) and only a small string handle
(``tier0_frame_ref``) rides graph state; the CATE / uplift / hierarchical nodes
resolve the SAME frame through the handle, so answer quality is unchanged.

These tests deliberately import NOTHING from the new registry module so they
collect (and fail RED for the right reason — the leak) on the unfixed code.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.agents.heterogeneous_optimizer.agent import HeterogeneousOptimizerAgent
from src.agents.heterogeneous_optimizer.nodes.cate_estimator import CATEEstimatorNode
from src.agents.heterogeneous_optimizer.nodes.profile_generator import (
    ProfileGeneratorNode,
)
from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState

# ---------------------------------------------------------------------------
# Real synthetic patient cohort (no mocks at the seam under test)
# ---------------------------------------------------------------------------


def _make_patient_frame(n: int = 240, seed: int = 1734) -> pd.DataFrame:
    """A real patient-level frame with a PLANTED heterogeneous treatment effect.

    Treatment helps high-severity patients more (+0.25 pp on top of a +0.05
    base uplift), so the CATE pipeline has genuine signal to recover — the
    aggregate assertions below pin real computed values, not ``is not None``.
    """
    rng = np.random.default_rng(seed)
    severity = rng.normal(5.0, 2.0, n)
    treatment = rng.integers(0, 2, n)
    p = 0.35 + 0.05 * treatment + 0.25 * treatment * (severity > 5.0)
    outcome = rng.binomial(1, np.clip(p, 0.0, 1.0))
    return pd.DataFrame(
        {
            "patient_journey_id": [f"pj-{i:05d}" for i in range(n)],
            "treatment_arm": treatment,
            "persistent_180d": outcome,
            "disease_severity": severity,
            "age_at_diagnosis": rng.normal(52.0, 9.0, n),
            "engagement_score": rng.normal(0.0, 1.0, n),
            "severity_band": np.where(severity > 5.0, "high", "low"),
        }
    )


def _make_input(df: pd.DataFrame) -> dict:
    return {
        "query": "Which patient segments respond most to treatment?",
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "segment_vars": ["severity_band"],
        "effect_modifiers": ["disease_severity", "age_at_diagnosis", "engagement_score"],
        "data_source": "unit_test_frame",
        "filters": None,
        "tier0_data": df,
        "n_estimators": 50,
        "min_samples_leaf": 10,
    }


def _make_agent() -> HeterogeneousOptimizerAgent:
    # data_connector=object(): the factory must NOT resolve a real Supabase
    # connector; the tier0 passthrough is priority-1 in every data-fetching node.
    return HeterogeneousOptimizerAgent(
        data_connector=object(),
        enable_memory=False,
        enable_mlflow=False,
        enable_opik=False,
    )


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
def test_het_state_schema_declares_no_dataframe_channel():
    """The state schema must not declare ``tier0_data``: LangGraph materializes
    a channel per declared key, and a declared frame channel is re-serialized
    into every nested on_chain_* event on the chat path (#1734). The
    lightweight ``tier0_frame_ref`` handle replaces it."""
    annotations = HeterogeneousOptimizerState.__annotations__
    assert "tier0_data" not in annotations, (
        "HeterogeneousOptimizerState must not declare a raw-frame channel; "
        "the tier0 frame rides the process-local frame registry (#1734)"
    )
    assert "tier0_frame_ref" in annotations


# ---------------------------------------------------------------------------
# Initial-state pin — the caller boundary must carry the handle, not the frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initial_state_carries_no_dataframe_and_no_patient_rows():
    """The top-level ``on_chain_start`` for a nested ``graph.ainvoke`` carries
    the CALLER's raw input dict before schema filtering (verified against the
    installed langgraph), so the initial state built by the agent must already
    be frame-free."""
    df = _make_patient_frame()
    agent = _make_agent()

    state = agent._build_initial_state(_make_input(df), session_id="s-1734-init")

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
async def test_graph_stream_carries_no_patient_rows_and_still_serves_real_cate(
    monkeypatch,
):
    """Run the REAL compiled het graph over a real 240-row cohort frame through
    ``astream_events`` — the exact channel the AG-UI/CoAgent bridge consumes —
    and pin BOTH halves of the #1734 acceptance across ALL streamed event
    types (chain events included, not just STATE_SNAPSHOT translations):

    1. no emitted event carries ``patient_journey_id`` or a ``tier0_data``
       payload (nor any frame), and
    2. the CATE aggregates are still REAL values computed from the full frame.
    """

    # External boundaries only (never the state/streaming seam under test):
    # - _llm_interpretation -> None engages the node's HONEST factual fallback,
    #   itself a real production branch (CI / no API key), and avoids the
    #   ~714MB eager dspy import on this memory-constrained box.
    # - _collect_dspy_signal is fire-and-forget training telemetry.
    async def _no_llm(self, state):
        return None

    async def _no_signal(self, *args, **kwargs):
        return None

    monkeypatch.setattr(ProfileGeneratorNode, "_llm_interpretation", _no_llm)
    monkeypatch.setattr(ProfileGeneratorNode, "_collect_dspy_signal", _no_signal)

    n = 240
    df = _make_patient_frame(n=n)
    agent = _make_agent()
    state = agent._build_initial_state(_make_input(df), session_id="s-1734-stream")

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
        f"frame (first 5: {leaks[:5]}) — the 377.6 MB eval-4.4 defect (#1734)"
    )

    # The emitted final state is frame-free under EVERY key.
    assert final_output is not None, "expected the top-level on_chain_end event"
    frame_keys = [k for k, v in final_output.items() if isinstance(v, pd.DataFrame)]
    assert frame_keys == []
    assert "tier0_data" not in final_output

    # (2) answer quality unchanged: REAL aggregates computed from the REAL frame.
    assert final_output["status"] == "completed"
    ate = final_output["overall_ate"]
    assert isinstance(ate, float) and math.isfinite(ate)
    cate = final_output["cate_by_segment"]["severity_band"]
    assert {r["segment_value"] for r in cate} == {"high", "low"}
    assert sum(int(r["sample_size"]) for r in cate) == n
    assert final_output["heterogeneity_score"] is not None
    assert final_output["executive_summary"]


# ---------------------------------------------------------------------------
# Direct node invocation keeps accepting an in-dict frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_direct_node_invocation_still_accepts_in_dict_frame():
    """Library/unit-test callers hand ``node.execute``/``_fetch_data`` a plain
    dict with the frame in it — a function argument, never graph state (the
    schema does not declare the channel, so LangGraph drops it at every graph
    boundary; verified against the installed langgraph). That contract stays."""
    df = _make_patient_frame(n=120)
    node = CATEEstimatorNode(None)
    state = {
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "effect_modifiers": ["disease_severity", "age_at_diagnosis", "engagement_score"],
        "segment_vars": ["severity_band"],
        "data_source": "unit_test_frame",
        "filters": None,
        "tier0_data": df,
    }
    out = await node._fetch_data(state)
    assert out is df
