"""TDD for ``src.utils.frame_registry`` (issue #1734) and the handle lifecycle
of its production callers.

The registry keeps patient-level DataFrames OUT of LangGraph state: callers
stash the frame and put only an opaque string handle (``tier0_frame_ref``) in
state; nodes resolve the SAME frame through the handle; the caller releases it
when the graph run completes. On the unfixed code this whole module is absent
(these tests fail at import — the feature does not exist).
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.frame_registry import (
    _clear_all_for_tests,
    live_frame_count,
    release_frame,
    resolve_frame,
    resolve_state_frame,
    stash_frame,
    stashed_frame,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _clear_all_for_tests()
    yield
    _clear_all_for_tests()


def _df(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame({"patient_journey_id": range(n), "y": range(n)})


# ---------------------------------------------------------------------------
# Registry primitives
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stash_resolve_release_roundtrip_identity():
    df = _df()
    ref = stash_frame(df, label="t")
    assert isinstance(ref, str) and ref
    assert resolve_frame(ref) is df  # identity — the SAME frame, not a copy
    assert live_frame_count() == 1
    release_frame(ref)
    assert resolve_frame(ref) is None
    assert live_frame_count() == 0
    release_frame(ref)  # idempotent
    release_frame(None)  # None-safe
    assert resolve_frame(None) is None


@pytest.mark.unit
def test_stashed_frame_context_releases_on_exit_and_is_none_safe():
    df = _df()
    with stashed_frame(df) as ref:
        assert resolve_frame(ref) is df
    assert resolve_frame(ref) is None  # released even without explicit call

    with stashed_frame(None) as none_ref:
        assert none_ref is None
    assert live_frame_count() == 0


@pytest.mark.unit
def test_stashed_frame_releases_when_body_raises():
    df = _df()
    captured = {}
    with pytest.raises(RuntimeError):
        with stashed_frame(df) as ref:
            captured["ref"] = ref
            raise RuntimeError("graph blew up")
    assert resolve_frame(captured["ref"]) is None
    assert live_frame_count() == 0


@pytest.mark.unit
def test_resolve_state_frame_prefers_ref_then_legacy_dict():
    df = _df()
    legacy = _df(4)
    ref = stash_frame(df)
    # the ref channel (the only one that exists in compiled graph state) wins
    assert resolve_state_frame({"tier0_frame_ref": ref}) is df
    assert resolve_state_frame({"tier0_frame_ref": ref, "tier0_data": legacy}) is df
    # direct-invocation fallback: an in-dict frame handed straight to a node
    # (never reachable through a compiled graph — the schema drops the key)
    assert resolve_state_frame({"tier0_data": legacy}) is legacy
    assert resolve_state_frame({}) is None
    assert resolve_state_frame({"tier0_frame_ref": None}) is None


# ---------------------------------------------------------------------------
# Production caller lifecycle: the het agent stashes for the graph run and
# releases afterwards. The graph boundary is stubbed, and the stub ASSERTS the
# interaction from inside the run (ref present, resolves to the SAME frame,
# no DataFrame anywhere in the state it received).
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_het_agent_stashes_ref_for_graph_and_releases_after_run():
    from src.agents.heterogeneous_optimizer.agent import HeterogeneousOptimizerAgent

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "patient_journey_id": [f"pj-{i}" for i in range(16)],
            "treatment_arm": rng.integers(0, 2, 16),
            "persistent_180d": rng.integers(0, 2, 16),
            "disease_severity": rng.normal(size=16),
        }
    )
    agent = HeterogeneousOptimizerAgent(
        data_connector=object(),
        enable_memory=False,
        enable_mlflow=False,
        enable_opik=False,
    )

    seen = {}

    class _GraphStub:
        async def ainvoke(self, state):
            seen["ref"] = state.get("tier0_frame_ref")
            seen["resolved_is_input_frame"] = resolve_frame(seen["ref"]) is df
            seen["frame_keys_in_state"] = [
                k for k, v in state.items() if isinstance(v, pd.DataFrame)
            ]
            return {**state, "status": "completed"}

    agent.graph = _GraphStub()

    out = await agent.run(
        {
            "query": "segments?",
            "treatment_var": "treatment_arm",
            "outcome_var": "persistent_180d",
            "segment_vars": [],
            "effect_modifiers": ["disease_severity"],
            "data_source": "unit_test_frame",
            "tier0_data": df,
        }
    )

    assert isinstance(seen["ref"], str) and seen["ref"]
    assert seen["resolved_is_input_frame"] is True  # nodes see the SAME frame
    assert seen["frame_keys_in_state"] == []  # never the frame itself
    assert resolve_frame(seen["ref"]) is None  # released by run()'s finally
    assert live_frame_count() == 0
    assert out["status"] == "completed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_het_agent_releases_ref_when_graph_raises():
    from src.agents.heterogeneous_optimizer.agent import HeterogeneousOptimizerAgent

    df = _df(16)
    agent = HeterogeneousOptimizerAgent(
        data_connector=object(),
        enable_memory=False,
        enable_mlflow=False,
        enable_opik=False,
    )

    class _BoomGraph:
        async def ainvoke(self, state):
            raise RuntimeError("mid-graph failure")

    agent.graph = _BoomGraph()

    with pytest.raises(RuntimeError):
        await agent.run(
            {
                "query": "segments?",
                "treatment_var": "treatment_arm",
                "outcome_var": "persistent_180d",
                "segment_vars": [],
                "effect_modifiers": ["y"],
                "data_source": "unit_test_frame",
                "tier0_data": df,
            }
        )

    assert live_frame_count() == 0  # no leak on the failure path


# ---------------------------------------------------------------------------
# All three data-fetching nodes resolve the handle to the SAME frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_three_data_nodes_resolve_the_ref_to_the_same_frame():
    from src.agents.heterogeneous_optimizer.nodes.cate_estimator import (
        CATEEstimatorNode,
    )
    from src.agents.heterogeneous_optimizer.nodes.hierarchical_analyzer import (
        HierarchicalAnalyzerNode,
    )
    from src.agents.heterogeneous_optimizer.nodes.uplift_analyzer import (
        UpliftAnalyzerNode,
    )

    n = 120  # >= cate's 100-row floor and hierarchical's min_segment_size*2
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "patient_journey_id": [f"pj-{i}" for i in range(n)],
            "treatment_arm": rng.integers(0, 2, n),
            "persistent_180d": rng.integers(0, 2, n),
            "disease_severity": rng.normal(size=n),
            "severity_band": rng.choice(["low", "high"], n),
        }
    )
    ref = stash_frame(df)
    state = {
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "effect_modifiers": ["disease_severity"],
        "segment_vars": ["severity_band"],
        "data_source": "unit_test_frame",
        "filters": None,
        "tier0_frame_ref": ref,
    }

    assert await CATEEstimatorNode(None)._fetch_data(state) is df
    assert await HierarchicalAnalyzerNode(data_connector=None)._get_data(state) is df
    assert await UpliftAnalyzerNode(data_connector=None)._get_data(state) is df
