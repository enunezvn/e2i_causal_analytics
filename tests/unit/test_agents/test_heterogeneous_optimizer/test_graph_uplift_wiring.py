"""Uplift node wiring (segment-analysis clinical-HTE rebuild).

The uplift_analyzer node was fully coded but NEVER wired into the compiled graph
(graph.py imported five nodes; uplift was not among them), so `overall_auuc` was
never populated and the page's Uplift tab was structurally empty on any substrate.
This wires it as a NON-FATAL complementary step. Tests stay light: graph-structure
introspection + the non-fatal wrapper + the tier0_data short-circuit (no full-graph
CForest invocation).
"""

import pandas as pd
import pytest

from src.agents.heterogeneous_optimizer.graph import (
    _run_uplift_nonfatal,
    create_heterogeneous_optimizer_graph,
)
from src.agents.heterogeneous_optimizer.nodes.uplift_analyzer import UpliftAnalyzerNode


def _node_names(compiled) -> set:
    return set(compiled.get_graph().nodes.keys())


@pytest.mark.unit
def test_uplift_node_is_wired_when_enabled():
    # Pass an explicit dummy connector so the factory does not attempt real
    # Supabase resolution; we only assert graph structure here.
    g = create_heterogeneous_optimizer_graph(data_connector=object(), enable_uplift=True)
    names = _node_names(g)
    assert "uplift_analysis" in names, "uplift node must be wired into the graph"
    # sanity: the rest of the pipeline is still present
    assert {"estimate_cate", "analyze_segments", "learn_policy", "generate_profiles"} <= names


@pytest.mark.unit
def test_uplift_node_absent_when_disabled():
    g = create_heterogeneous_optimizer_graph(data_connector=object(), enable_uplift=False)
    assert "uplift_analysis" not in _node_names(g)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_uplift_nonfatal_swallows_exceptions():
    class _Boom:
        async def execute(self, state):
            raise RuntimeError("no data source")

    out = await _run_uplift_nonfatal(_Boom(), {"treatment_var": "t", "outcome_var": "y"})
    # Degrades to a warning; never re-raises (CATE/responder/policy must survive).
    assert "warnings" in out and out["warnings"]
    assert "status" not in out  # must NOT mark the run failed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_data_prefers_tier0_passthrough_over_failclosed():
    # No connector + tier0_data present → must use tier0_data, NOT raise the
    # fail-closed RuntimeError. This is how the page feeds the real gold-standard
    # frame to uplift without a connector round-trip.
    node = UpliftAnalyzerNode(data_connector=None)
    frame = pd.DataFrame(
        {
            "treatment_arm": [0, 1, 0, 1],
            "persistent_180d": [0, 1, 0, 1],
            "disease_severity": [3.0, 8.0, 4.0, 9.0],
        }
    )
    state = {
        "tier0_data": frame,
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "effect_modifiers": ["disease_severity"],
        "segment_vars": ["disease_severity"],
        "data_source": "patient_journeys",
        "filters": None,
    }
    out = await node._get_data(state)
    assert out is frame


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uplift_binary_treatment_produces_metrics_not_out_of_bounds():
    """Regression: with binary treatment {0,1}, the node left ``UpliftConfig.control_name``
    at its default ``"control"`` — which matches neither ``"0"`` nor ``"1"``. CausalML
    then treated BOTH values as treatment groups and returned a 2-D ``(n, 2)`` uplift
    prediction. The metric / segment / targeting consumers flatten ``uplift_scores`` to
    ``2n`` and index the ``n``-length treatment array, raising
    ``index <2n-ish> is out of bounds for axis 0 with size <n>`` -> uplift failed and the
    live page's uplift came back empty. The node must identify the control group and
    produce per-unit (1-D) uplift aligned to ``n`` so AUUC/Qini compute.
    """
    import numpy as np

    n = 240
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "treatment_arm": rng.integers(0, 2, n),
            "persistent_180d": rng.integers(0, 2, n),
            "disease_severity": rng.normal(size=n),
            "engagement_score": rng.normal(size=n),
            "age_at_diagnosis": rng.normal(size=n),
            "disease_severity_band": rng.choice(["low", "medium", "high"], n),
        }
    )
    state = {
        "status": "completed",
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "effect_modifiers": ["disease_severity", "engagement_score", "age_at_diagnosis"],
        "segment_vars": ["disease_severity_band"],
        "tier0_data": df,
        "data_source": "patient_journeys",
        "filters": None,
    }
    node = UpliftAnalyzerNode(model_type="random_forest", n_estimators=10, max_depth=4)
    result = await node.execute(state)

    # Uplift must have SUCCEEDED — before the fix this hit the out-of-bounds and
    # returned the failed/empty path (no auuc, "failed" warning).
    assert "Uplift analysis failed - continuing with CATE results only" not in result.get(
        "warnings", []
    )
    assert result.get("overall_auuc") is not None
    assert result.get("uplift_by_segment")
