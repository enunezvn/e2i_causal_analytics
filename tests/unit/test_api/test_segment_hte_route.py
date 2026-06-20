"""Unit tests for the Segment Analysis clinical-HTE route slice.

Covers the API-route changes for the agent-driven Segment Analysis rebuild
(design: docs/superpowers/specs/2026-06-20-segment-analysis-clinical-hte-design.md):

  * the un-drop guard — ``_execute_segment_analysis`` must map
    ``strategic_interpretation``, ``mid_responders``, ``segment_heterogeneity``
    and the hierarchical fields FROM the final graph ``result`` dict. These
    were silently dropped at the route before the rebuild (codex LOW-2).
  * the curated allowlist guard — a treatment/outcome outside the
    ``patient_journeys`` allowlist must raise ``HTTPException`` 400 (codex
    HIGH-3), not a generic 500 / silent empty.
  * ``mid_responders`` defaults to ``[]`` when the graph omits it.

Import-light per the brief: import the specific functions, NEVER
``src.api.main.app`` (it OOMs the dev box). The graph factory is monkeypatched
to a stub so no real agent / Supabase / EconML runs — these tests exercise the
route's mapping + guards only, mirroring tests/unit/test_api/test_label_gater_converters.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.routes.segments import (
    AnalysisStatus,
    ResponderType,
    RunSegmentAnalysisRequest,
    SegmentAnalysisResponse,
    _execute_segment_analysis,
    _load_segment_hte_frame,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_stub_frame(n: int = 120) -> pd.DataFrame:
    """A minimal prepared patient_journeys-shaped frame for tier0 passthrough.

    >=100 rows so cate_estimator's tier0 priority-1 path accepts it; carries the
    fixed clinical contract columns so a (stubbed) graph never needs Supabase.
    """
    return pd.DataFrame(
        {
            "treatment_arm": [i % 2 for i in range(n)],
            "persistent_180d": [(i + 1) % 2 for i in range(n)],
            "disease_severity": [i % 10 for i in range(n)],
            "engagement_score": [float(i % 5) for i in range(n)],
            "age_at_diagnosis": [40 + (i % 40) for i in range(n)],
            "academic_hcp": [i % 2 for i in range(n)],
            "ecog_performance_status": [i % 3 for i in range(n)],
            "egfr": [60 + (i % 30) for i in range(n)],
            "proteinuria_g_day": [float(i % 4) for i in range(n)],
            "ldh_ratio": [1.0 + (i % 3) for i in range(n)],
            "urticaria_severity_uas7": [i % 7 for i in range(n)],
            "geographic_region": [
                ["midwest", "south", "northeast", "west"][i % 4] for i in range(n)
            ],
            "disease_severity_band": [["low", "medium", "high"][i % 3] for i in range(n)],
            "age_band": [["<50", "50-65", ">65"][i % 3] for i in range(n)],
        }
    )


@pytest.fixture
def stub_request() -> RunSegmentAnalysisRequest:
    """A request that omits treatment/outcome (route must default them) and
    carries only the query — the clinical contract is fixed server-side."""
    return RunSegmentAnalysisRequest(
        query="Which clinical segments respond best to treatment?",
    )


@pytest.fixture
def rich_graph_result() -> dict:
    """Final graph state carrying ALL fields the route must map (the un-drop set).

    Keys mirror what the real nodes emit: profile_generator ->
    strategic_interpretation; segment_analyzer -> mid_responders / segment_comparison;
    hierarchical_analyzer -> segment_heterogeneity / overall_hierarchical_ate /
    hierarchical_segment_results / n_segments_analyzed; uplift_analyzer ->
    uplift_by_segment.
    """
    return {
        "status": "completed",
        "cate_by_segment": {},
        "overall_ate": 0.19,
        "heterogeneity_score": 0.55,
        "feature_importance": {"disease_severity": 0.6},
        "strategic_interpretation": "High-severity patients respond strongest; prioritize them.",
        "mid_responders": [
            {
                "segment_id": "sev_medium",
                "responder_type": "average",
                "cate_estimate": 0.18,
                "defining_features": [{"feature": "disease_severity_band", "value": "medium"}],
                "size": 40,
                "size_percentage": 33.3,
                "recommendation": "Maintain current targeting.",
            }
        ],
        "high_responders": [
            {
                "segment_id": "sev_high",
                "responder_type": "high",
                "cate_estimate": 0.50,
                "defining_features": [{"feature": "disease_severity_band", "value": "high"}],
                "size": 40,
                "size_percentage": 33.3,
                "recommendation": "Increase intensity.",
            }
        ],
        "low_responders": [],
        "segment_comparison": {"effect_ratio": 3.3, "mid_count": 1},
        "segment_heterogeneity": 42.0,
        "overall_hierarchical_ate": 0.21,
        "hierarchical_segment_results": [
            {
                "segment_id": "sev_high",
                "segment_name": "disease_severity_band=high",
                "cate_mean": 0.5,
            }
        ],
        "n_segments_analyzed": 3,
        "segmentation_method_used": "threshold",
        "uplift_by_segment": {"disease_severity_band": [{"segment": "high", "uplift": 0.5}]},
        "policy_recommendations": [],
        "key_insights": ["High severity drives the effect"],
        "warnings": [],
        "confidence": 0.8,
    }


def _patch_graph(result: dict):
    """Patch the graph factory so ``ainvoke`` returns ``result`` (no real agent)."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=result)
    return patch(
        "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
        return_value=mock_graph,
    )


def _patch_loader(frame: pd.DataFrame):
    """Patch the server-side loader so no real Supabase read happens."""
    return patch(
        "src.api.routes.segments._load_segment_hte_frame",
        new=AsyncMock(return_value=frame),
    )


# =============================================================================
# Un-drop guard: the route must map the new fields FROM the graph result
# =============================================================================


@pytest.mark.asyncio
async def test_execute_maps_strategic_and_hierarchical_fields(stub_request, rich_graph_result):
    """KEY un-drop guard: strategic_interpretation, mid_responders,
    segment_heterogeneity, and the hierarchical fields must survive the route
    mapping (they were dropped before the rebuild)."""
    with _patch_loader(_make_stub_frame()), _patch_graph(rich_graph_result):
        result = await _execute_segment_analysis(stub_request)

    assert isinstance(result, SegmentAnalysisResponse)
    assert result.status == AnalysisStatus.COMPLETED

    # strategic_interpretation (profile_generator) — was dropped at the route.
    assert result.strategic_interpretation == (
        "High-severity patients respond strongest; prioritize them."
    )

    # mid_responders (segment_analyzer) — converter accepts responder_type="average".
    assert len(result.mid_responders) == 1
    assert result.mid_responders[0].responder_type == ResponderType.AVERAGE
    assert result.mid_responders[0].segment_id == "sev_medium"

    # hierarchical_analyzer fields — note the result dict key is
    # ``segment_heterogeneity`` (NOT ``segment_heterogeneity_score``).
    assert result.segment_heterogeneity == 42.0
    assert result.overall_hierarchical_ate == 0.21
    assert result.n_segments_analyzed == 3
    assert result.segmentation_method_used == "threshold"
    assert result.hierarchical_segment_results is not None
    assert result.hierarchical_segment_results[0]["segment_id"] == "sev_high"

    # segment_comparison + uplift_by_segment.
    assert result.segment_comparison == {"effect_ratio": 3.3, "mid_count": 1}
    assert result.uplift_by_segment == {
        "disease_severity_band": [{"segment": "high", "uplift": 0.5}]
    }


@pytest.mark.asyncio
async def test_execute_defaults_mid_responders_to_empty_list(stub_request, rich_graph_result):
    """mid_responders defaults to [] when the graph result omits the key."""
    result_without_mid = dict(rich_graph_result)
    result_without_mid.pop("mid_responders")

    with _patch_loader(_make_stub_frame()), _patch_graph(result_without_mid):
        result = await _execute_segment_analysis(stub_request)

    assert result.mid_responders == []


@pytest.mark.asyncio
async def test_execute_defaults_optional_fields_when_absent(stub_request):
    """When the graph emits a bare completed result, the new optional fields are
    None / [] — never fabricated."""
    bare_result = {"status": "completed", "cate_by_segment": {}}

    with _patch_loader(_make_stub_frame()), _patch_graph(bare_result):
        result = await _execute_segment_analysis(stub_request)

    assert result.strategic_interpretation is None
    assert result.mid_responders == []
    assert result.segment_heterogeneity is None
    assert result.overall_hierarchical_ate is None
    assert result.n_segments_analyzed is None
    assert result.segmentation_method_used is None
    assert result.hierarchical_segment_results is None
    assert result.segment_comparison is None
    assert result.uplift_by_segment is None


@pytest.mark.asyncio
async def test_execute_passes_clinical_contract_as_tier0_initial_state(
    stub_request, rich_graph_result
):
    """The route must pass the loaded frame as tier0_data and set the fixed
    clinical contract (effect_modifiers numeric, confounders=engagement_score
    with NO X/W overlap, banded segment_vars, data_source=patient_journeys)."""
    frame = _make_stub_frame()
    captured = {}

    mock_graph = MagicMock()

    async def _capture(initial_state):
        captured.update(initial_state)
        return rich_graph_result

    mock_graph.ainvoke = AsyncMock(side_effect=_capture)

    with (
        _patch_loader(frame),
        patch(
            "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
            return_value=mock_graph,
        ),
    ):
        await _execute_segment_analysis(stub_request)

    # tier0_data is the loaded frame.
    assert captured["tier0_data"] is frame
    assert captured["data_source"] == "patient_journeys"

    # defaults applied when the request omits treatment/outcome.
    assert captured["treatment_var"] == "treatment_arm"
    assert captured["outcome_var"] == "persistent_180d"

    # effect_modifiers (X) = numeric clinical covariates, region NOT among them,
    # and engagement_score NOT among them (it is the W confounder — no X/W overlap).
    assert "geographic_region" not in captured["effect_modifiers"]
    assert set(captured["effect_modifiers"]) == {
        "disease_severity",
        "age_at_diagnosis",
        "academic_hcp",
        "ecog_performance_status",
        "egfr",
        "proteinuria_g_day",
        "ldh_ratio",
        "urticaria_severity_uas7",
    }

    # confounders (W) = pure control engagement_score; must NOT also be in X
    # (codex MED-2: no variable in both the heterogeneity matrix and the nuisance
    # controls) and must NOT be a segment var.
    assert captured["confounders"] == ["engagement_score"]
    assert "engagement_score" not in captured["effect_modifiers"]

    # segment_vars are the banded / raw categoricals.
    assert captured["segment_vars"] == [
        "disease_severity_band",
        "age_band",
        "geographic_region",
        "ecog_performance_status",
        "academic_hcp",
    ]
    assert "engagement_score" not in captured["segment_vars"]


# =============================================================================
# Curated allowlist guard (codex HIGH-3): 400, not a generic 500 / silent empty
# =============================================================================


@pytest.mark.asyncio
async def test_loader_rejects_disallowed_treatment_with_400():
    """A treatment outside the patient_journeys allowlist => HTTPException 400."""
    with pytest.raises(HTTPException) as exc_info:
        await _load_segment_hte_frame(
            brand=None,
            treatment_var="rep_visits",  # not in the curated allowlist
            outcome_var="persistent_180d",
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_loader_rejects_disallowed_outcome_with_400():
    """An outcome outside the patient_journeys allowlist => HTTPException 400."""
    with pytest.raises(HTTPException) as exc_info:
        await _load_segment_hte_frame(
            brand=None,
            treatment_var="treatment_arm",
            outcome_var="trx",  # not in the curated allowlist
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_execute_propagates_allowlist_400(rich_graph_result):
    """A disallowed treatment/outcome on the request must surface as a 400 from
    _execute_segment_analysis — NOT be swallowed into a generic 500 / empty."""
    bad_request = RunSegmentAnalysisRequest(
        query="bad",
        treatment_var="rep_visits",
        outcome_var="trx",
    )
    # Graph patched too, to prove the 400 fires BEFORE graph invocation.
    with _patch_graph(rich_graph_result):
        with pytest.raises(HTTPException) as exc_info:
            await _execute_segment_analysis(bad_request)
    assert exc_info.value.status_code == 400
