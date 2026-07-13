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

import numpy as np
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
        # The hierarchical node emits the key 'segmentation_method' (NOT
        # '..._used'); the route maps it onto response.segmentation_method_used.
        "segmentation_method": "threshold",
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
async def test_execute_coerces_numpy_scalars_so_response_serializes(
    stub_request, rich_graph_result
):
    """Regression: pandas-derived numpy scalars (e.g. per-segment sizes/counts from
    groupby/value_counts) leaked into the ``Dict[str, Any]`` / ``List[Dict[str, Any]]``
    response fields (segment_comparison, hierarchical_segment_results, uplift_by_segment).

    ``numpy.int64`` is NOT a python ``int`` subclass, so Pydantic stored it
    un-coerced and the durable store's ``response.model_dump_json()`` raised
    ``Unable to serialize unknown type: <class 'numpy.int64'>`` -> the whole
    background analysis was marked FAILED on the live page. The route must coerce
    numpy scalars to native python so the response serializes cleanly.
    """
    result = dict(rich_graph_result)
    result["segment_comparison"] = {"effect_ratio": np.float64(3.3), "mid_count": np.int64(1)}
    result["hierarchical_segment_results"] = [
        {"segment_id": "sev_high", "size": np.int64(40), "cate_mean": np.float64(0.5)}
    ]
    result["uplift_by_segment"] = {"band": [{"segment": "high", "n": np.int64(7)}]}
    result["feature_importance"] = {"disease_severity": np.float64(0.6)}

    with _patch_loader(_make_stub_frame()), _patch_graph(result):
        resp = await _execute_segment_analysis(stub_request)

    # The durable store persists via model_dump_json(); this MUST NOT raise.
    payload = resp.model_dump_json()
    assert '"mid_count":1' in payload

    # numpy scalars are coerced to native python (round-trip + type).
    assert resp.segment_comparison["mid_count"] == 1
    assert isinstance(resp.segment_comparison["mid_count"], int)
    assert not isinstance(resp.segment_comparison["mid_count"], np.generic)
    assert resp.hierarchical_segment_results[0]["size"] == 40
    assert isinstance(resp.hierarchical_segment_results[0]["size"], int)
    assert resp.uplift_by_segment["band"][0]["n"] == 7
    assert isinstance(resp.uplift_by_segment["band"][0]["n"], int)


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
    """The route must pass the loaded frame as tier0_data and set the (Phase 2
    brand-aware) clinical contract. stub_request has brand=None (all brands), so the
    indication-specific clinical columns — NULL off-brand after gating — are excluded:
    effect_modifiers collapse to the universals, and ecog drops out of segment_vars.
    engagement_score stays the W confounder (no X/W overlap); data_source unchanged."""
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

    # brand=None: effect_modifiers (X) collapse to the universals (populated for every
    # brand). The 5 indication-specific clinical modifiers are excluded because each
    # is NULL for ~2/3 of an all-brands cohort after gating (would crash EconML).
    assert "geographic_region" not in captured["effect_modifiers"]
    assert set(captured["effect_modifiers"]) == {
        "disease_severity",
        "age_at_diagnosis",
        "academic_hcp",
    }

    # confounders (W) = pure control engagement_score; must NOT also be in X
    # (codex MED-2: no variable in both the heterogeneity matrix and the nuisance
    # controls) and must NOT be a segment var.
    assert captured["confounders"] == ["engagement_score"]
    assert "engagement_score" not in captured["effect_modifiers"]

    # segment_vars are the banded / universal categoricals; ecog (clinical) drops for
    # the all-brands cohort.
    assert captured["segment_vars"] == [
        "disease_severity_band",
        "age_band",
        "geographic_region",
        "academic_hcp",
    ]
    assert "engagement_score" not in captured["segment_vars"]


@pytest.mark.asyncio
async def test_execute_brand_scoped_contract_includes_own_clinical(rich_graph_result):
    """Phase 2: a single-brand request keeps THAT brand's clinical modifier/segment
    dimension. Kisqali -> ecog_performance_status is a valid X and segment var; the
    other brands' clinical columns (uas7, egfr, …) stay excluded (NULL for Kisqali)."""
    request = RunSegmentAnalysisRequest(
        query="Which Kisqali segments respond best?", brand="Kisqali"
    )
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
        await _execute_segment_analysis(request)

    assert set(captured["effect_modifiers"]) == {
        "disease_severity",
        "age_at_diagnosis",
        "academic_hcp",
        "ecog_performance_status",
    }
    # No off-brand clinical column leaks in.
    for off in ("egfr", "urticaria_severity_uas7", "ldh_ratio", "proteinuria_g_day"):
        assert off not in captured["effect_modifiers"]
    assert captured["segment_vars"] == [
        "disease_severity_band",
        "age_band",
        "geographic_region",
        "ecog_performance_status",
        "academic_hcp",
    ]


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


@pytest.mark.unit
def test_uplift_metrics_kept_when_auuc_is_zero():
    """A real overall_auuc of 0.0 is a valid (if poor) uplift result and must be
    surfaced — the prior `if not auuc` check dropped it as 'absent' (now wired,
    so it matters). None / missing still drops the card honestly."""
    from src.api.routes.segments import _convert_uplift_metrics

    kept = _convert_uplift_metrics(
        {
            "overall_auuc": 0.0,
            "overall_qini": 0.0,
            "targeting_efficiency": 0.5,
            "model_type_used": "random_forest",
        }
    )
    assert kept is not None and kept.overall_auuc == 0.0
    assert _convert_uplift_metrics({"overall_auuc": None}) is None
    assert _convert_uplift_metrics({}) is None


# =============================================================================
# Phase-0 DGP regression pin: adherent_180d accepted, unknown column rejected
# =============================================================================


@pytest.mark.asyncio
async def test_segment_route_accepts_adherent_180d_outcome():
    """adherent_180d is now an allowlisted outcome (Task 8, Phase-0 DGP enrichment):
    the loader must pass the allowlist guard — no 400 'not permitted' rejection.
    Unknown columns must still produce HTTPException 400.

    Uses the same _load_segment_hte_frame pattern as
    test_loader_rejects_disallowed_outcome_with_400 above.
    """
    # Positive case: adherent_180d must pass the allowlist check.  The loader
    # will proceed to a Supabase query (which may raise in the unit-test context
    # with no real DB), but it must NOT raise an HTTPException(400) from the
    # 'not permitted' allowlist guard.
    try:
        await _load_segment_hte_frame(
            brand=None,
            treatment_var="treatment_arm",
            outcome_var="adherent_180d",
        )
    except HTTPException as exc:
        if exc.status_code == 400 and "not permitted" in str(exc.detail):
            pytest.fail(
                "adherent_180d was rejected by the allowlist guard (400 'not "
                "permitted') — Task 8 Phase-0 DGP allowlist change is missing "
                "or has been reverted."
            )
        # Any other HTTPException (e.g. 503 no-rows from Supabase) is fine.
    except Exception:
        # Non-HTTP exceptions (Supabase connection, missing env var, etc.) are
        # expected in a unit-test context with no real DB — not an allowlist failure.
        pass

    # Negative case: an unknown column must still be rejected with 400.
    with pytest.raises(HTTPException) as exc_info:
        await _load_segment_hte_frame(
            brand=None,
            treatment_var="treatment_arm",
            outcome_var="made_up_outcome",
        )
    assert exc_info.value.status_code == 400
    assert "not permitted" in exc_info.value.detail
