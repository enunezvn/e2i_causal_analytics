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

import contextlib
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.dependencies import compute as compute_mod
from src.api.dependencies.compute import HeavyComputeSaturated
from src.api.routes.segments import (
    ResponderType,
    RunSegmentAnalysisRequest,
    SegmentAnalysisResponse,
    SegmentAnalysisStatus,
    _DurableAnalysesStore,
    _execute_segment_analysis,
    _load_segment_hte_frame,
    _run_segment_analysis_task,
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


@contextlib.contextmanager
def _saturate_heavy_compute() -> Iterator[None]:
    """Occupy the per-loop heavy-compute limiter so the NEXT ``heavy_compute_slot()``
    enter rejects fast (``HeavyComputeSaturated``) without any fit running.

    This is the honest way to force saturation used by the OOM guard: the limiter
    (default ``max_concurrency=1``) is a plain in-flight counter keyed by the
    running event loop, so acquiring it once here leaves the guarded path's own
    ``acquire()`` over budget. Must run inside a live loop (the limiter resolves
    ``asyncio.get_running_loop()``), i.e. inside the ``async def`` test body.
    """
    compute_mod._reset_limiter_cache_for_tests()
    limiter = compute_mod.get_heavy_compute_limiter()
    limiter.acquire()  # occupy the only slot -> now saturated
    try:
        yield
    finally:
        limiter.release()
        compute_mod._reset_limiter_cache_for_tests()


async def _failing_redis_factory() -> Any:
    """A Redis factory that always fails, forcing the analyses store onto its
    in-process fallback so background-task tests never touch a real Redis."""
    raise ConnectionError("no redis in unit test")


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
    assert result.status == SegmentAnalysisStatus.COMPLETED

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
    """The route must hand the loaded frame to the graph via the frame-registry
    handle (#1734 — the frame itself must never enter graph state: a nested
    streamed run re-serializes state into every on_chain_* event, the 377.6 MB
    eval-4.4 turn) and set the (Phase 2 brand-aware) clinical contract.
    stub_request has brand=None (all brands), so the indication-specific
    clinical columns — NULL off-brand after gating — are excluded:
    effect_modifiers collapse to the universals, and ecog drops out of
    segment_vars. engagement_score stays the W confounder (no X/W overlap);
    data_source unchanged.

    What the pre-#1734 ``captured["tier0_data"] is frame`` assertion protected
    — the nodes consume EXACTLY the server-side loaded banded frame — is
    preserved by resolving the handle INSIDE the (stubbed) graph run and
    asserting identity."""
    # Local import: this module must stay collectible on pre-#1734 checkouts
    # (the registry ships with the fix).
    from src.utils.frame_registry import resolve_frame

    frame = _make_stub_frame()
    captured = {}
    in_run = {}

    mock_graph = MagicMock()

    async def _capture(initial_state):
        captured.update(initial_state)
        # The handle must resolve to the SAME frame WHILE the graph runs
        # (the route releases it when the run completes).
        in_run["resolved_is_loaded_frame"] = (
            resolve_frame(initial_state.get("tier0_frame_ref")) is frame
        )
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

    # #1734: no DataFrame under ANY key of the state handed to the graph — the
    # frame travels via the registry handle, which resolved to the exact
    # server-side loaded frame during the run and is released afterwards.
    assert "tier0_data" not in captured
    assert [k for k, v in captured.items() if isinstance(v, pd.DataFrame)] == []
    assert in_run["resolved_is_loaded_frame"] is True
    assert resolve_frame(captured["tier0_frame_ref"]) is None  # released
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


# =============================================================================
# OOM guard (#1293): bound the background fit with heavy_compute_slot()
# =============================================================================


@pytest.mark.asyncio
async def test_execute_rejects_fast_when_heavy_compute_saturated(stub_request, rich_graph_result):
    """With the per-worker heavy-compute slot saturated, _execute_segment_analysis
    raises HeavyComputeSaturated on slot ENTER — the graph fit never runs (reject
    fast, nothing queued). The Tier-0 frame load still happens (it is light I/O
    OUTSIDE the slot), proving the guard boundary sits around the fit, not the load."""
    loader = AsyncMock(return_value=_make_stub_frame())
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=rich_graph_result)

    with (
        patch("src.api.routes.segments._load_segment_hte_frame", new=loader),
        patch(
            "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
            return_value=mock_graph,
        ),
    ):
        with _saturate_heavy_compute():
            with pytest.raises(HeavyComputeSaturated):
                await _execute_segment_analysis(stub_request)

    # Frame load ran OUTSIDE the slot (not blocked); the heavy fit was bounded out.
    loader.assert_awaited_once()
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_execute_happy_path_unaffected_by_slot_guard(stub_request, rich_graph_result):
    """The heavy_compute_slot guard must not break the happy path: with a free slot
    the (mocked) graph still runs, a COMPLETED response comes back, and the slot is
    released (in_flight returns to 0 — no leak)."""
    compute_mod._reset_limiter_cache_for_tests()
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=rich_graph_result)

    with (
        _patch_loader(_make_stub_frame()),
        patch(
            "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
            return_value=mock_graph,
        ),
    ):
        result = await _execute_segment_analysis(stub_request)

    assert result.status == SegmentAnalysisStatus.COMPLETED
    mock_graph.ainvoke.assert_awaited_once()
    assert compute_mod.get_heavy_compute_limiter().in_flight == 0
    compute_mod._reset_limiter_cache_for_tests()


@pytest.mark.asyncio
async def test_background_task_records_failed_on_saturation(stub_request):
    """When the slot is saturated, the background task records a FAILED analysis
    with a SPECIFIC 'capacity saturated' warning (reject fast) — not a generic
    internal-error warning and not a record stuck in ESTIMATING."""
    store = _DurableAnalysesStore(redis_factory=_failing_redis_factory)
    analysis_id = "seg_saturation_probe"
    await store.set(
        analysis_id,
        SegmentAnalysisResponse(
            analysis_id=analysis_id,
            status=SegmentAnalysisStatus.PENDING,
            question_type=stub_request.question_type,
        ),
    )

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"status": "completed", "cate_by_segment": {}})

    with (
        patch("src.api.routes.segments._analyses_store", store),
        _patch_loader(_make_stub_frame()),
        patch(
            "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
            return_value=mock_graph,
        ),
    ):
        with _saturate_heavy_compute():
            await _run_segment_analysis_task(analysis_id=analysis_id, request=stub_request)

    stored = await store.get(analysis_id)
    assert stored is not None
    assert stored.status == SegmentAnalysisStatus.FAILED
    assert any("capacity saturated" in w.lower() for w in stored.warnings)
    assert not any("internal error" in w.lower() for w in stored.warnings)
    # The heavy fit was bounded out — the task rejected before invoking the graph.
    mock_graph.ainvoke.assert_not_called()


# =============================================================================
# Registry-derived adjustment set + unmodeled-question warning (2026-08-29,
# /segment-analysis review follow-ups). The page's W was a FIXED
# ["engagement_score"]; copay_support's DGP backdoor (treatment_arm.ARM_REGISTRY)
# is {insurance_access_score, disease_severity} and insurance_access_score is in
# neither X nor W -> the copay estimate reported the confounded diff. causal_paths
# stores each modeled edge's ``confounders_controlled``; derive W from it (minus X,
# minus categoricals the segment loader keeps raw) and warn when the requested
# (treatment, outcome) has no registry edge at all — that is exactly the case where
# cross-library validation "FAILED" without saying why.
# =============================================================================

from src.api.routes.segments import (  # noqa: E402
    _segment_effect_modifiers,
    _segment_question_adjustment,
)


def _registry_rows(brand: str = "Remibrutinib") -> list[dict]:
    """Rows shaped like CausalPathRepository.get_distinct_questions()."""
    return [
        {
            "treatment": "copay_support",
            "outcome": "persistent_180d",
            "brand": brand,
            "confounders": ["insurance_access_score", "disease_severity"],
        },
        {
            "treatment": "treatment_arm",
            "outcome": "persistent_180d",
            "brand": brand,
            "confounders": ["disease_severity", "academic_hcp", "geographic_region"],
        },
        {
            "treatment": "psp_enrolled",
            "outcome": "persistent_180d",
            "brand": brand,
            "confounders": ["disease_severity", "engagement_score", "academic_hcp"],
        },
    ]


def _patch_registry(rows: Any = None, *, error: Exception | None = None):
    """Patch the causal_paths repo factory (function-locally imported from
    src.api.routes.causal) so no Supabase read happens."""
    repo = MagicMock()
    if error is not None:
        repo.get_distinct_questions = AsyncMock(side_effect=error)
    else:
        repo.get_distinct_questions = AsyncMock(return_value=rows or [])
    return patch(
        "src.api.routes.causal._get_causal_path_repo",
        new=AsyncMock(return_value=repo),
    )


@pytest.mark.asyncio
async def test_adjustment_adds_registry_backdoor_outside_x_to_default_w():
    """copay_support -> persistent_180d: the registry backdoor is
    {insurance_access_score, disease_severity}; disease_severity is already in X, so
    W = default control + insurance_access_score (no X/W overlap)."""
    modifiers = _segment_effect_modifiers("Remibrutinib")
    with _patch_registry(_registry_rows()):
        adj = await _segment_question_adjustment(
            treatment_var="copay_support",
            outcome_var="persistent_180d",
            brand="Remibrutinib",
            effect_modifiers=modifiers,
        )
    assert adj.confounders == ["engagement_score", "insurance_access_score"]
    assert adj.modeled is True
    assert adj.warnings == []
    assert not set(adj.confounders) & set(modifiers)


@pytest.mark.asyncio
@pytest.mark.parametrize("treatment", ["psp_enrolled", "treatment_arm"])
async def test_adjustment_pairs_already_covered_keep_default_w(treatment: str):
    """A registry set fully covered by X ∪ default-W (psp_enrolled) — or whose only
    extra member is a categorical the segment loader keeps RAW for segmentation
    (treatment_arm: geographic_region) — leaves W byte-identical to today."""
    with _patch_registry(_registry_rows()):
        adj = await _segment_question_adjustment(
            treatment_var=treatment,
            outcome_var="persistent_180d",
            brand="Remibrutinib",
            effect_modifiers=_segment_effect_modifiers("Remibrutinib"),
        )
    assert adj.confounders == ["engagement_score"]
    assert adj.modeled is True
    assert adj.warnings == []


@pytest.mark.asyncio
async def test_adjustment_ignores_off_allowlist_off_brand_and_self_columns():
    """Registry columns that are not numeric allowlisted covariates, are another
    brand's clinical column, or are the treatment/outcome itself never enter W."""
    rows = [
        {
            "treatment": "copay_support",
            "outcome": "persistent_180d",
            "brand": "Kisqali",
            "confounders": [
                "insurance_access_score",
                "urticaria_severity_uas7",  # Remibrutinib-only -> NULL for Kisqali
                "not_a_column",
                "copay_support",
                "persistent_180d",
            ],
        }
    ]
    with _patch_registry(rows):
        adj = await _segment_question_adjustment(
            treatment_var="copay_support",
            outcome_var="persistent_180d",
            brand="Kisqali",
            effect_modifiers=_segment_effect_modifiers("Kisqali"),
        )
    assert adj.confounders == ["engagement_score", "insurance_access_score"]


@pytest.mark.asyncio
async def test_adjustment_all_brands_unions_registry_rows():
    rows = [
        {
            "treatment": "copay_support",
            "outcome": "persistent_180d",
            "brand": "Fabhalta",
            "confounders": ["disease_severity"],
        },
        {
            "treatment": "copay_support",
            "outcome": "persistent_180d",
            "brand": "Kisqali",
            "confounders": ["insurance_access_score"],
        },
    ]
    with _patch_registry(rows) as factory:
        adj = await _segment_question_adjustment(
            treatment_var="copay_support",
            outcome_var="persistent_180d",
            brand=None,
            effect_modifiers=_segment_effect_modifiers(None),
        )
    repo = factory.return_value
    repo.get_distinct_questions.assert_awaited_once_with(brand=None, include_synthetic=True)
    assert adj.confounders == ["engagement_score", "insurance_access_score"]
    assert adj.modeled is True


@pytest.mark.asyncio
async def test_adjustment_unmodeled_pair_warns_and_keeps_default_w():
    """The user's original run: treatment_initiated -> persistent_180d has NO
    registry edge (no planted effect). The API still accepts it (allowlisted), so
    the run must carry a self-explanatory warning instead of a bare
    'cross-library validation FAILED'."""
    with _patch_registry(_registry_rows()):
        adj = await _segment_question_adjustment(
            treatment_var="treatment_initiated",
            outcome_var="persistent_180d",
            brand="Remibrutinib",
            effect_modifiers=_segment_effect_modifiers("Remibrutinib"),
        )
    assert adj.confounders == ["engagement_score"]
    assert adj.modeled is False
    assert len(adj.warnings) == 1
    warning = adj.warnings[0]
    assert "not a modeled causal question" in warning
    assert "treatment_initiated" in warning and "persistent_180d" in warning
    assert "Remibrutinib" in warning
    assert "/segments/datasets" in warning


@pytest.mark.asyncio
async def test_adjustment_registry_unavailable_fails_soft_with_warning():
    with _patch_registry(error=ConnectionError("no supabase in unit test")):
        adj = await _segment_question_adjustment(
            treatment_var="copay_support",
            outcome_var="persistent_180d",
            brand="Remibrutinib",
            effect_modifiers=_segment_effect_modifiers("Remibrutinib"),
        )
    assert adj.confounders == ["engagement_score"]
    assert adj.modeled is None
    assert len(adj.warnings) == 1
    assert "registry unavailable" in adj.warnings[0]
    assert "engagement_score" in adj.warnings[0]


def _copay_stub_frame(n: int = 120) -> pd.DataFrame:
    frame = _make_stub_frame(n)
    frame["copay_support"] = [i % 2 for i in range(n)]
    frame["insurance_access_score"] = [0.1 + 0.05 * (i % 10) for i in range(n)]
    return frame


@pytest.mark.asyncio
async def test_execute_routes_registry_w_into_state_and_loader(rich_graph_result):
    """The derived W must reach BOTH consumers in lock-step: the loader (so the
    column is selected + float-coerced) and the graph state (cate_estimator's
    explicit-confounders source)."""
    request = RunSegmentAnalysisRequest(
        query="Which Remibrutinib segments benefit most from copay support?",
        brand="Remibrutinib",
        treatment_var="copay_support",
        outcome_var="persistent_180d",
    )
    captured: dict = {}
    mock_graph = MagicMock()

    async def _capture(initial_state):
        captured.update(initial_state)
        return rich_graph_result

    mock_graph.ainvoke = AsyncMock(side_effect=_capture)
    loader = AsyncMock(return_value=_copay_stub_frame())
    with (
        _patch_registry(_registry_rows()),
        patch("src.api.routes.segments._load_segment_hte_frame", new=loader),
        patch(
            "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
            return_value=mock_graph,
        ),
    ):
        response = await _execute_segment_analysis(request)

    assert captured["confounders"] == ["engagement_score", "insurance_access_score"]
    assert loader.await_args.kwargs["confounders"] == [
        "engagement_score",
        "insurance_access_score",
    ]
    assert captured["warnings"] == []
    assert response.warnings == []


@pytest.mark.asyncio
async def test_execute_seeds_unmodeled_question_warning_before_graph_warnings(
    rich_graph_result,
):
    """An unmodeled pair's warning is seeded into the initial graph state so it
    lands FIRST in the persisted warnings, ahead of the validator's FAILED line
    (which then reads as the consequence, not the cause)."""
    request = RunSegmentAnalysisRequest(
        query="Does initiation drive persistence?",
        brand="Remibrutinib",
        treatment_var="treatment_initiated",
        outcome_var="persistent_180d",
        allow_unmodeled=True,  # #1827: the warn-and-run path is opt-in
    )
    failed = "Cross-library validation FAILED: EconML and CausalML agree only 42%"
    mock_graph = MagicMock()

    async def _echo(initial_state):
        # Faithful to the append_unique channel: seeded warnings survive, the
        # uplift node appends its own.
        return {**rich_graph_result, "warnings": [*initial_state["warnings"], failed]}

    mock_graph.ainvoke = AsyncMock(side_effect=_echo)
    with (
        _patch_registry(_registry_rows()),
        _patch_loader(_make_stub_frame()),
        patch(
            "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
            return_value=mock_graph,
        ),
    ):
        response = await _execute_segment_analysis(request)

    assert len(response.warnings) == 2
    assert "not a modeled causal question" in response.warnings[0]
    assert response.warnings[1] == failed


@pytest.mark.asyncio
async def test_execute_mock_fallback_keeps_mock_warning_first(monkeypatch):
    """Mock fallback (dev-only) keeps 'Using mock data' as warnings[0] (locked by
    test_import_error_fail_closed) and still carries the question warning."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    request = RunSegmentAnalysisRequest(
        query="q",
        brand="Remibrutinib",
        treatment_var="treatment_initiated",
        outcome_var="persistent_180d",
        allow_unmodeled=True,  # #1827: the warn-and-run path is opt-in
    )
    with (
        _patch_registry(_registry_rows()),
        _patch_loader(_make_stub_frame()),
        patch(
            "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
            side_effect=ImportError,
        ),
    ):
        response = await _execute_segment_analysis(request)

    assert "mock data" in response.warnings[0].lower()
    assert any("not a modeled causal question" in w for w in response.warnings[1:])


class _FakeQuery:
    """Chainable supabase-py query stub recording the select() column list."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.select_cols: list[str] = []

    def select(self, cols: str):
        self.select_cols = cols.split(",")
        return self

    def eq(self, *_a):
        return self

    def limit(self, _n):
        return self

    async def execute(self):
        return MagicMock(data=self.rows)


@pytest.mark.asyncio
async def test_loader_selects_and_float_coerces_registry_confounders():
    """The loader must SELECT the derived W columns and float-coerce them (the
    numeric-column set is the fixed curated list ∪ the run's confounders), so a
    registry-derived confounder never reaches EconML as a string/object column."""
    n = 120
    rows = [
        {
            "copay_support": i % 2,
            "persistent_180d": (i + 1) % 2,
            "disease_severity": i % 10,
            "age_at_diagnosis": 40 + (i % 40),
            "academic_hcp": i % 2,
            "urticaria_severity_uas7": i % 7,
            "engagement_score": float(i % 5),
            # PostgREST can hand numerics back as strings — must be coerced.
            "insurance_access_score": str(0.1 + 0.05 * (i % 10)),
            "geographic_region": ["midwest", "south"][i % 2],
            "brand": "Remibrutinib",
        }
        for i in range(n)
    ]
    query = _FakeQuery(rows)
    client = MagicMock()
    client.table = MagicMock(return_value=query)
    with patch(
        "src.memory.services.factories.get_async_supabase_client",
        new=AsyncMock(return_value=client),
    ):
        frame = await _load_segment_hte_frame(
            brand="Remibrutinib",
            treatment_var="copay_support",
            outcome_var="persistent_180d",
            effect_modifiers=_segment_effect_modifiers("Remibrutinib"),
            confounders=["engagement_score", "insurance_access_score"],
        )

    assert "insurance_access_score" in query.select_cols
    assert "engagement_score" in query.select_cols
    assert frame["insurance_access_score"].dtype.kind == "f"
    assert frame["insurance_access_score"].iloc[1] == pytest.approx(0.15)


# =============================================================================
# #1827: an UNMODELED pair is refused (400) before any compute unless the caller
# opts in with allow_unmodeled. Live 2026-08-30: treatment_initiated ->
# persistent_180d ran ~40 s to a plausible-looking ATE of +0.076 (confounding —
# the DGP plants no such effect) with nothing but a warning string guarding it.
# =============================================================================

from fastapi import BackgroundTasks  # noqa: E402

from src.api.routes.segments import (  # noqa: E402
    _refuse_unmodeled_question,
    _SegmentQuestionAdjustment,
    run_segment_analysis,
)


def _mock_store() -> MagicMock:
    store = MagicMock()
    store.set = AsyncMock()
    store.get = AsyncMock(return_value=None)
    # #1840: the handler claims an in-flight dedup marker before queuing; a
    # None claim means "no identical run in flight — queue this one".
    store.claim_inflight = AsyncMock(return_value=None)
    store.release_inflight = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_adjustment_names_modeled_outcomes_for_an_unmodeled_pair():
    """The refusal must name the alternatives: outcomes the registry DOES model for
    the requested treatment in scope, restricted to this dataset's outcome
    allowlist (an HCP-grain edge such as copay_support -> roi is not runnable
    here and must not be offered)."""
    rows = _registry_rows() + [
        {
            "treatment": "copay_support",
            "outcome": "adherent_180d",
            "brand": "Remibrutinib",
            "confounders": [],
        },
        {
            "treatment": "copay_support",
            "outcome": "roi",
            "brand": "Remibrutinib",
            "confounders": [],
        },
    ]
    with _patch_registry(rows):
        adj = await _segment_question_adjustment(
            treatment_var="copay_support",
            outcome_var="treatment_initiated",  # no such edge
            brand="Remibrutinib",
            effect_modifiers=_segment_effect_modifiers("Remibrutinib"),
        )
    assert adj.modeled is False
    assert adj.modeled_outcomes == ("adherent_180d", "persistent_180d")

    with _patch_registry(_registry_rows()):
        modeled = await _segment_question_adjustment(
            treatment_var="copay_support",
            outcome_var="persistent_180d",
            brand="Remibrutinib",
            effect_modifiers=_segment_effect_modifiers("Remibrutinib"),
        )
    assert modeled.modeled is True
    assert modeled.modeled_outcomes == ()


def test_refusal_gate_only_fires_on_modeled_false_without_opt_in():
    request = RunSegmentAnalysisRequest(query="q", brand="Remibrutinib")
    for modeled in (True, None):  # modeled / registry unavailable (fail-soft) run
        _refuse_unmodeled_question(
            request,
            _SegmentQuestionAdjustment(
                confounders=["engagement_score"], modeled=modeled, warnings=[]
            ),
            "treatment_arm",
            "persistent_180d",
        )

    with pytest.raises(HTTPException) as exc_info:
        _refuse_unmodeled_question(
            request,
            _SegmentQuestionAdjustment(
                confounders=["engagement_score"],
                modeled=False,
                warnings=[],
                modeled_outcomes=("adherent_180d", "persistent_180d"),
            ),
            "copay_support",
            "treatment_initiated",
        )
    assert exc_info.value.status_code == 400
    detail = str(exc_info.value.detail)
    assert "'copay_support -> treatment_initiated' is not a modeled causal question" in detail
    assert (
        "Modeled outcomes for 'copay_support' on Remibrutinib: adherent_180d, persistent_180d"
        in detail
    )
    assert "allow_unmodeled=true" in detail

    # No modeled outcome at all -> say so (treatment_initiated is an outcome here).
    with pytest.raises(HTTPException) as exc_info:
        _refuse_unmodeled_question(
            request,
            _SegmentQuestionAdjustment(
                confounders=["engagement_score"], modeled=False, warnings=[]
            ),
            "treatment_initiated",
            "persistent_180d",
        )
    assert "'treatment_initiated' has no modeled outcome on Remibrutinib" in str(
        exc_info.value.detail
    )

    # Explicit opt-in bypasses the gate.
    _refuse_unmodeled_question(
        request.model_copy(update={"allow_unmodeled": True}),
        _SegmentQuestionAdjustment(confounders=["engagement_score"], modeled=False, warnings=[]),
        "treatment_initiated",
        "persistent_180d",
    )


@pytest.mark.asyncio
async def test_execute_refuses_unmodeled_pair_with_400_before_any_load():
    """Direct callers of _execute_segment_analysis are gated identically, and the
    refusal fires BEFORE the frame load (no Supabase read, no fit)."""
    request = RunSegmentAnalysisRequest(
        query="Does initiation drive persistence?",
        brand="Remibrutinib",
        treatment_var="treatment_initiated",
        outcome_var="persistent_180d",
    )
    loader = AsyncMock(return_value=_make_stub_frame())
    with (
        _patch_registry(_registry_rows()),
        patch("src.api.routes.segments._load_segment_hte_frame", new=loader),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _execute_segment_analysis(request)
    assert exc_info.value.status_code == 400
    assert "not a modeled causal question" in str(exc_info.value.detail)
    loader.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_refuses_unmodeled_pair_before_queuing():
    """POST in async mode must 400 immediately: no pending record persisted and
    no background task queued (the client never gets an id that only fails)."""
    request = RunSegmentAnalysisRequest(
        query="q",
        brand="Remibrutinib",
        treatment_var="treatment_initiated",
        outcome_var="persistent_180d",
    )
    tasks = BackgroundTasks()
    store = _mock_store()
    with (
        _patch_registry(_registry_rows()),
        patch("src.api.routes.segments._analyses_store", store),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await run_segment_analysis(request, tasks, async_mode=True, user={})
    assert exc_info.value.status_code == 400
    assert tasks.tasks == []
    store.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_hands_resolved_adjustment_to_the_background_task():
    """A modeled pair is queued WITH the handler's adjustment, so the registry is
    read exactly once per request (handler), never again by the task."""
    request = RunSegmentAnalysisRequest(
        query="q",
        brand="Remibrutinib",
        treatment_var="copay_support",
        outcome_var="persistent_180d",
    )
    tasks = BackgroundTasks()
    store = _mock_store()
    with (
        _patch_registry(_registry_rows()) as factory,
        patch("src.api.routes.segments._analyses_store", store),
    ):
        response = await run_segment_analysis(request, tasks, async_mode=True, user={})
    assert response.status == SegmentAnalysisStatus.PENDING
    assert len(tasks.tasks) == 1
    handed = tasks.tasks[0].kwargs["adjustment"]
    assert handed.modeled is True
    assert "insurance_access_score" in handed.confounders
    factory.return_value.get_distinct_questions.assert_awaited_once()


@pytest.mark.asyncio
async def test_background_task_uses_handed_adjustment_without_registry_read(
    rich_graph_result,
):
    request = RunSegmentAnalysisRequest(
        query="q",
        brand="Remibrutinib",
        treatment_var="copay_support",
        outcome_var="persistent_180d",
    )
    store = _DurableAnalysesStore(redis_factory=_failing_redis_factory)
    analysis_id = "seg_handed_adjustment"
    await store.set(
        analysis_id,
        SegmentAnalysisResponse(
            analysis_id=analysis_id,
            status=SegmentAnalysisStatus.PENDING,
            question_type=request.question_type,
        ),
    )
    handed = _SegmentQuestionAdjustment(
        confounders=["engagement_score", "insurance_access_score"], modeled=True, warnings=[]
    )
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=rich_graph_result)
    with (
        patch("src.api.routes.segments._analyses_store", store),
        _patch_registry(error=RuntimeError("registry must not be read by the task")),
        _patch_loader(_make_stub_frame()),
        patch(
            "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
            return_value=mock_graph,
        ),
    ):
        await _run_segment_analysis_task(
            analysis_id=analysis_id, request=request, adjustment=handed
        )

    stored = await store.get(analysis_id)
    assert stored is not None
    assert stored.status == SegmentAnalysisStatus.COMPLETED
    assert not any("registry unavailable" in w for w in stored.warnings)
    initial_state = mock_graph.ainvoke.call_args.args[0]
    assert initial_state["confounders"] == ["engagement_score", "insurance_access_score"]
