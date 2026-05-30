"""OOM P1b: causal real-compute endpoints are bounded by the heavy-compute slot.

P1 (merged) added a per-worker reject-fast bound (``heavy_compute_slot``) on the
explain + digital_twin heavy routes. P1b extends that SAME bound to the causal
route's THREE real-compute endpoints, which were an unclosed OOM hole:

1. ``/causal/hierarchical/analyze`` — the EconML-within-segments fit
   (``analyzer.analyze`` inside ``_execute_hierarchical_analysis``).
2. ``/causal/pipeline/sequential`` (``demo_mode=False``) — the real 4-library
   ``SequentialPipeline`` via ``_run_real_sequential_pipeline``.
3. ``/causal/pipeline/parallel`` (``demo_mode=False``) — the real multi-library
   fan-out via ``_run_real_parallel_pipeline``.

These tests exercise the REAL limiter (env-bounded to 1 in-flight op, not mocked)
and assert that:

* When the single slot is already occupied (a concurrent in-flight heavy op),
  each real endpoint rejects fast by surfacing ``HeavyComputeSaturated`` — which
  the app exception handler maps to 503 + Retry-After — rather than swallowing it
  into a 500 or running another heavy op that could OOM-kill the cgroup. We also
  assert the heavy work itself was NOT invoked while saturated.
* The cheap ``demo_mode=True`` placeholder paths for sequential/parallel are NOT
  bounded: they still succeed even when the slot is saturated (they do no heavy
  work, so rejecting them would be pure load-shedding of trivial requests).

An always-pass gate is a mock; these assertions check real behavior — the heavy
callable must not run while saturated, and the demo path must run regardless.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks

from src.api.routes import causal as causal_module
from src.api.schemas.causal import (
    AnalysisStatus,
    CausalLibrary,
    HierarchicalAnalysisRequest,
    ParallelPipelineRequest,
    ParallelPipelineResponse,
    PipelineStageConfig,
    SequentialPipelineRequest,
    SequentialPipelineResponse,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def _heavy_compute_one_slot(monkeypatch):
    """Bound heavy compute to a single in-flight op and reset the limiter.

    Mirrors the digital_twin/explain route tests: exercises the REAL limiter
    (not a mock) so a saturated worker genuinely raises HeavyComputeSaturated.
    """
    monkeypatch.setenv("HEAVY_COMPUTE_MAX_CONCURRENCY", "1")
    import src.api.dependencies.compute as compute_mod

    compute_mod._reset_limiter_cache_for_tests()
    yield compute_mod
    compute_mod._reset_limiter_cache_for_tests()


def _seq_request() -> SequentialPipelineRequest:
    # The schema requires >= 2 stages (min_length=2).
    return SequentialPipelineRequest(
        treatment_var="treatment",
        outcome_var="outcome",
        stages=[
            PipelineStageConfig(
                library=CausalLibrary.DOWHY,
                estimator="propensity_score_matching",
            ),
            PipelineStageConfig(
                library=CausalLibrary.ECONML,
                estimator="causal_forest",
            ),
        ],
        data_source="test_data",
    )


def _parallel_request() -> ParallelPipelineRequest:
    # The schema requires >= 2 libraries (min_length=2).
    return ParallelPipelineRequest(
        treatment_var="treatment",
        outcome_var="outcome",
        libraries=[CausalLibrary.DOWHY, CausalLibrary.ECONML],
        data_source="test_data",
    )


def _hierarchical_request() -> HierarchicalAnalysisRequest:
    return HierarchicalAnalysisRequest(
        data_source="test_data",
        treatment_var="treatment",
        outcome_var="outcome",
        effect_modifiers=["mod_a"],
        timeout_seconds=30,
    )


def _seq_response(pipeline_id: str) -> SequentialPipelineResponse:
    return SequentialPipelineResponse(
        pipeline_id=pipeline_id,
        status=AnalysisStatus.COMPLETED,
        stages_completed=1,
        stages_total=1,
        stage_results=[],
        consensus_effect=None,
        consensus_ci_lower=None,
        consensus_ci_upper=None,
        library_agreement_score=None,
        effect_estimate_variance=None,
        total_latency_ms=1,
        created_at=datetime.now(timezone.utc),
        warnings=[],
    )


def _parallel_response(pipeline_id: str) -> ParallelPipelineResponse:
    return ParallelPipelineResponse(
        pipeline_id=pipeline_id,
        status=AnalysisStatus.COMPLETED,
        libraries_succeeded=["dowhy"],
        libraries_failed=[],
        library_results={},
        consensus_effect=None,
        consensus_ci_lower=None,
        consensus_ci_upper=None,
        library_agreement_score=None,
        consensus_method="weighted_average",
        total_latency_ms=1,
        created_at=datetime.now(timezone.utc),
        warnings=[],
    )


# --------------------------------------------------------------------------- #
# 1. /hierarchical/analyze
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_hierarchical_analyze_rejects_when_saturated(_heavy_compute_one_slot):
    """When the heavy-compute slot is occupied, the sync hierarchical endpoint
    must surface HeavyComputeSaturated (→ 503), NOT swallow it into a 500, and
    must NOT invoke the heavy analyzer.analyze."""
    from src.api.dependencies.compute import HeavyComputeSaturated

    # Occupy the single slot (simulate a concurrent in-flight heavy op).
    limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
    limiter.acquire()

    analyze_called = {"v": False}

    class _FakeAnalyzer:
        def __init__(self, *a, **k):
            pass

        async def analyze(self, *a, **k):  # pragma: no cover - must not run
            analyze_called["v"] = True
            raise AssertionError("analyzer.analyze ran while saturated")

    # Patch the heavy analyzer at its import site inside the engine module so the
    # in-function import in _execute_hierarchical_analysis resolves to the fake.
    with patch(
        "src.causal_engine.hierarchical.HierarchicalAnalyzer",
        _FakeAnalyzer,
    ):
        with pytest.raises(HeavyComputeSaturated):
            await causal_module.run_hierarchical_analysis(
                _hierarchical_request(),
                background_tasks=BackgroundTasks(),
                async_mode=False,
                user={"user_id": "test", "role": "analyst"},
            )

    assert analyze_called["v"] is False


@pytest.mark.asyncio
async def test_hierarchical_analyze_succeeds_when_slot_free(_heavy_compute_one_slot):
    """With a free slot the hierarchical endpoint acquires + releases the slot
    around the heavy analyze call and returns a real response."""
    import numpy as np

    from src.api.schemas.causal import HierarchicalAnalysisResponse

    analyze_called = {"v": False}

    class _Seg:
        segment_id = 0
        segment_name = "seg-0"
        n_samples = 100
        uplift_range = (0.0, 1.0)
        cate_mean = 0.5
        cate_std = 0.1
        cate_ci_lower = 0.4
        cate_ci_upper = 0.6
        success = True
        error_message = None

    class _Result:
        segment_results = [_Seg()]
        overall_ate = 0.5
        overall_ate_ci_lower = 0.4
        overall_ate_ci_upper = 0.6
        segment_heterogeneity = 0.0
        n_segments = 1
        warnings: list = []
        errors: list = []

    class _FakeAnalyzer:
        def __init__(self, *a, **k):
            pass

        async def analyze(self, *a, **k):
            analyze_called["v"] = True
            return _Result()

    # Keep nested-CI math out of scope: stub the aggregator to a no-op-ish path
    # by ensuring at least one successful segment (handled by _Seg above).
    with (
        patch(
            "src.causal_engine.hierarchical.HierarchicalAnalyzer",
            _FakeAnalyzer,
        ),
        patch.object(np.random, "seed", lambda *a, **k: None),
    ):
        result = await causal_module.run_hierarchical_analysis(
            _hierarchical_request(),
            background_tasks=BackgroundTasks(),
            async_mode=False,
            user={"user_id": "test", "role": "analyst"},
        )

    assert analyze_called["v"] is True
    assert isinstance(result, HierarchicalAnalysisResponse)
    # Slot released after a successful run.
    limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
    assert limiter.in_flight == 0


# --------------------------------------------------------------------------- #
# 2. /pipeline/sequential (real path bounded, demo path NOT bounded)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sequential_real_path_rejects_when_saturated(_heavy_compute_one_slot):
    """demo_mode=False: a saturated slot must reject fast (HeavyComputeSaturated
    → 503) without invoking the real pipeline."""
    from src.api.dependencies.compute import HeavyComputeSaturated

    limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
    limiter.acquire()

    real_called = {"v": False}

    async def fake_real(pipeline_id, request):  # pragma: no cover - must not run
        real_called["v"] = True
        return _seq_response(pipeline_id)

    with patch.object(causal_module, "_run_real_sequential_pipeline", fake_real):
        with pytest.raises(HeavyComputeSaturated):
            await causal_module.run_sequential_pipeline(
                _seq_request(),
                background_tasks=BackgroundTasks(),
                async_mode=False,
                demo_mode=False,
                user={"user_id": "test", "role": "analyst"},
            )

    assert real_called["v"] is False


@pytest.mark.asyncio
async def test_sequential_demo_path_not_bounded_when_saturated(_heavy_compute_one_slot):
    """demo_mode=True does NO heavy work, so it must NOT be bounded: it succeeds
    even when the heavy-compute slot is fully occupied."""
    limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
    limiter.acquire()  # saturate

    # The real pipeline must never be touched on the demo path.
    real_called = {"v": False}

    async def fake_real(pipeline_id, request):  # pragma: no cover
        real_called["v"] = True
        return _seq_response(pipeline_id)

    with patch.object(causal_module, "_run_real_sequential_pipeline", fake_real):
        result = await causal_module.run_sequential_pipeline(
            _seq_request(),
            background_tasks=BackgroundTasks(),
            async_mode=False,
            demo_mode=True,
            user={"user_id": "test", "role": "analyst"},
        )

    assert real_called["v"] is False
    assert result.status in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED)
    # Demo path is pinned-zero placeholders; it must succeed despite saturation.
    assert any("demo_mode=true" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# 3. /pipeline/parallel (real path bounded, demo path NOT bounded)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_parallel_real_path_rejects_when_saturated(_heavy_compute_one_slot):
    """demo_mode=False: a saturated slot must reject fast (HeavyComputeSaturated
    → 503), NOT a 408 timeout or a 500, and without invoking the real pipeline."""
    from src.api.dependencies.compute import HeavyComputeSaturated

    limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
    limiter.acquire()

    real_called = {"v": False}

    async def fake_real(pipeline_id, request):  # pragma: no cover - must not run
        real_called["v"] = True
        return _parallel_response(pipeline_id)

    with patch.object(causal_module, "_run_real_parallel_pipeline", fake_real):
        with pytest.raises(HeavyComputeSaturated):
            await causal_module.run_parallel_pipeline(
                _parallel_request(),
                demo_mode=False,
                user={"user_id": "test", "role": "analyst"},
            )

    assert real_called["v"] is False


@pytest.mark.asyncio
async def test_parallel_demo_path_not_bounded_when_saturated(_heavy_compute_one_slot):
    """demo_mode=True does NO heavy work, so it must NOT be bounded: it succeeds
    even when the heavy-compute slot is fully occupied."""
    limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
    limiter.acquire()  # saturate

    real_called = {"v": False}

    async def fake_real(pipeline_id, request):  # pragma: no cover
        real_called["v"] = True
        return _parallel_response(pipeline_id)

    with patch.object(causal_module, "_run_real_parallel_pipeline", fake_real):
        result = await causal_module.run_parallel_pipeline(
            _parallel_request(),
            demo_mode=True,
            user={"user_id": "test", "role": "analyst"},
        )

    assert real_called["v"] is False
    assert result.status in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED)
