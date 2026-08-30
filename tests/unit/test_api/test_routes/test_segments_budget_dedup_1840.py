"""#1840 — run budget on the heavy-compute slot + in-flight dedup of identical
``POST /segments/analyze`` requests.

Follow-up to #1836. Two backend gaps made a client-side duplicate POST
user-visible and could let one hung run poison a worker:

1. ``_execute_segment_analysis`` held ``heavy_compute_slot()`` around
   ``graph.ainvoke`` with NO timeout — a stalled run held the worker's only slot
   forever, so every later request on that worker was rejected with "compute
   capacity saturated" until the worker restarted.
2. An identical request submitted while the first was still pending/estimating
   was queued as a NEW analysis, rejected by the slot guard, and the client
   ended up polling the duplicate's FAILED record while the original completed
   unseen.

Import-light per the repo convention: never ``src.api.main.app``. The graph
factory / loader / registry are patched so no real agent, Supabase or EconML
runs; the durable store is exercised on its in-process fallback (failing Redis
factory) and, for the Redis claim path, against a minimal faithful fake.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.dependencies import compute as compute_mod
from src.api.routes.segments import (
    RunSegmentAnalysisRequest,
    SegmentAnalysisResponse,
    SegmentAnalysisStatus,
    _DurableAnalysesStore,
    _execute_segment_analysis,
    _run_segment_analysis_task,
    _SegmentQuestionAdjustment,
    run_segment_analysis,
)

# =============================================================================
# Fixtures / helpers
# =============================================================================


def _stub_frame(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treatment_arm": [i % 2 for i in range(n)],
            "copay_support": [i % 2 for i in range(n)],
            "persistent_180d": [(i + 1) % 2 for i in range(n)],
            "disease_severity": [i % 10 for i in range(n)],
            "engagement_score": [float(i % 5) for i in range(n)],
            "insurance_access_score": [float(i % 3) for i in range(n)],
            "age_at_diagnosis": [40 + (i % 40) for i in range(n)],
            "academic_hcp": [i % 2 for i in range(n)],
            "disease_severity_band": [["low", "medium", "high"][i % 3] for i in range(n)],
            "age_band": [["<50", "50-65", ">65"][i % 3] for i in range(n)],
            "geographic_region": [["midwest", "south"][i % 2] for i in range(n)],
        }
    )


async def _failing_redis_factory() -> Any:
    raise ConnectionError("no redis in unit test")


def _patch_loader():
    return patch(
        "src.api.routes.segments._load_segment_hte_frame",
        new=AsyncMock(return_value=_stub_frame()),
    )


def _patch_graph(ainvoke: Any):
    mock_graph = MagicMock()
    mock_graph.ainvoke = ainvoke
    return patch(
        "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
        return_value=mock_graph,
    )


def _patch_adjustment(confounders: list[str]):
    """Pin the resolved adjustment set (no registry read) for the POST handler."""
    return patch(
        "src.api.routes.segments._segment_question_adjustment",
        new=AsyncMock(
            return_value=_SegmentQuestionAdjustment(
                confounders=list(confounders), modeled=True, warnings=[]
            )
        ),
    )


async def _never_returns(*_args: Any, **_kwargs: Any) -> None:
    """A graph run that stalls forever (the #1840 pathological run)."""
    await asyncio.Event().wait()


def _completed_result() -> dict:
    return {"status": "completed", "cate_by_segment": {}, "warnings": []}


@pytest.fixture
def mock_user() -> dict:
    return {"user_id": "user123", "role": "analyst"}


@pytest.fixture
def curated_request() -> RunSegmentAnalysisRequest:
    return RunSegmentAnalysisRequest(
        query="Which clinical segments respond best to copay support?",
        brand="Remibrutinib",
        treatment_var="copay_support",
        outcome_var="persistent_180d",
    )


@pytest.fixture
def memory_store() -> _DurableAnalysesStore:
    return _DurableAnalysesStore(redis_factory=_failing_redis_factory)


@pytest.fixture(autouse=True)
def _fresh_limiter():
    compute_mod._reset_limiter_cache_for_tests()
    yield
    compute_mod._reset_limiter_cache_for_tests()


async def _seed_pending(
    store: _DurableAnalysesStore, analysis_id: str, request: RunSegmentAnalysisRequest
) -> None:
    await store.set(
        analysis_id,
        SegmentAnalysisResponse(
            analysis_id=analysis_id,
            status=SegmentAnalysisStatus.PENDING,
            question_type=request.question_type,
        ),
    )


# =============================================================================
# 1. Run budget
# =============================================================================


def test_budget_default_covers_frontend_all_brands_ceiling(monkeypatch):
    """The page waits up to 600 s for an all-brands run
    (SegmentAnalysis.tsx ALL_BRANDS_POLL_CEILING_MS); the backend must never
    give up on a run the page is still willing to wait for."""
    from src.api.routes.segments import (
        SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT,
        _segment_analysis_budget_seconds,
    )

    monkeypatch.delenv("SEGMENT_ANALYSIS_BUDGET_SECONDS", raising=False)
    assert _segment_analysis_budget_seconds() == SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT
    assert SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT >= 600
    # Measured deployed single-brand runs: 73-208 s (#1836) and 109-121 s
    # (2026-08-30 e2i_api logs). The default must sit well above the slowest.
    assert SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT >= 4 * 208


def test_budget_env_override_and_invalid_fallback(monkeypatch):
    from src.api.routes.segments import (
        SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT,
        _segment_analysis_budget_seconds,
    )

    monkeypatch.setenv("SEGMENT_ANALYSIS_BUDGET_SECONDS", "12.5")
    assert _segment_analysis_budget_seconds() == 12.5
    for bad in ("abc", "0", "-5", ""):
        monkeypatch.setenv("SEGMENT_ANALYSIS_BUDGET_SECONDS", bad)
        assert _segment_analysis_budget_seconds() == SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT


@pytest.mark.asyncio
async def test_stalled_graph_exceeds_budget_releases_slot_and_records_failed(
    curated_request, memory_store, monkeypatch
):
    """RED on main: the task never returns (no budget), the slot stays held.

    With the budget: the task returns within the budget, the record is FAILED
    with a warning that NAMES the budget, the slot is released (in_flight 0),
    and the NEXT analysis on this worker runs to completion instead of being
    rejected as 'capacity saturated'."""
    monkeypatch.setenv("SEGMENT_ANALYSIS_BUDGET_SECONDS", "0.3")
    analysis_id = "seg_budget_probe"
    await _seed_pending(memory_store, analysis_id, curated_request)
    handed = _SegmentQuestionAdjustment(confounders=["engagement_score"], modeled=True, warnings=[])

    with (
        patch("src.api.routes.segments._analyses_store", memory_store),
        _patch_loader(),
        _patch_graph(AsyncMock(side_effect=_never_returns)),
    ):
        try:
            await asyncio.wait_for(
                _run_segment_analysis_task(
                    analysis_id=analysis_id, request=curated_request, adjustment=handed
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:  # pragma: no cover - the RED arm
            pytest.fail("background task never returned: nothing bounds the graph run")

    stored = await memory_store.get(analysis_id)
    assert stored is not None
    assert stored.status == SegmentAnalysisStatus.FAILED
    joined = " ".join(stored.warnings).lower()
    assert "budget" in joined, stored.warnings
    assert "0.3" in joined, stored.warnings  # names the budget actually applied
    assert "internal error" not in joined
    assert "capacity saturated" not in joined
    # The slot was released with the cancelled run ...
    assert compute_mod.get_heavy_compute_limiter().in_flight == 0

    # ... so the next request on this worker is ACCEPTED and completes.
    with _patch_loader(), _patch_graph(AsyncMock(return_value=_completed_result())):
        nxt = await _execute_segment_analysis(curated_request, adjustment=handed)
    assert nxt.status == SegmentAnalysisStatus.COMPLETED
    assert compute_mod.get_heavy_compute_limiter().in_flight == 0


@pytest.mark.asyncio
async def test_sync_mode_budget_exceeded_maps_to_504_and_persists_failed(
    curated_request, memory_store, mock_user, monkeypatch
):
    monkeypatch.setenv("SEGMENT_ANALYSIS_BUDGET_SECONDS", "0.2")
    with (
        patch("src.api.routes.segments._analyses_store", memory_store),
        _patch_adjustment(["engagement_score"]),
        _patch_loader(),
        _patch_graph(AsyncMock(side_effect=_never_returns)),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(
                run_segment_analysis(
                    request=curated_request,
                    background_tasks=BackgroundTasks(),
                    async_mode=False,
                    user=mock_user,
                ),
                timeout=3.0,
            )
    assert exc_info.value.status_code == 504
    assert "budget" in str(exc_info.value.detail).lower()
    records = await memory_store.values()
    assert len(records) == 1 and records[0].status == SegmentAnalysisStatus.FAILED
    assert compute_mod.get_heavy_compute_limiter().in_flight == 0


# =============================================================================
# 2. In-flight dedup of identical POSTs
# =============================================================================


async def _post(request: RunSegmentAnalysisRequest, user: dict) -> tuple[Any, BackgroundTasks]:
    bg = BackgroundTasks()
    resp = await run_segment_analysis(
        request=request, background_tasks=bg, async_mode=True, user=user
    )
    return resp, bg


@pytest.mark.asyncio
async def test_identical_post_during_inflight_returns_existing_id(
    curated_request, memory_store, mock_user
):
    """RED on main: the second identical POST gets a NEW analysis_id and a NEW
    background task (which the slot guard then rejects, #1836).

    Contract chosen: 200 with the EXISTING record (same analysis_id, current
    status) — the page's POST-then-poll flow already polls whatever id comes
    back, so no response-shape change is needed."""
    with (
        patch("src.api.routes.segments._analyses_store", memory_store),
        _patch_adjustment(["engagement_score", "insurance_access_score"]),
    ):
        first, bg1 = await _post(curated_request, mock_user)
        assert first.status == SegmentAnalysisStatus.PENDING
        assert len(bg1.tasks) == 1

        twin = RunSegmentAnalysisRequest(**curated_request.model_dump())
        second, bg2 = await _post(twin, mock_user)

    assert second.analysis_id == first.analysis_id
    assert second.status in (SegmentAnalysisStatus.PENDING, SegmentAnalysisStatus.ESTIMATING)
    assert bg2.tasks == []  # nothing queued for the duplicate
    assert len(await memory_store.values()) == 1


@pytest.mark.asyncio
async def test_identical_post_while_estimating_still_dedups(
    curated_request, memory_store, mock_user
):
    with (
        patch("src.api.routes.segments._analyses_store", memory_store),
        _patch_adjustment(["engagement_score"]),
    ):
        first, _ = await _post(curated_request, mock_user)
        rec = await memory_store.get(first.analysis_id)
        assert rec is not None
        rec.status = SegmentAnalysisStatus.ESTIMATING
        await memory_store.set(first.analysis_id, rec)

        second, bg2 = await _post(curated_request, mock_user)
    assert second.analysis_id == first.analysis_id
    assert second.status == SegmentAnalysisStatus.ESTIMATING
    assert bg2.tasks == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", [SegmentAnalysisStatus.COMPLETED, SegmentAnalysisStatus.FAILED]
)
async def test_identical_post_after_terminal_twin_gets_new_id(
    curated_request, memory_store, mock_user, terminal
):
    """Dedup collapses ONLY in-flight records — a completed/failed twin must not
    swallow a fresh submission (the user may legitimately re-run)."""
    with (
        patch("src.api.routes.segments._analyses_store", memory_store),
        _patch_adjustment(["engagement_score"]),
    ):
        first, _ = await _post(curated_request, mock_user)
        rec = await memory_store.get(first.analysis_id)
        assert rec is not None
        rec.status = terminal
        await memory_store.set(first.analysis_id, rec)

        second, bg2 = await _post(curated_request, mock_user)
    assert second.analysis_id != first.analysis_id
    assert second.status == SegmentAnalysisStatus.PENDING
    assert len(bg2.tasks) == 1


@pytest.mark.asyncio
async def test_dedup_key_includes_resolved_adjustment_set(curated_request, memory_store, mock_user):
    """Same raw request, different resolved W (registry-derived) => different
    runs. The key must be built from the RESOLVED scoping, not the wire body."""
    with patch("src.api.routes.segments._analyses_store", memory_store):
        with _patch_adjustment(["engagement_score"]):
            first, _ = await _post(curated_request, mock_user)
        with _patch_adjustment(["engagement_score", "insurance_access_score"]):
            second, bg2 = await _post(curated_request, mock_user)
    assert second.analysis_id != first.analysis_id
    assert len(bg2.tasks) == 1


@pytest.mark.asyncio
async def test_different_brand_scoping_is_not_collapsed(memory_store, mock_user):
    """brand changes the effect-modifier set (and the cohort) — never dedup
    across brands even when everything else matches."""
    remi = RunSegmentAnalysisRequest(
        query="q",
        brand="Remibrutinib",
        treatment_var="copay_support",
        outcome_var="persistent_180d",
    )
    fab = RunSegmentAnalysisRequest(
        query="q", brand="Fabhalta", treatment_var="copay_support", outcome_var="persistent_180d"
    )
    with (
        patch("src.api.routes.segments._analyses_store", memory_store),
        _patch_adjustment(["engagement_score"]),
    ):
        first, _ = await _post(remi, mock_user)
        second, bg2 = await _post(fab, mock_user)
    assert second.analysis_id != first.analysis_id
    assert len(bg2.tasks) == 1


@pytest.mark.asyncio
async def test_task_releases_inflight_marker_when_run_ends(curated_request, memory_store):
    """After the task reaches a terminal state the marker is released, so a
    later identical POST starts a fresh run (belt: status check; braces: the
    marker itself is gone, not merely stale)."""
    from src.api.routes.segments import _segment_analysis_dedup_key

    handed = _SegmentQuestionAdjustment(confounders=["engagement_score"], modeled=True, warnings=[])
    key = _segment_analysis_dedup_key(curated_request, handed, ["disease_severity"])
    analysis_id = "seg_release_probe"
    await _seed_pending(memory_store, analysis_id, curated_request)
    assert await memory_store.claim_inflight(key, analysis_id, ttl_seconds=60) is None
    assert await memory_store.inflight_owner(key) == analysis_id

    with (
        patch("src.api.routes.segments._analyses_store", memory_store),
        _patch_loader(),
        _patch_graph(AsyncMock(return_value=_completed_result())),
    ):
        await _run_segment_analysis_task(
            analysis_id=analysis_id, request=curated_request, adjustment=handed, dedup_key=key
        )

    stored = await memory_store.get(analysis_id)
    assert stored is not None and stored.status == SegmentAnalysisStatus.COMPLETED
    assert await memory_store.inflight_owner(key) is None


# -----------------------------------------------------------------------------
# Store-level claim semantics (Redis NX path + graceful degradation)
# -----------------------------------------------------------------------------


class _TinyFakeRedis:
    """Only the commands the in-flight marker uses; ``SET NX EX`` is faithful
    (returns True when set, None when the key already exists). Record reads
    return None so the store falls through to its in-process mirror."""

    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.calls: list[tuple] = []

    async def set(self, key, value, ex=None, nx=False):  # noqa: ANN001
        self.calls.append(("set", key, value, ex, nx))
        if nx and key in self.strings:
            return None
        self.strings[key] = value
        return True

    async def get(self, key):  # noqa: ANN001
        return self.strings.get(key)

    async def delete(self, *keys):  # noqa: ANN001
        n = 0
        for k in keys:
            if self.strings.pop(k, None) is not None:
                n += 1
        return n


@pytest.mark.asyncio
async def test_claim_inflight_uses_set_nx_and_yields_to_live_owner(curated_request):
    fake = _TinyFakeRedis()

    async def factory() -> Any:
        return fake

    store = _DurableAnalysesStore(redis_factory=factory)
    # Seed the owner's record in the mirror only (fake ``get`` -> None for it).
    store._memory["seg_owner"] = SegmentAnalysisResponse(
        analysis_id="seg_owner", status=SegmentAnalysisStatus.ESTIMATING
    )

    assert await store.claim_inflight("k1", "seg_owner", ttl_seconds=30) is None
    assert ("set", store._inflight_key("k1"), "seg_owner", 30, True) in fake.calls
    # Second claimant loses to the live owner.
    assert await store.claim_inflight("k1", "seg_dup", ttl_seconds=30) == "seg_owner"
    assert fake.strings[store._inflight_key("k1")] == "seg_owner"

    # Owner reaches a terminal state: the marker is STALE, the next claim wins.
    store._memory["seg_owner"].status = SegmentAnalysisStatus.COMPLETED
    assert await store.claim_inflight("k1", "seg_new", ttl_seconds=30) is None
    assert fake.strings[store._inflight_key("k1")] == "seg_new"

    # Release only removes a marker that still points at the caller.
    await store.release_inflight("k1", "seg_owner")
    assert fake.strings[store._inflight_key("k1")] == "seg_new"
    await store.release_inflight("k1", "seg_new")
    assert store._inflight_key("k1") not in fake.strings


@pytest.mark.asyncio
async def test_claim_inflight_marker_with_missing_record_is_stale():
    """A marker whose record vanished (evicted / never written) must not block
    submissions forever — it is treated as stale and overwritten."""
    fake = _TinyFakeRedis()

    async def factory() -> Any:
        return fake

    store = _DurableAnalysesStore(redis_factory=factory)
    fake.strings[store._inflight_key("k2")] = "seg_ghost"
    assert await store.claim_inflight("k2", "seg_live", ttl_seconds=30) is None
    assert fake.strings[store._inflight_key("k2")] == "seg_live"


@pytest.mark.asyncio
async def test_claim_inflight_degrades_to_memory_when_redis_down(memory_store):
    memory_store._memory["seg_a"] = SegmentAnalysisResponse(
        analysis_id="seg_a", status=SegmentAnalysisStatus.PENDING
    )
    assert await memory_store.claim_inflight("k3", "seg_a", ttl_seconds=30) is None
    assert await memory_store.claim_inflight("k3", "seg_b", ttl_seconds=30) == "seg_a"
    await memory_store.release_inflight("k3", "seg_a")
    assert await memory_store.inflight_owner("k3") is None
    assert await memory_store.claim_inflight("k3", "seg_b", ttl_seconds=30) is None


@pytest.mark.asyncio
async def test_claim_inflight_memory_marker_expires(memory_store, monkeypatch):
    """The in-process marker honours its TTL (the crash-recovery backstop the
    Redis ``EX`` gives for free)."""
    import src.api.routes.segments as seg_mod

    now = {"t": 1000.0}
    monkeypatch.setattr(seg_mod.time, "monotonic", lambda: now["t"])
    memory_store._memory["seg_a"] = SegmentAnalysisResponse(
        analysis_id="seg_a", status=SegmentAnalysisStatus.PENDING
    )
    assert await memory_store.claim_inflight("k4", "seg_a", ttl_seconds=10) is None
    now["t"] += 5
    assert await memory_store.claim_inflight("k4", "seg_b", ttl_seconds=10) == "seg_a"
    now["t"] += 6  # past the marker's TTL
    assert await memory_store.claim_inflight("k4", "seg_b", ttl_seconds=10) is None


def test_dedup_key_is_stable_and_scoping_sensitive(curated_request):
    from src.api.routes.segments import _segment_analysis_dedup_key

    adj = _SegmentQuestionAdjustment(confounders=["engagement_score"], modeled=True, warnings=[])
    k1 = _segment_analysis_dedup_key(curated_request, adj, ["disease_severity", "age_at_diagnosis"])
    k2 = _segment_analysis_dedup_key(
        RunSegmentAnalysisRequest(**curated_request.model_dump()),
        adj,
        ["age_at_diagnosis", "disease_severity"],  # order-insensitive
    )
    assert k1 == k2
    other_w = _SegmentQuestionAdjustment(
        confounders=["engagement_score", "insurance_access_score"], modeled=True, warnings=[]
    )
    assert _segment_analysis_dedup_key(curated_request, other_w, ["disease_severity"]) != k1
    other_query = RunSegmentAnalysisRequest(**{**curated_request.model_dump(), "query": "other"})
    assert _segment_analysis_dedup_key(other_query, adj, ["disease_severity"]) != k1
