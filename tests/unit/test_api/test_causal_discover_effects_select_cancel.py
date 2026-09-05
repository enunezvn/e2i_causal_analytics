"""Unit coverage for the discover-effects question SELECTION and CANCEL paths.

A discovery run validates every SSOT candidate question for a (dataset, brand)
scope — ~3 min each, serially — so the user can (1) pick a SUBSET of the
candidates up front and (2) stop a run they no longer want. These tests wire a
real HTTP client over the router (query/body shapes on the wire) and drive the
background task directly with the agent run stubbed; the end-to-end agent runs
are covered by a faithful check on the box.

Design contracts pinned here:
- ``GET /causal/discover-effects/questions`` lists what a run WOULD validate,
  with the curated display labels, and is NOT shadowed by the ``{job_id}`` poll.
- ``POST /causal/discover-effects`` accepts an optional ``questions`` subset that
  must be a subset of the SSOT candidates (unknown pair -> 400, empty -> 400);
  no body / no ``questions`` keeps today's run-everything behaviour.
- Cancel is COOPERATIVE: the in-flight question finishes (sync estimators cannot
  be interrupted), the run stops at the next question boundary, finished rows
  are kept, the rest are marked ``cancelled`` — never fabricated.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.routes import causal as causal_routes
from src.api.schemas.causal import (
    AgentCausalAnalysisResponse,
    CausalDAGModel,
    DiscoveredEffect,
    DiscoverEffectsResponse,
    RefutationSummary,
)

CANDIDATES = [
    causal_routes._CandidateQuestion(
        "treatment_arm", "persistent_180d", "Remibrutinib", ["disease_severity"]
    ),
    causal_routes._CandidateQuestion(
        "sample_dropped", "treatment_initiated", "Remibrutinib", ["disease_severity"]
    ),
    causal_routes._CandidateQuestion(
        "copay_card_used", "persistent_180d", "Remibrutinib", ["disease_severity", "age"]
    ),
]


def _memory_store(prefix: str) -> DurableJobStore:
    """A store that degrades to its in-process fallback (no Redis in unit tests)."""

    async def boom():
        raise RuntimeError("redis not initialised")

    return DurableJobStore(prefix, DiscoverEffectsResponse, redis_factory=boom)


def _wire_client():
    """Real HTTP client over the causal router with viewer + analyst auth stubbed."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.dependencies.auth import require_analyst, require_viewer

    app = FastAPI()
    app.include_router(causal_routes.router)
    app.dependency_overrides[require_viewer] = lambda: {"role": "viewer"}
    app.dependency_overrides[require_analyst] = lambda: {"role": "analyst"}
    return TestClient(app)


@pytest.fixture
def scope(monkeypatch):
    """Patch the SSOT enumeration + brand list + job store + the background task
    so the route layer is exercised without a DB or an agent run."""
    monkeypatch.setattr(
        causal_routes, "_discover_candidate_questions", AsyncMock(return_value=list(CANDIDATES))
    )
    monkeypatch.setattr(
        causal_routes, "_list_dataset_brands", AsyncMock(return_value=["Remibrutinib"])
    )
    store = _memory_store("test:discover")
    monkeypatch.setattr(causal_routes, "_discover_effects_store", store)
    task = AsyncMock()
    monkeypatch.setattr(causal_routes, "_run_discover_effects_task", task)
    return {"store": store, "task": task}


# ---------------------------------------------------------------------------
# GET /discover-effects/questions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_questions_endpoint_lists_labelled_ssot_candidates(scope):
    """The selector shows exactly what a run would validate, labelled with the
    curated column labels (the same SSOT the leaderboard rows render). This path
    is a literal segment under ``/discover-effects/`` — a 404 "Unknown job
    'questions'" here means the ``{job_id}`` poll route shadowed it."""
    client = _wire_client()
    r = client.get(
        "/causal/discover-effects/questions",
        params={"dataset": "patient_journeys", "brand": "Remibrutinib"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == "patient_journeys"
    assert body["brand"] == "Remibrutinib"
    assert [(q["treatment"], q["outcome"]) for q in body["questions"]] == [
        (q.treatment, q.outcome) for q in CANDIDATES
    ]
    first = body["questions"][1]
    assert first["treatment_label"] == causal_routes._column_label("sample_dropped")
    assert first["treatment_label"] == "Product samples provided (rep sample drop)"
    assert first["outcome_label"] == causal_routes._column_label("treatment_initiated")
    assert first["brand"] == "Remibrutinib"
    assert first["adjustment_set"] == ["disease_severity"]
    causal_routes._discover_candidate_questions.assert_awaited_once_with(
        "patient_journeys", "Remibrutinib"
    )


@pytest.mark.unit
def test_questions_endpoint_validates_dataset_and_brand(scope):
    client = _wire_client()
    assert (
        client.get("/causal/discover-effects/questions", params={"dataset": "nope"}).status_code
        == 404
    )
    r = client.get(
        "/causal/discover-effects/questions",
        params={"dataset": "patient_journeys", "brand": "Acme"},
    )
    assert r.status_code == 400
    assert "Acme" in r.text


# ---------------------------------------------------------------------------
# POST /discover-effects — optional question subset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_post_without_a_selection_runs_every_candidate(scope):
    """Back-compat: the FE has always POSTed ``{}`` — that (and no body at all)
    still schedules every SSOT candidate."""
    client = _wire_client()
    for body in ({}, None):
        r = client.post(
            "/causal/discover-effects", params={"dataset": "patient_journeys"}, json=body
        )
        assert r.status_code == 200, r.text
        job = r.json()
        assert job["total"] == 3
        assert job["status"] == "pending"
        assert job["cancel_requested"] is False
    scheduled = scope["task"].await_args_list[-1].args
    assert [q.outcome for q in scheduled[2]] == [q.outcome for q in CANDIDATES]


@pytest.mark.unit
def test_post_with_a_selection_runs_only_those_candidates(scope):
    client = _wire_client()
    r = client.post(
        "/causal/discover-effects",
        params={"dataset": "patient_journeys", "brand": "Remibrutinib"},
        json={
            "questions": [
                {
                    "treatment": "copay_card_used",
                    "outcome": "persistent_180d",
                    "brand": "Remibrutinib",
                },
                {
                    "treatment": "treatment_arm",
                    "outcome": "persistent_180d",
                    "brand": "Remibrutinib",
                },
                # A duplicate selection is collapsed, never run twice.
                {
                    "treatment": "treatment_arm",
                    "outcome": "persistent_180d",
                    "brand": "Remibrutinib",
                },
            ]
        },
    )
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["total"] == 2
    assert job["completed"] == 0
    assert [(e["treatment"], e["outcome"]) for e in job["effects"]] == [
        ("copay_card_used", "persistent_180d"),
        ("treatment_arm", "persistent_180d"),
    ]
    # The SSOT row (its modeled adjustment set) is what runs — not the request's echo.
    assert job["effects"][0]["adjustment_set"] == ["disease_severity", "age"]
    scheduled = scope["task"].await_args_list[-1].args
    assert [q.treatment for q in scheduled[2]] == ["copay_card_used", "treatment_arm"]
    assert scheduled[2][0].adjustment_set == ["disease_severity", "age"]


@pytest.mark.unit
def test_post_rejects_a_pair_outside_the_ssot_candidates(scope):
    """The column-allowlist gate stays authoritative: a selection is a SUBSET of
    the SSOT candidates, never a free-form pair."""
    client = _wire_client()
    r = client.post(
        "/causal/discover-effects",
        params={"dataset": "patient_journeys", "brand": "Remibrutinib"},
        json={
            "questions": [
                {
                    "treatment": "treatment_arm",
                    "outcome": "persistent_180d",
                    "brand": "Remibrutinib",
                },
                {"treatment": "age", "outcome": "persistent_180d", "brand": "Remibrutinib"},
            ]
        },
    )
    assert r.status_code == 400
    assert "age" in r.text and "persistent_180d" in r.text
    assert "discover-effects/questions" in r.text
    scope["task"].assert_not_awaited()


@pytest.mark.unit
def test_post_rejects_an_empty_selection(scope):
    client = _wire_client()
    r = client.post(
        "/causal/discover-effects",
        params={"dataset": "patient_journeys"},
        json={"questions": []},
    )
    assert r.status_code == 400
    assert "at least one" in r.text.lower()
    scope["task"].assert_not_awaited()


@pytest.mark.unit
def test_post_selection_brand_must_match_the_candidate_row(scope):
    """Same (treatment, outcome) under a different brand is a different SSOT row."""
    client = _wire_client()
    r = client.post(
        "/causal/discover-effects",
        params={"dataset": "patient_journeys", "brand": "Remibrutinib"},
        json={
            "questions": [
                {"treatment": "treatment_arm", "outcome": "persistent_180d", "brand": "Kisqali"}
            ]
        },
    )
    assert r.status_code == 400
    scope["task"].assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /discover-effects/{job_id}/cancel
# ---------------------------------------------------------------------------


def _job(job_id: str, status: str, completed: int = 0) -> DiscoverEffectsResponse:
    return DiscoverEffectsResponse(
        job_id=job_id,
        status=status,
        dataset="patient_journeys",
        brand="Remibrutinib",
        total=3,
        completed=completed,
        effects=[causal_routes._pending_effect(q, "pending") for q in CANDIDATES],
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_route_flags_a_running_job(scope):
    store: DurableJobStore = scope["store"]
    await store.set("j-run", _job("j-run", "running", completed=1))
    # A LIVE run: its task is beating. (A `running` row with no heartbeat is an
    # orphan, which the cancel route reports as failed — see the orphan tests.)
    await store.touch_marker("j-run", causal_routes._DISCOVERY_ALIVE_MARKER, ttl_seconds=120)
    client = _wire_client()
    r = client.post("/causal/discover-effects/j-run/cancel")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "j-run"
    assert body["cancel_requested"] is True
    # Still running on the wire: the in-flight question finishes first.
    assert body["status"] == "running"
    assert await store.has_marker("j-run", "cancel") is True
    # Idempotent: a second cancel is a no-op 200, not an error.
    assert client.post("/causal/discover-effects/j-run/cancel").status_code == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_route_is_a_noop_on_a_finished_job_and_404_on_unknown(scope):
    store: DurableJobStore = scope["store"]
    await store.set("j-done", _job("j-done", "completed", completed=3))
    client = _wire_client()
    r = client.post("/causal/discover-effects/j-done/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert r.json()["cancel_requested"] is False
    assert await store.has_marker("j-done", "cancel") is False
    assert client.post("/causal/discover-effects/nope/cancel").status_code == 404


# ---------------------------------------------------------------------------
# The background task honours the cancel marker at question boundaries
# ---------------------------------------------------------------------------


def _completed_agent_response(aid: str, t: str, o: str) -> AgentCausalAnalysisResponse:
    return AgentCausalAnalysisResponse(
        analysis_id=aid,
        status="completed",
        treatment_var=t,
        outcome_var=o,
        dataset="patient_journeys",
        n_rows=500,
        data_source="synthetic",
        dag=CausalDAGModel(),
        ate=0.05,
        statistical_significance=True,
        selected_estimator="LinearDML",
        refutation=RefutationSummary(gate_decision="proceed", passed=True),
        latency_ms=10,
    )


@pytest.fixture
def task_env(monkeypatch):
    """Drive ``_run_discover_effects_task`` with the agent stubbed. ``calls``
    records every question the agent actually ran."""
    store = _memory_store("test:discover")
    agent_store = DurableJobStore(
        "test:agent",
        AgentCausalAnalysisResponse,
        redis_factory=store._redis_factory,
    )
    monkeypatch.setattr(causal_routes, "_discover_effects_store", store)
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", agent_store)

    async def identity_prerank(dataset, questions):
        return list(questions)

    monkeypatch.setattr(causal_routes, "_prerank_questions", identity_prerank)
    monkeypatch.setattr(causal_routes, "_attach_clinical_context", AsyncMock())

    calls: List[Dict[str, Any]] = []
    hooks: Dict[str, Any] = {"on_load": None}

    async def fake_load(**kw):
        calls.append(kw)
        if hooks["on_load"] is not None:
            await hooks["on_load"](len(calls))
        df = pd.DataFrame({kw["treatment_var"]: [0, 1], kw["outcome_var"]: [0, 1]})
        return df, [kw["treatment_var"], kw["outcome_var"], *kw["covariates"]]

    async def fake_agent(aid, req, df, cov, data_source):
        await agent_store.set(
            aid, _completed_agent_response(aid, req.treatment_var, req.outcome_var)
        )

    monkeypatch.setattr(causal_routes, "_load_agent_estimation_frame", fake_load)
    monkeypatch.setattr(causal_routes, "_run_agent_analysis_task", fake_agent)
    return {"store": store, "calls": calls, "hooks": hooks}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_stops_at_the_next_boundary_after_a_cancel(task_env):
    """A cancel that lands while question 1 is estimating: question 1 finishes
    and is KEPT (real estimate), questions 2-3 never run and are marked
    ``cancelled`` (no fabricated rows), the job ends ``cancelled`` with
    ``completed == 1`` and ``cancel_requested`` echoed."""
    store: DurableJobStore = task_env["store"]

    async def cancel_during_first_question(n_loads: int) -> None:
        if n_loads == 1:
            await store.set_marker("job-1", "cancel")

    task_env["hooks"]["on_load"] = cancel_during_first_question
    await causal_routes._run_discover_effects_task(
        "job-1", "patient_journeys", list(CANDIDATES), "synthetic", "Remibrutinib"
    )
    job = await store.get("job-1")
    assert job is not None
    assert job.status == "cancelled"
    assert job.cancel_requested is True
    assert job.completed == 1
    assert job.total == 3
    assert len(task_env["calls"]) == 1
    by_key = {(e.treatment, e.outcome): e for e in job.effects}
    done = by_key[("treatment_arm", "persistent_180d")]
    assert done.status == "completed" and done.ate == pytest.approx(0.05)
    assert done.analysis_id  # drill-down stays reachable
    for key in (("sample_dropped", "treatment_initiated"), ("copay_card_used", "persistent_180d")):
        assert by_key[key].status == "cancelled"
        assert by_key[key].ate is None and by_key[key].summary is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_cancelled_before_the_first_question_runs_nothing(task_env):
    store: DurableJobStore = task_env["store"]
    await store.set_marker("job-0", "cancel")
    await causal_routes._run_discover_effects_task(
        "job-0", "patient_journeys", list(CANDIDATES), "synthetic", "Remibrutinib"
    )
    job = await store.get("job-0")
    assert job is not None
    assert job.status == "cancelled"
    assert job.completed == 0
    assert task_env["calls"] == []
    assert {e.status for e in job.effects} == {"cancelled"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_without_a_cancel_completes_every_question(task_env):
    """Negative control for the boundary check: no marker -> the run is unchanged."""
    store: DurableJobStore = task_env["store"]
    await causal_routes._run_discover_effects_task(
        "job-all", "patient_journeys", list(CANDIDATES), "synthetic", "Remibrutinib"
    )
    job = await store.get("job-all")
    assert job is not None
    assert job.status == "completed"
    assert job.cancel_requested is False
    assert job.completed == 3
    assert len(task_env["calls"]) == 3
    assert {e.status for e in job.effects} == {"completed"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_after_the_last_question_is_still_a_completed_run(task_env):
    """A cancel that lands while the FINAL question is estimating has nothing left
    to skip — the run is complete, not cancelled."""
    store: DurableJobStore = task_env["store"]

    async def cancel_during_last(n_loads: int) -> None:
        if n_loads == 3:
            await store.set_marker("job-last", "cancel")

    task_env["hooks"]["on_load"] = cancel_during_last
    await causal_routes._run_discover_effects_task(
        "job-last", "patient_journeys", list(CANDIDATES), "synthetic", "Remibrutinib"
    )
    job = await store.get("job-last")
    assert job is not None
    assert job.status == "completed"
    assert job.completed == 3
    assert {e.status for e in job.effects} == {"completed"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_is_honoured_when_the_marker_write_degraded_on_another_worker(
    task_env, monkeypatch
):
    """Prod runs 2 gunicorn workers: the task lives on worker A and the cancel
    POST can land on worker B. If B's Redis marker SET fails transiently,
    ``set_marker`` degrades to B's process memory (invisible to A) while the
    route's row write does reach Redis and the route answers 200 with
    ``cancel_requested: true``. The task must honour the ROW flag as a fallback
    so the API can never acknowledge a cancel the run then ignores."""

    class _SharedRedis:
        def __init__(self) -> None:
            self.kv: Dict[str, str] = {}

        async def set(self, key, value, ex=None):
            if key.endswith(":cancel"):
                raise ConnectionError("marker SET lost on this worker")
            self.kv[key] = value

        async def get(self, key):
            return self.kv.get(key)

    shared = _SharedRedis()

    async def factory():
        return shared

    worker_a = DurableJobStore("test:discover", DiscoverEffectsResponse, redis_factory=factory)
    worker_b = DurableJobStore("test:discover", DiscoverEffectsResponse, redis_factory=factory)
    monkeypatch.setattr(causal_routes, "_discover_effects_store", worker_a)

    async def cancel_via_worker_b(n_loads: int) -> None:
        if n_loads != 1:
            return
        # The cancel request is served by worker B's process (its own store).
        monkeypatch.setattr(causal_routes, "_discover_effects_store", worker_b)
        try:
            resp = await causal_routes.cancel_discover_causal_effects("job-1", user={})
        finally:
            monkeypatch.setattr(causal_routes, "_discover_effects_store", worker_a)
        assert resp.status == "running" and resp.cancel_requested is True
        # The marker never reached Redis, so worker A cannot see it.
        assert await worker_a.has_marker("job-1", "cancel") is False

    task_env["hooks"]["on_load"] = cancel_via_worker_b
    await causal_routes._run_discover_effects_task(
        "job-1", "patient_journeys", list(CANDIDATES), "synthetic", "Remibrutinib"
    )
    job = await worker_a.get("job-1")
    assert job is not None
    assert job.status == "cancelled", "the acknowledged cancel was ignored by the task"
    assert job.completed == 1 and job.total == 3
    assert len(task_env["calls"]) == 1
    assert sum(1 for e in job.effects if e.status == "cancelled") == 2


# ---------------------------------------------------------------------------
# Orphaned runs. The task dies without publishing (API restart on deploy, a
# gunicorn worker recycled at --max-requests, a crash) and the row used to stay
# `running` until the 8h TTL while the FE polled forever. The task stamps a
# liveness heartbeat; a poll on ANY worker repairs a non-terminal row whose
# heartbeat is gone to an honest `failed`. No startup sweep: with 2 workers a
# fresh worker cannot tell whether the OTHER worker's job is still alive.
# ---------------------------------------------------------------------------


def _running_row(job_id: str = "job-orphan") -> DiscoverEffectsResponse:
    """A run mid-flight: one question kept (with its estimate), one in flight,
    one still queued."""
    q0, q1, q2 = CANDIDATES
    return DiscoverEffectsResponse(
        job_id=job_id,
        status="running",
        dataset="patient_journeys",
        brand="Remibrutinib",
        total=3,
        completed=1,
        effects=[
            DiscoveredEffect(
                treatment=q0.treatment,
                outcome=q0.outcome,
                brand=q0.brand,
                status="completed",
                ate=0.182,
            ),
            DiscoveredEffect(
                treatment=q1.treatment, outcome=q1.outcome, brand=q1.brand, status="running"
            ),
            DiscoveredEffect(
                treatment=q2.treatment, outcome=q2.outcome, brand=q2.brand, status="pending"
            ),
        ],
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_stamps_a_liveness_heartbeat(scope):
    """The row is alive from the moment it exists: a poll that lands before the
    task's first beat must not declare a brand-new job dead."""
    client = _wire_client()
    r = client.post(
        "/causal/discover-effects",
        params={"dataset": "patient_journeys", "brand": "Remibrutinib"},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    age = await scope["store"].marker_age_seconds(job_id, causal_routes._DISCOVERY_ALIVE_MARKER)
    assert age is not None and age < 5
    assert client.get(f"/causal/discover-effects/{job_id}").json()["status"] == "pending"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_repairs_a_running_job_whose_heartbeat_is_gone(scope):
    """API restarted (deploy) / worker recycled: the task is gone, the row says
    `running`, no heartbeat exists. The poll reports the run `failed` with the
    reason, keeps the finished row (and its estimate), marks the in-flight
    question `failed` and the unrun one `cancelled` — nothing fabricated — and
    PERSISTS the repair so every later poll (on any worker) agrees."""
    store = scope["store"]
    await store.set("job-orphan", _running_row())
    client = _wire_client()
    r = client.get("/causal/discover-effects/job-orphan")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "failed"
    assert "interrupted" in body["error"].lower()
    assert body["completed"] == 1 and body["total"] == 3
    by_t = {e["treatment"]: e for e in body["effects"]}
    assert by_t["treatment_arm"]["status"] == "completed"
    assert by_t["treatment_arm"]["ate"] == pytest.approx(0.182)
    assert by_t["sample_dropped"]["status"] == "failed" and by_t["sample_dropped"]["ate"] is None
    assert by_t["copay_card_used"]["status"] == "cancelled"
    assert by_t["copay_card_used"]["ate"] is None
    persisted = await store.get("job-orphan")
    assert persisted is not None and persisted.status == "failed"
    assert persisted.error == body["error"]
    assert client.get("/causal/discover-effects/job-orphan").json()["status"] == "failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_leaves_a_live_running_job_alone(scope):
    store = scope["store"]
    await store.set("job-live", _running_row("job-live"))
    await store.touch_marker("job-live", causal_routes._DISCOVERY_ALIVE_MARKER, ttl_seconds=120)
    client = _wire_client()
    body = client.get("/causal/discover-effects/job-live").json()
    assert body["status"] == "running" and body.get("error") is None
    assert [e["status"] for e in body["effects"]] == ["completed", "running", "pending"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_treats_a_stale_heartbeat_as_dead(scope, monkeypatch):
    """A heartbeat older than the budget is as good as none: the task stopped
    beating (killed mid-question) even though its last stamp is still there."""
    store = scope["store"]
    await store.set("job-stale", _running_row("job-stale"))
    await store.touch_marker("job-stale", causal_routes._DISCOVERY_ALIVE_MARKER, ttl_seconds=120)
    # Budget of -1s: any stamp, however fresh, is past it.
    monkeypatch.setattr(causal_routes, "_DISCOVERY_HEARTBEAT_TTL_SECONDS", -1)
    client = _wire_client()
    assert client.get("/causal/discover-effects/job-stale").json()["status"] == "failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_never_repairs_a_finished_job(scope):
    """Terminal rows have no live heartbeat by design (the task is gone because
    it FINISHED). completed / cancelled must stay exactly as published."""
    store = scope["store"]
    done = _running_row("job-done").model_copy(update={"status": "completed", "completed": 3})
    await store.set("job-done", done)
    stopped = _running_row("job-stop").model_copy(
        update={"status": "cancelled", "cancel_requested": True}
    )
    await store.set("job-stop", stopped)
    client = _wire_client()
    assert client.get("/causal/discover-effects/job-done").json()["status"] == "completed"
    assert client.get("/causal/discover-effects/job-stop").json()["status"] == "cancelled"
    persisted = await store.get("job-done")
    assert persisted is not None and persisted.status == "completed" and persisted.error is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_on_a_dead_job_reports_it_failed_not_stopping(scope):
    """A cancel that lands after the task died must not pretend the run will
    stop 'after the current question' — there is no current question."""
    store = scope["store"]
    await store.set("job-dead", _running_row("job-dead"))
    client = _wire_client()
    r = client.post("/causal/discover-effects/job-dead/cancel")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "failed"
    assert r.json()["cancel_requested"] is False
    assert await store.has_marker("job-dead", causal_routes._DISCOVERY_CANCEL_MARKER) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_beats_periodically_while_a_question_runs_and_stops_with_the_run(
    task_env, monkeypatch
):
    """The estimators run in a worker thread, so the event loop is free to beat.
    The beat must be PERIODIC (a single stamp at start would go stale during a
    3-minute question) and must stop when the run ends."""
    store = task_env["store"]
    monkeypatch.setattr(causal_routes, "_DISCOVERY_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    touches: List[int] = []
    real_touch = store.touch_marker

    async def counting_touch(job_id, marker, *, ttl_seconds):
        touches.append(ttl_seconds)
        await real_touch(job_id, marker, ttl_seconds=ttl_seconds)

    monkeypatch.setattr(store, "touch_marker", counting_touch)
    seen: Dict[str, Any] = {}

    async def slow_question(n_loads: int) -> None:
        if n_loads != 1:
            return
        before = len(touches)
        await asyncio.sleep(0.1)
        seen["beats_during_question"] = len(touches) - before
        seen["age_during_question"] = await store.marker_age_seconds(
            "job-hb", causal_routes._DISCOVERY_ALIVE_MARKER
        )

    task_env["hooks"]["on_load"] = slow_question
    await causal_routes._run_discover_effects_task(
        "job-hb", "patient_journeys", list(CANDIDATES), "synthetic", "Remibrutinib"
    )
    assert seen["beats_during_question"] >= 3
    assert seen["age_during_question"] is not None and seen["age_during_question"] < 1
    assert set(touches) == {causal_routes._DISCOVERY_HEARTBEAT_TTL_SECONDS}
    n_after = len(touches)
    await asyncio.sleep(0.1)
    assert len(touches) == n_after, "the heartbeat outlived the run"
    job = await store.get("job-hb")
    assert job is not None and job.status == "completed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_crash_outside_a_question_marks_the_run_failed_not_stuck(task_env, monkeypatch):
    """Per-question errors are already caught; an error OUTSIDE a question (the
    pre-rank, here) used to kill the task silently and leave the row `running`
    until the TTL. It must publish an honest `failed` row with the reason."""

    async def boom(dataset, questions):
        raise RuntimeError("prerank exploded")

    monkeypatch.setattr(causal_routes, "_prerank_questions", boom)
    await causal_routes._run_discover_effects_task(
        "job-crash", "patient_journeys", list(CANDIDATES), "synthetic", "Remibrutinib"
    )
    job = await task_env["store"].get("job-crash")
    assert job is not None and job.status == "failed"
    assert "prerank exploded" in (job.error or "")
    assert job.completed == 0
    assert {e.status for e in job.effects} == {"cancelled"}
    assert all(e.ate is None for e in job.effects)
    assert len(task_env["calls"]) == 0
