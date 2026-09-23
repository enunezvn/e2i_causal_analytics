"""#2233 defect 2: an agent-analyze row whose worker died must surface as ``failed``.

Live (2026-09-22, analysis b0cf6946): gunicorn aborted the worker (code 134) in
the middle of the background task; the row stayed ``running`` with no error for
the whole 8h TTL while the caller polled to its 1000 s cap. gunicorn's
``worker_abort`` hook never fires under ``UvicornWorker`` (config/gunicorn.conf.py,
measured), so the mechanism is the one ``discover-effects`` already uses on the
same store class: the task stamps a liveness heartbeat, and a poll on ANY worker
read-repairs a non-terminal row whose heartbeat is gone.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.routes.causal import agent as agent_routes
from src.api.schemas.causal import (
    AgentCausalAnalysisRequest,
    AgentCausalAnalysisResponse,
    CausalDAGModel,
    RefutationSummary,
)


def _memory_store() -> DurableJobStore:
    async def boom():
        raise RuntimeError("redis not initialised")

    return DurableJobStore("test:agent_analyze", AgentCausalAnalysisResponse, redis_factory=boom)


def _row(analysis_id: str, status: str) -> AgentCausalAnalysisResponse:
    return AgentCausalAnalysisResponse(
        analysis_id=analysis_id,
        status=status,
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
        n_rows=10,
        data_source="synthetic",
        dag=CausalDAGModel(),
        statistical_significance=False,
        refutation=RefutationSummary(),
        warnings=["Analysis submitted; poll GET /causal/agent-analyze/{id} for the result."],
        latency_ms=0,
    )


def _wire_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.dependencies.auth import require_analyst, require_viewer
    from src.api.routes.causal import router as causal_package_router

    app = FastAPI()
    app.include_router(causal_package_router)
    app.dependency_overrides[require_viewer] = lambda: {"role": "viewer"}
    app.dependency_overrides[require_analyst] = lambda: {"role": "analyst"}
    return TestClient(app)


@pytest.fixture
def store(monkeypatch):
    s = _memory_store()
    monkeypatch.setattr(agent_routes, "_agent_analysis_store", s)
    return s


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_repairs_a_running_row_whose_heartbeat_is_gone(store):
    await store.set("aid-orphan", _row("aid-orphan", "running"))
    r = _wire_client().get("/causal/agent-analyze/aid-orphan")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "failed"
    assert any("interrupted" in w.lower() for w in body["warnings"]), body["warnings"]
    assert body["ate"] is None  # nothing fabricated
    persisted = await store.get("aid-orphan")
    assert persisted is not None and persisted.status == "failed"
    assert _wire_client().get("/causal/agent-analyze/aid-orphan").json()["status"] == "failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_leaves_a_live_running_row_alone(store):
    await store.set("aid-live", _row("aid-live", "running"))
    await store.touch_marker("aid-live", agent_routes._AGENT_ALIVE_MARKER, ttl_seconds=120)
    body = _wire_client().get("/causal/agent-analyze/aid-live").json()
    assert body["status"] == "running"
    assert not any("interrupted" in w.lower() for w in body["warnings"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_treats_a_stale_heartbeat_as_dead(store, monkeypatch):
    await store.set("aid-stale", _row("aid-stale", "running"))
    await store.touch_marker("aid-stale", agent_routes._AGENT_ALIVE_MARKER, ttl_seconds=120)
    monkeypatch.setattr(agent_routes, "_AGENT_HEARTBEAT_TTL_SECONDS", -1)
    assert _wire_client().get("/causal/agent-analyze/aid-stale").json()["status"] == "failed"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "needs_review", "failed"])
async def test_poll_never_repairs_a_terminal_row(store, status):
    await store.set("aid-done", _row("aid-done", status))
    body = _wire_client().get("/causal/agent-analyze/aid-done").json()
    assert body["status"] == status
    assert not any("interrupted" in w.lower() for w in body["warnings"])


@pytest.mark.unit
def test_heartbeat_ttl_is_the_gunicorn_worker_timeout():
    """A loop stalled that long is killed anyway (docker/Dockerfile --timeout 120)."""
    assert agent_routes._AGENT_HEARTBEAT_TTL_SECONDS == 120
    assert 0 < agent_routes._AGENT_HEARTBEAT_INTERVAL_SECONDS < 120


class _BG:
    def __init__(self) -> None:
        self.scheduled: list = []

    def add_task(self, fn, *args, **kwargs):
        self.scheduled.append((fn, args))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_stamps_a_liveness_heartbeat(store):
    """The row is alive from the moment it exists: a poll that lands before the
    task's first beat must not declare a brand-new job dead."""
    frame = pd.DataFrame(
        {"treatment_arm": [1.0, 0.0], "persistent_180d": [1.0, 0.0], "disease_severity": [2.0, 1.0]}
    )
    req = AgentCausalAnalysisRequest(
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
        covariates=["disease_severity"],
        limit=1500,
    )
    loader = AsyncMock(return_value=(frame, list(frame.columns)))
    with (
        patch.object(agent_routes, "_load_agent_estimation_frame", loader),
        patch.object(agent_routes, "_run_agent_analysis_task", AsyncMock()),
    ):
        pending = await agent_routes.run_causal_agent_analysis(req, _BG(), user={"sub": "t"})
    age = await store.marker_age_seconds(pending.analysis_id, agent_routes._AGENT_ALIVE_MARKER)
    assert age is not None and age < 5
    assert _wire_client().get(f"/causal/agent-analyze/{pending.analysis_id}").json()["status"] == (
        "pending"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_beats_while_the_graph_runs_and_stops_after(store, monkeypatch):
    import src.agents.causal_impact.graph as graph_mod

    touches: list = []
    real_touch = store.touch_marker

    async def spy(job_id, marker, *, ttl_seconds):
        touches.append(marker)
        await real_touch(job_id, marker, ttl_seconds=ttl_seconds)

    monkeypatch.setattr(store, "touch_marker", spy)
    monkeypatch.setattr(agent_routes, "_AGENT_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    seen_alive: list = []

    class _SlowGraph:
        async def ainvoke(self, state, **kwargs):
            await asyncio.sleep(0.1)
            seen_alive.append(await agent_routes._agent_is_alive("aid-beat"))
            raise RuntimeError("stop: the mapping is not under test")

    monkeypatch.setattr(graph_mod, "create_causal_impact_graph", lambda: _SlowGraph())
    await store.set("aid-beat", _row("aid-beat", "pending"))
    frame = pd.DataFrame({"treatment_arm": [1.0, 0.0], "persistent_180d": [1.0, 0.0]})
    req = AgentCausalAnalysisRequest(
        treatment_var="treatment_arm", outcome_var="persistent_180d", dataset="patient_journeys"
    )
    await agent_routes._run_agent_analysis_task("aid-beat", req, frame, [], "synthetic")
    assert seen_alive == [True]
    assert len(touches) >= 3, touches  # the inline first beat + periodic beats
    n_after = len(touches)
    await asyncio.sleep(0.05)
    assert len(touches) == n_after, "the heartbeat outlived the run"
    persisted = await store.get("aid-beat")
    assert persisted is not None and persisted.status == "failed"


# ---------------------------------------------------------------------------
# Codex r1 HIGH: the read-repair and the task's own terminal write can race.
# The store has no CAS, so both sides re-read immediately before writing and a
# terminal row is never replaced: a repair that lost the race returns the real
# result, and a task whose row was already repaired lets the repaired row stand
# (the discover-effects contract, discovery.py "repaired").
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repair_does_not_overwrite_a_row_that_turned_terminal_meanwhile(store, monkeypatch):
    """The task publishes ``completed`` between the poll's liveness check and
    its repair write: the real result must stand and be what the poll returns."""
    await store.set("aid-race", _row("aid-race", "running"))
    completed = _row("aid-race", "completed").model_copy(update={"ate": 0.12})

    async def dead_but_finishing(analysis_id: str) -> bool:
        await store.set(analysis_id, completed)  # the producer wins the race
        return False

    monkeypatch.setattr(agent_routes, "_agent_is_alive", dead_but_finishing)
    body = _wire_client().get("/causal/agent-analyze/aid-race").json()
    assert body["status"] == "completed" and body["ate"] == pytest.approx(0.12)
    persisted = await store.get("aid-race")
    assert persisted is not None and persisted.status == "completed"
    assert not any("interrupted" in w.lower() for w in persisted.warnings)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_lets_a_repaired_row_stand(store, monkeypatch):
    """A poll on some worker already closed this row as ``failed`` (no live
    heartbeat seen — e.g. THIS worker's beats never reached Redis). The task's
    late terminal write must not resurrect or replace it."""
    import src.agents.causal_impact.graph as graph_mod

    repaired = _row("aid-stand", "failed").model_copy(
        update={"warnings": [agent_routes._AGENT_INTERRUPTED_WARNING]}
    )

    class _RepairedMeanwhile:
        async def ainvoke(self, state, **kwargs):
            await store.set("aid-stand", repaired)  # the poll's repair landed
            raise RuntimeError("the task then ends (any terminal outcome)")

    monkeypatch.setattr(graph_mod, "create_causal_impact_graph", lambda: _RepairedMeanwhile())
    await store.set("aid-stand", _row("aid-stand", "running"))
    frame = pd.DataFrame({"treatment_arm": [1.0, 0.0], "persistent_180d": [1.0, 0.0]})
    req = AgentCausalAnalysisRequest(
        treatment_var="treatment_arm", outcome_var="persistent_180d", dataset="patient_journeys"
    )
    await agent_routes._run_agent_analysis_task("aid-stand", req, frame, [], "synthetic")
    persisted = await store.get("aid-stand")
    assert persisted is not None and persisted.status == "failed"
    assert persisted.warnings == [agent_routes._AGENT_INTERRUPTED_WARNING]
