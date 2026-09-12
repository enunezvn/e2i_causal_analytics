"""The composition recorder against a prod-faithful database migrated through ml/041.

``CompositionRecorder`` enqueues every write on one background chain per composition and never
blocks ``compose()``. Each RPC carries the seed, so whichever write lands records the
composition; each write retries once, then is counted and dropped; the finish snapshot restores
every field a lost phase write carried and re-sends every step the recorder was given. These
tests inject transport faults around a REAL database (psycopg as service_role): the database
behaviour itself is never simulated.

Opt-in: ``E2I_DB_INTEGRATION=1``. Run with ``-n 0``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from src.agents.tool_composer import composer as _composer  # noqa: F401 - registers every tool
from src.agents.tool_composer import learning_recorder
from src.agents.tool_composer.executor import PlanExecutor
from src.agents.tool_composer.learning_recorder import CompositionRecorder, drain
from src.agents.tool_composer.registry_sync import RegistrySync
from src.tool_registry.registry import ToolSchema, get_registry
from tests.unit.test_agents.test_tool_composer.test_learning_recorder_serializer import (
    SENTINEL,
    _decomposition,
    _plan,
    _result,
    _sentinel_models,
    _step,
)
from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(300),
]

UPTO = "ml/041_composer_learning_loop_recording.sql"


class FaultPort:
    """A real PsycopgRpcPort with injected transport faults.

    ``fail`` maps an RPC name to how many calls fail. ``after=True`` executes the call first and
    then raises, as a response lost on the way back does. ``gate`` (an Event) makes the named
    RPCs fail until it is set.
    """

    def __init__(
        self,
        inner: _pg.PsycopgRpcPort,
        fail: Optional[Dict[str, int]] = None,
        *,
        after: bool = False,
        gate: Optional[asyncio.Event] = None,
        gated: tuple = (),
    ):
        self.inner = inner
        self.fail = dict(fail or {})
        self.after = after
        self.gate = gate
        self.gated = set(gated)
        self.calls: List[str] = []

    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        self.calls.append(name)
        if self.gate is not None and name in self.gated and not self.gate.is_set():
            raise ConnectionError(f"{name}: injected outage")
        if self.fail.get(name, 0) > 0:
            self.fail[name] -= 1
            if self.after:
                await self.inner.call(name, params)
                raise ConnectionError(f"{name}: injected lost response")
            raise ConnectionError(f"{name}: injected failure")
        return await self.inner.call(name, params)


def _seed(cid: str) -> Dict[str, Any]:
    return {
        "composition_id": cid,
        "query_text": "which regions drive TRx?",
        "session_id": f"sess-{cid}",
        "user_id": "user-1",
        "entry_point": "chat_tool",
        "brand": "Kisqali",
        "region": "US",
        "audit_workflow_id": None,
        "is_synthetic": True,
    }


def _recorder(cid: str, port: Any, *, sync_port: Any = None, **kw: Any) -> CompositionRecorder:
    settings = {"write_timeout_s": 2.0, "retry_delay_s": 0.05, "heartbeat_s": 3600.0}
    settings.update(kw)
    return CompositionRecorder(
        cid, _seed(cid), port=port, sync=RegistrySync(port=sync_port or port), **settings
    )


@pytest.fixture
def synced(clone_db) -> _pg.PgConn:
    """Migrated through 041 and synced to the live registry (all 20 tools present)."""
    db = clone_db("recorder")
    _pg.migrate(db, UPTO)
    asyncio.run(RegistrySync(port=_pg.PsycopgRpcPort(db)).sync_once())
    return db


def _one(db: _pg.PgConn, sql: str, *params: Any) -> Any:
    with db.connect() as conn:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None


def _episode(db: _pg.PgConn, cid: str) -> Dict[str, Any]:
    return _one(
        db, "select row_to_json(e)::jsonb from composer_episodes e where composition_id = %s", cid
    )


def _count(db: _pg.PgConn, table: str, cid: str) -> int:
    if table == "composer_episodes":
        return _one(db, "select count(*) from composer_episodes where composition_id = %s", cid)
    if table == "tool_performance":
        return _one(db, "select count(*) from tool_performance where composition_id = %s", cid)
    return _one(
        db,
        "select count(*) from composition_steps s join composer_episodes e using (episode_id) "
        "where e.composition_id = %s",
        cid,
    )


def _two_step_models():
    d = _decomposition(["CAUSAL", "COMPARATIVE"])
    ate = _step(
        "kpi_ate", "causal_effect_estimator", "sq_0", {"treatment": "brand", "outcome": "x"}
    )
    gap = _step("kpi_gap", "gap_calculator", "sq_1", {"metric": "region"}, deps=["kpi_ate"])
    plan = _plan(d, [ate, gap], [["kpi_ate"], ["kpi_gap"]])
    r0 = _result(ate, outcome_class="succeeded", result={"ate": 0.1})
    r1 = _result(gap, outcome_class="refused", error="nope", error_type="ToolRefusalError")
    return d, plan, [r0, r1]


def _run_full(recorder: CompositionRecorder, d, plan, results) -> None:
    recorder.start()
    recorder.decomposed(d, latency_ms=100)
    recorder.planned(plan, latency_ms=200, plan_source="llm")
    for n, r in enumerate(results):
        recorder.step(n, r)
    recorder.executed(latency_ms=300)
    recorder.finish(
        status="COMPLETED",
        outcome="partial",
        total_latency_ms=1000,
        synthesize_latency_ms=400,
        tools_executed=len(results),
        tools_succeeded=sum(r.outcome_class == "succeeded" for r in results),
    )


# ---------------------------------------------------------------------------
# Lost writes
# ---------------------------------------------------------------------------


async def test_start_failure_then_later_writes_create_episode(synced):
    port = FaultPort(_pg.PsycopgRpcPort(synced), {"composer_record_start": 2})
    d, plan, results = _two_step_models()
    _run_full(_recorder("lost_start", port), d, plan, results)
    await drain(timeout=30)
    ep = _episode(synced, "lost_start")
    assert port.calls.count("composer_record_start") == 2  # the write and its one retry
    assert _count(synced, "composer_episodes", "lost_start") == 1
    assert (ep["status"], ep["outcome"], ep["plan_source"]) == ("COMPLETED", "partial", "llm")
    assert ep["sub_questions"] == [
        {"index": 0, "intent": "CAUSAL"},
        {"index": 1, "intent": "COMPARATIVE"},
    ]
    assert [s["tool_name"] for s in ep["tool_plan"]["steps"]] == [
        "causal_effect_estimator",
        "gap_calculator",
    ]
    assert ep["parallelizable_groups"] == [[0], [1]]
    assert (ep["decompose_latency_ms"], ep["plan_latency_ms"], ep["execute_latency_ms"]) == (
        100,
        200,
        300,
    )
    assert _count(synced, "composition_steps", "lost_start") == 2


async def test_exhausted_phase_retry_restored_by_finish(synced):
    port = FaultPort(_pg.PsycopgRpcPort(synced), {"composer_record_phase": 100})
    d, plan, results = _two_step_models()
    _run_full(_recorder("lost_phase", port), d, plan, results)
    await drain(timeout=30)
    ep = _episode(synced, "lost_phase")
    assert port.calls.count("composer_record_phase") == 6  # 3 phase writes x (1 + 1 retry)
    assert ep["plan_source"] == "llm" and ep["parallelizable_groups"] == [[0], [1]]
    assert len(ep["tool_plan"]["steps"]) == 2 and len(ep["sub_questions"]) == 2
    assert (ep["decompose_latency_ms"], ep["plan_latency_ms"], ep["execute_latency_ms"]) == (
        100,
        200,
        300,
    )
    assert ep["synthesize_latency_ms"] == 400 and ep["total_latency_ms"] == 1000


async def test_lost_response_resend_no_duplicates(synced):
    port = FaultPort(_pg.PsycopgRpcPort(synced), {"composer_record_steps": 1}, after=True)
    d, plan, results = _two_step_models()
    _run_full(_recorder("lost_response", port), d, plan, results)
    await drain(timeout=30)
    assert _count(synced, "composition_steps", "lost_response") == 2
    assert _count(synced, "tool_performance", "lost_response") == 2
    assert _count(synced, "composer_episodes", "lost_response") == 1


async def test_exhausted_step_write_resent_by_finish(synced):
    port = FaultPort(_pg.PsycopgRpcPort(synced), {"composer_record_steps": 2})
    d, plan, results = _two_step_models()
    _run_full(_recorder("lost_step", port), d, plan, results)
    await drain(timeout=30)
    assert _count(synced, "composition_steps", "lost_step") == 2
    assert _count(synced, "tool_performance", "lost_step") == 2


# ---------------------------------------------------------------------------
# Cancel with a real executor
# ---------------------------------------------------------------------------


@pytest.fixture
def probe_tools():
    """psi_calculator (fast) and gap_calculator (slow) with probe callables: both names exist in
    the DB registry, so their steps can be recorded."""
    live = get_registry()
    snapshot = live.snapshot()
    live.clear()

    async def fast(**_: Any) -> Any:
        return {"psi": 0.01}

    async def slow(**_: Any) -> Any:
        await asyncio.sleep(30)
        return {"gap": 1.0}

    for name, fn in (("psi_calculator", fast), ("gap_calculator", slow)):
        live.register(
            schema=ToolSchema(
                name=name, description="recorder probe tool.", source_agent="drift_monitor", tier=3
            ),
            callable=fn,
        )
    try:
        yield live
    finally:
        live.restore_snapshot(snapshot)


@pytest.mark.parametrize("same_group", [True, False])
async def test_cancel_resends_retained_steps(synced, probe_tools, same_group):
    cid = f"cancel_{'same' if same_group else 'earlier'}"
    outage = asyncio.Event()
    port = FaultPort(_pg.PsycopgRpcPort(synced), gate=outage, gated=("composer_record_steps",))
    recorder = _recorder(cid, port)
    d = _decomposition(["CAUSAL"])
    fast = _step("fast", "psi_calculator", "sq_0", {})
    slow = _step("slow", "gap_calculator", "sq_0", {})
    groups = [["fast", "slow"]] if same_group else [["fast"], ["slow"]]
    plan = _plan(d, [fast, slow], groups)
    fast_reported = asyncio.Event()

    def on_step(n: int, result: Any) -> None:
        recorder.step(n, result)
        if result.step_id == "fast":
            fast_reported.set()

    recorder.start()
    recorder.decomposed(d, latency_ms=1)
    recorder.planned(plan, latency_ms=1, plan_source="llm")
    executor = PlanExecutor(tool_registry=probe_tools, enable_caching=False, max_retries=0)
    task = asyncio.create_task(executor.execute(plan, on_step_result=on_step))
    await asyncio.wait_for(fast_reported.wait(), timeout=10)
    if not same_group:
        await asyncio.sleep(0.2)  # the slow step is running in the next group
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Release the outage only after the fast step's write AND its retry failed, so the only way
    # its row can land is the finish re-sending the retained step.
    for _ in range(200):
        if port.calls.count("composer_record_steps") >= 2:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("the step write never ran into the outage")
    await asyncio.sleep(0.1)
    outage.set()  # the step writes that failed during the outage are re-sent by the finish
    recorder.cancelled("execute")
    await drain(timeout=30)

    ep = _episode(synced, cid)
    assert (ep["status"], ep["outcome"], ep["failed_phase"]) == ("FAILED", "cancelled", "execute")
    steps = _one(
        synced,
        "select jsonb_agg(s.tool_name order by s.step_number) from composition_steps s "
        "join composer_episodes e using (episode_id) where e.composition_id = %s",
        cid,
    )
    assert steps == ["psi_calculator"]
    assert _count(synced, "tool_performance", cid) == 1


# ---------------------------------------------------------------------------
# Structure only, end to end
# ---------------------------------------------------------------------------


async def test_sentinel_absent_after_real_db_round_trip(synced):
    d, plan, trace = _sentinel_models()
    recorder = _recorder("sentinel", _pg.PsycopgRpcPort(synced))
    recorder.start()
    recorder.decomposed(d, latency_ms=1)
    recorder.planned(plan, latency_ms=1, plan_source="llm")
    for n, result in enumerate(trace.step_results):
        recorder.step(n, result)
    recorder.executed(latency_ms=1)
    recorder.finish(status="FAILED", outcome="failed", failed_phase="execute", error_type="ExecutionError",
                    total_latency_ms=5, tools_executed=4, tools_succeeded=1)  # fmt: skip
    await drain(timeout=30)
    rows = [
        _episode(synced, "sentinel"),
        *(_one(
            synced,
            "select coalesce(jsonb_agg(row_to_json(s)), '[]') from composition_steps s "
            "join composer_episodes e using (episode_id) where e.composition_id = %s",
            "sentinel",
        ) or []),
        *(_one(
            synced,
            "select coalesce(jsonb_agg(row_to_json(p)), '[]') from tool_performance p where composition_id = %s",
            "sentinel",
        ) or []),
    ]  # fmt: skip
    assert len(rows) == 1 + 4 + 4
    assert SENTINEL not in json.dumps(rows)
    ep = rows[0]
    assert (
        ep["error_message"] is None
        and ep["synthesized_response"] is None
        and ep["tool_outputs"] == {}
    )
    assert all(r.get("error_message") is None for r in rows[1:5])


async def test_allowlist_none_keeps_no_names_end_to_end(synced):
    inner = _pg.PsycopgRpcPort(synced)
    port = FaultPort(inner, {"composer_public_column_names": 100})
    d, plan, results = _two_step_models()  # treatment="brand", metric="region": catalog columns
    _run_full(_recorder("no_allowlist", port), d, plan, results)
    await drain(timeout=30)
    stored = json.dumps(
        [
            _episode(synced, "no_allowlist")["tool_plan"],
            _one(
                synced,
                "select jsonb_agg(s.input_params) from composition_steps s join composer_episodes e "
                "using (episode_id) where e.composition_id = %s",
                "no_allowlist",
            ),
        ]
    )
    assert '"column"' not in stored and '"str"' in stored

    control_port = _pg.PsycopgRpcPort(synced)
    _run_full(_recorder("with_allowlist", control_port), *_two_step_models())
    await drain(timeout=30)
    assert '"column"' in json.dumps(
        _episode(synced, "with_allowlist")["tool_plan"]
    )  # positive control


async def test_unknown_tools_trigger_lazy_sync_then_resend(clone_db):
    db = clone_db("lazy_sync")
    _pg.migrate(db, UPTO)  # no sync: cohort_builder has no registry row yet
    port = _pg.PsycopgRpcPort(db)
    recorder = _recorder("lazy", port)
    d = _decomposition(["DESCRIPTIVE"])
    step = _step("build", "cohort_builder", "sq_0", {"brand": "brand"})
    plan = _plan(d, [step], [["build"]])
    recorder.start()
    recorder.planned(plan, latency_ms=1, plan_source="llm")
    recorder.step(0, _result(step, outcome_class="succeeded", result={"total_eligible": 3}))
    await drain(timeout=60)
    assert port.calls.count("sync_tool_registry") == 1
    assert port.calls.count("composer_record_steps") == 2  # reported unknown, then re-sent
    assert _count(db, "composition_steps", "lazy") == 1
    assert _one(db, "select count(*) from tool_registry where deprecated_at is null") == 20


async def test_concurrent_recorders_sharing_one_sync_both_record(clone_db):
    db = clone_db("lazy_race")
    _pg.migrate(db, UPTO)  # no sync yet
    port = _pg.PsycopgRpcPort(db)
    shared = RegistrySync(port=port)
    d = _decomposition(["DESCRIPTIVE"])
    step = _step("build", "cohort_builder", "sq_0", {"brand": "brand"})
    plan = _plan(d, [step], [["build"]])
    for cid in ("race_1", "race_2"):
        recorder = CompositionRecorder(
            cid, _seed(cid), port=port, sync=shared, retry_delay_s=0.05, heartbeat_s=3600.0
        )
        recorder.planned(plan, latency_ms=1, plan_source="llm")
        recorder.step(0, _result(step, outcome_class="succeeded", result={"total_eligible": 3}))
    await drain(timeout=60)
    assert port.calls.count("sync_tool_registry") == 1
    assert _count(db, "composition_steps", "race_1") == 1
    assert _count(db, "composition_steps", "race_2") == 1


# ---------------------------------------------------------------------------
# Liveness and drain
# ---------------------------------------------------------------------------


async def test_heartbeat_stops_once_another_finish_closed_the_episode(synced):
    beating = _recorder("closed_elsewhere", _pg.PsycopgRpcPort(synced), heartbeat_s=0.3)
    beating.start()
    await drain(timeout=30)
    closer = _recorder("closed_elsewhere", _pg.PsycopgRpcPort(synced))
    closer.finish(status="COMPLETED", outcome="success", total_latency_ms=1)
    await drain(timeout=30)
    for _ in range(40):
        if beating.heartbeat_stopped:
            break
        await asyncio.sleep(0.1)
    assert beating.heartbeat_stopped


async def test_heartbeat_keeps_activity_fresh(synced):
    recorder = _recorder("beating", _pg.PsycopgRpcPort(synced), heartbeat_s=1.0)
    recorder.start()
    await asyncio.sleep(0.5)
    lags = []
    for _ in range(10):  # a 10 s step
        await asyncio.sleep(1.0)
        lags.append(
            _one(
                synced,
                "select extract(epoch from now() - last_activity_at) from composer_episodes "
                "where composition_id = %s",
                "beating",
            )
        )
    recorder.finish(status="COMPLETED", outcome="success", total_latency_ms=10_500)
    await drain(timeout=30)
    assert all(lag is not None and float(lag) <= 2.5 for lag in lags), lags
    assert recorder.heartbeat_stopped


async def test_drain_flushes_pending(synced):
    d = _decomposition(["CAUSAL"])
    steps = [_step(f"s{i}", "psi_calculator", "sq_0", {}) for i in range(20)]
    plan = _plan(d, steps, [[s.step_id] for s in steps])
    recorder = _recorder("drained", _pg.PsycopgRpcPort(synced))
    recorder.start()
    recorder.planned(plan, latency_ms=1, plan_source="llm")
    for n, step in enumerate(steps):
        recorder.step(n, _result(step, outcome_class="succeeded", result={"psi": 0.1}))
    assert learning_recorder.pending_count() > 0
    await drain(timeout=30)
    assert learning_recorder.pending_count() == 0
    assert _count(synced, "composition_steps", "drained") == 20
