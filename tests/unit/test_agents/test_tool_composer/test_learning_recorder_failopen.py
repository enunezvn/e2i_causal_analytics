"""Recording is fail-open and off the user's path (spec §5.4). No database: CI-safe.

Nothing ``compose()`` calls on the recorder awaits I/O: every method is a synchronous enqueue
onto a background chain. So an unreachable database (connection refused) or a hanging one adds
no latency to a composition and never raises into it. A write that fails is logged once with the
RPC, the composition id and the error class, and counted in
``e2i_composer_record_failures_total{rpc}``: a transport failure or timeout after its one retry, a
permanent failure (a rejected payload, a database error) at once. No wait inside the chain is
unbounded (catalog fetch and registry sync included), a malformed value never stops the rest of
a record, the seed crosses the network only as its identity fields, and the heartbeat always ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pytest

from src.agents.tool_composer import composer as _composer  # noqa: F401 - registers every tool
from src.agents.tool_composer import learning_recorder
from src.agents.tool_composer.learning_recorder import (
    CompositionRecorder,
    drain,
    structure_value,
)
from src.agents.tool_composer.registry_sync import RegistrySync
from tests.unit.test_agents.test_tool_composer.test_learning_recorder_serializer import (
    SENTINEL,
    _decomposition,
    _plan,
    _result,
    _step,
)

METRIC_GLOBALS = (
    "_metrics_initialized",
    "_metrics_registry",
    "_request_counter",
    "_request_latency",
    "_active_requests",
    "_error_counter",
    "_agent_invocations",
    "_health_gauge",
    "_composer_record_failures",
)


class RefusedPort:
    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        raise ConnectionRefusedError(111, "Connection refused")


class HangingPort:
    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        await asyncio.sleep(10)


class RaisingPort:
    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls: List[str] = []

    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        self.calls.append(name)
        raise self.exc


class ScriptedPort:
    """Answers like an empty database; hangs the named RPCs; records every payload."""

    def __init__(self, hang: Set[str] = frozenset(), heartbeat: Any = True, unknown: tuple = ()):
        self.hang = set(hang)
        self.heartbeat = heartbeat
        self.unknown = list(unknown)
        self.payloads: List[Tuple[str, Dict[str, Any]]] = []

    @property
    def calls(self) -> List[str]:
        return [name for name, _ in self.payloads]

    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        self.payloads.append((name, params))
        if name in self.hang:
            await asyncio.sleep(30)
        if name == "composer_record_steps":
            return {
                "recorded": len(params["p_steps"]),
                "already_present": 0,
                "unknown_tools": self.unknown,
                "schema_mismatch_tools": [],
            }
        if name == "composer_record_heartbeat":
            return self.heartbeat
        if name == "composer_public_column_names":
            return ["brand", "region"]
        if name == "sync_tool_registry":
            return dict.fromkeys(
                (
                    "inserted",
                    "updated",
                    "deprecated",
                    "dependencies_upserted",
                    "dependencies_deleted",
                ),
                0,
            )
        return None


@pytest.fixture(autouse=True)
async def _no_leftover_tasks():
    yield
    await drain(timeout=0, cancel_heartbeats=True)
    for task in list(learning_recorder._pending):
        task.cancel()
    await asyncio.sleep(0)


@pytest.fixture
def metrics_registry():
    """A fresh metrics registry; every collector global is restored afterwards."""
    import src.api.routes.metrics as metrics

    saved = {name: getattr(metrics, name) for name in METRIC_GLOBALS}
    metrics._metrics_initialized = False
    metrics._metrics_registry = None
    metrics._init_metrics()
    try:
        yield metrics._metrics_registry
    finally:
        for name, value in saved.items():
            setattr(metrics, name, value)


def _recorder(port: Any, cid: str = "comp_failopen", **kw: Any) -> CompositionRecorder:
    return CompositionRecorder(
        cid,
        {"composition_id": cid, "query_text": "q", "is_synthetic": True},
        port=port,
        sync=RegistrySync(port=port),
        **kw,
    )


def _failures(registry: Any, rpc: str) -> float:
    return registry.get_sample_value("e2i_composer_record_failures_total", {"rpc": rpc}) or 0.0


def _models():
    d = _decomposition(["CAUSAL"])
    step = _step("s", "gap_calculator", "sq_0", {"metric": "brand"})
    plan = _plan(d, [step], [["s"]])
    return d, plan, _result(step, outcome_class="succeeded", result={"gap": 0.1})


async def _composition(recorder: Optional[CompositionRecorder]) -> None:
    """A composition's shape: phases with real awaits between the recorder calls."""
    d, plan, result = _models()
    if recorder:
        recorder.start()
    await asyncio.sleep(0.05)
    if recorder:
        recorder.decomposed(d, latency_ms=50)
    await asyncio.sleep(0.05)
    if recorder:
        recorder.planned(plan, latency_ms=50, plan_source="llm")
    await asyncio.sleep(0.05)
    if recorder:
        recorder.step(0, result)
        recorder.executed(latency_ms=50)
    await asyncio.sleep(0.05)
    if recorder:
        recorder.finish(status="COMPLETED", outcome="success", total_latency_ms=200)


# ---------------------------------------------------------------------------
# Off the user's path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("port", [RefusedPort(), HangingPort()], ids=["refused", "hanging"])
async def test_no_latency_on_user_path(port):
    d, plan, result = _models()
    recorder = _recorder(port)
    started = time.perf_counter()
    recorder.start()
    recorder.decomposed(d, latency_ms=1)
    recorder.planned(plan, latency_ms=1, plan_source="llm")
    recorder.step(0, result)
    recorder.executed(latency_ms=1)
    recorder.finish(status="COMPLETED", outcome="success", total_latency_ms=5)
    assert time.perf_counter() - started < 0.05

    baseline_started = time.perf_counter()
    await asyncio.wait_for(_composition(None), timeout=1)
    baseline = time.perf_counter() - baseline_started

    recorded_started = time.perf_counter()
    await asyncio.wait_for(_composition(_recorder(port)), timeout=1)
    recorded = time.perf_counter() - recorded_started
    assert recorded - baseline < 0.05, (recorded, baseline)


async def test_drain_returns_at_its_timeout_while_writes_hang():
    recorder = _recorder(HangingPort())
    recorder.start()
    started = time.perf_counter()
    unfinished = await drain(timeout=0.2)
    assert time.perf_counter() - started < 1.0
    assert unfinished >= 1


# ---------------------------------------------------------------------------
# Failure accounting
# ---------------------------------------------------------------------------


async def test_failed_write_is_logged_and_counted(caplog, metrics_registry):
    recorder = _recorder(RefusedPort(), retry_delay_s=0.01)
    with caplog.at_level(logging.WARNING, logger=learning_recorder.__name__):
        recorder.start()
        assert await drain(timeout=5) == 0
    assert _failures(metrics_registry, "composer_record_start") == 1.0
    messages = [r.getMessage() for r in caplog.records if r.name == learning_recorder.__name__]
    assert any(
        "composer_record_start" in m and "comp_failopen" in m and "ConnectionRefusedError" in m
        for m in messages
    ), messages
    assert not any("Connection refused" in m for m in messages)  # the class, not the text


class OperationalError(Exception):
    """Named like psycopg's connection-level error class."""


@pytest.mark.parametrize(
    "exc, attempts",
    [
        (ConnectionError("down"), 2),
        (TimeoutError(), 2),
        (OSError("reset by peer"), 2),
        (OperationalError("server closed the connection"), 2),
        (ValueError("rejected payload"), 1),
        (RuntimeError("check_violation"), 1),
    ],
)
async def test_only_transport_failures_are_retried(exc, attempts, metrics_registry):
    port = RaisingPort(exc)
    recorder = _recorder(port, retry_delay_s=0.01)
    recorder.start()
    assert await drain(timeout=5) == 0
    assert port.calls.count("composer_record_start") == attempts
    assert _failures(metrics_registry, "composer_record_start") == 1.0


def test_failure_counter_helper_is_a_no_op_before_metrics_init(metrics_registry):
    import src.api.routes.metrics as metrics

    metrics.inc_composer_record_failure("composer_record_start")
    assert _failures(metrics_registry, "composer_record_start") == 1.0
    metrics._metrics_initialized = False  # restored by the fixture
    metrics.inc_composer_record_failure("composer_record_start")
    assert _failures(metrics_registry, "composer_record_start") == 1.0  # unchanged


# ---------------------------------------------------------------------------
# Bounded waits, isolated failures, the seed, the heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hung", ["composer_public_column_names", "sync_tool_registry"])
async def test_a_hanging_catalog_fetch_or_sync_does_not_stall_the_chain(hung):
    port = ScriptedPort(hang={hung}, unknown=("gap_calculator",))
    recorder = _recorder(port, write_timeout_s=0.2)
    d, plan, result = _models()
    recorder.planned(plan, latency_ms=1, plan_source="llm")
    recorder.step(0, result)
    recorder.finish(status="COMPLETED", outcome="success", total_latency_ms=5)
    assert await drain(timeout=10) == 0
    assert "composer_record_finish" in port.calls
    assert port.calls.count("composer_record_steps") >= 2  # the step write and the finish re-send


async def test_resend_after_a_sync_another_recorder_ran():
    port = ScriptedPort(unknown=("gap_calculator",))
    shared = RegistrySync(port=port)
    d, plan, result = _models()
    recorders = [
        CompositionRecorder(cid, {"composition_id": cid, "query_text": "q"}, port=port, sync=shared)
        for cid in ("first", "second")
    ]
    for recorder in recorders:
        recorder.planned(plan, latency_ms=1, plan_source="llm")
        recorder.step(0, result)
    assert await drain(timeout=10) == 0
    sends = [
        params["p_seed"]["composition_id"]
        for name, params in port.payloads
        if name == "composer_record_steps"
    ]
    assert port.calls.count("sync_tool_registry") == 1
    assert (
        sends.count("first") == 2 and sends.count("second") == 2
    )  # both re-sent after the one sync


class _ShapelessFrame:
    columns = ["a"]

    @property
    def shape(self) -> Any:
        raise RuntimeError("no shape")


class _Unmeasurable(dict):
    def __len__(self) -> int:
        raise RuntimeError("no length")


def test_structure_value_is_total_for_malformed_values():
    assert structure_value(_ShapelessFrame(), allowlist=None) == {"type": "str", "len": None}
    assert structure_value(_Unmeasurable(a=1), allowlist=None) == {"type": "str", "len": None}


async def test_a_malformed_step_is_skipped_and_counted_while_the_rest_records(metrics_registry):
    port = ScriptedPort()
    recorder = _recorder(port)
    d = _decomposition(["CAUSAL"])
    good_step = _step("good", "gap_calculator", "sq_0", {})
    bad_step = _step("bad", "psi_calculator", "sq_0", {})
    plan = _plan(d, [good_step, bad_step], [["good", "bad"]])
    good = _result(good_step, outcome_class="succeeded", result={"gap": 0.1})
    bad = _result(bad_step, outcome_class="succeeded", result={"psi": 0.1}).model_copy(
        update={"started_at": "not a datetime"}
    )
    recorder.planned(plan, latency_ms=1, plan_source="llm")
    recorder.step(0, good)
    recorder.step(1, bad)
    recorder.finish(status="COMPLETED", outcome="success", total_latency_ms=5)
    assert await drain(timeout=10) == 0
    step_batches = [
        params["p_steps"] for name, params in port.payloads if name == "composer_record_steps"
    ]
    assert step_batches and all([s["step_number"] for s in batch] == [0] for batch in step_batches)
    assert "composer_record_finish" in port.calls
    assert _failures(metrics_registry, "composer_record_steps") >= 1.0


async def test_seed_is_projected_to_the_recorded_identity_fields():
    port = ScriptedPort()
    seed = {
        "composition_id": "comp_seed",
        "query_text": "q" * 2000,
        "session_id": "sess-1",
        "user_id": 7,
        "entry_point": "somewhere_else",
        "brand": "Kisqali",
        "region": None,
        "audit_workflow_id": "not-a-uuid",
        "is_synthetic": "yes",
        SENTINEL: SENTINEL,
        "estimation_data": pd.DataFrame({SENTINEL: [1]}),
    }
    recorder = CompositionRecorder("comp_seed", seed, port=port, sync=RegistrySync(port=port))
    recorder.start()
    recorder.finish(status="COMPLETED", outcome="success", total_latency_ms=1)
    assert await drain(timeout=5) == 0
    seeds = [params["p_seed"] for _, params in port.payloads if "p_seed" in params]
    assert len(seeds) == 2
    assert all(
        s
        == {
            "composition_id": "comp_seed",
            "query_text": "q" * 1000,
            "session_id": "sess-1",
            "user_id": None,
            "entry_point": None,
            "brand": "Kisqali",
            "region": None,
            "audit_workflow_id": None,
            "is_synthetic": False,
        }
        for s in seeds
    ), seeds
    assert SENTINEL not in json.dumps([params for _, params in port.payloads], default=str)


async def test_seed_keeps_valid_identity_values():
    port = ScriptedPort()
    audit = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    seed = {
        "composition_id": "comp_ok",
        "query_text": "which regions?",
        "session_id": "s",
        "user_id": "u",
        "entry_point": "orchestrator_agent",
        "brand": "Remibrutinib",
        "region": "Northeast",
        "audit_workflow_id": audit,
        "is_synthetic": True,
    }
    recorder = CompositionRecorder("comp_ok", seed, port=port, sync=RegistrySync(port=port))
    recorder.start()
    assert await drain(timeout=5) == 0
    assert port.payloads[0][1]["p_seed"] == seed


async def test_heartbeat_stops_when_the_episode_is_already_terminal():
    port = ScriptedPort(heartbeat=False)  # another finish already closed the episode
    recorder = _recorder(port, heartbeat_s=0.05)
    recorder.start()
    await asyncio.sleep(0.4)
    assert recorder.heartbeat_stopped
    assert port.calls.count("composer_record_heartbeat") == 1


async def test_heartbeat_ends_at_its_lifetime_cap_without_a_finish():
    port = ScriptedPort()
    recorder = _recorder(port, heartbeat_s=0.02, heartbeat_max_s=0.15)
    recorder.start()
    await asyncio.sleep(0.5)
    assert recorder.heartbeat_stopped
    beats = port.calls.count("composer_record_heartbeat")
    assert 1 <= beats <= 10


async def test_shutdown_drain_cancels_heartbeats():
    port = ScriptedPort()
    recorder = _recorder(port, heartbeat_s=0.05)
    recorder.start()
    await asyncio.sleep(0.1)
    assert not recorder.heartbeat_stopped
    assert await drain(timeout=2, cancel_heartbeats=True) == 0
    assert recorder.heartbeat_stopped


# ---------------------------------------------------------------------------
# Iteration 2: malformed seeds, timeouts that cool down, heartbeats bound to their owner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed",
    [
        {"composition_id": "comp_malformed", "entry_point": []},
        {"composition_id": "comp_malformed", "entry_point": {}},
        {
            "composition_id": ["x"],
            "brand": ["x"],
            "region": {"a": 1},
            "audit_workflow_id": {"a": 1},
        },
        None,
        "not a mapping",
    ],
)
async def test_a_malformed_seed_never_raises_and_records_the_identity(seed):
    port = ScriptedPort()
    recorder = CompositionRecorder("comp_malformed", seed, port=port, sync=RegistrySync(port=port))
    recorder.start()
    assert await drain(timeout=5) == 0
    (sent,) = [
        params["p_seed"] for name, params in port.payloads if name == "composer_record_start"
    ]
    assert sent["composition_id"] == "comp_malformed"
    assert (
        sent["entry_point"] is None and sent["brand"] is None and sent["audit_workflow_id"] is None
    )


async def test_a_timed_out_catalog_fetch_cools_down_for_the_rest_of_the_chain():
    port = ScriptedPort(hang={"composer_public_column_names"})
    shared = RegistrySync(port=port)
    recorder = CompositionRecorder(
        "comp_cool", {"composition_id": "comp_cool"}, port=port, sync=shared, write_timeout_s=0.2
    )
    d, plan, result = _models()
    started = time.perf_counter()
    recorder.planned(plan, latency_ms=1, plan_source="llm")
    recorder.step(0, result)
    recorder.finish(status="COMPLETED", outcome="success", total_latency_ms=5)
    assert await drain(timeout=10) == 0
    assert port.calls.count("composer_public_column_names") == 1  # timed out once, then cooled down
    assert time.perf_counter() - started < 1.5


async def test_a_timed_out_refresh_keeps_the_cached_allowlist():
    port = ScriptedPort()
    shared = RegistrySync(port=port, allowlist_ttl_s=0.1)
    first = await shared.column_allowlist(timeout=0.2)
    assert first == frozenset({"brand", "region"})
    await asyncio.sleep(0.15)  # expired
    port.hang = {"composer_public_column_names"}
    assert await shared.column_allowlist(timeout=0.2) is first
    assert await shared.column_allowlist(timeout=0.2) is first  # cooling down: no new attempt
    assert port.calls.count("composer_public_column_names") == 2


async def test_a_timed_out_sync_cools_down_for_every_recorder_sharing_it():
    port = ScriptedPort(hang={"sync_tool_registry"}, unknown=("gap_calculator",))
    shared = RegistrySync(port=port)
    d, plan, result = _models()
    # One after the other, so the shared lock is free for the second: only the cooldown can stop
    # it from waiting on the hung sync again.
    for cid in ("first", "second"):
        recorder = CompositionRecorder(
            cid,
            {"composition_id": cid},
            port=port,
            sync=shared,
            write_timeout_s=0.1,
            sync_timeout_s=0.3,
        )
        recorder.planned(plan, latency_ms=1, plan_source="llm")
        recorder.step(0, result)
        assert await drain(timeout=10) == 0
    assert port.calls.count("sync_tool_registry") == 1
    assert shared.synced is False


@pytest.mark.parametrize("ending", ["returns", "raises", "is cancelled"])
async def test_heartbeat_stops_when_its_owning_task_ends_without_finish(ending):
    port = ScriptedPort()
    started = asyncio.Event()
    holder: Dict[str, CompositionRecorder] = {}

    async def owner() -> None:
        recorder = _recorder(port, heartbeat_s=0.05)
        recorder.start()
        holder["recorder"] = recorder
        started.set()
        if ending == "raises":
            raise RuntimeError("the composition crashed before its finish")
        if ending == "is cancelled":
            await asyncio.sleep(30)

    task = asyncio.create_task(owner())
    await started.wait()
    if ending == "is cancelled":
        task.cancel()
    with contextlib.suppress(BaseException):
        await task
    await asyncio.sleep(0.3)  # the loop stays alive well past several heartbeat periods
    assert holder["recorder"].heartbeat_stopped
