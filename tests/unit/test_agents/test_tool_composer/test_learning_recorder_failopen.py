"""Recording is fail-open and off the user's path (spec §5.4). No database: CI-safe.

Nothing ``compose()`` calls on the recorder awaits I/O: every method is a synchronous enqueue
onto a background chain. So an unreachable database (connection refused) or a hanging one adds
no latency to a composition and never raises into it. A write that fails its one retry is logged
once with the RPC, the composition id and the error class, and counted in
``composer_record_failures_total{rpc}``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import pytest

from src.agents.tool_composer import composer as _composer  # noqa: F401 - registers every tool
from src.agents.tool_composer import learning_recorder
from src.agents.tool_composer.learning_recorder import CompositionRecorder, drain
from src.agents.tool_composer.registry_sync import RegistrySync
from tests.unit.test_agents.test_tool_composer.test_learning_recorder_serializer import (
    _decomposition,
    _plan,
    _result,
    _step,
)


class RefusedPort:
    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        raise ConnectionRefusedError(111, "Connection refused")


class HangingPort:
    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        await asyncio.sleep(10)


@pytest.fixture(autouse=True)
async def _no_leftover_tasks():
    yield
    for task in list(learning_recorder._pending):
        task.cancel()
    await asyncio.sleep(0)


def _recorder(port: Any, **kw: Any) -> CompositionRecorder:
    return CompositionRecorder(
        "comp_failopen",
        {"composition_id": "comp_failopen", "query_text": "q", "is_synthetic": True},
        port=port,
        sync=RegistrySync(port=port),
        **kw,
    )


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


async def test_failed_write_is_logged_and_counted(caplog):
    import src.api.routes.metrics as metrics

    original = (metrics._metrics_initialized, metrics._metrics_registry)
    metrics._metrics_initialized = False
    metrics._metrics_registry = None
    try:
        metrics._init_metrics()
        recorder = _recorder(RefusedPort(), retry_delay_s=0.01)
        with caplog.at_level(logging.WARNING, logger=learning_recorder.__name__):
            recorder.start()
            assert await drain(timeout=5) == 0
        sample = metrics._metrics_registry.get_sample_value(
            "composer_record_failures_total", {"rpc": "composer_record_start"}
        )
        assert sample == 1.0
        messages = [r.getMessage() for r in caplog.records if r.name == learning_recorder.__name__]
        assert any(
            "composer_record_start" in m and "comp_failopen" in m and "ConnectionRefusedError" in m
            for m in messages
        ), messages
        assert not any("Connection refused" in m for m in messages)  # the class, not the text
    finally:
        metrics._metrics_initialized, metrics._metrics_registry = original


def test_failure_counter_helper_is_a_no_op_before_metrics_init():
    import src.api.routes.metrics as metrics

    original = (metrics._metrics_initialized, metrics._metrics_registry)
    metrics._metrics_initialized = False
    try:
        metrics.inc_composer_record_failure("composer_record_start")
    finally:
        metrics._metrics_initialized, metrics._metrics_registry = original
