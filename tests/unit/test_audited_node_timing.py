"""Unit tests for the shared ``audited_node`` per-node timing wrapper.

Problem B (telemetry): the analytics latency panel reads ``duration_ms`` from
``audit_chain_entries``. Only ``causal_impact`` (via its inline ``traced_node``)
ever wrote a timed entry; the other ~11 agent graphs only emitted a genesis
``workflow_start`` entry (no ``duration_ms``) via ``create_workflow_initializer``.
So once those agents ran, ``/analytics/summary`` averaged an empty latency list
and returned ``avg_latency_ms = 0.0`` — a fake "0ms".

``audited_node`` is the shared, node-agnostic fix: it wraps ANY LangGraph node
callable, MEASURES real wall-clock duration, runs the node, and records a timed
``add_entry`` against the workflow's audit chain. These tests pin:

  1. A real ``duration_ms`` (>= 0, integer) is recorded for a wrapped node.
  2. The recorded ``action_type`` / ``agent_name`` / ``agent_tier`` are honest.
  3. When no workflow_id / no audit service is present, the node still runs and
     NO entry is fabricated (audit is best-effort, never blocks execution).
  4. A node raising still records a timed ``*_error`` entry and re-raises.
  5. ``validation_passed`` is only populated from a real result key, never faked.

The timing is REAL (a node that sleeps ~20ms records duration_ms >= ~15ms).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from uuid import uuid4

import pytest

from src.agents.base.audit_chain_mixin import (
    audited_node,
    set_audit_chain_service,
)
from src.utils.audit_chain import AgentTier


class _RecordingService:
    """Minimal faithful stand-in for AuditChainService.

    Records the exact kwargs passed to ``add_entry`` so we can assert on the
    REAL measured ``duration_ms`` rather than a mocked constant. ``start_workflow``
    is included so a test can mint a genuine workflow_id the way
    ``create_workflow_initializer`` does in production.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self._workflows: set = set()

    def start_workflow(self, **kwargs: Any) -> Any:
        wid = uuid4()
        self._workflows.add(wid)

        class _Entry:
            workflow_id = wid

        return _Entry()

    def add_entry(self, **kwargs: Any) -> Any:
        self.entries.append(kwargs)
        return object()


@pytest.fixture
def recording_service() -> _RecordingService:
    svc = _RecordingService()
    set_audit_chain_service(svc)  # type: ignore[arg-type]
    yield svc
    set_audit_chain_service(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_audited_node_records_real_duration(recording_service: _RecordingService) -> None:
    """A wrapped node that sleeps ~20ms records a real measured duration_ms."""

    async def slow_node(state: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.02)
        return {"status": "ok"}

    wrapped = audited_node(
        slow_node,
        agent_name="gap_analyzer",
        agent_tier=AgentTier.CAUSAL_ANALYTICS,
        node_name="gap_detector",
    )

    workflow_id = uuid4()
    result = await wrapped({"audit_workflow_id": workflow_id})

    assert result == {"status": "ok"}
    assert len(recording_service.entries) == 1
    entry = recording_service.entries[0]
    assert entry["agent_name"] == "gap_analyzer"
    assert entry["agent_tier"] == AgentTier.CAUSAL_ANALYTICS
    assert entry["action_type"] == "gap_detector"
    assert entry["workflow_id"] == workflow_id
    # REAL measurement: slept 20ms -> at least ~15ms recorded (allow scheduler jitter).
    assert isinstance(entry["duration_ms"], int)
    assert entry["duration_ms"] >= 15


@pytest.mark.asyncio
async def test_audited_node_noop_without_workflow_id(
    recording_service: _RecordingService,
) -> None:
    """No workflow_id in state -> node runs, but NO audit entry is fabricated."""

    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok"}

    wrapped = audited_node(
        node, agent_name="explainer", agent_tier=AgentTier.SELF_IMPROVEMENT, node_name="reason"
    )
    result = await wrapped({})  # no audit_workflow_id

    assert result == {"status": "ok"}
    assert recording_service.entries == []


@pytest.mark.asyncio
async def test_audited_node_noop_without_service() -> None:
    """No audit service installed -> node runs, no crash, no entry."""
    set_audit_chain_service(None)  # type: ignore[arg-type]

    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": 42}

    wrapped = audited_node(
        node, agent_name="drift_monitor", agent_tier=AgentTier.MONITORING, node_name="data_drift"
    )
    result = await wrapped({"audit_workflow_id": uuid4()})
    assert result == {"value": 42}


@pytest.mark.asyncio
async def test_audited_node_records_error_entry_and_reraises(
    recording_service: _RecordingService,
) -> None:
    """A raising node records a timed *_error entry then re-raises the exception."""

    async def boom(state: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.005)
        raise ValueError("kaboom")

    wrapped = audited_node(
        boom,
        agent_name="resource_optimizer",
        agent_tier=AgentTier.ML_PREDICTIONS,
        node_name="optimize",
    )

    with pytest.raises(ValueError, match="kaboom"):
        await wrapped({"audit_workflow_id": uuid4()})

    assert len(recording_service.entries) == 1
    entry = recording_service.entries[0]
    assert entry["action_type"] == "optimize_error"
    assert entry["validation_passed"] is False
    assert isinstance(entry["duration_ms"], int)
    assert entry["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_audited_node_validation_passed_only_from_real_key(
    recording_service: _RecordingService,
) -> None:
    """validation_passed is taken from a real result key, never invented."""

    async def node_with_validation(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"validation_passed": True, "status": "ok"}

    async def node_without(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok"}

    wrapped_with = audited_node(
        node_with_validation,
        agent_name="experiment_monitor",
        agent_tier=AgentTier.MONITORING,
        node_name="health_checker",
    )
    wrapped_without = audited_node(
        node_without,
        agent_name="experiment_monitor",
        agent_tier=AgentTier.MONITORING,
        node_name="srm_detector",
    )

    await wrapped_with({"audit_workflow_id": uuid4()})
    await wrapped_without({"audit_workflow_id": uuid4()})

    assert recording_service.entries[0]["validation_passed"] is True
    # No validation key -> None (UNMEASURED), not a fabricated default.
    assert recording_service.entries[1]["validation_passed"] is None


@pytest.mark.asyncio
async def test_audited_node_subms_node_records_at_least_one_ms(
    recording_service: _RecordingService,
) -> None:
    """A real sub-millisecond node records duration_ms >= 1, never 0.

    Recording 0 is indistinguishable downstream from "unmeasured" (analytics
    drops falsy duration_ms; the UI treats 0 as not-measured), so a fast-but-real
    node would silently vanish from the latency panel. The node DID run, so the
    floored 1ms is honest quantization, not a fabricated timing.
    """

    async def instant(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok"}

    wrapped = audited_node(
        instant, agent_name="orchestrator", agent_tier=AgentTier.COORDINATION, node_name="classify"
    )
    await wrapped({"audit_workflow_id": uuid4()})

    assert len(recording_service.entries) == 1
    assert recording_service.entries[0]["duration_ms"] >= 1


def test_audited_node_preserves_sync_callable(recording_service: _RecordingService) -> None:
    """A sync node callable is wrapped into an async node that still records timing.

    LangGraph supports sync and async nodes; experiment_designer feeds sync nodes
    through ``wrap_async_node``. ``audited_node`` must accept either and produce an
    awaitable node so graph wiring is uniform.
    """

    def sync_node(state: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(0.005)
        return {"status": "done"}

    wrapped = audited_node(
        sync_node,
        agent_name="health_score",
        agent_tier=AgentTier.MONITORING,
        node_name="component",
    )

    result = asyncio.run(wrapped({"audit_workflow_id": uuid4()}))
    assert result == {"status": "done"}
    assert len(recording_service.entries) == 1
    assert recording_service.entries[0]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_audited_node_soft_failure_records_error_entry(
    recording_service: _RecordingService,
) -> None:
    """A node that catches its own failure and returns ``{node}_error`` (the
    fail-closed convention, e.g. causal_impact estimation) records the same
    ``<node>_error`` entry a raising node does — the only readable execution
    outcome in audit_chain_entries (output_data is hashed, 2026-09-06). The
    result is still returned unchanged; nothing raises."""

    async def fail_closed(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"optimize_error": "solver unavailable", "status": "failed"}

    wrapped = audited_node(
        fail_closed,
        agent_name="resource_optimizer",
        agent_tier=AgentTier.ML_PREDICTIONS,
        node_name="optimize",
    )

    result = await wrapped({"audit_workflow_id": uuid4()})

    assert result["optimize_error"] == "solver unavailable"
    assert len(recording_service.entries) == 1
    entry = recording_service.entries[0]
    assert entry["action_type"] == "optimize_error"
    assert entry["validation_passed"] is False
