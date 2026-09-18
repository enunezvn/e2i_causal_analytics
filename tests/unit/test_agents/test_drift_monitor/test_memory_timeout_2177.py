"""#2177 codex R1: a slow memory backend must not hold a drift run hostage.

Co-drift writes are O(n²) in drifting features, so the contribution is bounded:
``run()`` waits at most ``memory_timeout_seconds`` and then returns its result.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import src.agents.drift_monitor.agent as agent_module
from src.agents.drift_monitor import DriftMonitorAgent, DriftMonitorInput


@pytest.mark.asyncio
async def test_run_returns_when_memory_contribution_exceeds_its_budget(monkeypatch):
    async def _slow(result, state, session_id):
        await asyncio.sleep(60)

    monkeypatch.setattr(agent_module, "contribute_to_memory", _slow)
    agent = DriftMonitorAgent()
    agent.memory_timeout_seconds = 0.5

    started = time.monotonic()
    output = await agent.run(
        DriftMonitorInput(query="Check drift", features_to_monitor=["feature1"])
    )

    assert time.monotonic() - started < 20
    assert output.overall_drift_score >= 0.0


class _BlockingGraph:
    """A synchronous graph handle whose every query blocks — a stalled FalkorDB."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls: list[dict] = []

    def query(self, cypher, params=None, timeout=None):
        self.calls.append({"timeout": timeout})
        time.sleep(self.seconds)


class _Semantic:
    def __init__(self, graph) -> None:
        self.graph = graph


@pytest.mark.asyncio
async def test_drift_writes_are_query_bounded_and_stop_at_the_write_budget(monkeypatch):
    import src.agents.drift_monitor.memory_hooks as hooks_module

    monkeypatch.setattr(hooks_module, "_GRAPH_WRITE_BUDGET_SECONDS", 0.3)
    graph = _BlockingGraph(seconds=0.1)
    hooks = hooks_module.DriftMonitorMemoryHooks()
    hooks._semantic_memory = _Semantic(graph)
    features = [f"f{i}" for i in range(30)]  # 29 co-drift writes for f0 alone
    result = {
        "features_with_drift": features,
        "data_drift_results": [{"feature": "f0", "drift_detected": True}],
    }

    started = time.monotonic()
    await hooks.store_drift_pattern("f0", "data", "high", result, {"model_id": "m"})

    # Every query carries a server-side timeout, and the thread stops issuing
    # queries once the write budget is spent instead of running all 33.
    assert all(c["timeout"] == hooks_module._GRAPH_QUERY_TIMEOUT_MS for c in graph.calls)
    assert len(graph.calls) < 10
    assert time.monotonic() - started < 2


@pytest.mark.asyncio
async def test_memory_wait_fits_inside_the_remaining_sla(monkeypatch):
    waited: list[float] = []
    real_wait_for = asyncio.wait_for

    async def _spy_wait_for(aw, timeout):
        waited.append(timeout)
        return await real_wait_for(aw, timeout)

    async def _noop(result, state, session_id):
        return {}

    monkeypatch.setattr(agent_module, "contribute_to_memory", _noop)
    monkeypatch.setattr(agent_module.asyncio, "wait_for", _spy_wait_for)
    agent = DriftMonitorAgent()

    await agent.run(DriftMonitorInput(query="Check drift", features_to_monitor=["feature1"]))

    assert len(waited) == 1
    assert 0 < waited[0] <= agent.sla_seconds


@pytest.mark.asyncio
async def test_exhausted_sla_still_records_with_the_floor(monkeypatch):
    """A run that already used its SLA still records its drift result, bounded by
    the 0.5 s floor — a deliberate trade: slow runs are the many-feature runs whose
    drift record matters most."""
    waited: list[float] = []
    real_wait_for = asyncio.wait_for

    async def _spy_wait_for(aw, timeout):
        waited.append(timeout)
        return await real_wait_for(aw, timeout)

    async def _noop(result, state, session_id):
        return {}

    monkeypatch.setattr(agent_module, "contribute_to_memory", _noop)
    monkeypatch.setattr(agent_module.asyncio, "wait_for", _spy_wait_for)
    agent = DriftMonitorAgent()
    agent.sla_seconds = 0  # detection alone exhausts it

    await agent.run(DriftMonitorInput(query="Check drift", features_to_monitor=["feature1"]))

    assert waited == [0.5]
