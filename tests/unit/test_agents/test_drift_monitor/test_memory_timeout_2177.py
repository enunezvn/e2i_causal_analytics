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
