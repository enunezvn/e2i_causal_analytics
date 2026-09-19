"""#2177: DriftMonitorAgent.run() must hand its result to the memory hooks.

``contribute_to_memory`` (working cache + episodic record + semantic drift
patterns) existed since 1253622c1 but had no caller, so none of drift_monitor's
memory writes ever ran. The real writers are exercised against a real FalkorDB in
``tests/integration/test_kg_writer_integrity.py``; this test pins the WIRING —
the one seam that was missing — by recording what ``run()`` passes to it. The
detection graph itself runs for real on the suite's mock data connector.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import src.agents.drift_monitor.agent as agent_module
from src.agents.drift_monitor import DriftMonitorAgent, DriftMonitorInput


@pytest.fixture
def recorded(monkeypatch) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    async def _record(result, state, session_id):
        calls.append({"result": result, "state": state, "session_id": session_id})
        return {"working": 0, "episodic": 0, "semantic": 0}

    monkeypatch.setattr(agent_module, "contribute_to_memory", _record)
    return calls


@pytest.mark.asyncio
async def test_run_contributes_the_detection_result_to_memory(recorded):
    output = await DriftMonitorAgent().run(
        DriftMonitorInput(
            query="Check drift", features_to_monitor=["feature1", "feature2"], model_id="m1"
        )
    )

    assert len(recorded) == 1
    call = recorded[0]
    assert call["state"]["model_id"] == "m1"
    assert call["state"]["features_to_monitor"] == ["feature1", "feature2"]
    assert call["result"]["overall_drift_score"] == output.overall_drift_score
    assert call["result"]["features_with_drift"] == output.features_with_drift
    # No conversation exists for a drift run; the id is a per-run handle (#2099).
    assert call["session_id"].startswith("drift_run_")


@pytest.mark.asyncio
async def test_memory_can_be_disabled(recorded):
    await DriftMonitorAgent(enable_memory=False).run(
        DriftMonitorInput(query="Check drift", features_to_monitor=["feature1"])
    )

    assert recorded == []


@pytest.mark.asyncio
async def test_a_memory_failure_never_fails_the_run(monkeypatch):
    async def _boom(result, state, session_id):
        raise RuntimeError("memory backend down")

    monkeypatch.setattr(agent_module, "contribute_to_memory", _boom)

    output = await DriftMonitorAgent().run(
        DriftMonitorInput(query="Check drift", features_to_monitor=["feature1"])
    )

    assert output.overall_drift_score >= 0.0
