"""#883 PR B unit tests: ExperimentDesignerAgent must WIRE contribute_to_memory.

The hooks existed since the 4-memory rollout but had zero callers (and the
agent_activities payload was schema-broken — both bugs masked each other).
These tests pin the wiring contract; the faithful real-DB proofs live in
tests/integration/test_agent_memory_wiring_883b.py (agent path) and
tests/integration/test_agent_activities_realign_883b.py (hook payloads):

* enable_memory=True (default) -> contribute_to_memory called once post-run
  (async arun AND sync run — the dispatcher invokes sync run() in an executor
  thread, where the no-running-loop guard admits asyncio.run);
* the agent mints a session id per run (ExperimentDesignerInput has no
  session concept) and exposes it as last_memory_session_id;
* enable_memory=False          -> no call attempted;
* 046-trap posture             -> a RAISING contribution never changes the
  run's output or raises to the caller;
* failed design                -> RuntimeError raised, nothing contributed.

The recorder/raiser patches re-patch the conftest autouse offline guard —
same attribute, test-local behavior.
"""

from typing import Any, Dict

import pytest

from src.agents.experiment_designer.agent import (
    ExperimentDesignerAgent,
    ExperimentDesignerInput,
)

_AGENT_ATTR = "src.agents.experiment_designer.agent.contribute_to_memory"

_FINAL_STATE: Dict[str, Any] = {
    "status": "completed",
    "design_type": "RCT",
    "design_rationale": "randomization feasible",
    "brand": "remibrutinib",
    "business_question": "Does rep frequency lift TRx?",
    "constraints": {},
    "validity_threats": [],
    "treatments": [],
    "outcomes": [],
    "errors": [],
}


class _StubGraph:
    """Deterministic graph double: _create_output tolerates this state."""

    def __init__(self, final_state: Dict[str, Any]):
        self.final_state = final_state

    async def ainvoke(self, _state):
        return dict(self.final_state)

    def invoke(self, _state):
        return dict(self.final_state)


def _make_agent(**kwargs: Any) -> ExperimentDesignerAgent:
    kwargs.setdefault("enable_mlflow", False)
    agent = ExperimentDesignerAgent(**kwargs)
    agent.graph = _StubGraph(_FINAL_STATE)
    return agent


_INPUT = ExperimentDesignerInput(
    business_question="Does increasing rep visit frequency lift Remibrutinib TRx?",
)


@pytest.fixture()
def recorder(monkeypatch):
    calls = []

    async def _record(result, state, session_id, brand=None):
        calls.append({"result": result, "state": state, "session_id": session_id, "brand": brand})
        return {"working": 0, "episodic": 0}

    monkeypatch.setattr(_AGENT_ATTR, _record)
    return calls


class TestMemoryWiringAsync:
    @pytest.mark.asyncio
    async def test_arun_contributes_once_with_output_state_brand(self, recorder):
        agent = _make_agent()
        output = await agent.arun(_INPUT)

        assert len(recorder) == 1, "arun must contribute to memory exactly once"
        call = recorder[0]
        assert call["result"]["design_type"] == output.design_type
        assert call["state"]["status"] == "completed"
        assert call["brand"] == "remibrutinib"
        # The agent mints and exposes the memory session id.
        assert call["session_id"] == agent.last_memory_session_id
        assert agent.last_memory_session_id

    @pytest.mark.asyncio
    async def test_default_agent_has_memory_enabled(self):
        assert _make_agent().enable_memory is True

    @pytest.mark.asyncio
    async def test_failed_design_contributes_nothing(self, recorder):
        agent = _make_agent()
        agent.graph = _StubGraph(
            {**_FINAL_STATE, "status": "failed", "errors": [{"error": "boom"}]}
        )

        with pytest.raises(RuntimeError, match="Experiment design failed"):
            await agent.arun(_INPUT)

        assert recorder == [], "a failed design must not be stored"


class TestMemoryWiringSync:
    def test_sync_run_contributes_when_no_event_loop(self, recorder):
        """The dispatcher executes sync run() via run_in_executor (no running
        loop in that thread) — the contribution must fire there too."""
        agent = _make_agent()
        output = agent.run(_INPUT)

        assert output.design_type == "RCT"
        assert len(recorder) == 1
        assert recorder[0]["session_id"] == agent.last_memory_session_id


class TestMemoryWiringDisabled:
    @pytest.mark.asyncio
    async def test_arun_skips_memory_when_disabled(self, recorder):
        agent = _make_agent(enable_memory=False)
        output = await agent.arun(_INPUT)

        assert recorder == []
        assert output.design_type == "RCT"
        assert agent.last_memory_session_id is None


class TestMemoryFailureNonBlocking:
    """046-trap posture: a memory failure can never poison the run."""

    @pytest.mark.asyncio
    async def test_raising_contribution_does_not_change_output(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise RuntimeError("fabricated memory failure (046-trap probe)")

        monkeypatch.setattr(_AGENT_ATTR, _boom)

        agent = _make_agent()
        output = await agent.arun(_INPUT)

        assert output.design_type == "RCT"
        assert all("046-trap probe" not in str(e) for e in output.errors)

    def test_sync_raising_contribution_does_not_change_output(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise RuntimeError("fabricated memory failure (046-trap probe)")

        monkeypatch.setattr(_AGENT_ATTR, _boom)

        agent = _make_agent()
        output = agent.run(_INPUT)

        assert output.design_type == "RCT"
        assert all("046-trap probe" not in str(e) for e in output.errors)
