"""#2099: ExperimentDesignerAgent must not mint a session id per run.

``_contribute_to_memory`` opened with ``session_id = str(uuid.uuid4())``. Its own
comment named the reason -- "ExperimentDesignerInput carries no session concept,
so the agent mints one per run" -- which is exactly the invented identity #2099
removes: the value reached ``agent_activities.input_data.session_id`` and keyed a
``experiment_designer:session:<id>`` working-memory entry no reader can ask for.

With no session concept in the input, the honest value is always ``None``:
``last_memory_session_id`` stays ``None``, the activity row records a NULL
session, and the session-keyed cache write is skipped rather than written under
an unreachable key. The design is still cached by question hash, which is the
key the reuse path actually reads.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest

from src.agents.experiment_designer.agent import (
    ExperimentDesignerAgent,
    ExperimentDesignerInput,
)
from src.agents.experiment_designer.memory_hooks import ExperimentDesignerMemoryHooks

_AGENT_ATTR = "src.agents.experiment_designer.agent.contribute_to_memory"
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

_FINAL_STATE: Dict[str, Any] = {
    "business_question": "Does increasing rep visit frequency lift Remibrutinib TRx?",
    "brand": "remibrutinib",
    "design_type": "RCT",
    "design_rationale": "",
    "randomization_unit": "individual",
    "randomization_method": "simple",
    "treatments": [],
    "outcomes": [],
    "power_analysis": {"required_sample_size": 100, "achieved_power": 0.8},
    "validity_threats": [],
    "overall_validity_score": 0.9,
    "duration_estimate_days": 30,
    "total_latency_ms": 5,
    "errors": [],
    "warnings": [],
    "status": "completed",
}


class _StubGraph:
    def __init__(self, final_state: Dict[str, Any]) -> None:
        self._final_state = final_state

    async def ainvoke(self, _state: Any, *_a: Any, **_kw: Any) -> Dict[str, Any]:
        return dict(self._final_state)

    def invoke(self, _state: Any, *_a: Any, **_kw: Any) -> Dict[str, Any]:
        return dict(self._final_state)


@pytest.fixture()
def recorder(monkeypatch):
    """Record the session id the memory writer receives."""
    calls: List[Dict[str, Any]] = []

    async def _record(result, state, session_id, brand=None):
        calls.append({"session_id": session_id, "brand": brand})
        return {"working": 0, "episodic": 0}

    monkeypatch.setattr(_AGENT_ATTR, _record)
    return calls


def _agent() -> ExperimentDesignerAgent:
    agent = ExperimentDesignerAgent(enable_mlflow=False)
    agent.graph = _StubGraph(_FINAL_STATE)
    return agent


_INPUT = ExperimentDesignerInput(
    business_question="Does increasing rep visit frequency lift Remibrutinib TRx?",
)


@pytest.mark.asyncio
async def test_arun_hands_the_writer_none_not_a_minted_uuid(recorder):
    """The input carries no session, so the writer must receive None."""
    agent = _agent()

    await agent.arun(_INPUT)

    assert len(recorder) == 1
    got = recorder[0]["session_id"]
    assert not (isinstance(got, str) and _UUID_RE.match(got)), f"agent minted a session id: {got!r}"
    assert got is None


@pytest.mark.asyncio
async def test_last_memory_session_id_stays_none_after_a_contribution(recorder):
    """The attribute reports the session that was stored -- there is none."""
    agent = _agent()

    await agent.arun(_INPUT)

    assert len(recorder) == 1
    assert agent.last_memory_session_id is None
    assert recorder[0]["session_id"] == agent.last_memory_session_id


def test_sync_run_hands_the_writer_none(recorder):
    """The dispatcher's sync path must not mint one either."""
    agent = _agent()

    agent.run(_INPUT)

    assert len(recorder) == 1
    assert recorder[0]["session_id"] is None
    assert agent.last_memory_session_id is None


@pytest.mark.asyncio
async def test_session_cache_write_is_skipped_without_a_session():
    """The session cache key embeds the id; None would write an unreachable key."""
    hooks = ExperimentDesignerMemoryHooks()
    written: List[str] = []

    async def _set_json(key: str, _value: Any, _ttl: int) -> None:
        written.append(key)

    hooks._working_memory = object()  # truthy: the write path is reachable
    hooks._wm_set_json = _set_json  # type: ignore[assignment]

    assert await hooks.cache_experiment_design(
        session_id=None,
        result={"design_type": "RCT"},
        business_question="Does rep frequency lift TRx?",
    )

    assert not any("experiment_designer:session:" in k for k in written), written
    assert any("experiment_designer:question:" in k for k in written), written


@pytest.mark.asyncio
async def test_caller_session_id_still_keys_the_session_cache():
    """The change is scoped to the session-less path."""
    hooks = ExperimentDesignerMemoryHooks()
    written: List[str] = []

    async def _set_json(key: str, _value: Any, _ttl: int) -> None:
        written.append(key)

    hooks._working_memory = object()
    hooks._wm_set_json = _set_json  # type: ignore[assignment]

    session_id: Optional[str] = "eeba22e7-4d9d-49ea-977b-b9e9d1549c53"
    assert await hooks.cache_experiment_design(
        session_id=session_id,
        result={"design_type": "RCT"},
        business_question="Does rep frequency lift TRx?",
    )

    assert f"experiment_designer:session:{session_id}" in written
