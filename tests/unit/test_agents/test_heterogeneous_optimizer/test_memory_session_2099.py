"""#2099: HeterogeneousOptimizerAgent must not mint a session id.

``run`` computed ``input_data.get("session_id") or str(uuid.uuid4())`` and spent
the result on five consumers at once: the memory context read, the episodic
write, ``state["session_id"]``, the Opik trace's ``session_id`` and MLflow's
``query_id``. Only the last two need a value when the caller has no session, and
neither of them needs it to be a session -- so they get their own per-run id and
the memory path passes the caller's ``None`` straight through.

The riskiest downstream site is ``profile_generator``, which coerced the state
session with ``or ""``. With the mint gone that would have fed an empty string --
a second invented-identity shape -- into the DSPy signal, so it normalises to
``None`` here too.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest

from src.agents.heterogeneous_optimizer.agent import HeterogeneousOptimizerAgent
from src.agents.heterogeneous_optimizer.memory_hooks import (
    CATEAnalysisContext,
    HeterogeneousOptimizerMemoryHooks,
)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_SESSION = "eeba22e7-4d9d-49ea-977b-b9e9d1549c53"

_INPUT: Dict[str, Any] = {
    "query": "which HCP segments respond to rep visits",
    "treatment_var": "rep_visits",
    "outcome_var": "trx",
    "segment_vars": ["specialty"],
    "effect_modifiers": ["specialty"],
    "data_source": "synthetic",
}

_FINAL_STATE: Dict[str, Any] = {
    "overall_ate": 0.1,
    "heterogeneity_score": 0.2,
    "cate_by_segment": {},
    "high_responders": [],
    "low_responders": [],
    "policy_recommendations": [],
    "expected_total_lift": 0.0,
    "confidence": 0.7,
    "total_latency_ms": 1,
    "errors": [],
    "warnings": [],
    "status": "completed",
}


class _FakeHooks:
    def __init__(self) -> None:
        self.get_context_calls: List[Optional[str]] = []

    async def get_context(self, session_id: Optional[str], **_kw: Any) -> CATEAnalysisContext:
        self.get_context_calls.append(session_id)
        return CATEAnalysisContext(session_id=session_id)


class _FakeGraph:
    async def ainvoke(self, _state: Any, *_a: Any, **_kw: Any) -> Dict[str, Any]:
        return dict(_FINAL_STATE)


@pytest.fixture
def recorder(monkeypatch):
    """Record the session id the episodic writer receives."""
    calls: List[Dict[str, Any]] = []

    async def _contribute(**kwargs: Any) -> Dict[str, int]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        "src.agents.heterogeneous_optimizer.agent.contribute_to_memory", _contribute
    )
    return calls


def _agent() -> tuple[HeterogeneousOptimizerAgent, _FakeHooks]:
    agent = HeterogeneousOptimizerAgent(enable_memory=True)
    agent.graph = _FakeGraph()  # type: ignore[assignment]
    agent._get_mlflow_tracker = lambda: None  # type: ignore[assignment]
    agent._get_opik_tracer = lambda: None  # type: ignore[assignment]
    hooks = _FakeHooks()
    agent._memory_hooks = hooks  # type: ignore[assignment]
    return agent, hooks


@pytest.mark.asyncio
async def test_no_session_reaches_the_writer_as_none(recorder):
    """No session in -> None out. Never a minted uuid."""
    agent, hooks = _agent()

    await agent.run(dict(_INPUT))

    assert len(recorder) == 1
    got = recorder[0]["session_id"]
    assert not (isinstance(got, str) and _UUID_RE.match(got)), f"agent minted a session: {got!r}"
    assert got is None
    assert hooks.get_context_calls == [None]


@pytest.mark.asyncio
async def test_caller_session_is_passed_through_unchanged(recorder):
    """The change is scoped to the session-less path."""
    agent, hooks = _agent()

    await agent.run({**_INPUT, "session_id": _SESSION})

    assert recorder[0]["session_id"] == _SESSION
    assert hooks.get_context_calls == [_SESSION]


@pytest.mark.asyncio
async def test_initial_state_session_is_none_not_minted(recorder):
    """state["session_id"] feeds the DSPy signal; it must stay honest."""
    agent, _hooks = _agent()
    captured: Dict[str, Any] = {}

    class _CapturingGraph:
        async def ainvoke(self, state: Dict[str, Any], *_a: Any, **_kw: Any) -> Dict[str, Any]:
            captured.update(state)
            return dict(_FINAL_STATE)

    agent.graph = _CapturingGraph()  # type: ignore[assignment]

    await agent.run(dict(_INPUT))

    assert captured["session_id"] is None


@pytest.mark.asyncio
async def test_mlflow_and_opik_still_get_a_run_handle(recorder):
    """Both tracers need a non-empty id; neither needs it to be a session."""
    agent, _hooks = _agent()
    opened: Dict[str, Any] = {}

    class _Tracer:
        def trace_analysis(self, **kwargs: Any) -> Any:
            opened["opik"] = kwargs

            class _Cm:
                async def __aenter__(self) -> Any:
                    class _Ctx:
                        def __getattr__(self, _n: str) -> Any:
                            return lambda *a, **k: None

                    return _Ctx()

                async def __aexit__(self, *_e: Any) -> bool:
                    return False

            return _Cm()

    class _Tracker:
        def start_analysis_run(self, **kwargs: Any) -> Any:
            opened["mlflow"] = kwargs

            class _Cm:
                async def __aenter__(self) -> None:
                    return None

                async def __aexit__(self, *_e: Any) -> bool:
                    return False

            return _Cm()

        async def log_analysis_result(self, *_a: Any, **_kw: Any) -> None:
            return None

    agent._get_opik_tracer = lambda: _Tracer()  # type: ignore[assignment]
    agent._get_mlflow_tracker = lambda: _Tracker()  # type: ignore[assignment]

    await agent.run(dict(_INPUT))

    trace_session = opened["opik"]["session_id"]
    query_id = opened["mlflow"]["query_id"]
    assert isinstance(query_id, str) and query_id, f"MLflow lost its handle: {query_id!r}"
    assert isinstance(trace_session, str) and trace_session, f"Opik lost its id: {trace_session!r}"
    assert query_id == trace_session, "both tracers should correlate on the same run id"
    assert recorder[0]["session_id"] is None


@pytest.mark.asyncio
async def test_working_memory_read_is_skipped_without_a_session():
    """A session-less working-memory read would query under an invented key."""
    from unittest.mock import AsyncMock

    hooks = HeterogeneousOptimizerMemoryHooks()
    working_memory = AsyncMock()
    hooks._working_memory = working_memory

    assert await hooks._get_working_memory_context(None) == []
    working_memory.get_messages.assert_not_called()


def test_profile_generator_does_not_coerce_the_session_to_empty_string():
    """``or ""`` would swap one invented identity for another."""
    import inspect

    from src.agents.heterogeneous_optimizer.nodes import profile_generator

    source = inspect.getsource(profile_generator)
    assert 'state.get("session_id") or ""' not in source, (
        "the empty-string session coercion is back in profile_generator"
    )


@pytest.mark.asyncio
async def test_dspy_signal_records_a_none_session_not_an_empty_string():
    """The signal feeds learning_signals.session_id."""
    from src.agents.heterogeneous_optimizer.dspy_integration import (
        get_heterogeneous_optimizer_signal_collector,
    )

    signal = get_heterogeneous_optimizer_signal_collector().collect_optimization_signal(
        session_id=None,
        query="q",
        treatment_var="t",
        outcome_var="o",
        segment_vars_count=0,
        effect_modifiers_count=0,
    )

    assert signal.session_id is None
