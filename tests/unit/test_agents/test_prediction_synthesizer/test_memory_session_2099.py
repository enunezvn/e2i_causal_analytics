"""#2099: PredictionSynthesizerAgent must not mint a session id.

``synthesize`` minted ``str(uuid.uuid4())`` for a session-less call and then used
the one value for three unrelated jobs: the memory session written to
``episodic_memories.session_id``, the DSPy signal's ``learning_signals.session_id``
and the Opik trace's ``synthesis_id``. Only the last of those needs a value when
no caller session exists, so the two identity columns now record an honest NULL
and the trace gets its own per-run id.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent
from src.agents.prediction_synthesizer.memory_hooks import (
    PredictionMemoryContext,
    PredictionSynthesizerMemoryHooks,
)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_SESSION = "eeba22e7-4d9d-49ea-977b-b9e9d1549c53"


class _FakeHooks:
    """Records the session id ``synthesize`` hands to the context read."""

    def __init__(self) -> None:
        self.get_context_calls: List[Optional[str]] = []

    async def get_context(
        self, session_id: Optional[str], **_kwargs: Any
    ) -> PredictionMemoryContext:
        self.get_context_calls.append(session_id)
        return PredictionMemoryContext(session_id=session_id)


class _FakeTraceCtx:
    """Swallows every ``log_*`` call the traced branch makes."""

    def __getattr__(self, _name: str) -> Any:
        return MagicMock()


class _FakeTracer:
    """Records the kwargs ``trace_synthesis`` is opened with."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def trace_synthesis(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class _Cm:
            async def __aenter__(self) -> _FakeTraceCtx:
                return _FakeTraceCtx()

            async def __aexit__(self, *_exc: Any) -> bool:
                return False

        return _Cm()


@pytest.fixture
def mock_graph():
    """A graph returning a completed, single-model synthesis state."""
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "entity_id": "hcp_1",
        "entity_type": "hcp",
        "prediction_target": "churn",
        "time_horizon": "30d",
        "individual_predictions": [],
        "models_succeeded": 1,
        "models_failed": 0,
        "ensemble_prediction": {
            "point_estimate": 0.4,
            "prediction_interval_lower": 0.3,
            "prediction_interval_upper": 0.5,
            "confidence": 0.9,
            "ensemble_method": "weighted",
            "model_agreement": 0.8,
        },
        "prediction_summary": "",
        "prediction_context": None,
        "total_latency_ms": 1,
        "timestamp": "2026-09-14T00:00:00+00:00",
        "errors": [],
        "warnings": [],
        "status": "completed",
    }
    return graph


@pytest.fixture
def recorder(monkeypatch):
    """Record the session id the memory writer and the DSPy emitter receive."""
    calls: Dict[str, List[Optional[str]]] = {"memory": [], "dspy": []}

    async def _contribute(
        result: Dict[str, Any],
        state: Dict[str, Any],
        memory_hooks: Any = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, int]:
        calls["memory"].append(session_id)
        return {}

    async def _emit(
        session_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        min_reward_threshold: float = 0.5,
    ) -> None:
        calls["dspy"].append(session_id)
        return None

    monkeypatch.setattr(
        "src.agents.prediction_synthesizer.memory_hooks.contribute_to_memory", _contribute
    )
    monkeypatch.setattr(
        "src.agents.prediction_synthesizer.dspy_integration.collect_and_emit_signal", _emit
    )
    return calls


def _agent(mock_graph) -> tuple[PredictionSynthesizerAgent, _FakeHooks]:
    agent = PredictionSynthesizerAgent(enable_opik=False, enable_memory=True, enable_dspy=True)
    agent._full_graph = mock_graph
    agent._simple_graph = mock_graph
    hooks = _FakeHooks()
    agent._memory_hooks = hooks  # type: ignore[assignment]
    return agent, hooks


async def _synthesize(agent, **kwargs):
    return await agent.synthesize(
        entity_id="hcp_1",
        prediction_target="churn",
        features={"x": 1.0},
        **kwargs,
    )


@pytest.mark.asyncio
async def test_no_session_id_reaches_either_writer_as_none(mock_graph, recorder):
    """No session in -> None to the episodic writer and to the learning signal."""
    agent, hooks = _agent(mock_graph)

    await _synthesize(agent)

    assert recorder["memory"] == [None], recorder["memory"]
    assert recorder["dspy"] == [None], recorder["dspy"]
    assert hooks.get_context_calls == [None]
    for got in recorder["memory"] + recorder["dspy"]:
        assert not (isinstance(got, str) and _UUID_RE.match(got)), f"minted a session: {got!r}"


@pytest.mark.asyncio
async def test_caller_session_id_is_passed_through_unchanged(mock_graph, recorder):
    """The change is scoped to the session-less path."""
    agent, hooks = _agent(mock_graph)

    await _synthesize(agent, session_id=_SESSION)

    assert recorder["memory"] == [_SESSION]
    assert recorder["dspy"] == [_SESSION]
    assert hooks.get_context_calls == [_SESSION]


@pytest.mark.asyncio
async def test_opik_trace_gets_its_own_run_id_not_the_session(mock_graph, recorder):
    """The trace still needs a handle; it must not be an invented session."""
    agent, _hooks = _agent(mock_graph)
    agent.enable_opik = True
    tracer = _FakeTracer()
    agent._tracer = tracer  # type: ignore[assignment]

    await _synthesize(agent)

    assert len(tracer.calls) == 1
    synthesis_id = tracer.calls[0]["synthesis_id"]
    assert isinstance(synthesis_id, str) and synthesis_id, (
        f"trace lost its handle: {synthesis_id!r}"
    )
    assert recorder["memory"] == [None]
    assert synthesis_id not in recorder["memory"]


@pytest.mark.asyncio
async def test_working_memory_read_is_skipped_without_a_session():
    """A session-less working-memory read would query under an invented key."""
    hooks = PredictionSynthesizerMemoryHooks()
    working_memory = AsyncMock()
    hooks._working_memory = working_memory

    assert await hooks._get_working_memory_context(None) == []
    working_memory.get_messages.assert_not_called()
