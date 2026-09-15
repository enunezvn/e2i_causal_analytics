"""#2099: the explainer's checkpointer thread id is not a memory session.

``_generate_session_id`` minted ``explainer_<hex12>`` and one value did two jobs:
the LangGraph checkpointer's ``config["configurable"]["thread_id"]``, which
genuinely needs a non-empty handle per run, and ``state["session_id"]``, which is
a claim about which conversation the row belongs to.

The two roles split here. The thread id keeps its mint under its real name; the
state session is the caller's session or ``None``.

There is no row-level change on today's traffic: ``coerce_session_uuid`` already
returns ``None`` for ``explainer_ab12...``, and all 43 live explainer episodic
rows carry a real caller session. What does change is the guard in
``narrative_generator``: it gated the whole episodic write on a truthy session,
so passing ``None`` would have stopped writing the explanation row at all rather
than writing it with a NULL session. Only the session-keyed cache is gated now.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest

from src.agents.explainer.agent import ExplainerAgent
from src.agents.explainer.nodes.context_assembler import ContextAssemblerNode
from src.agents.explainer.nodes.deep_reasoner import DeepReasonerNode
from src.agents.explainer.nodes.narrative_generator import NarrativeGeneratorNode

_MINT_RE = re.compile(r"^explainer_[0-9a-f]{12}$")
_SESSION = "eeba22e7-4d9d-49ea-977b-b9e9d1549c53"


class _RecordingGraph:
    """Captures the state and config the agent invokes the graph with."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(self, state: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        self.calls.append({"state": state, "config": config})
        return {
            "executive_summary": "s",
            "detailed_explanation": "d",
            "extracted_insights": [],
            "visual_suggestions": [],
            "follow_up_questions": [],
            "total_latency_ms": 1,
            "status": "completed",
            "errors": [],
            "warnings": [],
        }


def _agent(graph: _RecordingGraph) -> ExplainerAgent:
    agent = ExplainerAgent()
    agent._get_graph = lambda *_a, **_kw: graph  # type: ignore[assignment]
    return agent


async def _explain(agent: ExplainerAgent, **kwargs: Any) -> Any:
    return await agent.explain(
        query="why did TRx move",
        analysis_results=[{"finding": "up"}],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_state_session_is_none_when_the_caller_has_none():
    """The memory session must not inherit the checkpointer's minted handle."""
    graph = _RecordingGraph()

    await _explain(_agent(graph))

    state = graph.calls[0]["state"]
    got = state["session_id"]
    assert not (isinstance(got, str) and _MINT_RE.match(got)), f"minted session: {got!r}"
    assert got is None


@pytest.mark.asyncio
async def test_checkpointer_thread_id_still_gets_a_handle():
    """The checkpointer needs a per-run id; it just stops being a session."""
    graph = _RecordingGraph()

    await _explain(_agent(graph))

    thread_id = graph.calls[0]["config"]["configurable"]["thread_id"]
    assert isinstance(thread_id, str) and thread_id, f"thread lost its handle: {thread_id!r}"
    assert _MINT_RE.match(thread_id), thread_id
    assert thread_id != graph.calls[0]["state"]["session_id"]


@pytest.mark.asyncio
async def test_caller_session_drives_both_roles():
    """With a real session, nothing changes: it is both session and thread."""
    graph = _RecordingGraph()

    await _explain(_agent(graph), session_id=_SESSION)

    assert graph.calls[0]["state"]["session_id"] == _SESSION
    assert graph.calls[0]["config"]["configurable"]["thread_id"] == _SESSION


@pytest.mark.asyncio
async def test_two_runs_get_distinct_thread_ids():
    """A shared thread id would collide checkpoints across runs."""
    graph = _RecordingGraph()
    agent = _agent(graph)

    await _explain(agent)
    await _explain(agent)

    first = graph.calls[0]["config"]["configurable"]["thread_id"]
    second = graph.calls[1]["config"]["configurable"]["thread_id"]
    assert first != second


class _RecordingHooks:
    def __init__(self) -> None:
        self.cached: List[Optional[str]] = []
        self.stored: List[Optional[str]] = []

    async def cache_explanation(self, session_id: Any, explanation: Any) -> bool:
        self.cached.append(session_id)
        return True

    async def store_explanation(
        self, session_id: Any, explanation: Any, brand: Any = None, region: Any = None
    ) -> str:
        self.stored.append(session_id)
        return "mem-1"


def _node(hooks: _RecordingHooks) -> NarrativeGeneratorNode:
    node = NarrativeGeneratorNode(use_llm=False)
    node._memory_hooks = hooks  # type: ignore[assignment]
    return node


async def _reasoned_state(base_state: Dict[str, Any], analysis_results: List[Any]) -> Any:
    """Drive the real upstream nodes, as test_narrative_generator.py does.

    These tests go through ``NarrativeGeneratorNode.execute()`` rather than
    calling ``_store_explanation_in_memory`` directly, because the behaviour
    under test IS the guard in ``execute`` -- calling the helper straight would
    bypass the changed line and pass on base.
    """
    assembled = await ContextAssemblerNode().execute(
        {**base_state, "analysis_results": analysis_results}  # type: ignore[arg-type]
    )
    return await DeepReasonerNode(use_llm=False).execute(assembled)


@pytest.mark.asyncio
async def test_episodic_write_still_happens_without_a_session(
    base_explainer_state, sample_causal_analysis
):
    """The guard gated the whole write; a NULL session must not silence it."""
    hooks = _RecordingHooks()
    reasoned = await _reasoned_state(base_explainer_state, [sample_causal_analysis])

    result = await _node(hooks).execute({**reasoned, "session_id": None})

    assert result["status"] == "completed"
    assert hooks.stored == [None], "the episodic explanation row was not written"


@pytest.mark.asyncio
async def test_session_keyed_cache_is_skipped_without_a_session(
    base_explainer_state, sample_causal_analysis
):
    """The cache key embeds the session; None would write an unreachable key."""
    hooks = _RecordingHooks()
    reasoned = await _reasoned_state(base_explainer_state, [sample_causal_analysis])

    await _node(hooks).execute({**reasoned, "session_id": None})

    assert hooks.cached == [], f"cached under a session-less key: {hooks.cached!r}"


@pytest.mark.asyncio
async def test_both_writes_happen_with_a_real_session(base_explainer_state, sample_causal_analysis):
    """The change is scoped to the session-less path."""
    hooks = _RecordingHooks()
    reasoned = await _reasoned_state(base_explainer_state, [sample_causal_analysis])

    await _node(hooks).execute({**reasoned, "session_id": _SESSION})

    assert hooks.cached == [_SESSION]
    assert hooks.stored == [_SESSION]


def test_the_mint_is_named_for_what_it_is():
    """A ``_generate_session_id`` that feeds a thread id invites the confusion."""
    assert hasattr(ExplainerAgent, "_generate_thread_id")
    assert not hasattr(ExplainerAgent, "_generate_session_id")
