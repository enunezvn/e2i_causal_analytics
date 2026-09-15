"""#2116: causal_impact keeps the caller's chat session; discovery coerces it honestly.

Where the session was lost (read on main 3bc4ee00c): NOT in the dispatcher. For a
``method="run"`` agent without ``uses_kwargs`` the dispatcher MERGES the resolver's
output into the generic payload (``agent_input.update(resolved)``), so the
``session_id`` that ``_prepare_agent_input`` put there survives -- that is why the
heterogeneous_optimizer row carried the thread. The drop is one level down:
``CausalImpactAgent._initialize_state`` built the ``CausalImpactState`` literal from
``input_data`` and never copied ``session_id``, so the memory write at the end of the
run read ``state.get("session_id")`` as ``None`` on every dispatched turn (a NULL row
since #2113; a minted uuid before that).

Keeping the session in state exposes a second defect: ``GraphBuilderNode._run_discovery``
did ``UUID(session_id) if session_id else None``. The plain chat routes carry a composite
``{user}~{session}`` id, which never parses, so with the session in state the parse
raised a ``ValueError`` inside ``_run_discovery``; ``execute`` catches it, logs it as a
warning and surfaces it as ``discovery_skip_reason`` (also appended to the state's
warnings), so every plain-route causal turn with ``auto_discover`` set still answered
from the manual DAG -- visible to an operator reading the log or the state, not to the
user. Discovery now recovers the trailing session uuid with the shared
``coerce_session_uuid`` (a bare uuid in canonical form, ``None`` for anything malformed).

The session stays RAW in the agent's state. Four boundaries coerce it, each for its
own uuid column: discovery (``_run_discovery`` -> ``coerce_session_uuid``); the episodic
writer (``insert_episodic_memory_with_text``, ``_coerce_session_id`` at
episodic_memory.py:751, #1404); the audit chain (audit_chain_mixin.py:516 forwards the
raw state value, utils/audit_chain.py:326 coerces it, child entries inherit the parent's
at :401); and the discovered-DAG repository (``_as_uuid_str`` at discovered_dag.py:110,
called at :207 -- routed through the shared helper by this lane's Part C).

Doubles in these tests, and nothing else: the discovery runner
(``_discovery_runner.discover_dag``) and the discovery gate (``_discovery_gate.evaluate``);
the memory contribution recorder (``memory_hooks.contribute_to_memory``); the episodic
writer (``insert_episodic_memory_with_text``); the activity persist
(``memory_hooks.persist_agent_activity``); and two probe agents plus a fake resolver
registered under test-local dispatch entries. The dispatcher, its resolver merge,
``_initialize_state``, ``_contribute_to_memory``, ``store_causal_analysis`` and
``_run_discovery`` are the real ones.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pandas as pd
import pytest

from src.agents.causal_impact.agent import CausalImpactAgent
from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.agents.causal_impact.state import CausalImpactState
from src.agents.orchestrator import _agent_method_map
from src.agents.orchestrator._agent_method_map import AgentMethodSpec
from src.agents.orchestrator.nodes import dispatcher as dispatcher_module
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode

# Captured at import, before this directory's autouse fixture swaps the method for
# an AsyncMock (see conftest.py). T2 is about that method, so it calls the real one.
_REAL_CONTRIBUTE = CausalImpactAgent._contribute_to_memory

_USER = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
_SESSION = "a59e835e-2b1c-4f7d-9c0e-3d5a6b7c8d9e"
_COMPOSITE = f"{_USER}~{_SESSION}"  # the plain-route shape (chatbot_messages splits on '~')
_AUDIT = "9f2a1d4e-7c3b-4a15-9e28-6b0d5a7c1e39"

_OUTPUT: Dict[str, Any] = {
    "status": "success",
    "ate_estimate": 0.12,
    "confidence": 0.8,
    "refutation_passed": False,
}


def _base_input() -> Dict[str, Any]:
    """The contract's required fields (same shape as test_data_plumbing.py)."""
    return {
        "query": "What is the causal effect of hcp_visits on discontinuation_flag?",
        "query_id": "qid-2116",
        "treatment_var": "hcp_visits",
        "outcome_var": "discontinuation_flag",
        "confounders": [],
        "data_source": "patient_journeys",
    }


def _init(**extra: Any) -> CausalImpactState:
    return CausalImpactAgent(enable_mlflow=False)._initialize_state({**_base_input(), **extra})


# ---------------------------------------------------------------------------
# T1 -- _initialize_state keeps the caller's session, raw, and only when supplied
# ---------------------------------------------------------------------------


def test_initialize_state_keeps_the_callers_session_raw():
    """RED on the base (``KeyError``): a composite id is kept verbatim -- no parse, no
    coercion, no mint here."""
    assert _init(session_id=_COMPOSITE)["session_id"] == _COMPOSITE


def test_initialize_state_without_a_session_leaves_the_key_absent():
    """GREEN on the base (the key was never set): ``session_id`` is ``NotRequired``, a
    session-less call leaves it absent, so the memory write's
    ``state.get("session_id") or None`` stores an honest NULL."""
    assert "session_id" not in _init()


def test_initialize_state_treats_an_empty_session_as_absent():
    """GREEN on the base, pinned so the fix keeps ``""`` out of the state."""
    assert "session_id" not in _init(session_id="")


# ---------------------------------------------------------------------------
# T2 -- the memory write receives the session the dispatcher supplied
# ---------------------------------------------------------------------------


@pytest.fixture
def recorder(monkeypatch):
    """Record the kwargs the agent hands to the memory hook."""
    calls: List[Dict[str, Any]] = []

    async def _contribute(**kwargs: Any) -> Dict[str, int]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr("src.agents.causal_impact.memory_hooks.contribute_to_memory", _contribute)
    return calls


async def test_the_memory_write_receives_the_session_the_dispatcher_supplied(recorder):
    """The real ``_contribute_to_memory`` over a state the real ``_initialize_state``
    built from a payload carrying the composite: the hook sees the composite."""
    state = _init(session_id=_COMPOSITE)

    await _REAL_CONTRIBUTE(CausalImpactAgent(enable_mlflow=False), _OUTPUT, state)  # type: ignore[arg-type]

    assert len(recorder) == 1
    assert recorder[0]["session_id"] == _COMPOSITE


@patch("src.agents.causal_impact.memory_hooks.persist_agent_activity", return_value=None)
async def test_the_row_carries_the_session_and_keeps_the_audit_id_in_raw_content(_activity):
    """One level down, at the writer: the session reaches the row and the audit id
    still travels in ``raw_content`` (#2099), not in the session column."""
    captured: Dict[str, Any] = {}

    async def _insert(memory: Any, text_to_embed: str, session_id: Optional[str]) -> str:
        captured["raw_content"] = memory.raw_content
        captured["session_id"] = session_id
        return "mem-2116"

    from src.agents.causal_impact.memory_hooks import CausalImpactMemoryHooks

    state = _init(session_id=_COMPOSITE)
    # The audit chain stamps this later in the run (not in _initialize_state).
    state["audit_workflow_id"] = _AUDIT

    with patch("src.memory.episodic_memory.insert_episodic_memory_with_text", _insert):
        memory_id = await CausalImpactMemoryHooks().store_causal_analysis(
            # exactly what _contribute_to_memory passes
            session_id=state.get("session_id") or None,
            result=_OUTPUT,
            state=state,  # type: ignore[arg-type]
        )

    assert memory_id == "mem-2116"
    assert captured["session_id"] == _COMPOSITE
    assert captured["raw_content"].get("audit_workflow_id") == _AUDIT


# ---------------------------------------------------------------------------
# T3 -- discovery coerces the state's session instead of parsing it bare
# ---------------------------------------------------------------------------


def _discovery_node() -> GraphBuilderNode:
    """The M-gb1 setup (test_graph_builder.py): runner + gate stubbed, so the
    assertion is on the session wiring only, never on GES/PC."""
    node = GraphBuilderNode()
    fake_result = MagicMock()
    fake_result.n_edges = 1
    node._discovery_runner = MagicMock()
    node._discovery_runner.discover_dag = AsyncMock(return_value=fake_result)
    fake_eval = MagicMock()
    fake_eval.decision.value = "accept"
    fake_eval.to_dict.return_value = {"decision": "accept", "confidence": 0.9}
    node._discovery_gate = MagicMock()
    node._discovery_gate.evaluate = MagicMock(return_value=fake_eval)
    return node


async def _session_seen_by_discovery(session_id: Optional[str]) -> Optional[UUID]:
    node = _discovery_node()
    df = pd.DataFrame(
        {"hcp_visits": [0.1, 0.2, 0.3, 0.4], "discontinuation_flag": [1.0, 2.0, 3.0, 4.0]}
    )
    state: CausalImpactState = {
        "query": "test",
        "query_id": "qid-2116-discovery",
        "treatment_var": "hcp_visits",
        "outcome_var": "discontinuation_flag",
        "data_cache": {"estimation_data": df},
        "status": "pending",
    }
    if session_id is not None:
        state["session_id"] = session_id

    await node._run_discovery(state, "hcp_visits", "discontinuation_flag")

    node._discovery_runner.discover_dag.assert_awaited_once()
    return node._discovery_runner.discover_dag.await_args.kwargs["session_id"]


@pytest.mark.parametrize(
    "session_id, expected",
    [
        pytest.param(_COMPOSITE, UUID(_SESSION), id="composite-plain-route"),
        pytest.param(_SESSION, UUID(_SESSION), id="bare-uuid-agui-thread"),
        pytest.param(f"{_USER}~garbage", None, id="malformed-honest-none"),
        pytest.param(None, None, id="absent"),
    ],
)
async def test_discovery_receives_the_coerced_session_uuid(session_id, expected):
    """Composite -> the trailing session uuid; bare uuid -> itself; malformed -> None
    (never a ValueError that aborts discovery); absent -> None."""
    assert await _session_seen_by_discovery(session_id) == expected


# ---------------------------------------------------------------------------
# T4 -- dispatcher pin: why the session reaches a run(dict) agent at all
# ---------------------------------------------------------------------------

_AGENT = "resolver_probe_2116"


class _RunProbe:
    """A run(dict) agent, like causal_impact / heterogeneous_optimizer."""

    def __init__(self) -> None:
        self.seen: Dict[str, Any] = {}

    async def run(self, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        self.seen = dict(agent_input)
        return {"narrative": "ran", "success": True}


class _KwargsProbe:
    """A kwargs agent, like resource_optimizer / prediction_synthesizer."""

    def __init__(self) -> None:
        self.seen: Dict[str, Any] = {}

    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        self.seen = dict(kwargs)
        return {"narrative": "ran", "success": True}


def _resolver(agent_input: Dict[str, Any], dispatch: Any) -> Dict[str, Any]:
    return {"treatment_var": "t"}


async def _dispatch(monkeypatch, agent: Any, spec: AgentMethodSpec) -> Dict[str, Any]:
    monkeypatch.setitem(_agent_method_map.AGENT_METHOD_MAP, _AGENT, spec)
    monkeypatch.setitem(dispatcher_module.INPUT_RESOLVERS, _AGENT, _resolver)
    dispatcher = DispatcherNode(agent_registry={_AGENT: agent})
    state = {
        "query": "does t lift y",
        "session_id": "thread-1",
        "dispatch_plan": [
            {
                "agent_name": _AGENT,
                "priority": 1,
                "parameters": {},
                "timeout_ms": 30000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [[_AGENT]],
    }
    result = await dispatcher.execute(state)
    assert result["agent_results"][0]["success"] is True, result["agent_results"][0]
    return agent.seen


async def test_a_run_agent_sees_the_resolver_keys_merged_over_the_session(monkeypatch):
    """GREEN on the base: for a run(dict) agent the resolver output is MERGED into the
    generic payload, so the turn's session survives alongside the resolved inputs.
    This pins the corrected mechanism (#2116's issue text said REPLACES)."""
    seen = await _dispatch(monkeypatch, _RunProbe(), AgentMethodSpec(method="run"))

    assert seen["treatment_var"] == "t"
    assert seen["session_id"] == "thread-1"


async def test_a_kwargs_agent_sees_only_what_its_resolver_returned(monkeypatch):
    """GREEN on the base: for a uses_kwargs agent the resolver output REPLACES the
    payload -- which is why those resolvers copy ``session_id`` themselves."""
    seen = await _dispatch(
        monkeypatch, _KwargsProbe(), AgentMethodSpec(method="run", uses_kwargs=True)
    )

    assert seen == {"treatment_var": "t"}
