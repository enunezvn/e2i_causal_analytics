"""#2119 Part D: a sync agent's worker thread sees the context the turn bound.

``_dispatch_agent`` offloads agents whose spec says ``is_async=False`` (today
only ``experiment_designer``) to a worker thread. It used a bare
``loop.run_in_executor``, which starts the thread from an EMPTY context:
``asyncio.to_thread`` is the variant that runs the call under
``contextvars.copy_context()``. So every metered LLM call such an agent made
read ``get_attribution()`` as None and recorded ``surface='other'``, and its
``record_usage`` accumulated into nothing the caller's ``drain_run_usage`` could
see — on the plain routes and the AG-UI route alike, since both reach the
orchestrator's dispatcher.

The dispatcher, its spec lookup and the offload are the real ones; the only
double is the agent, registered under a test-local sync spec.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

import pytest

from src.agents.orchestrator import _agent_method_map
from src.agents.orchestrator._agent_method_map import AgentMethodSpec
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode
from src.api.routes.chatbot_tools import reset_chat_session_id, set_chat_session_id
from src.utils.llm_attribution import (
    clear_attribution,
    drain_run_usage,
    get_attribution,
    record_usage,
)

SESSION = "u~s"
REQUEST_ID = "req-2119-d"
AGENT = "sync_probe_2119"


class _SyncProbeAgent:
    """A sync agent that reports what its thread could see, and meters one call."""

    def __init__(self) -> None:
        self.seen: Dict[str, Any] = {}

    def run(self, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        from src.api.routes.chatbot_tools import chat_session_id_context

        self.seen["attribution"] = get_attribution()
        self.seen["session_var"] = chat_session_id_context.get()
        self.seen["worker_thread"] = threading.current_thread() is not threading.main_thread()
        record_usage("probe-model", 3, 5)
        return {"narrative": "designed", "success": True}


@pytest.fixture
def sync_agent(monkeypatch) -> _SyncProbeAgent:
    monkeypatch.setitem(
        _agent_method_map.AGENT_METHOD_MAP, AGENT, AgentMethodSpec(method="run", is_async=False)
    )
    return _SyncProbeAgent()


@pytest.fixture
def bound_turn():
    from src.utils.llm_attribution import set_chat_attribution

    clear_attribution()
    set_chat_attribution(SESSION, REQUEST_ID)
    token = set_chat_session_id(SESSION)
    yield
    reset_chat_session_id(token)
    clear_attribution()


async def test_a_sync_agents_thread_sees_the_turns_attribution_and_meters_into_its_drain(
    sync_agent, bound_turn
):
    dispatcher = DispatcherNode(agent_registry={AGENT: sync_agent})
    state = {
        "query": "design an experiment",
        "dispatch_plan": [
            {
                "agent_name": AGENT,
                "priority": 1,
                "parameters": {},
                "timeout_ms": 30000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [[AGENT]],
    }

    result = await dispatcher.execute(state)

    assert result["agent_results"][0]["success"] is True, result["agent_results"][0]
    assert sync_agent.seen["worker_thread"] is True, "the sync agent did not run in a worker"
    attribution = sync_agent.seen["attribution"]
    assert attribution is not None, "the worker thread saw no attribution"
    assert attribution.surface == "chat"
    assert attribution.session_id == SESSION
    assert attribution.request_id == REQUEST_ID
    assert sync_agent.seen["session_var"] == SESSION
    drained = drain_run_usage()
    assert drained is not None, "the usage metered in the thread never reached the drain"
    assert (drained.input_tokens, drained.output_tokens, drained.last_model) == (
        3,
        5,
        "probe-model",
    )
