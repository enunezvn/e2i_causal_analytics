"""Regression tests for #1636: control-plane LLM streams must never render as answer text.

#1547 closed this on the ``tools`` node. It reopened on a THIRD surface because
the filter matched a single literal node name rather than the property that
matters.

Root cause (eval 2026-08-15, turn 2.1). The AG-UI graph registers exactly three
nodes — ``chat``, ``tools``, ``synthesize`` (``copilotkit.py:3691-3693``). But
LangGraph's ``astream_events`` propagates callbacks into NESTED graphs invoked
inside a tool, and the metadata carries the INNERMOST node name. So the
orchestrator's intent classifier surfaced as
``metadata.langgraph_node == "classify"`` — a node that belongs to no graph the
AG-UI filter knew about. It matched neither the suppressed ``"tools"`` nor the
allowed ``"chat"``/``"synthesize"``, so it fell through the fail-open branch and
its raw JSON was delivered as the FIRST assistant message:

    ```json
    {"primary_intent": "segment_analysis", "confidence": 0.92,
     "requires_multi_agent": false}
    ```

Measured across all 51 turns of ``docs/demos/results/2026-08-15_copilot_chat_perf/
raw_agui.jsonl``, ``langgraph_node`` on ``on_chat_model_*`` events takes exactly
three values — ``chat`` (1372), ``synthesize`` (814), ``classify`` (6) — and is
NEVER missing (0 occurrences). Notably ``tools`` appears 0 times, which is why a
literal ``== "tools"`` match no longer catches tool-internal streams at all.

Contract under test: an ``on_chat_model_*`` event is suppressed when it carries a
``langgraph_node`` that is not an ANSWER node of the AG-UI graph. Unknown node
names are suppressed (they are by definition not this graph's answer nodes);
ABSENT metadata still fails open, preserving #1547's safety property that a
legitimate stream is never silenced on missing information.

Why the obvious alternative is wrong: ``copilotkit.py:1042`` already detects
duplicate lifecycles and only logs them. Suppressing "the extra lifecycle" by
POSITION would, on turn 2.1, keep the 101-char classifier blob and delete the
1364-char real answer — the leak must be discriminated by ORIGIN, never by order.
"""

from typing import Any, Dict, List, Optional

import pytest

from src.api.routes.copilotkit import _ANSWER_NODE_NAMES, LangGraphAgent

pytestmark = pytest.mark.unit


class _FakeChunk:
    def __init__(self, content: Any = "", chunk_id: str = "lc_run--test"):
        self.content = content
        self.id = chunk_id
        self.response_metadata: Dict[str, Any] = {}
        self.tool_call_chunks: List[Dict[str, Any]] = []
        self.additional_kwargs: Dict[str, Any] = {}


def _bare_agent() -> LangGraphAgent:
    agent = object.__new__(LangGraphAgent)
    agent.messages_in_process = {}
    agent.active_run = {"id": "run-1"}
    return agent


def _stream_event(node: Optional[str], content: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node} if node is not None else {},
        "data": {"chunk": _FakeChunk(content=content)},
    }


async def _collect(agent: LangGraphAgent, event: dict) -> list:
    return [e async for e in agent._handle_single_event(event, {})]


#: Verbatim first assistant message from turn 2.1 of the 2026-08-15 eval.
LEAKED_CLASSIFIER_JSON = (
    '```json\n{"primary_intent": "segment_analysis", "confidence": 0.92, '
    '"requires_multi_agent": false}\n```'
)


class TestClassifyNodeSuppressed:
    async def test_classify_node_stream_emits_nothing(self):
        """The 2.1 leak: the intent classifier's own generation must not render."""
        agent = _bare_agent()
        out = await _collect(agent, _stream_event("classify", LEAKED_CLASSIFIER_JSON))
        assert out == [], f"classifier JSON reached the answer stream: {out}"

    async def test_classify_node_leaves_no_message_in_progress(self):
        """A suppressed stream must not open a lifecycle either — a dangling
        TEXT_MESSAGE_START would still render an empty bubble."""
        agent = _bare_agent()
        await _collect(agent, _stream_event("classify", LEAKED_CLASSIFIER_JSON))
        assert agent.messages_in_process == {}


class TestClassIsClosedNotJustThisNode:
    """#1547 fixed one node name and the defect returned on another. These pin the
    PROPERTY (not an answer node) rather than an enumeration of known offenders."""

    @pytest.mark.parametrize(
        "node",
        ["classify", "tools", "plan", "decompose", "route", "gap_analyzer"],
    )
    async def test_any_non_answer_node_is_suppressed(self, node: str):
        agent = _bare_agent()
        out = await _collect(agent, _stream_event(node, "internal machinery"))
        assert out == [], f"node {node!r} leaked into the answer stream"

    @pytest.mark.parametrize("node", sorted(_ANSWER_NODE_NAMES))
    async def test_answer_nodes_still_stream(self, node: str):
        agent = _bare_agent()
        out = await _collect(agent, _stream_event(node, "Kisqali TRx is 11,298."))
        assert out, f"answer node {node!r} was wrongly silenced"

    async def test_absent_metadata_still_fails_open(self):
        """#1547's safety property is preserved: suppress on KNOWN-bad origin, never
        on missing information. Measured 0 occurrences in the 51-turn run, but the
        guarantee is what stops a metadata regression from muting all answers."""
        agent = _bare_agent()
        out = await _collect(agent, _stream_event(None, "legitimate answer text"))
        assert out, "a stream with no node metadata must not be silenced"


class TestAnswerNodeSet:
    def test_answer_nodes_are_exactly_the_graph_s_answer_nodes(self):
        """If a new answer-producing node is added to the graph, this fails loudly
        rather than silently muting it."""
        assert _ANSWER_NODE_NAMES == frozenset({"chat", "synthesize"})
