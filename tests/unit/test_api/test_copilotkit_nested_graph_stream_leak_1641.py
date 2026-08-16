"""Regression tests for #1641: a NESTED graph's answer-named node must not render.

#1547 suppressed the literal node name ``"tools"``. #1636 widened that to an
allow-list of the AG-UI graph's answer nodes (``_ANSWER_NODE_NAMES``). Both
keyed on ``metadata.langgraph_node`` — an identifier that is **not unique across
graph boundaries**, which is the gap this file closes.

Root cause (eval 2026-08-15, turn A.9-followup). Two different graphs each
register a node called ``synthesize``:

* ``src/agents/orchestrator/graph.py`` — ``dispatch -> synthesize -> END``
* ``src/api/routes/copilotkit.py``     — ``tools -> synthesize -> END``

``astream_events`` reports the INNERMOST node of a nested graph, so when
``orchestrator_tool`` runs inside the AG-UI graph's ``tools`` node the
orchestrator's own ``synthesize`` surfaces as ``langgraph_node == "synthesize"``
— matching the allow-list — and its dispatch summary was delivered as the FIRST
assistant message, a generic "# Strategic Insights Summary" asserting *"Both
analyses returned null results (0 findings)"* while the real answer below it
surfaced a lead. The real answer literally opens "Straight answer:".

MEASURED, ``docs/demos/results/2026-08-15_copilot_chat_perf/raw_agui.jsonl``,
``on_chat_model_*`` events only (51 turns):

===================================  =====  ===========================
``langgraph_checkpoint_ns`` (shape)  count  what it is
===================================  =====  ===========================
``chat``                              1372  outer graph, answer text
``synthesize``                         782  outer graph, answer text
``tools|synthesize``                    32  **the #1641 leak**
``tools|classify``                       6  the #1636 leak
===================================  =====  ===========================

On turn A.9-followup specifically the split is exact: 3472 chars stream from
``tools:<id>|synthesize:<id>`` (the template) and 2552 chars from
``synthesize:<id>`` (the real answer) — byte-for-byte the two assistant message
lengths reported on the issue.

Contract under test: a chat-model event raised inside a NESTED graph is
machinery and can never be answer text, whatever the innermost node is called.
LangGraph joins each nesting level of ``langgraph_checkpoint_ns`` with
``NS_SEP`` (``"|"``, documented in ``langgraph._internal._constants`` as
"separates each level (ie. graph|subgraph|subsubgraph)"), so the presence of
that separator IS the depth signal.

DEPTH DOES NOT REPLACE THE NAME CHECK — both rules are load-bearing, and each
catches leaks the other misses. Measured counter-example in the 2026-08-11
corpus: the #1547 leak on turn 2.6 carries ``langgraph_node == "tools"`` with
``langgraph_checkpoint_ns == "tools"`` — 4063 chars of planner JSON at depth
ZERO, because a direct LLM call inside the tools node never enters a subgraph.
A depth-only filter would readmit it. Conversely a nested answer-named node
(this issue) is invisible to a name-only filter. See
``TestDepthDoesNotReplaceTheNameCheck``.

NOT KEYED ON LIFECYCLE COUNT. Turns 1.7 / 6.1 / 6.2 / 6.5 legitimately emit more
than one ``TEXT_MESSAGE_START`` and are benign; ``copilotkit.py``'s existing
duplicate-lifecycle detector only logs for that reason. Suppressing by POSITION
would, on the #1636 sibling defect, keep the junk and delete the answer. Every
assertion here keys on ORIGIN.

#1547's fail-open contract is preserved at BOTH levels: a missing
``langgraph_node`` streams, and so does an answer node with a missing or
non-string ``langgraph_checkpoint_ns``. A metadata regression must degrade to
noise, never to a mute chatbot.
"""

from typing import Any, Dict, List, Optional

import pytest

from src.api.routes.copilotkit import (
    _ANSWER_NODE_NAMES,
    _CHECKPOINT_NS_SEPARATOR,
    LangGraphAgent,
)

pytestmark = pytest.mark.unit


class _FakeChunk:
    """Minimal AIMessageChunk stand-in (only the fields the translator reads)."""

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


def _ns(*segments: str) -> str:
    """Build a realistic ``langgraph_checkpoint_ns``: ``node:<task_id>`` per level.

    Task ids are real-shaped so nothing in the assertions can accidentally
    depend on them; only the number of levels is meaningful.
    """
    task_ids = [
        "3d4f0eaa-f96e-fc7c-d560-53338e695341",
        "8a1c77b2-2f10-4e55-b6ad-9c0f2e1d4477",
        "c05e91d3-77aa-4b21-9f3e-1d6b8a0c5522",
    ]
    return _CHECKPOINT_NS_SEPARATOR.join(
        f"{seg}:{task_ids[i % len(task_ids)]}" for i, seg in enumerate(segments)
    )


def _stream_event(
    node: Optional[str],
    content: str,
    checkpoint_ns: Optional[str] = None,
    chunk_id: str = "lc_run--test",
    include_ns: bool = True,
) -> dict:
    metadata: Dict[str, Any] = {}
    if node is not None:
        metadata["langgraph_node"] = node
    if include_ns:
        metadata["langgraph_checkpoint_ns"] = (
            checkpoint_ns if checkpoint_ns is not None else _ns(node or "unknown")
        )
    return {
        "event": "on_chat_model_stream",
        "metadata": metadata,
        "data": {"chunk": _FakeChunk(content=content, chunk_id=chunk_id)},
    }


def _end_event(node: str, checkpoint_ns: Optional[str] = None) -> dict:
    return {
        "event": "on_chat_model_end",
        "metadata": {
            "langgraph_node": node,
            "langgraph_checkpoint_ns": (checkpoint_ns if checkpoint_ns is not None else _ns(node)),
        },
        "data": {},
    }


async def _collect(agent: LangGraphAgent, event: dict) -> list:
    return [e async for e in agent._handle_single_event(event, {})]


#: Verbatim head of the leaked first assistant message, turn A.9-followup,
#: 2026-08-15 eval (full length 3472 chars, streamed from
#: ``tools:<id>|synthesize:<id>``).
LEAKED_ORCHESTRATOR_SUMMARY = (
    "# Strategic Insights Summary\n\n## 1. KEY FINDING\n"
    "**No actionable performance gaps or key findings identified** across "
    "analyzed segments. Both analyses returned null results (0 findings, 0 "
    "segments meeting threshold criteria)"
)

#: Verbatim head of the REAL answer for the same turn (full length 2552 chars,
#: streamed from ``synthesize:<id>``).
REAL_ANSWER = (
    "Straight answer: neither tool call actually surfaced a Northeast‑specific, "
    "Kisqali-specific root cause for the Q1 dip — here's exactly what came back "
    "and why it doesn't close the loop:"
)


class TestNestedAnswerNodeSuppressed:
    """The #1641 leak itself, keyed on namespace DEPTH not on the node name."""

    async def test_nested_synthesize_emits_nothing(self):
        agent = _bare_agent()
        out = await _collect(
            agent,
            _stream_event(
                "synthesize",
                LEAKED_ORCHESTRATOR_SUMMARY,
                checkpoint_ns=_ns("tools", "synthesize"),
            ),
        )
        assert out == [], (
            "the nested orchestrator's synthesize reached the answer stream: "
            f"{[getattr(e, 'type', e) for e in out]}"
        )

    async def test_nested_synthesize_leaves_no_message_in_progress(self):
        """A suppressed stream must not open a lifecycle either — a dangling
        TEXT_MESSAGE_START still renders an empty bubble."""
        agent = _bare_agent()
        await _collect(
            agent,
            _stream_event(
                "synthesize",
                LEAKED_ORCHESTRATOR_SUMMARY,
                checkpoint_ns=_ns("tools", "synthesize"),
            ),
        )
        assert agent.messages_in_process == {}

    async def test_outer_synthesize_with_same_name_still_streams(self):
        """The discriminator is depth, not the name: the identically-named OUTER
        node must survive, otherwise the fix blanks the turn."""
        agent = _bare_agent()
        out = await _collect(
            agent,
            _stream_event("synthesize", REAL_ANSWER, checkpoint_ns=_ns("synthesize")),
        )
        assert out, "the outer graph's synthesize was wrongly silenced"
        contents = [e for e in out if "TEXT_MESSAGE_CONTENT" in str(getattr(e, "type", ""))]
        assert contents, [str(getattr(e, "type", "")) for e in out]
        assert contents[0].delta == REAL_ANSWER


class TestDepthPropertyNotNodeName:
    """Pin the PROPERTY. These must hold for names that do not exist today, so a
    future nested graph reusing an answer-node name is covered without being
    enumerated first."""

    @pytest.mark.parametrize("node", sorted(_ANSWER_NODE_NAMES))
    async def test_any_answer_node_nested_one_level_is_suppressed(self, node: str):
        agent = _bare_agent()
        out = await _collect(
            agent,
            _stream_event(node, "nested machinery", checkpoint_ns=_ns("tools", node)),
        )
        assert out == [], f"nested {node!r} leaked into the answer stream"

    @pytest.mark.parametrize("node", sorted(_ANSWER_NODE_NAMES))
    async def test_any_answer_node_at_depth_zero_streams(self, node: str):
        agent = _bare_agent()
        out = await _collect(
            agent, _stream_event(node, "Kisqali TRx is 11,298.", checkpoint_ns=_ns(node))
        )
        assert out, f"top-level answer node {node!r} was wrongly silenced"

    async def test_two_level_nesting_is_suppressed(self):
        """``tools|dispatch|roi_calculator`` was measured two levels deep in the
        same run. Depth closes the whole class, not one more name."""
        agent = _bare_agent()
        out = await _collect(
            agent,
            _stream_event(
                "synthesize",
                "sub-sub-graph machinery",
                checkpoint_ns=_ns("tools", "dispatch", "synthesize"),
            ),
        )
        assert out == []

    async def test_suppression_survives_unknown_future_node_names(self):
        """Neither the outer node name nor the inner one is enumerated anywhere."""
        agent = _bare_agent()
        out = await _collect(
            agent,
            _stream_event(
                "chat",
                "machinery from a graph that does not exist yet",
                checkpoint_ns=_ns("some_future_tool_node", "chat"),
            ),
        )
        assert out == []


class TestDepthDoesNotReplaceTheNameCheck:
    """Measured counter-examples: real leaks that carry NO separator. If depth
    were the only rule these would be readmitted."""

    async def test_depth_zero_tools_node_still_suppressed(self):
        """The #1547 leak, 2026-08-11 turn 2.6: ``langgraph_node == "tools"`` with
        ``langgraph_checkpoint_ns == "tools"`` — 4063 chars of planner JSON at
        depth ZERO (a direct LLM call inside the tools node enters no subgraph)."""
        agent = _bare_agent()
        out = await _collect(
            agent,
            _stream_event(
                "tools",
                '{\n  "reasoning": "The query asks for a resource allocation',
                checkpoint_ns=_ns("tools"),
            ),
        )
        assert out == [], "the #1547 depth-0 tools leak was readmitted"

    @pytest.mark.parametrize("node", ["assemble", "reason", "generate", "audit_init"])
    async def test_depth_zero_machinery_from_a_separately_invoked_graph(self, node: str):
        """Measured in the same 2026-08-15 run: the explainer graph is invoked with
        a FRESH config, so its nodes carry depth-0 namespaces (``assemble``,
        ``reason``, ``generate``, ``audit_init``) despite being machinery. Only
        the name allow-list catches these."""
        agent = _bare_agent()
        out = await _collect(agent, _stream_event(node, "internal", checkpoint_ns=_ns(node)))
        assert out == [], f"depth-0 machinery node {node!r} leaked"


class TestFailOpenContractPreserved:
    """#1547's safety property, now at BOTH levels of the check. Suppress only on
    a KNOWN-bad origin; a metadata regression degrades to noise, never to a mute
    chatbot."""

    async def test_absent_node_metadata_still_streams(self):
        agent = _bare_agent()
        out = await _collect(agent, _stream_event(None, "legitimate answer text"))
        assert out, "a stream with no node metadata must not be silenced"

    async def test_answer_node_with_absent_checkpoint_ns_still_streams(self):
        """The new signal must not become a new way to mute the chatbot."""
        agent = _bare_agent()
        out = await _collect(agent, _stream_event("synthesize", "answer text", include_ns=False))
        assert out, "an answer node with no checkpoint_ns must not be silenced"

    @pytest.mark.parametrize("bad_ns", [None, 123, ["tools", "synthesize"], {"ns": "x"}])
    async def test_answer_node_with_non_string_checkpoint_ns_still_streams(self, bad_ns: Any):
        agent = _bare_agent()
        event = _stream_event("synthesize", "answer text")
        event["metadata"]["langgraph_checkpoint_ns"] = bad_ns
        out = await _collect(agent, event)
        assert out, f"checkpoint_ns={bad_ns!r} must fail open, not silence the answer"

    async def test_empty_metadata_dict_still_streams(self):
        agent = _bare_agent()
        out = await _collect(
            agent,
            {
                "event": "on_chat_model_stream",
                "metadata": {},
                "data": {"chunk": _FakeChunk(content="answer text")},
            },
        )
        assert out


class TestNonChatModelEventsUnaffected:
    async def test_nested_tool_lifecycle_events_pass(self):
        """Only chat-model callbacks are dropped. Tool/chain lifecycle events from
        nested graphs drive AG-UI state and must still flow."""
        for name in ("on_tool_start", "on_tool_end", "on_chain_start", "on_custom_event"):
            assert not LangGraphAgent._is_tool_internal_llm_event(
                {
                    "event": name,
                    "metadata": {
                        "langgraph_node": "synthesize",
                        "langgraph_checkpoint_ns": _ns("tools", "synthesize"),
                    },
                }
            ), f"{name} from a nested graph was wrongly matched"

    async def test_non_dict_event_not_matched(self):
        assert not LangGraphAgent._is_tool_internal_llm_event("RUN_STARTED")


class TestTurnA9FollowupReplay:
    """Replay the measured A.9-followup ORDER: the nested template streams FIRST,
    the real answer second. Exactly one lifecycle must survive and it must be the
    real answer — the acceptance criterion on the issue."""

    async def test_only_the_real_answer_survives(self):
        agent = _bare_agent()
        emitted: List[Any] = []
        for ev in (
            _stream_event(
                "synthesize",
                LEAKED_ORCHESTRATOR_SUMMARY,
                checkpoint_ns=_ns("tools", "synthesize"),
                chunk_id="lc_run--orchestrator",
            ),
            _end_event("synthesize", checkpoint_ns=_ns("tools", "synthesize")),
            _stream_event(
                "synthesize",
                REAL_ANSWER,
                checkpoint_ns=_ns("synthesize"),
                chunk_id="lc_run--outer",
            ),
            _end_event("synthesize", checkpoint_ns=_ns("synthesize")),
        ):
            emitted.extend(await _collect(agent, ev))

        types = [str(getattr(e, "type", "")) for e in emitted]
        starts = [e for e, t in zip(emitted, types, strict=True) if "TEXT_MESSAGE_START" in t]
        ends = [e for e, t in zip(emitted, types, strict=True) if "TEXT_MESSAGE_END" in t]
        contents = [e for e, t in zip(emitted, types, strict=True) if "TEXT_MESSAGE_CONTENT" in t]

        # Exactly one lifecycle, and it is the OUTER one — pinning the message
        # ids is what proves the SURVIVOR is the answer rather than the template,
        # which a bare count could not distinguish.
        assert len(starts) == 1, types
        assert len(ends) == 1, types
        assert [starts[0].message_id, ends[0].message_id] == ["lc_run--outer", "lc_run--outer"]
        assert all(c.message_id == "lc_run--outer" for c in contents)

        delivered = "".join(c.delta for c in contents)
        assert delivered == REAL_ANSWER
        assert "Strategic Insights Summary" not in delivered

        # The ended lifecycle left clean bookkeeping behind (idiom from #1547):
        # the suppressed nested stream must not have corrupted it.
        assert agent.messages_in_process.get("run-1") is None

    async def test_turn_is_not_blanked_when_only_outer_text_exists(self):
        """The failure mode that would make this fix worse than the bug: a turn
        whose only answer text is nested would render EMPTY. Measured over all 51
        turns of the 2026-08-15 run, every turn has non-zero depth-0 answer text,
        so no turn is blanked — this pins the surviving path."""
        agent = _bare_agent()
        emitted: List[Any] = []
        for ev in (
            _stream_event("synthesize", REAL_ANSWER, checkpoint_ns=_ns("synthesize")),
            _end_event("synthesize", checkpoint_ns=_ns("synthesize")),
        ):
            emitted.extend(await _collect(agent, ev))
        contents = [e for e in emitted if "TEXT_MESSAGE_CONTENT" in str(getattr(e, "type", ""))]
        assert contents, "a turn with only outer answer text must still render"
        assert "".join(c.delta for c in contents) == REAL_ANSWER


class TestLangGraphCoupling:
    """The separator is LangGraph's, not ours. Pin the coupling so an upgrade that
    changes it fails HERE instead of silently disabling the depth check (the
    established idiom in test_copilotkit_tool_stream_leak_1547.py)."""

    def test_separator_matches_langgraph_ns_sep(self):
        from langgraph._internal._constants import NS_SEP

        assert _CHECKPOINT_NS_SEPARATOR == NS_SEP

    def test_langgraph_still_documents_ns_sep_as_the_level_separator(self):
        """``NS_SEP`` is only meaningful to us because it separates NESTING
        LEVELS. If a future LangGraph repurposes it, the depth reading is wrong
        even though the character still matches."""
        import inspect

        import langgraph._internal._constants as consts

        source = inspect.getsource(consts)
        assert "separates each level" in source


class TestAnswerGraphIsAlwaysTopLevel:
    """The depth rule reads "nested == machinery". That is only sound while the
    AG-UI graph is itself always the OUTERMOST graph — if it were ever compiled
    INTO another graph, its own ``chat``/``synthesize`` would acquire a ``|`` and
    the filter would MUTE the chatbot rather than merely leak.

    Verified by inspection when the fix landed: ``create_e2i_chat_agent`` is
    consumed only as a top-level graph (module-level ``e2i_chat_graph``, the
    ``LangGraphAgent`` ``graph_factory``, and ``chat_bridge.py`` which streams it
    directly). That is a property of the CALLERS, not of this module, so pin it
    — a future ``add_node("...", e2i_chat_graph)`` anywhere in ``src/`` must fail
    HERE instead of silently blanking every answer.

    SCOPE, STATED HONESTLY: the scan below is a TRIPWIRE, not a proof. It parses
    ``src/`` and taints names bound to the factory's result (through imports,
    aliases and simple assignments), which covers the realistic refactors — but a
    sufficiently indirect construction (a registry, a lambda, a dict of graphs)
    would still slip past. It buys early warning on the likely path, not a
    guarantee.
    """

    def test_chat_graph_is_never_embedded_as_a_subgraph(self):
        import ast
        import pathlib

        import src

        src_root = pathlib.Path(src.__file__).parent
        origin = "src.api.routes.copilotkit"
        seeds = {"create_e2i_chat_agent", "e2i_chat_graph"}
        offenders = []

        for py in src_root.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            except SyntaxError:  # pragma: no cover - defensive
                continue

            # Local names that refer to the chat graph or its factory: the seed
            # symbols themselves, any alias they were imported under, and any
            # variable assigned from the factory call or the compiled graph.
            tainted = set(seeds)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(origin):
                    for alias in node.names:
                        if alias.name in seeds:
                            tainted.add(alias.asname or alias.name)
                elif isinstance(node, ast.Assign):
                    value = node.value
                    referent = value.func if isinstance(value, ast.Call) else value
                    name = getattr(referent, "id", None) or getattr(referent, "attr", None)
                    if name in tainted:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                tainted.add(target.id)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "attr", None) != "add_node":
                    continue
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    referent = arg.func if isinstance(arg, ast.Call) else arg
                    name = getattr(referent, "id", None) or getattr(referent, "attr", None)
                    if name in tainted:
                        offenders.append(f"{py}:{node.lineno}: add_node(... {name} ...)")

        assert not offenders, (
            "the AG-UI chat graph is being embedded as a SUBGRAPH; its answer "
            "nodes would then carry a nested checkpoint namespace and be "
            "suppressed as machinery, muting the chatbot (#1641):\n" + "\n".join(offenders)
        )

    def test_the_subgraph_tripwire_actually_detects_an_embedding(self):
        """A scan that can never fire is worse than no scan. Prove this one trips
        on the exact refactor it is meant to catch — an aliased import assigned to
        a local and then embedded — so it cannot silently rot into a no-op."""
        import ast

        source = (
            "from src.api.routes.copilotkit import create_e2i_chat_agent as make_graph\n"
            "inner = make_graph()\n"
            "workflow.add_node('chat_subgraph', inner)\n"
        )
        tree = ast.parse(source)
        tainted = {"create_e2i_chat_agent", "e2i_chat_graph"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                "src.api.routes.copilotkit"
            ):
                for alias in node.names:
                    if alias.name in tainted:
                        tainted.add(alias.asname or alias.name)
            elif isinstance(node, ast.Assign):
                value = node.value
                referent = value.func if isinstance(value, ast.Call) else value
                name = getattr(referent, "id", None) or getattr(referent, "attr", None)
                if name in tainted:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            tainted.add(target.id)

        hits = [
            arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_node"
            for arg in node.args
            if getattr(arg, "id", None) in tainted
        ]
        assert hits, "the tripwire failed to detect an aliased subgraph embedding"

    def test_graph_registers_the_answer_nodes_at_its_own_top_level(self):
        """The other half of the invariant: the nodes the allow-list names really
        are THIS graph's own nodes, so they are the ones running at depth 0."""
        from src.api.routes.copilotkit import _TOOL_NODE_NAME, e2i_chat_graph

        nodes = set(e2i_chat_graph.nodes)
        assert _ANSWER_NODE_NAMES <= nodes, (
            f"allow-listed answer nodes missing from the graph: {_ANSWER_NODE_NAMES - nodes}"
        )
        assert _TOOL_NODE_NAME in nodes
