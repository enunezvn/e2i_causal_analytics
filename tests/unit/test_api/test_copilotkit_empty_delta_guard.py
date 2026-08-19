"""Empty-delta stream-kill guard (2026-08-19 review, session no90vkf).

The browser's ``@ag-ui/core`` ``TextMessageContentEventSchema`` refines
``delta`` with ``s.length > 0`` ("Delta must not be an empty string") and one
invalid event aborts the ENTIRE CopilotKit run client-side — while the backend
200s, finishes the run, and persists the answer. The live chain, measured
2026-08-19 (3/3 endpoint replays, SSE frames 194/197/198/200):

1. sonnet-5 with ``reasoning_effort="medium"`` engages adaptive thinking on
   real synthesis prompts and streams thinking chunks first —
   ``content=[{"thinking": "", "type": "thinking", "index": 0}]`` — truthy
   lists with no ``"text"`` key.
2. The node loops extracted ``""`` from them and called
   ``copilotkit_emit_message(config, "")``.
3. ``execute()``'s CUSTOM -> TEXT_MESSAGE conversion emitted
   ``TEXT_MESSAGE_CONTENT`` with ``delta: ""`` (and opened a message lifecycle
   for it).
4. The frontend Zod refine rejected it and killed the stream; the user saw
   "The assistant didn't respond" on every tool+synthesis turn.

Contract under test (boundary layer): whatever a node emits, the SSE stream
leaving ``execute()`` NEVER contains a ``TEXT_MESSAGE_CONTENT`` event with an
empty ``delta``, and never opens a TEXT_MESSAGE lifecycle for content that is
entirely empty.

Harness: same shape as ``test_agui_stream_health_1667_1669.py`` — a REAL
compiled StateGraph whose node calls the REAL ``copilotkit_emit_message``, run
through a REAL ``LangGraphAgent.execute()``. No part of the event pipeline is
replaced; only the LLM leg is absent because the defect lives entirely in the
emit -> convert path.
"""

import asyncio
import json
from typing import Annotated, Any, Dict, List, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


#: The exact first-chunk shape captured live from sonnet-5 adaptive thinking
#: (stream_dump.jsonl frame 194): a truthy list whose only block has no "text".
THINKING_BLOCK_PAYLOAD: List[Dict[str, Any]] = [{"thinking": "", "type": "thinking", "index": 0}]

#: The signature block that follows it (frame 198).
SIGNATURE_BLOCK_PAYLOAD: List[Dict[str, Any]] = [
    {"signature": "EqMDCkYIChABGAIqQPT-live-shape", "type": "signature", "index": 0}
]

REAL_ANSWER = "West-region TRx shortfall is driven by Remibrutinib access barriers."


class _StubState(TypedDict):
    messages: Annotated[list, add_messages]
    tools: list


def _emitting_graph(payloads: List[Any]):
    """A real compiled StateGraph whose node manually emits ``payloads`` in
    order via the REAL ``copilotkit_emit_message`` — the same call the
    chat/synthesize nodes make per streamed chunk."""
    from copilotkit.langgraph import copilotkit_emit_message
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableConfig

    async def chat(state: _StubState, config: RunnableConfig) -> dict:
        for payload in payloads:
            await copilotkit_emit_message(config, payload)
            await asyncio.sleep(0)  # let each CUSTOM event flush individually
        return {"messages": [AIMessage(content="done")]}

    workflow = StateGraph(_StubState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile(checkpointer=MemorySaver())


def _agent(graph):
    from src.api.routes.copilotkit import LangGraphAgent

    return LangGraphAgent(name="default", description="empty-delta guard stub", graph=graph)


async def _sse_events(payloads: List[Any]) -> List[Dict[str, Any]]:
    """Run execute() over a graph that emits ``payloads`` and parse every SSE
    data frame — the exact byte stream the browser Zod-validates."""
    agent = _agent(_emitting_graph(payloads))
    events: List[Dict[str, Any]] = []
    async for chunk in agent.execute(
        thread_id="thread-empty-delta",
        state={},
        messages=[{"id": "m1", "role": "user", "content": "I am asking about Remi"}],
        config=None,
        actions=[],
    ):
        text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        for line in text.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[len("data: ") :]))
                except json.JSONDecodeError:
                    pass
    return events


def _contents(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if str(e.get("type")) == "TEXT_MESSAGE_CONTENT"]


def _empty_deltas(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Events the frontend @ag-ui/core Zod refine would reject."""
    return [e for e in _contents(events) if not e.get("delta")]


class TestEmptyStringEmit:
    async def test_empty_string_emit_never_streams_empty_delta(self):
        """The minimal defect shape: an emit of "" must produce NO
        TEXT_MESSAGE_CONTENT with empty delta (Zod would kill the run)."""
        events = await _sse_events(["", REAL_ANSWER])
        assert _empty_deltas(events) == [], (
            "Zod-fatal empty delta reached the wire: "
            f"{[e for e in _contents(events)]}"
        )

    async def test_answer_still_delivered_after_empty_emit(self):
        """The guard must DROP the empty chunk, not the stream."""
        events = await _sse_events(["", REAL_ANSWER])
        delivered = "".join(e.get("delta") or "" for e in _contents(events))
        assert REAL_ANSWER in delivered, f"answer lost: {delivered!r}"


async def _sse_from_run_events(run_events: List[Any]) -> List[Dict[str, Any]]:
    """Drive execute()'s conversion/serialization over a controlled event
    sequence — the shapes the CONTAINER's SDK layer measurably forwards
    (manual-emit CUSTOM events can carry str OR content-block-list payloads;
    ag-ui-protocol 0.1.18, the image's pin, does not validate delta)."""
    agent = _agent(_emitting_graph([]))

    async def _fake_run(_run_input):
        for ev in run_events:
            yield ev

    agent.run = _fake_run  # type: ignore[method-assign]
    events: List[Dict[str, Any]] = []
    async for chunk in agent.execute(
        thread_id="thread-run-events",
        state={},
        messages=[{"id": "m1", "role": "user", "content": "q"}],
        config=None,
        actions=[],
    ):
        text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        for line in text.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[len("data: ") :]))
                except json.JSONDecodeError:
                    pass
    return events


def _manual_emit(payload: Any):
    from ag_ui.core import CustomEvent, EventType

    return CustomEvent(
        type=EventType.CUSTOM,
        name="copilotkit_manually_emit_message",
        value={"message": payload, "message_id": "manual-1"},
    )


def _run_lifecycle(inner: List[Any]) -> List[Any]:
    from ag_ui.core import EventType, RunFinishedEvent, RunStartedEvent

    return [
        RunStartedEvent(type=EventType.RUN_STARTED, thread_id="t", run_id="r"),
        *inner,
        RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id="t", run_id="r"),
    ]


class TestThinkingBlockConversion:
    """The route's CUSTOM -> TEXT_MESSAGE conversion receives the manual-emit
    payloads; the live no90vkf sequence had thinking/signature blocks whose
    block-join extracts to "" (v1.21.4 documents the list shape reaching this
    converter). None of it may open or feed a lifecycle."""

    async def test_live_thinking_chunk_shape_never_streams_empty_delta(self):
        events = await _sse_from_run_events(
            _run_lifecycle(
                [
                    _manual_emit(THINKING_BLOCK_PAYLOAD),
                    _manual_emit(SIGNATURE_BLOCK_PAYLOAD),
                    _manual_emit([{"text": REAL_ANSWER, "type": "text", "index": 1}]),
                ]
            )
        )
        assert _empty_deltas(events) == [], f"live no90vkf shape still fatal: {_contents(events)}"
        delivered = "".join(e.get("delta") or "" for e in _contents(events))
        assert REAL_ANSWER in delivered

    async def test_no_lifecycle_opened_for_empty_only_emits(self):
        """If every emitted chunk is empty, no TEXT_MESSAGE lifecycle may open
        at all — a START with no non-empty CONTENT is the other half of the
        broken wire shape (the old code opened the lifecycle on the FIRST
        chunk regardless of content)."""
        events = await _sse_from_run_events(
            _run_lifecycle([_manual_emit(""), _manual_emit(THINKING_BLOCK_PAYLOAD)])
        )
        starts = [e for e in events if str(e.get("type")) == "TEXT_MESSAGE_START"]
        assert starts == [], f"lifecycle opened for empty-only content: {starts}"

    async def test_lifecycle_opens_on_first_nonempty_chunk(self):
        """With leading empty chunks, the single lifecycle must open at the
        first real chunk and carry the full answer."""
        events = await _sse_from_run_events(
            _run_lifecycle(
                [
                    _manual_emit(""),
                    _manual_emit(THINKING_BLOCK_PAYLOAD),
                    _manual_emit(REAL_ANSWER),
                ]
            )
        )
        starts = [e for e in events if str(e.get("type")) == "TEXT_MESSAGE_START"]
        assert len(starts) == 1, f"expected exactly one lifecycle: {starts}"
        contents = _contents(events)
        assert contents and contents[0].get("delta") == REAL_ANSWER
        assert all(c.get("messageId") == starts[0].get("messageId") for c in contents)


class TestPassthroughGuard:
    """A TEXT_MESSAGE_CONTENT with empty delta arriving ALREADY-FORMED from the
    SDK layer (ag-ui-protocol 0.1.18, the image's pin, does NOT validate
    delta) must be dropped by execute()'s passthrough serialization — it is
    just as Zod-fatal as one our own conversion mints."""

    async def test_sdk_native_empty_content_event_is_dropped(self):
        from ag_ui.core import (
            EventType,
            RunFinishedEvent,
            RunStartedEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
        )

        run_events = [
            RunStartedEvent(type=EventType.RUN_STARTED, thread_id="t", run_id="r"),
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START, message_id="sdk-m1", role="assistant"
            ),
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT, message_id="sdk-m1", delta=""
            ),
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT, message_id="sdk-m1", delta=REAL_ANSWER
            ),
            TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id="sdk-m1"),
            RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id="t", run_id="r"),
        ]
        events = await _sse_from_run_events(run_events)
        assert _empty_deltas(events) == [], f"SDK-native empty delta passed through: {_contents(events)}"
        delivered = "".join(e.get("delta") or "" for e in _contents(events))
        assert REAL_ANSWER in delivered

    async def test_string_serialized_empty_content_event_is_dropped(self):
        """The str-event branch (pre-serialized JSON from the SDK) is the other
        passthrough door; same contract."""
        run_events = [
            json.dumps({"type": "RUN_STARTED", "threadId": "t", "runId": "r"}),
            json.dumps(
                {"type": "TEXT_MESSAGE_START", "messageId": "sdk-m2", "role": "assistant"}
            ),
            json.dumps({"type": "TEXT_MESSAGE_CONTENT", "messageId": "sdk-m2", "delta": ""}),
            json.dumps(
                {"type": "TEXT_MESSAGE_CONTENT", "messageId": "sdk-m2", "delta": REAL_ANSWER}
            ),
            json.dumps({"type": "TEXT_MESSAGE_END", "messageId": "sdk-m2"}),
            json.dumps({"type": "RUN_FINISHED", "threadId": "t", "runId": "r"}),
        ]
        events = await _sse_from_run_events(run_events)
        assert _empty_deltas(events) == [], f"str-branch empty delta passed through: {_contents(events)}"
        delivered = "".join(e.get("delta") or "" for e in _contents(events))
        assert REAL_ANSWER in delivered
