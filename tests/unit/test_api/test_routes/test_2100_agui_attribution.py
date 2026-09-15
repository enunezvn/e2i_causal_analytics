"""#2100: an AG-UI turn's attribution survives the keepalive wrapper's frame pulls.

``execute()`` calls ``set_chat_attribution`` before its first yield, so the write
lands in whichever task pulled that frame. ``with_sse_keepalive`` pulled every
frame in a *fresh* task, and a ``Task`` copies the context at creation and
discards its writes — so the four readers that run on a later pull
(``UsageRecorderCallback.on_llm_end``, ``record_litellm_success``,
``_ensure_conversation_exists`` and ``drain_run_usage``) all saw no attribution.
Live effect since 2026-08-16: no ``llm_usage_events`` row with ``surface='chat'``,
AG-UI conversations owned by the anonymous sentinel, and assistant rows with NULL
``tokens_used``.

These tests drive the REAL ``LangGraphAgent.execute()`` under the REAL
``with_sse_keepalive`` and assert on the side effects those readers produce. The
only doubles are the ones a unit test cannot avoid: the chat model (whose real
usage-capture callback is still the thing that reads the attribution), the
Supabase client, and the ``llm_usage_events`` queue sink.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, LLMResult

from src.api.routes import chatbot_tools, copilotkit
from src.api.utils.sse_keepalive import SSE_KEEPALIVE_FRAME, with_sse_keepalive
from src.utils.llm_attribution import (
    ANONYMOUS_USER_ID,
    clear_attribution,
    get_attribution,
    set_authenticated_user,
    set_chat_attribution,
)
from src.utils.llm_usage_callback import UsageRecorderCallback

USER = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
MODEL = "claude-2100-test"
ANSWER = "The system health score is 82."
QUESTION = "What is the health score?"
INPUT_TOKENS = 11
OUTPUT_TOKENS = 7


# ----------------------------------------------------------------- doubles


class _Result:
    def __init__(self, data: list[dict]) -> None:
        self.data = data


class _FakeTable:
    """Chainable stand-in for ``client.table(name)``.

    Reports "no rows" for every read (so ``_ensure_conversation_exists`` takes
    its create branch) and echoes back what was inserted, which is what the
    production code reads the row id from.
    """

    def __init__(self, name: str, inserts: dict[str, list[dict]]) -> None:
        self._name = name
        self._inserts = inserts
        self._pending: dict | None = None

    def insert(self, payload: dict, *args: Any, **kwargs: Any) -> "_FakeTable":
        self._pending = payload
        self._inserts.setdefault(self._name, []).append(payload)
        return self

    def select(self, *args: Any, **kwargs: Any) -> "_FakeTable":
        return self

    def update(self, *args: Any, **kwargs: Any) -> "_FakeTable":
        return self

    def eq(self, *args: Any, **kwargs: Any) -> "_FakeTable":
        return self

    def neq(self, *args: Any, **kwargs: Any) -> "_FakeTable":
        return self

    def is_(self, *args: Any, **kwargs: Any) -> "_FakeTable":
        return self

    def order(self, *args: Any, **kwargs: Any) -> "_FakeTable":
        return self

    def limit(self, *args: Any, **kwargs: Any) -> "_FakeTable":
        return self

    def execute(self) -> _Result:
        if self._pending is not None:
            row = dict(self._pending)
            row.setdefault("id", f"{self._name}-{len(self._inserts[self._name])}")
            self._pending = None
            return _Result([row])
        return _Result([])


class _FakeSupabase:
    def __init__(self) -> None:
        self.inserts: dict[str, list[dict]] = {}

    def table(self, name: str) -> _FakeTable:
        return _FakeTable(name, self.inserts)


def _llm_result() -> LLMResult:
    """The shape langchain hands ``on_llm_end`` after an aggregated stream."""
    message = AIMessage(
        content=ANSWER,
        usage_metadata={
            "input_tokens": INPUT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
        },
        response_metadata={"model_name": MODEL},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class _ScriptedChatModel:
    """Answers in text, then fires the REAL usage-capture callback.

    ``llm_factory`` attaches ``UsageRecorderCallback`` at model construction, so
    it runs inside whatever task drives the stream — which is the frame-pull
    task. The model is scripted; the reader under test is not.
    """

    def __init__(self, seen: dict[str, Any]) -> None:
        self._seen = seen
        self._recorder = UsageRecorderCallback(provider="anthropic", default_model=MODEL)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedChatModel":
        return _ScriptedChatModel(self._seen)

    async def astream(self, messages: Any, *args: Any, **kwargs: Any):
        # Read the per-run channels from inside the frame-pull task, where the
        # graph's nodes actually run.
        self._seen["session_var"] = copilotkit._session_id_context.get()
        self._seen["run_var"] = copilotkit._run_id_context.get()
        self._seen["raw_query_var"] = chatbot_tools._raw_user_query_context.get()
        yield AIMessageChunk(content=ANSWER)
        self._recorder.on_llm_end(_llm_result())


@pytest.fixture
def agui_turn(monkeypatch):
    """A real AG-UI agent wired to the doubles above; returns a driver function."""
    supabase = _FakeSupabase()
    usage_events: list[Any] = []
    seen: dict[str, Any] = {}

    monkeypatch.setattr(copilotkit, "get_chat_llm", lambda **kwargs: _ScriptedChatModel(seen))
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: supabase)
    monkeypatch.setattr("src.utils.llm_usage_callback.enqueue", usage_events.append)

    async def _no_learning_signal(**kwargs) -> None:
        return None

    monkeypatch.setattr(copilotkit, "_collect_copilot_learning_signal", _no_learning_signal)

    agent = copilotkit.LangGraphAgent(
        name="default",
        description="#2100",
        graph=copilotkit.e2i_chat_graph,
        graph_factory=copilotkit.create_e2i_chat_agent,
    )

    async def drive(thread: str) -> list[str]:
        frames: list[str] = []
        stream = with_sse_keepalive(
            agent.execute(
                thread_id=thread,
                state={},
                messages=[{"id": "u-2100", "role": "user", "content": QUESTION}],
                actions=[],
            )
        )
        async for frame in stream:
            frames.append(frame)
        return frames

    drive.supabase = supabase  # type: ignore[attr-defined]
    drive.usage_events = usage_events  # type: ignore[attr-defined]
    drive.seen = seen  # type: ignore[attr-defined]
    return drive


@pytest.fixture
def verified_user():
    """Bind the JWT-verified user in the REQUEST task, as the auth gate does."""
    clear_attribution()
    set_authenticated_user(USER)
    try:
        yield USER
    finally:
        set_authenticated_user(None)
        clear_attribution()


def _rows(supabase: _FakeSupabase, table: str) -> list[dict]:
    return supabase.inserts.get(table, [])


# ----------------------------------------------------------------- the defect


async def test_an_agui_turn_attributes_its_llm_usage_to_the_chat_user_and_session(
    agui_turn, verified_user
):
    """``llm_usage_events`` must carry surface 'chat' plus the user and thread.

    The callback runs on a later frame pull than the ``set_chat_attribution``
    call, so before the fix it saw ``None`` and wrote the honest-but-useless
    platform fallback (``surface='other'``, NULL user/session) — which is exactly
    what production has emitted since 2026-08-16.
    """
    thread = str(uuid4())

    await agui_turn(thread)

    assert len(agui_turn.usage_events) == 1, "the usage callback never fired"
    event = agui_turn.usage_events[0]
    assert event.surface == "chat"
    assert event.user_id == USER
    assert event.session_id == thread
    assert event.request_id is not None, "run_id lost with the attribution"


async def test_an_agui_turn_owns_its_new_conversation(agui_turn, verified_user):
    """``_ensure_conversation_exists`` reads the attribution for the owner."""
    thread = str(uuid4())

    await agui_turn(thread)

    conversations = _rows(agui_turn.supabase, "chatbot_conversations")
    assert len(conversations) == 1, conversations
    assert conversations[0]["session_id"] == thread
    assert conversations[0]["user_id"] == USER
    assert conversations[0]["user_id"] != ANONYMOUS_USER_ID


async def test_the_assistant_row_carries_the_runs_tokens_and_model(agui_turn, verified_user):
    """``drain_run_usage`` is the fourth reader: no attribution, no accumulator."""
    thread = str(uuid4())

    await agui_turn(thread)

    assistant = [
        r for r in _rows(agui_turn.supabase, "chatbot_messages") if r["role"] == "assistant"
    ]
    assert len(assistant) == 1, _rows(agui_turn.supabase, "chatbot_messages")
    assert assistant[0]["tokens_used"] == INPUT_TOKENS + OUTPUT_TOKENS
    assert assistant[0]["model_used"] == MODEL


# ------------------------------------------- the other vars the same fix revives


async def test_the_frame_context_carries_the_session_run_and_query_vars(agui_turn, verified_user):
    """Intended side effects: each revived var EQUALS its state-borne fallback.

    ``execute()`` writes ``session_id`` and ``run_id`` into both the contextvar
    and ``state``; #2064 made the graph read state because the var never
    arrived. Both channels must now agree, so state stays the authority and
    nothing downstream changes meaning.
    """
    thread = str(uuid4())

    frames = await agui_turn(thread)

    # RUN_STARTED carries execute()'s own run_id — the same value it wrote into
    # state. (Later frames carry ids the AG-UI SDK mints for itself.)
    run_ids = {
        payload["runId"]
        for payload in (_json_payload(f) for f in frames)
        if isinstance(payload, dict) and payload.get("type") == "RUN_STARTED"
    }
    assert len(run_ids) == 1, run_ids
    run_id = run_ids.pop()

    # var == the value execute() also put in state
    assert agui_turn.seen["session_var"] == thread
    assert agui_turn.seen["run_var"] == run_id
    assert agui_turn.seen["raw_query_var"] == QUESTION

    # and the row persisted through the state-borne fallback names the same run
    messages = _rows(agui_turn.supabase, "chatbot_messages")
    assert messages, "nothing persisted"
    assert {m["metadata"].get("run_id") for m in messages} == {run_id}


def _json_payload(frame: str) -> Any:
    """The JSON object inside an SSE ``data:`` line, or None."""
    for line in frame.splitlines():
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                return None
    return None


# ------------------------------------------------------ teardown under one context


async def _next_body_frame(stream: Any, timeout: float = 5.0) -> Any:
    """The next frame from ``stream`` that is not a keepalive heartbeat.

    A heartbeat is emitted whenever a pull outlives the interval, and a loaded
    box can make that happen before a frame the test is waiting for. Skipping
    them keeps the frame assertions from racing the clock. The heartbeat
    assertions in the test are gated on a pull that provably cannot finish, so
    they stay exact rather than going through here.

    Bounded on purpose: a body that stops yielding would otherwise be skipped
    over forever and surface only as the suite timeout killing this worker,
    which reports nothing about which frame never came.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    skipped = 0

    def _expired() -> AssertionError:
        return AssertionError(f"no body frame within {timeout}s ({skipped} keepalive(s) seen)")

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _expired()
        try:
            frame = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
        except TimeoutError as exc:
            raise _expired() from exc
        if frame != SSE_KEEPALIVE_FRAME:
            return frame
        skipped += 1


async def test_a_keepalive_then_an_early_disconnect_keep_the_shared_frame_context():
    """The two moments the wrapper re-enters the one context every pull shares.

    First a real keepalive emission: the wrapper times out, yields a heartbeat
    and goes back to waiting on the SAME pull, which then produces a frame. The
    attribution read on THAT frame is the one a heartbeat could have cost us.

    Then an early disconnect with a pull still OUTSTANDING — the case a
    disconnect between frames never reaches, because the wrapper has already
    nulled ``pending`` and its teardown cancels nothing. Here ``pending.cancel()``
    itself runs in the caller's context; the pending task then resumes in the
    shared frame context to receive its ``CancelledError`` and run the body's
    handlers, which is the last entry into that context and the place a
    ``RuntimeError: cannot enter context`` would surface. The wrapper's own
    ``await pending`` and ``aclose()`` execute in the consumer's context, which
    is why the assertions below inspect the abandoned task rather than the
    wrapper.

    Both halves are driven by events rather than wall-clock sleeps, so the pull
    is guaranteed pending rather than merely likely. The body stands in for
    ``execute()``; the wrapper is the real one.
    """
    thread = str(uuid4())
    resume = asyncio.Event()  # released by the consumer once a heartbeat arrives
    abandon = asyncio.Event()  # never set: this pull is walked away from
    reads: list[Any] = []
    body_saw: list[str] = []

    def _surface() -> Any:
        attribution = get_attribution()
        return None if attribution is None else attribution.surface

    async def body():
        set_chat_attribution(thread, "run-2100-keepalive")
        try:
            reads.append(_surface())
            yield "frame-0"
            await resume.wait()
            reads.append(_surface())  # the read AFTER a real keepalive emission
            yield "frame-1"
            await abandon.wait()
            yield "frame-2"  # pragma: no cover — the consumer never asks for it
        except asyncio.CancelledError:
            body_saw.append("cancelled")
            raise
        finally:
            body_saw.append("finalised")

    stream = with_sse_keepalive(body(), interval_seconds=0.01)
    assert await _next_body_frame(stream) == "frame-0"
    # The body now waits on an event nobody has set, so the pull CANNOT finish:
    # a heartbeat is the proof that a pull is outstanding, not merely slow.
    assert await stream.__anext__() == SSE_KEEPALIVE_FRAME
    resume.set()
    assert await _next_body_frame(stream) == "frame-1"
    # ... and the next pull, on an event never set, is the one abandoned below.
    assert await stream.__anext__() == SSE_KEEPALIVE_FRAME

    pulls = [
        task
        for task in asyncio.all_tasks()
        if getattr(task.get_coro(), "__qualname__", "") == "_pull_next"
    ]
    assert len(pulls) == 1, pulls
    pending_pull = pulls[0]
    assert not pending_pull.done()

    await asyncio.wait_for(stream.aclose(), timeout=5)

    assert pending_pull.done(), "the abandoned pull was left running"
    assert body_saw == ["cancelled", "finalised"], body_saw
    # Both reads are 'chat', and the second one happened after a heartbeat.
    assert reads == ["chat", "chat"], reads
    # The write stayed in the frame context's copy; the request task is clean.
    assert get_attribution() is None
