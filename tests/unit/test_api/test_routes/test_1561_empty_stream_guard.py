"""#1561: /chat/stream zero-char HTTP-200 guard at the stream-writer seam.

Measured (2026-08-12 wave-2 post-deploy sweep, turn 5.1): HTTP 200, 27.1 s,
the generate node ran ~19 s, ZERO characters streamed to the client, no error
event anywhere in the envelope — the UI renders an empty assistant turn with
no indication anything failed. The solo retry was clean, so the upstream
emptiness is transient — but a zero-byte 200 must be impossible by
construction at the writer seam.

Scope pin (wave-1 finding): individual zero-char TEXT_MESSAGE lifecycles on
the AG-UI surface are BENIGN envelopes. This guard is about the TOTAL /chat
response text being empty at stream end, never about any single empty chunk.

These tests pin the guard in ``_stream_chat_response``:

1. A graph run that ends with no streamed text emits an honest explanatory
   ``text`` envelope (never a silent empty 200), flags
   ``dispatch_info.empty_response_fallback`` for recurrence counting, and
   still closes with ``done``.
2. Whitespace-only output counts as empty.
3. A normal answering turn is untouched — no fallback appended, flag False.
4. A #1336 bridge-authored answer streams as-is: the bridge fires INSIDE
   orchestrator_node on complete orchestrator failure and its text reaches
   this writer as ordinary ``response_text``, so guard and bridge are
   mutually exclusive — no masking, no double-wrap.
5. The guard emits no ``error`` event — the stream COMPLETED; the error
   channel stays reserved for exceptions (Finding 3 generic message).
"""

import json

import pytest
from langchain_core.messages import AIMessage

import src.api.routes.chatbot_graph as g
import src.api.routes.copilotkit as ck

# =============================================================================
# Helpers
# =============================================================================


async def _collect_events(request: ck.ChatRequest) -> list[dict]:
    events = []
    async for chunk in ck._stream_chat_response(request, "auth-user"):
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def _texts(events: list[dict]) -> list[str]:
    return [e["data"] for e in events if e["type"] == "text"]


def _dispatch(events: list[dict]) -> dict:
    dispatch = [e for e in events if e["type"] == "dispatch_info"]
    assert len(dispatch) == 1, f"expected exactly one dispatch_info, got {dispatch}"
    return dispatch[0]["data"]


def _request(qid: str) -> ck.ChatRequest:
    return ck.ChatRequest(query="what is TRx?", user_id="u", request_id=qid, session_id=f"s-{qid}")


# =============================================================================
# 1+5. Empty stream end -> explanatory envelope, flagged, no error event
# =============================================================================


class TestEmptyStreamGuard:
    @pytest.mark.asyncio
    async def test_empty_stream_emits_explanatory_envelope(self, monkeypatch):
        """The measured 5.1 shape: nodes ran, nothing textual reached the
        writer. The client must receive an honest fallback text envelope."""

        async def fake_stream(**kwargs):
            yield {"classify_intent": {"intent": "kpi_query", "intent_confidence": 0.9}}
            yield {"generate": {"response_text": ""}}
            yield {"finalize": {"response_text": ""}}

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        events = await _collect_events(_request("r-empty"))

        texts = _texts(events)
        assert texts, "silent empty 200: no text event reached the client"
        assert texts == [ck._EMPTY_STREAM_FALLBACK]
        # Recurrence must be countable (#1561 asks for zero-char visibility).
        assert _dispatch(events)["empty_response_fallback"] is True
        # The stream still closes properly.
        assert events[-1]["type"] == "done"
        # The stream COMPLETED — the error channel stays reserved for
        # exceptions (Finding 3), so no error event on the guard path.
        assert [e for e in events if e["type"] == "error"] == []

    @pytest.mark.asyncio
    async def test_empty_ai_message_stream_also_guarded(self, monkeypatch):
        """Empty AIMessage content (the messages channel) is also nothing."""

        async def fake_stream(**kwargs):
            yield {"generate": {"messages": [AIMessage(content="")]}}

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        events = await _collect_events(_request("r-empty-msg"))

        assert _texts(events) == [ck._EMPTY_STREAM_FALLBACK]
        assert _dispatch(events)["empty_response_fallback"] is True

    @pytest.mark.asyncio
    async def test_whitespace_only_stream_counts_as_empty(self, monkeypatch):
        """Whitespace chars stream (they are truthy), but a whitespace-only
        TOTAL is still an unanswered turn — the envelope must follow."""

        async def fake_stream(**kwargs):
            yield {"finalize": {"response_text": " \n "}}

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        events = await _collect_events(_request("r-ws"))

        assert _texts(events)[-1] == ck._EMPTY_STREAM_FALLBACK
        assert _dispatch(events)["empty_response_fallback"] is True

    def test_fallback_text_is_honest_not_an_answer(self):
        """The envelope explains a fault and invites retry; it must never be
        mistakable for an analytical answer (no fabricated content)."""
        low = ck._EMPTY_STREAM_FALLBACK.lower()
        assert "no response" in low or "no text" in low or "without producing" in low
        assert "again" in low  # invites retry


# =============================================================================
# 3. Normal answering turns are untouched
# =============================================================================


class TestNonEmptyStreamUntouched:
    @pytest.mark.asyncio
    async def test_answering_turn_gets_no_fallback(self, monkeypatch):
        async def fake_stream(**kwargs):
            yield {"finalize": {"response_text": "TRx for Kisqali is 12,867."}}

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        events = await _collect_events(_request("r-ok"))

        assert _texts(events) == ["TRx for Kisqali is 12,867."]
        assert _dispatch(events)["empty_response_fallback"] is False


# =============================================================================
# 4. #1336 bridge interaction: no masking, no double-wrap
# =============================================================================


class TestBridgePathNotDoubleWrapped:
    @pytest.mark.asyncio
    async def test_bridge_authored_answer_streams_verbatim(self, monkeypatch):
        """A bridge-authored turn (complete orchestrator failure rescued by
        the AG-UI brain, #1336) arrives at the writer as ordinary
        response_text — the guard must not fire on it."""
        bridge_text = (
            "Answered from live platform data, pulled through the analytics "
            "tools just now. The deeper multi-agent analysis did not run for "
            "this question.\n\nTRx for Kisqali is 12,867."
        )

        async def fake_stream(**kwargs):
            yield {
                "orchestrator": {
                    "response_text": bridge_text,
                    "orchestrator_used": True,
                    "agents_dispatched": ["causal_impact"],
                }
            }
            yield {"finalize": {"response_text": bridge_text}}

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        events = await _collect_events(_request("r-bridge"))

        assert _texts(events) == [bridge_text]
        assert _dispatch(events)["empty_response_fallback"] is False

    @pytest.mark.asyncio
    async def test_fail_closed_summary_not_masked(self, monkeypatch):
        """When the bridge itself fails, orchestrator_node keeps the #883
        fail-closed summary — real text, so the guard stays inert and the
        summary reaches the client unmasked."""

        async def fake_stream(**kwargs):
            yield {"finalize": {"response_text": "Analysis could not be completed."}}

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        events = await _collect_events(_request("r-failclosed"))

        assert _texts(events) == ["Analysis could not be completed."]
        assert _dispatch(events)["empty_response_fallback"] is False
