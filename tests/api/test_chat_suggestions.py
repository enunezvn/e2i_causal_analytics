"""
POST /api/chat/suggestions — shape, clipping, and failure contract.

The LLM call is monkeypatched at the module boundary (``get_fast_llm`` in
``src.api.routes.chat``) so these tests pin the route's wiring and its
fail-fast contract (502 on any generation/parsing failure — the frontend
falls back to static pills) without a network call. Prompt QUALITY is not
testable here; it was validated live against the real fast-tier model.
"""

import json

import pytest

import src.api.routes.chat as chat_module
from src.api.routes.chat import _parse_suggestions

# =============================================================================
# FAKES
# =============================================================================


class _FakeReply:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str = "", exc: Exception | None = None) -> None:
        self._content = content
        self._exc = exc
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self._exc is not None:
            raise self._exc
        return _FakeReply(self._content)


def _payload(**overrides):
    payload = {
        "messages": [
            {"role": "user", "content": "What is the current TRx?"},
            {"role": "assistant", "content": "TRx is 11,634 this month."},
        ],
        "page": "/time-series",
        "brand": "Remibrutinib",
    }
    payload.update(overrides)
    return payload


def _good_reply(n: int = 4) -> str:
    return json.dumps(
        {"suggestions": [{"title": f"Pill {i}", "message": f"Question {i}?"} for i in range(n)]}
    )


# =============================================================================
# ROUTE CONTRACT
# =============================================================================


def test_returns_generated_pills(test_client, auth_headers, monkeypatch):
    fake = _FakeLLM(content=_good_reply(4))
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: fake)

    resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    assert [s["title"] for s in body["suggestions"]] == [f"Pill {i}" for i in range(4)]
    # The transcript and UI context must actually reach the LLM.
    sent = fake.calls[0][1].content
    assert "What is the current TRx?" in sent
    assert "/time-series" in sent
    assert "Remibrutinib" in sent


def test_clips_to_four_suggestions(test_client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        chat_module, "get_fast_llm", lambda **kwargs: _FakeLLM(content=_good_reply(9))
    )
    resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()["suggestions"]) == 4


def test_llm_failure_returns_502(test_client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_fast_llm",
        lambda **kwargs: _FakeLLM(exc=RuntimeError("model down")),
    )
    resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)
    assert resp.status_code == 502


def test_unusable_reply_returns_502(test_client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        chat_module, "get_fast_llm", lambda **kwargs: _FakeLLM(content="not json at all")
    )
    resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)
    assert resp.status_code == 502


def test_transcript_without_user_turn_is_422(test_client, auth_headers):
    payload = _payload(messages=[{"role": "assistant", "content": "How can I help you?"}])
    resp = test_client.post("/api/chat/suggestions", json=payload, headers=auth_headers)
    assert resp.status_code == 422


def test_oversized_transcript_is_422(test_client, auth_headers):
    payload = _payload(messages=[{"role": "user", "content": f"q{i}"} for i in range(13)])
    resp = test_client.post("/api/chat/suggestions", json=payload, headers=auth_headers)
    assert resp.status_code == 422


# =============================================================================
# PARSER
# =============================================================================


def test_parse_strips_markdown_fence():
    fenced = "```json\n" + _good_reply(2) + "\n```"
    assert len(_parse_suggestions(fenced)) == 2


def test_parse_accepts_bare_list():
    bare = json.dumps([{"title": "T", "message": "M?"}])
    assert _parse_suggestions(bare)[0].title == "T"


def test_parse_skips_malformed_items_and_truncates():
    reply = json.dumps(
        {
            "suggestions": [
                {"title": "  ", "message": "empty title"},
                {"title": "ok", "message": 42},
                "not a dict",
                {"title": "x" * 100, "message": "y" * 900},
            ]
        }
    )
    pills = _parse_suggestions(reply)
    assert len(pills) == 1
    assert len(pills[0].title) == 60
    assert len(pills[0].message) == 500


@pytest.mark.parametrize("raw", ["", "{}", '{"suggestions": []}', "[1, 2]"])
def test_parse_rejects_unusable_replies(raw):
    with pytest.raises(ValueError):
        _parse_suggestions(raw)
