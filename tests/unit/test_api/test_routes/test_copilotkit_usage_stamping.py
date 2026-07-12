"""_persist_message_sync stamps the run's captured tokens/model onto
assistant rows (drain = read-and-reset — no double-count across persists),
and leaves user rows and drained runs unstamped (honest NULL)."""

from types import SimpleNamespace

import pytest

from src.utils.llm_attribution import clear_attribution, record_usage, set_chat_attribution

USER = "11111111-1111-1111-1111-111111111111"
SESSION = f"{USER}~conv-1"


@pytest.fixture()
def captured(monkeypatch):
    rows = []

    class _Table:
        def insert(self, data):
            rows.append(data)
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{"id": len(rows)}]))

    class _Client:
        def table(self, name):
            assert name == "chatbot_messages"
            return _Table()

    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: _Client())
    clear_attribution()
    return rows


def test_assistant_row_stamped_then_drained(captured):
    from src.api.routes.copilotkit import _persist_message_sync

    set_chat_attribution(SESSION, request_id="run-1")
    record_usage("claude-sonnet-4-6", 100, 50)

    _persist_message_sync(SESSION, "assistant", "answer one")
    assert captured[0]["tokens_used"] == 150
    assert captured[0]["model_used"] == "claude-sonnet-4-6"

    # accumulator drained: next assistant row must NOT repeat the tokens
    _persist_message_sync(SESSION, "assistant", "answer two")
    assert "tokens_used" not in captured[1]
    assert "model_used" not in captured[1]


def test_user_row_never_stamped(captured):
    from src.api.routes.copilotkit import _persist_message_sync

    set_chat_attribution(SESSION)
    record_usage("gpt-4o", 10, 5)
    _persist_message_sync(SESSION, "user", "question")
    assert "tokens_used" not in captured[0]
    # user persist must not consume the accumulator either
    _persist_message_sync(SESSION, "assistant", "answer")
    assert captured[1]["tokens_used"] == 15
