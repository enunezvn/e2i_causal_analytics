"""Unit tests for scripts/replay_golden_set.py.

Network-free: payload construction, JWT sub decoding, dry-run discipline,
and golden-dataset sanity. The live path is exercised manually (smoke run
--limit 2 against prod) per the plan's Task 8.
"""

from __future__ import annotations

import base64
import json

from scripts.replay_golden_set import build_chat_payload, jwt_sub, main


def _fake_jwt(sub: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode()).decode().rstrip("=")
    return f"{header}.{payload}."


def test_jwt_sub_decodes_unpadded_payload():
    assert jwt_sub(_fake_jwt("user-123")) == "user-123"


def test_build_chat_payload_shape():
    p = build_chat_payload("What drives Kisqali TRx?", "u1", "goldset-replay-20260715-q01")
    assert p == {
        "query": "What drives Kisqali TRx?",
        "user_id": "u1",
        "session_id": "goldset-replay-20260715-q01",
    }


def test_dry_run_sends_nothing(monkeypatch, capsys):
    import scripts.replay_golden_set as mod

    def _boom(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("network I/O attempted during --dry-run")

    monkeypatch.setattr(mod.urllib.request, "urlopen", _boom)
    rc = main(["--dry-run", "--limit", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("goldset-replay-") >= 3
    assert "3 questions" in out


def test_golden_dataset_has_30_nonempty_queries():
    """Guards accidental dataset edits — the replay banks on 30 real questions."""
    from src.rag.evaluation import get_default_evaluation_dataset

    samples = get_default_evaluation_dataset()
    assert len(samples) == 30
    assert all(s.query.strip() for s in samples)


def test_send_chat_never_raises_on_connection_reset(monkeypatch):
    import scripts.replay_golden_set as mod

    def _reset(*args, **kwargs):
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(mod.urllib.request, "urlopen", _reset)
    ok, detail = mod.send_chat("http://api.invalid", "tok", {"query": "q"}, 1)
    assert ok is False
    assert "ConnectionResetError" in detail


def test_main_reminting_on_401(monkeypatch):
    import scripts.replay_golden_set as mod

    minted = []

    def _mint():
        minted.append(f"t{len(minted) + 1}")
        return minted[-1]

    def _send(api_base, token, payload, timeout):
        if token == "t1":
            return False, "HTTP 401: token expired"
        return True, "ok"

    monkeypatch.setattr(mod, "mint_token", _mint)
    monkeypatch.setattr(mod, "jwt_sub", lambda token: "u1")
    monkeypatch.setattr(mod, "send_chat", _send)
    rc = mod.main(["--limit", "1", "--sleep", "0", "--target", "chat"])
    assert rc == 0
    assert minted == ["t1", "t2"]


# --- #1242: --target {chat,cognitive}, default cognitive -------------------
#
# The chat target feeds chatbot_training_signals (chatbot optimization; the
# Tier-5 feedback learner NEVER reads that table). Only the cognitive target
# (POST /api/cognitive/rag, 4-phase workflow, Phase-4 Reflector) writes the
# learning_signals dspy_signal rows the feedback learner consumes — so
# cognitive must be the default or the replay structurally starves the
# learner (the #1240 mechanism).


def test_build_cognitive_payload_shape():
    from scripts.replay_golden_set import build_cognitive_payload

    p = build_cognitive_payload("What drives Kisqali TRx?", "goldset-replay-20260715-q01")
    assert p == {
        "query": "What drives Kisqali TRx?",
        "conversation_id": "goldset-replay-20260715-q01",
    }


def test_default_target_is_cognitive(monkeypatch):
    """Without --target, main must route through send_cognitive (the learner path)."""
    import scripts.replay_golden_set as mod

    calls = {"cognitive": 0, "chat": 0}

    def _send_cognitive(api_base, token, payload, timeout):
        calls["cognitive"] += 1
        assert set(payload) == {"query", "conversation_id"}
        return True, "ok"

    def _send_chat(api_base, token, payload, timeout):  # pragma: no cover - must not run
        calls["chat"] += 1
        return True, "ok"

    monkeypatch.setattr(mod, "mint_token", lambda: "tok")
    monkeypatch.setattr(mod, "send_cognitive", _send_cognitive)
    monkeypatch.setattr(mod, "send_chat", _send_chat)
    rc = mod.main(["--limit", "2", "--sleep", "0"])
    assert rc == 0
    assert calls == {"cognitive": 2, "chat": 0}


def test_target_chat_still_available(monkeypatch):
    """--target chat keeps the chatbot-training replay path usable."""
    import scripts.replay_golden_set as mod

    calls = {"cognitive": 0, "chat": 0}

    def _send_cognitive(api_base, token, payload, timeout):  # pragma: no cover
        calls["cognitive"] += 1
        return True, "ok"

    def _send_chat(api_base, token, payload, timeout):
        calls["chat"] += 1
        assert set(payload) == {"query", "user_id", "session_id"}
        return True, "ok"

    monkeypatch.setattr(mod, "mint_token", lambda: "tok")
    monkeypatch.setattr(mod, "jwt_sub", lambda token: "u1")
    monkeypatch.setattr(mod, "send_cognitive", _send_cognitive)
    monkeypatch.setattr(mod, "send_chat", _send_chat)
    rc = mod.main(["--limit", "1", "--sleep", "0", "--target", "chat"])
    assert rc == 0
    assert calls == {"cognitive": 0, "chat": 1}


def test_send_cognitive_ok_requires_response_and_no_error(monkeypatch):
    import io

    import scripts.replay_golden_set as mod

    def _urlopen_factory(body: dict):
        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _urlopen(req, timeout=None):
            assert req.full_url.endswith("/cognitive/rag")
            return _Resp(json.dumps(body).encode())

        return _urlopen

    good = {"response": "Kisqali adoption rose...", "error": None, "latency_ms": 15500.0}
    monkeypatch.setattr(mod.urllib.request, "urlopen", _urlopen_factory(good))
    ok, detail = mod.send_cognitive("http://api.invalid", "tok", {"query": "q"}, 1)
    assert ok is True

    errored = {"response": "", "error": "workflow failed", "latency_ms": 10.0}
    monkeypatch.setattr(mod.urllib.request, "urlopen", _urlopen_factory(errored))
    ok, detail = mod.send_cognitive("http://api.invalid", "tok", {"query": "q"}, 1)
    assert ok is False
    assert "workflow failed" in detail


def test_send_cognitive_never_raises_on_connection_reset(monkeypatch):
    import scripts.replay_golden_set as mod

    def _reset(*args, **kwargs):
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(mod.urllib.request, "urlopen", _reset)
    ok, detail = mod.send_cognitive("http://api.invalid", "tok", {"query": "q"}, 1)
    assert ok is False
    assert "ConnectionResetError" in detail


def test_dry_run_names_cognitive_endpoint(monkeypatch, capsys):
    import scripts.replay_golden_set as mod

    def _boom(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("network I/O attempted during --dry-run")

    monkeypatch.setattr(mod.urllib.request, "urlopen", _boom)
    rc = main(["--dry-run", "--limit", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "/cognitive/rag" in out
