"""LLMObservabilityService.llm_usage aggregation: summary, daily buckets,
per-user + per-session rollups (chat), platform grouping (NULL-user rows),
unpriced-model honesty."""

from types import SimpleNamespace

import pytest

from src.services.llm_observability_service import LLMObservabilityService

U1 = "11111111-1111-1111-1111-111111111111"
U2 = "22222222-2222-2222-2222-222222222222"
S1 = f"{U1}~conv-a"
S2 = f"{U1}~conv-b"
S3 = f"{U2}~conv-c"

USERS = [
    {"id": U1, "email": "alice@x.com"},
    {"id": U2, "email": "bob@x.com"},
]


def _ev(model, i, o, user=None, session=None, surface="chat", component=None, day="2026-07-10"):
    return {
        "created_at": f"{day}T12:00:00+00:00",
        "provider": "anthropic" if "claude" in model else "openai",
        "model": model,
        "input_tokens": i,
        "output_tokens": o,
        "surface": surface if user is None else "chat",
        "component": component,
        "user_id": user,
        "session_id": session,
    }


class _Query:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, _name):
        def _chain(*_a, **_k):
            return self

        return _chain

    def execute(self):
        return SimpleNamespace(data=self._data)


class _Client:
    """Scripted per-table responses, popped in call order."""

    def __init__(self, script):
        self._script = {k: list(v) for k, v in script.items()}

    def table(self, name):
        return _Query(self._script[name].pop(0))


def _service(events, conversations=None, first_event=None):
    script = {
        "llm_usage_events": [
            events,  # _fetch_events page 1 (< page size => single page)
            [first_event] if first_event else [],  # _tracking_since
        ],
        "chatbot_conversations": [conversations or []],
    }
    return LLMObservabilityService(client=_Client(script))


def test_aggregation_end_to_end():
    events = [
        _ev("claude-sonnet-4-6", 1000, 500, user=U1, session=S1),
        _ev("claude-sonnet-4-6", 2000, 1000, user=U1, session=S1, day="2026-07-11"),
        _ev("gpt-4o", 500, 250, user=U1, session=S2),
        _ev("gpt-4o-mini", 100, 50, user=U2, session=S3),
        _ev("gpt-4o", 4000, 2000, surface="insights", component="ExecutiveBrief"),
    ]
    convs = [{"session_id": S1, "title": "Kisqali TRx", "created_at": "2026-07-10T11:59:00+00:00"}]
    svc = _service(
        events, conversations=convs, first_event={"created_at": "2026-07-01T00:00:00+00:00"}
    )

    result = svc.llm_usage(30, USERS)

    s = result["summary"]
    assert s["calls"] == 5
    assert s["input_tokens"] == 7600
    assert s["output_tokens"] == 3800
    assert s["distinct_users"] == 2
    assert s["tracking_since"] == "2026-07-01T00:00:00+00:00"
    assert s["total_cost_usd"] > 0

    assert [d["date"] for d in result["daily"]] == ["2026-07-10", "2026-07-11"]
    assert result["daily"][0]["platform_cost_usd"] > 0
    assert result["daily"][1]["platform_cost_usd"] == 0

    by_user = {u["user_id"]: u for u in result["by_user"]}
    assert by_user[U1]["email"] == "alice@x.com"
    assert by_user[U1]["sessions"] == 2
    assert by_user[U1]["calls"] == 3
    assert "claude-sonnet-4-6" in by_user[U1]["models"]

    sessions_u1 = {r["session_id"]: r for r in result["sessions"][U1]}
    assert sessions_u1[S1]["title"] == "Kisqali TRx"
    assert sessions_u1[S1]["calls"] == 2
    assert sessions_u1[S2]["title"] is None

    assert len(result["platform"]) == 1
    p = result["platform"][0]
    assert (p["surface"], p["component"], p["model"]) == ("insights", "ExecutiveBrief", "gpt-4o")

    assert result["unpriced_models"] == []
    assert result["pricing_version"]


def test_unpriced_model_counted_but_not_costed():
    events = [_ev("mystery-lm-9", 1000, 1000, user=U1, session=S1)]
    svc = _service(events, first_event={"created_at": "2026-07-10T00:00:00+00:00"})
    result = svc.llm_usage(30, USERS)
    assert result["unpriced_models"] == ["mystery-lm-9"]
    assert result["summary"]["total_cost_usd"] is None  # all calls unpriced: honest null, not $0
    assert result["summary"]["input_tokens"] == 1000  # tokens still honest
    assert result["by_user"][0]["cost_usd"] is None  # honest "—", never $0
    assert result["sessions"][U1][0]["cost_usd"] is None


def test_mixed_priced_and_unpriced_costs_only_priced_rows():
    """Regression guard (#1211): with priced and unpriced calls mixed in one
    window, cost must equal the priced rows exactly (never None, never
    inflated) while calls/tokens honestly include both — at summary, per-user,
    and per-session levels."""
    events = [
        _ev("gpt-4o", 1000, 500, user=U1, session=S1),
        _ev("mystery-lm-9", 2000, 1000, user=U1, session=S1),
    ]
    svc = _service(events, first_event={"created_at": "2026-07-10T00:00:00+00:00"})
    result = svc.llm_usage(30, USERS)

    # gpt-4o only: (1000 * 2.50 + 500 * 10.00) / 1M
    expected = pytest.approx(0.0075)
    assert result["summary"]["total_cost_usd"] == expected
    assert result["summary"]["calls"] == 2
    assert result["summary"]["input_tokens"] == 3000
    assert result["summary"]["output_tokens"] == 1500
    assert result["unpriced_models"] == ["mystery-lm-9"]

    user_row = result["by_user"][0]
    assert user_row["cost_usd"] == expected
    assert user_row["calls"] == 2
    assert user_row["input_tokens"] == 3000

    session_row = result["sessions"][U1][0]
    assert session_row["cost_usd"] == expected
    assert session_row["calls"] == 2


def test_empty_window():
    svc = _service([], first_event=None)
    result = svc.llm_usage(7, USERS)
    assert result["summary"]["calls"] == 0
    assert result["summary"]["tracking_since"] is None
    assert result["by_user"] == []
    assert result["daily"] == []
    assert result["platform"] == []


def test_constructor_rejects_missing_client(monkeypatch):
    # get_supabase() returns None when Supabase is down; the constructor must
    # raise so the admin route's singleton getter never caches a dead service.
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: None)
    with pytest.raises(RuntimeError):
        LLMObservabilityService()
