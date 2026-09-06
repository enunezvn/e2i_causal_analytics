"""
POST /api/chat/suggestions — shape, clipping, and failure contract.

The LLM call is monkeypatched at the module boundary (``get_fast_llm`` in
``src.api.routes.chat``) so these tests pin the route's wiring and its
fail-fast contract (502 on any generation/parsing failure — the frontend
falls back to static pills) without a network call. Prompt QUALITY is not
testable here; it was validated live against the real fast-tier model.
"""

import asyncio
import json

import pytest

import src.api.routes.chat as chat_module
from src.api.routes.chat import _parse_suggestions
from src.services import chat_capability_catalog as catalog_module

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


async def _fake_coverage():
    return [
        {"kpi_id": "WS3-BI-005", "brand": "", "points": 24},
        {"kpi_id": "WS3-BI-007", "brand": "Kisqali", "points": 24},
    ]


async def _fake_outcomes():
    return ["persistent_180d", "treatment_initiated", "roi", "adopted"]


def make_fake_catalog():
    return asyncio.run(
        catalog_module.build_capability_catalog(
            coverage_loader=_fake_coverage, outcomes_loader=_fake_outcomes
        )
    )


@pytest.fixture(autouse=True)
def _fake_catalog(monkeypatch):
    """Route tests never build the real catalog (Supabase). The fake is built
    with the module's own builder and injected loaders."""
    catalog = make_fake_catalog()

    async def _get():
        return catalog

    monkeypatch.setattr(chat_module, "get_capability_catalog", _get)
    return catalog


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
# OPENER MODE (empty transcript, pane just opened)
# =============================================================================


def test_opener_with_page_context_reaches_llm(test_client, auth_headers, monkeypatch):
    fake = _FakeLLM(content=_good_reply(4))
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: fake)

    payload = _payload(
        messages=[],
        page="/segment-analysis",
        page_context="Top responder segment: ECOG 0-1, CATE +0.21; overall ATE +0.12.",
    )
    resp = test_client.post("/api/chat/suggestions", json=payload, headers=auth_headers)

    assert resp.status_code == 200
    assert len(resp.json()["suggestions"]) == 4
    sent = fake.calls[0][1].content
    assert "ECOG 0-1" in sent
    assert "/segment-analysis" in sent


def test_opener_without_page_context_still_generates(test_client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        chat_module, "get_fast_llm", lambda **kwargs: _FakeLLM(content=_good_reply(3))
    )
    payload = _payload(messages=[])
    payload.pop("page_context", None)
    resp = test_client.post("/api/chat/suggestions", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()["suggestions"]) == 3


def test_oversized_page_context_is_422(test_client, auth_headers):
    payload = _payload(messages=[], page_context="x" * 4001)
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


# =============================================================================
# CONTENT-BLOCK REGRESSION (#1350/#1358 sweep)
# =============================================================================


def test_content_block_list_reply_still_generates(test_client, auth_headers, monkeypatch):
    """ChatAnthropic on adaptive-thinking models returns a block LIST.

    Pre-fix the route did ``str(reply.content)`` — stringifying the list into
    unparseable garbage → 502. The fast tier (haiku) returns str today, so
    this is the latent model-upgrade failure mode pinned as a regression.
    """
    blocks = [
        {"type": "thinking", "thinking": "chain of thought..."},
        {"type": "text", "text": _good_reply(4)},
    ]
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: _FakeLLM(content=blocks))

    resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)

    assert resp.status_code == 200
    assert [s["title"] for s in resp.json()["suggestions"]] == [f"Pill {i}" for i in range(4)]


# =============================================================================
# PILL BRAND GROUNDING (2026-08-19 review)
# =============================================================================


def test_prompt_requires_brand_in_pill_messages_when_filter_set():
    """A pill click sends ONLY the message text; a brand-less message hits the
    assistant's brand-clarification wall (measured 2026-08-19: filter set to
    Remibrutinib, chat answered "which brand?"). The prompt must require every
    pill MESSAGE to name the active brand when brand_filter is set — not use
    the filter as a mere tiebreaker — while brand="All" stays unconstrained.
    """
    prompt = chat_module._SYSTEM_PROMPT
    assert "brand_filter is set" in prompt
    assert "name that brand" in prompt
    assert '"All"' in prompt


# =============================================================================
# PROMPT TEMPLATE (2026-09-05 capability catalog)
# =============================================================================


def test_build_system_prompt_interpolates_catalog_and_route_hint():
    catalog = make_fake_catalog()
    prompt = chat_module.build_system_prompt(catalog, "/time-series")
    assert "{capability_catalog}" not in prompt and "{route_hint}" not in prompt
    assert "WHAT THE ASSISTANT CAN DO" in prompt
    assert "Total Prescriptions (TRx)" in prompt
    assert "persistent_180d" in prompt
    assert "The E2I system has" in prompt
    assert "PAGE HINT" in prompt
    assert catalog_module.ROUTE_HINTS["/time-series"] in prompt
    # the JSON output instruction survives the placeholder fill
    assert '{"suggestions": [{"title": "...", "message": "..."}, ...]}' in prompt


def test_build_system_prompt_omits_hint_block_for_unknown_page():
    catalog = make_fake_catalog()
    prompt = chat_module.build_system_prompt(catalog, "/nope")
    assert "PAGE HINT" not in prompt
    assert "\n\n\n" not in prompt


def test_prompt_tells_the_model_the_assistant_sees_page_content():
    """Part C of the design publishes page_content to the assistant, so the
    prompt must say pills MAY read on-screen values and must NOT extend them."""
    prompt = chat_module.build_system_prompt(make_fake_catalog(), "/")
    assert "ALSO shown to the assistant" in prompt
    assert "must NOT ask for anything beyond those literal values" in prompt
    assert "at least two different letters" in prompt


# =============================================================================
# ROUTE: catalog in the prompt, validator on the output
# =============================================================================


def test_llm_receives_catalog_and_route_hint(test_client, auth_headers, monkeypatch):
    fake = _FakeLLM(content=_good_reply(4))
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: fake)

    resp = test_client.post(
        "/api/chat/suggestions", json=_payload(page="/time-series"), headers=auth_headers
    )

    assert resp.status_code == 200
    system = fake.calls[0][0].content
    assert "WHAT THE ASSISTANT CAN DO" in system
    assert "Total Prescriptions (TRx)" in system
    assert "persistent_180d" in system
    assert "PAGE HINT" in system and "Time Series:" in system
    assert "{capability_catalog}" not in system


def test_unsupported_pills_are_dropped_and_logged(test_client, auth_headers, monkeypatch, caplog):
    reply = json.dumps(
        {
            "suggestions": [
                {"title": "TRx trend", "message": "Chart the TRx trend for Kisqali."},
                {
                    "title": "T-114",
                    "message": "Why did territory T-114 gain field force for Kisqali?",
                },
                {"title": "Drivers", "message": "What drives persistent_180d for Kisqali?"},
                {
                    "title": "Persistence rate",
                    "message": "Chart the persistent_180d rate for Kisqali by region.",
                },
            ]
        }
    )
    fake = _FakeLLM(content=reply)
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: fake)

    with caplog.at_level("INFO", logger="src.api.routes.chat"):
        resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)

    assert resp.status_code == 200
    assert [s["title"] for s in resp.json()["suggestions"]] == ["TRx trend", "Drivers"]
    dropped = [r.message for r in caplog.records if "chat suggestion dropped" in r.message]
    assert len(dropped) == 2
    assert any("rule=territory_detail" in m for m in dropped)
    assert any("rule=outcome_as_kpi:persistent_180d" in m for m in dropped)


def test_all_pills_dropped_returns_502(test_client, auth_headers, monkeypatch):
    reply = json.dumps(
        {
            "suggestions": [
                {"title": "SHAP", "message": "Which SHAP features drive Kisqali adoption?"}
            ]
        }
    )
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: _FakeLLM(content=reply))

    resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)

    assert resp.status_code == 502
    assert "no supported pills" in resp.json()["message"]


def test_fast_llm_gets_600_tokens(test_client, auth_headers, monkeypatch):
    seen = {}

    def _factory(**kwargs):
        seen.update(kwargs)
        return _FakeLLM(content=_good_reply(2))

    monkeypatch.setattr(chat_module, "get_fast_llm", _factory)
    test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)
    assert seen == {"max_tokens": 600, "timeout": 8}


def test_degraded_catalog_still_serves_pills(test_client, auth_headers, monkeypatch):
    """Both DB-backed catalog fields failed: the route still answers 200 with
    pills grounded in registry KPIs, the prompt says the data is unavailable,
    and the static validator rules still apply. Outcome-as-KPI pills are NOT
    dropped in this state (there are no known outcomes to match) - the
    accepted trade-off; the cache retries within 60 s."""

    async def _nothing():
        return []

    degraded = asyncio.run(
        catalog_module.build_capability_catalog(coverage_loader=_nothing, outcomes_loader=_nothing)
    )
    assert degraded.degraded == ("trend_coverage", "causal_outcomes")

    async def _get():
        return degraded

    monkeypatch.setattr(chat_module, "get_capability_catalog", _get)
    reply = json.dumps(
        {
            "suggestions": [
                {"title": "TRx trend", "message": "Chart the TRx trend for Kisqali."},
                {
                    "title": "T-114",
                    "message": "Why did territory T-114 gain field force for Kisqali?",
                },
                {
                    "title": "Persistence rate",
                    "message": "Chart the persistent_180d rate for Kisqali by region.",
                },
            ]
        }
    )
    fake = _FakeLLM(content=reply)
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: fake)

    resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)

    assert resp.status_code == 200
    assert "unavailable right now" in fake.calls[0][0].content
    titles = [s["title"] for s in resp.json()["suggestions"]]
    assert "T-114" not in titles  # static rule still applies
    assert titles == ["TRx trend", "Persistence rate"]  # outcome rule inert while outcomes unknown
