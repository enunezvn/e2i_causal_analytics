"""Capture hooks: extract usage from LangChain LLMResult shapes and litellm
success payloads; enqueue with current attribution; zero-usage => no event."""

from types import SimpleNamespace

import src.utils.llm_usage_callback as cb_mod
from src.utils.litellm_usage_logger import record_litellm_success
from src.utils.llm_attribution import clear_attribution, drain_run_usage, set_chat_attribution
from src.utils.llm_usage_callback import UsageRecorderCallback, _extract_usage

USER = "11111111-1111-1111-1111-111111111111"


def setup_function(_fn):
    clear_attribution()


def _anthropic_stream_result():
    # langchain-anthropic 1.3.x aggregated stream: usage on message.usage_metadata
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 11, "output_tokens": 7},
        response_metadata={"model_name": "claude-sonnet-4-6"},
    )
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output=None)


def _openai_llm_output_result():
    # langchain-openai fallback shape: usage in llm_output.token_usage
    return SimpleNamespace(
        generations=[[SimpleNamespace(message=None)]],
        llm_output={
            "token_usage": {"prompt_tokens": 9, "completion_tokens": 3},
            "model_name": "gpt-4o",
        },
    )


def test_extract_usage_message_metadata():
    model, i, o = _extract_usage(_anthropic_stream_result(), "claude-sonnet-4-6")
    assert (model, i, o) == ("claude-sonnet-4-6", 11, 7)


def test_extract_usage_llm_output_fallback():
    model, i, o = _extract_usage(_openai_llm_output_result(), "gpt-4o")
    assert (model, i, o) == ("gpt-4o", 9, 3)


def test_extract_usage_empty_result_is_zero():
    empty = SimpleNamespace(generations=[], llm_output=None)
    model, i, o = _extract_usage(empty, "gpt-4o")
    assert (i, o) == (0, 0)
    assert model == "gpt-4o"


def test_callback_enqueues_with_attribution(monkeypatch):
    events = []
    monkeypatch.setattr(cb_mod, "enqueue", lambda e: events.append(e) or True)
    set_chat_attribution(f"{USER}~s1", request_id="run-1")

    cb = UsageRecorderCallback(provider="anthropic", default_model="claude-sonnet-4-6")
    cb.on_llm_end(_anthropic_stream_result())

    assert len(events) == 1
    ev = events[0]
    assert ev.user_id == USER
    assert ev.session_id == f"{USER}~s1"
    assert ev.surface == "chat"
    assert ev.request_id == "run-1"
    assert (ev.input_tokens, ev.output_tokens) == (11, 7)
    # accumulator updated for persist-time stamping
    drained = drain_run_usage()
    assert drained.input_tokens == 11 and drained.last_model == "claude-sonnet-4-6"


def test_callback_without_attribution_is_platform_row(monkeypatch):
    events = []
    monkeypatch.setattr(cb_mod, "enqueue", lambda e: events.append(e) or True)
    cb = UsageRecorderCallback(provider="openai", default_model="gpt-4o")
    cb.on_llm_end(_openai_llm_output_result())
    assert events[0].user_id is None
    assert events[0].session_id is None
    assert events[0].surface == "other"


def test_callback_zero_usage_no_event(monkeypatch):
    events = []
    monkeypatch.setattr(cb_mod, "enqueue", lambda e: events.append(e) or True)
    cb = UsageRecorderCallback(provider="openai", default_model="gpt-4o")
    cb.on_llm_end(SimpleNamespace(generations=[], llm_output=None))
    assert events == []  # never fabricate


def test_callback_never_raises(monkeypatch):
    def _boom(_e):
        raise RuntimeError("recorder exploded")

    monkeypatch.setattr(cb_mod, "enqueue", _boom)
    cb = UsageRecorderCallback(provider="openai", default_model="gpt-4o")
    cb.on_llm_end(_openai_llm_output_result())  # must not raise


def test_factory_attaches_callback_and_stream_usage(monkeypatch):
    """llm_factory must construct models with the capture callback (and
    stream_usage=True for OpenAI) — the whole point of the chokepoint."""
    import src.utils.llm_factory as factory

    captured = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
    factory._create_openai_llm("gpt-4o", 100, 0.3, None)
    assert captured["stream_usage"] is True
    assert any(isinstance(c, UsageRecorderCallback) for c in captured["callbacks"])


# ---------------------------------------------------------------- litellm ---


def _litellm_response(model="gpt-4o", prompt=9, completion=4):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def test_litellm_success_enqueues(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.services.llm_usage_recorder.enqueue", lambda e: events.append(e) or True
    )
    record_litellm_success({"model": "gpt-4o"}, _litellm_response())
    assert len(events) == 1
    assert events[0].provider == "openai"
    assert (events[0].input_tokens, events[0].output_tokens) == (9, 4)


def test_litellm_anthropic_provider_detection(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.services.llm_usage_recorder.enqueue", lambda e: events.append(e) or True
    )
    record_litellm_success(
        {"model": "anthropic/claude-sonnet-4-6"},
        _litellm_response(model="claude-sonnet-4-6"),
    )
    assert events[0].provider == "anthropic"


def test_litellm_cache_hit_skipped(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.services.llm_usage_recorder.enqueue", lambda e: events.append(e) or True
    )
    record_litellm_success({"model": "gpt-4o", "cache_hit": True}, _litellm_response())
    assert events == []  # cached replay spent no tokens


def test_litellm_zero_usage_skipped(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.services.llm_usage_recorder.enqueue", lambda e: events.append(e) or True
    )
    record_litellm_success({"model": "gpt-4o"}, SimpleNamespace(model="gpt-4o", usage=None))
    assert events == []


def test_litellm_never_raises(monkeypatch):
    def _boom(_e):
        raise RuntimeError("recorder exploded")

    monkeypatch.setattr("src.services.llm_usage_recorder.enqueue", _boom)
    record_litellm_success({"model": "gpt-4o"}, _litellm_response())  # must not raise
