"""Shard 05: shared DSPy LM configuration helper."""

from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy")


def test_ensure_dspy_configured_sets_lm(monkeypatch):
    from src.optimization import dspy_lm

    # Force a clean slate. Provider-aware: pin the provider to match the key set.
    dspy.settings.configure(lm=None)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # no call made; just construction
    assert dspy_lm.ensure_dspy_configured() is True
    assert dspy.settings.lm is not None


def test_ensure_dspy_configured_is_idempotent(monkeypatch):
    from src.optimization import dspy_lm

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    dspy_lm.ensure_dspy_configured()
    first = dspy.settings.lm
    dspy_lm.ensure_dspy_configured()
    assert dspy.settings.lm is first  # not reconfigured


def test_ensure_dspy_configured_returns_false_without_key(monkeypatch):
    from src.optimization import dspy_lm

    dspy.settings.configure(lm=None)
    # Provider is anthropic but its key is absent -> not configured (provider-aware).
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert dspy_lm.ensure_dspy_configured() is False


# ---------------------------------------------------------------------------
# Provider-aware model resolver (#ai-insights brief fix): DSPy must use the same
# working model as the rest of the app, never a hardcoded retired one.
# ---------------------------------------------------------------------------


def test_get_default_dspy_model_openai_default(monkeypatch):
    from src.optimization import dspy_lm

    monkeypatch.delenv("DSPY_LM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)  # conftest loads .env; isolate
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert dspy_lm.get_default_dspy_model() == "openai/gpt-5.6-terra"


def test_get_default_dspy_model_openai_honors_llm_model(monkeypatch):
    """LLM_MODEL pins the OpenAI model deployment-wide (model refresh 2026-07-18),
    mirroring llm_factory's override, so DSPy and LangChain paths stay on the
    same workhorse model."""
    from src.optimization import dspy_lm

    monkeypatch.delenv("DSPY_LM_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6-sol")
    assert dspy_lm.get_default_dspy_model() == "openai/gpt-5.6-sol"


def test_get_default_dspy_model_anthropic_uses_env_model(monkeypatch):
    from src.optimization import dspy_lm

    monkeypatch.delenv("DSPY_LM_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
    assert dspy_lm.get_default_dspy_model() == "anthropic/claude-opus-4-5-20251101"


def test_get_default_dspy_model_never_returns_retired_model(monkeypatch):
    """The retired model that 404'd the Executive AI Brief must never be the default."""
    from src.optimization import dspy_lm

    monkeypatch.delenv("DSPY_LM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    for provider in ("openai", "anthropic", ""):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        assert "claude-sonnet-4-20250514" not in dspy_lm.get_default_dspy_model()


def test_get_default_dspy_model_explicit_override_wins(monkeypatch):
    from src.optimization import dspy_lm

    monkeypatch.setenv("DSPY_LM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")  # ignored when explicit override present
    assert dspy_lm.get_default_dspy_model() == "openai/gpt-4o-mini"


def test_dspy_provider_api_key_present_is_provider_aware(monkeypatch):
    from src.optimization import dspy_lm

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert dspy_lm.dspy_provider_api_key_present() is True

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    # Now anthropic is the active provider but only the OpenAI key is set.
    assert dspy_lm.dspy_provider_api_key_present() is False
