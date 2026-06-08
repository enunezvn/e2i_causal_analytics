"""Shard 05: shared DSPy LM configuration helper."""

from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy")


def test_ensure_dspy_configured_sets_lm(monkeypatch):
    from src.optimization import dspy_lm

    # Force a clean slate.
    dspy.settings.configure(lm=None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # no call made; just construction
    assert dspy_lm.ensure_dspy_configured() is True
    assert dspy.settings.lm is not None


def test_ensure_dspy_configured_is_idempotent(monkeypatch):
    from src.optimization import dspy_lm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    dspy_lm.ensure_dspy_configured()
    first = dspy.settings.lm
    dspy_lm.ensure_dspy_configured()
    assert dspy.settings.lm is first  # not reconfigured


def test_ensure_dspy_configured_returns_false_without_key(monkeypatch):
    from src.optimization import dspy_lm

    dspy.settings.configure(lm=None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert dspy_lm.ensure_dspy_configured() is False
