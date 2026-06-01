"""Opt-in marked-mock LLM contract — issue #606 item C.

The Tier 1-5 harness runs keyless (no LLM API key). LLM-dependent agents may
construct a clearly-MARKED mock LLM, but ONLY behind the explicit opt-in flag
``E2I_ALLOW_MOCK_LLM`` AND only when a real key is absent — production stays
fail-loud. These tests pin that contract for the shared util and for each wired
agent construction site.

Lives in test_utils (NOT test_<agent>/) so the per-agent autouse LLM-patching
conftests don't mask the real factory's fail-loud path. Pure construction — no
services / heavy ML.
"""

from __future__ import annotations

import asyncio

import pytest

from src.utils.mock_llm import (
    MOCK_MARKER,
    MarkedMockChatLLM,
    MarkedMockResponse,
    mock_llm_allowed,
)

_KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def _clear_keys(monkeypatch):
    for v in _KEY_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_mock_llm_allowed_flag(monkeypatch):
    monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
    assert mock_llm_allowed() is False
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", truthy)
        assert mock_llm_allowed() is True
    monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "0")
    assert mock_llm_allowed() is False


def test_marked_mock_chat_llm_marks_its_response():
    llm = MarkedMockChatLLM('{"ok": true}')
    resp = asyncio.run(llm.ainvoke("anything"))
    assert isinstance(resp, MarkedMockResponse)
    assert resp.content == '{"ok": true}'
    assert resp.response_metadata.get(MOCK_MARKER) is True
    assert resp.additional_kwargs.get(MOCK_MARKER) is True
    # chainable no-ops keep call sites working
    assert llm.bind(foo=1) is llm
    assert llm.with_structured_output(object) is llm


def test_design_reasoning_fails_loud_without_opt_in(monkeypatch):
    """No key + no opt-in flag -> construction still raises (prod fail-loud)."""
    _clear_keys(monkeypatch)
    monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
    from src.agents.experiment_designer.nodes.design_reasoning import DesignReasoningNode

    with pytest.raises(ValueError):
        DesignReasoningNode()


def test_design_reasoning_uses_marked_mock_with_opt_in(monkeypatch):
    """No key + E2I_ALLOW_MOCK_LLM=1 -> constructs with a MARKED mock LLM."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
    from src.agents.experiment_designer.nodes.design_reasoning import DesignReasoningNode

    node = DesignReasoningNode()
    assert isinstance(node.llm, MarkedMockChatLLM)
    assert isinstance(node.fallback_llm, MarkedMockChatLLM)
