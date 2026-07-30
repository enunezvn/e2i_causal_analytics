"""Opt-in marked-mock LLM contract — issue #606 item C.

The Tier 1-5 harness runs keyless (no LLM API key). LLM-dependent agents may
construct a clearly-MARKED mock LLM, but ONLY behind the explicit opt-in flag
``E2I_ALLOW_MOCK_LLM`` AND only when a real key is absent — production stays
fail-loud. These tests pin that contract for the shared util and for each wired
agent construction site.

Lives in test_utils (NOT test_<agent>/) so the per-agent autouse LLM-patching
conftests don't interfere. The "no key" condition is simulated DETERMINISTICALLY
by patching the relevant ``get_*_llm`` factory to raise ``ValueError`` (what it
does on a missing key) — so the tests don't depend on ambient env keys and are
xdist-safe. Pure construction — no services / heavy ML.
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


def _raise_no_key(*_a, **_k):
    raise ValueError("OPENAI_API_KEY environment variable is not set")


# ---------------------------------------------------------------------------
# Shared util
# ---------------------------------------------------------------------------


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
    assert llm.bind(foo=1) is llm
    assert llm.with_structured_output(object) is llm


def test_marked_mock_phase_responses_select_by_message_content():
    """Multi-phase agents (tool_composer) get a phase-appropriate payload chosen
    from the message text; first matching keyword wins, else the default."""

    class _Msg:
        def __init__(self, content):
            self.content = content

    llm = MarkedMockChatLLM(
        "DEFAULT",
        phase_responses=[
            ("decomposition", "DECOMP"),
            ("synth", "SYNTH"),
            ("planning", "PLAN"),
            ("tool", "PLAN"),
        ],
    )
    assert asyncio.run(llm.ainvoke([_Msg("Decomposition system prompt")])).content == "DECOMP"
    # "synth" must win over "tool" even though synthesis prompts mention tools
    assert asyncio.run(llm.ainvoke([_Msg("Synthesis: combine the tools")])).content == "SYNTH"
    assert asyncio.run(llm.ainvoke([_Msg("Planning: map tools")])).content == "PLAN"
    assert asyncio.run(llm.ainvoke([_Msg("nothing relevant")])).content == "DEFAULT"


# ---------------------------------------------------------------------------
# experiment_designer / design_reasoning
# ---------------------------------------------------------------------------


def test_design_reasoning_fails_loud_without_opt_in(monkeypatch):
    import src.agents.experiment_designer.nodes.design_reasoning as dr

    monkeypatch.setattr(dr, "get_chat_llm", _raise_no_key)
    monkeypatch.setattr(dr, "get_fast_llm", _raise_no_key)
    monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
    with pytest.raises(ValueError):
        dr.DesignReasoningNode()


def test_design_reasoning_uses_marked_mock_with_opt_in(monkeypatch):
    import src.agents.experiment_designer.nodes.design_reasoning as dr

    monkeypatch.setattr(dr, "get_chat_llm", _raise_no_key)
    monkeypatch.setattr(dr, "get_fast_llm", _raise_no_key)
    monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
    node = dr.DesignReasoningNode()
    assert isinstance(node.llm, MarkedMockChatLLM)
    assert isinstance(node.fallback_llm, MarkedMockChatLLM)


# ---------------------------------------------------------------------------
# orchestrator / intent_classifier + synthesizer
# ---------------------------------------------------------------------------


def test_orchestrator_intent_classifier_keyless(monkeypatch):
    import src.agents.orchestrator.nodes.intent_classifier as ic

    monkeypatch.setattr(ic, "get_fast_llm", _raise_no_key)

    monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
    with pytest.raises(ValueError):
        ic.IntentClassifierNode()

    monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
    assert isinstance(ic.IntentClassifierNode().llm, MarkedMockChatLLM)


def test_orchestrator_synthesizer_keyless(monkeypatch):
    import src.agents.orchestrator.nodes.synthesizer as syn

    monkeypatch.setattr(syn, "get_fast_llm", _raise_no_key)

    monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
    with pytest.raises(ValueError):
        syn.SynthesizerNode()

    monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
    assert isinstance(syn.SynthesizerNode().llm, MarkedMockChatLLM)


# ---------------------------------------------------------------------------
# tool_composer (lazy client init in _ensure_composer; imports factory inside)
# ---------------------------------------------------------------------------


def test_tool_composer_keyless_ensure_composer(monkeypatch):
    # #1365: tool_composer no longer PRE-BUILDS one shared get_standard_llm()
    # client — when a key is present it leaves llm_client=None so ToolComposer
    # builds a correctly-sized client PER PHASE. So "no key" is now detected
    # directly from the provider key env var (ToolComposerAgent._provider_key_
    # present), simulated here by removing it, not by patching get_standard_llm.
    from src.agents.tool_composer.agent import ToolComposerAgent

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
    with pytest.raises(RuntimeError, match="requires an LLM client"):
        ToolComposerAgent()._ensure_composer()

    monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
    agent = ToolComposerAgent()
    agent._ensure_composer()
    assert isinstance(agent.llm_client, MarkedMockChatLLM)
