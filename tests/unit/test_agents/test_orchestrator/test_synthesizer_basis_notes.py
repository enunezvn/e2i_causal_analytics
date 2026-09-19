"""Standing notes survive every synthesis branch, and no LLM synthesis runs beside a
per-HCP cohort result (canonical TRx lane, codex r7-r9).

When the router sends cohort_profiler together with another agent (router.py:426-454),
SynthesizerNode would compose fresh LLM text, and orchestrator_tool forwards only the
synthesized text (chatbot_tools.py:1661/1696). r9: a detector over LLM wording can always
be bypassed (codex reproduced "Can<!--x-->onical" and "Market-wide"), so when any result
carries ``cohort_profile.basis_note`` the LLM is not called at all. The answer is the
deterministic composition of the agents' own narratives, and the notes are re-added
verbatim at the single exit.
"""

import inspect
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.agents.orchestrator.agent as orchestrator_agent_module
import src.api.routes.chatbot_tools as chatbot_tools
from src.agents.cohort_profiler import agent
from src.agents.orchestrator.nodes import synthesizer
from src.agents.orchestrator.nodes.synthesizer import (
    SynthesizerNode,
    append_basis_notes,
    strip_standing_notes,
)
from tests.unit.test_agents.test_orchestrator.test_cohort_profiler_panel_labeling import (
    _CANONICAL_THRESHOLD,
    _THRESHOLD_ROWS,
    _assert_never_presented_as_canonical,
)
from tests.unit.test_agents.test_orchestrator.test_cohort_profiler_volume_tiers import _Q15, _agent

_DISCLOSURE = agent.CANONICAL_GRAIN_DISCLOSURE
_REQUEST = agent.CANONICAL_REQUEST_SENTENCE
# What a fake LLM WOULD say: codex r9's bypasses of the r8 detector guard. It must never be asked.
_BYPASS_TEXTS = [
    "Can<!--x-->onical TRx for these HCPs is 1,240",
    "Market-wide TRx for these HCPs is 1,240",
]
_PANEL_FIGURES = "900 TRx Panel events combined"
_GAP_NARRATIVE = "Northeast Kisqali gap vs target is 12%."
_GAP_RESULT = {
    "agent_name": "gap_analyzer",
    "success": True,
    "result": {"narrative": _GAP_NARRATIVE, "recommendations": [], "confidence": 0.7},
    "error": None,
    "latency_ms": 900,
}
_CAUSAL_RESULT = {
    "agent_name": "causal_impact",
    "success": True,
    "result": {
        "narrative": "HCP engagement drives conversions.",
        "recommendations": [],
        "confidence": 0.87,
    },
    "error": None,
    "latency_ms": 1500,
}
_LLM_TEXT = "Synthesized response combining all agent insights."


async def _cohort_result(query: str = _Q15):
    """A REAL served cohort_profiler result, wrapped the way the dispatcher wraps it."""
    agent_, _db = _agent(db_rows=[_THRESHOLD_ROWS], today=date(2026, 8, 19))
    out = await agent_.analyze({"query": query})
    assert out["cohort_profile"]["basis_note"] == _DISCLOSURE
    return {
        "agent_name": "cohort_profiler",
        "success": True,
        "result": out,
        "error": None,
        "latency_ms": 1200,
    }


def _node(monkeypatch, *, llm_text=None, llm_error=None):
    monkeypatch.setattr(synthesizer, "_get_opik_connector", lambda: None)
    node = SynthesizerNode()
    if llm_error is not None:
        node.llm = MagicMock(ainvoke=AsyncMock(side_effect=llm_error))
    else:
        response = SimpleNamespace(content=llm_text, response_metadata={})
        node.llm = MagicMock(ainvoke=AsyncMock(return_value=response))
    return node


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [_Q15, _CANONICAL_THRESHOLD])
@pytest.mark.parametrize("would_say", _BYPASS_TEXTS)
async def test_no_llm_synthesis_runs_beside_a_cohort_result(monkeypatch, query, would_say):
    node = _node(monkeypatch, llm_text=would_say)
    state = await node.execute({"agent_results": [await _cohort_result(query), _GAP_RESULT]})
    node.llm.ainvoke.assert_not_called()  # the channel is removed, not policed
    text = state["synthesized_response"]
    assert "1,240" not in text
    assert would_say not in text
    assert "Analysis Summary" in text  # the deterministic composition
    assert _PANEL_FIGURES in text  # grounded panel figures from the cohort narrative
    assert _GAP_NARRATIVE in text  # and the other agent's own narrative
    assert text.count(_DISCLOSURE) == 1
    assert text.count(_REQUEST) == (1 if query == _CANONICAL_THRESHOLD else 0)
    assert state["response_confidence"] == 0.8  # recomposed like the fallback: mean of 0.9 and 0.7
    _assert_never_presented_as_canonical(text)


@pytest.mark.asyncio
async def test_multi_agent_synthesis_without_a_cohort_result_still_uses_the_llm(monkeypatch):
    node = _node(monkeypatch, llm_text=_LLM_TEXT)
    state = await node.execute({"agent_results": [_CAUSAL_RESULT, _GAP_RESULT]})
    node.llm.ainvoke.assert_awaited_once()
    assert state["synthesized_response"] == _LLM_TEXT


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [_Q15, _CANONICAL_THRESHOLD])
async def test_single_result_branch_keeps_exactly_one_copy_of_each_note(monkeypatch, query):
    node = _node(monkeypatch, llm_text="unused")
    state = await node.execute({"agent_results": [await _cohort_result(query)]})
    node.llm.ainvoke.assert_not_awaited()
    text = state["synthesized_response"]
    assert text.count(_DISCLOSURE) == 1
    assert text.count(_REQUEST) == (1 if query == _CANONICAL_THRESHOLD else 0)


def test_note_helpers_are_verbatim_once_in_result_order():
    a = {"result": {"cohort_profile": {"canonical_request_note": "ask A", "basis_note": "note A"}}}
    b = {"result": {"cohort_profile": {"basis_note": "note B"}}}
    duplicate = {"result": {"cohort_profile": {"basis_note": "note A"}}}
    plain = {"result": {"narrative": "no profile"}}
    assert (
        append_basis_notes("text", [a, plain, b, duplicate])
        == "text\n\n_ask A_\n\n_note A_\n\n_note B_"
    )
    assert append_basis_notes("text already says note B", [b]) == "text already says note B"
    assert append_basis_notes("text", [plain, {"result": None}]) == "text"
    assert strip_standing_notes("ask A\n\n_note A_\n\nfigures", ["ask A", "note A"]) == "figures"


@pytest.mark.asyncio
@pytest.mark.parametrize("would_say", _BYPASS_TEXTS)
async def test_orchestrator_tool_payload_carries_the_composition_not_llm_text(
    monkeypatch, would_say
):
    """The seam the chat model receives. The graph is stubbed at ``orchestrator.run``, whose
    only synthesis mapping is agent.py's ``"response_text": state.get("synthesized_response", "")``
    (pinned below). The REAL SynthesizerNode produces that state and the REAL tool builds the
    payload."""
    source = inspect.getsource(orchestrator_agent_module)
    assert '"response_text": state.get("synthesized_response", "")' in source
    node = _node(monkeypatch, llm_text=would_say)
    cohort = await _cohort_result(_CANONICAL_THRESHOLD)

    async def _run(*_args, **_kwargs):
        state = await node.execute({"agent_results": [cohort, _GAP_RESULT]})
        return {
            "response_text": state["synthesized_response"],
            "response_confidence": state["response_confidence"],
            "agents_dispatched": ["cohort_profiler", "gap_analyzer"],
            "status": "completed",
        }

    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(side_effect=_run)
    monkeypatch.setattr(chatbot_tools, "get_orchestrator", lambda: orchestrator)
    payload = await chatbot_tools.orchestrator_tool.ainvoke(
        {"query": "Profile HCPs above 50 TRx and the regional gaps for Kisqali", "brand": "Kisqali"}
    )
    orchestrator.run.assert_awaited_once()
    node.llm.ainvoke.assert_not_called()
    assert payload["success"] is True
    response = payload["response"]
    assert "1,240" not in response
    assert would_say not in response
    assert _PANEL_FIGURES in response
    assert response.count(_DISCLOSURE) == 1
    assert response.count(_REQUEST) == 1
    _assert_never_presented_as_canonical(response)
