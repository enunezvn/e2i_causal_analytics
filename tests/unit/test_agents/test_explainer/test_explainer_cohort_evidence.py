"""The explainer explains a per-HCP cohort result deterministically (canonical TRx lane, codex r10).

The explainer is the only agent that CONSUMES another agent's result as input: the
dispatcher binds same-turn results into ``analysis_results`` (dispatcher.py
``_resolve_explainer_input`` branch 2a, flattened by ``_successful_results``), and the
router's generic multi-intent fallback (router.py:426-454) can dispatch cohort_profiler
together with explainer. With a cohort result in its evidence the explainer runs the
use_llm=False graph, which makes no LLM call, whatever the explicit setting or
auto-detect says.
"""

import logging
import re
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.api.routes.chatbot_tools as chatbot_tools
from src.agents.cohort_profiler import agent as cohort_agent
from src.agents.cohort_profiler.ask import mentions_canonical
from src.agents.cohort_profiler.notes import (
    cohort_key_findings,
    has_cohort_evidence,
    standing_notes,
)
from src.agents.explainer import ExplainerAgent
from src.agents.explainer import agent as explainer_module
from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes import synthesizer
from src.agents.orchestrator.nodes.dispatcher import _successful_results
from src.agents.orchestrator.nodes.synthesizer import SynthesizerNode
from tests.unit.test_agents.test_orchestrator.test_cohort_profiler_panel_labeling import (
    _CANONICAL_THRESHOLD,
    _THRESHOLD_ROWS,
    _assert_never_presented_as_canonical,
)
from tests.unit.test_agents.test_orchestrator.test_cohort_profiler_volume_tiers import (
    _ALL_BRANDS_90D,
    _KISQALI_NRX,
    _Q15,
    _Q43_BRANDED,
    _agent,
    _RecordingCalc,
)

_DISCLOSURE = cohort_agent.CANONICAL_GRAIN_DISCLOSURE
# What a spy LLM WOULD say: the codex r9 detector bypasses. No LLM may be asked.
_BYPASS_TEXTS = [
    "Can<!--x-->onical TRx for these HCPs is 1,240",
    "Market-wide TRx for these HCPs is 1,240",
]
_OVERRIDE_REASON = "cohort_evidence_deterministic"


def _spy_llm(would_say: str):
    """An LLM that records every use. ``model`` is a real string because deep_reasoner reads it
    into ``model_used`` (measured: a MagicMock ``model`` fails ExplainerOutput validation)."""
    llm = MagicMock()
    llm.model = "spy-llm"
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content=would_say, response_metadata={}))
    return llm


async def _cohort_agent_result():
    """A REAL served cohort_profiler result, wrapped the way the dispatcher wraps it."""
    agent_, _db = _agent(db_rows=[_THRESHOLD_ROWS], today=date(2026, 8, 19))
    out = await agent_.analyze({"query": _CANONICAL_THRESHOLD})
    assert out["cohort_profile"]["basis_note"] == _DISCLOSURE
    return {
        "agent_name": "cohort_profiler",
        "success": True,
        "result": out,
        "error": None,
        "latency_ms": 1200,
    }


def _record_graph_modes(explainer, modes, *, stub=False):
    original = explainer._get_graph

    def _get_graph(use_llm):
        modes.append(use_llm)
        if stub:
            return SimpleNamespace(ainvoke=AsyncMock(return_value={"status": "completed"}))
        return original(use_llm)

    explainer._get_graph = _get_graph


def test_the_shared_rule_reads_both_result_shapes():
    wrapped = {
        "result": {"cohort_profile": {"basis_note": "note A", "canonical_request_note": "ask A"}}
    }
    flattened = {"cohort_profile": {"basis_note": "note B"}}
    assert standing_notes([wrapped, flattened, {"result": None}, {"narrative": "x"}]) == [
        "ask A",
        "note A",
        "note B",
    ]
    assert has_cohort_evidence([flattened]) is True
    assert (
        has_cohort_evidence([{"findings": ["no cohort"]}, {"result": {"narrative": "x"}}]) is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("would_say", _BYPASS_TEXTS)
async def test_explicit_llm_mode_is_overridden_by_cohort_evidence(caplog, would_say):
    caplog.set_level(logging.INFO, logger=explainer_module.logger.name)
    spy = _spy_llm(would_say)
    explainer = ExplainerAgent(use_llm=True, llm=spy)
    modes = []
    _record_graph_modes(explainer, modes)
    evidence = _successful_results(
        [await _cohort_agent_result()]
    )  # the dispatcher's real flattening
    out = await explainer.explain(analysis_results=evidence, query="Explain this HCP cohort")
    assert spy.ainvoke.await_count == 0
    assert spy.mock_calls == []
    assert modes == [False]
    assert out.status == "completed"
    assert out.model_used == "deterministic"
    assert any(_OVERRIDE_REASON in r.getMessage() for r in caplog.records)
    assert "1,240" not in out.executive_summary
    assert would_say not in out.executive_summary


@pytest.mark.asyncio
@pytest.mark.parametrize("would_say", _BYPASS_TEXTS)
async def test_cohort_to_explainer_to_synthesizer_to_tool_payload_never_serves_llm_text(
    monkeypatch, would_say
):
    """Real cohort agent -> the dispatcher's real flattening -> real ExplainerAgent -> real
    SynthesizerNode -> real orchestrator_tool payload. The graph is stubbed only at
    ``orchestrator.run``, whose synthesis mapping (agent.py:423) the synthesizer test pins."""
    monkeypatch.setattr(synthesizer, "_get_opik_connector", lambda: None)
    cohort = await _cohort_agent_result()
    explainer_llm = _spy_llm(would_say)
    explainer = ExplainerAgent(use_llm=True, llm=explainer_llm)
    explained = await explainer.explain(
        analysis_results=_successful_results([cohort]), query="Explain this HCP cohort"
    )
    explainer_result = {
        "agent_name": "explainer",
        "success": True,
        "result": explained.model_dump(),
        "error": None,
        "latency_ms": 800,
    }
    synth = SynthesizerNode()
    synth_llm = _spy_llm(would_say)
    synth.llm = synth_llm

    async def _run(*_args, **_kwargs):
        state = await synth.execute({"agent_results": [cohort, explainer_result]})
        return {
            "response_text": state["synthesized_response"],
            "response_confidence": state["response_confidence"],
            "agents_dispatched": ["cohort_profiler", "explainer"],
            "status": "completed",
        }

    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(side_effect=_run)
    monkeypatch.setattr(chatbot_tools, "get_orchestrator", lambda: orchestrator)
    payload = await chatbot_tools.orchestrator_tool.ainvoke(
        {"query": "Profile HCPs above 50 TRx and explain the cohort", "brand": "Kisqali"}
    )
    assert explainer_llm.ainvoke.await_count == 0
    assert synth_llm.ainvoke.await_count == 0
    assert payload["success"] is True
    response = payload["response"]
    assert "1,240" not in response
    assert would_say not in response
    assert response.count(_DISCLOSURE) == 1
    _assert_never_presented_as_canonical(response)


@pytest.mark.asyncio
async def test_without_cohort_evidence_the_llm_mode_is_still_selected(caplog):
    caplog.set_level(logging.INFO, logger=explainer_module.logger.name)
    explainer = ExplainerAgent(use_llm=True, llm=_spy_llm("unused"))
    modes = []
    _record_graph_modes(explainer, modes, stub=True)
    out = await explainer.explain(
        analysis_results=[{"findings": ["Kisqali Northeast gap vs target is 12%"]}],
        query="Explain the gap",
    )
    assert modes == [True]
    assert out.status == "completed"
    assert not any(_OVERRIDE_REASON in r.getMessage() for r in caplog.records)


# ------------------------------------------ codex r11: substantive deterministic explanation

_FOLLOW_UP = "Explain that cohort"
_NO_FINDINGS_HUSK = re.compile(r"(?<!\d)0 key finding")
_TIER_FIGURES = ("3,427", "1,266", "Remibrutinib", "2026-04-01")
_SHAPE_FIGURES = {
    "hcp_threshold": ["12 HCPs", "900 events combined", "top prescriber 120 events", "WS3-BI-011"],
    "hcp_volume_tiers": ["Remibrutinib", "2026-04-01", "3,427 HCPs", "1,266 HCPs", "WS3-BI-011"],
    "patient_legacy": ["Kisqali", "3,256 new-Rx patients"],
    "patient_criteria": ["Kisqali", "100 new-Rx patients", "moderate 40", "severe 60"],
}


async def _cohort_shapes():
    """The four real cohort_profile shapes the agent emits (cohort_profiler/agent.py return sites)."""
    shapes = {}
    agent_, _db = _agent(db_rows=[_THRESHOLD_ROWS], today=date(2026, 8, 19))
    shapes["hcp_threshold"] = await agent_.analyze({"query": _Q15})
    agent_, _db = _agent(db_rows=[_ALL_BRANDS_90D], today=date(2026, 8, 19))
    shapes["hcp_volume_tiers"] = await agent_.analyze({"query": _Q43_BRANDED})
    agent_, _db = _agent(calc=_RecordingCalc(_KISQALI_NRX), today=date(2026, 8, 19))
    shapes["patient_legacy"] = await agent_.analyze({"query": "Profile the Kisqali patient cohort"})
    rows = [
        {"nrx": 40, "severity": "moderate", "therapy_line": "1L"},
        {"nrx": 60, "severity": "severe", "therapy_line": "2L"},
    ]
    agent_, _db = _agent(db_rows=[rows], today=date(2026, 8, 19))
    shapes["patient_criteria"] = await agent_.analyze(
        {"query": "Profile Kisqali patients older than 64"}
    )
    return shapes


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(_SHAPE_FIGURES))
async def test_cohort_key_findings_carry_each_shapes_own_figures(shape):
    out = (await _cohort_shapes())[shape]
    assert out["status"] == "completed"
    findings = cohort_key_findings(out)
    assert findings and findings == cohort_key_findings(out)  # non-empty and deterministic
    text = "\n".join(findings)
    for figure in _SHAPE_FIGURES[shape]:
        assert figure in text, (figure, findings)
    for finding in findings:
        assert not mentions_canonical(finding), finding
        if "trx" in finding.lower() and re.search(r"\d", finding):
            assert "TRx Panel" in finding, finding


@pytest.mark.asyncio
async def test_explainer_only_follow_up_on_carried_cohort_evidence_narrates_the_panel_figures(
    monkeypatch,
):
    """Branch 2c. A later turn asks only to explain the earlier cohort. The carried prior-turn result
    reaches the explainer through the REAL resolver; the explainer is forced deterministic and
    narrates the cohort's own figures. The operator.add ``agent_results`` channel
    (orchestrator/state.py:233) that the resolver read from also hands the synthesizer that
    cohort result, so the disclosure is appended once."""
    monkeypatch.setattr(synthesizer, "_get_opik_connector", lambda: None)
    agent_, _db = _agent(db_rows=[_ALL_BRANDS_90D], today=date(2026, 8, 19))
    tiers = await agent_.analyze({"query": _Q43_BRANDED})
    prior = {
        "agent_name": "cohort_profiler",
        "success": True,
        "result": tiers,
        "error": None,
        "latency_ms": 1200,
    }
    agent_input = {
        "query": _FOLLOW_UP,
        "session_id": "sess-r11",
        "user_context": {},
        "parsed_query": {"entities": []},
        "agent_results": [prior],
    }
    dispatch = {
        "agent_name": "explainer",
        "priority": "high",
        "parameters": {},
        "timeout_ms": 15000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }
    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, dispatch)
    assert isinstance(resolved, dict), resolved
    assert (
        resolved["analysis_results"][0]["cohort_profile"]["cohort_size"] == 3427
    )  # branch 2c bound it

    explainer_llm = _spy_llm(_BYPASS_TEXTS[0])
    explained = await ExplainerAgent(use_llm=True, llm=explainer_llm).explain(**resolved)
    narrative = f"{explained.executive_summary}\n{explained.detailed_explanation}"
    for figure in _TIER_FIGURES:
        assert figure in narrative, (figure, narrative)
    assert not _NO_FINDINGS_HUSK.search(narrative), narrative

    explainer_result = {
        "agent_name": "explainer",
        "success": True,
        "result": explained.model_dump(),
        "error": None,
        "latency_ms": 800,
    }
    synth = SynthesizerNode()
    synth_llm = _spy_llm(_BYPASS_TEXTS[0])
    synth.llm = synth_llm

    async def _run(*_args, **_kwargs):
        state = await synth.execute({"agent_results": [prior, explainer_result]})
        return {
            "response_text": state["synthesized_response"],
            "response_confidence": state["response_confidence"],
            "agents_dispatched": ["explainer"],
            "status": "completed",
        }

    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(side_effect=_run)
    monkeypatch.setattr(chatbot_tools, "get_orchestrator", lambda: orchestrator)
    payload = await chatbot_tools.orchestrator_tool.ainvoke(
        {"query": _FOLLOW_UP, "brand": "Remibrutinib"}
    )
    assert explainer_llm.ainvoke.await_count == 0
    assert synth_llm.ainvoke.await_count == 0
    assert payload["success"] is True
    response = payload["response"]
    for figure in _TIER_FIGURES:
        assert figure in response, (figure, response)
    assert not _NO_FINDINGS_HUSK.search(response), response
    assert "1,240" not in response
    assert _BYPASS_TEXTS[0] not in response
    assert response.count(_DISCLOSURE) == 1
    _assert_never_presented_as_canonical(response)
