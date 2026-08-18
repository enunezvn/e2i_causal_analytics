"""#1698: the chat model's query rewrite drops servable cohort criteria.

Measured defect (2026-08-18 certification eval, turn 2.1): the user asked

    "Build a patient cohort for Remibrutinib CSU with inclusion criteria for
    adults over 18 diagnosed in 2024"

and the chat model dispatched

    orchestrator_tool(query="Profile the Remibrutinib patient cohort (CSU)
    — aggregate counts and breakdowns")

— both criteria stripped BEFORE parse_cohort_ask ever ran, so the #1696
criteria accounting (servable binding + criteria_not_applied) had nothing to
account for, and the answer blamed the cohort model for the model's own
rewrite. parse_cohort_ask on the ORIGINAL text yields age_min (servable) and
diagnosis_year (honest-unserved) — verified against HEAD before this fix.

Fix chain under test (each link has its own test):
  1. orchestrator_tool threads the raw user ask (contextvar set by the
     copilotkit handler) into user_context["raw_user_query"];
  2. _resolve_cohort_profiler_input passes it through to the agent;
  3. CohortProfilerAgent.analyze parses BOTH texts and merges
     (merge_cohort_asks) so every criterion in the original ask reaches the
     accounting — rewrite-parse wins on collision, original fills the gaps.
"""

import pytest

from src.agents.cohort_profiler import CohortProfilerAgent
from src.agents.cohort_profiler.ask import parse_cohort_ask

# The measured 2.1 pair, verbatim (question set + raw_agui.jsonl tool args).
ORIGINAL = (
    "Build a patient cohort for Remibrutinib CSU with inclusion criteria "
    "for adults over 18 diagnosed in 2024"
)
REWRITE = "Profile the Remibrutinib patient cohort (CSU) — aggregate counts and breakdowns"


# ---------------------------------------------------------------- merge unit


def test_merge_adds_dropped_criteria():
    from src.agents.cohort_profiler.ask import merge_cohort_asks

    merged = merge_cohort_asks(parse_cohort_ask(REWRITE), parse_cohort_ask(ORIGINAL))

    kinds = {c.kind: c for c in merged.criteria}
    assert set(kinds) == {"age_min", "diagnosis_year"}
    assert kinds["age_min"].servable and kinds["age_min"].value == 18
    assert not kinds["diagnosis_year"].servable
    assert kinds["diagnosis_year"].value == 2024
    # Identity fields stay the primary's.
    assert merged.entity_type == "patient"
    assert merged.brand == "Remibrutinib"


def test_merge_primary_wins_on_kind_collision():
    from src.agents.cohort_profiler.ask import merge_cohort_asks

    primary = parse_cohort_ask("patient cohort for Remibrutinib, adults over 21")
    supplement = parse_cohort_ask(ORIGINAL)
    merged = merge_cohort_asks(primary, supplement)

    age = next(c for c in merged.criteria if c.kind == "age_min")
    assert age.value == 21  # the rewrite may have resolved anaphora — it wins


def test_merge_noop_when_supplement_parses_nothing():
    from src.agents.cohort_profiler.ask import merge_cohort_asks

    primary = parse_cohort_ask(ORIGINAL)
    merged = merge_cohort_asks(primary, parse_cohort_ask("thanks, run that again"))

    assert merged.criteria == primary.criteria
    assert merged.window == primary.window
    assert merged.threshold == primary.threshold


def test_merge_fills_missing_window_and_threshold():
    from src.agents.cohort_profiler.ask import merge_cohort_asks

    primary = parse_cohort_ask(REWRITE)
    supplement = parse_cohort_ask(
        "HCPs who prescribed more than 50 TRx last quarter for Remibrutinib"
    )
    merged = merge_cohort_asks(primary, supplement)

    assert merged.window is not None and merged.window.label == "last quarter"
    assert merged.threshold is not None


# ------------------------------------------------------------ agent threading


@pytest.mark.asyncio
async def test_analyze_threads_raw_user_query_into_parse(monkeypatch):
    agent = CohortProfilerAgent()
    seen = {}

    async def _capture(ask, *a, **k):
        seen["ask"] = ask
        return {"status": "completed", "narrative": ""}

    monkeypatch.setattr(agent, "_analyze_patients", _capture)

    await agent.analyze({"query": REWRITE, "raw_user_query": ORIGINAL, "brand": "Remibrutinib"})

    kinds = {c.kind for c in seen["ask"].criteria}
    assert kinds == {"age_min", "diagnosis_year"}


@pytest.mark.asyncio
async def test_analyze_without_raw_user_query_unchanged(monkeypatch):
    agent = CohortProfilerAgent()
    seen = {}

    async def _capture(ask, *a, **k):
        seen["ask"] = ask
        return {"status": "completed", "narrative": ""}

    monkeypatch.setattr(agent, "_analyze_patients", _capture)

    await agent.analyze({"query": REWRITE, "brand": "Remibrutinib"})

    assert seen["ask"].criteria == ()


# ---------------------------------------------------------- resolver threading


def test_resolver_threads_raw_user_query():
    from src.agents.orchestrator.nodes.dispatcher import (
        NeedsStructuredInput,
        _resolve_cohort_profiler_input,
    )

    dispatch = {"agent_name": "cohort_profiler", "parameters": {}}
    resolved = _resolve_cohort_profiler_input(
        {
            "query": REWRITE,
            "user_context": {"brand": "Remibrutinib", "raw_user_query": ORIGINAL},
        },
        dispatch,
    )
    assert not isinstance(resolved, NeedsStructuredInput)
    assert resolved["raw_user_query"] == ORIGINAL

    # Absent (non-chat callers, direct invocations) → key stays absent.
    resolved_bare = _resolve_cohort_profiler_input({"query": REWRITE}, dispatch)
    assert "raw_user_query" not in resolved_bare


# ------------------------------------------------------- orchestrator_tool leg


class _FakeOrchestrator:
    def __init__(self):
        self.payload = None

    async def run(self, payload):
        self.payload = payload
        return {
            "response_text": "ok",
            "response_confidence": 0.9,
            "agents_dispatched": ["cohort_profiler"],
            "status": "completed",
        }


@pytest.mark.asyncio
async def test_orchestrator_tool_stashes_raw_user_query(monkeypatch):
    from src.api.routes import chatbot_tools

    fake = _FakeOrchestrator()
    monkeypatch.setattr(chatbot_tools, "get_orchestrator", lambda: fake)

    token = chatbot_tools.set_raw_user_query(ORIGINAL)
    try:
        await chatbot_tools.orchestrator_tool.ainvoke({"query": REWRITE})
    finally:
        chatbot_tools.reset_raw_user_query(token)

    assert fake.payload["user_context"]["raw_user_query"] == ORIGINAL


@pytest.mark.asyncio
async def test_orchestrator_tool_without_contextvar_omits_key(monkeypatch):
    from src.api.routes import chatbot_tools

    fake = _FakeOrchestrator()
    monkeypatch.setattr(chatbot_tools, "get_orchestrator", lambda: fake)

    await chatbot_tools.orchestrator_tool.ainvoke({"query": REWRITE})

    assert "raw_user_query" not in fake.payload["user_context"]
