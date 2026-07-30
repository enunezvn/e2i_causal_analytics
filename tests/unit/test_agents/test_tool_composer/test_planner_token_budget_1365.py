"""Regression tests for #1365 — tool_composer planning JSON truncation.

Root cause (verified against the real Anthropic API, 2026-07-30): the planning
call ran claude-sonnet-5 with adaptive thinking ON and only the factory-default
``max_tokens=2048``. Thinking tokens count against that budget and stochastically
run away — a real 8192-budget trial produced 2058 output tokens, i.e. the natural
(thinking + JSON) generation EXCEEDS 2048, so at the 2048 cap the JSON is cut
mid-first-value (``Unterminated string ... line 2 column 16``) and the whole
composition fails with 0 tools executed.

Intent: the per-phase ``model=`` / ``max_tokens=`` knobs were DESIGNED to be
honored — the original ``_call_llm`` passed ``model=self.model, max_tokens=3000``
to the raw Anthropic SDK; the LangChain migration (55a7f749) replaced that with a
shared, pre-built client and silently dropped them (and lowered the budget to
2048). This suite pins the fix: each phase gets a correctly-sized client, the
planning phase disables thinking (guaranteed-safe structured JSON), and the
naive fence slicing is replaced by the fence-tolerant ``parse_llm_json``.
"""

from __future__ import annotations

import json

import pytest

from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.planner import PlanningError, ToolPlanner

# ---------------------------------------------------------------------------
# Per-phase client construction (factory mode: no client injected)
# ---------------------------------------------------------------------------


class _SpyFactory:
    """Records every get_chat_llm(**kwargs) call and returns an inert sentinel."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return object()  # inert client sentinel; never invoked at construction


@pytest.fixture
def spy_factory(monkeypatch):
    spy = _SpyFactory()
    # _resolve_phase_client imports get_chat_llm from the factory module at call
    # time, so patching the source symbol is sufficient.
    monkeypatch.setattr("src.utils.llm_factory.get_chat_llm", spy)
    return spy


class TestPerPhaseClientBudget:
    """Factory mode (llm_client=None) builds a sized client per phase."""

    def test_plan_phase_client_disables_thinking_and_raises_budget(
        self, spy_factory, mock_tool_registry
    ):
        ToolComposer(llm_client=None, tool_registry=mock_tool_registry)

        # Exactly one phase disables thinking: the planner (structured mapping —
        # thinking is wasted AND causes the truncation).
        plan_calls = [c for c in spy_factory.calls if c.get("reasoning_effort") == "none"]
        assert len(plan_calls) == 1, spy_factory.calls
        assert plan_calls[0]["max_tokens"] >= 4096
        assert plan_calls[0].get("model_tier") == "standard"

    def test_decompose_and_synthesize_budgets_do_not_regress(self, spy_factory, mock_tool_registry):
        ToolComposer(llm_client=None, tool_registry=mock_tool_registry)

        # The two sibling phases keep adaptive thinking but get a budget >= the
        # old shared 2048/2000 caps (no silent regression, headroom over the
        # ~2058-token natural output measured against the real API).
        sibling_calls = [c for c in spy_factory.calls if c.get("reasoning_effort") != "none"]
        assert len(sibling_calls) == 2, spy_factory.calls
        for c in sibling_calls:
            assert c["max_tokens"] >= 4096

    def test_per_phase_config_knobs_are_honored(self, spy_factory, mock_tool_registry):
        # The designed-but-severed knobs are live again: an explicit per-phase
        # config sizes that phase's client.
        config = {"phases": {"plan": {"max_tokens": 6000, "reasoning_effort": "low"}}}
        ToolComposer(llm_client=None, tool_registry=mock_tool_registry, config=config)

        low_calls = [c for c in spy_factory.calls if c.get("reasoning_effort") == "low"]
        assert len(low_calls) == 1, spy_factory.calls
        assert low_calls[0]["max_tokens"] == 6000

    def test_di_mode_uses_injected_client_without_factory(
        self, spy_factory, mock_llm_client, mock_tool_registry
    ):
        # DI mode: when a client is injected (tests, or the chatbot_tools path),
        # every phase shares it and the factory is NOT touched.
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)

        assert spy_factory.calls == []
        assert composer.decomposer.llm_client is mock_llm_client
        assert composer.planner.llm_client is mock_llm_client
        assert composer.synthesizer.llm_client is mock_llm_client


# ---------------------------------------------------------------------------
# _parse_response: fence-tolerant via parse_llm_json (PR #1364)
# ---------------------------------------------------------------------------


_VALID_PLAN = {
    "reasoning": "r",
    "tool_mappings": [],
    "execution_steps": [],
    "parallel_groups": [],
}


class TestParseResponseFenceTolerant:
    def _planner(self, mock_llm_client, mock_tool_registry) -> ToolPlanner:
        return ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)

    def test_parse_bare_json(self, mock_llm_client, mock_tool_registry):
        planner = self._planner(mock_llm_client, mock_tool_registry)
        assert planner._parse_response(json.dumps(_VALID_PLAN)) == _VALID_PLAN

    def test_parse_fenced_json(self, mock_llm_client, mock_tool_registry):
        planner = self._planner(mock_llm_client, mock_tool_registry)
        text = f"Here you go:\n```json\n{json.dumps(_VALID_PLAN)}\n```"
        assert planner._parse_response(text) == _VALID_PLAN

    def test_parse_unterminated_fence(self, mock_llm_client, mock_tool_registry):
        # Models truncate the CLOSING fence when they run low on tokens; the JSON
        # body is still complete and must parse (naive slicing dropped the last
        # char and failed).
        planner = self._planner(mock_llm_client, mock_tool_registry)
        text = f"```json\n{json.dumps(_VALID_PLAN)}"
        assert planner._parse_response(text) == _VALID_PLAN

    def test_string_value_containing_backticks(self, mock_llm_client, mock_tool_registry):
        # Bare-JSON-first: a value legitimately containing ``` must not be
        # mangled by fence logic.
        planner = self._planner(mock_llm_client, mock_tool_registry)
        payload = {**_VALID_PLAN, "reasoning": "use ```json fences``` carefully"}
        assert planner._parse_response(json.dumps(payload)) == payload

    def test_genuinely_truncated_json_raises_planning_error(
        self, mock_llm_client, mock_tool_registry
    ):
        # The #1365 defect payload: cut mid-first-value. Unrecoverable -> clear error.
        planner = self._planner(mock_llm_client, mock_tool_registry)
        truncated = '{\n  "reasoning": "sq_1 and sq_2 are descriptive trend analyses of TRx market'
        with pytest.raises(PlanningError) as exc:
            planner._parse_response(truncated)
        assert "JSON" in str(exc.value)
