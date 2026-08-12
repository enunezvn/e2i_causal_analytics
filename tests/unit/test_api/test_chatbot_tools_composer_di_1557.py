"""Regression tests for #1557 — chat entry point must not bypass #1365 sizing.

Defect (measured during the #1547 lane, eval turn 2.6): ``tool_composer_tool``
injected ONE shared ``get_chat_llm(model_tier="reasoning", max_tokens=4096)``
client into ``compose_query``. That takes ``ToolComposer``'s dependency-injection
mode, whose documented contract is "every phase SHARES it unchanged" — so the
PLAN phase ran claude-sonnet-5 with adaptive thinking ON against the shared 4096
budget. Thinking tokens count against ``max_tokens``; the planner's JSON hit the
ceiling mid-object and raised ``PlanningError: Invalid JSON``.

#1365 already built the cure: factory mode (no injected client) sizes a client
PER PHASE from ``_PHASE_LLM_DEFAULTS`` — notably plan -> ``reasoning_effort:
"none"`` (thinking disabled; truncation impossible). The orchestrator-side entry
point (``ToolComposerAgent._get_composer``) already deliberately leaves
``llm_client=None`` for exactly this reason. This suite pins the CHAT entry
point to the same choice:

- ``tool_composer_tool`` calls ``compose_query`` WITHOUT injecting a client
  (factory mode -> per-phase sizing);
- ``compose_query`` makes that expressible (``llm_client`` defaults to None);
- the DI contract for tests (explicitly injected client is forwarded shared)
  stays intact.

Per-phase sizing itself (plan thinking-off, budgets, config overrides) is
pinned by ``test_planner_token_budget_1365.py`` — together with this file the
chain "chat entry -> factory mode -> sized planner client" is closed without
duplicating that suite.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.tool_composer import composer as composer_module
from src.api.routes import chatbot_tools
from src.api.routes.chatbot_tools import tool_composer_tool


def _mock_composition_result() -> MagicMock:
    """Minimal CompositionResult-shaped mock for the route envelope."""
    result = MagicMock()
    result.success = True
    result.status.value = "completed"
    result.decomposition.sub_questions = []
    result.execution.tools_executed = []
    result.execution.get_all_outputs.return_value = {}
    result.plan.get_execution_order.return_value = []
    result.plan.parallel_groups = []
    result.response.answer = "ok"
    result.response.confidence = 0.9
    return result


@pytest.fixture(autouse=True)
def _no_data_resolution():
    """Keep these tests hermetic: no cohort/KPI substrate resolution (DB-free).

    The seam under test is the LLM-client dependency injection, not the
    estimation-data threading (pinned elsewhere).
    """
    with (
        patch.object(chatbot_tools, "_resolve_cohort_frame", return_value=None),
        patch.object(chatbot_tools.kpi_resolution, "recognize_kpi", return_value=None),
    ):
        yield


@pytest.fixture(autouse=True)
def _no_entrypoint_client_build():
    """Hermetic guard for the RED state of this suite.

    Pre-fix, ``tool_composer_tool`` builds a real client via ``get_chat_llm``
    before ``compose_query`` runs, which requires a provider API key.
    ``create=True`` keeps the patch valid post-fix, when the entry point no
    longer imports the factory at all.
    """
    with patch.object(chatbot_tools, "get_chat_llm", create=True) as mock_llm:
        mock_llm.return_value = MagicMock(name="entrypoint-built-client")
        yield mock_llm


class TestChatEntryPointFactoryMode:
    """#1557 acceptance: chat traffic reaches the composer in factory mode."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.compose_query", new_callable=AsyncMock)
    async def test_no_shared_client_injected_into_compose_query(self, mock_compose):
        """The entry point must NOT pass a pre-built shared client.

        Injecting any client puts ALL phases in DI mode (shared unchanged),
        which bypasses the plan phase's thinking-off sizing and re-opens the
        truncation -> PlanningError failure.
        """
        mock_compose.return_value = _mock_composition_result()

        await tool_composer_tool.ainvoke(
            {"query": "Compare TRx trends across brands and explain the causal factors"}
        )

        mock_compose.assert_called_once()
        injected = mock_compose.call_args.kwargs.get("llm_client")
        assert injected is None, (
            "chat entry point injected a shared client into compose_query "
            f"({injected!r}) — this forces DI mode on every composer phase and "
            "bypasses #1365 per-phase sizing (plan must run thinking-off)"
        )

    @pytest.mark.asyncio
    async def test_real_compose_query_reaches_composer_in_factory_mode(self):
        """Through the REAL ``compose_query``, the composer sees llm_client=None.

        Fake only the ``ToolComposer`` class at its construction seam and let
        the entry point's actual call + ``compose_query``'s actual forwarding
        run, so a regression in either hop is caught.
        """
        captured: dict = {}

        class _FakeComposer:
            def __init__(self, llm_client=None, **kwargs):
                captured["llm_client"] = llm_client

            async def compose(self, query, context=None):
                return _mock_composition_result()

        with patch.object(composer_module, "ToolComposer", _FakeComposer):
            result = await tool_composer_tool.ainvoke(
                {"query": "What is TRx and why did it change?", "brand": "Kisqali"}
            )

        assert "llm_client" in captured, "composer was never constructed"
        assert captured["llm_client"] is None, (
            f"composer constructed with an injected client ({captured['llm_client']!r}) "
            "instead of factory mode — per-phase #1365 sizing is bypassed"
        )
        assert result["success"] is True

    def test_compose_query_llm_client_is_optional(self):
        """``compose_query`` must allow omitting the client (factory mode).

        ``ToolComposer(llm_client=None)`` is the documented factory mode; the
        convenience wrapper the chat entry point uses must expose it.
        """
        param = inspect.signature(composer_module.compose_query).parameters["llm_client"]
        assert param.default is None, (
            "compose_query requires llm_client — callers cannot request factory "
            "mode without explicitly passing None"
        )


class TestDIContractIntact:
    """The mock-injection contract the composer test suite relies on."""

    @pytest.mark.asyncio
    async def test_explicitly_injected_client_is_still_forwarded(self):
        """An explicitly injected client must still reach the composer (DI mode)."""
        sentinel = object()
        captured: dict = {}

        class _FakeComposer:
            def __init__(self, llm_client=None, **kwargs):
                captured["llm_client"] = llm_client

            async def compose(self, query, context=None):
                return _mock_composition_result()

        with patch.object(composer_module, "ToolComposer", _FakeComposer):
            await composer_module.compose_query(query="q", llm_client=sentinel)

        assert captured["llm_client"] is sentinel
