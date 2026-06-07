"""
R3 — Planner data & schema awareness.

Tests that ToolPlanner.plan(available_columns=...) threads real dataset columns
into the planning prompt (F6 schema-binding) and that composer.compose derives
those columns from context["estimation_data"] (F2 data wiring).
"""

import logging

import pandas as pd
import pytest

from src.agents.tool_composer.planner import ToolPlanner


class TestPlannerColumnAwareness:
    """T1: available_columns reaches the planning prompt with a binding instruction."""

    @pytest.mark.asyncio
    async def test_available_columns_injected_into_prompt(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        columns = ["rep_visits", "rx_volume", "region", "patient_age"]

        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,  # force a real LLM call (no cached-plan short-circuit)
            use_episodic_memory=False,
        )

        await planner.plan(sample_decomposition, available_columns=columns)

        # The mock records the full user message of the planning ainvoke call.
        assert mock_llm_client.call_history, "planner did not call the LLM"
        user_msg = mock_llm_client.call_history[-1]["user"]

        # F6: the exact column names must appear as a comma-list section
        assert "## Available dataset columns:" in user_msg
        assert "rep_visits, rx_volume, region, patient_age" in user_msg
        # F6: an explicit instruction to bind to EXACT names / not invent
        assert "EXACT names" in user_msg
        assert "do NOT invent column names" in user_msg

    @pytest.mark.asyncio
    async def test_no_columns_omits_section(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )

        await planner.plan(sample_decomposition)  # no available_columns

        user_msg = mock_llm_client.call_history[-1]["user"]
        assert "## Available dataset columns:" not in user_msg


class TestComposerDerivesColumns:
    """T2: compose() pulls available_columns from a DataFrame in context."""

    @pytest.mark.asyncio
    async def test_compose_passes_dataframe_columns_to_planner(
        self, mock_llm_client, mock_tool_registry
    ):
        from src.agents.tool_composer.composer import ToolComposer

        frame = pd.DataFrame(
            {
                "rep_visits": [1, 2, 3],
                "rx_volume": [10, 20, 30],
                "region": ["NE", "MW", "NE"],
            }
        )

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_memory_contribution=False,
            config={"phases": {"plan": {"use_episodic_memory": False}}},
        )

        recorded: dict = {}
        real_plan = composer.planner.plan

        async def _spy(decomposition, available_columns=None):
            recorded["available_columns"] = available_columns
            return await real_plan(decomposition, available_columns=available_columns)

        composer.planner.plan = _spy  # type: ignore[method-assign]

        await composer.compose("causal effect of rep visits", context={"estimation_data": frame})

        assert recorded["available_columns"] == ["rep_visits", "rx_volume", "region"]

    @pytest.mark.asyncio
    async def test_compose_no_dataframe_passes_none(self, mock_llm_client, mock_tool_registry):
        from src.agents.tool_composer.composer import ToolComposer

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_memory_contribution=False,
            config={"phases": {"plan": {"use_episodic_memory": False}}},
        )

        recorded: dict = {}
        real_plan = composer.planner.plan

        async def _spy(decomposition, available_columns=None):
            recorded["available_columns"] = available_columns
            return await real_plan(decomposition, available_columns=available_columns)

        composer.planner.plan = _spy  # type: ignore[method-assign]

        # context carries a non-DataFrame value under the key — must be ignored.
        await composer.compose("causal effect", context={"estimation_data": {"not": "a frame"}})

        assert recorded["available_columns"] is None


class TestUnboundColumnWarning:
    """T3: a plan literal referencing a non-existent column logs a warning (no raise)."""

    @pytest.mark.asyncio
    async def test_warns_on_unbound_column_literal(
        self, mock_llm_client, mock_tool_registry, sample_decomposition, caplog
    ):
        # The default mock planning response binds treatment->rep_visits,
        # outcome->rx_volume. Provide columns that OMIT rx_volume so the
        # outcome literal is unbound.
        columns = ["rep_visits", "region"]

        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )

        with caplog.at_level(logging.WARNING):
            plan = await planner.plan(sample_decomposition, available_columns=columns)

        # Plan still succeeds (no hard-fail).
        assert len(plan.steps) > 0
        # Warning names the offending arg + value.
        assert any(
            "rx_volume" in rec.message and "not in available dataset columns" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_all_columns_bound(
        self, mock_llm_client, mock_tool_registry, sample_decomposition, caplog
    ):
        # All literal column args (rep_visits, rx_volume, region) are present.
        columns = ["rep_visits", "rx_volume", "region", "extra"]

        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )

        with caplog.at_level(logging.WARNING):
            await planner.plan(sample_decomposition, available_columns=columns)

        assert not any("not in available dataset columns" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_step_refs_never_warn(
        self, mock_llm_client, mock_tool_registry, sample_decomposition, caplog
    ):
        # The default response's step_2 uses dimension -> "region" (a column)
        # and effect -> "$step_1.effect" (a step ref). With region present and
        # a $-ref, no warning should fire for step_2.
        columns = ["rep_visits", "rx_volume", "region"]

        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )

        with caplog.at_level(logging.WARNING):
            await planner.plan(sample_decomposition, available_columns=columns)

        assert not any("$step_1.effect" in rec.message for rec in caplog.records)
