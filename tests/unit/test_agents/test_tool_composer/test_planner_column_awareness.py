"""
R3 — Planner data & schema awareness.

Tests that ToolPlanner.plan(available_columns=...) threads real dataset columns
into the planning prompt (F6 schema-binding) and that composer.compose derives
those columns from context["estimation_data"] (F2 data wiring).
"""

import pandas as pd
import pytest

from src.agents.tool_composer.planner import PlanningError, ToolPlanner


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

        async def _spy(
            decomposition, available_columns=None, column_profiles=None, outcome_hint=None
        ):
            recorded["available_columns"] = available_columns
            recorded["column_profiles"] = column_profiles
            recorded["outcome_hint"] = outcome_hint
            return await real_plan(
                decomposition,
                available_columns=available_columns,
                column_profiles=column_profiles,
                outcome_hint=outcome_hint,
            )

        composer.planner.plan = _spy  # type: ignore[method-assign]

        await composer.compose("causal effect of rep visits", context={"estimation_data": frame})

        assert recorded["available_columns"] == ["rep_visits", "rx_volume", "region"]
        # F6(b): composer now also threads a rich column profile.
        assert recorded["column_profiles"] is not None
        assert {p["name"] for p in recorded["column_profiles"]} == {
            "rep_visits",
            "rx_volume",
            "region",
        }

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

        async def _spy(
            decomposition, available_columns=None, column_profiles=None, outcome_hint=None
        ):
            recorded["available_columns"] = available_columns
            recorded["column_profiles"] = column_profiles
            recorded["outcome_hint"] = outcome_hint
            return await real_plan(
                decomposition,
                available_columns=available_columns,
                column_profiles=column_profiles,
                outcome_hint=outcome_hint,
            )

        composer.planner.plan = _spy  # type: ignore[method-assign]

        # context carries a non-DataFrame value under the key — must be ignored.
        await composer.compose("causal effect", context={"estimation_data": {"not": "a frame"}})

        assert recorded["available_columns"] is None
        assert recorded["column_profiles"] is None


class TestUnboundColumnEnforcement:
    """T3 (F6(b)): when a schema is supplied the planner ENFORCES bindings.

    Under F6 the unbound-column path was warn-only. F6(b) supersedes it: a
    column-typed literal that cannot be resolved to a real column FAILS FAST
    (PlanningError) so the bad binding never reaches the fail-closed tool. The
    warn-only behavior is retained ONLY as a no-schema fallback (covered by the
    F6(b) suite ``test_no_schema_does_not_enforce``).
    """

    @pytest.mark.asyncio
    async def test_enforces_unbound_column_literal(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        # The default mock planning response binds treatment->rep_visits,
        # outcome->rx_volume. Provide columns that OMIT rx_volume so the
        # outcome literal is unbound and unresolvable.
        columns = ["rep_visits", "region"]

        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )

        with pytest.raises(PlanningError) as exc:
            await planner.plan(sample_decomposition, available_columns=columns)
        assert "rx_volume" in str(exc.value)
        assert "unbound column" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_no_raise_when_all_columns_bound(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        # All literal column args (rep_visits, rx_volume, region) are present.
        columns = ["rep_visits", "rx_volume", "region", "extra"]

        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )

        plan = await planner.plan(sample_decomposition, available_columns=columns)
        assert len(plan.steps) == 2

    @pytest.mark.asyncio
    async def test_step_refs_never_fail(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        # The default response's step_2 uses dimension -> "region" (a column)
        # and effect -> "$step_1.effect" (a step ref). With region present and
        # a $-ref, enforcement must not raise on step_2.
        columns = ["rep_visits", "rx_volume", "region"]

        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )

        plan = await planner.plan(sample_decomposition, available_columns=columns)
        step2 = next(s for s in plan.steps if s.step_id == "step_2")
        assert step2.input_mapping["effect"] == "$step_1.effect"
