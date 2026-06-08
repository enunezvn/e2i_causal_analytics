"""
F6(b) — Planner semantic/entity binding + enforcement + cate schema fix.

These deterministic, network-free tests pin the new behavior:

1. ``ToolComposer._extract_column_profiles`` classifies dtype families
   (binary / numeric-continuous / categorical / other), counts cardinality,
   lists low-cardinality value sets, and is ROBUST to unhashable cell values
   (real cohort frames carry list-valued columns such as ``comorbidities``).
2. The planner prompt carries the richer column PROFILE (names + dtype family
   + value distributions) so the LLM avoids inventing columns AND avoids
   near-constant / degenerate targets.
3. Enforcement (not just warn): after parsing, the planner best-effort resolves
   column-typed args to the real schema (case-insensitive / alias / substring)
   and FAILS FAST (PlanningError) on a column-typed arg that resolves to no
   real column — so the bad binding never reaches the fail-closed tool.
   Non-column args (literals, ``$step`` refs, scalar params) are untouched.
4. ``CateAnalyzerInput.segments`` is ``List[str]`` (not ``List[Dict]``).

The single faithful real-LLM proof lives in
``tests/integration/test_tool_composer_functional_e2e.py`` gated behind
``E2I_RUN_REAL_LLM_E2E=1``.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.planner import PlanningError, ToolPlanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kisqali_like_frame() -> pd.DataFrame:
    """A small frame mirroring the REAL Kisqali/Northeast cohort dtype shapes.

    Includes a binary treatment, a degenerate near-constant binary, numeric
    continuous outcomes, low-card categoricals, a constant categorical, and an
    unhashable (list-valued) column — exactly the families the real frame has.
    """
    return pd.DataFrame(
        {
            "academic_hcp": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            # near-constant binary (degenerate as a target)
            "treatment_initiated": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "days_to_treatment": [73.0, 13.0, 83.0, 33.0, 55.0, 27.0],
            "engagement_score": [8.54, 9.15, 9.14, 8.56, 8.41, 8.15],
            "age_group": ["50-64", "35-49", "18-34", "65+", "50-64", "35-49"],
            "geographic_region": ["northeast"] * 6,  # constant categorical
            "comorbidities": [["a", "b"], ["c"], [], ["a"], ["b", "c"], []],  # unhashable
        }
    )


# ---------------------------------------------------------------------------
# 1. Column-profile extraction (composer)
# ---------------------------------------------------------------------------


class TestColumnProfileExtraction:
    def _profiles(self) -> dict:
        composer = ToolComposer.__new__(ToolComposer)  # no LLM needed for this static method
        frame = _kisqali_like_frame()
        profiles = ToolComposer._extract_column_profiles(composer, {"estimation_data": frame})
        assert profiles is not None
        return {p["name"]: p for p in profiles}

    def test_classifies_binary(self):
        by_name = self._profiles()
        assert by_name["academic_hcp"]["dtype_family"] == "binary"
        assert by_name["academic_hcp"]["n_unique"] == 2

    def test_classifies_numeric_continuous(self):
        by_name = self._profiles()
        assert by_name["days_to_treatment"]["dtype_family"] == "numeric-continuous"
        assert by_name["engagement_score"]["dtype_family"] == "numeric-continuous"

    def test_classifies_categorical_with_value_list(self):
        by_name = self._profiles()
        prof = by_name["age_group"]
        assert prof["dtype_family"] == "categorical"
        # low-card -> the actual values are exposed
        assert set(prof["values"]) == {"50-64", "35-49", "18-34", "65+"}

    def test_constant_categorical_flagged_low_card(self):
        by_name = self._profiles()
        prof = by_name["geographic_region"]
        assert prof["n_unique"] == 1
        assert prof["values"] == ["northeast"]

    def test_unhashable_column_does_not_crash(self):
        # The real cohort frame has list-valued columns; profile extraction
        # must NOT raise TypeError: unhashable type: 'list'.
        by_name = self._profiles()
        assert "comorbidities" in by_name
        # cardinality still computed via a safe fallback (non-negative int)
        assert by_name["comorbidities"]["n_unique"] >= 1
        assert by_name["comorbidities"]["dtype_family"] in {"other", "categorical"}

    def test_nonnull_counts_present(self):
        by_name = self._profiles()
        assert by_name["academic_hcp"]["n_nonnull"] == 6

    def test_no_frame_returns_none(self):
        composer = ToolComposer.__new__(ToolComposer)
        assert ToolComposer._extract_column_profiles(composer, {}) is None
        assert (
            ToolComposer._extract_column_profiles(composer, {"estimation_data": {"x": 1}}) is None
        )


# ---------------------------------------------------------------------------
# 2. Profile threaded into the planning prompt
# ---------------------------------------------------------------------------


class TestProfileInPrompt:
    @pytest.mark.asyncio
    async def test_profile_block_in_prompt(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        profiles = [
            {
                "name": "academic_hcp",
                "dtype_family": "binary",
                "n_unique": 2,
                "n_nonnull": 6,
                "values": [0, 1],
            },
            {
                "name": "days_to_treatment",
                "dtype_family": "numeric-continuous",
                "n_unique": 78,
                "n_nonnull": 6,
                "values": None,
            },
        ]
        # Bind the plan to the profile's real columns so enforcement does not
        # fail-fast — this test only verifies the prompt block content.
        mock_llm_client.set_planning_response(
            '{"reasoning":"r",'
            '"tool_mappings":[{"sub_question_id":"sq_1","tool_name":"causal_effect_estimator",'
            '"confidence":0.9,"reasoning":"x"}],'
            '"execution_steps":[{"step_id":"step_1","sub_question_id":"sq_1",'
            '"tool_name":"causal_effect_estimator",'
            '"input_mapping":{"treatment":"academic_hcp","outcome":"days_to_treatment"},'
            '"depends_on_steps":[]}],'
            '"parallel_groups":[["step_1"]]}'
        )
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )
        await planner.plan(sample_decomposition, column_profiles=profiles)

        user_msg = mock_llm_client.call_history[-1]["user"]
        # the column block is present with dtype-family guidance
        assert "academic_hcp" in user_msg
        assert "binary" in user_msg
        assert "days_to_treatment" in user_msg
        assert "numeric-continuous" in user_msg
        # explicit semantic guidance for treatment/outcome/segments
        assert "treatment" in user_msg.lower()
        assert "NEVER" in user_msg  # never bind a brand/region VALUE as a column

    @pytest.mark.asyncio
    async def test_available_columns_back_compat(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        # The legacy available_columns kwarg must still work (derives a minimal
        # name-only profile internally).
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )
        await planner.plan(
            sample_decomposition,
            available_columns=["rep_visits", "rx_volume", "region"],
        )
        user_msg = mock_llm_client.call_history[-1]["user"]
        assert "rep_visits" in user_msg
        assert "rx_volume" in user_msg


# ---------------------------------------------------------------------------
# 3. Enforcement: best-effort resolve, then fail fast
# ---------------------------------------------------------------------------


class TestEnforcement:
    @pytest.mark.asyncio
    async def test_bad_binding_fails_fast(self, mock_llm_client, mock_tool_registry):
        # Plan binds outcome='conversion_rate' which is NOT a real column and
        # cannot be resolved to one. Enforcement must raise PlanningError.
        from src.agents.tool_composer.models.composition_models import (
            DecompositionResult,
            SubQuestion,
        )

        decomposition = DecompositionResult(
            original_query="what drove conversion",
            sub_questions=[
                SubQuestion(
                    id="sq_1",
                    question="what drove conversion?",
                    intent="CAUSAL",
                    entities=["conversion"],
                    depends_on=[],
                )
            ],
            decomposition_reasoning="t",
        )
        mock_llm_client.set_planning_response(
            '{"reasoning":"r",'
            '"tool_mappings":[{"sub_question_id":"sq_1","tool_name":"causal_effect_estimator",'
            '"confidence":0.9,"reasoning":"x"}],'
            '"execution_steps":[{"step_id":"step_1","sub_question_id":"sq_1",'
            '"tool_name":"causal_effect_estimator",'
            '"input_mapping":{"treatment":"academic_hcp","outcome":"conversion_rate"},'
            '"depends_on_steps":[]}],'
            '"parallel_groups":[["step_1"]]}'
        )
        profiles = [
            {
                "name": "academic_hcp",
                "dtype_family": "binary",
                "n_unique": 2,
                "n_nonnull": 6,
                "values": [0, 1],
            },
            {
                "name": "days_to_treatment",
                "dtype_family": "numeric-continuous",
                "n_unique": 78,
                "n_nonnull": 6,
                "values": None,
            },
        ]
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )
        with pytest.raises(PlanningError) as exc:
            await planner.plan(decomposition, column_profiles=profiles)
        assert "conversion_rate" in str(exc.value)
        assert "unbound column" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_good_binding_passes(self, mock_llm_client, mock_tool_registry):
        from src.agents.tool_composer.models.composition_models import (
            DecompositionResult,
            SubQuestion,
        )

        decomposition = DecompositionResult(
            original_query="effect of academic hcp on time to treatment",
            sub_questions=[
                SubQuestion(
                    id="sq_1",
                    question="effect of academic hcp on time to treatment?",
                    intent="CAUSAL",
                    entities=[],
                    depends_on=[],
                )
            ],
            decomposition_reasoning="t",
        )
        mock_llm_client.set_planning_response(
            '{"reasoning":"r",'
            '"tool_mappings":[{"sub_question_id":"sq_1","tool_name":"causal_effect_estimator",'
            '"confidence":0.9,"reasoning":"x"}],'
            '"execution_steps":[{"step_id":"step_1","sub_question_id":"sq_1",'
            '"tool_name":"causal_effect_estimator",'
            '"input_mapping":{"treatment":"academic_hcp","outcome":"days_to_treatment"},'
            '"depends_on_steps":[]}],'
            '"parallel_groups":[["step_1"]]}'
        )
        profiles = [
            {
                "name": "academic_hcp",
                "dtype_family": "binary",
                "n_unique": 2,
                "n_nonnull": 6,
                "values": [0, 1],
            },
            {
                "name": "days_to_treatment",
                "dtype_family": "numeric-continuous",
                "n_unique": 78,
                "n_nonnull": 6,
                "values": None,
            },
        ]
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )
        plan = await planner.plan(decomposition, column_profiles=profiles)
        assert len(plan.steps) == 1
        assert plan.steps[0].input_mapping["outcome"] == "days_to_treatment"

    @pytest.mark.asyncio
    async def test_case_insensitive_resolution(self, mock_llm_client, mock_tool_registry):
        # A near-miss casing (Days_To_Treatment) is RESOLVED to the real column
        # rather than failing — best-effort resolution before fail-fast.
        from src.agents.tool_composer.models.composition_models import (
            DecompositionResult,
            SubQuestion,
        )

        decomposition = DecompositionResult(
            original_query="q",
            sub_questions=[
                SubQuestion(id="sq_1", question="q?", intent="CAUSAL", entities=[], depends_on=[])
            ],
            decomposition_reasoning="t",
        )
        mock_llm_client.set_planning_response(
            '{"reasoning":"r",'
            '"tool_mappings":[{"sub_question_id":"sq_1","tool_name":"causal_effect_estimator",'
            '"confidence":0.9,"reasoning":"x"}],'
            '"execution_steps":[{"step_id":"step_1","sub_question_id":"sq_1",'
            '"tool_name":"causal_effect_estimator",'
            '"input_mapping":{"treatment":"academic_hcp","outcome":"Days_To_Treatment"},'
            '"depends_on_steps":[]}],'
            '"parallel_groups":[["step_1"]]}'
        )
        profiles = [
            {
                "name": "academic_hcp",
                "dtype_family": "binary",
                "n_unique": 2,
                "n_nonnull": 6,
                "values": [0, 1],
            },
            {
                "name": "days_to_treatment",
                "dtype_family": "numeric-continuous",
                "n_unique": 78,
                "n_nonnull": 6,
                "values": None,
            },
        ]
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )
        plan = await planner.plan(decomposition, column_profiles=profiles)
        # resolved to the real, correctly-cased column
        assert plan.steps[0].input_mapping["outcome"] == "days_to_treatment"

    @pytest.mark.asyncio
    async def test_step_ref_and_literal_args_never_fail(
        self, mock_llm_client, mock_tool_registry, sample_decomposition, caplog
    ):
        # Default mock plan: step_2 uses dimension->"region" and a $step ref.
        # Provide a profile that INCLUDES rep_visits/rx_volume/region so all
        # column args resolve; the $step ref must never be treated as a column.
        profiles = [
            {
                "name": "rep_visits",
                "dtype_family": "numeric-continuous",
                "n_unique": 3,
                "n_nonnull": 3,
                "values": None,
            },
            {
                "name": "rx_volume",
                "dtype_family": "numeric-continuous",
                "n_unique": 3,
                "n_nonnull": 3,
                "values": None,
            },
            {
                "name": "region",
                "dtype_family": "categorical",
                "n_unique": 2,
                "n_nonnull": 3,
                "values": ["NE", "MW"],
            },
        ]
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )
        with caplog.at_level(logging.WARNING):
            plan = await planner.plan(sample_decomposition, column_profiles=profiles)
        # All column args resolve; $step ref untouched; plan succeeds.
        assert len(plan.steps) == 2
        step2 = next(s for s in plan.steps if s.step_id == "step_2")
        assert step2.input_mapping["effect"] == "$step_1.effect"

    @pytest.mark.asyncio
    async def test_no_schema_does_not_enforce(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        # With NO column profile/columns, enforcement is a no-op: the planner
        # cannot validate, so even an invented column must NOT raise.
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_caching=False,
            use_episodic_memory=False,
        )
        plan = await planner.plan(sample_decomposition)  # no schema
        assert len(plan.steps) == 2


# ---------------------------------------------------------------------------
# 4. cate_analyzer input schema fix
# ---------------------------------------------------------------------------


class TestCateSchemaFix:
    def test_cate_input_segments_is_list_of_str(self):
        from src.agents.tool_composer.tool_registrations import CateAnalyzerInput

        model = CateAnalyzerInput(
            treatment="academic_hcp",
            outcome="days_to_treatment",
            segments=["age_group", "gender"],
        )
        assert model.segments == ["age_group", "gender"]
        # field type is List[str] -> a list of dicts must be rejected
        with pytest.raises(Exception):
            CateAnalyzerInput(
                treatment="academic_hcp",
                outcome="days_to_treatment",
                segments=[{"name": "age_group"}],
            )

    def test_cate_registration_uses_corrected_model(self):
        # importing tool_registrations triggers the @composable_tool decorator
        # that registers cate_analyzer with the corrected input model.
        import src.agents.tool_composer.tool_registrations as tr  # noqa: F401
        from src.agents.tool_composer.tool_registrations import CateAnalyzerInput
        from src.tool_registry.registry import get_registry

        registered = get_registry()._tools.get("cate_analyzer")
        assert registered is not None
        assert registered.pydantic_input_model is CateAnalyzerInput
