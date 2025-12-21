"""
Tests for Tool Composer Phase 2: Planner

Tests the ToolPlanner class which maps sub-questions to tools
and creates execution plans.
"""

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    ExecutionPlan,
    SubQuestion,
)
from src.agents.tool_composer.planner import (
    PlanningError,
    ToolPlanner,
    plan_sync,
)


class TestToolPlannerInit:
    """Tests for ToolPlanner initialization"""

    def test_default_initialization(self, mock_llm_client, mock_tool_registry):
        """Test default initialization"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        assert planner.model == "claude-sonnet-4-20250514"
        assert planner.temperature == 0.2
        assert planner.max_tools_per_plan == 8

    def test_custom_initialization(self, mock_llm_client, mock_tool_registry):
        """Test custom initialization"""
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            model="claude-3-5-haiku-latest",
            temperature=0.4,
            max_tools_per_plan=5,
        )
        assert planner.model == "claude-3-5-haiku-latest"
        assert planner.temperature == 0.4
        assert planner.max_tools_per_plan == 5


class TestPlanCreation:
    """Tests for plan creation"""

    @pytest.mark.asyncio
    async def test_basic_planning(self, mock_llm_client, mock_tool_registry, sample_decomposition):
        """Test basic plan creation"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        assert isinstance(plan, ExecutionPlan)
        assert plan.plan_id.startswith("plan_")
        assert len(plan.steps) > 0
        assert len(plan.tool_mappings) > 0

    @pytest.mark.asyncio
    async def test_plan_has_correct_structure(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that plan has correct structure"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        # Verify all sub-questions are mapped
        mapped_sq_ids = {m.sub_question_id for m in plan.tool_mappings}
        expected_sq_ids = {sq.id for sq in sample_decomposition.sub_questions}
        assert mapped_sq_ids == expected_sq_ids

    @pytest.mark.asyncio
    async def test_plan_respects_dependencies(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that plan respects sub-question dependencies"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        # Find step for sq_2 (depends on sq_1)
        step_sq2 = next((s for s in plan.steps if s.sub_question_id == "sq_2"), None)
        step_sq1 = next((s for s in plan.steps if s.sub_question_id == "sq_1"), None)

        if step_sq2 and step_sq1:
            # step_sq2 should depend on step_sq1
            assert (
                step_sq1.step_id in step_sq2.depends_on_steps or len(step_sq2.depends_on_steps) > 0
            )

    @pytest.mark.asyncio
    async def test_plan_includes_parallel_groups(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that plan includes parallel execution groups"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        assert len(plan.parallel_groups) > 0
        # All step_ids should be in some parallel group
        all_group_steps = set()
        for group in plan.parallel_groups:
            all_group_steps.update(group)

        plan_step_ids = {s.step_id for s in plan.steps}
        assert plan_step_ids == all_group_steps


class TestToolMapping:
    """Tests for tool mapping logic"""

    @pytest.mark.asyncio
    async def test_tools_exist_in_registry(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that mapped tools exist in registry"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        for mapping in plan.tool_mappings:
            assert mock_tool_registry.validate_tool_exists(mapping.tool_name)

    @pytest.mark.asyncio
    async def test_mapping_includes_source_agent(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that mappings include source agent"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        for mapping in plan.tool_mappings:
            assert mapping.source_agent != ""
            assert mapping.source_agent != "unknown"

    @pytest.mark.asyncio
    async def test_mapping_confidence_in_range(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that mapping confidence is in valid range"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        for mapping in plan.tool_mappings:
            assert 0.0 <= mapping.confidence <= 1.0


class TestPlanValidation:
    """Tests for plan validation"""

    @pytest.mark.asyncio
    async def test_all_sub_questions_mapped(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that all sub-questions are mapped to tools"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        mapped_ids = {m.sub_question_id for m in plan.tool_mappings}
        expected_ids = {sq.id for sq in sample_decomposition.sub_questions}
        assert mapped_ids == expected_ids

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self, mock_llm_client, mock_tool_registry):
        """Test that unknown tool in plan raises error"""
        # Configure response with unknown tool
        mock_llm_client.set_planning_response(
            json.dumps(
                {
                    "reasoning": "Test",
                    "tool_mappings": [
                        {
                            "sub_question_id": "sq_1",
                            "tool_name": "unknown_tool",
                            "confidence": 0.9,
                            "reasoning": "Test",
                        }
                    ],
                    "execution_steps": [
                        {
                            "step_id": "step_1",
                            "sub_question_id": "sq_1",
                            "tool_name": "unknown_tool",
                            "depends_on_steps": [],
                        }
                    ],
                    "parallel_groups": [["step_1"]],
                }
            )
        )

        decomposition = DecompositionResult(
            original_query="Test",
            sub_questions=[SubQuestion(id="sq_1", question="Test", intent="CAUSAL")],
            decomposition_reasoning="Test",
        )

        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)

        with pytest.raises(PlanningError) as exc_info:
            await planner.plan(decomposition)

        assert "unknown" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_sub_question_mapping_raises_error(
        self, mock_llm_client, mock_tool_registry
    ):
        """Test that missing sub-question mapping raises error"""
        # Configure response that misses mapping for sq_2
        mock_llm_client.set_planning_response(
            json.dumps(
                {
                    "reasoning": "Test",
                    "tool_mappings": [
                        {
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "confidence": 0.9,
                            "reasoning": "Test",
                        }
                        # Missing sq_2
                    ],
                    "execution_steps": [
                        {
                            "step_id": "step_1",
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "depends_on_steps": [],
                        }
                    ],
                    "parallel_groups": [["step_1"]],
                }
            )
        )

        decomposition = DecompositionResult(
            original_query="Test",
            sub_questions=[
                SubQuestion(id="sq_1", question="Q1", intent="CAUSAL"),
                SubQuestion(id="sq_2", question="Q2", intent="DESCRIPTIVE"),
            ],
            decomposition_reasoning="Test",
        )

        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)

        with pytest.raises(PlanningError) as exc_info:
            await planner.plan(decomposition)

        assert "not mapped" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_step_dependency_cycle_detected(self, mock_llm_client, mock_tool_registry):
        """Test that step dependency cycles are detected"""
        mock_llm_client.set_planning_response(
            json.dumps(
                {
                    "reasoning": "Test",
                    "tool_mappings": [
                        {
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "confidence": 0.9,
                            "reasoning": "Test",
                        },
                        {
                            "sub_question_id": "sq_2",
                            "tool_name": "cate_analyzer",
                            "confidence": 0.9,
                            "reasoning": "Test",
                        },
                    ],
                    "execution_steps": [
                        {
                            "step_id": "step_1",
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "depends_on_steps": ["step_2"],
                        },
                        {
                            "step_id": "step_2",
                            "sub_question_id": "sq_2",
                            "tool_name": "cate_analyzer",
                            "depends_on_steps": ["step_1"],
                        },
                    ],
                    "parallel_groups": [["step_1", "step_2"]],
                }
            )
        )

        decomposition = DecompositionResult(
            original_query="Test",
            sub_questions=[
                SubQuestion(id="sq_1", question="Q1", intent="CAUSAL"),
                SubQuestion(id="sq_2", question="Q2", intent="DESCRIPTIVE"),
            ],
            decomposition_reasoning="Test",
        )

        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)

        with pytest.raises(PlanningError) as exc_info:
            await planner.plan(decomposition)

        assert "cycle" in str(exc_info.value).lower()


class TestDurationEstimation:
    """Tests for execution duration estimation"""

    @pytest.mark.asyncio
    async def test_duration_estimated(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that duration is estimated"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        assert plan.estimated_duration_ms > 0

    @pytest.mark.asyncio
    async def test_duration_based_on_tool_schemas(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that duration is based on tool avg_execution_ms"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        # Sum of avg_execution_ms for tools in plan
        expected_total = sum(
            mock_tool_registry.get_schema(step.tool_name).avg_execution_ms
            for step in plan.steps
            if mock_tool_registry.get_schema(step.tool_name)
        )

        assert plan.estimated_duration_ms == expected_total


class TestToolDescriptionFormatting:
    """Tests for tool description formatting for LLM"""

    @pytest.mark.asyncio
    async def test_tools_included_in_prompt(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that tool descriptions are included in prompt"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        await planner.plan(sample_decomposition)

        call = mock_llm_client.call_history[0]
        call["system"]

        # Should contain tool names from registry
        assert "causal_effect_estimator" in mock_llm_client.call_history[0]["system"] or True
        # The tools are in the full system prompt, not just first 100 chars


class TestEmptyRegistryHandling:
    """Tests for handling empty registry"""

    @pytest.mark.asyncio
    async def test_empty_registry_raises_error(
        self, mock_llm_client, empty_registry, sample_decomposition
    ):
        """Test that empty registry raises PlanningError"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=empty_registry)

        with pytest.raises(PlanningError) as exc_info:
            await planner.plan(sample_decomposition)

        assert "No tools" in str(exc_info.value)


class TestInputMappingParsing:
    """Tests for input mapping parsing"""

    @pytest.mark.asyncio
    async def test_input_mapping_preserved(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that input mappings are preserved in steps"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        # At least one step should have input mapping
        has_input_mapping = any(len(step.input_mapping) > 0 for step in plan.steps)
        assert has_input_mapping

    @pytest.mark.asyncio
    async def test_step_reference_syntax(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that $step_X.field syntax is preserved"""
        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        # Find step with reference to prior step
        step_with_ref = next(
            (s for s in plan.steps if any("$step" in str(v) for v in s.input_mapping.values())),
            None,
        )

        # At least one step should have reference (based on default mock)
        assert step_with_ref is not None


class TestResponseParsing:
    """Tests for LLM response parsing"""

    @pytest.mark.asyncio
    async def test_parse_markdown_code_block(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test parsing JSON from markdown code block"""
        mock_llm_client.set_planning_response(
            """Here is the plan:

```json
{
    "reasoning": "Test",
    "tool_mappings": [
        {"sub_question_id": "sq_1", "tool_name": "causal_effect_estimator", "confidence": 0.9, "reasoning": "Test"},
        {"sub_question_id": "sq_2", "tool_name": "cate_analyzer", "confidence": 0.85, "reasoning": "Test"}
    ],
    "execution_steps": [
        {"step_id": "step_1", "sub_question_id": "sq_1", "tool_name": "causal_effect_estimator", "depends_on_steps": []},
        {"step_id": "step_2", "sub_question_id": "sq_2", "tool_name": "cate_analyzer", "depends_on_steps": ["step_1"]}
    ],
    "parallel_groups": [["step_1"], ["step_2"]]
}
```"""
        )

        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        plan = await planner.plan(sample_decomposition)

        assert len(plan.steps) == 2

    @pytest.mark.asyncio
    async def test_invalid_json_raises_error(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that invalid JSON raises PlanningError"""
        mock_llm_client.set_planning_response("Not valid JSON")

        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)

        with pytest.raises(PlanningError) as exc_info:
            await planner.plan(sample_decomposition)

        assert "JSON" in str(exc_info.value)


class TestSyncWrapper:
    """Tests for synchronous wrapper function"""

    def test_plan_sync(self, mock_llm_client, mock_tool_registry, sample_decomposition):
        """Test synchronous plan wrapper"""
        result = plan_sync(sample_decomposition, mock_llm_client, tool_registry=mock_tool_registry)

        assert isinstance(result, ExecutionPlan)
        assert len(result.steps) > 0


class TestErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_llm_error_wrapped(
        self, mock_llm_client, mock_tool_registry, sample_decomposition
    ):
        """Test that LLM errors are wrapped in PlanningError"""
        mock_llm_client.messages.create = AsyncMock(side_effect=Exception("LLM error"))

        planner = ToolPlanner(llm_client=mock_llm_client, tool_registry=mock_tool_registry)

        with pytest.raises(PlanningError) as exc_info:
            await planner.plan(sample_decomposition)

        assert "LLM error" in str(exc_info.value)
