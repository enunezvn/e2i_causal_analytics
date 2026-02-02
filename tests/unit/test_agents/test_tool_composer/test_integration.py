"""
Integration tests for Tool Composer 4-phase pipeline.

Tests cover:
- Full pipeline execution (decompose -> plan -> execute -> synthesize)
- Phase handoffs and state propagation
- Error handling and partial success scenarios
- Contract compliance for all models
- Edge cases (single question, all parallel, deep dependencies)
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.tool_composer.composer import (
    ToolComposer,
    ToolComposerIntegration,
    compose_query,
)
from src.agents.tool_composer.decomposer import QueryDecomposer
from src.agents.tool_composer.executor import PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    ComposedResponse,
    CompositionResult,
    DecompositionResult,
    ExecutionPlan,
    ExecutionTrace,
    SubQuestion,
)
from src.agents.tool_composer.planner import ToolPlanner
from src.agents.tool_composer.synthesizer import ResponseSynthesizer


class TestFullPipelineIntegration:
    """End-to-end tests for the complete composition pipeline."""

    @pytest.mark.asyncio
    async def test_simple_query_end_to_end(self, mock_llm_client, mock_tool_registry):
        """Test complete pipeline with a simple multi-faceted query."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("What is the causal effect of rep visits on Rx volume?")

        assert result is not None
        assert result.success is True
        assert result.query == "What is the causal effect of rep visits on Rx volume?"
        assert result.response is not None
        assert result.response.answer is not None
        assert len(result.response.answer) > 0

    @pytest.mark.asyncio
    async def test_complex_query_with_dependencies(self, mock_llm_client, mock_tool_registry):
        """Test pipeline with query requiring sequential execution."""
        # Configure response with dependencies
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Needs causal then regional analysis",
                    "sub_questions": [
                        {
                            "id": "sq_1",
                            "question": "What is the causal effect of rep visits?",
                            "intent": "CAUSAL",
                            "entities": ["rep_visits"],
                            "depends_on": [],
                        },
                        {
                            "id": "sq_2",
                            "question": "How does this vary by region?",
                            "intent": "COMPARATIVE",
                            "entities": ["region"],
                            "depends_on": ["sq_1"],
                        },
                    ],
                }
            )
        )

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Compare causal impact of rep visits by region")

        assert result.success is True
        assert result.decomposition.question_count == 2
        # Check that dependency exists in decomposition
        sq2 = result.decomposition.sub_questions[1]
        assert "sq_1" in sq2.depends_on

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self, mock_llm_client, mock_tool_registry):
        """Test pipeline executes independent tools in parallel groups."""
        # Configure all-parallel decomposition
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Three independent analyses",
                    "sub_questions": [
                        {
                            "id": "sq_1",
                            "question": "Effect of rep visits?",
                            "intent": "CAUSAL",
                            "entities": ["rep_visits"],
                            "depends_on": [],
                        },
                        {
                            "id": "sq_2",
                            "question": "Effect of speaker programs?",
                            "intent": "CAUSAL",
                            "entities": ["speaker_programs"],
                            "depends_on": [],
                        },
                    ],
                }
            )
        )

        mock_llm_client.set_planning_response(
            json.dumps(
                {
                    "reasoning": "Both can run in parallel",
                    "tool_mappings": [
                        {
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "confidence": 0.9,
                            "reasoning": "Matches causal intent",
                        },
                        {
                            "sub_question_id": "sq_2",
                            "tool_name": "causal_effect_estimator",
                            "confidence": 0.9,
                            "reasoning": "Matches causal intent",
                        },
                    ],
                    "execution_steps": [
                        {
                            "step_id": "step_1",
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "input_mapping": {"treatment": "rep_visits", "outcome": "rx"},
                            "depends_on_steps": [],
                        },
                        {
                            "step_id": "step_2",
                            "sub_question_id": "sq_2",
                            "tool_name": "causal_effect_estimator",
                            "input_mapping": {"treatment": "speaker_programs", "outcome": "rx"},
                            "depends_on_steps": [],
                        },
                    ],
                    "parallel_groups": [["step_1", "step_2"]],  # Both in same group
                }
            )
        )

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Compare rep visits vs speaker programs")

        assert result.success is True
        # Both steps should be in the same parallel group
        assert len(result.plan.parallel_groups) == 1
        assert "step_1" in result.plan.parallel_groups[0]
        assert "step_2" in result.plan.parallel_groups[0]

    @pytest.mark.asyncio
    async def test_error_recovery_partial_success(self, mock_llm_client, mock_tool_registry):
        """Test pipeline handles partial tool failures gracefully."""
        # Register a failing tool alongside working ones
        from src.tool_registry.registry import ToolParameter, ToolSchema

        def failing_tool(**kwargs):
            raise ValueError("Simulated failure")

        failing_schema = ToolSchema(
            name="failing_tool",
            description="A tool that always fails",
            source_agent="test",
            tier=2,
            input_parameters=[ToolParameter("input", "str", "Input", True)],
            output_schema="FailResult",
            avg_execution_ms=100,
        )
        mock_tool_registry.register(schema=failing_schema, callable=failing_tool)

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        # Even with failures, composer should return a result
        result = await composer.compose("Test query")

        assert result is not None
        # The synthesis phase should acknowledge any failures

    @pytest.mark.asyncio
    async def test_context_propagation(self, mock_llm_client, mock_tool_registry):
        """Test that context is propagated through the pipeline."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        context = {
            "brand": "Kisqali",
            "region": "Northeast",
            "time_period": "Q4_2024",
        }

        result = await composer.compose(
            "What is the causal effect for Kisqali in Northeast?",
            context=context,
        )

        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_full_trace_recorded(self, mock_llm_client, mock_tool_registry):
        """Test that complete execution trace is recorded."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Compare causal impact of different interventions")

        assert result is not None
        # Check that all phases recorded durations
        assert "decompose" in result.phase_durations
        assert "plan" in result.phase_durations
        assert "execute" in result.phase_durations
        assert "synthesize" in result.phase_durations

        # Check execution trace exists
        assert result.execution is not None
        assert hasattr(result.execution, "plan_id")

    @pytest.mark.asyncio
    async def test_composition_id_generated(self, mock_llm_client, mock_tool_registry):
        """Test that composition_id is generated for tracking."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Test query")

        assert result is not None
        assert hasattr(result, "composition_id")
        assert result.composition_id is not None
        assert len(result.composition_id) > 0


class TestPhaseHandoffs:
    """Tests for data handoffs between pipeline phases."""

    @pytest.mark.asyncio
    async def test_decomposition_to_planning_handoff(self, mock_llm_client, mock_tool_registry):
        """Test that decomposition results are passed to planner."""
        # Create decomposer and planner
        decomposer = QueryDecomposer(llm_client=mock_llm_client)
        planner = ToolPlanner(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        # Decompose
        decomposition = await decomposer.decompose("Test query")

        # Plan should accept decomposition
        plan = await planner.plan(decomposition)

        assert plan is not None
        # Plan should reference the decomposition
        assert plan.decomposition == decomposition
        assert len(plan.steps) >= 0

    @pytest.mark.asyncio
    async def test_planning_to_execution_handoff(
        self, mock_llm_client, mock_tool_registry, sample_execution_plan
    ):
        """Test that plan is correctly passed to executor."""
        executor = PlanExecutor(tool_registry=mock_tool_registry)

        # Execute the plan
        trace = await executor.execute(sample_execution_plan)

        assert trace is not None
        assert trace.plan_id == sample_execution_plan.plan_id

    @pytest.mark.asyncio
    async def test_execution_to_synthesis_handoff(self, mock_llm_client, sample_synthesis_input):
        """Test that execution trace is passed to synthesizer."""
        synthesizer = ResponseSynthesizer(llm_client=mock_llm_client)

        # Synthesize
        response = await synthesizer.synthesize(sample_synthesis_input)

        assert response is not None
        assert response.answer is not None

    @pytest.mark.asyncio
    async def test_state_preserved_across_phases(self, mock_llm_client, mock_tool_registry):
        """Test that key state is preserved across all phases."""
        original_query = "Original test query for state preservation"

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose(original_query)

        # Query should be preserved
        assert result.query == original_query
        # Decomposition should have the original query
        assert result.decomposition.original_query == original_query


class TestContractCompliance:
    """Tests verifying contract compliance for composition models."""

    def test_composition_result_contract(self):
        """Test CompositionResult has all required fields."""
        decomposition = DecompositionResult(
            original_query="Test",
            sub_questions=[],
            decomposition_reasoning="Test",
        )
        plan = ExecutionPlan(
            decomposition=decomposition,
            steps=[],
            tool_mappings=[],
            planning_reasoning="Test",
        )
        execution = ExecutionTrace(plan_id=plan.plan_id)
        response = ComposedResponse(answer="Test answer", confidence=0.9)

        result = CompositionResult(
            query="Test",
            decomposition=decomposition,
            plan=plan,
            execution=execution,
            response=response,
            total_duration_ms=100,
            phase_durations={},
            success=True,
        )

        # Check required fields exist
        assert result.query is not None
        assert result.decomposition is not None
        assert result.plan is not None
        assert result.execution is not None
        assert result.response is not None
        assert result.total_duration_ms >= 0
        assert result.success in [True, False]
        assert hasattr(result, "composition_id")

    def test_decomposition_result_contract(self):
        """Test DecompositionResult has all required fields."""
        result = DecompositionResult(
            original_query="Test query",
            sub_questions=[
                SubQuestion(
                    id="sq_1",
                    question="What?",
                    intent="CAUSAL",
                    entities=["test"],
                    depends_on=[],
                )
            ],
            decomposition_reasoning="Reasoning",
        )

        assert result.original_query is not None
        assert result.sub_questions is not None
        assert len(result.sub_questions) >= 0
        assert result.decomposition_reasoning is not None
        assert hasattr(result, "question_count")

    def test_execution_trace_contract(self):
        """Test ExecutionTrace has all required fields."""
        trace = ExecutionTrace(plan_id="test_plan")

        assert trace.plan_id is not None
        assert hasattr(trace, "step_results")  # Not "results"
        assert hasattr(trace, "started_at")
        assert hasattr(trace, "tools_executed")
        assert hasattr(trace, "tools_succeeded")

    def test_composed_response_contract(self):
        """Test ComposedResponse has all required fields."""
        response = ComposedResponse(
            answer="Test answer",
            confidence=0.85,
            supporting_data={"key": "value"},
            citations=["step_1"],
            caveats=["Note"],
            failed_components=[],
        )

        assert response.answer is not None
        assert 0 <= response.confidence <= 1
        assert hasattr(response, "supporting_data")
        assert hasattr(response, "citations")
        assert hasattr(response, "caveats")
        assert hasattr(response, "failed_components")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_minimal_sub_questions(self, mock_llm_client, mock_tool_registry):
        """Test pipeline handles minimal (2) sub-questions correctly.

        Note: Tool Composer is designed for MULTI-FACETED queries,
        so minimum 2 sub-questions is required by design.
        """
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Simple two-part question",
                    "sub_questions": [
                        {
                            "id": "sq_1",
                            "question": "What is the effect?",
                            "intent": "CAUSAL",
                            "entities": ["effect"],
                            "depends_on": [],
                        },
                        {
                            "id": "sq_2",
                            "question": "What is the magnitude?",
                            "intent": "DESCRIPTIVE",
                            "entities": ["magnitude"],
                            "depends_on": [],
                        },
                    ],
                }
            )
        )

        mock_llm_client.set_planning_response(
            json.dumps(
                {
                    "reasoning": "Two tools needed",
                    "tool_mappings": [
                        {
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "confidence": 0.95,
                            "reasoning": "Matches causal intent",
                        },
                        {
                            "sub_question_id": "sq_2",
                            "tool_name": "gap_calculator",
                            "confidence": 0.9,
                            "reasoning": "Provides magnitude",
                        },
                    ],
                    "execution_steps": [
                        {
                            "step_id": "step_1",
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "input_mapping": {"treatment": "x", "outcome": "y"},
                            "depends_on_steps": [],
                        },
                        {
                            "step_id": "step_2",
                            "sub_question_id": "sq_2",
                            "tool_name": "gap_calculator",
                            "input_mapping": {"metric": "effect"},
                            "depends_on_steps": [],
                        },
                    ],
                    "parallel_groups": [["step_1", "step_2"]],
                }
            )
        )

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Simple causal question with magnitude")

        assert result.success is True
        assert result.decomposition.question_count == 2

    @pytest.mark.asyncio
    async def test_all_parallel_steps(self, mock_llm_client, mock_tool_registry):
        """Test pipeline with all steps being parallel (no dependencies)."""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "All independent",
                    "sub_questions": [
                        {
                            "id": f"sq_{i}",
                            "question": f"Question {i}",
                            "intent": "CAUSAL",
                            "entities": [],
                            "depends_on": [],
                        }
                        for i in range(1, 4)
                    ],
                }
            )
        )

        # Add planning response with mappings for all 3 sub-questions
        mock_llm_client.set_planning_response(
            json.dumps(
                {
                    "reasoning": "All can run in parallel",
                    "tool_mappings": [
                        {
                            "sub_question_id": f"sq_{i}",
                            "tool_name": "causal_effect_estimator",
                            "confidence": 0.9,
                            "reasoning": f"Matches question {i}",
                        }
                        for i in range(1, 4)
                    ],
                    "execution_steps": [
                        {
                            "step_id": f"step_{i}",
                            "sub_question_id": f"sq_{i}",
                            "tool_name": "causal_effect_estimator",
                            "input_mapping": {"treatment": f"var_{i}", "outcome": "rx"},
                            "depends_on_steps": [],
                        }
                        for i in range(1, 4)
                    ],
                    "parallel_groups": [["step_1", "step_2", "step_3"]],
                }
            )
        )

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Multiple independent questions")

        assert result is not None
        assert result.decomposition.question_count == 3
        # All steps should be in a single parallel group
        assert len(result.plan.parallel_groups) == 1
        assert len(result.plan.parallel_groups[0]) == 3

    @pytest.mark.asyncio
    async def test_deep_dependency_chain(self, mock_llm_client, mock_tool_registry):
        """Test pipeline with deep dependency chain (A -> B -> C -> D)."""
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Sequential analysis chain",
                    "sub_questions": [
                        {
                            "id": "sq_1",
                            "question": "Step 1",
                            "intent": "CAUSAL",
                            "entities": [],
                            "depends_on": [],
                        },
                        {
                            "id": "sq_2",
                            "question": "Step 2",
                            "intent": "CAUSAL",
                            "entities": [],
                            "depends_on": ["sq_1"],
                        },
                        {
                            "id": "sq_3",
                            "question": "Step 3",
                            "intent": "CAUSAL",
                            "entities": [],
                            "depends_on": ["sq_2"],
                        },
                    ],
                }
            )
        )

        mock_llm_client.set_planning_response(
            json.dumps(
                {
                    "reasoning": "Fully sequential",
                    "tool_mappings": [
                        {
                            "sub_question_id": f"sq_{i}",
                            "tool_name": "causal_effect_estimator",
                            "confidence": 0.9,
                            "reasoning": "Sequential step",
                        }
                        for i in range(1, 4)
                    ],
                    "execution_steps": [
                        {
                            "step_id": "step_1",
                            "sub_question_id": "sq_1",
                            "tool_name": "causal_effect_estimator",
                            "input_mapping": {"treatment": "x", "outcome": "y"},
                            "depends_on_steps": [],
                        },
                        {
                            "step_id": "step_2",
                            "sub_question_id": "sq_2",
                            "tool_name": "causal_effect_estimator",
                            "input_mapping": {"treatment": "$step_1.effect", "outcome": "z"},
                            "depends_on_steps": ["step_1"],
                        },
                        {
                            "step_id": "step_3",
                            "sub_question_id": "sq_3",
                            "tool_name": "causal_effect_estimator",
                            "input_mapping": {"treatment": "$step_2.effect", "outcome": "w"},
                            "depends_on_steps": ["step_2"],
                        },
                    ],
                    "parallel_groups": [["step_1"], ["step_2"], ["step_3"]],
                }
            )
        )

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Deep chain analysis")

        assert result is not None
        # Should have 3 sequential parallel groups
        assert len(result.plan.parallel_groups) == 3

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_llm_client, mock_tool_registry):
        """Test pipeline handles empty query gracefully."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        # Empty query should still complete (may fail gracefully)
        result = await composer.compose("")

        assert result is not None
        # Either succeeds or has an error message
        assert result.response is not None


class TestOrchestratorIntegration:
    """Tests for integration with the Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_interface(self, mock_llm_client, mock_tool_registry):
        """Test ToolComposerIntegration interface."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )
        integration = ToolComposerIntegration(composer)

        result = await integration.handle_multi_faceted_query(
            query="Compare causal effects",
            extracted_entities={"brand": "Test"},
            user_context={"region": "Northeast"},
        )

        assert result is not None
        assert "success" in result
        assert "response" in result
        assert "confidence" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_orchestrator_response_format(self, mock_llm_client, mock_tool_registry):
        """Test response format matches Orchestrator expectations."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )
        integration = ToolComposerIntegration(composer)

        result = await integration.handle_multi_faceted_query(
            query="Test",
            extracted_entities={},
            user_context={},
        )

        # Check expected structure
        assert isinstance(result["success"], bool)
        assert isinstance(result["response"], str)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["metadata"], dict)
        assert "composition_id" in result["metadata"]
        assert "sub_questions" in result["metadata"]
        assert "tools_executed" in result["metadata"]
        assert "total_duration_ms" in result["metadata"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_compose_query_function(self, mock_llm_client, mock_tool_registry):
        """Test compose_query convenience function."""
        result = await compose_query(
            query="Test query",
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        assert result is not None
        assert isinstance(result, CompositionResult)


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_decomposition_error_handled(self, mock_tool_registry):
        """Test that decomposition errors are handled gracefully."""
        # Create a client that returns invalid JSON
        bad_client = Mock()
        bad_client.messages = Mock()
        bad_client.messages.create = AsyncMock(
            return_value=Mock(content=[Mock(text="invalid json {")])
        )

        composer = ToolComposer(
            llm_client=bad_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Test query")

        # Should return error result, not crash
        assert result is not None
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_planning_error_handled(self, mock_llm_client, mock_tool_registry):
        """Test that planning errors are handled gracefully."""
        # Set valid decomposition but invalid planning response
        mock_llm_client.set_planning_response("not valid json {{{")

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Test query")

        assert result is not None
        # May succeed or fail, but should not crash
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_execution_timeout_handled(self, mock_llm_client, mock_tool_registry):
        """Test that execution timeouts are handled."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            config={
                "phases": {
                    "execute": {
                        "max_execution_time_seconds": 0.001,  # Very short timeout
                    }
                }
            },
        )

        # This may or may not timeout, but shouldn't crash
        result = await composer.compose("Test query")

        assert result is not None


class TestPhaseMetrics:
    """Tests for phase timing and metrics."""

    @pytest.mark.asyncio
    async def test_phase_durations_recorded(self, mock_llm_client, mock_tool_registry):
        """Test that all phase durations are recorded."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Test query")

        assert result.phase_durations is not None
        assert "decompose" in result.phase_durations
        assert "plan" in result.phase_durations
        assert "execute" in result.phase_durations
        assert "synthesize" in result.phase_durations

        # All should be non-negative
        for phase, duration in result.phase_durations.items():
            assert duration >= 0, f"{phase} has negative duration"

    @pytest.mark.asyncio
    async def test_total_duration_is_sum(self, mock_llm_client, mock_tool_registry):
        """Test that total duration is at least sum of phases."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        result = await composer.compose("Test query")

        phase_sum = sum(result.phase_durations.values())
        # Total should be at least sum of phases (may include overhead)
        assert result.total_duration_ms >= phase_sum - 10  # Allow 10ms margin


class TestToolRegistryInteraction:
    """Tests for interaction with the tool registry."""

    @pytest.mark.asyncio
    async def test_composer_uses_provided_registry(self, mock_llm_client, mock_tool_registry):
        """Test that composer uses the provided registry."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )

        assert composer.registry == mock_tool_registry

    @pytest.mark.asyncio
    async def test_empty_registry_handled(self, mock_llm_client, empty_registry):
        """Test that empty registry is handled gracefully."""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=empty_registry,
        )

        # May fail, but should not crash
        result = await composer.compose("Test query")
        assert result is not None
