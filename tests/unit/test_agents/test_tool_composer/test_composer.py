"""
Tests for Tool Composer Main Orchestrator

Integration tests for the full 4-phase tool composition pipeline.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.tool_composer.composer import (
    ToolComposer,
    ToolComposerIntegration,
    compose_query,
    compose_query_sync,
)
from src.agents.tool_composer.models.composition_models import (
    CompositionResult,
)


class TestToolComposerInit:
    """Tests for ToolComposer initialization"""

    def test_default_initialization(self, mock_llm_client, mock_tool_registry):
        """Test default initialization"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        assert composer.llm_client == mock_llm_client
        assert composer.registry == mock_tool_registry
        assert composer.decomposer is not None
        assert composer.planner is not None
        assert composer.executor is not None
        assert composer.synthesizer is not None

    def test_initialization_with_config(self, mock_llm_client, mock_tool_registry):
        """Test initialization with custom config"""
        config = {
            "phases": {
                "decompose": {"model": "custom-decompose-model", "temperature": 0.1},
                "plan": {"max_tools_per_plan": 10},
                "execute": {"parallel_execution_limit": 5},
                "synthesize": {"max_tokens": 3000},
            }
        }
        composer = ToolComposer(
            llm_client=mock_llm_client, tool_registry=mock_tool_registry, config=config
        )
        assert composer.decomposer.model == "custom-decompose-model"
        assert composer.decomposer.temperature == 0.1


class TestFullPipeline:
    """Tests for full pipeline execution"""

    @pytest.mark.asyncio
    async def test_successful_composition(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test successful full pipeline execution"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        assert isinstance(result, CompositionResult)
        assert result.success is True
        assert result.query == sample_query
        assert result.decomposition is not None
        assert result.plan is not None
        assert result.execution is not None
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_composition_returns_answer(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that composition returns a meaningful answer"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        assert result.response.answer != ""
        assert 0 <= result.response.confidence <= 1

    @pytest.mark.asyncio
    async def test_composition_with_context(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test composition with context"""
        context = {"brand": "Remibrutinib", "region": "Northeast", "time_period": "Q1 2024"}
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query, context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_composition_records_phases(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that composition records phase durations"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        assert "decompose" in result.phase_durations
        assert "plan" in result.phase_durations
        assert "execute" in result.phase_durations
        assert "synthesize" in result.phase_durations

    @pytest.mark.asyncio
    async def test_composition_records_total_duration(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that total duration is recorded"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        # Duration can be 0ms with fast mocks, but must be non-negative
        assert result.total_duration_ms >= 0
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at


class TestErrorHandling:
    """Tests for error handling during composition"""

    @pytest.mark.asyncio
    async def test_decomposition_error(self, mock_llm_client, mock_tool_registry):
        """Test handling of decomposition errors"""
        # Configure mock to return invalid response
        mock_llm_client.set_decomposition_response("Not valid JSON at all")

        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose("Test query")

        assert result.success is False
        assert "decompose" in result.response.failed_components
        assert result.response.confidence == 0.0

    @pytest.mark.asyncio
    async def test_planning_error(self, mock_llm_client, mock_tool_registry):
        """Test handling of planning errors"""
        # Configure mock to succeed on decomposition but fail on planning
        mock_llm_client.set_decomposition_response(
            json.dumps(
                {
                    "reasoning": "Test",
                    "sub_questions": [
                        {"id": "sq_1", "question": "Q1", "intent": "CAUSAL"},
                        {"id": "sq_2", "question": "Q2", "intent": "DESCRIPTIVE"},
                    ],
                }
            )
        )
        mock_llm_client.set_planning_response("Not valid JSON")

        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose("Test query")

        assert result.success is False
        assert "plan" in result.response.failed_components

    @pytest.mark.asyncio
    async def test_unexpected_error(self, mock_llm_client, mock_tool_registry):
        """Test handling of unexpected errors"""
        # Use the LangChain interface to inject an error
        mock_llm_client.set_error(RuntimeError("Unexpected error"))

        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose("Test query")

        assert result.success is False
        assert result.error is not None
        assert "Unexpected" in result.error

    @pytest.mark.asyncio
    async def test_error_result_structure(self, mock_llm_client, mock_tool_registry):
        """Test that error results have proper structure"""
        mock_llm_client.set_decomposition_response("invalid")

        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose("Test query")

        # Even failed results should have all fields
        assert result.query == "Test query"
        assert result.decomposition is not None
        assert result.plan is not None
        assert result.execution is not None
        assert result.response is not None


class TestPhaseExecution:
    """Tests for individual phase execution"""

    @pytest.mark.asyncio
    async def test_decomposition_phase(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test decomposition phase creates sub-questions"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        assert result.decomposition.question_count >= 2
        assert result.decomposition.original_query == sample_query

    @pytest.mark.asyncio
    async def test_planning_phase(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test planning phase creates execution steps"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        assert result.plan.step_count >= 1
        assert len(result.plan.tool_mappings) >= 1

    @pytest.mark.asyncio
    async def test_execution_phase(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test execution phase runs tools"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        # Execution should have attempted to run tools
        assert result.execution.plan_id is not None

    @pytest.mark.asyncio
    async def test_synthesis_phase(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test synthesis phase creates response"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        assert result.response.answer != ""
        assert result.response.response_id.startswith("resp_")


@pytest.mark.xdist_group(name="sync_wrappers")
class TestConvenienceFunctions:
    """Tests for convenience functions"""

    @pytest.mark.asyncio
    async def test_compose_query(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test compose_query convenience function"""
        result = await compose_query(
            sample_query, mock_llm_client, tool_registry=mock_tool_registry
        )

        assert isinstance(result, CompositionResult)
        assert result.query == sample_query

    @pytest.mark.asyncio
    async def test_compose_query_with_context(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test compose_query with context"""
        context = {"brand": "Remibrutinib"}
        result = await compose_query(
            sample_query, mock_llm_client, context=context, tool_registry=mock_tool_registry
        )

        assert isinstance(result, CompositionResult)

    def test_compose_query_sync(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test synchronous compose_query_sync wrapper"""
        result = compose_query_sync(sample_query, mock_llm_client, tool_registry=mock_tool_registry)

        assert isinstance(result, CompositionResult)


class TestToolComposerIntegration:
    """Tests for ToolComposerIntegration class"""

    @pytest.mark.asyncio
    async def test_handle_multi_faceted_query(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test handling multi-faceted query from orchestrator"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        integration = ToolComposerIntegration(composer)

        response = await integration.handle_multi_faceted_query(
            query=sample_query,
            extracted_entities={"brand": "Remibrutinib"},
            user_context={"region": "Northeast"},
        )

        assert "success" in response
        assert "response" in response
        assert "confidence" in response
        assert "metadata" in response

    @pytest.mark.asyncio
    async def test_integration_response_format(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that integration response has correct format"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        integration = ToolComposerIntegration(composer)

        response = await integration.handle_multi_faceted_query(
            query=sample_query, extracted_entities={}, user_context={}
        )

        # Check metadata structure
        assert "composition_id" in response["metadata"]
        assert "sub_questions" in response["metadata"]
        assert "tools_executed" in response["metadata"]
        assert "total_duration_ms" in response["metadata"]
        assert "phase_durations" in response["metadata"]

    @pytest.mark.asyncio
    async def test_integration_merges_context(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that integration merges entity and user context"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        integration = ToolComposerIntegration(composer)

        # Both contexts should be available
        response = await integration.handle_multi_faceted_query(
            query=sample_query,
            extracted_entities={"brand": "Remibrutinib", "hcp_type": "Oncologist"},
            user_context={"region": "Northeast", "time_period": "Q1 2024"},
        )

        assert response["success"] is True


class TestLogging:
    """Tests for logging behavior"""

    @pytest.mark.asyncio
    async def test_logging_on_composition(
        self, mock_llm_client, mock_tool_registry, sample_query, caplog
    ):
        """Test that composition logs key events"""
        import logging

        caplog.set_level(logging.INFO)

        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        await composer.compose(sample_query)

        # Should log phase completion
        log_text = caplog.text
        assert "Phase 1" in log_text or "Decomposing" in log_text


class TestEdgeCases:
    """Tests for edge cases"""

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_llm_client, mock_tool_registry):
        """Test handling of empty query"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        # Empty query should still be processed (may fail at decomposition)
        result = await composer.compose("")
        assert isinstance(result, CompositionResult)

    @pytest.mark.asyncio
    async def test_very_long_query(self, mock_llm_client, mock_tool_registry):
        """Test handling of very long query"""
        long_query = "What is the causal effect? " * 100

        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(long_query)

        assert isinstance(result, CompositionResult)

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, mock_llm_client, mock_tool_registry):
        """Test handling of special characters in query"""
        query = "What's the effect of X → Y with α=0.05 & β=0.80?"

        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(query)

        assert isinstance(result, CompositionResult)

    @pytest.mark.asyncio
    async def test_none_context(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test composition with None context"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query, context=None)

        assert result.success is True


class TestResultSummary:
    """Tests for composition result summary"""

    @pytest.mark.asyncio
    async def test_to_summary(self, mock_llm_client, mock_tool_registry, sample_query):
        """Test to_summary method on result"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        summary = result.to_summary()

        assert "composition_id" in summary
        assert "sub_questions" in summary
        assert "tools_executed" in summary
        assert "success" in summary


class TestPhaseDurationCalculation:
    """Tests for phase duration calculation"""

    @pytest.mark.asyncio
    async def test_phase_durations_are_positive(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that all phase durations are positive"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        for phase, duration in result.phase_durations.items():
            assert duration >= 0, f"{phase} had negative duration"

    @pytest.mark.asyncio
    async def test_total_duration_sums_phases(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that total duration roughly equals sum of phases"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        result = await composer.compose(sample_query)

        phase_sum = sum(result.phase_durations.values())
        # Total should be close to phase sum (allowing for small overhead)
        assert abs(result.total_duration_ms - phase_sum) < 100  # 100ms tolerance


class TestRegistryInteraction:
    """Tests for tool registry interaction"""

    def test_uses_provided_registry(self, mock_llm_client, mock_tool_registry):
        """Test that composer uses provided registry"""
        composer = ToolComposer(llm_client=mock_llm_client, tool_registry=mock_tool_registry)
        assert composer.registry == mock_tool_registry

    def test_creates_default_registry_if_not_provided(self, mock_llm_client):
        """Test that composer creates default registry if not provided"""
        composer = ToolComposer(llm_client=mock_llm_client)
        assert composer.registry is not None


class TestMemoryContribution:
    """Tests for memory contribution after composition (G1, G2)"""

    @pytest.mark.asyncio
    async def test_memory_contribution_enabled_by_default(
        self, mock_llm_client, mock_tool_registry
    ):
        """Test that memory contribution is enabled by default"""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )
        assert composer.enable_memory_contribution is True

    @pytest.mark.asyncio
    async def test_memory_contribution_can_be_disabled(
        self, mock_llm_client, mock_tool_registry
    ):
        """Test that memory contribution can be disabled"""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            enable_memory_contribution=False,
        )
        assert composer.enable_memory_contribution is False

    @pytest.mark.asyncio
    async def test_memory_hooks_initialized(
        self, mock_llm_client, mock_tool_registry
    ):
        """Test that memory hooks are initialized"""
        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )
        # Should have memory_hooks attribute (even if None from factory)
        assert hasattr(composer, "memory_hooks")

    @pytest.mark.asyncio
    async def test_custom_memory_hooks_accepted(
        self, mock_llm_client, mock_tool_registry
    ):
        """Test that custom memory hooks can be provided"""
        mock_memory_hooks = AsyncMock()

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            memory_hooks=mock_memory_hooks,
        )

        assert composer.memory_hooks == mock_memory_hooks

    @pytest.mark.asyncio
    async def test_contribute_to_memory_called_on_success(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that memory contribution is called after successful composition"""
        mock_memory_hooks = AsyncMock()

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            memory_hooks=mock_memory_hooks,
            enable_memory_contribution=True,
        )

        # Patch the contribute_to_memory function
        with patch(
            "src.agents.tool_composer.composer.contribute_to_memory",
            new_callable=AsyncMock,
            return_value={"episodic_stored": 1, "procedural_stored": 1, "working_cached": 1},
        ) as mock_contribute:
            await composer.compose(sample_query)

            # Verify contribute was called
            mock_contribute.assert_called_once()
            call_kwargs = mock_contribute.call_args.kwargs
            assert "result" in call_kwargs
            assert call_kwargs["memory_hooks"] == mock_memory_hooks

    @pytest.mark.asyncio
    async def test_contribute_to_memory_not_called_when_disabled(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that memory contribution is not called when disabled"""
        mock_memory_hooks = AsyncMock()

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            memory_hooks=mock_memory_hooks,
            enable_memory_contribution=False,  # Disabled
        )

        with patch(
            "src.agents.tool_composer.composer.contribute_to_memory",
            new_callable=AsyncMock,
        ) as mock_contribute:
            await composer.compose(sample_query)

            # Should NOT be called when disabled
            mock_contribute.assert_not_called()

    @pytest.mark.asyncio
    async def test_contribute_to_memory_error_does_not_fail_composition(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that memory contribution errors don't fail the composition"""
        mock_memory_hooks = AsyncMock()

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            memory_hooks=mock_memory_hooks,
            enable_memory_contribution=True,
        )

        with patch(
            "src.agents.tool_composer.composer.contribute_to_memory",
            new_callable=AsyncMock,
            side_effect=Exception("Memory write failed"),
        ):
            # Should still complete successfully
            result = await composer.compose(sample_query)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_contribute_to_memory_receives_context(
        self, mock_llm_client, mock_tool_registry, sample_query
    ):
        """Test that memory contribution receives context from compose call"""
        mock_memory_hooks = AsyncMock()

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            memory_hooks=mock_memory_hooks,
            enable_memory_contribution=True,
        )

        context = {
            "session_id": "test-session-123",
            "brand": "Kisqali",
            "region": "Northeast",
        }

        with patch(
            "src.agents.tool_composer.composer.contribute_to_memory",
            new_callable=AsyncMock,
            return_value={"episodic_stored": 1, "procedural_stored": 1, "working_cached": 1},
        ) as mock_contribute:
            await composer.compose(sample_query, context=context)

            call_kwargs = mock_contribute.call_args.kwargs
            assert call_kwargs["session_id"] == "test-session-123"
            assert call_kwargs["brand"] == "Kisqali"
            assert call_kwargs["region"] == "Northeast"

    @pytest.mark.asyncio
    async def test_memory_hooks_passed_to_planner(
        self, mock_llm_client, mock_tool_registry
    ):
        """Test that memory hooks are passed to planner for episodic lookup"""
        mock_memory_hooks = AsyncMock()

        composer = ToolComposer(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            memory_hooks=mock_memory_hooks,
        )

        # Verify planner has memory hooks
        assert composer.planner.memory_hooks == mock_memory_hooks
        assert composer.planner.use_episodic_memory is True
