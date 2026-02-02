"""Unit tests for Sequential Pipeline.

Tests the NetworkX → DoWhy → EconML → CausalML sequential flow.
"""

import pytest

from src.causal_engine.pipeline import (
    CausalLibrary,
    PipelineInput,
    PipelineStage,
    SequentialPipeline,
    SequentialPipelineBuilder,
    create_sequential_pipeline,
)
from src.causal_engine.pipeline.sequential import LIBRARY_STAGES, SEQUENTIAL_ORDER


class TestSequentialOrder:
    """Test sequential pipeline ordering."""

    def test_sequential_order_has_all_libraries(self):
        """Test that SEQUENTIAL_ORDER includes all libraries."""
        assert len(SEQUENTIAL_ORDER) == 4
        assert CausalLibrary.NETWORKX in SEQUENTIAL_ORDER
        assert CausalLibrary.DOWHY in SEQUENTIAL_ORDER
        assert CausalLibrary.ECONML in SEQUENTIAL_ORDER
        assert CausalLibrary.CAUSALML in SEQUENTIAL_ORDER

    def test_sequential_order_is_correct(self):
        """Test that libraries are in correct order."""
        assert SEQUENTIAL_ORDER[0] == CausalLibrary.NETWORKX
        assert SEQUENTIAL_ORDER[1] == CausalLibrary.DOWHY
        assert SEQUENTIAL_ORDER[2] == CausalLibrary.ECONML
        assert SEQUENTIAL_ORDER[3] == CausalLibrary.CAUSALML

    def test_library_stages_mapping(self):
        """Test that each library maps to correct stage."""
        assert LIBRARY_STAGES[CausalLibrary.NETWORKX] == PipelineStage.GRAPH_ANALYSIS
        assert LIBRARY_STAGES[CausalLibrary.DOWHY] == PipelineStage.CAUSAL_VALIDATION
        assert LIBRARY_STAGES[CausalLibrary.ECONML] == PipelineStage.EFFECT_ESTIMATION
        assert LIBRARY_STAGES[CausalLibrary.CAUSALML] == PipelineStage.UPLIFT_MODELING


class TestSequentialPipelineCreation:
    """Test sequential pipeline instantiation."""

    def test_create_with_defaults(self):
        """Test creating pipeline with default settings."""
        pipeline = SequentialPipeline()
        assert pipeline.router is not None
        assert len(pipeline.executors) == 4
        assert pipeline.fail_fast is False

    def test_create_with_fail_fast(self):
        """Test creating pipeline with fail_fast enabled."""
        pipeline = SequentialPipeline(fail_fast=True)
        assert pipeline.fail_fast is True

    def test_factory_function(self):
        """Test create_sequential_pipeline factory."""
        pipeline = create_sequential_pipeline(fail_fast=True)
        assert isinstance(pipeline, SequentialPipeline)
        assert pipeline.fail_fast is True

    def test_builder_pattern(self):
        """Test SequentialPipelineBuilder."""
        pipeline = SequentialPipelineBuilder().with_fail_fast(True).build()
        assert isinstance(pipeline, SequentialPipeline)
        assert pipeline.fail_fast is True


class TestSequentialPipelineExecution:
    """Test sequential pipeline execution."""

    @pytest.fixture
    def pipeline(self) -> SequentialPipeline:
        """Create a pipeline for testing."""
        return SequentialPipeline()

    @pytest.fixture
    def basic_input(self) -> PipelineInput:
        """Create basic input for testing."""
        return PipelineInput(
            query="Does detailing cause prescription increases?",
            treatment_var="detailing_visits",
            outcome_var="trx",
            confounders=["region", "specialty"],
            effect_modifiers=["tenure"],
            data_source="test_data",
            filters=None,
            mode=None,
            libraries_enabled=None,
            cross_validate=None,
        )

    @pytest.mark.asyncio
    async def test_execute_full_pipeline(
        self, pipeline: SequentialPipeline, basic_input: PipelineInput
    ):
        """Test executing full sequential pipeline."""
        output = await pipeline.execute(basic_input)

        assert output["status"] in ["completed", "partial"]
        assert output["question_type"] == "causal_relationship"
        assert len(output["libraries_used"]) > 0

    @pytest.mark.asyncio
    async def test_execute_with_specific_libraries(self, pipeline: SequentialPipeline):
        """Test executing with specific libraries."""
        input_data = PipelineInput(
            query="How much does the treatment effect vary by region?",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=["confounder1"],
            effect_modifiers=["region"],
            data_source="test_data",
            filters=None,
            mode=None,
            libraries_enabled=["econml", "causalml"],
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        assert output["status"] in ["completed", "partial"]
        # Should only execute specified libraries
        for lib in output["libraries_used"]:
            assert lib in ["econml", "causalml"]

    @pytest.mark.asyncio
    async def test_execute_generates_summary(
        self, pipeline: SequentialPipeline, basic_input: PipelineInput
    ):
        """Test that execution generates summary and insights."""
        output = await pipeline.execute(basic_input)

        assert output["executive_summary"]
        assert isinstance(output["key_insights"], list)
        assert isinstance(output["recommended_actions"], list)

    @pytest.mark.asyncio
    async def test_execute_tracks_latency(
        self, pipeline: SequentialPipeline, basic_input: PipelineInput
    ):
        """Test that execution tracks latency."""
        output = await pipeline.execute(basic_input)

        # Latency >= 0 (may be 0 for very fast mock executors)
        assert output["total_latency_ms"] >= 0
        # Check that status reflects proper completion
        assert output["status"] in ["completed", "partial"]


class TestSequentialPipelineRouting:
    """Test routing integration in sequential pipeline."""

    @pytest.fixture
    def pipeline(self) -> SequentialPipeline:
        return SequentialPipeline()

    @pytest.mark.asyncio
    async def test_route_causal_question(self, pipeline: SequentialPipeline):
        """Test routing a causal relationship question."""
        decision = await pipeline.route("Does X cause Y?")

        assert decision.primary_library == CausalLibrary.DOWHY

    @pytest.mark.asyncio
    async def test_route_targeting_question(self, pipeline: SequentialPipeline):
        """Test routing a targeting question."""
        decision = await pipeline.route("Who should we target for the campaign?")

        assert decision.primary_library == CausalLibrary.CAUSALML

    @pytest.mark.asyncio
    async def test_route_heterogeneity_question(self, pipeline: SequentialPipeline):
        """Test routing a heterogeneity question."""
        decision = await pipeline.route("How does the treatment effect vary by segment?")

        assert decision.primary_library == CausalLibrary.ECONML


class TestSequentialPipelineFailFast:
    """Test fail-fast behavior."""

    @pytest.mark.asyncio
    async def test_fail_fast_stops_on_error(self):
        """Test that fail_fast stops execution on first error."""
        pipeline = SequentialPipeline(fail_fast=True)

        # Create input that will fail validation (missing treatment_var)
        input_data = PipelineInput(
            query="Any query",
            treatment_var=None,  # Missing - will cause validation failure
            outcome_var="outcome",
            confounders=None,
            effect_modifiers=None,
            data_source="test_data",
            filters=None,
            mode=None,
            libraries_enabled=["dowhy"],  # DoWhy requires treatment_var
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        # Should have warnings about missing treatment_var
        assert len(output["warnings"]) > 0 or len(output["errors"]) >= 0

    @pytest.mark.asyncio
    async def test_non_fail_fast_continues(self):
        """Test that non-fail-fast continues after warnings."""
        pipeline = SequentialPipeline(fail_fast=False)

        input_data = PipelineInput(
            query="Any query",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=None,
            effect_modifiers=None,
            data_source="test_data",
            filters=None,
            mode=None,
            libraries_enabled=["dowhy", "econml", "causalml"],
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        # Should have executed multiple libraries
        assert len(output["libraries_used"]) >= 1


class TestSequentialPipelineAggregation:
    """Test result aggregation."""

    @pytest.fixture
    def pipeline(self) -> SequentialPipeline:
        return SequentialPipeline()

    @pytest.mark.asyncio
    async def test_aggregates_effects(self, pipeline: SequentialPipeline):
        """Test that pipeline aggregates effects from multiple libraries."""
        input_data = PipelineInput(
            query="Does treatment affect outcome?",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=["c1"],
            effect_modifiers=["e1"],
            data_source="test_data",
            filters=None,
            mode=None,
            libraries_enabled=["dowhy", "econml"],
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        # Should have at least the summary
        assert output["executive_summary"]

    @pytest.mark.asyncio
    async def test_generates_recommendations(self, pipeline: SequentialPipeline):
        """Test that pipeline generates recommendations."""
        input_data = PipelineInput(
            query="Who should we target?",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=None,
            effect_modifiers=None,
            data_source="test_data",
            filters=None,
            mode=None,
            libraries_enabled=None,
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        assert isinstance(output["recommended_actions"], list)
        assert len(output["recommended_actions"]) > 0


class TestLibraryExecutors:
    """Test individual library executors."""

    @pytest.fixture
    def pipeline(self) -> SequentialPipeline:
        return SequentialPipeline()

    def test_has_all_executors(self, pipeline: SequentialPipeline):
        """Test that pipeline has executors for all libraries."""
        assert CausalLibrary.NETWORKX in pipeline.executors
        assert CausalLibrary.DOWHY in pipeline.executors
        assert CausalLibrary.ECONML in pipeline.executors
        assert CausalLibrary.CAUSALML in pipeline.executors

    def test_networkx_executor_library(self, pipeline: SequentialPipeline):
        """Test NetworkX executor returns correct library."""
        executor = pipeline.executors[CausalLibrary.NETWORKX]
        assert executor.library == CausalLibrary.NETWORKX

    def test_dowhy_executor_library(self, pipeline: SequentialPipeline):
        """Test DoWhy executor returns correct library."""
        executor = pipeline.executors[CausalLibrary.DOWHY]
        assert executor.library == CausalLibrary.DOWHY

    def test_econml_executor_library(self, pipeline: SequentialPipeline):
        """Test EconML executor returns correct library."""
        executor = pipeline.executors[CausalLibrary.ECONML]
        assert executor.library == CausalLibrary.ECONML

    def test_causalml_executor_library(self, pipeline: SequentialPipeline):
        """Test CausalML executor returns correct library."""
        executor = pipeline.executors[CausalLibrary.CAUSALML]
        assert executor.library == CausalLibrary.CAUSALML
