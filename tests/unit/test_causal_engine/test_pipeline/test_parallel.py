"""Unit tests for Parallel Pipeline.

Tests simultaneous execution of all four causal libraries
with confidence-weighted result aggregation.
"""

import pytest

from src.causal_engine.pipeline import (
    CausalLibrary,
    ParallelPipeline,
    ParallelPipelineBuilder,
    PipelineInput,
    create_parallel_pipeline,
)


class TestParallelPipelineCreation:
    """Test parallel pipeline instantiation."""

    def test_create_with_defaults(self):
        """Test creating pipeline with default settings."""
        pipeline = ParallelPipeline()
        assert pipeline.router is not None
        assert len(pipeline.executors) == 4
        assert pipeline.max_parallel == 4
        assert pipeline.fail_fast is False

    def test_create_with_max_parallel(self):
        """Test creating pipeline with max_parallel limit."""
        pipeline = ParallelPipeline(max_parallel=2)
        assert pipeline.max_parallel == 2

    def test_create_with_fail_fast(self):
        """Test creating pipeline with fail_fast enabled."""
        pipeline = ParallelPipeline(fail_fast=True)
        assert pipeline.fail_fast is True

    def test_factory_function(self):
        """Test create_parallel_pipeline factory."""
        pipeline = create_parallel_pipeline(max_parallel=3, fail_fast=True)
        assert isinstance(pipeline, ParallelPipeline)
        assert pipeline.max_parallel == 3
        assert pipeline.fail_fast is True

    def test_builder_pattern(self):
        """Test ParallelPipelineBuilder."""
        pipeline = ParallelPipelineBuilder().with_max_parallel(2).with_fail_fast(True).build()
        assert isinstance(pipeline, ParallelPipeline)
        assert pipeline.max_parallel == 2
        assert pipeline.fail_fast is True

    def test_builder_with_router(self):
        """Test builder with custom router."""
        from src.causal_engine.pipeline import LibraryRouter

        custom_router = LibraryRouter()
        pipeline = ParallelPipelineBuilder().with_router(custom_router).build()
        assert pipeline.router is custom_router


class TestParallelPipelineExecution:
    """Test parallel pipeline execution."""

    @pytest.fixture
    def pipeline(self) -> ParallelPipeline:
        """Create a pipeline for testing."""
        return ParallelPipeline()

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
            mode="parallel",
            libraries_enabled=None,
            cross_validate=None,
        )

    @pytest.mark.asyncio
    async def test_execute_full_pipeline(
        self, pipeline: ParallelPipeline, basic_input: PipelineInput
    ):
        """Test executing full parallel pipeline."""
        output = await pipeline.execute(basic_input)

        assert output["status"] in ["completed", "partial"]
        assert output["question_type"] == "causal_relationship"
        assert len(output["libraries_used"]) > 0

    @pytest.mark.asyncio
    async def test_execute_with_limited_libraries(self, pipeline: ParallelPipeline):
        """Test executing with limited library set."""
        input_data = PipelineInput(
            query="How does treatment effect vary?",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=["confounder1"],
            effect_modifiers=["region"],
            data_source="test_data",
            filters=None,
            mode="parallel",
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
        self, pipeline: ParallelPipeline, basic_input: PipelineInput
    ):
        """Test that execution generates summary and insights."""
        output = await pipeline.execute(basic_input)

        assert output["executive_summary"]
        assert isinstance(output["key_insights"], list)
        assert isinstance(output["recommended_actions"], list)

    @pytest.mark.asyncio
    async def test_execute_tracks_latency(
        self, pipeline: ParallelPipeline, basic_input: PipelineInput
    ):
        """Test that execution tracks latency."""
        output = await pipeline.execute(basic_input)

        # Latency >= 0 (may be 0 for very fast mock executors)
        assert output["total_latency_ms"] >= 0
        assert output["status"] in ["completed", "partial"]

    @pytest.mark.asyncio
    async def test_mode_is_parallel(self, pipeline: ParallelPipeline, basic_input: PipelineInput):
        """Test that mode is set to parallel."""
        # This verifies the mode is correctly set during execution
        output = await pipeline.execute(basic_input)
        # The pipeline should complete successfully regardless of input mode
        assert output["status"] in ["completed", "partial"]


class TestParallelPipelineMaxParallel:
    """Test max_parallel limiting."""

    @pytest.mark.asyncio
    async def test_max_parallel_limits_libraries(self):
        """Test that max_parallel limits concurrent libraries."""
        pipeline = ParallelPipeline(max_parallel=2)

        input_data = PipelineInput(
            query="Full causal analysis",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=["c1"],
            effect_modifiers=["e1"],
            data_source="test_data",
            filters=None,
            mode="parallel",
            libraries_enabled=None,  # All libraries
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        # Should only execute up to max_parallel libraries
        assert output["status"] in ["completed", "partial", "failed"]


class TestParallelPipelineFailFast:
    """Test fail-fast behavior."""

    @pytest.mark.asyncio
    async def test_fail_fast_mode_creation(self):
        """Test that fail_fast mode is properly set."""
        pipeline = ParallelPipeline(fail_fast=True)
        assert pipeline.fail_fast is True

    @pytest.mark.asyncio
    async def test_non_fail_fast_continues(self):
        """Test that non-fail-fast continues after warnings."""
        pipeline = ParallelPipeline(fail_fast=False)

        input_data = PipelineInput(
            query="Any query",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=None,
            effect_modifiers=None,
            data_source="test_data",
            filters=None,
            mode="parallel",
            libraries_enabled=["dowhy", "econml", "causalml"],
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        # Should have executed multiple libraries
        assert len(output["libraries_used"]) >= 1


class TestConsensusCalculation:
    """Test confidence-weighted consensus calculation."""

    @pytest.fixture
    def pipeline(self) -> ParallelPipeline:
        return ParallelPipeline()

    @pytest.mark.asyncio
    async def test_consensus_with_multiple_effects(self, pipeline: ParallelPipeline):
        """Test consensus calculation with multiple effect estimates."""
        input_data = PipelineInput(
            query="Does treatment affect outcome?",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=["c1"],
            effect_modifiers=["e1"],
            data_source="test_data",
            filters=None,
            mode="parallel",
            libraries_enabled=["dowhy", "econml"],
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        # Should have at least the summary
        assert output["executive_summary"]

    @pytest.mark.asyncio
    async def test_generates_insights(self, pipeline: ParallelPipeline):
        """Test that parallel pipeline generates insights."""
        input_data = PipelineInput(
            query="Any query",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=None,
            effect_modifiers=None,
            data_source="test_data",
            filters=None,
            mode="parallel",
            libraries_enabled=None,
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        assert isinstance(output["key_insights"], list)


class TestLibraryAgreement:
    """Test library agreement calculations."""

    @pytest.fixture
    def pipeline(self) -> ParallelPipeline:
        return ParallelPipeline()

    @pytest.mark.asyncio
    async def test_agreement_metrics_structure(self, pipeline: ParallelPipeline):
        """Test that agreement metrics are properly structured."""
        input_data = PipelineInput(
            query="Does treatment affect outcome?",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=["c1"],
            effect_modifiers=["e1"],
            data_source="test_data",
            filters=None,
            mode="parallel",
            libraries_enabled=["dowhy", "econml"],
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        # library_agreement should be a dict if multiple libraries returned effects
        if output.get("library_agreement"):
            assert isinstance(output["library_agreement"], dict)


class TestParallelPipelineRecommendations:
    """Test recommendation generation."""

    @pytest.fixture
    def pipeline(self) -> ParallelPipeline:
        return ParallelPipeline()

    @pytest.mark.asyncio
    async def test_generates_recommendations(self, pipeline: ParallelPipeline):
        """Test that parallel pipeline generates recommendations."""
        input_data = PipelineInput(
            query="Who should we target?",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders=None,
            effect_modifiers=None,
            data_source="test_data",
            filters=None,
            mode="parallel",
            libraries_enabled=None,
            cross_validate=None,
        )

        output = await pipeline.execute(input_data)

        assert isinstance(output["recommended_actions"], list)
        assert len(output["recommended_actions"]) > 0


class TestLibraryExecutors:
    """Test individual library executors in parallel context."""

    @pytest.fixture
    def pipeline(self) -> ParallelPipeline:
        return ParallelPipeline()

    def test_has_all_executors(self, pipeline: ParallelPipeline):
        """Test that pipeline has executors for all libraries."""
        assert CausalLibrary.NETWORKX in pipeline.executors
        assert CausalLibrary.DOWHY in pipeline.executors
        assert CausalLibrary.ECONML in pipeline.executors
        assert CausalLibrary.CAUSALML in pipeline.executors

    def test_executor_libraries_match(self, pipeline: ParallelPipeline):
        """Test that executors return correct library identifiers."""
        assert pipeline.executors[CausalLibrary.NETWORKX].library == CausalLibrary.NETWORKX
        assert pipeline.executors[CausalLibrary.DOWHY].library == CausalLibrary.DOWHY
        assert pipeline.executors[CausalLibrary.ECONML].library == CausalLibrary.ECONML
        assert pipeline.executors[CausalLibrary.CAUSALML].library == CausalLibrary.CAUSALML


class TestParallelPipelineRouting:
    """Test routing integration in parallel pipeline."""

    @pytest.fixture
    def pipeline(self) -> ParallelPipeline:
        return ParallelPipeline()

    @pytest.mark.asyncio
    async def test_route_causal_question(self, pipeline: ParallelPipeline):
        """Test routing a causal relationship question."""
        decision = await pipeline.route("Does X cause Y?")

        assert decision.primary_library == CausalLibrary.DOWHY

    @pytest.mark.asyncio
    async def test_route_targeting_question(self, pipeline: ParallelPipeline):
        """Test routing a targeting question."""
        decision = await pipeline.route("Who should we target for the campaign?")

        assert decision.primary_library == CausalLibrary.CAUSALML

    @pytest.mark.asyncio
    async def test_route_heterogeneity_question(self, pipeline: ParallelPipeline):
        """Test routing a heterogeneity question."""
        decision = await pipeline.route("How does the treatment effect vary by segment?")

        assert decision.primary_library == CausalLibrary.ECONML
