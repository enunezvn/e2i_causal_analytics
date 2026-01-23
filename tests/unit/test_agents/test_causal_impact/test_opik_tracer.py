"""Unit tests for CausalImpactOpikTracer.

Tests comprehensive Opik tracing for the Causal Impact 5-node pipeline,
including NodeSpanContext, AnalysisTraceContext, and the singleton tracer.

Version: 1.0.0
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.causal_impact.opik_tracer import (
    AGENT_METADATA,
    AnalysisTraceContext,
    CausalImpactOpikTracer,
    NodeSpanContext,
    _get_pipeline_position,
    get_causal_impact_tracer,
    reset_tracer,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_tracer()
    yield
    reset_tracer()


@pytest.fixture
def mock_opik_connector():
    """Create mock Opik connector."""
    mock = MagicMock()
    mock.is_enabled = True

    # Create mock span context manager
    mock_span = MagicMock()
    mock_span.span_id = "span_123"
    mock_span.set_output = MagicMock()

    mock.trace_agent = MagicMock()
    mock.trace_agent.return_value.__aenter__ = AsyncMock(return_value=mock_span)
    mock.trace_agent.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock


@pytest.fixture
def tracer():
    """Create a CausalImpactOpikTracer instance."""
    return CausalImpactOpikTracer(project_name="test_causal_analytics")


@pytest.fixture
def node_span_context():
    """Create a sample NodeSpanContext."""
    return NodeSpanContext(
        trace_id="trace_abc123",
        span_id="span_xyz789",
        node_name="estimation",
    )


@pytest.fixture
def analysis_trace_context():
    """Create a sample AnalysisTraceContext."""
    return AnalysisTraceContext(
        trace_id="trace_abc123",
        query="What is the impact of marketing spend on sales?",
        treatment_var="marketing_spend",
        outcome_var="sales_revenue",
        brand="TestBrand",
    )


# ==============================================================================
# TestAgentMetadata
# ==============================================================================


class TestAgentMetadata:
    """Tests for AGENT_METADATA constant."""

    def test_metadata_contains_required_fields(self):
        """Test that metadata contains all required fields."""
        assert "name" in AGENT_METADATA
        assert "tier" in AGENT_METADATA
        assert "type" in AGENT_METADATA
        assert "pipeline" in AGENT_METADATA

    def test_metadata_values(self):
        """Test metadata values are correct."""
        assert AGENT_METADATA["name"] == "causal_impact"
        assert AGENT_METADATA["tier"] == 2
        assert AGENT_METADATA["type"] == "hybrid"
        assert "graph_builder" in AGENT_METADATA["pipeline"]
        assert "interpretation" in AGENT_METADATA["pipeline"]


# ==============================================================================
# TestGetPipelinePosition
# ==============================================================================


class TestGetPipelinePosition:
    """Tests for _get_pipeline_position helper function."""

    def test_graph_builder_position(self):
        """Test graph_builder is position 1."""
        assert _get_pipeline_position("graph_builder") == 1

    def test_estimation_position(self):
        """Test estimation is position 2."""
        assert _get_pipeline_position("estimation") == 2

    def test_refutation_position(self):
        """Test refutation is position 3."""
        assert _get_pipeline_position("refutation") == 3

    def test_sensitivity_position(self):
        """Test sensitivity is position 4."""
        assert _get_pipeline_position("sensitivity") == 4

    def test_interpretation_position(self):
        """Test interpretation is position 5."""
        assert _get_pipeline_position("interpretation") == 5

    def test_unknown_node_returns_zero(self):
        """Test unknown node returns 0."""
        assert _get_pipeline_position("unknown_node") == 0
        assert _get_pipeline_position("") == 0


# ==============================================================================
# TestNodeSpanContext
# ==============================================================================


class TestNodeSpanContext:
    """Tests for NodeSpanContext dataclass."""

    def test_initialization(self, node_span_context):
        """Test NodeSpanContext initialization."""
        assert node_span_context.trace_id == "trace_abc123"
        assert node_span_context.span_id == "span_xyz789"
        assert node_span_context.node_name == "estimation"
        assert node_span_context.end_time is None
        assert node_span_context.duration_ms is None
        assert node_span_context.metadata == {}
        assert node_span_context._opik_span is None

    def test_start_time_default(self):
        """Test that start_time is set by default."""
        ctx = NodeSpanContext(
            trace_id="trace_1",
            span_id="span_1",
            node_name="graph_builder",
        )

        assert ctx.start_time is not None
        assert isinstance(ctx.start_time, datetime)
        assert ctx.start_time.tzinfo == timezone.utc

    def test_log_graph_construction(self, node_span_context):
        """Test log_graph_construction method."""
        node_span_context.log_graph_construction(
            num_nodes=10,
            num_edges=15,
            confidence=0.85,
            adjustment_sets=3,
            discovery_enabled=True,
            discovery_algorithms=["PC", "GES"],
        )

        assert node_span_context.metadata["graph_nodes"] == 10
        assert node_span_context.metadata["graph_edges"] == 15
        assert node_span_context.metadata["graph_confidence"] == 0.85
        assert node_span_context.metadata["adjustment_sets"] == 3
        assert node_span_context.metadata["discovery_enabled"] is True
        assert node_span_context.metadata["discovery_algorithms"] == ["PC", "GES"]

    def test_log_graph_construction_defaults(self, node_span_context):
        """Test log_graph_construction with default values."""
        node_span_context.log_graph_construction(
            num_nodes=5,
            num_edges=8,
            confidence=0.7,
        )

        assert node_span_context.metadata["adjustment_sets"] is None
        assert node_span_context.metadata["discovery_enabled"] is False
        assert node_span_context.metadata["discovery_algorithms"] == []

    def test_log_estimation(self, node_span_context):
        """Test log_estimation method."""
        node_span_context.log_estimation(
            ate=0.15,
            ci_lower=0.10,
            ci_upper=0.20,
            method="doubly_robust",
            sample_size=1000,
            energy_score=0.85,
            selection_strategy="energy_based",
            n_estimators_evaluated=5,
        )

        assert node_span_context.metadata["ate"] == 0.15
        assert node_span_context.metadata["ci_lower"] == 0.10
        assert node_span_context.metadata["ci_upper"] == 0.20
        assert node_span_context.metadata["ci_width"] == 0.10
        assert node_span_context.metadata["method"] == "doubly_robust"
        assert node_span_context.metadata["sample_size"] == 1000
        assert node_span_context.metadata["energy_score"] == 0.85
        assert node_span_context.metadata["selection_strategy"] == "energy_based"
        assert node_span_context.metadata["n_estimators_evaluated"] == 5

    def test_log_refutation(self, node_span_context):
        """Test log_refutation method."""
        individual_tests = {
            "placebo": True,
            "random_common_cause": True,
            "subset": False,
        }

        node_span_context.log_refutation(
            tests_passed=2,
            tests_total=3,
            refutation_rate=0.667,
            individual_tests=individual_tests,
        )

        assert node_span_context.metadata["refutation_tests_passed"] == 2
        assert node_span_context.metadata["refutation_tests_total"] == 3
        assert node_span_context.metadata["refutation_rate"] == 0.667
        assert node_span_context.metadata["individual_tests"] == individual_tests

    def test_log_refutation_defaults(self, node_span_context):
        """Test log_refutation with default individual_tests."""
        node_span_context.log_refutation(
            tests_passed=3,
            tests_total=3,
            refutation_rate=1.0,
        )

        assert node_span_context.metadata["individual_tests"] == {}

    def test_log_sensitivity(self, node_span_context):
        """Test log_sensitivity method."""
        node_span_context.log_sensitivity(
            e_value=2.5,
            robustness_score=0.8,
            sensitivity_passed=True,
            unmeasured_confounding_threshold=0.1,
        )

        assert node_span_context.metadata["e_value"] == 2.5
        assert node_span_context.metadata["robustness_score"] == 0.8
        assert node_span_context.metadata["sensitivity_passed"] is True
        assert node_span_context.metadata["unmeasured_confounding_threshold"] == 0.1

    def test_log_sensitivity_defaults(self, node_span_context):
        """Test log_sensitivity with default values."""
        node_span_context.log_sensitivity()

        assert node_span_context.metadata["e_value"] is None
        assert node_span_context.metadata["robustness_score"] is None
        assert node_span_context.metadata["sensitivity_passed"] is True

    def test_log_interpretation(self, node_span_context):
        """Test log_interpretation method."""
        node_span_context.log_interpretation(
            summary_generated=True,
            summary_length=150,
            confidence_level="high",
        )

        assert node_span_context.metadata["summary_generated"] is True
        assert node_span_context.metadata["summary_length"] == 150
        assert node_span_context.metadata["confidence_level"] == "high"

    def test_log_interpretation_defaults(self, node_span_context):
        """Test log_interpretation with default values."""
        node_span_context.log_interpretation()

        assert node_span_context.metadata["summary_generated"] is True
        assert node_span_context.metadata["summary_length"] is None
        assert node_span_context.metadata["confidence_level"] is None

    def test_set_error(self, node_span_context):
        """Test set_error method."""
        node_span_context.set_error(
            error="Estimation failed due to insufficient data",
            error_type="estimation_error",
        )

        assert node_span_context.metadata["error"] == "Estimation failed due to insufficient data"
        assert node_span_context.metadata["error_type"] == "estimation_error"
        assert node_span_context.metadata["status"] == "error"

    def test_set_error_default_type(self, node_span_context):
        """Test set_error with default error type."""
        node_span_context.set_error("Unknown error occurred")

        assert node_span_context.metadata["error_type"] == "unknown_error"


# ==============================================================================
# TestAnalysisTraceContext
# ==============================================================================


class TestAnalysisTraceContext:
    """Tests for AnalysisTraceContext dataclass."""

    def test_initialization(self, analysis_trace_context):
        """Test AnalysisTraceContext initialization."""
        assert analysis_trace_context.trace_id == "trace_abc123"
        assert analysis_trace_context.query == "What is the impact of marketing spend on sales?"
        assert analysis_trace_context.treatment_var == "marketing_spend"
        assert analysis_trace_context.outcome_var == "sales_revenue"
        assert analysis_trace_context.brand == "TestBrand"
        assert analysis_trace_context.node_spans == []
        assert analysis_trace_context._tracer is None
        assert analysis_trace_context._opik_span is None

    def test_start_time_default(self):
        """Test that start_time is set by default."""
        ctx = AnalysisTraceContext(
            trace_id="trace_1",
            query="test query",
            treatment_var="X",
            outcome_var="Y",
        )

        assert ctx.start_time is not None
        assert isinstance(ctx.start_time, datetime)

    @pytest.mark.asyncio
    async def test_trace_node_creates_span_context(self, analysis_trace_context):
        """Test that trace_node creates NodeSpanContext."""
        async with analysis_trace_context.trace_node("graph_builder") as node:
            assert isinstance(node, NodeSpanContext)
            assert node.node_name == "graph_builder"
            assert node.trace_id == analysis_trace_context.trace_id

    @pytest.mark.asyncio
    async def test_trace_node_appends_to_spans(self, analysis_trace_context):
        """Test that trace_node appends span to node_spans."""
        assert len(analysis_trace_context.node_spans) == 0

        async with analysis_trace_context.trace_node("estimation") as node:
            pass

        assert len(analysis_trace_context.node_spans) == 1
        assert analysis_trace_context.node_spans[0].node_name == "estimation"

    @pytest.mark.asyncio
    async def test_trace_node_sets_end_time(self, analysis_trace_context):
        """Test that trace_node sets end_time on exit."""
        async with analysis_trace_context.trace_node("refutation") as node:
            assert node.end_time is None

        assert analysis_trace_context.node_spans[0].end_time is not None

    @pytest.mark.asyncio
    async def test_trace_node_calculates_duration(self, analysis_trace_context):
        """Test that trace_node calculates duration_ms."""
        async with analysis_trace_context.trace_node("sensitivity") as node:
            pass

        assert analysis_trace_context.node_spans[0].duration_ms is not None
        assert analysis_trace_context.node_spans[0].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_trace_node_handles_exception(self, analysis_trace_context):
        """Test that trace_node handles exceptions and sets error."""
        with pytest.raises(ValueError):
            async with analysis_trace_context.trace_node("estimation") as node:
                raise ValueError("Test error")

        # Span should still be appended
        assert len(analysis_trace_context.node_spans) == 1
        assert analysis_trace_context.node_spans[0].metadata["error"] == "Test error"
        assert analysis_trace_context.node_spans[0].metadata["status"] == "error"

    def test_log_analysis_complete(self, analysis_trace_context):
        """Test log_analysis_complete method."""
        analysis_trace_context.log_analysis_complete(
            success=True,
            ate=0.15,
            refutation_rate=0.8,
            confidence_score=0.9,
            final_status="completed",
        )

        # Method logs completion info - verify no exceptions

    def test_log_analysis_complete_with_node_durations(self, analysis_trace_context):
        """Test log_analysis_complete includes node durations."""
        # Add some mock node spans
        span1 = NodeSpanContext(
            trace_id="trace_1",
            span_id="span_1",
            node_name="estimation",
        )
        span1.duration_ms = 100.0
        analysis_trace_context.node_spans.append(span1)

        span2 = NodeSpanContext(
            trace_id="trace_1",
            span_id="span_2",
            node_name="refutation",
        )
        span2.duration_ms = 50.0
        analysis_trace_context.node_spans.append(span2)

        # Should not raise
        analysis_trace_context.log_analysis_complete(success=True)


# ==============================================================================
# TestCausalImpactOpikTracer
# ==============================================================================


class TestCausalImpactOpikTracer:
    """Tests for CausalImpactOpikTracer class."""

    def test_initialization(self, tracer):
        """Test tracer initialization."""
        assert tracer.project_name == "test_causal_analytics"
        assert tracer._opik is None
        assert tracer._initialized is False

    def test_default_project_name(self):
        """Test tracer with default project name."""
        tracer = CausalImpactOpikTracer()
        assert tracer.project_name == "e2i_causal_analytics"

    def test_ensure_initialized_lazy_loads(self, tracer):
        """Test that _ensure_initialized lazy loads connector."""
        assert tracer._initialized is False

        mock_connector = MagicMock(is_enabled=True)
        with patch.dict(
            "sys.modules",
            {"src.mlops.opik_connector": MagicMock(get_opik_connector=MagicMock(return_value=mock_connector))}
        ):
            tracer._ensure_initialized()

        assert tracer._initialized is True

    def test_ensure_initialized_handles_import_error(self, tracer):
        """Test graceful handling when connector unavailable."""
        # Force uninitialized state
        tracer._initialized = False
        tracer._opik = None

        # Mock the import to fail
        original_import = __builtins__.__dict__.get('__import__', __import__)

        def mock_import(name, *args, **kwargs):
            if 'opik_connector' in name:
                raise ImportError("Opik not installed")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            tracer._ensure_initialized()

        assert tracer._initialized is True
        assert tracer._opik is None

    def test_is_enabled_when_opik_available(self, tracer, mock_opik_connector):
        """Test is_enabled returns True when Opik available."""
        # Directly set the connector to simulate successful initialization
        tracer._opik = mock_opik_connector
        tracer._initialized = True
        assert tracer.is_enabled is True

    def test_is_enabled_when_opik_unavailable(self, tracer):
        """Test is_enabled returns False when Opik unavailable."""
        # Simulate failed initialization
        tracer._opik = None
        tracer._initialized = True
        assert tracer.is_enabled is False

    def test_is_enabled_when_opik_disabled(self, tracer):
        """Test is_enabled returns False when Opik is disabled."""
        mock_connector = MagicMock()
        mock_connector.is_enabled = False

        tracer._opik = mock_connector
        tracer._initialized = True
        assert tracer.is_enabled is False


# ==============================================================================
# TestTraceAnalysis
# ==============================================================================


class TestTraceAnalysis:
    """Tests for trace_analysis context manager."""

    @pytest.mark.asyncio
    async def test_yields_analysis_context(self, tracer):
        """Test that trace_analysis yields AnalysisTraceContext."""
        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="X",
            outcome_var="Y",
        ) as ctx:
            assert isinstance(ctx, AnalysisTraceContext)
            assert ctx.query == "Test query"
            assert ctx.treatment_var == "X"
            assert ctx.outcome_var == "Y"

    @pytest.mark.asyncio
    async def test_sets_trace_id(self, tracer):
        """Test that trace_analysis sets a trace_id."""
        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="X",
            outcome_var="Y",
        ) as ctx:
            assert ctx.trace_id is not None
            assert len(ctx.trace_id) > 0

    @pytest.mark.asyncio
    async def test_sets_brand(self, tracer):
        """Test that brand is set correctly."""
        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="X",
            outcome_var="Y",
            brand="TestBrand",
        ) as ctx:
            assert ctx.brand == "TestBrand"

    @pytest.mark.asyncio
    async def test_sets_tracer_reference(self, tracer):
        """Test that _tracer reference is set."""
        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="X",
            outcome_var="Y",
        ) as ctx:
            assert ctx._tracer is tracer

    @pytest.mark.asyncio
    async def test_works_without_opik(self, tracer):
        """Test that trace_analysis works when Opik unavailable."""
        # Simulate Opik not being available
        tracer._opik = None
        tracer._initialized = True

        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="X",
            outcome_var="Y",
        ) as ctx:
            assert isinstance(ctx, AnalysisTraceContext)

    @pytest.mark.asyncio
    async def test_creates_opik_span_when_available(self, tracer, mock_opik_connector):
        """Test that Opik span is created when connector available."""
        tracer._opik = mock_opik_connector
        tracer._initialized = True

        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="X",
            outcome_var="Y",
        ) as ctx:
            mock_opik_connector.trace_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_exception(self, tracer):
        """Test that exceptions are propagated."""
        with pytest.raises(ValueError):
            async with tracer.trace_analysis(
                query="Test query",
                treatment_var="X",
                outcome_var="Y",
            ) as ctx:
                raise ValueError("Test error")


# ==============================================================================
# TestTraceNodeDecorator
# ==============================================================================


class TestTraceNodeDecorator:
    """Tests for trace_node_decorator method."""

    @pytest.mark.asyncio
    async def test_decorator_without_opik(self, tracer):
        """Test decorator works when Opik unavailable."""
        # Simulate Opik not being available
        tracer._opik = None
        tracer._initialized = True

        @tracer.trace_node_decorator("estimation")
        async def estimate_effect(state):
            return {"ate": 0.15, "status": "completed"}

        result = await estimate_effect({"query": "test", "treatment_var": "X"})

        assert result["ate"] == 0.15
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_decorator_with_opik(self, tracer, mock_opik_connector):
        """Test decorator creates span when Opik available."""
        tracer._opik = mock_opik_connector
        tracer._initialized = True

        @tracer.trace_node_decorator("refutation")
        async def run_refutation(state):
            return {"refutation_rate": 0.8, "status": "completed"}

        result = await run_refutation({"query": "test", "outcome_var": "Y"})

        assert result["refutation_rate"] == 0.8
        mock_opik_connector.trace_agent.assert_called()

    @pytest.mark.asyncio
    async def test_decorator_passes_state(self, tracer):
        """Test that decorator passes state correctly."""
        # Simulate Opik not being available
        tracer._opik = None
        tracer._initialized = True

        @tracer.trace_node_decorator("graph_builder")
        async def build_graph(state):
            return {
                "graph_nodes": state.get("expected_nodes", 0),
                "treatment_var": state.get("treatment_var"),
            }

        result = await build_graph({
            "expected_nodes": 10,
            "treatment_var": "marketing_spend",
        })

        assert result["graph_nodes"] == 10
        assert result["treatment_var"] == "marketing_spend"

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self, tracer):
        """Test that decorator preserves function name."""
        # Simulate Opik not being available
        tracer._opik = None
        tracer._initialized = True

        @tracer.trace_node_decorator("sensitivity")
        async def run_sensitivity_analysis(state):
            return {"e_value": 2.0}

        assert run_sensitivity_analysis.__name__ == "run_sensitivity_analysis"


# ==============================================================================
# TestSingleton
# ==============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_causal_impact_tracer_returns_instance(self):
        """Test that get_causal_impact_tracer returns a tracer."""
        tracer = get_causal_impact_tracer()
        assert isinstance(tracer, CausalImpactOpikTracer)

    def test_get_causal_impact_tracer_returns_same_instance(self):
        """Test that get_causal_impact_tracer returns singleton."""
        tracer1 = get_causal_impact_tracer()
        tracer2 = get_causal_impact_tracer()
        assert tracer1 is tracer2

    def test_reset_tracer_clears_singleton(self):
        """Test that reset_tracer clears the singleton."""
        tracer1 = get_causal_impact_tracer()
        reset_tracer()
        tracer2 = get_causal_impact_tracer()
        assert tracer1 is not tracer2


# ==============================================================================
# TestIntegration
# ==============================================================================


class TestIntegration:
    """Integration tests for the full tracing flow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_trace(self, tracer):
        """Test tracing a full 5-node pipeline."""
        # Simulate Opik not being available
        tracer._opik = None
        tracer._initialized = True

        async with tracer.trace_analysis(
            query="What is the causal effect of X on Y?",
            treatment_var="X",
            outcome_var="Y",
            brand="TestBrand",
        ) as trace:
                # Node 1: Graph Builder
                async with trace.trace_node("graph_builder") as node:
                    node.log_graph_construction(
                        num_nodes=10,
                        num_edges=15,
                        confidence=0.85,
                    )

                # Node 2: Estimation
                async with trace.trace_node("estimation") as node:
                    node.log_estimation(
                        ate=0.15,
                        ci_lower=0.10,
                        ci_upper=0.20,
                        method="doubly_robust",
                        sample_size=1000,
                    )

                # Node 3: Refutation
                async with trace.trace_node("refutation") as node:
                    node.log_refutation(
                        tests_passed=3,
                        tests_total=3,
                        refutation_rate=1.0,
                    )

                # Node 4: Sensitivity
                async with trace.trace_node("sensitivity") as node:
                    node.log_sensitivity(
                        e_value=2.5,
                        robustness_score=0.9,
                    )

                # Node 5: Interpretation
                async with trace.trace_node("interpretation") as node:
                    node.log_interpretation(
                        summary_generated=True,
                        summary_length=150,
                        confidence_level="high",
                    )

                trace.log_analysis_complete(
                    success=True,
                    ate=0.15,
                    refutation_rate=1.0,
                    confidence_score=0.9,
                )

        # Verify all nodes were traced
        assert len(trace.node_spans) == 5
        assert trace.node_spans[0].node_name == "graph_builder"
        assert trace.node_spans[1].node_name == "estimation"
        assert trace.node_spans[2].node_name == "refutation"
        assert trace.node_spans[3].node_name == "sensitivity"
        assert trace.node_spans[4].node_name == "interpretation"

        # Verify metrics were logged
        assert trace.node_spans[0].metadata["graph_nodes"] == 10
        assert trace.node_spans[1].metadata["ate"] == 0.15
        assert trace.node_spans[2].metadata["refutation_rate"] == 1.0
        assert trace.node_spans[3].metadata["e_value"] == 2.5
        assert trace.node_spans[4].metadata["summary_generated"] is True

    @pytest.mark.asyncio
    async def test_partial_pipeline_with_error(self, tracer):
        """Test tracing a pipeline that fails partway through."""
        # Simulate Opik not being available
        tracer._opik = None
        tracer._initialized = True

        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="X",
            outcome_var="Y",
        ) as trace:
                async with trace.trace_node("graph_builder") as node:
                    node.log_graph_construction(
                        num_nodes=5,
                        num_edges=8,
                        confidence=0.7,
                    )

                # Estimation fails
                with pytest.raises(RuntimeError):
                    async with trace.trace_node("estimation") as node:
                        raise RuntimeError("Estimation failed")

        # Both spans should be recorded
        assert len(trace.node_spans) == 2
        assert trace.node_spans[1].metadata["error"] == "Estimation failed"
        assert trace.node_spans[1].metadata["status"] == "error"


# ==============================================================================
# TestExports
# ==============================================================================


class TestExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.agents.causal_impact import opik_tracer

        assert hasattr(opik_tracer, "CausalImpactOpikTracer")
        assert hasattr(opik_tracer, "AnalysisTraceContext")
        assert hasattr(opik_tracer, "NodeSpanContext")
        assert hasattr(opik_tracer, "get_causal_impact_tracer")
        assert hasattr(opik_tracer, "reset_tracer")
        assert hasattr(opik_tracer, "AGENT_METADATA")
