"""Tests for Tool Composer Opik Tracer.

Version: 1.0.0
Tests the Opik observability integration for Tool Composer 4-phase pipeline.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.agents.tool_composer.opik_tracer import (
    CompositionTraceContext,
    PhaseSpanContext,
    ToolComposerOpikTracer,
    get_tool_composer_tracer,
    reset_tool_composer_tracer,
    trace_composition,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the tracer singleton before each test."""
    reset_tool_composer_tracer()
    yield
    reset_tool_composer_tracer()


class TestPhaseSpanContext:
    """Tests for PhaseSpanContext dataclass."""

    def test_create_phase_span_context(self):
        """Test creating a phase span context."""
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="decompose",
        )

        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.phase_name == "decompose"
        assert ctx.end_time is None
        assert ctx.duration_ms is None
        assert ctx.metadata == {}

    def test_log_decomposition(self):
        """Test logging decomposition phase metrics."""
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="decompose",
        )

        ctx.log_decomposition(
            sub_question_count=3,
            intents=["CAUSAL", "COMPARATIVE", "PREDICTIVE"],
            extracted_entities=["brand_a", "region_midwest"],
        )

        assert ctx.metadata["sub_question_count"] == 3
        assert ctx.metadata["intents"] == ["CAUSAL", "COMPARATIVE", "PREDICTIVE"]
        assert ctx.metadata["extracted_entities"] == ["brand_a", "region_midwest"]

    def test_log_decomposition_with_opik_span(self):
        """Test logging decomposition with Opik span attached."""
        mock_span = MagicMock()
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="decompose",
            _opik_span=mock_span,
        )

        ctx.log_decomposition(
            sub_question_count=2,
            intents=["CAUSAL", "DESCRIPTIVE"],
        )

        mock_span.set_attribute.assert_any_call("sub_question_count", 2)
        mock_span.set_attribute.assert_any_call("intent_count", 2)
        mock_span.add_event.assert_called_once()

    def test_log_planning(self):
        """Test logging planning phase metrics."""
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="plan",
        )

        ctx.log_planning(
            step_count=4,
            tool_mappings=["causal_effect_estimator", "segment_ranker"],
            parallel_groups=2,
            avg_confidence=0.85,
        )

        assert ctx.metadata["step_count"] == 4
        assert ctx.metadata["tool_mappings"] == ["causal_effect_estimator", "segment_ranker"]
        assert ctx.metadata["parallel_groups"] == 2
        assert ctx.metadata["avg_confidence"] == 0.85

    def test_log_planning_with_opik_span(self):
        """Test logging planning with Opik span attached."""
        mock_span = MagicMock()
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="plan",
            _opik_span=mock_span,
        )

        ctx.log_planning(
            step_count=3,
            tool_mappings=["tool1", "tool2"],
            parallel_groups=1,
            avg_confidence=0.9,
        )

        mock_span.set_attribute.assert_any_call("step_count", 3)
        mock_span.set_attribute.assert_any_call("parallel_groups", 1)
        mock_span.set_attribute.assert_any_call("avg_confidence", 0.9)
        mock_span.add_event.assert_called_once()

    def test_log_execution(self):
        """Test logging execution phase metrics."""
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="execute",
        )

        ctx.log_execution(
            tools_executed=5,
            tools_succeeded=4,
            retry_count=1,
            parallel_executions=2,
            step_durations_ms=[100, 200, 150, 300, 250],
        )

        assert ctx.metadata["tools_executed"] == 5
        assert ctx.metadata["tools_succeeded"] == 4
        assert ctx.metadata["success_rate"] == 0.8  # 4/5
        assert ctx.metadata["retry_count"] == 1
        assert ctx.metadata["parallel_executions"] == 2

    def test_log_execution_with_opik_span(self):
        """Test logging execution with Opik span attached."""
        mock_span = MagicMock()
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="execute",
            _opik_span=mock_span,
        )

        ctx.log_execution(
            tools_executed=3,
            tools_succeeded=3,
            retry_count=0,
            parallel_executions=1,
        )

        mock_span.set_attribute.assert_any_call("tools_executed", 3)
        mock_span.set_attribute.assert_any_call("tools_succeeded", 3)
        mock_span.set_attribute.assert_any_call("success_rate", 1.0)
        mock_span.set_attribute.assert_any_call("parallel_executions", 1)
        mock_span.add_event.assert_called_once()

    def test_log_execution_handles_zero_tools(self):
        """Test logging execution handles zero tools gracefully."""
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="execute",
        )

        ctx.log_execution(
            tools_executed=0,
            tools_succeeded=0,
        )

        assert ctx.metadata["tools_executed"] == 0
        assert ctx.metadata["success_rate"] == 0.0

    def test_log_synthesis(self):
        """Test logging synthesis phase metrics."""
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="synthesize",
        )

        ctx.log_synthesis(
            answer_length=500,
            confidence=0.85,
            caveat_count=2,
            failed_components=["tool_3"],
        )

        assert ctx.metadata["answer_length"] == 500
        assert ctx.metadata["confidence"] == 0.85
        assert ctx.metadata["caveat_count"] == 2
        assert ctx.metadata["failed_components"] == ["tool_3"]

    def test_log_synthesis_with_opik_span(self):
        """Test logging synthesis with Opik span attached."""
        mock_span = MagicMock()
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="synthesize",
            _opik_span=mock_span,
        )

        ctx.log_synthesis(
            answer_length=300,
            confidence=0.9,
            caveat_count=0,
        )

        mock_span.set_attribute.assert_any_call("answer_length", 300)
        mock_span.set_attribute.assert_any_call("confidence", 0.9)
        mock_span.set_attribute.assert_any_call("caveat_count", 0)
        mock_span.add_event.assert_called_once()

    def test_set_output(self):
        """Test setting output data on phase span."""
        mock_span = MagicMock()
        ctx = PhaseSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            phase_name="decompose",
            _opik_span=mock_span,
        )

        ctx.set_output({"sub_questions": ["q1", "q2"]})

        mock_span.set_output.assert_called_once_with({"sub_questions": ["q1", "q2"]})


class TestCompositionTraceContext:
    """Tests for CompositionTraceContext dataclass."""

    def test_create_trace_context(self):
        """Test creating a composition trace context."""
        ctx = CompositionTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Compare causal impact of rep visits vs speaker programs",
        )

        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert "Compare causal" in ctx.query
        assert ctx.end_time is None
        assert ctx.duration_ms is None
        assert ctx.phase_spans == {}
        assert ctx.phase_durations == {}

    @pytest.mark.asyncio
    async def test_trace_phase_creates_phase_context(self):
        """Test trace_phase creates PhaseSpanContext."""
        ctx = CompositionTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Test query",
        )

        async with ctx.trace_phase("decompose") as phase:
            assert isinstance(phase, PhaseSpanContext)
            assert phase.phase_name == "decompose"
            assert phase.trace_id == "trace-123"

        # Phase should be recorded
        assert "decompose" in ctx.phase_spans
        assert "decompose" in ctx.phase_durations

    @pytest.mark.asyncio
    async def test_trace_phase_records_duration(self):
        """Test trace_phase records phase duration."""
        ctx = CompositionTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Test query",
        )

        async with ctx.trace_phase("plan"):
            await asyncio.sleep(0.01)  # Small delay for measurable duration

        assert ctx.phase_durations["plan"] >= 10  # At least 10ms
        assert ctx.phase_spans["plan"].duration_ms >= 10

    @pytest.mark.asyncio
    async def test_trace_multiple_phases(self):
        """Test tracing multiple phases in sequence."""
        ctx = CompositionTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Multi-phase query",
        )

        async with ctx.trace_phase("decompose") as phase:
            phase.log_decomposition(2, ["CAUSAL", "DESCRIPTIVE"])

        async with ctx.trace_phase("plan") as phase:
            phase.log_planning(3, ["tool1", "tool2"], 1, 0.85)

        async with ctx.trace_phase("execute") as phase:
            phase.log_execution(2, 2, 0, 1)

        async with ctx.trace_phase("synthesize") as phase:
            phase.log_synthesis(200, 0.9, 0)

        assert len(ctx.phase_spans) == 4
        assert len(ctx.phase_durations) == 4
        assert all(
            phase in ctx.phase_durations for phase in ["decompose", "plan", "execute", "synthesize"]
        )

    def test_get_phase_index(self):
        """Test phase index lookup."""
        ctx = CompositionTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Test query",
        )

        assert ctx._get_phase_index("decompose") == 0
        assert ctx._get_phase_index("plan") == 1
        assert ctx._get_phase_index("execute") == 2
        assert ctx._get_phase_index("synthesize") == 3
        assert ctx._get_phase_index("unknown") == -1

    def test_log_composition_complete(self):
        """Test logging composition completion."""
        ctx = CompositionTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Test query",
        )
        ctx.phase_durations = {"decompose": 100, "plan": 200, "execute": 500, "synthesize": 100}

        ctx.log_composition_complete(
            status="success",
            success=True,
            total_duration_ms=900,
            sub_question_count=3,
            tools_executed=4,
            tools_succeeded=4,
            confidence=0.9,
            parallel_executions=2,
        )

        # Verify no exceptions and logs info

    def test_log_composition_complete_with_opik_span(self):
        """Test logging composition completion with Opik span."""
        mock_span = MagicMock()
        ctx = CompositionTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Test query",
            _opik_span=mock_span,
        )
        ctx.phase_durations = {"decompose": 100}

        ctx.log_composition_complete(
            status="partial",
            success=True,
            total_duration_ms=500,
            tools_executed=3,
            tools_succeeded=2,
            confidence=0.7,
        )

        mock_span.set_attribute.assert_any_call("status", "partial")
        mock_span.set_attribute.assert_any_call("success", True)
        mock_span.set_attribute.assert_any_call("total_duration_ms", 500)
        mock_span.set_output.assert_called_once()


class TestToolComposerOpikTracer:
    """Tests for ToolComposerOpikTracer class."""

    def test_init_defaults(self):
        """Test tracer initialization with defaults."""
        tracer = ToolComposerOpikTracer()

        assert tracer.project_name == "e2i-tool-composer"
        assert tracer.enabled is True
        assert tracer.sample_rate == 1.0
        assert tracer._initialized is False

    def test_init_custom_params(self):
        """Test tracer initialization with custom parameters."""
        tracer = ToolComposerOpikTracer(
            project_name="custom-project",
            enabled=False,
            sample_rate=0.5,
        )

        assert tracer.project_name == "custom-project"
        assert tracer.enabled is False
        assert tracer.sample_rate == 0.5

    @patch("src.mlops.opik_connector.get_opik_connector")
    def test_lazy_initialization(self, mock_get_connector):
        """Test lazy initialization of OpikConnector."""
        mock_connector = MagicMock()
        mock_connector.is_enabled = True
        mock_get_connector.return_value = mock_connector

        tracer = ToolComposerOpikTracer()
        assert tracer._initialized is False

        # Access is_enabled to trigger initialization
        _ = tracer.is_enabled

        assert tracer._initialized is True
        mock_get_connector.assert_called_once()

    @patch("src.mlops.opik_connector.get_opik_connector")
    def test_is_enabled_with_connector(self, mock_get_connector):
        """Test is_enabled property when connector is available."""
        mock_connector = MagicMock()
        mock_connector.is_enabled = True
        mock_get_connector.return_value = mock_connector

        tracer = ToolComposerOpikTracer()
        assert tracer.is_enabled is True

    @patch("src.mlops.opik_connector.get_opik_connector")
    def test_is_enabled_when_disabled(self, mock_get_connector):
        """Test is_enabled property when tracer is disabled."""
        mock_connector = MagicMock()
        mock_connector.is_enabled = True
        mock_get_connector.return_value = mock_connector

        tracer = ToolComposerOpikTracer(enabled=False)
        assert tracer.is_enabled is False

    @patch("src.mlops.opik_connector.get_opik_connector")
    def test_is_enabled_when_connector_unavailable(self, mock_get_connector):
        """Test is_enabled property when connector import fails."""
        mock_get_connector.side_effect = ImportError("opik not available")

        tracer = ToolComposerOpikTracer()
        assert tracer.is_enabled is False
        assert tracer._initialized is True  # Still marked as initialized

    def test_should_trace_with_full_sample_rate(self):
        """Test _should_trace returns True with 1.0 sample rate."""
        tracer = ToolComposerOpikTracer(sample_rate=1.0)
        # Should always return True
        assert all(tracer._should_trace() for _ in range(10))

    def test_should_trace_with_zero_sample_rate(self):
        """Test _should_trace returns False with 0.0 sample rate."""
        tracer = ToolComposerOpikTracer(sample_rate=0.0)
        # Should always return False
        assert not any(tracer._should_trace() for _ in range(10))

    @pytest.mark.asyncio
    async def test_trace_composition_without_opik(self):
        """Test trace_composition works without Opik connector."""
        tracer = ToolComposerOpikTracer(enabled=False)

        async with tracer.trace_composition(query="Test query") as trace:
            assert isinstance(trace, CompositionTraceContext)
            assert trace.query == "Test query"
            assert trace.trace_id  # Should have a trace ID
            assert trace._opik_span is None  # No Opik span

    @pytest.mark.asyncio
    async def test_trace_composition_with_context(self):
        """Test trace_composition with context metadata."""
        tracer = ToolComposerOpikTracer(enabled=False)

        context = {
            "session_id": "sess-123",
            "brand": "Kisqali",
            "region": "Midwest",
        }

        async with tracer.trace_composition(
            query="Analyze Kisqali performance",
            context=context,
        ) as trace:
            assert trace.metadata["session_id"] == "sess-123"
            assert trace.metadata["brand"] == "Kisqali"
            assert trace.metadata["region"] == "Midwest"

    @pytest.mark.asyncio
    async def test_trace_composition_with_metadata(self):
        """Test trace_composition with additional metadata."""
        tracer = ToolComposerOpikTracer(enabled=False)

        async with tracer.trace_composition(
            query="Test query",
            metadata={"user_id": "user-456", "priority": "high"},
        ) as trace:
            assert trace.metadata["user_id"] == "user-456"
            assert trace.metadata["priority"] == "high"
            assert trace.metadata["query_length"] == 10  # len("Test query")

    @pytest.mark.asyncio
    async def test_trace_composition_records_duration(self):
        """Test trace_composition records total duration."""
        tracer = ToolComposerOpikTracer(enabled=False)

        async with tracer.trace_composition(query="Test query") as trace:
            await asyncio.sleep(0.01)  # Small delay

        assert trace.duration_ms is not None
        assert trace.duration_ms >= 10
        assert trace.end_time is not None

    @pytest.mark.asyncio
    async def test_trace_composition_full_pipeline(self):
        """Test full composition trace with all phases."""
        tracer = ToolComposerOpikTracer(enabled=False)

        async with tracer.trace_composition(
            query="Compare rep visits vs speaker programs"
        ) as trace:
            async with trace.trace_phase("decompose") as phase:
                phase.log_decomposition(3, ["CAUSAL", "COMPARATIVE", "PREDICTIVE"])

            async with trace.trace_phase("plan") as phase:
                phase.log_planning(4, ["tool1", "tool2", "tool3", "tool4"], 2, 0.85)

            async with trace.trace_phase("execute") as phase:
                phase.log_execution(4, 3, 1, 2)

            async with trace.trace_phase("synthesize") as phase:
                phase.log_synthesis(500, 0.8, 1, ["tool_4"])

            trace.log_composition_complete(
                status="partial",
                success=True,
                total_duration_ms=1000,
                sub_question_count=3,
                tools_executed=4,
                tools_succeeded=3,
                confidence=0.8,
                parallel_executions=2,
                errors=["Tool 4 timeout"],
            )

        assert len(trace.phase_spans) == 4
        assert all(
            k in trace.phase_durations for k in ["decompose", "plan", "execute", "synthesize"]
        )


class TestTraceCompositionDecorator:
    """Tests for trace_composition decorator."""

    @pytest.mark.asyncio
    async def test_decorator_creates_trace(self):
        """Test decorator creates trace context."""

        @trace_composition()
        async def compose(trace: CompositionTraceContext, query: str, context=None):
            assert isinstance(trace, CompositionTraceContext)
            assert trace.query == query
            return {"result": "success"}

        result = await compose(query="Test query")
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_decorator_with_custom_params(self):
        """Test decorator with custom parameter names."""

        @trace_composition(query_param="q", context_param="ctx")
        async def compose(trace: CompositionTraceContext, q: str, ctx=None):
            return trace.query

        result = await compose(q="Custom query")
        assert result == "Custom query"

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves original function metadata."""

        @trace_composition()
        async def my_compose_function(trace: CompositionTraceContext, query: str):
            """This is the docstring."""
            return query

        assert my_compose_function.__name__ == "my_compose_function"
        # Decorated function wraps the original


class TestGetToolComposerTracer:
    """Tests for get_tool_composer_tracer singleton."""

    def test_returns_tracer_instance(self):
        """Test get_tool_composer_tracer returns tracer instance."""
        tracer = get_tool_composer_tracer()
        assert isinstance(tracer, ToolComposerOpikTracer)

    def test_returns_same_instance(self):
        """Test get_tool_composer_tracer returns same instance."""
        tracer1 = get_tool_composer_tracer()
        tracer2 = get_tool_composer_tracer()
        assert tracer1 is tracer2

    def test_first_call_sets_config(self):
        """Test first call to get_tool_composer_tracer sets config."""
        tracer = get_tool_composer_tracer(
            project_name="custom-project",
            enabled=False,
            sample_rate=0.5,
        )

        assert tracer.project_name == "custom-project"
        assert tracer.enabled is False
        assert tracer.sample_rate == 0.5

    def test_subsequent_calls_ignore_config(self):
        """Test subsequent calls ignore config (uses cached instance)."""
        tracer1 = get_tool_composer_tracer(project_name="first-project")
        tracer2 = get_tool_composer_tracer(project_name="second-project")

        assert tracer1 is tracer2
        assert tracer2.project_name == "first-project"


class TestResetToolComposerTracer:
    """Tests for reset_tool_composer_tracer function."""

    def test_reset_clears_singleton(self):
        """Test reset_tool_composer_tracer clears the singleton."""
        tracer1 = get_tool_composer_tracer(project_name="project-a")

        reset_tool_composer_tracer()

        tracer2 = get_tool_composer_tracer(project_name="project-b")

        assert tracer1 is not tracer2
        assert tracer2.project_name == "project-b"


class TestOpikIntegration:
    """Integration tests with mocked Opik connector."""

    @pytest.mark.asyncio
    @patch("src.mlops.opik_connector.get_opik_connector")
    async def test_trace_composition_with_opik_enabled(self, mock_get_connector):
        """Test trace_composition with Opik connector enabled."""
        # Create mock connector with trace_agent context manager
        mock_span = MagicMock()
        mock_span.set_output = MagicMock()

        mock_connector = MagicMock()
        mock_connector.is_enabled = True

        # Create async context manager for trace_agent
        @asynccontextmanager
        async def mock_trace_agent(**kwargs):
            yield mock_span

        mock_connector.trace_agent = mock_trace_agent
        mock_get_connector.return_value = mock_connector

        tracer = ToolComposerOpikTracer(sample_rate=1.0)

        async with tracer.trace_composition(query="Test with Opik") as trace:
            assert trace._opik_span is mock_span

    @pytest.mark.asyncio
    @patch("src.mlops.opik_connector.get_opik_connector")
    async def test_trace_phase_with_opik_enabled(self, mock_get_connector):
        """Test trace_phase creates child spans in Opik."""
        mock_span = MagicMock()
        mock_child_span = MagicMock()

        mock_connector = MagicMock()
        mock_connector.is_enabled = True

        @asynccontextmanager
        async def mock_trace_agent(**kwargs):
            # Return parent span for root, child span for phases
            if kwargs.get("operation") == "compose":
                yield mock_span
            else:
                yield mock_child_span

        mock_connector.trace_agent = mock_trace_agent
        mock_get_connector.return_value = mock_connector

        tracer = ToolComposerOpikTracer(sample_rate=1.0)

        async with tracer.trace_composition(query="Test phases") as trace:
            async with trace.trace_phase("decompose") as phase:
                phase.log_decomposition(2, ["CAUSAL"])
                # Phase span should be mock_child_span
                assert phase._opik_span is mock_child_span

    @pytest.mark.asyncio
    @patch("src.mlops.opik_connector.get_opik_connector")
    async def test_graceful_degradation_on_opik_error(self, mock_get_connector):
        """Test tracing continues gracefully when Opik fails."""
        mock_connector = MagicMock()
        mock_connector.is_enabled = True

        @asynccontextmanager
        async def mock_trace_agent(**kwargs):
            raise RuntimeError("Opik connection failed")
            yield  # Never reached

        mock_connector.trace_agent = mock_trace_agent
        mock_get_connector.return_value = mock_connector

        tracer = ToolComposerOpikTracer(sample_rate=1.0)

        # Should not raise, should fall back to non-traced version
        async with tracer.trace_composition(query="Test fallback") as trace:
            assert trace._opik_span is None  # No span due to error

            async with trace.trace_phase("decompose") as phase:
                phase.log_decomposition(1, ["DESCRIPTIVE"])

        # Execution should complete successfully
        assert "decompose" in trace.phase_spans


# Import for async context manager helper
from contextlib import asynccontextmanager
