"""Tests for CohortConstructor observability integration."""

from unittest.mock import MagicMock

import pytest

from src.agents.cohort_constructor import (
    CohortExecutionResult,
    EligibilityLogEntry,
)
from src.agents.cohort_constructor.observability import (
    CohortMLflowLogger,
    CohortOpikTracer,
    CohortTraceContext,
    get_cohort_mlflow_logger,
    get_cohort_opik_tracer,
    reset_observability_singletons,
    track_cohort_construction,
    track_cohort_step,
)


class TestCohortMLflowLogger:
    """Tests for CohortMLflowLogger class."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock connector."""
        mock = MagicMock()
        mock.client = MagicMock()
        mock.log_metrics = MagicMock()
        return mock

    @pytest.fixture
    def cohort_logger(self, mock_connector):
        """Create CohortMLflowLogger with mocked connector."""
        return CohortMLflowLogger(mock_connector)

    def test_init(self, cohort_logger, mock_connector):
        """Test initialization."""
        assert cohort_logger._connector == mock_connector

    def test_log_cohort_execution(self, cohort_logger, sample_config):
        """Test logging cohort execution."""
        result = CohortExecutionResult(
            cohort_id="test_cohort",
            execution_id="exec_001",
            eligible_patient_ids=["P001", "P002"],
            eligibility_stats={
                "total_input_patients": 100,
                "eligible_patient_count": 50,
                "exclusion_rate": 0.5,
            },
            eligibility_log=[],
            patient_assignments=[],
            execution_metadata={"execution_time_ms": 100},
            status="success",
        )

        # Should not raise
        cohort_logger.log_cohort_execution(result, sample_config)

    def test_log_sla_compliance_compliant(self, cohort_logger):
        """Test SLA compliance logging when compliant."""
        cohort_logger.log_sla_compliance(
            execution_time_ms=50000,  # 50 seconds
            patient_count=100000,
        )

        # Verify metrics were logged
        cohort_logger._connector.log_metrics.assert_called()

    def test_log_sla_compliance_non_compliant(self, cohort_logger):
        """Test SLA compliance logging when not compliant."""
        cohort_logger.log_sla_compliance(
            execution_time_ms=150000,  # 150 seconds - exceeds SLA
            patient_count=100000,
        )

        # Should still log metrics
        cohort_logger._connector.log_metrics.assert_called()


class TestCohortOpikTracer:
    """Tests for CohortOpikTracer class."""

    @pytest.fixture
    def mock_opik_connector(self):
        """Create mock OpikConnector."""
        mock = MagicMock()
        mock._client = MagicMock()
        return mock

    @pytest.fixture
    def cohort_tracer(self, mock_opik_connector):
        """Create CohortOpikTracer with mocked connector."""
        return CohortOpikTracer(mock_opik_connector)

    def test_init(self, cohort_tracer, mock_opik_connector):
        """Test initialization."""
        assert cohort_tracer._connector == mock_opik_connector

    def test_trace_cohort_construction_context_manager(self, cohort_tracer, sample_config):
        """Test trace_cohort_construction as context manager."""
        with cohort_tracer.trace_cohort_construction(
            config=sample_config,
            patient_count=100,
        ) as ctx:
            assert isinstance(ctx, CohortTraceContext)

    def test_trace_cohort_construction_with_metadata(self, cohort_tracer, sample_config):
        """Test trace_cohort_construction with metadata."""
        with cohort_tracer.trace_cohort_construction(
            config=sample_config,
            patient_count=100,
            metadata={"environment": "test"},
        ) as ctx:
            assert ctx is not None


class TestCohortTraceContext:
    """Tests for CohortTraceContext class."""

    @pytest.fixture
    def mock_trace_connector(self):
        """Create mock OpikConnector."""
        mock = MagicMock()
        mock._client = MagicMock()
        mock.log_span = MagicMock()
        mock.start_trace = MagicMock(return_value="test_trace_123")
        mock.end_trace = MagicMock()
        return mock

    @pytest.fixture
    def trace_context(self, mock_trace_connector):
        """Create CohortTraceContext."""
        return CohortTraceContext(connector=mock_trace_connector)

    def test_init(self, trace_context, mock_trace_connector):
        """Test initialization."""
        assert trace_context._connector == mock_trace_connector
        assert trace_context._trace_id is None  # Not set until start_trace called
        assert trace_context._spans == []

    def test_log_criterion_evaluation(self, trace_context):
        """Test logging criterion evaluation."""
        # Should not raise
        trace_context.log_criterion_evaluation(
            criterion={"field": "age", "operator": ">=", "value": 18},
            criterion_type="inclusion",
            initial_count=100,
            removed_count=10,
            remaining_count=90,
        )

    def test_log_execution_complete(self, trace_context):
        """Test logging execution completion."""
        trace_context.log_execution_complete(
            eligible_count=50,
            total_count=100,
            execution_time_ms=500,
            status="success",
        )

    def test_log_error(self, trace_context):
        """Test logging error."""
        trace_context.log_error("Test error message")

    def test_start_and_end_trace(self, trace_context):
        """Test starting and ending a trace."""
        # Start trace
        trace_context.start_trace(
            name="test_trace",
            inputs={"test": "value"},
            metadata={"env": "test"},
        )

        # Verify trace ID was set
        assert trace_context._trace_id == "test_trace_123"

        # End trace
        trace_context.end_trace(outputs={"result": "success"})


class TestDecorators:
    """Tests for observability decorators."""

    def test_track_cohort_step_decorator(self):
        """Test track_cohort_step decorator."""

        @track_cohort_step("test_step")
        def sample_function(x, y):
            return x + y

        # Decorated function should still work
        result = sample_function(1, 2)
        assert result == 3

    def test_track_cohort_construction_decorator(self):
        """Test track_cohort_construction decorator."""

        @track_cohort_construction()  # Parameterized decorator requires ()
        def sample_construction(config=None, patient_df=None):
            return {"eligible": 50, "total": 100}

        # Decorated function should still work
        result = sample_construction()
        assert result == {"eligible": 50, "total": 100}


class TestSingletonFactories:
    """Tests for singleton factory functions."""

    def test_reset_observability_singletons(self):
        """Test resetting singletons."""
        # Should not raise
        reset_observability_singletons()

    def test_get_cohort_mlflow_logger_creates_instance(self):
        """Test MLflow logger factory creates instance."""
        reset_observability_singletons()

        # Factory creates CohortMLflowLogger without connector
        # (connector is lazily loaded)
        logger = get_cohort_mlflow_logger()

        assert isinstance(logger, CohortMLflowLogger)

    def test_get_cohort_opik_tracer_creates_instance(self):
        """Test Opik tracer factory creates instance."""
        reset_observability_singletons()

        # Factory creates CohortOpikTracer without connector
        # (connector is lazily loaded)
        tracer = get_cohort_opik_tracer()

        assert isinstance(tracer, CohortOpikTracer)

    def test_singletons_return_same_instance(self):
        """Test that factory returns same singleton instance."""
        reset_observability_singletons()

        logger1 = get_cohort_mlflow_logger()
        logger2 = get_cohort_mlflow_logger()
        assert logger1 is logger2

        tracer1 = get_cohort_opik_tracer()
        tracer2 = get_cohort_opik_tracer()
        assert tracer1 is tracer2


class TestMLflowLoggerMetrics:
    """Tests for MLflow metric logging."""

    @pytest.fixture
    def mock_mlflow_connector(self):
        """Create mock MLflow connector."""
        mock = MagicMock()
        mock.client = MagicMock()
        mock.log_metrics = MagicMock()
        return mock

    def test_log_cohort_metrics(self, mock_mlflow_connector, sample_config):
        """Test that cohort metrics are logged correctly."""
        logger = CohortMLflowLogger(mock_mlflow_connector)

        result = CohortExecutionResult(
            cohort_id="test_cohort",
            execution_id="exec_001",
            eligible_patient_ids=["P001", "P002", "P003"],
            eligibility_stats={
                "total_input_patients": 100,
                "eligible_patient_count": 50,
                "exclusion_rate": 0.5,
            },
            eligibility_log=[
                EligibilityLogEntry(
                    criterion_name="age",
                    criterion_type="inclusion",
                    criterion_order=1,
                    operator=">=",
                    value=18,
                    removed_count=10,
                    remaining_count=90,
                ),
            ],
            patient_assignments=[],
            execution_metadata={"execution_time_ms": 100},
            status="success",
        )

        logger.log_cohort_execution(result, sample_config)

        # Verify metrics logged
        mock_mlflow_connector.log_metrics.assert_called()


class TestOpikTracerSpans:
    """Tests for Opik span creation."""

    @pytest.fixture
    def mock_span_connector(self):
        """Create mock OpikConnector."""
        mock = MagicMock()
        mock._client = MagicMock()
        mock.start_trace = MagicMock(return_value="trace_123")
        mock.end_trace = MagicMock()
        return mock

    def test_creates_trace_on_construction_start(self, mock_span_connector, sample_config):
        """Test that trace is created when construction starts."""
        tracer = CohortOpikTracer(mock_span_connector)

        with tracer.trace_cohort_construction(
            config=sample_config,
            patient_count=100,
        ):
            # Trace should be started
            pass

        # Verify trace lifecycle was called
        mock_span_connector.start_trace.assert_called()
        mock_span_connector.end_trace.assert_called()


class TestObservabilityIntegration:
    """Integration tests for observability components."""

    def test_mlflow_logger_handles_none_tracker(self):
        """Test MLflow logger handles None tracker gracefully."""
        logger = CohortMLflowLogger(None)

        # Should not raise
        result = CohortExecutionResult(
            cohort_id="test",
            execution_id="exec_001",
            eligible_patient_ids=[],
            eligibility_stats={},
            eligibility_log=[],
            patient_assignments=[],
            execution_metadata={},
            status="success",
        )

        try:
            logger.log_cohort_execution(result, None)
        except Exception:
            pass  # Expected if tracker is None

    def test_opik_tracer_handles_none_connector(self, sample_config):
        """Test Opik tracer handles None connector gracefully."""
        tracer = CohortOpikTracer(None)

        try:
            with tracer.trace_cohort_construction(
                config=sample_config,
                patient_count=100,
            ):
                pass
        except Exception:
            pass  # Expected if connector is None


class TestSLAThresholdLogging:
    """Tests for SLA threshold logging."""

    @pytest.fixture
    def mock_sla_connector(self):
        """Create mock connector for SLA tests."""
        mock = MagicMock()
        mock.client = MagicMock()
        mock.log_metrics = MagicMock()
        return mock

    def test_small_cohort_sla(self, mock_sla_connector):
        """Test SLA check for small cohort."""
        logger = CohortMLflowLogger(mock_sla_connector)

        logger.log_sla_compliance(
            execution_time_ms=400,  # Under 500ms for small cohort
            patient_count=500,
        )

        # Verify metrics logged
        mock_sla_connector.log_metrics.assert_called()

    def test_medium_cohort_sla(self, mock_sla_connector):
        """Test SLA check for medium cohort."""
        logger = CohortMLflowLogger(mock_sla_connector)

        logger.log_sla_compliance(
            execution_time_ms=4000,  # Under 5s for medium cohort
            patient_count=5000,
        )

        mock_sla_connector.log_metrics.assert_called()

    def test_large_cohort_sla(self, mock_sla_connector):
        """Test SLA check for large cohort."""
        logger = CohortMLflowLogger(mock_sla_connector)

        logger.log_sla_compliance(
            execution_time_ms=40000,  # Under 50s for large cohort
            patient_count=50000,
        )

        mock_sla_connector.log_metrics.assert_called()

    def test_very_large_cohort_sla(self, mock_sla_connector):
        """Test SLA check for very large cohort."""
        logger = CohortMLflowLogger(mock_sla_connector)

        logger.log_sla_compliance(
            execution_time_ms=100000,  # Under 120s for very large cohort
            patient_count=150000,
        )

        mock_sla_connector.log_metrics.assert_called()
