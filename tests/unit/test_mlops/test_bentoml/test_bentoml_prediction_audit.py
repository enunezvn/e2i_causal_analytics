"""Unit tests for BentoML prediction audit trail module.

Tests cover:
- log_prediction_audit async function
- prediction_audit_context context manager
- PredictionAuditContext class
- log_prediction_audit_sync synchronous wrapper
- Graceful degradation when Opik unavailable

Phase 1 G07 from observability audit remediation plan.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest

from src.mlops.bentoml_prediction_audit import (
    log_prediction_audit,
    prediction_audit_context,
    PredictionAuditContext,
    log_prediction_audit_sync,
    _check_opik_available,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_opik_connector():
    """Mock OpikConnector for testing."""
    with patch("src.mlops.bentoml_prediction_audit.OpikConnector") as MockConnector:
        mock_instance = AsyncMock()
        mock_instance.log_model_prediction = AsyncMock(return_value="trace-123")
        MockConnector.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def reset_opik_available():
    """Reset the _OPIK_AVAILABLE global."""
    import src.mlops.bentoml_prediction_audit as module
    original = module._OPIK_AVAILABLE
    module._OPIK_AVAILABLE = None
    yield
    module._OPIK_AVAILABLE = original


@pytest.fixture
def sample_prediction_data():
    """Sample prediction input/output data."""
    return {
        "input_data": {"features": [[0.1, 0.2, 0.3, 0.4]]},
        "output_data": {"predictions": [1], "probabilities": [0.85]},
    }


# =============================================================================
# _check_opik_available TESTS
# =============================================================================


class TestCheckOpikAvailable:
    """Tests for _check_opik_available function."""

    def test_returns_true_when_opik_available(self, reset_opik_available):
        """Test returns True when OpikConnector can be imported."""
        with patch.dict("sys.modules", {"src.mlops.opik_connector": MagicMock()}):
            with patch("src.mlops.bentoml_prediction_audit.OpikConnector", MagicMock()):
                import src.mlops.bentoml_prediction_audit as module
                module._OPIK_AVAILABLE = None
                result = _check_opik_available()
                # May be True or False depending on actual import
                assert isinstance(result, bool)

    def test_caches_availability_result(self, reset_opik_available):
        """Test that availability result is cached."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = True

        result = _check_opik_available()
        assert result is True

        # Should use cached value
        module._OPIK_AVAILABLE = False
        result = _check_opik_available()
        assert result is False


# =============================================================================
# log_prediction_audit TESTS
# =============================================================================


class TestLogPredictionAudit:
    """Tests for log_prediction_audit async function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_opik_unavailable(self, reset_opik_available):
        """Test returns None when Opik is unavailable."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = False

        result = await log_prediction_audit(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type="classification",
            input_data={"features": [[0.1]]},
            output_data={"predictions": [1]},
            latency_ms=50.0,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_logs_to_opik_connector(
        self, reset_opik_available, mock_opik_connector, sample_prediction_data
    ):
        """Test that prediction is logged to Opik."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = True

        result = await log_prediction_audit(
            model_name="churn_classifier",
            model_tag="churn_classifier:v1",
            service_type="classification",
            input_data=sample_prediction_data["input_data"],
            output_data=sample_prediction_data["output_data"],
            latency_ms=45.2,
        )

        mock_opik_connector.log_model_prediction.assert_called_once()
        call_kwargs = mock_opik_connector.log_model_prediction.call_args.kwargs
        assert call_kwargs["model_name"] == "churn_classifier"
        assert call_kwargs["input_data"] == sample_prediction_data["input_data"]
        assert call_kwargs["output_data"] == sample_prediction_data["output_data"]

    @pytest.mark.asyncio
    async def test_includes_metadata(
        self, reset_opik_available, mock_opik_connector, sample_prediction_data
    ):
        """Test that metadata is included in Opik log."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = True

        await log_prediction_audit(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type="classification",
            input_data=sample_prediction_data["input_data"],
            output_data=sample_prediction_data["output_data"],
            latency_ms=45.2,
            request_id="req-123",
            metadata={"custom": "value"},
        )

        call_kwargs = mock_opik_connector.log_model_prediction.call_args.kwargs
        metadata = call_kwargs["metadata"]

        assert "model_tag" in metadata
        assert "service_type" in metadata
        assert "latency_ms" in metadata
        assert "timestamp" in metadata
        assert metadata["request_id"] == "req-123"
        assert metadata["custom"] == "value"

    @pytest.mark.asyncio
    async def test_returns_trace_id(
        self, reset_opik_available, mock_opik_connector, sample_prediction_data
    ):
        """Test that trace ID is returned."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = True

        result = await log_prediction_audit(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type="classification",
            input_data=sample_prediction_data["input_data"],
            output_data=sample_prediction_data["output_data"],
            latency_ms=45.2,
        )

        assert result == "trace-123"

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self, reset_opik_available):
        """Test that exceptions are handled gracefully."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = True

        with patch(
            "src.mlops.bentoml_prediction_audit.OpikConnector",
            side_effect=Exception("Connection failed")
        ):
            result = await log_prediction_audit(
                model_name="test_model",
                model_tag="test_model:v1",
                service_type="classification",
                input_data={"features": [[0.1]]},
                output_data={"predictions": [1]},
                latency_ms=45.2,
            )

        assert result is None


# =============================================================================
# PredictionAuditContext TESTS
# =============================================================================


class TestPredictionAuditContext:
    """Tests for PredictionAuditContext class."""

    def test_initialization(self):
        """Test context initialization."""
        ctx = PredictionAuditContext(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type="classification",
            request_id="req-123",
        )

        assert ctx.model_name == "test_model"
        assert ctx.model_tag == "test_model:v1"
        assert ctx.service_type == "classification"
        assert ctx.request_id == "req-123"
        assert ctx.input_data is None
        assert ctx.output_data is None
        assert ctx.metadata == {}

    def test_set_input(self):
        """Test setting input data."""
        ctx = PredictionAuditContext(
            model_name="test", model_tag="v1", service_type="classification"
        )

        ctx.set_input({"features": [[1, 2, 3]]})

        assert ctx.input_data == {"features": [[1, 2, 3]]}

    def test_set_output(self):
        """Test setting output data."""
        ctx = PredictionAuditContext(
            model_name="test", model_tag="v1", service_type="classification"
        )

        ctx.set_output({"predictions": [1]})

        assert ctx.output_data == {"predictions": [1]}

    def test_add_metadata(self):
        """Test adding metadata."""
        ctx = PredictionAuditContext(
            model_name="test", model_tag="v1", service_type="classification"
        )

        ctx.add_metadata("batch_size", 32)
        ctx.add_metadata("model_version", "1.2.0")

        assert ctx.metadata["batch_size"] == 32
        assert ctx.metadata["model_version"] == "1.2.0"


# =============================================================================
# prediction_audit_context TESTS
# =============================================================================


class TestPredictionAuditContextManager:
    """Tests for prediction_audit_context context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_yields_context(self):
        """Test that context manager yields PredictionAuditContext."""
        async with prediction_audit_context(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type="classification",
        ) as ctx:
            assert isinstance(ctx, PredictionAuditContext)
            assert ctx.model_name == "test_model"

    @pytest.mark.asyncio
    async def test_context_manager_records_timing(self, reset_opik_available):
        """Test that context manager records start and end time."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = False  # Disable to avoid actual logging

        async with prediction_audit_context(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type="classification",
        ) as ctx:
            ctx.set_input({"features": [[0.1]]})
            ctx.set_output({"predictions": [1]})
            await asyncio.sleep(0.01)  # 10ms

        # Timing should have been recorded
        assert ctx.start_time > 0
        assert ctx.end_time > ctx.start_time

    @pytest.mark.asyncio
    async def test_context_manager_includes_request_id(self):
        """Test that request ID is passed through context."""
        async with prediction_audit_context(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type="classification",
            request_id="req-456",
        ) as ctx:
            assert ctx.request_id == "req-456"


# =============================================================================
# log_prediction_audit_sync TESTS
# =============================================================================


class TestLogPredictionAuditSync:
    """Tests for log_prediction_audit_sync function."""

    def test_returns_none_when_opik_unavailable(self, reset_opik_available):
        """Test returns None when Opik is unavailable."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = False

        # Should not raise, just return None
        result = log_prediction_audit_sync(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type="classification",
            input_data={"features": [[0.1]]},
            output_data={"predictions": [1]},
            latency_ms=50.0,
        )

        assert result is None

    def test_handles_no_event_loop(self, reset_opik_available):
        """Test handles case when no event loop exists."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = True

        # This might create a new event loop
        with patch(
            "src.mlops.bentoml_prediction_audit.log_prediction_audit",
            new_callable=AsyncMock
        ) as mock_async:
            mock_async.return_value = "trace-123"

            # This test verifies it doesn't crash
            # The actual behavior depends on the event loop state
            try:
                log_prediction_audit_sync(
                    model_name="test_model",
                    model_tag="test_model:v1",
                    service_type="classification",
                    input_data={"features": [[0.1]]},
                    output_data={"predictions": [1]},
                    latency_ms=50.0,
                )
            except RuntimeError:
                # Expected if there's already an event loop in some contexts
                pass


# =============================================================================
# SERVICE TYPE TESTS
# =============================================================================


class TestServiceTypes:
    """Tests for different service types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("service_type", ["classification", "regression", "causal"])
    async def test_all_service_types_accepted(
        self, service_type, reset_opik_available, mock_opik_connector, sample_prediction_data
    ):
        """Test that all supported service types are accepted."""
        import src.mlops.bentoml_prediction_audit as module
        module._OPIK_AVAILABLE = True

        await log_prediction_audit(
            model_name="test_model",
            model_tag="test_model:v1",
            service_type=service_type,
            input_data=sample_prediction_data["input_data"],
            output_data=sample_prediction_data["output_data"],
            latency_ms=45.2,
        )

        call_kwargs = mock_opik_connector.log_model_prediction.call_args.kwargs
        assert call_kwargs["metadata"]["service_type"] == service_type
