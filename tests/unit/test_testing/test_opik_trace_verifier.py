"""Unit tests for OpikTraceVerifier.

Tests the Opik trace verification functionality for agent observability.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.testing.opik_trace_verifier import (
    OpikTraceVerifier,
    TraceVerificationResult,
    verify_opik_available,
)

# Set testing mode
os.environ["E2I_TESTING_MODE"] = "true"


@pytest.mark.unit
class TestTraceVerificationResult:
    """Test TraceVerificationResult dataclass."""

    def test_summary_trace_not_found(self):
        """Test summary for trace not found."""
        result = TraceVerificationResult(
            trace_exists=False,
            error="Trace not found",
        )
        assert "TRACE NOT FOUND" in result.summary
        assert "Trace not found" in result.summary

    def test_summary_valid_trace(self):
        """Test summary for valid trace."""
        result = TraceVerificationResult(
            trace_exists=True,
            metadata_valid=True,
            span_count=5,
            duration_ms=123.4,
        )
        assert "VALID" in result.summary
        assert "5 spans" in result.summary
        assert "123.4ms" in result.summary

    def test_summary_invalid_metadata(self):
        """Test summary for invalid metadata."""
        result = TraceVerificationResult(
            trace_exists=True,
            metadata_valid=False,
            span_count=3,
            duration_ms=50.0,
        )
        assert "INVALID" in result.summary
        assert "3 spans" in result.summary


@pytest.mark.unit
class TestOpikTraceVerifier:
    """Test OpikTraceVerifier class."""

    @pytest.fixture
    def verifier(self):
        """Create an OpikTraceVerifier instance."""
        return OpikTraceVerifier(opik_base_url="http://localhost:5173", timeout=5.0)

    @pytest.mark.asyncio
    async def test_verify_trace_exists_success(self, verifier):
        """Test verifying trace exists when it does."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.verify_trace_exists("trace_123")
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_trace_exists_not_found(self, verifier):
        """Test verifying trace exists when it doesn't."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.verify_trace_exists("trace_123")
            assert result is False

    @pytest.mark.asyncio
    async def test_verify_trace_exists_http_error(self, verifier):
        """Test verifying trace exists when HTTP error occurs."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

            result = await verifier.verify_trace_exists("trace_123")
            assert result is False

    @pytest.mark.asyncio
    async def test_get_trace_details_success(self, verifier):
        """Test getting trace details successfully."""
        trace_data = {
            "trace_id": "trace_123",
            "metadata": {"agent": "test_agent"},
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": "2024-01-01T12:00:01Z",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = trace_data
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.get_trace_details("trace_123")
            assert result == trace_data

    @pytest.mark.asyncio
    async def test_get_trace_details_not_found(self, verifier):
        """Test getting trace details when trace not found."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.get_trace_details("trace_123")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_trace_spans(self, verifier):
        """Test getting trace spans."""
        spans_data = {
            "content": [
                {"span_id": "span_1", "name": "init"},
                {"span_id": "span_2", "name": "process"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = spans_data
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.get_trace_spans("trace_123")
            assert len(result) == 2
            assert result[0]["name"] == "init"

    @pytest.mark.asyncio
    async def test_verify_agent_trace_success(self, verifier):
        """Test verifying agent trace with valid metadata."""
        trace_data = {
            "trace_id": "trace_123",
            "metadata": {
                "agent_name": "causal_impact",
                "tier": 2,
                "framework": "langgraph",
            },
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": "2024-01-01T12:00:01.500Z",
        }

        spans_data = {
            "content": [
                {"name": "analyze_impact"},
                {"name": "estimate_ate"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock both get_trace_details and get_trace_spans calls
            trace_response = MagicMock()
            trace_response.status_code = 200
            trace_response.json.return_value = trace_data

            spans_response = MagicMock()
            spans_response.status_code = 200
            spans_response.json.return_value = spans_data

            mock_client.get = AsyncMock(side_effect=[trace_response, spans_response])

            result = await verifier.verify_agent_trace(
                agent_name="causal_impact",
                trace_id="trace_123",
                tier=2,
            )

            assert result.trace_exists is True
            assert result.metadata_valid is True
            assert result.span_count == 2
            assert result.duration_ms is not None
            assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_verify_agent_trace_metadata_mismatch(self, verifier):
        """Test verifying agent trace with metadata mismatch."""
        trace_data = {
            "trace_id": "trace_123",
            "metadata": {
                "agent_name": "wrong_agent",
                "tier": 3,
                "framework": "langgraph",
            },
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": "2024-01-01T12:00:01Z",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            trace_response = MagicMock()
            trace_response.status_code = 200
            trace_response.json.return_value = trace_data

            spans_response = MagicMock()
            spans_response.status_code = 200
            spans_response.json.return_value = {"content": []}

            mock_client.get = AsyncMock(side_effect=[trace_response, spans_response])

            result = await verifier.verify_agent_trace(
                agent_name="causal_impact",
                trace_id="trace_123",
                tier=2,
            )

            assert result.trace_exists is True
            assert result.metadata_valid is False
            assert len(result.metadata_errors) > 0

    @pytest.mark.asyncio
    async def test_verify_trace_metadata(self, verifier):
        """Test verifying trace metadata."""
        trace_data = {
            "metadata": {
                "custom_field": "custom_value",
                "another_field": 123,
            }
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = trace_data
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.verify_trace_metadata(
                trace_id="trace_123",
                expected_metadata={"custom_field": "custom_value"},
            )

            assert result.trace_exists is True
            assert result.metadata_valid is True

    @pytest.mark.asyncio
    async def test_check_opik_health_healthy(self, verifier):
        """Test Opik health check when service is healthy."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.check_opik_health()

            assert result["healthy"] is True
            assert result["base_url"] == "http://localhost:5173"
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_check_opik_health_connection_error(self, verifier):
        """Test Opik health check when connection fails."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

            result = await verifier.check_opik_health()

            assert result["healthy"] is False
            assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_check_opik_health_timeout(self, verifier):
        """Test Opik health check when request times out."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

            result = await verifier.check_opik_health()

            assert result["healthy"] is False
            assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_recent_traces(self, verifier):
        """Test listing recent traces."""
        traces_data = {
            "content": [
                {"trace_id": "trace_1", "project_name": "test_project"},
                {"trace_id": "trace_2", "project_name": "test_project"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = traces_data
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.list_recent_traces(project_name="test_project", limit=10)

            assert len(result) == 2
            assert result[0]["trace_id"] == "trace_1"

    @pytest.mark.asyncio
    async def test_custom_base_url(self):
        """Test verifier with custom base URL."""
        custom_verifier = OpikTraceVerifier(opik_base_url="http://custom:8080/")
        assert custom_verifier.base_url == "http://custom:8080"

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        """Test verifier with custom timeout."""
        custom_verifier = OpikTraceVerifier(timeout=30.0)
        assert custom_verifier.timeout == 30.0


@pytest.mark.unit
class TestVerifyOpikAvailable:
    """Test verify_opik_available convenience function."""

    @pytest.mark.asyncio
    async def test_verify_opik_available_success(self):
        """Test verify_opik_available when Opik is available."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verify_opik_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_opik_available_failure(self):
        """Test verify_opik_available when Opik is unavailable."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))

            result = await verify_opik_available()
            assert result is False


@pytest.mark.unit
class TestOpikTraceVerifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def verifier(self):
        return OpikTraceVerifier()

    @pytest.mark.asyncio
    async def test_trace_with_error_captured(self, verifier):
        """Test trace verification with error captured."""
        trace_data = {
            "metadata": {},
            "error": "Some error occurred",
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": "2024-01-01T12:00:01Z",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            trace_response = MagicMock()
            trace_response.status_code = 200
            trace_response.json.return_value = trace_data

            spans_response = MagicMock()
            spans_response.status_code = 200
            spans_response.json.return_value = {"content": []}

            mock_client.get = AsyncMock(side_effect=[trace_response, spans_response])

            result = await verifier.verify_agent_trace(
                agent_name="test_agent",
                trace_id="trace_123",
                tier=1,
            )

            assert result.error_captured is True

    @pytest.mark.asyncio
    async def test_trace_without_timestamps(self, verifier):
        """Test trace verification without timestamps."""
        trace_data = {
            "metadata": {
                "agent_name": "test_agent",
                "tier": 1,
                "framework": "langgraph",
            },
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            trace_response = MagicMock()
            trace_response.status_code = 200
            trace_response.json.return_value = trace_data

            spans_response = MagicMock()
            spans_response.status_code = 200
            spans_response.json.return_value = {"content": []}

            mock_client.get = AsyncMock(side_effect=[trace_response, spans_response])

            result = await verifier.verify_agent_trace(
                agent_name="test_agent",
                trace_id="trace_123",
                tier=1,
            )

            assert result.duration_ms is None

    @pytest.mark.asyncio
    async def test_spans_with_alternative_format(self, verifier):
        """Test getting spans with alternative response format."""
        spans_data = {
            "spans": [  # Alternative key
                {"name": "span_1"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = spans_data
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await verifier.get_trace_spans("trace_123")
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_empty_metadata(self, verifier):
        """Test verification with empty metadata."""
        trace_data = {"metadata": {}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            trace_response = MagicMock()
            trace_response.status_code = 200
            trace_response.json.return_value = trace_data

            spans_response = MagicMock()
            spans_response.status_code = 200
            spans_response.json.return_value = {"content": []}

            mock_client.get = AsyncMock(side_effect=[trace_response, spans_response])

            result = await verifier.verify_agent_trace(
                agent_name="test_agent",
                trace_id="trace_123",
                tier=1,
            )

            assert result.trace_exists is True
            assert result.metadata_valid is False
            assert len(result.metadata_errors) > 0
