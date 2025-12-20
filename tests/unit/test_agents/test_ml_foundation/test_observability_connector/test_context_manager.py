"""Tests for context_manager nodes (create_context, extract_context, inject_context)."""

import pytest
from src.agents.ml_foundation.observability_connector.nodes.context_manager import (
    create_context,
    extract_context,
    inject_context,
)


class TestCreateContext:
    """Test create_context node."""

    @pytest.mark.asyncio
    async def test_create_context_success(self):
        """Test successful context creation."""
        state = {
            "request_id": "req_123",
            "experiment_id": "exp_456",
            "user_id": "user_789",
            "sample_rate": 1.0,
        }

        result = await create_context(state)

        assert "current_trace_id" in result
        assert "current_span_id" in result
        assert result["current_parent_span_id"] is None  # Root span
        assert result["request_id"] == "req_123"
        assert result["experiment_id"] == "exp_456"
        assert result["user_id"] == "user_789"
        assert result["sampled"] in [True, False]
        assert result["sample_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_create_context_minimal(self):
        """Test context creation with minimal inputs."""
        state = {}

        result = await create_context(state)

        assert "current_trace_id" in result
        assert "current_span_id" in result
        assert "request_id" in result
        assert result["sampled"] in [True, False]

    @pytest.mark.asyncio
    async def test_create_context_generates_trace_id(self):
        """Test that trace_id is generated."""
        state = {"request_id": "req_123"}

        result = await create_context(state)

        trace_id = result["current_trace_id"]
        assert len(trace_id) == 32  # UUID hex is 32 chars

    @pytest.mark.asyncio
    async def test_create_context_generates_span_id(self):
        """Test that span_id is generated."""
        state = {"request_id": "req_123"}

        result = await create_context(state)

        span_id = result["current_span_id"]
        assert len(span_id) == 16  # Span ID is 16 chars

    @pytest.mark.asyncio
    async def test_create_context_sampling(self):
        """Test sampling with different rates."""
        # Test 100% sampling
        state = {"sample_rate": 1.0}
        result = await create_context(state)
        assert result["sampled"] is True

        # Test 0% sampling (multiple tries to verify)
        sampled_count = 0
        for _ in range(10):
            state = {"sample_rate": 0.0}
            result = await create_context(state)
            if result["sampled"]:
                sampled_count += 1
        assert sampled_count == 0  # None should be sampled


class TestExtractContext:
    """Test extract_context node."""

    @pytest.mark.asyncio
    async def test_extract_context_with_traceparent(self):
        """Test context extraction from valid traceparent header."""
        state = {
            "headers": {
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
                "tracestate": "request_id=req_123,experiment_id=exp_456,sampled=1",
            }
        }

        result = await extract_context(state)

        assert result["current_trace_id"] == "0af7651916cd43dd8448eb211c80319c"
        assert result["current_parent_span_id"] == "b7ad6b7169203331"
        assert "current_span_id" in result  # New span ID generated
        assert result["request_id"] == "req_123"
        assert result["experiment_id"] == "exp_456"
        assert result["sampled"] is True

    @pytest.mark.asyncio
    async def test_extract_context_without_traceparent(self):
        """Test context extraction without traceparent (creates new)."""
        state = {"headers": {}}

        result = await extract_context(state)

        assert "current_trace_id" in result
        assert "current_span_id" in result
        assert result["current_parent_span_id"] is None  # No parent

    @pytest.mark.asyncio
    async def test_extract_context_invalid_traceparent(self):
        """Test context extraction with invalid traceparent."""
        state = {"headers": {"traceparent": "invalid-format"}}

        result = await extract_context(state)

        # Should create new context
        assert "current_trace_id" in result
        assert "current_span_id" in result

    @pytest.mark.asyncio
    async def test_extract_context_tracestate_parsing(self):
        """Test tracestate parsing."""
        state = {
            "headers": {
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
                "tracestate": "request_id=req_123,user_id=user_789,sample_rate=0.5",
            }
        }

        result = await extract_context(state)

        assert result["request_id"] == "req_123"
        assert result["user_id"] == "user_789"
        assert result["sample_rate"] == 0.5


class TestInjectContext:
    """Test inject_context node."""

    @pytest.mark.asyncio
    async def test_inject_context_success(self):
        """Test successful context injection."""
        state = {
            "current_trace_id": "0af7651916cd43dd8448eb211c80319c",
            "current_span_id": "b7ad6b7169203331",
            "sampled": True,
            "request_id": "req_123",
            "experiment_id": "exp_456",
            "sample_rate": 1.0,
        }

        result = await inject_context(state)

        headers = result["headers"]
        assert "traceparent" in headers
        assert "tracestate" in headers

        # Check traceparent format
        traceparent = headers["traceparent"]
        assert traceparent.startswith("00-")
        assert "0af7651916cd43dd8448eb211c80319c" in traceparent
        assert "b7ad6b7169203331" in traceparent
        assert traceparent.endswith("-01")  # Sampled flag

    @pytest.mark.asyncio
    async def test_inject_context_not_sampled(self):
        """Test context injection when not sampled."""
        state = {
            "current_trace_id": "0af7651916cd43dd8448eb211c80319c",
            "current_span_id": "b7ad6b7169203331",
            "sampled": False,
            "request_id": "req_123",
        }

        result = await inject_context(state)

        headers = result["headers"]
        traceparent = headers["traceparent"]
        assert traceparent.endswith("-00")  # Not sampled flag

    @pytest.mark.asyncio
    async def test_inject_context_with_baggage(self):
        """Test context injection with baggage."""
        state = {
            "current_trace_id": "0af7651916cd43dd8448eb211c80319c",
            "current_span_id": "b7ad6b7169203331",
            "sampled": True,
            "request_id": "req_123",
            "experiment_id": "exp_456",
            "user_id": "user_789",
            "sample_rate": 0.5,
        }

        result = await inject_context(state)

        headers = result["headers"]
        tracestate = headers["tracestate"]

        # Check baggage items
        assert "request_id=req_123" in tracestate
        assert "experiment_id=exp_456" in tracestate
        assert "user_id=user_789" in tracestate
        assert "sample_rate=0.5" in tracestate

    @pytest.mark.asyncio
    async def test_inject_context_missing_trace_id(self):
        """Test context injection with missing trace_id."""
        state = {
            "current_span_id": "b7ad6b7169203331",
            "sampled": True,
        }

        result = await inject_context(state)

        assert "error" in result
        assert "Missing trace_id or span_id" in result["error"]

    @pytest.mark.asyncio
    async def test_inject_context_missing_span_id(self):
        """Test context injection with missing span_id."""
        state = {
            "current_trace_id": "0af7651916cd43dd8448eb211c80319c",
            "sampled": True,
        }

        result = await inject_context(state)

        assert "error" in result
        assert "Missing trace_id or span_id" in result["error"]

    @pytest.mark.asyncio
    async def test_inject_context_empty_baggage_values(self):
        """Test that empty baggage values are excluded."""
        state = {
            "current_trace_id": "0af7651916cd43dd8448eb211c80319c",
            "current_span_id": "b7ad6b7169203331",
            "sampled": True,
            "request_id": "req_123",
            "experiment_id": "",  # Empty
            "user_id": None,  # None
        }

        result = await inject_context(state)

        headers = result["headers"]
        tracestate = headers["tracestate"]

        # Only non-empty values should be included
        assert "request_id=req_123" in tracestate
        assert "experiment_id=" not in tracestate or tracestate.count("experiment_id") == 0
