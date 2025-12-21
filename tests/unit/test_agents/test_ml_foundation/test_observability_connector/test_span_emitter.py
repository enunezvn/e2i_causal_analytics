"""Tests for span_emitter node (emit_spans)."""

import pytest

from src.agents.ml_foundation.observability_connector.nodes.span_emitter import (
    emit_spans,
)


class TestEmitSpans:
    """Test emit_spans node."""

    @pytest.mark.asyncio
    async def test_emit_spans_success(self):
        """Test successful span emission."""
        state = {
            "events_to_log": [
                {
                    "span_id": "span_123",
                    "trace_id": "trace_456",
                    "agent_name": "scope_definer",
                    "operation": "execute",
                    "started_at": "2025-12-18T10:00:00Z",
                    "completed_at": "2025-12-18T10:00:02Z",
                    "duration_ms": 2000,
                    "status": "ok",
                }
            ]
        }

        result = await emit_spans(state)

        assert result["emission_successful"] is True
        assert result["events_logged"] == 1
        assert "span_123" in result["span_ids_logged"]
        assert "trace_456" in result["trace_ids_logged"]
        assert result["opik_project"] == "e2i-causal-analytics"
        assert result["opik_workspace"] == "default"
        assert result["db_writes_successful"] is True
        assert result["db_write_count"] == 1

    @pytest.mark.asyncio
    async def test_emit_spans_empty_events(self):
        """Test span emission with no events."""
        state = {"events_to_log": []}

        result = await emit_spans(state)

        assert result["emission_successful"] is True
        assert result["events_logged"] == 0
        assert result["span_ids_logged"] == []
        assert result["trace_ids_logged"] == []
        assert result["db_write_count"] == 0

    @pytest.mark.asyncio
    async def test_emit_spans_multiple_events(self):
        """Test span emission with multiple events."""
        state = {
            "events_to_log": [
                {
                    "span_id": "span_1",
                    "trace_id": "trace_1",
                    "agent_name": "scope_definer",
                    "operation": "execute",
                    "status": "ok",
                },
                {
                    "span_id": "span_2",
                    "trace_id": "trace_1",  # Same trace
                    "agent_name": "data_preparer",
                    "operation": "execute",
                    "status": "ok",
                },
                {
                    "span_id": "span_3",
                    "trace_id": "trace_2",  # Different trace
                    "agent_name": "model_selector",
                    "operation": "execute",
                    "status": "ok",
                },
            ]
        }

        result = await emit_spans(state)

        assert result["emission_successful"] is True
        assert result["events_logged"] == 3
        assert len(result["span_ids_logged"]) == 3
        assert len(result["trace_ids_logged"]) == 2  # 2 unique traces

    @pytest.mark.asyncio
    async def test_emit_spans_with_error_status(self):
        """Test span emission with error status."""
        state = {
            "events_to_log": [
                {
                    "span_id": "span_error",
                    "trace_id": "trace_error",
                    "agent_name": "model_trainer",
                    "operation": "execute",
                    "status": "error",
                    "error": "Training failed",
                    "error_type": "TrainingError",
                }
            ]
        }

        result = await emit_spans(state)

        assert result["emission_successful"] is True
        assert result["events_logged"] == 1

    @pytest.mark.asyncio
    async def test_emit_spans_with_llm_metrics(self):
        """Test span emission with LLM token metrics."""
        state = {
            "events_to_log": [
                {
                    "span_id": "span_llm",
                    "trace_id": "trace_llm",
                    "agent_name": "feature_analyzer",
                    "operation": "interpret",
                    "status": "ok",
                    "model_used": "claude-sonnet-4-20250514",
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "tokens_used": 1500,
                }
            ]
        }

        result = await emit_spans(state)

        assert result["emission_successful"] is True
        assert result["events_logged"] == 1

    @pytest.mark.asyncio
    async def test_emit_spans_with_metadata(self):
        """Test span emission with custom metadata."""
        state = {
            "events_to_log": [
                {
                    "span_id": "span_meta",
                    "trace_id": "trace_meta",
                    "agent_name": "scope_definer",
                    "operation": "execute",
                    "status": "ok",
                    "metadata": {
                        "experiment_id": "exp_123",
                        "problem_type": "classification",
                        "custom_field": "value",
                    },
                }
            ]
        }

        result = await emit_spans(state)

        assert result["emission_successful"] is True
        assert result["events_logged"] == 1

    @pytest.mark.asyncio
    async def test_emit_spans_generates_ids_if_missing(self):
        """Test that span/trace IDs are generated if missing."""
        state = {
            "events_to_log": [
                {
                    # Missing span_id and trace_id
                    "agent_name": "test_agent",
                    "operation": "execute",
                    "status": "ok",
                }
            ]
        }

        result = await emit_spans(state)

        assert result["emission_successful"] is True
        assert result["events_logged"] == 1
        assert len(result["span_ids_logged"]) == 1
        assert len(result["trace_ids_logged"]) == 1

    @pytest.mark.asyncio
    async def test_emit_spans_opik_url(self):
        """Test that Opik URL is included."""
        state = {
            "events_to_log": [
                {
                    "span_id": "span_1",
                    "trace_id": "trace_1",
                    "agent_name": "test_agent",
                    "operation": "execute",
                    "status": "ok",
                }
            ]
        }

        result = await emit_spans(state)

        assert result["opik_url"] == "https://www.comet.com/opik"
