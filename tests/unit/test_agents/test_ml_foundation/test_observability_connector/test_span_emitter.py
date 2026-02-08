"""Tests for span_emitter node (emit_spans).

Version: 2.0.0 (Phase 2 Integration)
Tests use mocked OpikConnector and ObservabilitySpanRepository.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.ml_foundation.observability_connector.nodes.span_emitter import (
    emit_spans,
)


@pytest.fixture
def mock_opik_connector():
    """Create a mock OpikConnector."""
    mock = MagicMock()
    mock.is_enabled = True
    mock.config.project_name = "e2i-causal-analytics"
    mock.config.workspace = "default"
    mock.log_metric = MagicMock()
    return mock


@pytest.fixture
def mock_span_repository():
    """Create a mock ObservabilitySpanRepository."""
    mock = MagicMock()
    mock.insert_span = AsyncMock(return_value=MagicMock(span_id="test-span"))
    return mock


@pytest.fixture(autouse=True)
def reset_module_singletons():
    """Reset module-level singletons before each test."""
    import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

    module._opik_connector = None
    module._span_repository = None
    yield
    module._opik_connector = None
    module._span_repository = None


class TestEmitSpans:
    """Test emit_spans node."""

    @pytest.mark.asyncio
    async def test_emit_spans_success(self, mock_opik_connector, mock_span_repository):
        """Test successful span emission with mocked dependencies."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        # Inject mocks
        module._opik_connector = mock_opik_connector
        module._span_repository = mock_span_repository

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

        # Verify mock interactions
        mock_opik_connector.log_metric.assert_called_once()
        mock_span_repository.insert_span.assert_called_once()

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
    async def test_emit_spans_multiple_events(self, mock_opik_connector, mock_span_repository):
        """Test span emission with multiple events."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        module._opik_connector = mock_opik_connector
        module._span_repository = mock_span_repository

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
        assert result["db_write_count"] == 3  # 3 DB writes
        assert mock_span_repository.insert_span.call_count == 3

    @pytest.mark.asyncio
    async def test_emit_spans_with_error_status(self, mock_opik_connector, mock_span_repository):
        """Test span emission with error status."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        module._opik_connector = mock_opik_connector
        module._span_repository = mock_span_repository

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
        assert result["db_write_count"] == 1

    @pytest.mark.asyncio
    async def test_emit_spans_with_llm_metrics(self, mock_opik_connector, mock_span_repository):
        """Test span emission with LLM token metrics."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        module._opik_connector = mock_opik_connector
        module._span_repository = mock_span_repository

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
        assert result["db_write_count"] == 1

    @pytest.mark.asyncio
    async def test_emit_spans_with_metadata(self, mock_opik_connector, mock_span_repository):
        """Test span emission with custom metadata."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        module._opik_connector = mock_opik_connector
        module._span_repository = mock_span_repository

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
    async def test_emit_spans_generates_ids_if_missing(
        self, mock_opik_connector, mock_span_repository
    ):
        """Test that span/trace IDs are generated if missing."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        module._opik_connector = mock_opik_connector
        module._span_repository = mock_span_repository

        state = {
            "events_to_log": [
                {
                    # Missing span_id and trace_id
                    "agent_name": "orchestrator",  # Use valid agent name
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
    async def test_emit_spans_opik_url(self, mock_opik_connector, mock_span_repository):
        """Test that Opik URL is included."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        module._opik_connector = mock_opik_connector
        module._span_repository = mock_span_repository

        state = {
            "events_to_log": [
                {
                    "span_id": "span_1",
                    "trace_id": "trace_1",
                    "agent_name": "orchestrator",
                    "operation": "execute",
                    "status": "ok",
                }
            ]
        }

        result = await emit_spans(state)

        assert result["opik_url"] == "https://www.comet.com/opik"


class TestEmitSpansGracefulDegradation:
    """Test graceful degradation when connectors are unavailable."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset module-level singletons."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        module._opik_connector = None
        module._span_repository = None
        yield
        module._opik_connector = None
        module._span_repository = None

    @pytest.mark.asyncio
    async def test_emit_spans_without_opik(self, mock_span_repository):
        """Test emission works without Opik connector."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        # No Opik, only repository
        module._opik_connector = None
        module._span_repository = mock_span_repository

        state = {
            "events_to_log": [
                {
                    "span_id": "span_1",
                    "trace_id": "trace_1",
                    "agent_name": "orchestrator",
                    "operation": "execute",
                    "status": "ok",
                }
            ]
        }

        result = await emit_spans(state)

        # Should still succeed with DB-only
        assert result["emission_successful"] is True
        assert result["db_write_count"] == 1

    @pytest.mark.asyncio
    async def test_emit_spans_without_repository(self, mock_opik_connector):
        """Test emission works without repository."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module
        from unittest.mock import patch

        # Only Opik, no repository
        module._opik_connector = mock_opik_connector
        module._span_repository = None

        state = {
            "events_to_log": [
                {
                    "span_id": "span_1",
                    "trace_id": "trace_1",
                    "agent_name": "orchestrator",
                    "operation": "execute",
                    "status": "ok",
                }
            ]
        }

        with patch.object(module, 'get_span_repository', return_value=None):
            result = await emit_spans(state)

        # Should still succeed with Opik-only
        assert result["emission_successful"] is True
        assert result["db_write_count"] == 0
        assert result["db_writes_successful"] is True  # None repository = success

    @pytest.mark.asyncio
    async def test_emit_spans_without_both_connectors(self):
        """Test emission handles missing connectors gracefully."""
        import src.agents.ml_foundation.observability_connector.nodes.span_emitter as module

        # Neither connector available
        module._opik_connector = None
        module._span_repository = None

        state = {
            "events_to_log": [
                {
                    "span_id": "span_1",
                    "trace_id": "trace_1",
                    "agent_name": "orchestrator",
                    "operation": "execute",
                    "status": "ok",
                }
            ]
        }

        result = await emit_spans(state)

        # Should not raise, but emission may fail
        assert result["events_logged"] == 1
        assert result["db_write_count"] == 0
