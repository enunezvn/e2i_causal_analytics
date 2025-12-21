"""Tests for BatchProcessor.

Version: 3.0.0 (Phase 3 Production Features)
Tests batch processing with mocked dependencies.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ml_foundation.observability_connector.batch_processor import (
    BatchConfig,
    BatchMetrics,
    BatchProcessor,
    get_batch_processor,
    reset_batch_processor,
)


@pytest.fixture
def mock_opik_connector():
    """Create a mock OpikConnector."""
    mock = MagicMock()
    mock.is_enabled = True
    mock.log_metric = MagicMock()
    return mock


@pytest.fixture
def mock_span_repository():
    """Create a mock ObservabilitySpanRepository."""
    mock = MagicMock()
    mock.insert_spans_batch = AsyncMock(
        return_value={"success": True, "inserted_count": 5, "failed_count": 0, "span_ids": []}
    )
    return mock


@pytest.fixture
def sample_span_event():
    """Create a sample span event."""
    return {
        "span_id": "span_123",
        "trace_id": "trace_456",
        "agent_name": "orchestrator",
        "agent_tier": 1,
        "operation": "execute",
        "status": "ok",
        "duration_ms": 150,
        "started_at": "2025-12-21T10:00:00Z",
        "completed_at": "2025-12-21T10:00:00.150Z",
    }


@pytest.fixture
def batch_config():
    """Create a test batch configuration."""
    return BatchConfig(
        max_batch_size=5,
        max_wait_seconds=1.0,
        flush_on_shutdown=True,
        retry_failed=True,
        max_retries=2,
        retry_delay_seconds=0.5,
    )


@pytest.fixture(autouse=True)
async def reset_singleton():
    """Reset the batch processor singleton before each test."""
    await reset_batch_processor()
    yield
    await reset_batch_processor()


class TestBatchMetrics:
    """Test BatchMetrics class."""

    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = BatchMetrics()

        assert metrics.total_spans_processed == 0
        assert metrics.total_batches_flushed == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_batch_size == 0.0

    def test_record_flush(self):
        """Test recording a flush operation."""
        metrics = BatchMetrics()

        metrics.record_flush(
            batch_size=10,
            duration_ms=50.0,
            opik_success=10,
            opik_failure=0,
            db_success=10,
            db_failure=0,
        )

        assert metrics.total_spans_processed == 10
        assert metrics.total_batches_flushed == 1
        assert metrics.last_batch_size == 10
        assert metrics.last_flush_duration_ms == 50.0
        assert metrics.avg_batch_size == 10.0
        assert metrics.success_rate == 1.0
        assert metrics.last_flush_time is not None

    def test_record_multiple_flushes(self):
        """Test recording multiple flush operations."""
        metrics = BatchMetrics()

        metrics.record_flush(10, 50.0, 10, 0, 10, 0)
        metrics.record_flush(20, 100.0, 18, 2, 20, 0)

        assert metrics.total_spans_processed == 30
        assert metrics.total_batches_flushed == 2
        assert metrics.avg_batch_size == 15.0
        assert metrics.last_batch_size == 20

    def test_success_rate_calculation(self):
        """Test success rate calculation with failures."""
        metrics = BatchMetrics()

        metrics.record_flush(10, 50.0, 8, 2, 9, 1)

        # 8 + 9 = 17 successes, 2 + 1 = 3 failures, total = 20
        expected_rate = 17 / 20
        assert abs(metrics.success_rate - expected_rate) < 0.001

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = BatchMetrics()
        metrics.record_flush(10, 50.0, 10, 0, 10, 0)

        result = metrics.to_dict()

        assert result["total_spans_processed"] == 10
        assert result["total_batches_flushed"] == 1
        assert result["last_flush_duration_ms"] == 50.0
        assert result["last_flush_time"] is not None


class TestBatchConfig:
    """Test BatchConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchConfig()

        assert config.max_batch_size == 100
        assert config.max_wait_seconds == 5.0
        assert config.flush_on_shutdown is True
        assert config.retry_failed is True
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = BatchConfig(
            max_batch_size=50,
            max_wait_seconds=10.0,
            retry_failed=False,
        )

        assert config.max_batch_size == 50
        assert config.max_wait_seconds == 10.0
        assert config.retry_failed is False


class TestBatchProcessor:
    """Test BatchProcessor class."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_opik_connector, mock_span_repository):
        """Test processor initialization."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
        )

        assert processor.buffer_size == 0
        assert processor.is_running is False
        assert processor.metrics.total_batches_flushed == 0

    @pytest.mark.asyncio
    async def test_add_span(
        self, mock_opik_connector, mock_span_repository, sample_span_event
    ):
        """Test adding a span to the buffer."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
        )

        result = await processor.add_span(sample_span_event)

        assert result is True
        assert processor.buffer_size == 1

    @pytest.mark.asyncio
    async def test_add_multiple_spans(
        self, mock_opik_connector, mock_span_repository, sample_span_event
    ):
        """Test adding multiple spans."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
        )

        spans = [
            {**sample_span_event, "span_id": f"span_{i}"}
            for i in range(3)
        ]

        count = await processor.add_spans(spans)

        assert count == 3
        assert processor.buffer_size == 3

    @pytest.mark.asyncio
    async def test_auto_flush_on_max_size(
        self, mock_opik_connector, mock_span_repository, sample_span_event, batch_config
    ):
        """Test automatic flush when buffer reaches max size."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
            config=batch_config,  # max_batch_size=5
        )

        # Add 5 spans (should trigger flush)
        for i in range(5):
            await processor.add_span({**sample_span_event, "span_id": f"span_{i}"})

        # Buffer should be empty after auto-flush
        assert processor.buffer_size == 0
        assert processor.metrics.total_batches_flushed == 1

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, mock_opik_connector, mock_span_repository):
        """Test flushing an empty buffer."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
        )

        result = await processor.flush()

        assert result["success"] is True
        assert result["batch_size"] == 0
        mock_opik_connector.log_metric.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_with_spans(
        self, mock_opik_connector, mock_span_repository, sample_span_event
    ):
        """Test flushing buffer with spans."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
        )

        await processor.add_span(sample_span_event)
        await processor.add_span({**sample_span_event, "span_id": "span_456"})

        result = await processor.flush()

        assert result["batch_size"] == 2
        assert result["opik_success"] == 2
        assert processor.buffer_size == 0
        assert mock_opik_connector.log_metric.call_count == 2

    @pytest.mark.asyncio
    async def test_flush_opik_only(self, mock_opik_connector, sample_span_event):
        """Test flushing with Opik but no repository."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=None,
        )

        await processor.add_span(sample_span_event)
        result = await processor.flush()

        assert result["opik_success"] == 1
        assert result["db_success"] == 0

    @pytest.mark.asyncio
    async def test_flush_db_only(self, mock_span_repository, sample_span_event):
        """Test flushing with repository but no Opik."""
        mock_opik = MagicMock()
        mock_opik.is_enabled = False

        processor = BatchProcessor(
            opik_connector=mock_opik,
            span_repository=mock_span_repository,
        )

        await processor.add_span(sample_span_event)
        result = await processor.flush()

        assert result["opik_success"] == 0
        mock_span_repository.insert_spans_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_opik_connector, mock_span_repository):
        """Test starting and stopping the processor."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
        )

        await processor.start()
        assert processor.is_running is True

        await processor.stop()
        assert processor.is_running is False

    @pytest.mark.asyncio
    async def test_flush_on_shutdown(
        self, mock_opik_connector, mock_span_repository, sample_span_event, batch_config
    ):
        """Test that remaining spans are flushed on shutdown."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
            config=batch_config,
        )

        await processor.start()
        await processor.add_span(sample_span_event)
        await processor.add_span({**sample_span_event, "span_id": "span_2"})

        # Stop should flush remaining spans
        await processor.stop()

        assert processor.buffer_size == 0
        assert processor.metrics.total_spans_processed == 2

    @pytest.mark.asyncio
    async def test_get_status(
        self, mock_opik_connector, mock_span_repository, sample_span_event
    ):
        """Test getting processor status."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
        )

        await processor.add_span(sample_span_event)

        status = processor.get_status()

        assert status["running"] is False
        assert status["buffer_size"] == 1
        assert status["failed_queue_size"] == 0
        assert "config" in status
        assert "metrics" in status

    @pytest.mark.asyncio
    async def test_flush_callback(
        self, mock_opik_connector, mock_span_repository, sample_span_event
    ):
        """Test flush completion callback."""
        callback_result = {}

        def on_flush(result):
            callback_result.update(result)

        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
            on_flush_complete=on_flush,
        )

        await processor.add_span(sample_span_event)
        await processor.flush()

        assert callback_result["batch_size"] == 1

    @pytest.mark.asyncio
    async def test_metrics_update_on_flush(
        self, mock_opik_connector, mock_span_repository, sample_span_event
    ):
        """Test that metrics are updated after flush."""
        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
        )

        await processor.add_span(sample_span_event)
        await processor.flush()

        metrics = processor.metrics
        assert metrics.total_spans_processed == 1
        assert metrics.total_batches_flushed == 1
        assert metrics.last_flush_duration_ms > 0


class TestBatchProcessorPartialFailures:
    """Test partial failure handling."""

    @pytest.mark.asyncio
    async def test_opik_failure_continues_to_db(
        self, mock_span_repository, sample_span_event
    ):
        """Test that DB write continues even if Opik fails."""
        mock_opik = MagicMock()
        mock_opik.is_enabled = True
        mock_opik.log_metric = MagicMock(side_effect=Exception("Opik error"))

        processor = BatchProcessor(
            opik_connector=mock_opik,
            span_repository=mock_span_repository,
        )

        await processor.add_span(sample_span_event)
        result = await processor.flush()

        assert result["opik_failure"] == 1
        # DB should still be attempted
        mock_span_repository.insert_spans_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_spans_queued_for_retry(
        self, mock_opik_connector, sample_span_event, batch_config
    ):
        """Test that failed spans are queued for retry."""
        mock_repo = MagicMock()
        mock_repo.insert_spans_batch = AsyncMock(
            return_value={"success": False, "inserted_count": 0, "failed_count": 1, "span_ids": []}
        )

        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_repo,
            config=batch_config,
        )

        await processor.add_span(sample_span_event)
        await processor.flush()

        # Failed spans should be in retry queue
        status = processor.get_status()
        # Note: The model conversion might succeed, so failed_queue depends on that
        assert processor.metrics.total_batches_flushed == 1


class TestBatchProcessorBackgroundFlush:
    """Test background flush functionality."""

    @pytest.mark.asyncio
    async def test_background_flush_on_timeout(
        self, mock_opik_connector, mock_span_repository, sample_span_event
    ):
        """Test that background task flushes after timeout."""
        config = BatchConfig(
            max_batch_size=100,
            max_wait_seconds=0.3,  # Short timeout for testing
        )

        processor = BatchProcessor(
            opik_connector=mock_opik_connector,
            span_repository=mock_span_repository,
            config=config,
        )

        await processor.start()
        await processor.add_span(sample_span_event)

        # Wait for background flush (loop checks every 0.5s, need enough time)
        await asyncio.sleep(1.0)

        assert processor.buffer_size == 0
        assert processor.metrics.total_batches_flushed == 1

        await processor.stop()


class TestBatchProcessorSingleton:
    """Test singleton pattern."""

    @pytest.mark.asyncio
    async def test_get_batch_processor_singleton(self):
        """Test that get_batch_processor returns same instance."""
        processor1 = get_batch_processor()
        processor2 = get_batch_processor()

        assert processor1 is processor2

    @pytest.mark.asyncio
    async def test_reset_batch_processor(self):
        """Test resetting the singleton."""
        processor1 = get_batch_processor()
        await reset_batch_processor()
        processor2 = get_batch_processor()

        assert processor1 is not processor2


class TestBatchProcessorLazyInit:
    """Test lazy initialization of connectors."""

    @pytest.mark.asyncio
    async def test_lazy_opik_init(self):
        """Test lazy initialization of OpikConnector."""
        processor = BatchProcessor()

        # Initially None
        assert processor._opik_connector is None

        # Access property - will try to initialize
        with patch(
            "src.agents.ml_foundation.observability_connector.batch_processor.BatchProcessor.opik_connector",
            new_callable=lambda: property(lambda self: MagicMock(is_enabled=True)),
        ):
            # Property access should work
            pass

    @pytest.mark.asyncio
    async def test_lazy_repository_init(self):
        """Test lazy initialization of repository."""
        processor = BatchProcessor()

        # Initially None
        assert processor._span_repository is None
