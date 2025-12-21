"""Batch Processor for Observability Spans.

Provides efficient batching of span emissions to reduce overhead:
- Memory buffer with configurable max size and timeout
- Background periodic flush
- Partial failure handling with retry logic
- Batch metrics tracking

Version: 3.0.0 (Phase 3 Production Features)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from src.agents.ml_foundation.observability_connector.models import ObservabilitySpan
    from src.mlops.opik_connector import OpikConnector
    from src.repositories.observability_span import ObservabilitySpanRepository

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Metrics for batch processing operations."""

    total_spans_processed: int = 0
    total_batches_flushed: int = 0
    total_opik_successes: int = 0
    total_opik_failures: int = 0
    total_db_successes: int = 0
    total_db_failures: int = 0
    last_flush_time: Optional[datetime] = None
    last_flush_duration_ms: float = 0.0
    last_batch_size: int = 0
    avg_batch_size: float = 0.0
    success_rate: float = 1.0

    def record_flush(
        self,
        batch_size: int,
        duration_ms: float,
        opik_success: int,
        opik_failure: int,
        db_success: int,
        db_failure: int,
    ) -> None:
        """Record metrics from a flush operation."""
        self.total_spans_processed += batch_size
        self.total_batches_flushed += 1
        self.total_opik_successes += opik_success
        self.total_opik_failures += opik_failure
        self.total_db_successes += db_success
        self.total_db_failures += db_failure
        self.last_flush_time = datetime.now(timezone.utc)
        self.last_flush_duration_ms = duration_ms
        self.last_batch_size = batch_size

        # Update averages
        if self.total_batches_flushed > 0:
            self.avg_batch_size = self.total_spans_processed / self.total_batches_flushed

        # Calculate success rate
        total_ops = (
            self.total_opik_successes
            + self.total_opik_failures
            + self.total_db_successes
            + self.total_db_failures
        )
        if total_ops > 0:
            successes = self.total_opik_successes + self.total_db_successes
            self.success_rate = successes / total_ops

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_spans_processed": self.total_spans_processed,
            "total_batches_flushed": self.total_batches_flushed,
            "total_opik_successes": self.total_opik_successes,
            "total_opik_failures": self.total_opik_failures,
            "total_db_successes": self.total_db_successes,
            "total_db_failures": self.total_db_failures,
            "last_flush_time": (
                self.last_flush_time.isoformat() if self.last_flush_time else None
            ),
            "last_flush_duration_ms": self.last_flush_duration_ms,
            "last_batch_size": self.last_batch_size,
            "avg_batch_size": self.avg_batch_size,
            "success_rate": self.success_rate,
        }


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    max_batch_size: int = 100
    max_wait_seconds: float = 5.0
    flush_on_shutdown: bool = True
    retry_failed: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class BatchProcessor:
    """Batch processor for observability spans.

    Buffers spans in memory and flushes them in batches to reduce
    per-span overhead. Supports:
    - Size-based flushing (max 100 spans)
    - Time-based flushing (every 5 seconds)
    - Background periodic flush task
    - Partial failure handling
    - Metrics tracking

    Usage:
        processor = BatchProcessor(opik_connector, span_repository)
        await processor.start()

        # Add spans (non-blocking)
        await processor.add_span(span)

        # Force flush if needed
        await processor.flush()

        # Stop processor (flushes remaining)
        await processor.stop()
    """

    def __init__(
        self,
        opik_connector: Optional["OpikConnector"] = None,
        span_repository: Optional["ObservabilitySpanRepository"] = None,
        config: Optional[BatchConfig] = None,
        on_flush_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize batch processor.

        Args:
            opik_connector: OpikConnector for SDK emission
            span_repository: Repository for database persistence
            config: Batch configuration (uses defaults if None)
            on_flush_complete: Optional callback after flush completes
        """
        self._opik_connector = opik_connector
        self._span_repository = span_repository
        self._config = config or BatchConfig()
        self._on_flush_complete = on_flush_complete

        # Buffer for pending spans
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()

        # Background task
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_flush_time = time.monotonic()

        # Metrics
        self._metrics = BatchMetrics()

        # Failed spans for retry
        self._failed_spans: List[Dict[str, Any]] = []

    @property
    def opik_connector(self) -> Optional["OpikConnector"]:
        """Get OpikConnector (lazy initialization)."""
        if self._opik_connector is None:
            try:
                from src.mlops.opik_connector import OpikConnector

                self._opik_connector = OpikConnector()
            except Exception as e:
                logger.warning(f"Failed to initialize OpikConnector: {e}")
        return self._opik_connector

    @property
    def span_repository(self) -> Optional["ObservabilitySpanRepository"]:
        """Get ObservabilitySpanRepository (lazy initialization)."""
        if self._span_repository is None:
            try:
                from src.repositories import get_supabase_client
                from src.repositories.observability_span import ObservabilitySpanRepository

                client = get_supabase_client()
                if client:
                    self._span_repository = ObservabilitySpanRepository(client=client)
            except Exception as e:
                logger.warning(f"Failed to initialize span repository: {e}")
        return self._span_repository

    @property
    def metrics(self) -> BatchMetrics:
        """Get batch processing metrics."""
        return self._metrics

    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    @property
    def is_running(self) -> bool:
        """Check if background flush task is running."""
        return self._running

    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._background_flush_loop())
        logger.info(
            f"BatchProcessor started (max_size={self._config.max_batch_size}, "
            f"max_wait={self._config.max_wait_seconds}s)"
        )

    async def stop(self) -> None:
        """Stop the background flush task and flush remaining spans."""
        if not self._running:
            return

        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining spans
        if self._config.flush_on_shutdown and self._buffer:
            await self.flush()

        logger.info("BatchProcessor stopped")

    async def add_span(self, span_event: Dict[str, Any]) -> bool:
        """Add a span event to the buffer.

        Args:
            span_event: Span event dictionary

        Returns:
            True if span was added, False if buffer is full and flush failed
        """
        async with self._buffer_lock:
            self._buffer.append(span_event)
            buffer_size = len(self._buffer)

        # Trigger flush if buffer is full
        if buffer_size >= self._config.max_batch_size:
            await self.flush()

        return True

    async def add_spans(self, span_events: List[Dict[str, Any]]) -> int:
        """Add multiple span events to the buffer.

        Args:
            span_events: List of span event dictionaries

        Returns:
            Number of spans added
        """
        async with self._buffer_lock:
            self._buffer.extend(span_events)
            buffer_size = len(self._buffer)

        # Trigger flush if buffer exceeds max size
        if buffer_size >= self._config.max_batch_size:
            await self.flush()

        return len(span_events)

    async def flush(self) -> Dict[str, Any]:
        """Flush all buffered spans to Opik and database.

        Returns:
            Flush result with success counts and errors
        """
        # Get and clear buffer atomically
        async with self._buffer_lock:
            if not self._buffer:
                return {
                    "success": True,
                    "batch_size": 0,
                    "opik_success": 0,
                    "opik_failure": 0,
                    "db_success": 0,
                    "db_failure": 0,
                    "errors": [],
                }

            spans_to_flush = self._buffer.copy()
            self._buffer.clear()

        start_time = time.monotonic()
        result = await self._emit_batch(spans_to_flush)
        duration_ms = (time.monotonic() - start_time) * 1000

        # Record metrics
        self._metrics.record_flush(
            batch_size=len(spans_to_flush),
            duration_ms=duration_ms,
            opik_success=result["opik_success"],
            opik_failure=result["opik_failure"],
            db_success=result["db_success"],
            db_failure=result["db_failure"],
        )

        self._last_flush_time = time.monotonic()

        # Handle failed spans
        if result["failed_spans"] and self._config.retry_failed:
            self._failed_spans.extend(result["failed_spans"])

        # Callback
        if self._on_flush_complete:
            try:
                self._on_flush_complete(result)
            except Exception as e:
                logger.warning(f"Flush callback failed: {e}")

        logger.debug(
            f"Flushed {len(spans_to_flush)} spans in {duration_ms:.1f}ms "
            f"(opik: {result['opik_success']}/{result['opik_failure']}, "
            f"db: {result['db_success']}/{result['db_failure']})"
        )

        return result

    async def retry_failed(self) -> Dict[str, Any]:
        """Retry previously failed spans.

        Returns:
            Retry result with success counts
        """
        if not self._failed_spans:
            return {"success": True, "retried": 0, "recovered": 0}

        spans_to_retry = self._failed_spans.copy()
        self._failed_spans.clear()

        result = await self._emit_batch(spans_to_retry)

        # Add back any still-failed spans (up to max retries)
        if result["failed_spans"]:
            for span in result["failed_spans"]:
                retries = span.get("_retry_count", 0)
                if retries < self._config.max_retries:
                    span["_retry_count"] = retries + 1
                    self._failed_spans.append(span)

        recovered = len(spans_to_retry) - len(result["failed_spans"])
        return {
            "success": True,
            "retried": len(spans_to_retry),
            "recovered": recovered,
            "still_failed": len(result["failed_spans"]),
        }

    async def _emit_batch(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Emit a batch of spans to Opik and database.

        Args:
            spans: List of span event dictionaries

        Returns:
            Emission results
        """
        opik_success = 0
        opik_failure = 0
        db_success = 0
        db_failure = 0
        errors: List[str] = []
        failed_spans: List[Dict[str, Any]] = []

        # Emit to Opik
        opik = self.opik_connector
        if opik and opik.is_enabled:
            for span in spans:
                try:
                    opik.log_metric(
                        name="span_emitted",
                        value=1.0,
                        trace_id=span.get("trace_id"),
                        metadata={
                            "span_id": span.get("span_id"),
                            "agent_name": span.get("agent_name"),
                            "operation": span.get("operation"),
                            "status": span.get("status"),
                            "duration_ms": span.get("duration_ms"),
                            "batch_processed": True,
                        },
                    )
                    opik_success += 1
                except Exception as e:
                    opik_failure += 1
                    errors.append(f"Opik emit failed for {span.get('span_id')}: {e}")

        # Emit to database (batch insert)
        repository = self.span_repository
        if repository:
            try:
                # Convert to ObservabilitySpan models
                from src.agents.ml_foundation.observability_connector.models import (
                    AgentNameEnum,
                    AgentTierEnum,
                    ObservabilitySpan,
                    SpanStatusEnum,
                )

                span_models = []
                for span_event in spans:
                    try:
                        span_model = self._event_to_span_model(
                            span_event,
                            AgentNameEnum,
                            AgentTierEnum,
                            ObservabilitySpan,
                            SpanStatusEnum,
                        )
                        span_models.append(span_model)
                    except Exception as e:
                        db_failure += 1
                        failed_spans.append(span_event)
                        errors.append(
                            f"Model conversion failed for {span_event.get('span_id')}: {e}"
                        )

                if span_models:
                    result = await repository.insert_spans_batch(span_models)
                    db_success = result.get("inserted_count", 0)
                    db_failure += result.get("failed_count", 0)

            except Exception as e:
                db_failure += len(spans)
                failed_spans.extend(spans)
                errors.append(f"Batch DB insert failed: {e}")

        return {
            "success": (opik_failure == 0 and db_failure == 0),
            "batch_size": len(spans),
            "opik_success": opik_success,
            "opik_failure": opik_failure,
            "db_success": db_success,
            "db_failure": db_failure,
            "errors": errors,
            "failed_spans": failed_spans,
        }

    def _event_to_span_model(
        self,
        event: Dict[str, Any],
        AgentNameEnum,
        AgentTierEnum,
        ObservabilitySpan,
        SpanStatusEnum,
    ) -> "ObservabilitySpan":
        """Convert span event dict to ObservabilitySpan model."""
        from datetime import datetime, timezone

        # Parse agent name
        agent_name = event.get("agent_name", "")
        try:
            agent_name_enum = AgentNameEnum(agent_name) if agent_name else AgentNameEnum.ORCHESTRATOR
        except ValueError:
            agent_name_enum = AgentNameEnum.ORCHESTRATOR

        # Parse agent tier
        tier = event.get("agent_tier", 0)
        tier_mapping = {
            0: "ml_foundation",
            1: "coordination",
            2: "causal_analytics",
            3: "monitoring",
            4: "ml_predictions",
            5: "self_improvement",
        }
        tier_str = tier_mapping.get(tier, "coordination") if isinstance(tier, int) else str(tier)
        try:
            agent_tier_enum = AgentTierEnum(tier_str)
        except ValueError:
            agent_tier_enum = AgentTierEnum.COORDINATION

        # Parse status
        status_str = event.get("status", "success").lower()
        status_map = {
            "ok": SpanStatusEnum.SUCCESS,
            "success": SpanStatusEnum.SUCCESS,
            "completed": SpanStatusEnum.SUCCESS,
            "error": SpanStatusEnum.ERROR,
            "timeout": SpanStatusEnum.TIMEOUT,
        }
        status = status_map.get(status_str, SpanStatusEnum.SUCCESS)

        # Parse timestamps
        started_at = event.get("started_at")
        if isinstance(started_at, str):
            try:
                started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            except ValueError:
                started_at = datetime.now(timezone.utc)
        elif not isinstance(started_at, datetime):
            started_at = datetime.now(timezone.utc)

        ended_at = event.get("completed_at") or event.get("ended_at")
        if isinstance(ended_at, str):
            try:
                ended_at = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            except ValueError:
                ended_at = None

        return ObservabilitySpan(
            trace_id=event.get("trace_id", ""),
            span_id=event.get("span_id", ""),
            parent_span_id=event.get("parent_span_id"),
            agent_name=agent_name_enum,
            agent_tier=agent_tier_enum,
            operation_type=event.get("operation", "unknown"),
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=event.get("duration_ms"),
            model_name=event.get("model_used"),
            input_tokens=event.get("input_tokens"),
            output_tokens=event.get("output_tokens"),
            total_tokens=event.get("tokens_used"),
            status=status,
            error_type=event.get("error_type"),
            error_message=event.get("error"),
            fallback_used=event.get("metadata", {}).get("fallback_used", False),
            attributes=event.get("metadata", {}),
        )

    async def _background_flush_loop(self) -> None:
        """Background task that flushes buffer periodically."""
        while self._running:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms

                # Check if max wait time exceeded
                time_since_flush = time.monotonic() - self._last_flush_time
                if time_since_flush >= self._config.max_wait_seconds and self._buffer:
                    await self.flush()

                # Retry failed spans periodically
                if self._failed_spans and time_since_flush >= self._config.retry_delay_seconds:
                    await self.retry_failed()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background flush error: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    def get_status(self) -> Dict[str, Any]:
        """Get current processor status.

        Returns:
            Status dictionary with buffer info and metrics
        """
        return {
            "running": self._running,
            "buffer_size": len(self._buffer),
            "failed_queue_size": len(self._failed_spans),
            "config": {
                "max_batch_size": self._config.max_batch_size,
                "max_wait_seconds": self._config.max_wait_seconds,
                "retry_failed": self._config.retry_failed,
                "max_retries": self._config.max_retries,
            },
            "metrics": self._metrics.to_dict(),
        }


# Module-level singleton
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor(
    opik_connector: Optional["OpikConnector"] = None,
    span_repository: Optional["ObservabilitySpanRepository"] = None,
    config: Optional[BatchConfig] = None,
) -> BatchProcessor:
    """Get or create the BatchProcessor singleton.

    Args:
        opik_connector: Optional OpikConnector (uses lazy init if None)
        span_repository: Optional repository (uses lazy init if None)
        config: Optional batch configuration

    Returns:
        BatchProcessor singleton instance
    """
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(
            opik_connector=opik_connector,
            span_repository=span_repository,
            config=config,
        )
    return _batch_processor


async def reset_batch_processor() -> None:
    """Reset the batch processor singleton (for testing)."""
    global _batch_processor
    if _batch_processor:
        await _batch_processor.stop()
        _batch_processor = None
