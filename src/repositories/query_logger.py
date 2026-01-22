"""
Database Query Logging and Monitoring.

G13 from observability audit remediation plan:
- Query execution time tracking
- Slow query detection and alerting
- Connection pool metrics
- Prometheus metrics integration

Version: 1.0.0
"""

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# =============================================================================
# Prometheus Metrics Integration
# =============================================================================

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, query metrics disabled")


@dataclass
class QueryMetrics:
    """Container for database query metrics."""

    registry: Optional[Any] = None
    query_duration: Optional[Any] = None
    query_total: Optional[Any] = None
    query_errors: Optional[Any] = None
    slow_queries: Optional[Any] = None
    active_connections: Optional[Any] = None
    connection_pool_size: Optional[Any] = None
    connection_wait_time: Optional[Any] = None

    _initialized: bool = False

    def initialize(self, registry: Optional[Any] = None) -> None:
        """Initialize query monitoring metrics."""
        if self._initialized or not PROMETHEUS_AVAILABLE:
            return

        self.registry = registry or CollectorRegistry()

        # Query duration histogram with operation type labels
        self.query_duration = Histogram(
            "db_query_duration_seconds",
            "Database query execution time in seconds",
            ["operation", "table", "status"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Query count by operation type
        self.query_total = Counter(
            "db_query_total",
            "Total number of database queries",
            ["operation", "table"],
            registry=self.registry,
        )

        # Query errors
        self.query_errors = Counter(
            "db_query_errors_total",
            "Total number of database query errors",
            ["operation", "table", "error_type"],
            registry=self.registry,
        )

        # Slow query counter
        self.slow_queries = Counter(
            "db_slow_queries_total",
            "Total number of slow queries detected",
            ["operation", "table"],
            registry=self.registry,
        )

        # Connection pool metrics
        self.active_connections = Gauge(
            "db_active_connections",
            "Number of active database connections",
            registry=self.registry,
        )

        self.connection_pool_size = Gauge(
            "db_connection_pool_size",
            "Total size of connection pool",
            registry=self.registry,
        )

        self.connection_wait_time = Histogram(
            "db_connection_wait_seconds",
            "Time spent waiting for a connection from the pool",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry,
        )

        self._initialized = True
        logger.info("Database query metrics initialized")


# Global metrics instance
query_metrics = QueryMetrics()


# =============================================================================
# Slow Query Detection
# =============================================================================


@dataclass
class SlowQueryRecord:
    """Record of a slow query for analysis."""

    operation: str
    table: str
    duration_ms: float
    timestamp: datetime
    query_params: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


@dataclass
class SlowQueryConfig:
    """Configuration for slow query detection."""

    # Default threshold in seconds
    default_threshold_sec: float = 1.0

    # Per-operation thresholds (more lenient for complex operations)
    operation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "select": 0.5,      # 500ms for selects
        "insert": 0.25,     # 250ms for inserts
        "update": 0.25,     # 250ms for updates
        "delete": 0.25,     # 250ms for deletes
        "upsert": 0.5,      # 500ms for upserts
        "rpc": 2.0,         # 2s for stored procedures
    })

    # Per-table thresholds (for known complex tables)
    table_thresholds: Dict[str, float] = field(default_factory=dict)

    # Maximum slow queries to retain in memory
    max_retained_queries: int = 100

    # Whether to log slow queries
    log_slow_queries: bool = True

    # Whether to include query parameters in logs (may expose sensitive data)
    include_params_in_logs: bool = False

    def get_threshold(self, operation: str, table: str) -> float:
        """Get threshold for a specific operation and table."""
        # Table-specific threshold takes precedence
        if table in self.table_thresholds:
            return self.table_thresholds[table]

        # Then operation-specific threshold
        if operation in self.operation_thresholds:
            return self.operation_thresholds[operation]

        # Default threshold
        return self.default_threshold_sec


class SlowQueryDetector:
    """
    Detects and tracks slow database queries.

    Features:
    - Configurable thresholds per operation/table
    - In-memory retention of recent slow queries
    - Prometheus metrics integration
    - Optional alerting callbacks
    """

    def __init__(
        self,
        config: Optional[SlowQueryConfig] = None,
        alert_callback: Optional[Callable[[SlowQueryRecord], None]] = None,
    ):
        """
        Initialize slow query detector.

        Args:
            config: Slow query configuration
            alert_callback: Optional callback for slow query alerts
        """
        self.config = config or SlowQueryConfig()
        self.alert_callback = alert_callback
        self._slow_queries: List[SlowQueryRecord] = []
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

    def check_query(
        self,
        operation: str,
        table: str,
        duration_sec: float,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[SlowQueryRecord]:
        """
        Check if a query is slow and record it if so.

        Args:
            operation: Query operation type (select, insert, etc.)
            table: Table name
            duration_sec: Query duration in seconds
            query_params: Optional query parameters

        Returns:
            SlowQueryRecord if query is slow, None otherwise
        """
        threshold = self.config.get_threshold(operation, table)

        if duration_sec < threshold:
            return None

        # Create slow query record
        record = SlowQueryRecord(
            operation=operation,
            table=table,
            duration_ms=duration_sec * 1000,
            timestamp=datetime.now(timezone.utc),
            query_params=query_params if self.config.include_params_in_logs else None,
        )

        # Store in memory (with limit)
        self._slow_queries.append(record)
        if len(self._slow_queries) > self.config.max_retained_queries:
            self._slow_queries = self._slow_queries[-self.config.max_retained_queries:]

        # Update metrics
        if query_metrics._initialized:
            query_metrics.slow_queries.labels(
                operation=operation,
                table=table,
            ).inc()

        # Log if configured
        if self.config.log_slow_queries:
            logger.warning(
                f"Slow query detected: {operation} on {table} "
                f"took {record.duration_ms:.2f}ms (threshold: {threshold * 1000:.0f}ms)"
            )

        # Fire alert callback
        if self.alert_callback:
            try:
                self.alert_callback(record)
            except Exception as e:
                logger.error(f"Slow query alert callback failed: {e}")

        return record

    def get_recent_slow_queries(
        self,
        limit: int = 10,
        operation: Optional[str] = None,
        table: Optional[str] = None,
    ) -> List[SlowQueryRecord]:
        """
        Get recent slow queries with optional filtering.

        Args:
            limit: Maximum number of queries to return
            operation: Filter by operation type
            table: Filter by table name

        Returns:
            List of slow query records
        """
        queries = self._slow_queries

        if operation:
            queries = [q for q in queries if q.operation == operation]
        if table:
            queries = [q for q in queries if q.table == table]

        return queries[-limit:]

    def get_slow_query_stats(self) -> Dict[str, Any]:
        """
        Get statistics about slow queries.

        Returns:
            Dictionary with slow query statistics
        """
        if not self._slow_queries:
            return {
                "total_slow_queries": 0,
                "by_operation": {},
                "by_table": {},
                "avg_duration_ms": 0,
                "max_duration_ms": 0,
            }

        by_operation: Dict[str, int] = {}
        by_table: Dict[str, int] = {}
        total_duration = 0.0
        max_duration = 0.0

        for query in self._slow_queries:
            by_operation[query.operation] = by_operation.get(query.operation, 0) + 1
            by_table[query.table] = by_table.get(query.table, 0) + 1
            total_duration += query.duration_ms
            max_duration = max(max_duration, query.duration_ms)

        return {
            "total_slow_queries": len(self._slow_queries),
            "by_operation": by_operation,
            "by_table": by_table,
            "avg_duration_ms": total_duration / len(self._slow_queries),
            "max_duration_ms": max_duration,
        }

    def clear(self) -> None:
        """Clear stored slow queries."""
        self._slow_queries.clear()


# Global slow query detector
slow_query_detector = SlowQueryDetector()


# =============================================================================
# Query Logging Wrapper
# =============================================================================


class QueryLogger:
    """
    Wrapper for database clients that adds query logging and metrics.

    This class wraps database operations to provide:
    - Execution time tracking
    - Slow query detection
    - Prometheus metrics
    - Structured logging

    Usage:
        from src.repositories.query_logger import QueryLogger, query_logger

        # Wrap a Supabase query
        result = await query_logger.execute(
            operation="select",
            table="kpi_values",
            func=lambda: supabase.table("kpi_values").select("*").execute()
        )
    """

    def __init__(
        self,
        slow_query_detector: Optional[SlowQueryDetector] = None,
        registry: Optional[Any] = None,
    ):
        """
        Initialize query logger.

        Args:
            slow_query_detector: Optional slow query detector instance
            registry: Optional Prometheus registry
        """
        self.slow_query_detector = slow_query_detector or SlowQueryDetector()
        query_metrics.initialize(registry)

    def execute(
        self,
        operation: str,
        table: str,
        func: Callable[[], Any],
        query_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a database operation with logging and metrics.

        Args:
            operation: Operation type (select, insert, update, delete, upsert, rpc)
            table: Table name
            func: Function that performs the database operation
            query_params: Optional query parameters for logging

        Returns:
            Result of the database operation
        """
        start_time = time.perf_counter()
        status = "success"
        error_type = None

        try:
            result = func()
            return result

        except Exception as e:
            status = "error"
            error_type = type(e).__name__

            # Record error metric
            if query_metrics._initialized:
                query_metrics.query_errors.labels(
                    operation=operation,
                    table=table,
                    error_type=error_type,
                ).inc()

            logger.error(
                f"Database query error: {operation} on {table} - {error_type}: {e}"
            )
            raise

        finally:
            duration = time.perf_counter() - start_time

            # Record duration metric
            if query_metrics._initialized:
                query_metrics.query_duration.labels(
                    operation=operation,
                    table=table,
                    status=status,
                ).observe(duration)

                query_metrics.query_total.labels(
                    operation=operation,
                    table=table,
                ).inc()

            # Check for slow query
            self.slow_query_detector.check_query(
                operation=operation,
                table=table,
                duration_sec=duration,
                query_params=query_params,
            )

            # Debug logging for all queries
            logger.debug(
                f"Query executed: {operation} on {table} "
                f"took {duration * 1000:.2f}ms (status: {status})"
            )

    async def execute_async(
        self,
        operation: str,
        table: str,
        func: Callable[[], Any],
        query_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute an async database operation with logging and metrics.

        Args:
            operation: Operation type (select, insert, update, delete, upsert, rpc)
            table: Table name
            func: Async function that performs the database operation
            query_params: Optional query parameters for logging

        Returns:
            Result of the database operation
        """
        start_time = time.perf_counter()
        status = "success"
        error_type = None

        try:
            result = await func()
            return result

        except Exception as e:
            status = "error"
            error_type = type(e).__name__

            # Record error metric
            if query_metrics._initialized:
                query_metrics.query_errors.labels(
                    operation=operation,
                    table=table,
                    error_type=error_type,
                ).inc()

            logger.error(
                f"Database query error: {operation} on {table} - {error_type}: {e}"
            )
            raise

        finally:
            duration = time.perf_counter() - start_time

            # Record duration metric
            if query_metrics._initialized:
                query_metrics.query_duration.labels(
                    operation=operation,
                    table=table,
                    status=status,
                ).observe(duration)

                query_metrics.query_total.labels(
                    operation=operation,
                    table=table,
                ).inc()

            # Check for slow query
            self.slow_query_detector.check_query(
                operation=operation,
                table=table,
                duration_sec=duration,
                query_params=query_params,
            )

            # Debug logging for all queries
            logger.debug(
                f"Query executed: {operation} on {table} "
                f"took {duration * 1000:.2f}ms (status: {status})"
            )

    @contextmanager
    def track_query(
        self,
        operation: str,
        table: str,
        query_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracking query execution.

        Usage:
            with query_logger.track_query("select", "kpi_values") as tracker:
                result = supabase.table("kpi_values").select("*").execute()
                tracker.set_result(result)

        Args:
            operation: Operation type
            table: Table name
            query_params: Optional query parameters
        """
        tracker = _QueryTracker(
            logger=self,
            operation=operation,
            table=table,
            query_params=query_params,
        )
        try:
            yield tracker
        except Exception as e:
            tracker.error = e
            raise
        finally:
            tracker.finish()

    @asynccontextmanager
    async def track_query_async(
        self,
        operation: str,
        table: str,
        query_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Async context manager for tracking query execution.

        Usage:
            async with query_logger.track_query_async("select", "kpi_values") as tracker:
                result = await async_query()
                tracker.set_result(result)

        Args:
            operation: Operation type
            table: Table name
            query_params: Optional query parameters
        """
        tracker = _QueryTracker(
            logger=self,
            operation=operation,
            table=table,
            query_params=query_params,
        )
        try:
            yield tracker
        except Exception as e:
            tracker.error = e
            raise
        finally:
            tracker.finish()


class _QueryTracker:
    """Internal helper class for tracking queries within context managers."""

    def __init__(
        self,
        logger: QueryLogger,
        operation: str,
        table: str,
        query_params: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger
        self.operation = operation
        self.table = table
        self.query_params = query_params
        self.start_time = time.perf_counter()
        self.result = None
        self.error: Optional[Exception] = None

    def set_result(self, result: Any) -> None:
        """Set the query result."""
        self.result = result

    def finish(self) -> None:
        """Finalize tracking and record metrics."""
        duration = time.perf_counter() - self.start_time
        status = "error" if self.error else "success"
        error_type = type(self.error).__name__ if self.error else None

        # Record metrics
        if query_metrics._initialized:
            query_metrics.query_duration.labels(
                operation=self.operation,
                table=self.table,
                status=status,
            ).observe(duration)

            query_metrics.query_total.labels(
                operation=self.operation,
                table=self.table,
            ).inc()

            if self.error:
                query_metrics.query_errors.labels(
                    operation=self.operation,
                    table=self.table,
                    error_type=error_type,
                ).inc()

        # Check for slow query
        self.logger.slow_query_detector.check_query(
            operation=self.operation,
            table=self.table,
            duration_sec=duration,
            query_params=self.query_params,
        )

        # Logging
        if self.error:
            logging.getLogger(__name__).error(
                f"Database query error: {self.operation} on {self.table} "
                f"- {error_type}: {self.error}"
            )
        else:
            logging.getLogger(__name__).debug(
                f"Query executed: {self.operation} on {self.table} "
                f"took {duration * 1000:.2f}ms (status: {status})"
            )


# Global query logger instance
query_logger = QueryLogger(slow_query_detector=slow_query_detector)


# =============================================================================
# Decorator for Repository Methods
# =============================================================================

T = TypeVar("T")


def logged_query(operation: str, table: str):
    """
    Decorator for repository methods that adds query logging.

    Usage:
        class KPIRepository:
            @logged_query("select", "kpi_values")
            def get_kpi_values(self, kpi_id: str):
                return self.supabase.table("kpi_values").select("*").eq("kpi_id", kpi_id).execute()

    Args:
        operation: Operation type
        table: Table name
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return query_logger.execute(
                operation=operation,
                table=table,
                func=lambda: func(*args, **kwargs),
                query_params=kwargs if kwargs else None,
            )
        return wrapper
    return decorator


def logged_query_async(operation: str, table: str):
    """
    Async decorator for repository methods that adds query logging.

    Usage:
        class KPIRepository:
            @logged_query_async("select", "kpi_values")
            async def get_kpi_values(self, kpi_id: str):
                return await self.async_query()

    Args:
        operation: Operation type
        table: Table name
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await query_logger.execute_async(
                operation=operation,
                table=table,
                func=lambda: func(*args, **kwargs),
                query_params=kwargs if kwargs else None,
            )
        return wrapper
    return decorator


# =============================================================================
# Convenience Functions
# =============================================================================


def get_query_stats() -> Dict[str, Any]:
    """
    Get comprehensive query statistics.

    Returns:
        Dictionary with query statistics
    """
    return {
        "slow_queries": slow_query_detector.get_slow_query_stats(),
        "metrics_initialized": query_metrics._initialized,
    }


def configure_slow_query_thresholds(
    default_threshold_sec: Optional[float] = None,
    operation_thresholds: Optional[Dict[str, float]] = None,
    table_thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """
    Configure slow query detection thresholds.

    Args:
        default_threshold_sec: Default threshold in seconds
        operation_thresholds: Per-operation thresholds
        table_thresholds: Per-table thresholds
    """
    if default_threshold_sec is not None:
        slow_query_detector.config.default_threshold_sec = default_threshold_sec

    if operation_thresholds:
        slow_query_detector.config.operation_thresholds.update(operation_thresholds)

    if table_thresholds:
        slow_query_detector.config.table_thresholds.update(table_thresholds)

    logger.info(
        f"Slow query thresholds updated: default={slow_query_detector.config.default_threshold_sec}s"
    )
