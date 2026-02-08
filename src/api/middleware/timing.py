"""Request Timing Middleware.

Tracks request latency and records metrics for Prometheus scraping.
Adds Server-Timing header to responses for debugging.

Quick Win QW3 from observability audit remediation plan.

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
import time
from typing import Callable, cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Try to import metrics recording functions
try:
    from src.api.routes.metrics import record_error, record_request

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("metrics module not available, timing metrics will not be recorded")

    def record_request(*args, **kwargs):  # type: ignore[misc]
        pass

    def record_error(*args, **kwargs):  # type: ignore[misc]
        pass


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks request timing and records latency metrics.

    Features:
    - Measures total request duration (including response body streaming)
    - Records metrics to Prometheus registry
    - Adds Server-Timing header for debugging
    - Tracks slow requests (configurable threshold)
    - Excludes health check endpoints from metrics

    Configuration via environment variables:
    - TIMING_SLOW_THRESHOLD_MS: Threshold for logging slow requests (default: 1000ms)
    - TIMING_EXCLUDE_PATHS: Comma-separated paths to exclude from timing
    """

    def __init__(
        self,
        app: Callable,
        slow_threshold_ms: float = 1000.0,
        exclude_paths: list[str] | None = None,
        add_server_timing: bool = True,
    ):
        """Initialize timing middleware.

        Args:
            app: The ASGI application
            slow_threshold_ms: Threshold in ms for logging slow requests
            exclude_paths: Paths to exclude from timing metrics
            add_server_timing: Whether to add Server-Timing header
        """
        super().__init__(app)
        self.slow_threshold_ms = slow_threshold_ms
        self.add_server_timing = add_server_timing

        # Default paths to exclude from metrics (high-volume health checks)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/healthz",
            "/ready",
            "/metrics",
            "/metrics/health",
        ]

    def _should_track(self, path: str) -> bool:
        """Check if request path should be tracked.

        Args:
            path: Request URL path

        Returns:
            True if the request should be timed and recorded
        """
        return not any(path.startswith(exclude) for exclude in self.exclude_paths)

    def _normalize_path(self, path: str) -> str:
        """Normalize path for metric labeling.

        Replaces dynamic path segments (UUIDs, IDs) with placeholders
        to prevent metric cardinality explosion.

        Args:
            path: Request URL path

        Returns:
            Normalized path for use as metric label
        """
        import re

        # Replace UUIDs (including UUID7 format)
        path = re.sub(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            "{uuid}",
            path,
        )

        # Replace numeric IDs
        path = re.sub(r"/\d+(?=/|$)", "/{id}", path)

        # Replace date-like segments (YYYY-MM-DD)
        path = re.sub(r"\d{4}-\d{2}-\d{2}", "{date}", path)

        return path

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track timing.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response with optional Server-Timing header
        """
        path = request.url.path
        method = request.method

        # Check if we should track this request
        should_track = self._should_track(path)

        # Record start time
        start_time = time.perf_counter()

        # Track any errors that occur
        error_occurred = False
        error_type = None

        try:
            response: Response = cast(Response, await call_next(request))
            status_code = response.status_code

            # Track client/server errors
            if status_code >= 400:
                error_occurred = True
                if status_code >= 500:
                    error_type = "server_error"
                elif status_code == 429:
                    error_type = "rate_limited"
                elif status_code == 401 or status_code == 403:
                    error_type = "auth_error"
                elif status_code == 404:
                    error_type = "not_found"
                else:
                    error_type = "client_error"

        except Exception as exc:
            # Re-raise after recording, let error handlers deal with it
            error_occurred = True
            error_type = type(exc).__name__
            raise

        finally:
            # Calculate duration
            duration = time.perf_counter() - start_time
            duration_ms = duration * 1000

            # Record metrics if tracking is enabled
            if should_track:
                normalized_path = self._normalize_path(path)

                # Record request metric
                record_request(
                    method=method,
                    endpoint=normalized_path,
                    status_code=status_code if "response" in dir() else 500,
                    latency=duration,
                )

                # Record error metric if applicable
                if error_occurred and error_type:
                    record_error(
                        method=method,
                        endpoint=normalized_path,
                        error_type=error_type,
                    )

                # Log slow requests
                if duration_ms > self.slow_threshold_ms:
                    logger.warning(
                        f"Slow request: {method} {path} took {duration_ms:.2f}ms "
                        f"(threshold: {self.slow_threshold_ms}ms)",
                        extra={
                            "method": method,
                            "path": path,
                            "duration_ms": duration_ms,
                            "status_code": status_code if "response" in dir() else None,
                        },
                    )

        # Add Server-Timing header if enabled and response exists
        if self.add_server_timing and "response" in dir():
            # Server-Timing header for browser DevTools
            response.headers["Server-Timing"] = f"total;dur={duration_ms:.2f}"

        return response


class RequestTimingContext:
    """Context class for storing timing information in request state.

    Used to track timing across middleware layers and handlers.
    """

    def __init__(self) -> None:
        self.start_time: float = time.perf_counter()
        self.checkpoints: dict[str, float] = {}

    def checkpoint(self, name: str) -> float:
        """Record a timing checkpoint.

        Args:
            name: Name of the checkpoint

        Returns:
            Time since start in milliseconds
        """
        elapsed = (time.perf_counter() - self.start_time) * 1000
        self.checkpoints[name] = elapsed
        return elapsed

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000

    def to_server_timing(self) -> str:
        """Generate Server-Timing header value from checkpoints.

        Returns:
            Server-Timing header value
        """
        parts = []
        for name, elapsed in self.checkpoints.items():
            parts.append(f"{name};dur={elapsed:.2f}")

        # Add total
        parts.append(f"total;dur={self.elapsed_ms:.2f}")

        return ", ".join(parts)
