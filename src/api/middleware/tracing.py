"""Trace Header Extraction Middleware.

Extracts distributed tracing headers and makes trace context available
throughout the request lifecycle. Supports multiple tracing formats.

Quick Win QW5 from observability audit remediation plan.

Supported Headers:
- X-Request-ID: Unique request identifier
- X-Correlation-ID: Cross-service correlation ID
- traceparent: W3C Trace Context format (OpenTelemetry)
- X-B3-TraceId: Zipkin B3 format
- X-Trace-ID: Generic trace ID

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Import structured logging context (G14)
from src.utils.logging_config import (
    set_request_context as set_logging_context,
    clear_request_context as clear_logging_context,
)

# Use UUID7 for new request IDs (required by Opik)
try:
    from uuid_extensions import uuid7

    def generate_request_id() -> str:
        return str(uuid7())
except ImportError:
    import uuid

    def generate_request_id() -> str:
        return str(uuid.uuid4())


logger = logging.getLogger(__name__)

# Context variables for thread-safe trace context access
_request_id: ContextVar[str] = ContextVar("request_id", default="")
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_span_id: ContextVar[str] = ContextVar("span_id", default="")
_trace_flags: ContextVar[str] = ContextVar("trace_flags", default="00")


@dataclass
class TraceContext:
    """Trace context extracted from request headers.

    Attributes:
        request_id: Unique request identifier (generated if not provided)
        correlation_id: Cross-service correlation ID (may be same as request_id)
        trace_id: Distributed trace ID (from traceparent or other headers)
        span_id: Current span ID (from traceparent)
        trace_flags: Trace flags (sampling decision)
        parent_id: Parent span ID if available
    """

    request_id: str
    correlation_id: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    trace_flags: str = "00"
    parent_id: Optional[str] = None

    @classmethod
    def from_headers(cls, headers: dict) -> "TraceContext":
        """Create TraceContext from request headers.

        Args:
            headers: Request headers dict (lowercase keys)

        Returns:
            TraceContext with extracted or generated values
        """
        # Extract request ID (or generate one)
        request_id = (
            headers.get("x-request-id")
            or headers.get("x-amzn-trace-id")
            or generate_request_id()
        )

        # Extract correlation ID (or use request ID)
        correlation_id = headers.get("x-correlation-id") or request_id

        # Initialize trace context
        trace_id = None
        span_id = None
        trace_flags = "00"
        parent_id = None

        # Try W3C Trace Context format (OpenTelemetry standard)
        # Format: 00-<trace-id>-<parent-id>-<trace-flags>
        traceparent = headers.get("traceparent")
        if traceparent:
            match = re.match(
                r"^(\d{2})-([a-f0-9]{32})-([a-f0-9]{16})-([a-f0-9]{2})$",
                traceparent,
            )
            if match:
                version, trace_id, parent_id, trace_flags = match.groups()
                # Generate new span ID for this request
                span_id = generate_request_id()[:16].replace("-", "")

        # Fallback to Zipkin B3 format
        if not trace_id:
            trace_id = headers.get("x-b3-traceid")
            span_id = headers.get("x-b3-spanid")
            parent_id = headers.get("x-b3-parentspanid")
            if headers.get("x-b3-sampled") == "1":
                trace_flags = "01"

        # Fallback to generic trace ID header
        if not trace_id:
            trace_id = headers.get("x-trace-id")

        return cls(
            request_id=request_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            parent_id=parent_id,
        )


# Global accessor functions for trace context
def get_request_id() -> str:
    """Get current request ID from context."""
    return _request_id.get()


def get_correlation_id() -> str:
    """Get current correlation ID from context."""
    return _correlation_id.get()


def get_trace_id() -> str:
    """Get current trace ID from context."""
    return _trace_id.get()


def get_span_id() -> str:
    """Get current span ID from context."""
    return _span_id.get()


def get_trace_context() -> TraceContext:
    """Get full trace context from context variables."""
    return TraceContext(
        request_id=_request_id.get(),
        correlation_id=_correlation_id.get(),
        trace_id=_trace_id.get() or None,
        span_id=_span_id.get() or None,
        trace_flags=_trace_flags.get(),
    )


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware that extracts and propagates trace context.

    Features:
    - Extracts trace headers from incoming requests
    - Generates request ID if not provided (using UUID7)
    - Stores trace context in request state and context variables
    - Adds trace headers to responses for correlation
    - Supports W3C Trace Context, Zipkin B3, and custom formats

    Configuration:
    - add_response_headers: Add trace headers to response (default: True)
    - log_trace_context: Log trace context for each request (default: False)
    """

    def __init__(
        self,
        app: Callable,
        add_response_headers: bool = True,
        log_trace_context: bool = False,
    ):
        """Initialize tracing middleware.

        Args:
            app: The ASGI application
            add_response_headers: Whether to add trace headers to responses
            log_trace_context: Whether to log trace context for debugging
        """
        super().__init__(app)
        self.add_response_headers = add_response_headers
        self.log_trace_context = log_trace_context

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Extract trace context and process request.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response with trace headers added
        """
        # Extract headers (normalize to lowercase)
        headers = {k.lower(): v for k, v in request.headers.items()}

        # Create trace context from headers
        trace_ctx = TraceContext.from_headers(headers)

        # Store in context variables for thread-safe access
        request_id_token = _request_id.set(trace_ctx.request_id)
        correlation_id_token = _correlation_id.set(trace_ctx.correlation_id)
        trace_id_token = _trace_id.set(trace_ctx.trace_id or "")
        span_id_token = _span_id.set(trace_ctx.span_id or "")
        trace_flags_token = _trace_flags.set(trace_ctx.trace_flags)

        # Set structured logging context (G14)
        set_logging_context(
            request_id=trace_ctx.request_id,
            trace_id=trace_ctx.trace_id,
            span_id=trace_ctx.span_id,
        )

        # Store in request state for handler access
        request.state.trace_context = trace_ctx
        request.state.request_id = trace_ctx.request_id
        request.state.correlation_id = trace_ctx.correlation_id

        # Log trace context if enabled
        if self.log_trace_context:
            logger.debug(
                f"Trace context: request_id={trace_ctx.request_id}, "
                f"correlation_id={trace_ctx.correlation_id}, "
                f"trace_id={trace_ctx.trace_id}, "
                f"span_id={trace_ctx.span_id}"
            )

        try:
            response = await call_next(request)

            # Add trace headers to response
            if self.add_response_headers:
                response.headers["X-Request-ID"] = trace_ctx.request_id
                response.headers["X-Correlation-ID"] = trace_ctx.correlation_id

                # Add traceparent if we have trace context
                if trace_ctx.trace_id and trace_ctx.span_id:
                    response.headers["traceparent"] = (
                        f"00-{trace_ctx.trace_id}-{trace_ctx.span_id}-{trace_ctx.trace_flags}"
                    )

            return response

        finally:
            # Reset context variables
            _request_id.reset(request_id_token)
            _correlation_id.reset(correlation_id_token)
            _trace_id.reset(trace_id_token)
            _span_id.reset(span_id_token)
            _trace_flags.reset(trace_flags_token)

            # Clear structured logging context (G14)
            clear_logging_context()


# Utility for structured logging with trace context
def with_trace_context(extra: dict | None = None) -> dict:
    """Add trace context to logging extra dict.

    Args:
        extra: Existing extra dict to augment

    Returns:
        Dict with trace context fields added
    """
    result = extra or {}
    result["request_id"] = get_request_id()
    result["correlation_id"] = get_correlation_id()
    if trace_id := get_trace_id():
        result["trace_id"] = trace_id
    if span_id := get_span_id():
        result["span_id"] = span_id
    return result
