"""OpenTelemetry Configuration and Initialization.

Configures distributed tracing with OpenTelemetry SDK for the E2I platform.
Supports multiple exporters (OTLP, Console, Jaeger) and integrates with
existing trace context from the TracingMiddleware.

Phase 1 G02 from observability audit remediation plan.

Exporters:
- OTLP (default): For Jaeger, Tempo, or any OTLP-compatible backend
- Console: For development debugging
- None: Disabled (sampling rate 0)

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
import os

logger = logging.getLogger(__name__)

# Track initialization state
_otel_initialized = False
_tracer = None

# Configuration from environment
OTEL_ENABLED = os.environ.get("OTEL_ENABLED", "true").lower() in ("1", "true", "yes")
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "e2i-causal-analytics")
OTEL_SERVICE_VERSION = os.environ.get("OTEL_SERVICE_VERSION", "4.2.0")
OTEL_EXPORTER_TYPE = os.environ.get("OTEL_EXPORTER_TYPE", "otlp")  # otlp, console, none
OTEL_EXPORTER_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
OTEL_SAMPLING_RATE = float(os.environ.get("OTEL_SAMPLING_RATE", "1.0"))


def init_opentelemetry() -> bool:
    """Initialize OpenTelemetry with configured exporter.

    Returns:
        True if initialization succeeded, False otherwise
    """
    global _otel_initialized, _tracer

    if _otel_initialized:
        logger.debug("OpenTelemetry already initialized")
        return True

    if not OTEL_ENABLED:
        logger.info("OpenTelemetry: DISABLED (OTEL_ENABLED=false)")
        _otel_initialized = True
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.baggage.propagation import W3CBaggagePropagator
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.propagators.composite import CompositeHTTPPropagator
        from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_OFF,
            ALWAYS_ON,
            ParentBasedTraceIdRatio,
        )
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        # Configure sampler based on rate
        if OTEL_SAMPLING_RATE <= 0:
            sampler = ALWAYS_OFF
            logger.info("OpenTelemetry: Sampling DISABLED (rate=0)")
        elif OTEL_SAMPLING_RATE >= 1.0:
            sampler = ALWAYS_ON
            logger.info("OpenTelemetry: Sampling ALL traces (rate=1.0)")
        else:
            sampler = ParentBasedTraceIdRatio(OTEL_SAMPLING_RATE)
            logger.info(f"OpenTelemetry: Sampling {OTEL_SAMPLING_RATE * 100:.1f}% of traces")

        # Create resource with service info
        resource = Resource.create(
            {
                SERVICE_NAME: OTEL_SERVICE_NAME,
                SERVICE_VERSION: OTEL_SERVICE_VERSION,
                "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
                "host.name": os.environ.get("HOSTNAME", "unknown"),
            }
        )

        # Create tracer provider
        provider = TracerProvider(
            resource=resource,
            sampler=sampler,
        )

        # Configure exporter
        span_processor = _create_span_processor()
        if span_processor:
            provider.add_span_processor(span_processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Configure propagators (W3C Trace Context + Baggage)
        propagator = CompositeHTTPPropagator(
            [TraceContextTextMapPropagator(), W3CBaggagePropagator()]
        )
        set_global_textmap(propagator)

        # Get tracer for this module
        _tracer = trace.get_tracer(__name__, OTEL_SERVICE_VERSION)

        _otel_initialized = True
        logger.info(
            f"OpenTelemetry: ENABLED (service={OTEL_SERVICE_NAME}, exporter={OTEL_EXPORTER_TYPE})"
        )
        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry: SDK not available - {e}")
        _otel_initialized = True
        return False
    except Exception as e:
        logger.error(f"OpenTelemetry: Initialization failed - {e}")
        _otel_initialized = True
        return False


def _create_span_processor():
    """Create span processor based on configured exporter type.

    Returns:
        SpanProcessor instance or None
    """
    if OTEL_EXPORTER_TYPE == "none":
        logger.info("OpenTelemetry: No exporter configured")
        return None

    if OTEL_EXPORTER_TYPE == "console":
        try:
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            logger.info("OpenTelemetry: Using ConsoleSpanExporter")
            return SimpleSpanProcessor(ConsoleSpanExporter())
        except ImportError:
            logger.warning("ConsoleSpanExporter not available")
            return None

    if OTEL_EXPORTER_TYPE == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = OTLPSpanExporter(
                endpoint=OTEL_EXPORTER_ENDPOINT,
                insecure=os.environ.get("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
            )
            logger.info(f"OpenTelemetry: Using OTLP exporter ({OTEL_EXPORTER_ENDPOINT})")
            return BatchSpanProcessor(exporter)
        except ImportError:
            logger.warning("OTLP exporter not available. Install opentelemetry-exporter-otlp")
            # Fall back to console
            return _create_console_processor()

    logger.warning(f"Unknown exporter type: {OTEL_EXPORTER_TYPE}")
    return None


def _create_console_processor():
    """Create console span processor as fallback."""
    try:
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        logger.info("OpenTelemetry: Falling back to ConsoleSpanExporter")
        return SimpleSpanProcessor(ConsoleSpanExporter())
    except ImportError:
        return None


def get_tracer(name: str = __name__):
    """Get a tracer instance.

    Args:
        name: Tracer name (typically __name__ of calling module)

    Returns:
        Tracer instance or NoOpTracer if not initialized
    """
    if not _otel_initialized:
        init_opentelemetry()

    try:
        from opentelemetry import trace

        return trace.get_tracer(name, OTEL_SERVICE_VERSION)
    except ImportError:
        return _NoOpTracer()


def shutdown_opentelemetry() -> None:
    """Gracefully shutdown OpenTelemetry, flushing any pending spans."""
    global _otel_initialized

    if not _otel_initialized:
        return

    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
            logger.info("OpenTelemetry: Shutdown complete")
    except Exception as e:
        logger.warning(f"OpenTelemetry: Shutdown error - {e}")

    _otel_initialized = False


class _NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""

    def start_span(self, name, *args, **kwargs):
        return _NoOpSpan()

    def start_as_current_span(self, name, *args, **kwargs):
        return _NoOpContextManager()


class _NoOpSpan:
    """No-op span."""

    def set_attribute(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpContextManager:
    """No-op context manager for spans."""

    def __enter__(self):
        return _NoOpSpan()

    def __exit__(self, *args):
        pass


# =============================================================================
# ASGI Middleware Integration
# =============================================================================


def get_opentelemetry_middleware():
    """Get OpenTelemetry ASGI middleware if available.

    Returns:
        ASGI middleware class or None if not available
    """
    if not OTEL_ENABLED:
        return None

    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        return OpenTelemetryMiddleware
    except ImportError:
        logger.warning(
            "OpenTelemetry ASGI middleware not available. "
            "Install opentelemetry-instrumentation-asgi"
        )
        return None


def instrument_fastapi(app) -> bool:
    """Instrument a FastAPI application with OpenTelemetry.

    This adds automatic tracing for all HTTP requests.

    Args:
        app: FastAPI application instance

    Returns:
        True if instrumentation succeeded
    """
    if not OTEL_ENABLED or not _otel_initialized:
        return False

    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        # Wrap with ASGI middleware
        # Note: This must be added after other middleware for proper span hierarchy
        app.add_middleware(
            OpenTelemetryMiddleware,
            excluded_urls="health,healthz,ready,metrics",
        )
        logger.info("OpenTelemetry: FastAPI instrumentation added")
        return True
    except ImportError:
        logger.warning("OpenTelemetry ASGI instrumentation not available")
        return False
    except Exception as e:
        logger.error(f"OpenTelemetry: FastAPI instrumentation failed - {e}")
        return False


# =============================================================================
# Span Utilities
# =============================================================================


def create_span_from_trace_context(name: str, trace_context: dict):
    """Create a span linked to existing trace context.

    Useful for linking to trace context extracted by TracingMiddleware.

    Args:
        name: Span name
        trace_context: Dict with trace_id, span_id, trace_flags

    Returns:
        Span context manager
    """
    tracer = get_tracer(__name__)

    if not trace_context.get("trace_id"):
        return tracer.start_as_current_span(name)

    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanContext, TraceFlags

        # Create span context from extracted values
        span_context = SpanContext(
            trace_id=int(trace_context["trace_id"], 16),
            span_id=int(trace_context.get("span_id", "0" * 16), 16),
            is_remote=True,
            trace_flags=TraceFlags(int(trace_context.get("trace_flags", "00"), 16)),
        )

        # Create span with parent context
        ctx = trace.set_span_in_context(trace.NonRecordingSpan(span_context))
        return tracer.start_as_current_span(name, context=ctx)

    except Exception as e:
        logger.debug(f"Could not link to existing trace context: {e}")
        return tracer.start_as_current_span(name)
