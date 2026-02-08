"""
Structured Logging Configuration.

G14 from observability audit remediation plan:
- Standardized structured logging across all modules
- JSON format for production (log aggregation tools)
- Human-readable format for development
- Context propagation (request_id, trace_id, span_id)
- Performance-optimized for high-throughput logging

Version: 1.0.0
"""

import json
import logging
import os
import sys
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# =============================================================================
# Context Variables for Request Tracking
# =============================================================================

# Request-scoped context variables
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
agent_name_var: ContextVar[Optional[str]] = ContextVar("agent_name", default=None)
operation_var: ContextVar[Optional[str]] = ContextVar("operation", default=None)


def set_request_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    operation: Optional[str] = None,
) -> None:
    """
    Set request context for logging.

    These values will be included in all log records from the current context.

    Args:
        request_id: Unique request identifier (X-Request-ID header)
        trace_id: W3C Trace Context trace_id (traceparent header)
        span_id: W3C Trace Context span_id
        user_id: Authenticated user identifier
        agent_name: Currently executing agent name
        operation: Current operation name
    """
    if request_id is not None:
        request_id_var.set(request_id)
    if trace_id is not None:
        trace_id_var.set(trace_id)
    if span_id is not None:
        span_id_var.set(span_id)
    if user_id is not None:
        user_id_var.set(user_id)
    if agent_name is not None:
        agent_name_var.set(agent_name)
    if operation is not None:
        operation_var.set(operation)


def clear_request_context() -> None:
    """Clear all request context variables."""
    request_id_var.set(None)
    trace_id_var.set(None)
    span_id_var.set(None)
    user_id_var.set(None)
    agent_name_var.set(None)
    operation_var.set(None)


def get_request_context() -> Dict[str, Optional[str]]:
    """Get current request context as a dictionary."""
    return {
        "request_id": request_id_var.get(),
        "trace_id": trace_id_var.get(),
        "span_id": span_id_var.get(),
        "user_id": user_id_var.get(),
        "agent_name": agent_name_var.get(),
        "operation": operation_var.get(),
    }


# =============================================================================
# JSON Formatter for Structured Logging
# =============================================================================


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Produces JSON log lines suitable for log aggregation tools like:
    - Elasticsearch/Kibana (ELK Stack)
    - Grafana Loki
    - AWS CloudWatch
    - Datadog

    Format:
    {
        "timestamp": "2025-01-22T12:00:00.000Z",
        "level": "INFO",
        "logger": "src.api.main",
        "message": "Request completed",
        "request_id": "abc123",
        "trace_id": "def456",
        "span_id": "ghi789",
        "extra": {...}
    }
    """

    def __init__(
        self,
        include_hostname: bool = True,
        include_thread: bool = False,
        include_process: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize JSON formatter.

        Args:
            include_hostname: Include hostname in log records
            include_thread: Include thread name in log records
            include_process: Include process ID in log records
            extra_fields: Static fields to include in all log records
        """
        super().__init__()
        self.include_hostname = include_hostname
        self.include_thread = include_thread
        self.include_process = include_process
        self.extra_fields = extra_fields or {}
        self._hostname: Optional[str] = None

        # Cache hostname
        if include_hostname:
            import socket

            self._hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build base log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context
        request_id = request_id_var.get()
        trace_id = trace_id_var.get()
        span_id = span_id_var.get()
        user_id = user_id_var.get()
        agent_name = agent_name_var.get()
        operation = operation_var.get()

        if request_id:
            log_entry["request_id"] = request_id
        if trace_id:
            log_entry["trace_id"] = trace_id
        if span_id:
            log_entry["span_id"] = span_id
        if user_id:
            log_entry["user_id"] = user_id
        if agent_name:
            log_entry["agent_name"] = agent_name
        if operation:
            log_entry["operation"] = operation

        # Add optional fields
        if self._hostname:
            log_entry["hostname"] = self._hostname
        if self.include_thread:
            log_entry["thread"] = record.threadName
        if self.include_process:
            log_entry["process"] = record.process

        # Add static extra fields
        log_entry.update(self.extra_fields)

        # Add source location for errors
        if record.levelno >= logging.WARNING:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add any extra fields from the record
        if hasattr(record, "__dict__"):
            # Standard fields to exclude
            standard_fields = {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            }
            extra = {
                k: v
                for k, v in record.__dict__.items()
                if k not in standard_fields and not k.startswith("_")
            }
            if extra:
                log_entry["extra"] = extra

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# =============================================================================
# Human-Readable Formatter for Development
# =============================================================================


class ColoredFormatter(logging.Formatter):
    """
    Colored human-readable formatter for development.

    Format:
    2025-01-22 12:00:00 [INFO ] src.api.main - Message here [req:abc123 trace:def456]
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
        "GRAY": "\033[90m",
        "BOLD": "\033[1m",
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize colored formatter.

        Args:
            use_colors: Whether to use ANSI colors (disable for CI/CD logs)
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Get colors
        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, "")
            reset = self.COLORS["RESET"]
            gray = self.COLORS["GRAY"]
        else:
            level_color = reset = gray = ""

        # Format level (padded to 8 chars)
        level = f"{level_color}[{record.levelname:8s}]{reset}"

        # Format logger name (truncated if too long)
        logger_name = record.name
        if len(logger_name) > 30:
            logger_name = "..." + logger_name[-27:]

        # Format message
        message = record.getMessage()

        # Build context string
        context_parts = []
        request_id = request_id_var.get()
        trace_id = trace_id_var.get()
        agent_name = agent_name_var.get()

        if request_id:
            context_parts.append(f"req:{request_id[:8]}")
        if trace_id:
            context_parts.append(f"trace:{trace_id[:8]}")
        if agent_name:
            context_parts.append(f"agent:{agent_name}")

        context_str = f" {gray}[{' '.join(context_parts)}]{reset}" if context_parts else ""

        # Base format
        output = f"{timestamp} {level} {logger_name} - {message}{context_str}"

        # Add exception if present
        if record.exc_info:
            output += "\n" + self.formatException(record.exc_info)

        return output


# =============================================================================
# Context-Aware Handler
# =============================================================================


class ContextFilter(logging.Filter):
    """
    Filter that adds context variables to log records.

    This allows using context in format strings like:
    %(request_id)s %(trace_id)s
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record and always return True."""
        record.request_id = request_id_var.get() or "-"
        record.trace_id = trace_id_var.get() or "-"
        record.span_id = span_id_var.get() or "-"
        record.user_id = user_id_var.get() or "-"
        record.agent_name = agent_name_var.get() or "-"
        record.operation = operation_var.get() or "-"
        return True


# =============================================================================
# Logging Configuration
# =============================================================================


class LoggingConfig:
    """
    Centralized logging configuration.

    Supports:
    - JSON format for production (LOG_FORMAT=json)
    - Human-readable format for development (LOG_FORMAT=text)
    - Configurable log levels per module
    - Request context propagation
    """

    # Singleton instance
    _instance: Optional["LoggingConfig"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls) -> "LoggingConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Read configuration from environment
        self.log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        self.log_format = os.environ.get("LOG_FORMAT", "text").lower()
        self.service_name = os.environ.get("SERVICE_NAME", "e2i-causal-analytics")
        self.environment = os.environ.get("ENVIRONMENT", "development")

        # Module-specific log levels (comma-separated: module=level,module2=level2)
        self.module_levels: Dict[str, str] = {}
        module_levels_str = os.environ.get("LOG_LEVELS", "")
        if module_levels_str:
            for pair in module_levels_str.split(","):
                if "=" in pair:
                    module, level = pair.split("=", 1)
                    self.module_levels[module.strip()] = level.strip().upper()

        self._initialized = True

    def configure(self) -> None:
        """Configure logging for the application."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Allow all, filter at handler level

        # Remove existing handlers
        root_logger.handlers.clear()

        # Create handler based on format
        if self.log_format == "json":
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                JSONFormatter(
                    include_hostname=True,
                    extra_fields={
                        "service": self.service_name,
                        "environment": self.environment,
                    },
                )
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            use_colors = os.environ.get("NO_COLOR", "").lower() not in ("1", "true", "yes")
            handler.setFormatter(ColoredFormatter(use_colors=use_colors))

        # Set base level
        handler.setLevel(getattr(logging, self.log_level, logging.INFO))

        # Add context filter
        handler.addFilter(ContextFilter())

        # Add handler to root logger
        root_logger.addHandler(handler)

        # Configure module-specific levels
        for module, level in self.module_levels.items():
            module_logger = logging.getLogger(module)
            module_logger.setLevel(getattr(logging, level, logging.INFO))

        # Quiet noisy third-party loggers
        for noisy_logger in [
            "httpx",
            "httpcore",
            "urllib3",
            "asyncio",
            "uvicorn.access",
            "uvicorn.error",
            "watchfiles",
            "hpack",
            "charset_normalizer",
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        # Log configuration
        logger = logging.getLogger(__name__)
        logger.info(
            f"Logging configured: format={self.log_format}, level={self.log_level}, "
            f"service={self.service_name}, environment={self.environment}"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def configure_logging() -> None:
    """Configure application logging (call once at startup)."""
    config = LoggingConfig()
    config.configure()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    This is a convenience wrapper around logging.getLogger that ensures
    logging is configured.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# =============================================================================
# Log Level Context Manager
# =============================================================================


class log_level_context:
    """
    Temporarily change log level for a block of code.

    Usage:
        with log_level_context(logging.DEBUG, "src.agents"):
            # Debug logging enabled for this block
            ...
    """

    def __init__(self, level: int, logger_name: Optional[str] = None):
        """
        Initialize context manager.

        Args:
            level: Target log level
            logger_name: Logger to modify (None for root)
        """
        self.level = level
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level

    def __enter__(self):
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        return False


# =============================================================================
# Performance Logging Utilities
# =============================================================================


class timed_operation:
    """
    Context manager for timing operations and logging duration.

    Usage:
        with timed_operation("database_query", logger):
            result = await db.query(...)
    """

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        level: int = logging.DEBUG,
        warn_threshold_ms: Optional[float] = None,
    ):
        """
        Initialize timed operation context.

        Args:
            operation_name: Name of the operation being timed
            logger: Logger to use (defaults to module logger)
            level: Log level for timing messages
            warn_threshold_ms: Log as WARNING if duration exceeds this
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.level = level
        self.warn_threshold_ms = warn_threshold_ms
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.duration_ms = (end_time - self.start_time) * 1000

        # Determine log level
        level = self.level
        if self.warn_threshold_ms and self.duration_ms > self.warn_threshold_ms:
            level = logging.WARNING

        # Log timing
        self.logger.log(
            level,
            f"{self.operation_name} completed in {self.duration_ms:.2f}ms",
            extra={"operation": self.operation_name, "duration_ms": self.duration_ms},
        )

        return False
