"""Unit tests for Structured Logging Configuration module.

Tests cover:
- Context variables for request tracking
- JSONFormatter for structured logging
- ColoredFormatter for development
- ContextFilter for adding context to records
- LoggingConfig singleton and configuration
- Log level context manager
- Timed operation context manager

G14 from observability audit remediation plan.
"""

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from src.utils.logging_config import (
    ColoredFormatter,
    ContextFilter,
    # Formatters
    JSONFormatter,
    # Configuration
    LoggingConfig,
    agent_name_var,
    clear_request_context,
    configure_logging,
    get_logger,
    get_request_context,
    # Utilities
    log_level_context,
    operation_var,
    # Context variables
    request_id_var,
    set_request_context,
    span_id_var,
    timed_operation,
    trace_id_var,
    user_id_var,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clear_context():
    """Clear request context before and after each test."""
    clear_request_context()
    yield
    clear_request_context()


@pytest.fixture
def sample_log_record():
    """Create a sample log record."""
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/path/to/file.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    return record


@pytest.fixture
def error_log_record():
    """Create an error log record with exception info."""
    try:
        raise ValueError("Test error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname="/path/to/file.py",
        lineno=42,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )
    return record


# =============================================================================
# CONTEXT VARIABLE TESTS
# =============================================================================


class TestContextVariables:
    """Tests for request context variables."""

    def test_request_id_default_none(self):
        """Test request_id_var defaults to None."""
        assert request_id_var.get() is None

    def test_trace_id_default_none(self):
        """Test trace_id_var defaults to None."""
        assert trace_id_var.get() is None

    def test_span_id_default_none(self):
        """Test span_id_var defaults to None."""
        assert span_id_var.get() is None

    def test_user_id_default_none(self):
        """Test user_id_var defaults to None."""
        assert user_id_var.get() is None

    def test_agent_name_default_none(self):
        """Test agent_name_var defaults to None."""
        assert agent_name_var.get() is None

    def test_operation_default_none(self):
        """Test operation_var defaults to None."""
        assert operation_var.get() is None


class TestSetRequestContext:
    """Tests for set_request_context function."""

    def test_set_request_id(self):
        """Test setting request_id."""
        set_request_context(request_id="req-123")
        assert request_id_var.get() == "req-123"

    def test_set_trace_id(self):
        """Test setting trace_id."""
        set_request_context(trace_id="trace-456")
        assert trace_id_var.get() == "trace-456"

    def test_set_multiple_values(self):
        """Test setting multiple context values."""
        set_request_context(
            request_id="req-123",
            trace_id="trace-456",
            user_id="user-789",
            agent_name="orchestrator",
            operation="query",
        )
        assert request_id_var.get() == "req-123"
        assert trace_id_var.get() == "trace-456"
        assert user_id_var.get() == "user-789"
        assert agent_name_var.get() == "orchestrator"
        assert operation_var.get() == "query"

    def test_set_none_preserves_value(self):
        """Test that passing None explicitly preserves existing value."""
        set_request_context(request_id="req-123")
        set_request_context(trace_id="trace-456")  # Should not clear request_id
        assert request_id_var.get() == "req-123"


class TestClearRequestContext:
    """Tests for clear_request_context function."""

    def test_clear_all_values(self):
        """Test clearing all context values."""
        set_request_context(
            request_id="req-123",
            trace_id="trace-456",
            user_id="user-789",
        )
        clear_request_context()
        assert request_id_var.get() is None
        assert trace_id_var.get() is None
        assert user_id_var.get() is None


class TestGetRequestContext:
    """Tests for get_request_context function."""

    def test_get_empty_context(self):
        """Test getting empty context."""
        ctx = get_request_context()
        assert ctx["request_id"] is None
        assert ctx["trace_id"] is None

    def test_get_populated_context(self):
        """Test getting populated context."""
        set_request_context(request_id="req-123", trace_id="trace-456")
        ctx = get_request_context()
        assert ctx["request_id"] == "req-123"
        assert ctx["trace_id"] == "trace-456"


# =============================================================================
# JSON FORMATTER TESTS
# =============================================================================


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_formatter_creation(self):
        """Test JSONFormatter creation."""
        formatter = JSONFormatter()
        assert formatter is not None

    def test_format_returns_valid_json(self, sample_log_record):
        """Test format returns valid JSON."""
        formatter = JSONFormatter()
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_format_includes_required_fields(self, sample_log_record):
        """Test format includes required fields."""
        formatter = JSONFormatter()
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert "timestamp" in data
        assert "level" in data
        assert "logger" in data
        assert "message" in data

    def test_format_includes_context(self, sample_log_record):
        """Test format includes request context."""
        set_request_context(request_id="req-123", trace_id="trace-456")
        formatter = JSONFormatter()
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert data["request_id"] == "req-123"
        assert data["trace_id"] == "trace-456"

    def test_format_includes_hostname(self, sample_log_record):
        """Test format includes hostname when configured."""
        formatter = JSONFormatter(include_hostname=True)
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert "hostname" in data

    def test_format_excludes_hostname(self, sample_log_record):
        """Test format excludes hostname when disabled."""
        formatter = JSONFormatter(include_hostname=False)
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert "hostname" not in data

    def test_format_includes_thread(self, sample_log_record):
        """Test format includes thread when configured."""
        formatter = JSONFormatter(include_thread=True)
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert "thread" in data

    def test_format_includes_process(self, sample_log_record):
        """Test format includes process when configured."""
        formatter = JSONFormatter(include_process=True)
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert "process" in data

    def test_format_includes_extra_fields(self, sample_log_record):
        """Test format includes static extra fields."""
        formatter = JSONFormatter(extra_fields={"service": "test-service"})
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert data["service"] == "test-service"

    def test_format_includes_source_for_warnings(self, sample_log_record):
        """Test format includes source info for warnings."""
        sample_log_record.levelno = logging.WARNING
        formatter = JSONFormatter()
        output = formatter.format(sample_log_record)
        data = json.loads(output)
        assert "source" in data
        assert "file" in data["source"]
        assert "line" in data["source"]

    def test_format_includes_exception_info(self, error_log_record):
        """Test format includes exception info."""
        formatter = JSONFormatter()
        output = formatter.format(error_log_record)
        data = json.loads(output)
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"


# =============================================================================
# COLORED FORMATTER TESTS
# =============================================================================


class TestColoredFormatter:
    """Tests for ColoredFormatter class."""

    def test_formatter_creation(self):
        """Test ColoredFormatter creation."""
        formatter = ColoredFormatter()
        assert formatter is not None

    def test_format_returns_string(self, sample_log_record):
        """Test format returns string."""
        formatter = ColoredFormatter(use_colors=False)
        output = formatter.format(sample_log_record)
        assert isinstance(output, str)

    def test_format_includes_timestamp(self, sample_log_record):
        """Test format includes timestamp."""
        formatter = ColoredFormatter(use_colors=False)
        output = formatter.format(sample_log_record)
        # Should contain date-like pattern
        assert "-" in output  # Date separator

    def test_format_includes_level(self, sample_log_record):
        """Test format includes level."""
        formatter = ColoredFormatter(use_colors=False)
        output = formatter.format(sample_log_record)
        assert "INFO" in output

    def test_format_includes_message(self, sample_log_record):
        """Test format includes message."""
        formatter = ColoredFormatter(use_colors=False)
        output = formatter.format(sample_log_record)
        assert "Test message" in output

    def test_format_includes_context(self, sample_log_record):
        """Test format includes context when set."""
        set_request_context(request_id="req-12345678")
        formatter = ColoredFormatter(use_colors=False)
        output = formatter.format(sample_log_record)
        assert "req:req-1234" in output  # Truncated to 8 chars


# =============================================================================
# CONTEXT FILTER TESTS
# =============================================================================


class TestContextFilter:
    """Tests for ContextFilter class."""

    def test_filter_creation(self):
        """Test ContextFilter creation."""
        filter = ContextFilter()
        assert filter is not None

    def test_filter_always_returns_true(self, sample_log_record):
        """Test filter always returns True."""
        filter = ContextFilter()
        result = filter.filter(sample_log_record)
        assert result is True

    def test_filter_adds_context_to_record(self, sample_log_record):
        """Test filter adds context to record."""
        set_request_context(request_id="req-123", trace_id="trace-456")
        filter = ContextFilter()
        filter.filter(sample_log_record)
        assert sample_log_record.request_id == "req-123"
        assert sample_log_record.trace_id == "trace-456"

    def test_filter_uses_dash_for_missing(self, sample_log_record):
        """Test filter uses '-' for missing context."""
        filter = ContextFilter()
        filter.filter(sample_log_record)
        assert sample_log_record.request_id == "-"
        assert sample_log_record.trace_id == "-"


# =============================================================================
# LOGGING CONFIG TESTS
# =============================================================================


class TestLoggingConfig:
    """Tests for LoggingConfig class."""

    def test_config_singleton(self):
        """Test LoggingConfig is singleton."""
        config1 = LoggingConfig()
        config2 = LoggingConfig()
        assert config1 is config2

    def test_config_reads_environment(self):
        """Test LoggingConfig reads from environment."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG", "LOG_FORMAT": "json"}):
            # Reset singleton for test
            LoggingConfig._instance = None
            LoggingConfig._initialized = False
            config = LoggingConfig()
            assert config.log_level == "DEBUG"
            assert config.log_format == "json"

    def test_config_defaults(self):
        """Test LoggingConfig defaults."""
        with patch.dict(os.environ, {}, clear=True):
            LoggingConfig._instance = None
            LoggingConfig._initialized = False
            config = LoggingConfig()
            assert config.log_level == "INFO"
            assert config.log_format == "text"


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_callable(self):
        """Test configure_logging is callable."""
        # Just test it doesn't raise
        # Note: This modifies global logging config
        configure_logging()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"


# =============================================================================
# LOG LEVEL CONTEXT TESTS
# =============================================================================


class TestLogLevelContext:
    """Tests for log_level_context manager."""

    def test_context_changes_level(self):
        """Test context manager changes log level."""
        logger = logging.getLogger("test.level.context")
        original_level = logger.level

        with log_level_context(logging.DEBUG, "test.level.context"):
            assert logger.level == logging.DEBUG

        assert logger.level == original_level

    def test_context_restores_on_exit(self):
        """Test context manager restores level on exit."""
        logger = logging.getLogger("test.restore")
        logger.setLevel(logging.WARNING)

        with log_level_context(logging.DEBUG, "test.restore"):
            pass

        assert logger.level == logging.WARNING

    def test_context_restores_on_exception(self):
        """Test context manager restores level on exception."""
        logger = logging.getLogger("test.exception")
        logger.setLevel(logging.WARNING)

        try:
            with log_level_context(logging.DEBUG, "test.exception"):
                raise ValueError("Test error")
        except ValueError:
            pass

        assert logger.level == logging.WARNING


# =============================================================================
# TIMED OPERATION TESTS
# =============================================================================


class TestTimedOperation:
    """Tests for timed_operation context manager."""

    def test_timed_operation_captures_duration(self):
        """Test timed_operation captures duration."""
        with timed_operation("test_op") as timer:
            pass

        assert timer.duration_ms is not None
        assert timer.duration_ms >= 0

    def test_timed_operation_uses_logger(self):
        """Test timed_operation uses provided logger."""
        mock_logger = MagicMock()

        with timed_operation("test_op", logger=mock_logger):
            pass

        mock_logger.log.assert_called()

    def test_timed_operation_warns_on_slow(self):
        """Test timed_operation logs WARNING for slow operations."""
        mock_logger = MagicMock()

        import time

        with timed_operation("test_op", logger=mock_logger, warn_threshold_ms=0.001):
            time.sleep(0.01)  # Sleep 10ms, threshold is 0.001ms

        # Check that WARNING level was used
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.WARNING

    def test_timed_operation_records_name(self):
        """Test timed_operation records operation name."""
        with timed_operation("my_operation") as timer:
            pass

        assert timer.operation_name == "my_operation"

    def test_timed_operation_restores_on_exception(self):
        """Test timed_operation still captures duration on exception."""
        try:
            with timed_operation("failing_op") as timer:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert timer.duration_ms is not None
