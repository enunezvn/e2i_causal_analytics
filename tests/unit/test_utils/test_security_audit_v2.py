"""
Comprehensive unit tests for src/utils/security_audit.py

Tests cover:
- SecurityEventType enum
- SecurityEventSeverity enum
- SecurityAuditEvent dataclass
- SecurityAuditService class
  - Core logging
  - Authentication events
  - Authorization events
  - Rate limiting events
  - API security events
  - Data access events
  - Admin events
  - Query methods
- Singleton pattern
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import pytest

from src.utils.security_audit import (
    SecurityAuditEvent,
    SecurityAuditService,
    SecurityEventSeverity,
    SecurityEventType,
    get_security_audit_service,
    reset_security_audit_service,
)


# =============================================================================
# Test Enums
# =============================================================================


def test_security_event_type_auth_events():
    """Test authentication event types."""
    assert SecurityEventType.AUTH_LOGIN_SUCCESS.value == "auth.login.success"
    assert SecurityEventType.AUTH_LOGIN_FAILURE.value == "auth.login.failure"
    assert SecurityEventType.AUTH_LOGOUT.value == "auth.logout"
    assert SecurityEventType.AUTH_TOKEN_INVALID.value == "auth.token.invalid"


def test_security_event_type_authz_events():
    """Test authorization event types."""
    assert SecurityEventType.AUTHZ_ACCESS_DENIED.value == "authz.access.denied"
    assert (
        SecurityEventType.AUTHZ_PRIVILEGE_ESCALATION.value
        == "authz.privilege.escalation"
    )


def test_security_event_type_rate_limit_events():
    """Test rate limiting event types."""
    assert SecurityEventType.RATE_LIMIT_WARNING.value == "rate_limit.warning"
    assert SecurityEventType.RATE_LIMIT_EXCEEDED.value == "rate_limit.exceeded"
    assert SecurityEventType.RATE_LIMIT_BLOCKED.value == "rate_limit.blocked"


def test_security_event_type_api_security_events():
    """Test API security event types."""
    assert SecurityEventType.API_INVALID_REQUEST.value == "api.invalid_request"
    assert (
        SecurityEventType.API_SUSPICIOUS_ACTIVITY.value == "api.suspicious_activity"
    )
    assert SecurityEventType.API_INJECTION_ATTEMPT.value == "api.injection_attempt"


def test_security_event_severity_levels():
    """Test severity level enum."""
    assert SecurityEventSeverity.DEBUG.value == "debug"
    assert SecurityEventSeverity.INFO.value == "info"
    assert SecurityEventSeverity.WARNING.value == "warning"
    assert SecurityEventSeverity.ERROR.value == "error"
    assert SecurityEventSeverity.CRITICAL.value == "critical"


# =============================================================================
# Test SecurityAuditEvent Dataclass
# =============================================================================


def test_security_audit_event_creation():
    """Test SecurityAuditEvent creation."""
    event = SecurityAuditEvent(
        event_id=UUID("12345678-1234-5678-1234-567812345678"),
        event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
        severity=SecurityEventSeverity.INFO,
        timestamp=datetime.now(timezone.utc),
        message="Test event",
        user_id="user123",
        client_ip="192.168.1.1",
    )

    assert event.event_type == SecurityEventType.AUTH_LOGIN_SUCCESS
    assert event.severity == SecurityEventSeverity.INFO
    assert event.user_id == "user123"
    assert event.client_ip == "192.168.1.1"


def test_security_audit_event_to_dict():
    """Test SecurityAuditEvent.to_dict()."""
    timestamp = datetime.now(timezone.utc)
    event_id = UUID("12345678-1234-5678-1234-567812345678")

    event = SecurityAuditEvent(
        event_id=event_id,
        event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
        severity=SecurityEventSeverity.INFO,
        timestamp=timestamp,
        message="Test event",
        user_id="user123",
        client_ip="192.168.1.1",
        metadata={"key": "value"},
    )

    event_dict = event.to_dict()

    assert event_dict["event_id"] == str(event_id)
    assert event_dict["event_type"] == "auth.login.success"
    assert event_dict["severity"] == "info"
    assert event_dict["message"] == "Test event"
    assert event_dict["user_id"] == "user123"
    assert event_dict["metadata"] == {"key": "value"}
    # None values should be filtered out
    assert "error_code" not in event_dict


def test_security_audit_event_to_json():
    """Test SecurityAuditEvent.to_json()."""
    event = SecurityAuditEvent(
        event_id=UUID("12345678-1234-5678-1234-567812345678"),
        event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
        severity=SecurityEventSeverity.INFO,
        timestamp=datetime.now(timezone.utc),
        message="Test event",
    )

    json_str = event.to_json()

    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["message"] == "Test event"


# =============================================================================
# Test SecurityAuditService - Initialization
# =============================================================================


def test_security_audit_service_init_default():
    """Test SecurityAuditService default initialization."""
    service = SecurityAuditService()

    assert service.db is None
    assert service.log_file is None
    assert service.log_to_stdout is True
    assert service.min_severity == SecurityEventSeverity.INFO
    assert len(service._in_memory_log) == 0


def test_security_audit_service_init_with_params():
    """Test SecurityAuditService initialization with parameters."""
    mock_db = MagicMock()

    service = SecurityAuditService(
        supabase_client=mock_db,
        log_file="/tmp/test.log",
        log_to_stdout=False,
        min_severity=SecurityEventSeverity.WARNING,
    )

    assert service.db == mock_db
    assert service.log_file == "/tmp/test.log"
    assert service.log_to_stdout is False
    assert service.min_severity == SecurityEventSeverity.WARNING


# =============================================================================
# Test SecurityAuditService - Core Logging
# =============================================================================


def test_log_event_to_memory():
    """Test logging event to in-memory log."""
    service = SecurityAuditService(log_to_stdout=False)

    event = SecurityAuditEvent(
        event_id=UUID("12345678-1234-5678-1234-567812345678"),
        event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
        severity=SecurityEventSeverity.INFO,
        timestamp=datetime.now(timezone.utc),
        message="Test event",
    )

    event_id = service.log_event(event)

    assert event_id == event.event_id
    assert len(service._in_memory_log) == 1
    assert service._in_memory_log[0] == event


def test_log_event_respects_min_severity():
    """Test that events below min_severity are not logged."""
    service = SecurityAuditService(
        log_to_stdout=False,
        min_severity=SecurityEventSeverity.WARNING,
    )

    # Info event should be filtered
    info_event = SecurityAuditEvent(
        event_id=UUID("12345678-1234-5678-1234-567812345678"),
        event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
        severity=SecurityEventSeverity.INFO,
        timestamp=datetime.now(timezone.utc),
        message="Info event",
    )

    service.log_event(info_event)
    assert len(service._in_memory_log) == 0

    # Warning event should be logged
    warning_event = SecurityAuditEvent(
        event_id=UUID("12345678-1234-5678-1234-567812345679"),
        event_type=SecurityEventType.AUTH_LOGIN_FAILURE,
        severity=SecurityEventSeverity.WARNING,
        timestamp=datetime.now(timezone.utc),
        message="Warning event",
    )

    service.log_event(warning_event)
    assert len(service._in_memory_log) == 1


def test_log_event_memory_limit():
    """Test in-memory log circular buffer."""
    service = SecurityAuditService(log_to_stdout=False)

    # Add 11,000 events (should trigger pruning at 10,000)
    for i in range(11000):
        event = SecurityAuditEvent(
            event_id=UUID(f"00000000-0000-0000-0000-{i:012d}"),
            event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
            severity=SecurityEventSeverity.INFO,
            timestamp=datetime.now(timezone.utc),
            message=f"Event {i}",
        )
        service.log_event(event)

    # After adding 10001 events, prunes to 5000, then adds 999 more = 5999
    assert len(service._in_memory_log) == 5999


def test_log_event_to_file():
    """Test logging event to file."""
    with NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        log_file = f.name

    try:
        service = SecurityAuditService(
            log_file=log_file,
            log_to_stdout=False,
        )

        event = SecurityAuditEvent(
            event_id=UUID("12345678-1234-5678-1234-567812345678"),
            event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
            severity=SecurityEventSeverity.INFO,
            timestamp=datetime.now(timezone.utc),
            message="Test event",
        )

        service.log_event(event)

        # Read file and verify
        with open(log_file, "r") as f:
            line = f.readline()
            logged_event = json.loads(line)
            assert logged_event["message"] == "Test event"
    finally:
        Path(log_file).unlink(missing_ok=True)


def test_log_event_to_database():
    """Test logging event to database."""
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = None
    mock_db.table.return_value = mock_table

    service = SecurityAuditService(
        supabase_client=mock_db,
        log_to_stdout=False,
    )

    event = SecurityAuditEvent(
        event_id=UUID("12345678-1234-5678-1234-567812345678"),
        event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
        severity=SecurityEventSeverity.INFO,
        timestamp=datetime.now(timezone.utc),
        message="Test event",
    )

    service.log_event(event)

    mock_db.table.assert_called_once_with("security_audit_log")
    mock_table.insert.assert_called_once()


# =============================================================================
# Test SecurityAuditService - Authentication Methods
# =============================================================================


def test_log_auth_success():
    """Test logging successful authentication."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_auth_success(
        user_id="user123",
        user_email="test@example.com",
        client_ip="192.168.1.1",
    )

    assert isinstance(event_id, UUID)
    assert len(service._in_memory_log) == 1

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.AUTH_LOGIN_SUCCESS
    assert event.severity == SecurityEventSeverity.INFO
    assert event.user_id == "user123"
    assert event.user_email == "test@example.com"


def test_log_auth_failure():
    """Test logging failed authentication."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_auth_failure(
        user_email="test@example.com",
        client_ip="192.168.1.1",
        reason="Invalid password",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.AUTH_LOGIN_FAILURE
    assert event.severity == SecurityEventSeverity.WARNING
    assert event.error_details == "Invalid password"


def test_log_token_invalid():
    """Test logging invalid token."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_token_invalid(
        client_ip="192.168.1.1",
        endpoint="/api/data",
        reason="Token expired",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.AUTH_TOKEN_INVALID
    assert event.severity == SecurityEventSeverity.WARNING


# =============================================================================
# Test SecurityAuditService - Authorization Methods
# =============================================================================


def test_log_access_denied():
    """Test logging access denied event."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_access_denied(
        user_id="user123",
        user_email="test@example.com",
        resource_type="patient_data",
        resource_id="PAT123",
        action_attempted="read",
        client_ip="192.168.1.1",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.AUTHZ_ACCESS_DENIED
    assert event.severity == SecurityEventSeverity.WARNING
    assert event.resource_type == "patient_data"
    assert event.resource_id == "PAT123"


# =============================================================================
# Test SecurityAuditService - Rate Limiting Methods
# =============================================================================


def test_log_rate_limit_exceeded():
    """Test logging rate limit exceeded."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_rate_limit_exceeded(
        client_ip="192.168.1.1",
        endpoint="/api/query",
        limit=100,
        current=101,
        window_seconds=60,
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED
    assert event.severity == SecurityEventSeverity.WARNING
    assert event.metadata["limit"] == 100
    assert event.metadata["current"] == 101


def test_log_rate_limit_blocked():
    """Test logging rate limit block."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_rate_limit_blocked(
        client_ip="192.168.1.1",
        endpoint="/api/query",
        block_duration_seconds=300,
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.RATE_LIMIT_BLOCKED
    assert event.severity == SecurityEventSeverity.ERROR


# =============================================================================
# Test SecurityAuditService - API Security Methods
# =============================================================================


def test_log_suspicious_activity():
    """Test logging suspicious activity."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_suspicious_activity(
        client_ip="192.168.1.1",
        endpoint="/api/admin",
        description="Multiple failed login attempts",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.API_SUSPICIOUS_ACTIVITY
    assert event.severity == SecurityEventSeverity.ERROR


def test_log_injection_attempt():
    """Test logging injection attempt."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_injection_attempt(
        client_ip="192.168.1.1",
        endpoint="/api/query",
        injection_type="sql",
        payload_preview="SELECT * FROM users WHERE 1=1",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.API_INJECTION_ATTEMPT
    assert event.severity == SecurityEventSeverity.CRITICAL
    assert event.metadata["injection_type"] == "sql"


def test_log_cors_violation():
    """Test logging CORS violation."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_cors_violation(
        client_ip="192.168.1.1",
        origin="https://evil.com",
        endpoint="/api/data",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.API_CORS_VIOLATION
    assert event.severity == SecurityEventSeverity.WARNING


# =============================================================================
# Test SecurityAuditService - Data Access Methods
# =============================================================================


def test_log_sensitive_data_access():
    """Test logging sensitive data access."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_sensitive_data_access(
        user_id="user123",
        resource_type="patient_data",
        resource_id="PAT123",
        data_classification="pii",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.DATA_SENSITIVE_ACCESS
    assert event.severity == SecurityEventSeverity.INFO
    assert event.metadata["data_classification"] == "pii"


def test_log_data_export():
    """Test logging data export."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_data_export(
        user_id="user123",
        export_type="csv",
        record_count=1000,
        destination="s3://bucket/data.csv",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.DATA_EXPORT
    assert event.metadata["record_count"] == 1000


# =============================================================================
# Test SecurityAuditService - Admin Methods
# =============================================================================


def test_log_config_change():
    """Test logging configuration change."""
    service = SecurityAuditService(log_to_stdout=False)

    event_id = service.log_config_change(
        user_id="admin123",
        config_key="rate_limit.max_requests",
        old_value="100",
        new_value="200",
    )

    assert isinstance(event_id, UUID)

    event = service._in_memory_log[0]
    assert event.event_type == SecurityEventType.ADMIN_CONFIG_CHANGE
    assert event.severity == SecurityEventSeverity.WARNING
    assert event.metadata["config_key"] == "rate_limit.max_requests"


# =============================================================================
# Test SecurityAuditService - Query Methods
# =============================================================================


def test_get_recent_events():
    """Test getting recent events."""
    service = SecurityAuditService(log_to_stdout=False)

    # Log multiple events
    for i in range(5):
        service.log_auth_success(
            user_id=f"user{i}",
            user_email=f"user{i}@example.com",
        )

    recent = service.get_recent_events(limit=3)

    assert len(recent) == 3
    # Should be in reverse chronological order
    assert recent[0].user_id == "user4"


def test_get_recent_events_filtered():
    """Test getting recent events with filters."""
    service = SecurityAuditService(log_to_stdout=False)

    # Log mixed events
    service.log_auth_success(user_id="user1", user_email="user1@example.com")
    service.log_auth_failure(user_email="user2@example.com", reason="Bad password")
    service.log_auth_success(user_id="user3", user_email="user3@example.com")

    # Filter by event type
    login_successes = service.get_recent_events(
        event_types=[SecurityEventType.AUTH_LOGIN_SUCCESS]
    )

    assert len(login_successes) == 2

    # Filter by severity
    warnings = service.get_recent_events(min_severity=SecurityEventSeverity.WARNING)

    assert len(warnings) == 1
    assert warnings[0].event_type == SecurityEventType.AUTH_LOGIN_FAILURE


def test_get_events_by_user():
    """Test getting events for a specific user."""
    service = SecurityAuditService(log_to_stdout=False)

    service.log_auth_success(user_id="user1", user_email="user1@example.com")
    service.log_auth_success(user_id="user2", user_email="user2@example.com")
    service.log_auth_success(user_id="user1", user_email="user1@example.com")

    user1_events = service.get_events_by_user("user1")

    assert len(user1_events) == 2


def test_get_events_by_ip():
    """Test getting events from a specific IP."""
    service = SecurityAuditService(log_to_stdout=False)

    service.log_auth_success(user_id="user1", user_email="u1@example.com", client_ip="192.168.1.1")
    service.log_auth_success(user_id="user2", user_email="u2@example.com", client_ip="192.168.1.2")
    service.log_auth_success(user_id="user3", user_email="u3@example.com", client_ip="192.168.1.1")

    ip_events = service.get_events_by_ip("192.168.1.1")

    assert len(ip_events) == 2


def test_count_events_by_type():
    """Test counting events by type."""
    service = SecurityAuditService(log_to_stdout=False)

    service.log_auth_success(user_id="user1", user_email="u1@example.com")
    service.log_auth_success(user_id="user2", user_email="u2@example.com")
    service.log_auth_failure(user_email="u3@example.com", reason="Bad password")

    counts = service.count_events_by_type()

    assert counts["auth.login.success"] == 2
    assert counts["auth.login.failure"] == 1


# =============================================================================
# Test Singleton Pattern
# =============================================================================


def test_get_security_audit_service_singleton():
    """Test singleton pattern for get_security_audit_service."""
    try:
        # Reset first
        reset_security_audit_service()

        service1 = get_security_audit_service()
        service2 = get_security_audit_service()

        assert service1 is service2
    finally:
        # Clean up after test
        reset_security_audit_service()


def test_reset_security_audit_service():
    """Test resetting the singleton."""
    # Get instance
    service1 = get_security_audit_service()

    # Reset
    reset_security_audit_service()

    # Get new instance
    service2 = get_security_audit_service()

    # Should be different instances
    assert service1 is not service2
