"""
E2I Causal Analytics - Security Audit Logging Service

Provides centralized security audit logging for:
- Authentication events (login, logout, token refresh, failures)
- Authorization events (access denied, privilege escalation attempts)
- Rate limiting events (threshold hits, blocks)
- API security events (invalid tokens, suspicious requests)
- Data access events (sensitive data access, exports)

Complements the agent audit chain (audit_chain.py) which tracks agent actions.

Version: 1.0
Date: January 2026
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Categories of security events to audit."""

    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTH_TOKEN_INVALID = "auth.token.invalid"
    AUTH_TOKEN_EXPIRED = "auth.token.expired"
    AUTH_MFA_SUCCESS = "auth.mfa.success"
    AUTH_MFA_FAILURE = "auth.mfa.failure"

    # Authorization events
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_PRIVILEGE_ESCALATION = "authz.privilege.escalation"
    AUTHZ_ROLE_CHANGE = "authz.role.change"
    AUTHZ_PERMISSION_GRANTED = "authz.permission.granted"
    AUTHZ_PERMISSION_REVOKED = "authz.permission.revoked"

    # Rate limiting events
    RATE_LIMIT_WARNING = "rate_limit.warning"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    RATE_LIMIT_BLOCKED = "rate_limit.blocked"

    # API security events
    API_INVALID_REQUEST = "api.invalid_request"
    API_SUSPICIOUS_ACTIVITY = "api.suspicious_activity"
    API_INJECTION_ATTEMPT = "api.injection_attempt"
    API_CORS_VIOLATION = "api.cors_violation"

    # Data access events
    DATA_SENSITIVE_ACCESS = "data.sensitive.access"
    DATA_EXPORT = "data.export"
    DATA_BULK_QUERY = "data.bulk_query"
    DATA_PII_ACCESS = "data.pii.access"

    # Session events
    SESSION_CREATED = "session.created"
    SESSION_TERMINATED = "session.terminated"
    SESSION_HIJACK_ATTEMPT = "session.hijack_attempt"

    # Admin events
    ADMIN_CONFIG_CHANGE = "admin.config.change"
    ADMIN_USER_MODIFIED = "admin.user.modified"
    ADMIN_SYSTEM_ACCESS = "admin.system.access"


class SecurityEventSeverity(Enum):
    """Severity levels for security events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityAuditEvent:
    """
    A single security audit event.

    Captures all relevant context for security analysis and compliance.
    """

    event_id: UUID
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    timestamp: datetime
    message: str

    # Actor information
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_roles: Optional[List[str]] = None

    # Request context
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    http_method: Optional[str] = None

    # Additional context
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action_attempted: Optional[str] = None
    action_result: Optional[str] = None

    # Error details
    error_code: Optional[str] = None
    error_details: Optional[str] = None

    # Extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for storage/serialization."""
        data = {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "user_roles": self.user_roles,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "endpoint": self.endpoint,
            "http_method": self.http_method,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action_attempted": self.action_attempted,
            "action_result": self.action_result,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "metadata": self.metadata,
        }
        return {k: v for k, v in data.items() if v is not None}

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class SecurityAuditService:
    """
    Service for logging and managing security audit events.

    Supports multiple backends:
    - In-memory (for testing)
    - File-based (JSON lines)
    - Supabase (production database)
    - External logging (stdout for container logs)

    Usage:
        service = SecurityAuditService()

        # Log an authentication failure
        service.log_auth_failure(
            user_email="test@example.com",
            client_ip="192.168.1.1",
            reason="Invalid password"
        )

        # Log a rate limit event
        service.log_rate_limit(
            client_ip="192.168.1.1",
            endpoint="/api/query",
            limit=100,
            current=101
        )
    """

    def __init__(
        self,
        supabase_client=None,
        log_file: Optional[str] = None,
        log_to_stdout: bool = True,
        min_severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
    ):
        """
        Initialize the security audit service.

        Args:
            supabase_client: Optional Supabase client for database logging
            log_file: Optional file path for JSON line logging
            log_to_stdout: Whether to log to stdout (container logs)
            min_severity: Minimum severity level to log
        """
        self.db = supabase_client
        self.log_file = log_file
        self.log_to_stdout = log_to_stdout
        self.min_severity = min_severity
        self._in_memory_log: List[SecurityAuditEvent] = []

        # Severity ordering for filtering
        self._severity_order = {
            SecurityEventSeverity.DEBUG: 0,
            SecurityEventSeverity.INFO: 1,
            SecurityEventSeverity.WARNING: 2,
            SecurityEventSeverity.ERROR: 3,
            SecurityEventSeverity.CRITICAL: 4,
        }

    # =========================================================================
    # Core Logging
    # =========================================================================

    def log_event(self, event: SecurityAuditEvent) -> UUID:
        """
        Log a security audit event to all configured backends.

        Args:
            event: The security audit event to log

        Returns:
            The event ID
        """
        # Check severity threshold
        if self._severity_order[event.severity] < self._severity_order[self.min_severity]:
            return event.event_id

        # Always keep in memory (for recent events query)
        self._in_memory_log.append(event)
        if len(self._in_memory_log) > 10000:
            self._in_memory_log = self._in_memory_log[-5000:]

        # Log to stdout for container log aggregation
        if self.log_to_stdout:
            self._log_to_stdout(event)

        # Log to file if configured
        if self.log_file:
            self._log_to_file(event)

        # Log to database if configured
        if self.db:
            self._log_to_database(event)

        return event.event_id

    def _log_to_stdout(self, event: SecurityAuditEvent) -> None:
        """Log event to stdout in structured format."""
        log_data = {
            "log_type": "security_audit",
            **event.to_dict(),
        }
        # Use appropriate log level
        log_message = json.dumps(log_data, default=str)

        if event.severity == SecurityEventSeverity.CRITICAL:
            logger.critical(log_message)
        elif event.severity == SecurityEventSeverity.ERROR:
            logger.error(log_message)
        elif event.severity == SecurityEventSeverity.WARNING:
            logger.warning(log_message)
        elif event.severity == SecurityEventSeverity.DEBUG:
            logger.debug(log_message)
        else:
            logger.info(log_message)

    def _log_to_file(self, event: SecurityAuditEvent) -> None:
        """Append event to JSON lines file."""
        if self.log_file is None:
            return
        try:
            with open(self.log_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write security audit to file: {e}")

    def _log_to_database(self, event: SecurityAuditEvent) -> None:
        """Insert event into Supabase database."""
        try:
            self.db.table("security_audit_log").insert(event.to_dict()).execute()
        except Exception as e:
            logger.error(f"Failed to write security audit to database: {e}")
            # Don't raise - audit logging should not break the application

    # =========================================================================
    # Convenience Methods - Authentication
    # =========================================================================

    def log_auth_success(
        self,
        user_id: str,
        user_email: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Log a successful authentication."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
            severity=SecurityEventSeverity.INFO,
            timestamp=datetime.now(timezone.utc),
            message=f"Successful login for user {user_email}",
            user_id=user_id,
            user_email=user_email,
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            action_result="success",
            metadata=metadata or {},
        )
        return self.log_event(event)

    def log_auth_failure(
        self,
        user_email: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        reason: str = "Invalid credentials",
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Log a failed authentication attempt."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.AUTH_LOGIN_FAILURE,
            severity=SecurityEventSeverity.WARNING,
            timestamp=datetime.now(timezone.utc),
            message=f"Failed login attempt: {reason}",
            user_email=user_email,
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            action_result="failure",
            error_details=reason,
            metadata=metadata or {},
        )
        return self.log_event(event)

    def log_token_invalid(
        self,
        client_ip: Optional[str] = None,
        endpoint: Optional[str] = None,
        reason: str = "Invalid token",
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log an invalid token attempt."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.AUTH_TOKEN_INVALID,
            severity=SecurityEventSeverity.WARNING,
            timestamp=datetime.now(timezone.utc),
            message=f"Invalid token presented: {reason}",
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=request_id,
            error_details=reason,
        )
        return self.log_event(event)

    # =========================================================================
    # Convenience Methods - Authorization
    # =========================================================================

    def log_access_denied(
        self,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        resource_type: str = "unknown",
        resource_id: Optional[str] = None,
        action_attempted: str = "access",
        client_ip: Optional[str] = None,
        endpoint: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log an access denied event."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.AUTHZ_ACCESS_DENIED,
            severity=SecurityEventSeverity.WARNING,
            timestamp=datetime.now(timezone.utc),
            message=f"Access denied to {resource_type}/{resource_id or 'unknown'}",
            user_id=user_id,
            user_email=user_email,
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=request_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action_attempted=action_attempted,
            action_result="denied",
        )
        return self.log_event(event)

    # =========================================================================
    # Convenience Methods - Rate Limiting
    # =========================================================================

    def log_rate_limit_exceeded(
        self,
        client_ip: str,
        endpoint: str,
        limit: int,
        current: int,
        window_seconds: int = 60,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log a rate limit exceeded event."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SecurityEventSeverity.WARNING,
            timestamp=datetime.now(timezone.utc),
            message=f"Rate limit exceeded: {current}/{limit} requests in {window_seconds}s",
            user_id=user_id,
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=request_id,
            metadata={
                "limit": limit,
                "current": current,
                "window_seconds": window_seconds,
            },
        )
        return self.log_event(event)

    def log_rate_limit_blocked(
        self,
        client_ip: str,
        endpoint: str,
        block_duration_seconds: int,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log a request blocked due to rate limiting."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.RATE_LIMIT_BLOCKED,
            severity=SecurityEventSeverity.ERROR,
            timestamp=datetime.now(timezone.utc),
            message=f"Request blocked for {block_duration_seconds}s due to rate limit",
            user_id=user_id,
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=request_id,
            action_result="blocked",
            metadata={
                "block_duration_seconds": block_duration_seconds,
            },
        )
        return self.log_event(event)

    # =========================================================================
    # Convenience Methods - API Security
    # =========================================================================

    def log_suspicious_activity(
        self,
        client_ip: str,
        endpoint: str,
        description: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Log suspicious API activity."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.API_SUSPICIOUS_ACTIVITY,
            severity=SecurityEventSeverity.ERROR,
            timestamp=datetime.now(timezone.utc),
            message=f"Suspicious activity detected: {description}",
            user_id=user_id,
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=request_id,
            metadata=metadata or {},
        )
        return self.log_event(event)

    def log_injection_attempt(
        self,
        client_ip: str,
        endpoint: str,
        injection_type: str,  # sql, xss, command, etc.
        payload_preview: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log a potential injection attack attempt."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.API_INJECTION_ATTEMPT,
            severity=SecurityEventSeverity.CRITICAL,
            timestamp=datetime.now(timezone.utc),
            message=f"Potential {injection_type} injection attempt detected",
            user_id=user_id,
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=request_id,
            metadata={
                "injection_type": injection_type,
                "payload_preview": payload_preview[:200],  # Truncate for safety
            },
        )
        return self.log_event(event)

    def log_cors_violation(
        self,
        client_ip: str,
        origin: str,
        endpoint: str,
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log a CORS policy violation."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.API_CORS_VIOLATION,
            severity=SecurityEventSeverity.WARNING,
            timestamp=datetime.now(timezone.utc),
            message=f"CORS violation from origin: {origin}",
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=request_id,
            metadata={
                "origin": origin,
            },
        )
        return self.log_event(event)

    # =========================================================================
    # Convenience Methods - Data Access
    # =========================================================================

    def log_sensitive_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        data_classification: str = "sensitive",
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log access to sensitive data for compliance tracking."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.DATA_SENSITIVE_ACCESS,
            severity=SecurityEventSeverity.INFO,
            timestamp=datetime.now(timezone.utc),
            message=f"Sensitive data access: {resource_type}/{resource_id or 'bulk'}",
            user_id=user_id,
            client_ip=client_ip,
            request_id=request_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata={
                "data_classification": data_classification,
            },
        )
        return self.log_event(event)

    def log_data_export(
        self,
        user_id: str,
        export_type: str,
        record_count: int,
        destination: Optional[str] = None,
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log data export events for compliance."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.DATA_EXPORT,
            severity=SecurityEventSeverity.INFO,
            timestamp=datetime.now(timezone.utc),
            message=f"Data export: {export_type} ({record_count} records)",
            user_id=user_id,
            client_ip=client_ip,
            request_id=request_id,
            metadata={
                "export_type": export_type,
                "record_count": record_count,
                "destination": destination,
            },
        )
        return self.log_event(event)

    # =========================================================================
    # Convenience Methods - Admin Events
    # =========================================================================

    def log_config_change(
        self,
        user_id: str,
        config_key: str,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> UUID:
        """Log configuration changes."""
        event = SecurityAuditEvent(
            event_id=uuid4(),
            event_type=SecurityEventType.ADMIN_CONFIG_CHANGE,
            severity=SecurityEventSeverity.WARNING,
            timestamp=datetime.now(timezone.utc),
            message=f"Configuration change: {config_key}",
            user_id=user_id,
            client_ip=client_ip,
            request_id=request_id,
            metadata={
                "config_key": config_key,
                "old_value": old_value,
                "new_value": new_value,
            },
        )
        return self.log_event(event)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_recent_events(
        self,
        limit: int = 100,
        event_types: Optional[List[SecurityEventType]] = None,
        min_severity: Optional[SecurityEventSeverity] = None,
    ) -> List[SecurityAuditEvent]:
        """
        Get recent security events from in-memory log.

        Args:
            limit: Maximum number of events to return
            event_types: Filter by specific event types
            min_severity: Filter by minimum severity

        Returns:
            List of matching events (most recent first)
        """
        events = self._in_memory_log.copy()

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if min_severity:
            min_order = self._severity_order[min_severity]
            events = [e for e in events if self._severity_order[e.severity] >= min_order]

        # Return most recent first
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_events_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[SecurityAuditEvent]:
        """Get security events for a specific user."""
        return [e for e in self._in_memory_log if e.user_id == user_id][-limit:]

    def get_events_by_ip(
        self,
        client_ip: str,
        limit: int = 100,
    ) -> List[SecurityAuditEvent]:
        """Get security events from a specific IP address."""
        return [e for e in self._in_memory_log if e.client_ip == client_ip][-limit:]

    def count_events_by_type(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Count events by type for analytics.

        Args:
            since: Only count events after this timestamp

        Returns:
            Dictionary mapping event type to count
        """
        events = self._in_memory_log
        if since:
            events = [e for e in events if e.timestamp >= since]

        counts: Dict[str, int] = {}
        for event in events:
            key = event.event_type.value
            counts[key] = counts.get(key, 0) + 1

        return counts


# =============================================================================
# Singleton Instance
# =============================================================================

_security_audit_service: Optional[SecurityAuditService] = None


def get_security_audit_service() -> SecurityAuditService:
    """
    Get or create the singleton security audit service.

    The service is configured based on environment:
    - SECURITY_AUDIT_LOG_FILE: Path to JSON lines log file
    - SECURITY_AUDIT_TO_STDOUT: Whether to log to stdout (default: true)
    - SECURITY_AUDIT_MIN_SEVERITY: Minimum severity to log (default: info)
    """
    global _security_audit_service

    if _security_audit_service is None:
        log_file = os.environ.get("SECURITY_AUDIT_LOG_FILE")
        log_to_stdout = os.environ.get("SECURITY_AUDIT_TO_STDOUT", "true").lower() == "true"
        min_severity_str = os.environ.get("SECURITY_AUDIT_MIN_SEVERITY", "info")

        try:
            min_severity = SecurityEventSeverity(min_severity_str)
        except ValueError:
            min_severity = SecurityEventSeverity.INFO

        # Try to get Supabase client if configured
        supabase_client = None
        try:
            from src.api.deps import get_supabase

            supabase_client = get_supabase()
        except Exception:
            pass  # Running without database is fine

        _security_audit_service = SecurityAuditService(
            supabase_client=supabase_client,
            log_file=log_file,
            log_to_stdout=log_to_stdout,
            min_severity=min_severity,
        )

    return _security_audit_service


def reset_security_audit_service() -> None:
    """Reset the singleton for testing purposes."""
    global _security_audit_service
    _security_audit_service = None
