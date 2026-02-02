"""
E2I Causal Analytics - API Error Handling
=========================================

Comprehensive typed error hierarchy for API responses with detailed context.
Provides actionable error messages that help users and developers understand
what went wrong and how to fix it.

Phase 3 - Type Safety Enhancement

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import traceback
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class ErrorCategory(str, Enum):
    """Categories of errors for classification and routing."""

    # Client errors (4xx)
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    CONFLICT = "conflict"

    # Server errors (5xx)
    INTERNAL = "internal"
    AGENT_ERROR = "agent_error"
    DEPENDENCY_ERROR = "dependency_error"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"


class ErrorSeverity(str, Enum):
    """Severity levels for error tracking and alerting."""

    LOW = "low"  # User-correctable, no action needed
    MEDIUM = "medium"  # May need investigation
    HIGH = "high"  # Requires attention
    CRITICAL = "critical"  # Immediate attention required


class E2IError(Exception):
    """
    Base exception for all E2I API errors.

    Provides structured error responses with:
    - Unique error ID for tracking
    - Category and severity classification
    - Detailed context for debugging
    - Suggested actions for resolution
    - Timestamp for correlation

    All custom exceptions inherit from this class.
    """

    # Default HTTP status code for this error type
    status_code: int = 500
    category: ErrorCategory = ErrorCategory.INTERNAL
    severity: ErrorSeverity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        suggested_action: Optional[str] = None,
        error_id: Optional[str] = None,
        include_trace: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        self.suggested_action = suggested_action
        self.error_id = error_id or str(uuid4())[:8]  # Short ID for user reference
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.include_trace = include_trace

        # Capture stack trace if requested or in debug mode
        if include_trace and original_error:
            self._traceback = "".join(
                traceback.format_exception(
                    type(original_error), original_error, original_error.__traceback__
                )
            )
        else:
            self._traceback = None

    def to_dict(self, include_debug: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON API response.

        Args:
            include_debug: Include stack trace and internal details

        Returns:
            Structured error response dictionary
        """
        result = {
            "error": self.__class__.__name__,
            "error_id": self.error_id,
            "category": self.category.value,
            "message": self.message,
            "timestamp": self.timestamp,
        }

        if self.suggested_action:
            result["suggested_action"] = self.suggested_action

        if self.details:
            result["details"] = self.details

        if include_debug:
            result["severity"] = self.severity.value
            if self.original_error:
                result["original_error"] = str(self.original_error)
            if self._traceback:
                result["traceback"] = self._traceback

        return result


# =============================================================================
# VALIDATION ERRORS (400)
# =============================================================================


class ValidationError(E2IError):
    """
    Raised when request validation fails.

    Examples:
    - Invalid request body schema
    - Missing required fields
    - Invalid field values
    """

    status_code = 400
    category = ErrorCategory.VALIDATION
    severity = ErrorSeverity.LOW

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraints: Optional[List[str]] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["provided_value"] = str(value)[:100]  # Truncate for safety
        if constraints:
            details["constraints"] = constraints

        # Allow subclasses to override suggested_action
        kwargs.setdefault(
            "suggested_action", "Check the API documentation for valid request format."
        )

        super().__init__(
            message,
            details=details,
            **kwargs,
        )


class SchemaValidationError(ValidationError):
    """Raised when request doesn't match expected schema."""

    def __init__(
        self,
        message: str,
        *,
        schema_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if schema_errors:
            details["schema_errors"] = schema_errors

        # Allow override of suggested_action
        kwargs.setdefault("suggested_action", "Review the schema errors and correct your request.")

        super().__init__(
            message,
            details=details,
            **kwargs,
        )


# =============================================================================
# AUTHENTICATION ERRORS (401)
# =============================================================================


class AuthenticationError(E2IError):
    """
    Raised when authentication fails.

    Examples:
    - Missing token
    - Expired token
    - Invalid token
    """

    status_code = 401
    category = ErrorCategory.AUTHENTICATION
    severity = ErrorSeverity.LOW

    def __init__(self, message: str = "Authentication required", **kwargs):
        # Allow subclasses to override suggested_action
        kwargs.setdefault("suggested_action", "Provide a valid authentication token.")

        super().__init__(
            message,
            **kwargs,
        )


class TokenExpiredError(AuthenticationError):
    """Raised when the authentication token has expired."""

    def __init__(self, **kwargs):
        kwargs.setdefault("suggested_action", "Refresh your authentication token and retry.")
        super().__init__(
            "Authentication token has expired",
            **kwargs,
        )


class InvalidTokenError(AuthenticationError):
    """Raised when the authentication token is invalid."""

    def __init__(self, reason: Optional[str] = None, **kwargs):
        message = "Invalid authentication token"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, **kwargs)


# =============================================================================
# AUTHORIZATION ERRORS (403)
# =============================================================================


class AuthorizationError(E2IError):
    """
    Raised when user lacks permission for the requested operation.

    Examples:
    - Insufficient role
    - Resource not owned by user
    - Feature not enabled
    """

    status_code = 403
    category = ErrorCategory.AUTHORIZATION
    severity = ErrorSeverity.LOW

    def __init__(
        self,
        message: str = "Access denied",
        *,
        required_permission: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if required_permission:
            details["required_permission"] = required_permission

        super().__init__(
            message,
            details=details,
            suggested_action="Contact your administrator if you need access.",
            **kwargs,
        )


# =============================================================================
# NOT FOUND ERRORS (404)
# =============================================================================


class NotFoundError(E2IError):
    """
    Raised when a requested resource doesn't exist.
    """

    status_code = 404
    category = ErrorCategory.NOT_FOUND
    severity = ErrorSeverity.LOW

    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        **kwargs,
    ):
        message = f"{resource_type} not found"
        if resource_id:
            message = f"{resource_type} '{resource_id}' not found"

        details = kwargs.pop("details", {})
        details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        # Allow subclasses to override suggested_action
        suggested_action = kwargs.pop(
            "suggested_action", f"Verify the {resource_type.lower()} identifier is correct."
        )

        super().__init__(
            message,
            details=details,
            suggested_action=suggested_action,
            **kwargs,
        )


class EndpointNotFoundError(NotFoundError):
    """Raised when the requested API endpoint doesn't exist."""

    def __init__(self, path: str, **kwargs):
        super().__init__(
            "Endpoint",
            resource_id=path,
            suggested_action="Check the API documentation at /api/docs for available endpoints.",
            **kwargs,
        )


# =============================================================================
# RATE LIMITING ERRORS (429)
# =============================================================================


class RateLimitError(E2IError):
    """Raised when rate limit is exceeded."""

    status_code = 429
    category = ErrorCategory.RATE_LIMITED
    severity = ErrorSeverity.LOW

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        message = f"Rate limit exceeded: {limit} requests per {window_seconds} seconds"

        details = kwargs.pop("details", {})
        details["limit"] = limit
        details["window_seconds"] = window_seconds
        if retry_after:
            details["retry_after_seconds"] = retry_after

        suggested = f"Wait {retry_after or window_seconds} seconds before retrying."

        super().__init__(
            message,
            details=details,
            suggested_action=suggested,
            **kwargs,
        )


# =============================================================================
# AGENT ERRORS (500)
# =============================================================================


class AgentError(E2IError):
    """
    Raised when an agent fails during execution.

    Provides detailed context about:
    - Which agent failed
    - What operation it was performing
    - What inputs it was processing
    """

    status_code = 500
    category = ErrorCategory.AGENT_ERROR
    severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        *,
        agent_name: str,
        operation: Optional[str] = None,
        agent_tier: Optional[int] = None,
        input_summary: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["agent_name"] = agent_name

        if operation:
            details["operation"] = operation
        if agent_tier is not None:
            details["agent_tier"] = agent_tier
        if input_summary:
            details["input_summary"] = input_summary[:200]  # Truncate

        # Allow subclasses to override suggested_action
        kwargs.setdefault(
            "suggested_action",
            f"The {agent_name} agent encountered an issue. "
            "Try rephrasing your request or contact support if the problem persists.",
        )

        super().__init__(
            message,
            details=details,
            **kwargs,
        )


class AgentTimeoutError(AgentError):
    """Raised when an agent times out."""

    category = ErrorCategory.TIMEOUT

    def __init__(
        self,
        agent_name: str,
        timeout_seconds: float,
        operation: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["timeout_seconds"] = timeout_seconds

        message = f"Agent '{agent_name}' timed out after {timeout_seconds}s"
        if operation:
            message = f"Agent '{agent_name}' timed out during {operation} after {timeout_seconds}s"

        # Allow subclasses/callers to override suggested_action
        kwargs.setdefault(
            "suggested_action",
            "The operation took too long. Try simplifying your request "
            "or breaking it into smaller parts.",
        )

        super().__init__(
            message,
            agent_name=agent_name,
            operation=operation,
            details=details,
            **kwargs,
        )


class AgentConfigurationError(AgentError):
    """Raised when an agent is misconfigured."""

    category = ErrorCategory.CONFIGURATION
    severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        agent_name: str,
        config_issue: str,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        details["config_issue"] = config_issue

        # Allow subclasses/callers to override suggested_action
        kwargs.setdefault(
            "suggested_action", "Contact support - this is a system configuration issue."
        )

        super().__init__(
            f"Agent '{agent_name}' configuration error: {config_issue}",
            agent_name=agent_name,
            details=details,
            **kwargs,
        )


class OrchestratorError(AgentError):
    """Raised when the orchestrator fails to route or coordinate agents."""

    severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        *,
        dispatched_agents: Optional[List[str]] = None,
        failed_agents: Optional[List[str]] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if dispatched_agents:
            details["dispatched_agents"] = dispatched_agents
        if failed_agents:
            details["failed_agents"] = failed_agents

        super().__init__(
            message,
            agent_name="orchestrator",
            agent_tier=1,
            details=details,
            **kwargs,
        )


# =============================================================================
# DEPENDENCY ERRORS (503)
# =============================================================================


class DependencyError(E2IError):
    """
    Raised when an external dependency fails.

    Examples:
    - Database unavailable
    - Redis connection lost
    - External API failure
    """

    status_code = 503
    category = ErrorCategory.DEPENDENCY_ERROR
    severity = ErrorSeverity.HIGH

    def __init__(
        self,
        dependency: str,
        operation: Optional[str] = None,
        **kwargs,
    ):
        message = f"Dependency '{dependency}' is unavailable"
        if operation:
            message = f"Dependency '{dependency}' failed during {operation}"

        details = kwargs.pop("details", {})
        details["dependency"] = dependency
        if operation:
            details["operation"] = operation

        super().__init__(
            message,
            details=details,
            suggested_action=(
                "The service is experiencing issues. "
                "Please try again in 30 seconds. "
                "If the problem persists, contact support with error ID."
            ),
            **kwargs,
        )


class DatabaseError(DependencyError):
    """Raised when database operations fail."""

    def __init__(
        self,
        operation: str,
        table: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if table:
            details["table"] = table

        super().__init__(
            "database",
            operation=operation,
            details=details,
            **kwargs,
        )


class CacheError(DependencyError):
    """Raised when cache operations fail."""

    severity = ErrorSeverity.MEDIUM  # Cache failures are less critical

    def __init__(self, operation: str, **kwargs):
        super().__init__("cache", operation=operation, **kwargs)


class ExternalServiceError(DependencyError):
    """Raised when an external service fails."""

    def __init__(
        self,
        service: str,
        operation: Optional[str] = None,
        response_status: Optional[int] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if response_status:
            details["response_status"] = response_status

        super().__init__(
            service,
            operation=operation,
            details=details,
            **kwargs,
        )


# =============================================================================
# TIMEOUT ERRORS (504)
# =============================================================================


class TimeoutError(E2IError):
    """Raised when an operation times out."""

    status_code = 504
    category = ErrorCategory.TIMEOUT
    severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        **kwargs,
    ):
        message = f"Operation '{operation}' timed out after {timeout_seconds}s"

        details = kwargs.pop("details", {})
        details["operation"] = operation
        details["timeout_seconds"] = timeout_seconds

        super().__init__(
            message,
            details=details,
            suggested_action="The request took too long. Try again or simplify your request.",
            **kwargs,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def error_response(
    error: E2IError,
    include_debug: bool = False,
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.

    Args:
        error: The E2IError instance
        include_debug: Include debug information (stack traces, etc.)

    Returns:
        Dictionary ready for JSONResponse
    """
    return error.to_dict(include_debug=include_debug)


def wrap_exception(
    exc: Exception,
    agent_name: Optional[str] = None,
    operation: Optional[str] = None,
    include_trace: bool = False,
) -> E2IError:
    """
    Wrap a generic exception in an appropriate E2IError.

    Intelligently categorizes exceptions based on type and message.

    Args:
        exc: The original exception
        agent_name: Optional agent name if this occurred in an agent
        operation: Optional operation description
        include_trace: Include stack trace in error details

    Returns:
        An appropriate E2IError subclass
    """
    # Already an E2IError - return as-is
    if isinstance(exc, E2IError):
        return exc

    exc_str = str(exc).lower()
    exc_type = type(exc).__name__

    # Timeout detection
    if "timeout" in exc_str or "timed out" in exc_str:
        if agent_name:
            return AgentTimeoutError(
                agent_name=agent_name,
                timeout_seconds=30.0,  # Default timeout
                operation=operation,
                original_error=exc,
                include_trace=include_trace,
            )
        return TimeoutError(
            operation=operation or "request",
            timeout_seconds=30.0,
            original_error=exc,
            include_trace=include_trace,
        )

    # Connection/dependency errors
    if any(kw in exc_str for kw in ["connection", "connect", "unreachable", "unavailable"]):
        dependency = "unknown"
        if "redis" in exc_str:
            dependency = "redis"
        elif "postgres" in exc_str or "database" in exc_str or "supabase" in exc_str:
            dependency = "database"
        elif "falkordb" in exc_str:
            dependency = "falkordb"

        return DependencyError(
            dependency=dependency,
            operation=operation,
            original_error=exc,
            include_trace=include_trace,
        )

    # Validation errors
    if "validation" in exc_str or exc_type == "ValidationError":
        return ValidationError(
            str(exc),
            original_error=exc,
            include_trace=include_trace,
        )

    # Agent error (if agent context provided)
    if agent_name:
        return AgentError(
            f"Agent error: {exc}",
            agent_name=agent_name,
            operation=operation,
            original_error=exc,
            include_trace=include_trace,
        )

    # Default to internal error
    return E2IError(
        f"An unexpected error occurred: {exc_type}",
        original_error=exc,
        suggested_action=(
            "Please try again in a few moments. "
            "If the issue continues, try simplifying your request or contact support with the error ID."
        ),
        include_trace=include_trace,
    )
