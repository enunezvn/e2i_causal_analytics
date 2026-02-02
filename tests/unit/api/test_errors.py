"""
Tests for E2I API Error Handling Module
======================================

Tests for the typed error hierarchy and error response formatting.

Phase 3 - Type Safety Enhancement
"""

import pytest

from src.api.errors import (
    AgentConfigurationError,
    AgentError,
    AgentTimeoutError,
    AuthenticationError,
    AuthorizationError,
    CacheError,
    DatabaseError,
    DependencyError,
    E2IError,
    EndpointNotFoundError,
    ErrorCategory,
    ErrorSeverity,
    ExternalServiceError,
    InvalidTokenError,
    NotFoundError,
    OrchestratorError,
    RateLimitError,
    SchemaValidationError,
    TimeoutError,
    TokenExpiredError,
    ValidationError,
    error_response,
    wrap_exception,
)

# =============================================================================
# BASE E2I ERROR TESTS
# =============================================================================


class TestE2IError:
    """Tests for the base E2IError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = E2IError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.status_code == 500
        assert error.category == ErrorCategory.INTERNAL
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.error_id is not None
        assert error.timestamp is not None

    def test_error_with_details(self):
        """Test error with additional details."""
        error = E2IError(
            "Operation failed",
            details={"key": "value", "count": 42},
            suggested_action="Try again later.",
        )

        assert error.details == {"key": "value", "count": 42}
        assert error.suggested_action == "Try again later."

    def test_error_with_original_error(self):
        """Test error wrapping original exception."""
        original = ValueError("Original error")
        error = E2IError(
            "Wrapped error",
            original_error=original,
        )

        assert error.original_error is original

    def test_to_dict_basic(self):
        """Test basic dictionary conversion."""
        error = E2IError("Test error")
        result = error.to_dict()

        assert result["error"] == "E2IError"
        assert result["category"] == "internal"
        assert result["message"] == "Test error"
        assert "error_id" in result
        assert "timestamp" in result
        assert "severity" not in result  # Not included without debug

    def test_to_dict_with_debug(self):
        """Test dictionary conversion with debug info."""
        original = ValueError("Original")
        error = E2IError(
            "Test error",
            original_error=original,
            include_trace=True,
        )
        result = error.to_dict(include_debug=True)

        assert result["severity"] == "medium"
        assert result["original_error"] == "Original"

    def test_custom_error_id(self):
        """Test custom error ID."""
        error = E2IError("Test", error_id="custom-123")
        assert error.error_id == "custom-123"


# =============================================================================
# VALIDATION ERROR TESTS
# =============================================================================


class TestValidationErrors:
    """Tests for validation-related errors."""

    def test_validation_error(self):
        """Test basic validation error."""
        error = ValidationError(
            "Invalid input",
            field="email",
            value="not-an-email",
        )

        assert error.status_code == 400
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.LOW
        assert error.details["field"] == "email"
        assert "not-an-email" in error.details["provided_value"]

    def test_validation_error_with_constraints(self):
        """Test validation error with constraints."""
        error = ValidationError(
            "Value out of range",
            field="age",
            value=150,
            constraints=["min: 0", "max: 120"],
        )

        assert error.details["constraints"] == ["min: 0", "max: 120"]

    def test_schema_validation_error(self):
        """Test schema validation error."""
        schema_errors = [
            {"field": "name", "message": "required"},
            {"field": "age", "message": "must be positive"},
        ]
        error = SchemaValidationError(
            "Schema validation failed",
            schema_errors=schema_errors,
        )

        assert error.status_code == 400
        assert error.details["schema_errors"] == schema_errors


# =============================================================================
# AUTHENTICATION ERROR TESTS
# =============================================================================


class TestAuthenticationErrors:
    """Tests for authentication-related errors."""

    def test_authentication_error(self):
        """Test basic authentication error."""
        error = AuthenticationError()

        assert error.status_code == 401
        assert error.category == ErrorCategory.AUTHENTICATION
        assert "Authentication required" in error.message

    def test_token_expired_error(self):
        """Test token expired error."""
        error = TokenExpiredError()

        assert error.status_code == 401
        assert "expired" in error.message.lower()
        assert "refresh" in error.suggested_action.lower()

    def test_invalid_token_error(self):
        """Test invalid token error."""
        error = InvalidTokenError(reason="malformed")

        assert error.status_code == 401
        assert "malformed" in error.message


# =============================================================================
# AUTHORIZATION ERROR TESTS
# =============================================================================


class TestAuthorizationErrors:
    """Tests for authorization-related errors."""

    def test_authorization_error(self):
        """Test basic authorization error."""
        error = AuthorizationError()

        assert error.status_code == 403
        assert error.category == ErrorCategory.AUTHORIZATION
        assert "Access denied" in error.message

    def test_authorization_error_with_permission(self):
        """Test authorization error with required permission."""
        error = AuthorizationError(
            "Cannot delete resource",
            required_permission="admin:delete",
        )

        assert error.details["required_permission"] == "admin:delete"


# =============================================================================
# NOT FOUND ERROR TESTS
# =============================================================================


class TestNotFoundErrors:
    """Tests for not-found errors."""

    def test_not_found_error(self):
        """Test basic not found error."""
        error = NotFoundError("KPI", "WS1-DQ-001")

        assert error.status_code == 404
        assert error.category == ErrorCategory.NOT_FOUND
        assert "KPI" in error.message
        assert "WS1-DQ-001" in error.message
        assert error.details["resource_type"] == "KPI"
        assert error.details["resource_id"] == "WS1-DQ-001"

    def test_not_found_error_without_id(self):
        """Test not found error without ID."""
        error = NotFoundError("User")

        assert "User not found" in error.message
        assert "resource_id" not in error.details

    def test_endpoint_not_found_error(self):
        """Test endpoint not found error."""
        error = EndpointNotFoundError("/api/nonexistent")

        assert error.status_code == 404
        assert "/api/nonexistent" in error.message
        assert "/api/docs" in error.suggested_action


# =============================================================================
# RATE LIMIT ERROR TESTS
# =============================================================================


class TestRateLimitErrors:
    """Tests for rate limiting errors."""

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError(
            limit=100,
            window_seconds=60,
            retry_after=30,
        )

        assert error.status_code == 429
        assert error.category == ErrorCategory.RATE_LIMITED
        assert error.details["limit"] == 100
        assert error.details["window_seconds"] == 60
        assert error.details["retry_after_seconds"] == 30
        assert "30" in error.suggested_action


# =============================================================================
# AGENT ERROR TESTS
# =============================================================================


class TestAgentErrors:
    """Tests for agent-related errors."""

    def test_agent_error(self):
        """Test basic agent error."""
        error = AgentError(
            "Analysis failed",
            agent_name="causal_impact",
            operation="effect_estimation",
            agent_tier=2,
        )

        assert error.status_code == 500
        assert error.category == ErrorCategory.AGENT_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert error.details["agent_name"] == "causal_impact"
        assert error.details["operation"] == "effect_estimation"
        assert error.details["agent_tier"] == 2

    def test_agent_timeout_error(self):
        """Test agent timeout error."""
        error = AgentTimeoutError(
            agent_name="prediction_synthesizer",
            timeout_seconds=120.0,
            operation="batch_prediction",
        )

        assert error.category == ErrorCategory.TIMEOUT
        assert error.details["timeout_seconds"] == 120.0
        assert "prediction_synthesizer" in error.message
        assert "120" in error.message

    def test_agent_configuration_error(self):
        """Test agent configuration error."""
        error = AgentConfigurationError(
            agent_name="drift_monitor",
            config_issue="Missing threshold configuration",
        )

        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.details["config_issue"] == "Missing threshold configuration"

    def test_orchestrator_error(self):
        """Test orchestrator error."""
        error = OrchestratorError(
            "Failed to route query",
            dispatched_agents=["causal_impact", "gap_analyzer"],
            failed_agents=["gap_analyzer"],
        )

        assert error.details["agent_name"] == "orchestrator"
        assert error.details["agent_tier"] == 1
        assert error.details["dispatched_agents"] == ["causal_impact", "gap_analyzer"]
        assert error.details["failed_agents"] == ["gap_analyzer"]


# =============================================================================
# DEPENDENCY ERROR TESTS
# =============================================================================


class TestDependencyErrors:
    """Tests for dependency-related errors."""

    def test_dependency_error(self):
        """Test basic dependency error."""
        error = DependencyError(
            dependency="redis",
            operation="cache_get",
        )

        assert error.status_code == 503
        assert error.category == ErrorCategory.DEPENDENCY_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert error.details["dependency"] == "redis"
        assert error.details["operation"] == "cache_get"

    def test_database_error(self):
        """Test database error."""
        error = DatabaseError(
            operation="insert",
            table="kpi_results",
        )

        assert error.details["dependency"] == "database"
        assert error.details["table"] == "kpi_results"

    def test_cache_error(self):
        """Test cache error."""
        error = CacheError(operation="set")

        assert error.severity == ErrorSeverity.MEDIUM  # Less severe
        assert error.details["dependency"] == "cache"

    def test_external_service_error(self):
        """Test external service error."""
        error = ExternalServiceError(
            service="openai",
            operation="embedding",
            response_status=429,
        )

        assert error.details["dependency"] == "openai"
        assert error.details["response_status"] == 429


# =============================================================================
# TIMEOUT ERROR TESTS
# =============================================================================


class TestTimeoutErrors:
    """Tests for timeout errors."""

    def test_timeout_error(self):
        """Test basic timeout error."""
        error = TimeoutError(
            operation="causal_analysis",
            timeout_seconds=30.0,
        )

        assert error.status_code == 504
        assert error.category == ErrorCategory.TIMEOUT
        assert error.details["operation"] == "causal_analysis"
        assert error.details["timeout_seconds"] == 30.0


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestErrorResponse:
    """Tests for error_response helper."""

    def test_error_response_basic(self):
        """Test basic error response."""
        error = ValidationError("Bad input")
        response = error_response(error)

        assert response["error"] == "ValidationError"
        assert response["message"] == "Bad input"
        assert "error_id" in response

    def test_error_response_with_debug(self):
        """Test error response with debug info."""
        original = ValueError("Original")
        error = E2IError("Wrapped", original_error=original)
        response = error_response(error, include_debug=True)

        assert "severity" in response
        assert response["original_error"] == "Original"


class TestWrapException:
    """Tests for wrap_exception helper."""

    def test_wrap_e2i_error_returns_same(self):
        """Test that E2IError is returned as-is."""
        original = ValidationError("Already E2I")
        wrapped = wrap_exception(original)

        assert wrapped is original

    def test_wrap_timeout_exception(self):
        """Test wrapping timeout exception."""
        original = Exception("Connection timed out")
        wrapped = wrap_exception(original)

        assert isinstance(wrapped, TimeoutError)

    def test_wrap_timeout_with_agent(self):
        """Test wrapping timeout with agent context."""
        original = Exception("Operation timed out")
        wrapped = wrap_exception(
            original,
            agent_name="causal_impact",
            operation="analysis",
        )

        assert isinstance(wrapped, AgentTimeoutError)
        assert wrapped.details["agent_name"] == "causal_impact"

    def test_wrap_connection_error(self):
        """Test wrapping connection error."""
        original = Exception("Redis connection refused")
        wrapped = wrap_exception(original)

        assert isinstance(wrapped, DependencyError)
        assert wrapped.details["dependency"] == "redis"

    def test_wrap_database_error(self):
        """Test wrapping database error."""
        original = Exception("PostgreSQL connection failed")
        wrapped = wrap_exception(original)

        assert isinstance(wrapped, DependencyError)
        assert wrapped.details["dependency"] == "database"

    def test_wrap_validation_exception(self):
        """Test wrapping validation exception."""
        original = Exception("Validation error: field required")
        wrapped = wrap_exception(original)

        assert isinstance(wrapped, ValidationError)

    def test_wrap_generic_with_agent(self):
        """Test wrapping generic exception with agent context."""
        original = Exception("Something broke")
        wrapped = wrap_exception(
            original,
            agent_name="explainer",
            operation="generate_explanation",
        )

        assert isinstance(wrapped, AgentError)
        assert wrapped.details["agent_name"] == "explainer"

    def test_wrap_generic_exception(self):
        """Test wrapping unknown exception."""
        original = RuntimeError("Unknown error")
        wrapped = wrap_exception(original)

        assert isinstance(wrapped, E2IError)
        assert wrapped.original_error is original

    def test_wrap_with_trace(self):
        """Test wrapping with stack trace."""
        try:
            raise ValueError("With trace")
        except ValueError as e:
            wrapped = wrap_exception(e, include_trace=True)
            wrapped.to_dict(include_debug=True)
            # Trace is captured when there's an original error
            assert wrapped.original_error is not None


# =============================================================================
# ERROR HIERARCHY TESTS
# =============================================================================


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_e2i_error(self):
        """Test that all custom errors inherit from E2IError."""
        error_classes = [
            ValidationError,
            SchemaValidationError,
            AuthenticationError,
            TokenExpiredError,
            InvalidTokenError,
            AuthorizationError,
            NotFoundError,
            EndpointNotFoundError,
            RateLimitError,
            AgentError,
            AgentTimeoutError,
            AgentConfigurationError,
            OrchestratorError,
            DependencyError,
            DatabaseError,
            CacheError,
            ExternalServiceError,
            TimeoutError,
        ]

        for error_class in error_classes:
            assert issubclass(error_class, E2IError), (
                f"{error_class.__name__} should inherit from E2IError"
            )

    def test_catching_all_errors(self):
        """Test that all errors can be caught with E2IError."""
        errors = [
            ValidationError("test"),
            AuthenticationError(),
            AgentError("test", agent_name="test"),
            DependencyError(dependency="test"),
        ]

        for error in errors:
            try:
                raise error
            except E2IError as e:
                assert e is error
            except Exception:
                pytest.fail(f"{type(error).__name__} was not caught by E2IError")
