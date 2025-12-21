"""
E2I Hybrid RAG - Custom Exceptions

This module defines custom exceptions for the hybrid RAG system.
All exceptions inherit from RAGError for easy catching.

Part of Phase 1, Checkpoint 1.1.
"""

from typing import Any, Dict, Optional


class RAGError(Exception):
    """
    Base exception for all RAG-related errors.

    All custom RAG exceptions inherit from this class,
    allowing callers to catch all RAG errors with a single except clause.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API responses."""
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }
        if self.original_error:
            result["original_error"] = str(self.original_error)
        return result


class ConfigurationError(RAGError):
    """
    Raised when RAG configuration is invalid.

    Examples:
    - Weights don't sum to 1.0
    - Missing required environment variables
    - Invalid timeout values
    """
    pass


class EmbeddingError(RAGError):
    """
    Raised when embedding generation fails.

    Examples:
    - OpenAI API rate limit
    - Invalid API key
    - Network timeout
    - Batch too large
    """

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message, details, original_error)
        self.model = model
        self.batch_size = batch_size
        if model:
            self.details["model"] = model
        if batch_size:
            self.details["batch_size"] = batch_size


class RetrieverError(RAGError):
    """
    Raised when retrieval operations fail.

    Examples:
    - All backends failed
    - Required source unavailable
    - Fusion error
    """

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message, details, original_error)
        self.source = source
        self.query = query
        if source:
            self.details["source"] = source
        if query:
            self.details["query"] = query[:100]  # Truncate for logging


class BackendTimeoutError(RetrieverError):
    """
    Raised when a search backend times out.

    Includes which backend timed out and the configured timeout.
    """

    def __init__(
        self,
        backend: str,
        timeout_ms: float,
        query: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        message = f"{backend} search timed out after {timeout_ms}ms"
        super().__init__(
            message=message,
            source=backend,
            query=query,
            details={"timeout_ms": timeout_ms},
            original_error=original_error
        )
        self.backend = backend
        self.timeout_ms = timeout_ms


class BackendUnavailableError(RetrieverError):
    """
    Raised when a search backend is unavailable.

    This can happen when:
    - Database connection fails
    - Circuit breaker is open
    - Backend is marked unhealthy
    """

    def __init__(
        self,
        backend: str,
        reason: str,
        original_error: Optional[Exception] = None
    ):
        message = f"{backend} is unavailable: {reason}"
        super().__init__(
            message=message,
            source=backend,
            details={"reason": reason},
            original_error=original_error
        )
        self.backend = backend
        self.reason = reason


class VectorSearchError(RetrieverError):
    """Raised when Supabase vector search fails."""

    def __init__(
        self,
        message: str,
        backend: str = "supabase_vector",
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            source=backend,
            query=query,
            details=details,
            original_error=original_error
        )
        self.backend = backend


class FulltextSearchError(RetrieverError):
    """Raised when Supabase full-text search fails."""

    def __init__(
        self,
        message: str,
        backend: str = "supabase_fulltext",
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            source=backend,
            query=query,
            details=details,
            original_error=original_error
        )
        self.backend = backend


class GraphSearchError(RetrieverError):
    """Raised when FalkorDB graph search fails."""

    def __init__(
        self,
        message: str,
        backend: str = "falkordb_graph",
        query: Optional[str] = None,
        cypher_query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        merged_details = details or {}
        if cypher_query:
            merged_details["cypher_query"] = cypher_query[:200]  # Truncate

        super().__init__(
            message=message,
            source=backend,
            query=query,
            details=merged_details,
            original_error=original_error
        )
        self.backend = backend
        self.cypher_query = cypher_query


class EntityExtractionError(RAGError):
    """
    Raised when entity extraction fails.

    Examples:
    - Vocabulary file not found
    - Invalid query format
    - Fuzzy matching error
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if query:
            details["query"] = query[:100]
        super().__init__(message, details, original_error)
        self.query = query


class FusionError(RAGError):
    """
    Raised when result fusion (RRF) fails.

    Examples:
    - No results from any backend
    - Score normalization error
    """
    pass


class HealthCheckError(RAGError):
    """
    Raised when health check fails.

    Includes which backend failed and why.
    """

    def __init__(
        self,
        backend: str,
        reason: str,
        original_error: Optional[Exception] = None
    ):
        message = f"Health check failed for {backend}: {reason}"
        super().__init__(
            message=message,
            details={"backend": backend, "reason": reason},
            original_error=original_error
        )
        self.backend = backend
        self.reason = reason


class CacheError(RAGError):
    """
    Raised when cache operations fail.

    Non-fatal - the system should continue without cache.
    """
    pass


class CircuitBreakerOpenError(RAGError):
    """
    Raised when a circuit breaker is open for a backend.

    The backend has failed too many times and is temporarily blocked.
    """

    def __init__(
        self,
        backend: str,
        reset_time_seconds: float
    ):
        message = f"Circuit breaker open for {backend}, resets in {reset_time_seconds:.1f}s"
        super().__init__(
            message=message,
            details={
                "backend": backend,
                "reset_time_seconds": reset_time_seconds
            }
        )
        self.backend = backend
        self.reset_time_seconds = reset_time_seconds
