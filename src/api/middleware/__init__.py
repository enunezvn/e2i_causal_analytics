"""API Middleware package.

Provides middleware for:
- JWT Authentication (Supabase)
- Security Headers (XSS, clickjacking, MIME sniffing protection)
- Rate Limiting (Redis-backed with in-memory fallback)
- Distributed Tracing (W3C Trace Context, Zipkin B3)
- Request correlation IDs

Author: E2I Causal Analytics Team
Version: 4.2.1
"""

from src.api.middleware.auth_middleware import (
    JWTAuthMiddleware,
    get_public_paths,
)
from src.api.middleware.rate_limit_middleware import RateLimitMiddleware
from src.api.middleware.security_middleware import SecurityHeadersMiddleware
from src.api.middleware.tracing import (
    TraceContext,
    TracingMiddleware,
    get_correlation_id,
    get_request_id,
    get_trace_context,
    get_trace_id,
    with_trace_context,
)

__all__ = [
    # Authentication
    "JWTAuthMiddleware",
    "get_public_paths",
    # Rate Limiting
    "RateLimitMiddleware",
    # Security Headers
    "SecurityHeadersMiddleware",
    # Tracing & Correlation
    "TracingMiddleware",
    "TraceContext",
    "get_request_id",
    "get_correlation_id",
    "get_trace_id",
    "get_trace_context",
    "with_trace_context",
]
