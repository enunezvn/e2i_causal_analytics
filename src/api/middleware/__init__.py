"""API Middleware package.

Provides middleware for:
- JWT Authentication (Supabase)
- Security Headers (XSS, clickjacking, MIME sniffing protection)
- Rate Limiting (Redis-backed with in-memory fallback)
- Request logging (future)

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

from src.api.middleware.auth_middleware import (
    JWTAuthMiddleware,
    get_public_paths,
)
from src.api.middleware.rate_limit_middleware import RateLimitMiddleware
from src.api.middleware.security_middleware import SecurityHeadersMiddleware

__all__ = [
    "JWTAuthMiddleware",
    "get_public_paths",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
]
