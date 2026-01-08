"""API Middleware package.

Provides middleware for:
- JWT Authentication (Supabase)
- Rate limiting (future)
- Request logging (future)

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

from src.api.middleware.auth_middleware import (
    JWTAuthMiddleware,
    get_public_paths,
)

__all__ = [
    "JWTAuthMiddleware",
    "get_public_paths",
]
