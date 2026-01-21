"""JWT Authentication Middleware for FastAPI.

Protects API routes by validating Supabase JWT tokens.
Configurable public paths that bypass authentication.

Author: E2I Causal Analytics Team
Version: 4.2.2
"""

import logging
import os
import re
from typing import Callable, List, Set, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.dependencies.auth import is_auth_enabled, verify_supabase_token

logger = logging.getLogger(__name__)

# Testing mode - bypasses authentication for integration/e2e tests
# Set E2I_TESTING_MODE=true to disable auth during testing
TESTING_MODE = os.environ.get("E2I_TESTING_MODE", "").lower() in ("true", "1", "yes")

# Get allowed origins for CORS headers on error responses
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://localhost:8080,http://127.0.0.1:5173,http://127.0.0.1:5174"
).split(",")

# Paths that don't require authentication
# Format: (method_pattern, path_pattern) - use "*" for any method
PUBLIC_PATHS: List[Tuple[str, str]] = [
    # Health endpoints - always public
    ("*", "/"),
    ("*", "/health"),
    ("*", "/healthz"),
    ("*", "/ready"),
    ("*", "/health/bentoml"),
    # Documentation - always public
    ("*", "/api/docs"),
    ("*", "/api/docs/oauth2-redirect"),
    ("*", "/api/redoc"),
    ("*", "/api/openapi.json"),
    # Authentication endpoints - must be public for login/register
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/register"),
    ("POST", "/api/auth/refresh"),
    # Read-only KPI endpoints - public
    ("GET", "/api/kpis"),
    ("GET", "/api/kpis/workstreams"),
    ("GET", "/api/kpis/health"),
    # Read-only Causal endpoints - public
    ("GET", "/causal/estimators"),
    ("GET", "/causal/health"),
    # Graph endpoints - public for demo visualization
    ("GET", "/api/graph/health"),
    ("GET", "/api/graph/nodes"),
    ("GET", "/api/graph/relationships"),
    ("GET", "/api/graph/stats"),
    # CopilotKit - Only status endpoint is public (requires auth for chat/feedback/analytics)
    ("GET", "/api/copilotkit/status"),
]

# Paths that match patterns (for dynamic routes)
PUBLIC_PATH_PATTERNS: List[Tuple[str, str]] = [
    # KPI metadata by ID is public
    ("GET", r"^/api/kpis/[^/]+/metadata$"),
    # CopilotKit - ONLY status endpoint is public (see PUBLIC_PATHS above)
    # All other CopilotKit endpoints (chat, feedback, analytics) require authentication
    # The frontend includes Authorization header via axios interceptors
]


def _is_public_path(method: str, path: str) -> bool:
    """
    Check if a request path is public (no auth required).

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path

    Returns:
        True if public, False if auth required
    """
    # Normalize path (remove trailing slash)
    path = path.rstrip("/") or "/"

    # Check exact matches
    for allowed_method, allowed_path in PUBLIC_PATHS:
        if allowed_method == "*" or allowed_method == method:
            if path == allowed_path:
                return True

    # Check pattern matches
    for allowed_method, pattern in PUBLIC_PATH_PATTERNS:
        if allowed_method == "*" or allowed_method == method:
            if re.match(pattern, path):
                return True

    return False


def _get_cors_headers(request: Request) -> dict:
    """
    Get CORS headers for error responses.

    Ensures that 401/403 responses include proper CORS headers so browsers
    can read the error message instead of showing a generic CORS error.

    Args:
        request: The incoming request

    Returns:
        Dictionary of CORS headers to add to the response
    """
    origin = request.headers.get("origin", "")

    # Check if origin is allowed
    if origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
        return {
            "Access-Control-Allow-Origin": origin or "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept, Origin, X-Requested-With",
        }

    # Default headers if origin not in allowed list
    return {}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces JWT authentication on protected routes.

    Public paths bypass authentication entirely.
    Protected paths require a valid Supabase JWT in the Authorization header.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and enforce authentication."""
        method = request.method
        path = request.url.path

        # Skip auth for public paths
        if _is_public_path(method, path):
            return await call_next(request)

        # Skip auth for OPTIONS (CORS preflight)
        if method == "OPTIONS":
            return await call_next(request)

        # Skip auth in testing mode (for integration/e2e tests)
        if TESTING_MODE:
            # Provide a mock user for testing
            request.state.user = {
                "id": "test-user-id",
                "email": "test@e2i-analytics.com",
                "role": "authenticated",
                "aud": "authenticated",
                "app_metadata": {"role": "admin"},
                "user_metadata": {"name": "Test User"},
            }
            logger.debug(f"Testing mode - bypassing auth for {method} {path}")
            return await call_next(request)

        # Check if auth is configured
        if not is_auth_enabled():
            logger.warning(
                f"Auth not configured - allowing unauthenticated access to {method} {path}"
            )
            return await call_next(request)

        # Get CORS headers for error responses
        cors_headers = _get_cors_headers(request)

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")

        if not auth_header:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "authentication_required",
                    "message": "Missing Authorization header",
                    "hint": "Include 'Authorization: Bearer <token>' header",
                },
                headers={"WWW-Authenticate": "Bearer", **cors_headers},
            )

        # Parse Bearer token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "invalid_authorization",
                    "message": "Invalid Authorization header format",
                    "hint": "Use format: 'Bearer <token>'",
                },
                headers={"WWW-Authenticate": "Bearer", **cors_headers},
            )

        token = parts[1]

        # Verify token with Supabase
        user = await verify_supabase_token(token)

        if user is None:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "invalid_token",
                    "message": "Invalid or expired token",
                    "hint": "Login again to get a fresh token",
                },
                headers={"WWW-Authenticate": "Bearer", **cors_headers},
            )

        # Attach user to request state for downstream use
        request.state.user = user

        # Log authenticated request
        logger.debug(f"Authenticated request: {method} {path} by {user.get('email')}")

        return await call_next(request)


def get_public_paths() -> List[str]:
    """Get list of public paths for documentation."""
    paths = []
    for method, path in PUBLIC_PATHS:
        prefix = f"[{method}] " if method != "*" else ""
        paths.append(f"{prefix}{path}")
    for method, pattern in PUBLIC_PATH_PATTERNS:
        prefix = f"[{method}] " if method != "*" else ""
        paths.append(f"{prefix}{pattern} (pattern)")
    return paths
