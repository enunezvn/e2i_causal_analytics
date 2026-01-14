"""JWT Authentication dependency for FastAPI with Supabase.

Validates JWT tokens issued by Supabase Auth.
Tokens are passed in the Authorization header as Bearer tokens.

Usage:
    from src.api.dependencies.auth import get_current_user, require_auth

    # Get user info (optional auth)
    @app.get("/profile")
    async def profile(user: Optional[dict] = Depends(get_current_user)):
        ...

    # Require authentication
    @app.post("/protected")
    async def protected(user: dict = Depends(require_auth)):
        ...

Author: E2I Causal Analytics Team
Version: 4.2.1
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")

# Testing mode - bypasses authentication for integration/e2e tests
TESTING_MODE = os.environ.get("E2I_TESTING_MODE", "").lower() in ("true", "1", "yes")

# Mock user for testing mode
TEST_USER: Dict[str, Any] = {
    "id": "test-user-id",
    "email": "test@e2i-analytics.com",
    "role": "authenticated",
    "aud": "authenticated",
    "created_at": None,
    "app_metadata": {"role": "admin"},
    "user_metadata": {"name": "Test User"},
}

# Security scheme for OpenAPI docs
security = HTTPBearer(auto_error=False)


class AuthError(HTTPException):
    """Authentication error with standard format."""

    def __init__(self, detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
        super().__init__(
            status_code=status_code,
            detail={"error": "authentication_error", "message": detail},
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_supabase_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Supabase JWT token.

    Args:
        token: The JWT token from Authorization header

    Returns:
        User data dict if valid, None if invalid

    Note:
        Uses Supabase's auth.getUser() which validates the token
        against Supabase's auth service.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        logger.warning("Supabase not configured - auth disabled")
        return None

    try:
        from supabase import create_client

        # Create client with the user's token for verification
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

        # Verify token by getting user - this validates with Supabase
        response = client.auth.get_user(token)

        if response and response.user:
            user_data = {
                "id": response.user.id,
                "email": response.user.email,
                "role": response.user.role,
                "aud": response.user.aud,
                "created_at": str(response.user.created_at) if response.user.created_at else None,
                "app_metadata": response.user.app_metadata or {},
                "user_metadata": response.user.user_metadata or {},
            }
            logger.debug(f"Token verified for user: {user_data['email']}")
            return user_data

        return None

    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        return None


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """
    Get current user from JWT token (optional).

    Returns None if no token provided or token is invalid.
    Use this for endpoints where auth is optional.

    Args:
        request: FastAPI request object
        credentials: Bearer token from Authorization header

    Returns:
        User dict if authenticated, None otherwise
    """
    if credentials is None:
        return None

    token = credentials.credentials
    user = await verify_supabase_token(token)

    if user:
        # Attach user to request state for logging/audit
        request.state.user = user

    return user


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    Require valid JWT authentication.

    Raises 401 if no token or invalid token.
    Use this for protected endpoints.

    Args:
        request: FastAPI request object
        credentials: Bearer token from Authorization header

    Returns:
        User dict if authenticated

    Raises:
        AuthError: If not authenticated
    """
    # In testing mode, return mock user
    if TESTING_MODE:
        request.state.user = TEST_USER
        return TEST_USER

    if credentials is None:
        raise AuthError("Missing authorization header")

    token = credentials.credentials
    user = await verify_supabase_token(token)

    if user is None:
        raise AuthError("Invalid or expired token")

    # Attach user to request state
    request.state.user = user

    return user


async def require_admin(
    user: Dict[str, Any] = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Require admin role.

    Args:
        user: Authenticated user from require_auth

    Returns:
        User dict if admin

    Raises:
        AuthError: If not admin
    """
    # Check for admin in app_metadata or role
    is_admin = (
        user.get("role") == "admin"
        or user.get("app_metadata", {}).get("role") == "admin"
        or user.get("app_metadata", {}).get("is_admin", False)
    )

    if not is_admin:
        raise AuthError(
            "Admin privileges required",
            status_code=status.HTTP_403_FORBIDDEN,
        )

    return user


# Convenience function to check if auth is configured
def is_auth_enabled() -> bool:
    """Check if Supabase auth is configured and not in testing mode."""
    if TESTING_MODE:
        return False
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def is_testing_mode() -> bool:
    """Check if running in testing mode."""
    return TESTING_MODE
