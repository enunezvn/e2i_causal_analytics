"""JWT Authentication dependency for FastAPI with Supabase.

Validates JWT tokens issued by Supabase Auth.
Tokens are passed in the Authorization header as Bearer tokens.

Role-Based Access Control (RBAC):
    Hierarchical roles: ADMIN > OPERATOR > ANALYST > VIEWER
    - viewer: Read-only dashboard access
    - analyst: Run analyses (causal, gap, segment)
    - operator: Manage experiments, feedback learning, digital twin
    - admin: System management (cache, retraining, user management)

Usage:
    from src.api.dependencies.auth import get_current_user, require_auth
    from src.api.dependencies.auth import require_viewer, require_analyst, require_operator, require_admin

    # Get user info (optional auth)
    @app.get("/profile")
    async def profile(user: Optional[dict] = Depends(get_current_user)):
        ...

    # Require authentication (any role)
    @app.post("/protected")
    async def protected(user: dict = Depends(require_auth)):
        ...

    # Require specific role level
    @app.post("/analyze")
    async def analyze(user: dict = Depends(require_analyst)):
        ...

Author: E2I Causal Analytics Team
Version: 4.3.0
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class UserRole(str, Enum):
    """User roles for RBAC with hierarchical permissions.

    Hierarchy: ADMIN > OPERATOR > ANALYST > VIEWER
    Higher roles inherit all permissions from lower roles.
    """

    VIEWER = "viewer"
    ANALYST = "analyst"
    OPERATOR = "operator"
    ADMIN = "admin"


# Role hierarchy levels - higher number = more privileges
ROLE_LEVELS: Dict[UserRole, int] = {
    UserRole.VIEWER: 1,
    UserRole.ANALYST: 2,
    UserRole.OPERATOR: 3,
    UserRole.ADMIN: 4,
}

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")

# Testing mode - bypasses authentication for integration/e2e tests
TESTING_MODE = os.environ.get("E2I_TESTING_MODE", "").lower() in ("true", "1", "yes")
_ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
if TESTING_MODE and _ENVIRONMENT == "production":
    import warnings

    warnings.warn(
        "E2I_TESTING_MODE is set but ENVIRONMENT=production -- testing mode DISABLED",
        RuntimeWarning,
    )
    TESTING_MODE = False

# Mock user for testing mode (defaults to admin for full access in tests)
TEST_USER: Dict[str, Any] = {
    "id": "test-user-id",
    "email": "test@e2i-analytics.com",
    "role": "authenticated",
    "aud": "authenticated",
    "created_at": None,
    "app_metadata": {"role": "admin"},  # RBAC role stored here
    "user_metadata": {"name": "Test User"},
}


def get_user_role(user: Dict[str, Any]) -> UserRole:
    """Extract the RBAC role from user data.

    Looks for role in the following order:
    1. app_metadata.role (preferred - Supabase convention)
    2. user.role (fallback)
    3. Default to VIEWER if not found

    Args:
        user: User dict from authentication

    Returns:
        UserRole enum value
    """
    # Check app_metadata.role first (Supabase convention)
    role_str = user.get("app_metadata", {}).get("role")

    # Fallback to top-level role field
    if not role_str:
        role_str = user.get("role")

    # Handle legacy is_admin flag
    if not role_str and user.get("app_metadata", {}).get("is_admin"):
        return UserRole.ADMIN

    # Convert string to enum, default to viewer
    if role_str:
        try:
            return UserRole(role_str.lower())
        except ValueError:
            logger.warning(f"Unknown role '{role_str}', defaulting to viewer")

    return UserRole.VIEWER


def has_role(user: Dict[str, Any], required_role: UserRole) -> bool:
    """Check if user has at least the required role level.

    Uses hierarchical comparison: ADMIN > OPERATOR > ANALYST > VIEWER

    Args:
        user: User dict from authentication
        required_role: Minimum required role

    Returns:
        True if user's role level >= required role level
    """
    user_role = get_user_role(user)
    user_level = ROLE_LEVELS.get(user_role, 0)
    required_level = ROLE_LEVELS.get(required_role, 0)
    return user_level >= required_level


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


async def require_viewer(
    user: Dict[str, Any] = Depends(require_auth),
) -> Dict[str, Any]:
    """Require at least viewer role (any authenticated user).

    This is effectively the same as require_auth but explicitly
    documents the minimum role requirement.

    Args:
        user: Authenticated user from require_auth

    Returns:
        User dict if viewer or higher

    Raises:
        AuthError: If not authenticated (via require_auth)
    """
    # All authenticated users have at least viewer access
    if not has_role(user, UserRole.VIEWER):
        raise AuthError(
            "Viewer access required",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    return user


async def require_analyst(
    user: Dict[str, Any] = Depends(require_auth),
) -> Dict[str, Any]:
    """Require at least analyst role.

    Analysts can run analyses (causal, gap, segment).

    Args:
        user: Authenticated user from require_auth

    Returns:
        User dict if analyst or higher

    Raises:
        AuthError: If insufficient role
    """
    if not has_role(user, UserRole.ANALYST):
        raise AuthError(
            "Analyst privileges required",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    return user


async def require_operator(
    user: Dict[str, Any] = Depends(require_auth),
) -> Dict[str, Any]:
    """Require at least operator role.

    Operators can manage experiments, feedback learning, digital twin.

    Args:
        user: Authenticated user from require_auth

    Returns:
        User dict if operator or higher

    Raises:
        AuthError: If insufficient role
    """
    if not has_role(user, UserRole.OPERATOR):
        raise AuthError(
            "Operator privileges required",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    return user


async def require_admin(
    user: Dict[str, Any] = Depends(require_auth),
) -> Dict[str, Any]:
    """Require admin role.

    Admins have full system access including cache invalidation,
    model retraining, and user management.

    Args:
        user: Authenticated user from require_auth

    Returns:
        User dict if admin

    Raises:
        AuthError: If not admin
    """
    if not has_role(user, UserRole.ADMIN):
        raise AuthError(
            "Admin privileges required",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    return user


# Convenience function to check if auth is configured
def is_auth_enabled() -> bool:
    """Check if Supabase auth is configured and not in testing mode."""
    if is_testing_mode():
        return False
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def is_testing_mode() -> bool:
    """Check if running in testing mode."""
    return TESTING_MODE
