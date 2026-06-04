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

import base64
import binascii
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Request, WebSocket, status
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


def _warn_missing_auth_secrets() -> List[str]:
    """Emit startup warnings for missing auth secrets and return the messages.

    Returns the list of warnings emitted (empty if all configured). Returning
    them — rather than only logging — makes the messages assertable in tests.

    Accuracy note (LOW finding fix): token verification in this module runs
    through ``verify_supabase_token`` -> Supabase ``client.auth.get_user()``,
    which needs SUPABASE_URL + SUPABASE_ANON_KEY. It does NOT use
    SUPABASE_JWT_SECRET, so the absence of that secret does not disable JWT
    verification. The old message ("JWT verification will be disabled") was
    therefore inaccurate; SUPABASE_JWT_SECRET is currently optional/unused on
    this path, and the real "auth disabled" condition is missing URL/ANON_KEY.
    """
    messages: List[str] = []
    if not SUPABASE_URL:
        messages.append(
            "SUPABASE_URL is not set — auth will be disabled. Set this in .env for production."
        )
    if not SUPABASE_ANON_KEY:
        messages.append(
            "SUPABASE_ANON_KEY is not set — auth will be disabled. Set this in .env for production."
        )
    if not SUPABASE_JWT_SECRET:
        # Informational only: this secret is not consumed by the current
        # get_user()-based verification path, so its absence does not turn
        # verification off. Phrase it as optional, not as "verification disabled".
        messages.append(
            "SUPABASE_JWT_SECRET is not set. It is optional for the current "
            "Supabase get_user() verification path (which uses SUPABASE_URL + "
            "SUPABASE_ANON_KEY) and does NOT disable JWT verification. Set it "
            "only if you add local HS256 signature verification."
        )

    _auth_logger = logging.getLogger(__name__)
    for message in messages:
        _auth_logger.warning(message)
    return messages


# Warn at startup if critical auth secrets are missing (skip in test mode)
if not TESTING_MODE:
    _warn_missing_auth_secrets()
_ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
if TESTING_MODE and _ENVIRONMENT == "production":
    import warnings

    warnings.warn(
        "E2I_TESTING_MODE is set but ENVIRONMENT=production -- testing mode DISABLED",
        RuntimeWarning,
        stacklevel=2,
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


def get_user_brands(user: Dict[str, Any]) -> List[str]:
    """Extract brand-access grants from user data.

    Returns the list of brand strings the user is permitted to read/operate
    on. ``['all']`` means cross-brand access (typically admin operators).

    Look-up order (matches ``get_user_role``):
    1. ``app_metadata.brands`` (Supabase convention)
    2. top-level ``brands`` field
    3. Empty list when neither is set

    Used by routes that enforce per-tenant access until full RLS lands —
    e.g. ``GET /api/sentinels`` filters by this set so an Operator with
    Brand-X grant cannot list Brand-Y sentinels via ``?brand=Brand-Y``.
    """
    brands = user.get("app_metadata", {}).get("brands")
    if brands is None:
        brands = user.get("brands", [])
    if isinstance(brands, str):
        return [brands]
    return list(brands or [])


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
        # #471: surface per-var truthiness — "auth disabled" was a
        # serious posture change collapsed under a non-actionable log.
        from src.utils.env_diagnostics import env_state

        logger.warning(
            "Supabase not configured - auth disabled. Diagnostic: %s; %s. "
            "If .env contains these, ensure load_dotenv() ran before "
            "module import.",
            env_state("SUPABASE_URL"),
            env_state("SUPABASE_ANON_KEY"),
        )
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


# Sentinel subprotocol the frontend sends as the FIRST offered protocol to mark
# that the SECOND offered protocol carries a base64url-encoded bearer JWT.
# See frontend/src/hooks/use-websocket.ts (PR #679).
WS_BEARER_SUBPROTOCOL = "bearer"

# Defense-in-depth bound on the encoded subprotocol token before we allocate a
# buffer and decode it. Real Supabase JWTs are well under 4 KB; 8 KB is generous.
# uvicorn's h11 parser already caps total handshake headers (~16 KB), so this is
# belt-and-suspenders for non-uvicorn ASGI servers / proxies with larger limits.
_MAX_SUBPROTOCOL_TOKEN_LEN = 8192


def _decode_subprotocol_token(encoded: str) -> Optional[str]:
    """Decode a base64url subprotocol value back into the raw JWT string.

    The frontend encodes the token as ``btoa(token)`` with ``+`` -> ``-``,
    ``/`` -> ``_`` and ``=`` padding STRIPPED (RFC 6455 subprotocol values must
    be HTTP tokens, which forbid ``.``/``+``/``/``/``=``). This restores the
    padding and url-safe-decodes it.

    Returns the decoded JWT, or ``None`` if the value is missing, oversized, or
    not decodable (all of which must be treated as an auth failure, never a 500).
    """
    if not encoded or len(encoded) > _MAX_SUBPROTOCOL_TOKEN_LEN:
        return None
    try:
        # Restore the stripped '=' padding to a multiple of 4.
        padded = encoded + "=" * (-len(encoded) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        return raw.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError) as e:
        logger.warning("Failed to decode WebSocket bearer subprotocol: %s", e)
        return None


async def authenticate_websocket(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    """Authenticate a WebSocket handshake from its offered subprotocols.

    Browsers cannot set arbitrary headers on a WebSocket handshake, so the
    bearer token is carried in ``Sec-WebSocket-Protocol`` as two values:
    ``['bearer', base64url(jwt)]`` (see ``_decode_subprotocol_token``).

    This helper ONLY verifies the token (authentication). It does NOT call
    ``websocket.accept()`` / ``close()`` — the endpoint owns the handshake
    lifecycle (including echoing the ``bearer`` subprotocol on accept).

    Returns the verified user dict, or ``None`` when no valid bearer token was
    offered (caller decides whether that is allowed under the fail-open posture).
    """
    subprotocols = websocket.scope.get("subprotocols") or []
    if len(subprotocols) < 2 or subprotocols[0] != WS_BEARER_SUBPROTOCOL:
        # No bearer token offered (anonymous handshake).
        return None

    token = _decode_subprotocol_token(subprotocols[1])
    if not token:
        return None

    return await verify_supabase_token(token)


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
