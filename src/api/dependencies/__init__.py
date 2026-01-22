"""API Dependencies for FastAPI.

This module provides dependency injection for API routes.

Available Dependencies:
- Auth: JWT authentication with RBAC (viewer, analyst, operator, admin)
- BentoML: ML model serving
- Redis: Caching, sessions, rate limiting
- FalkorDB: Knowledge graph
- Supabase: PostgreSQL database
"""

from src.api.dependencies.auth import (
    AuthError,
    ROLE_LEVELS,
    UserRole,
    get_current_user,
    get_user_role,
    has_role,
    is_auth_enabled,
    is_testing_mode,
    require_admin,
    require_analyst,
    require_auth,
    require_operator,
    require_viewer,
)
from src.api.dependencies.bentoml_client import (
    BentoMLClient,
    BentoMLClientConfig,
    get_bentoml_client,
)
from src.api.dependencies.falkordb_client import (
    close_falkordb,
    falkordb_health_check,
    get_falkordb,
    get_graph,
    init_falkordb,
)
from src.api.dependencies.redis_client import (
    close_redis,
    get_redis,
    init_redis,
    redis_health_check,
)
from src.api.dependencies.supabase_client import (
    close_supabase,
    get_supabase,
    init_supabase,
    supabase_health_check,
)

__all__ = [
    # Auth (RBAC)
    "AuthError",
    "UserRole",
    "ROLE_LEVELS",
    "get_current_user",
    "get_user_role",
    "has_role",
    "require_auth",
    "require_viewer",
    "require_analyst",
    "require_operator",
    "require_admin",
    "is_auth_enabled",
    "is_testing_mode",
    # BentoML
    "BentoMLClient",
    "BentoMLClientConfig",
    "get_bentoml_client",
    # Redis
    "init_redis",
    "get_redis",
    "close_redis",
    "redis_health_check",
    # FalkorDB
    "init_falkordb",
    "get_falkordb",
    "get_graph",
    "close_falkordb",
    "falkordb_health_check",
    # Supabase
    "init_supabase",
    "get_supabase",
    "close_supabase",
    "supabase_health_check",
]
