"""API Dependencies for FastAPI.

This module provides dependency injection for API routes.

Available Dependencies:
- BentoML: ML model serving
- Redis: Caching, sessions, rate limiting
- FalkorDB: Knowledge graph
- Supabase: PostgreSQL database
"""

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
