"""
E2I Agentic Memory - Service Layer
Provides factory functions for all external service dependencies.

Usage:
    from src.memory.services import (
        get_redis_client,
        get_supabase_client,
        get_falkordb_client,
        get_embedding_service,
        get_llm_service,
        load_memory_config,
    )
"""

from src.memory.services.config import load_memory_config, get_config, MemoryConfig
from src.memory.services.factories import (
    get_redis_client,
    get_supabase_client,
    get_falkordb_client,
    get_embedding_service,
    get_llm_service,
    ServiceConnectionError,
)

__all__ = [
    # Config
    "load_memory_config",
    "get_config",
    "MemoryConfig",
    # Factories
    "get_redis_client",
    "get_supabase_client",
    "get_falkordb_client",
    "get_embedding_service",
    "get_llm_service",
    "ServiceConnectionError",
]
