"""Mock classes for external services.

This module provides consistent mock implementations for:
- LLM clients (simple and advanced)
- Database clients (Supabase, Redis, FalkorDB)
- Store classes (metrics, pipeline, context)
- Registry classes (tools, models, agents)
"""

from tests.fixtures.mocks.llm import (
    MockLLMClient,
    SimpleMockLLM,
    mock_llm,
    mock_simple_llm,
    mock_advanced_llm,
)
from tests.fixtures.mocks.databases import (
    MockDatabaseClient,
    MockSupabaseClient,
    MockSupabaseQuery,
    mock_supabase_client,
)

__all__ = [
    "MockLLMClient",
    "SimpleMockLLM",
    "mock_llm",
    "mock_simple_llm",
    "mock_advanced_llm",
    "MockDatabaseClient",
    "MockSupabaseClient",
    "MockSupabaseQuery",
    "mock_supabase_client",
]
