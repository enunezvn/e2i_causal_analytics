"""Centralized fixture library for E2I Causal Analytics tests.

This module provides reusable fixtures, mocks, and utilities used across
the test suite. It consolidates common patterns to reduce duplication
and ensure consistency.

Structure:
- mocks/: Mock classes for external services (LLM, databases, stores)
- agents/: Agent-specific fixtures with shared patterns
- helpers.py: Shared helper functions for fixture creation

Usage:
    # In conftest.py files:
    from tests.fixtures.mocks.llm import mock_llm, MockLLMClient
    from tests.fixtures.mocks.databases import MockSupabaseClient
    from tests.fixtures.agents.base import create_base_state
    from tests.fixtures.helpers import make_decomposition_response
"""

# Lazy imports to avoid circular dependencies
# Import from submodules as needed in test files

__all__ = [
    # LLM Mocks
    "MockLLMClient",
    "SimpleMockLLM",
    # Database Mocks
    "MockDatabaseClient",
    "MockSupabaseClient",
    "MockSupabaseQuery",
    # Agent Base
    "StateProgression",
    "create_base_state",
    "create_workflow_states",
    # Helpers
    "make_decomposition_response",
    "make_planning_response",
    "make_synthesis_response",
]
