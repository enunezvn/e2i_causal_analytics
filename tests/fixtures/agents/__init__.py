"""Agent-specific fixtures with shared patterns.

This module provides:
- Base state factory functions for consistent agent state creation
- StateProgression helper for workflow state management
- Agent-specific fixture modules for each agent type
"""

from tests.fixtures.agents.base import (
    StateProgression,
    create_base_state,
    create_workflow_states,
)

__all__ = [
    "StateProgression",
    "create_base_state",
    "create_workflow_states",
]
