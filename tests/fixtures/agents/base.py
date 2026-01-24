"""Base fixtures and factories for agent state management.

Provides consistent patterns for creating agent workflow states:
- StateProgression: Helper class for managing state transitions
- create_base_state(): Factory for creating base state objects
- create_workflow_states(): Factory for generating state progressions

Usage:
    from tests.fixtures.agents.base import create_base_state, StateProgression

    # Create a base state with common fields
    base = create_base_state(
        agent_name="feedback_learner",
        query="What is the TRx trend?",
        context={"user_id": "123"},
    )

    # Create state progressions for workflow testing
    states = StateProgression(base)
    states.add_stage("with_feedback", feedback_items=[...])
    states.add_stage("with_patterns", detected_patterns=[...])
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from uuid import uuid4

import pytest


T = TypeVar("T")


@dataclass
class StateProgression:
    """Helper class for managing agent state progressions.

    Agents typically follow a workflow pattern where state evolves
    through multiple stages. This class helps create consistent
    state fixtures for each stage.

    Example workflow:
        base_state -> with_data -> with_analysis -> with_response
    """

    base_state: Dict[str, Any]
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _current: str = "base"

    def add_stage(
        self,
        name: str,
        **updates: Any,
    ) -> "StateProgression":
        """Add a new stage to the progression.

        Args:
            name: Stage name (e.g., "with_feedback", "after_analysis")
            **updates: Fields to update/add in this stage

        Returns:
            Self for chaining
        """
        # Start from the previous stage or base
        if self._current == "base":
            prev_state = copy.deepcopy(self.base_state)
        else:
            prev_state = copy.deepcopy(self.stages[self._current])

        # Apply updates
        prev_state.update(updates)
        self.stages[name] = prev_state
        self._current = name
        return self

    def get_state(self, stage: str = "base") -> Dict[str, Any]:
        """Get the state at a specific stage.

        Args:
            stage: Stage name, or "base" for initial state

        Returns:
            Copy of state at that stage
        """
        if stage == "base":
            return copy.deepcopy(self.base_state)
        if stage not in self.stages:
            raise KeyError(f"Stage '{stage}' not found. Available: {list(self.stages.keys())}")
        return copy.deepcopy(self.stages[stage])

    def get_all_stages(self) -> Dict[str, Dict[str, Any]]:
        """Get all stages including base."""
        all_stages = {"base": copy.deepcopy(self.base_state)}
        for name, state in self.stages.items():
            all_stages[name] = copy.deepcopy(state)
        return all_stages

    @property
    def stage_names(self) -> List[str]:
        """Get list of all stage names."""
        return ["base"] + list(self.stages.keys())


def create_base_state(
    agent_name: str,
    query: str = "Test query",
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    include_timestamps: bool = True,
    include_metadata: bool = True,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Factory function for creating consistent base agent states.

    Creates a standard base state structure that all agents can use
    as a starting point for their workflow.

    Args:
        agent_name: Name of the agent (e.g., "feedback_learner")
        query: The user query being processed
        context: Optional context dictionary
        session_id: Optional session ID (generated if not provided)
        user_id: Optional user ID (defaults to "test-user")
        include_timestamps: Include created_at/updated_at fields
        include_metadata: Include metadata dict
        extra_fields: Additional fields to include in base state

    Returns:
        Base state dictionary with standard fields
    """
    state = {
        "agent_name": agent_name,
        "query": query,
        "session_id": session_id or str(uuid4()),
        "user_id": user_id or "test-user",
        "context": context or {},
        "messages": [],
        "errors": [],
        "status": "pending",
    }

    if include_timestamps:
        now = datetime.utcnow().isoformat()
        state["created_at"] = now
        state["updated_at"] = now

    if include_metadata:
        state["metadata"] = {
            "agent_version": "1.0.0",
            "workflow_id": str(uuid4()),
        }

    if extra_fields:
        state.update(extra_fields)

    return state


def create_workflow_states(
    base_state: Dict[str, Any],
    progressions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Create multiple workflow states from a base state.

    Convenience function for creating all stages at once.

    Args:
        base_state: The initial state
        progressions: List of dicts with "name" and updates
            Example: [
                {"name": "with_data", "data": [...]},
                {"name": "with_analysis", "analysis": {...}},
            ]

    Returns:
        Dict mapping stage names to state dicts
    """
    progression = StateProgression(base_state)
    for p in progressions:
        name = p.pop("name")
        progression.add_stage(name, **p)
    return progression.get_all_stages()


# =============================================================================
# COMMON STATE FIELD HELPERS
# =============================================================================


def create_messages_field(
    messages: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Create a messages field with optional initial messages."""
    if messages is None:
        return []
    return [
        {
            "role": m.get("role", "user"),
            "content": m.get("content", ""),
            "timestamp": m.get("timestamp", datetime.utcnow().isoformat()),
        }
        for m in messages
    ]


def create_error_field(
    errors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Create an errors field with optional initial errors."""
    if errors is None:
        return []
    return [
        {
            "message": e,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "error",
        }
        for e in errors
    ]


def create_context_field(
    brand: Optional[str] = None,
    kpi: Optional[str] = None,
    region: Optional[str] = None,
    time_range: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Create a context field with common E2I context values."""
    context = {}
    if brand:
        context["brand"] = brand
    if kpi:
        context["kpi"] = kpi
    if region:
        context["region"] = region
    if time_range:
        context["time_range"] = time_range
    context.update(extra)
    return context


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def base_agent_state() -> Dict[str, Any]:
    """Generic base state fixture for agent tests."""
    return create_base_state(
        agent_name="test_agent",
        query="What is the TRx for Kisqali in Q4?",
        context=create_context_field(
            brand="Kisqali",
            kpi="TRx",
            time_range="Q4 2024",
        ),
    )


@pytest.fixture
def state_progression_factory() -> Callable[..., StateProgression]:
    """Factory fixture for creating state progressions."""
    def _create(
        agent_name: str = "test_agent",
        query: str = "Test query",
        **base_kwargs: Any,
    ) -> StateProgression:
        base = create_base_state(agent_name=agent_name, query=query, **base_kwargs)
        return StateProgression(base)
    return _create


@pytest.fixture
def workflow_states_factory() -> Callable[..., Dict[str, Dict[str, Any]]]:
    """Factory fixture for creating complete workflow state sets."""
    def _create(
        agent_name: str,
        progressions: List[Dict[str, Any]],
        **base_kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        base = create_base_state(agent_name=agent_name, **base_kwargs)
        return create_workflow_states(base, progressions)
    return _create
