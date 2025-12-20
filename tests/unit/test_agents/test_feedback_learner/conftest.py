"""
Pytest fixtures for Feedback Learner agent tests.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def sample_feedback_items() -> List[Dict[str, Any]]:
    """Sample feedback items for testing."""
    return [
        {
            "feedback_id": "F001",
            "source_agent": "causal_impact",
            "query": "What caused the TRx drop?",
            "agent_response": "The drop was caused by competitor launch.",
            "user_feedback": 4,
            "feedback_type": "rating",
            "timestamp": "2024-01-15T10:00:00Z",
        },
        {
            "feedback_id": "F002",
            "source_agent": "causal_impact",
            "query": "Show market share trend",
            "agent_response": "Market share is declining by 2%.",
            "user_feedback": 2,
            "feedback_type": "rating",
            "timestamp": "2024-01-15T11:00:00Z",
        },
        {
            "feedback_id": "F003",
            "source_agent": "gap_analyzer",
            "query": "Find ROI opportunities",
            "agent_response": "Top opportunity is Region A.",
            "user_feedback": "The opportunity should be Region B, not A",
            "feedback_type": "correction",
            "timestamp": "2024-01-15T12:00:00Z",
        },
        {
            "feedback_id": "F004",
            "source_agent": "gap_analyzer",
            "query": "Identify gaps",
            "agent_response": "Gap in territory coverage.",
            "user_feedback": {"actual": 0.75, "predicted": 0.85, "error": -0.10},
            "feedback_type": "outcome",
            "timestamp": "2024-01-15T13:00:00Z",
        },
        {
            "feedback_id": "F005",
            "source_agent": "prediction_synthesizer",
            "query": "Predict next quarter TRx",
            "agent_response": "Predicted TRx: 15,000",
            "user_feedback": {"actual": 14000, "predicted": 15000, "error": -1000},
            "feedback_type": "outcome",
            "timestamp": "2024-01-15T14:00:00Z",
        },
        {
            "feedback_id": "F006",
            "source_agent": "prediction_synthesizer",
            "query": "Forecast market share",
            "agent_response": "Market share will be 25%.",
            "user_feedback": 5,
            "feedback_type": "rating",
            "timestamp": "2024-01-15T15:00:00Z",
        },
    ]


@pytest.fixture
def low_rating_feedback() -> List[Dict[str, Any]]:
    """Feedback items with low ratings for pattern detection."""
    return [
        {
            "feedback_id": f"F{i:03d}",
            "source_agent": "explainer",
            "query": f"Query {i}",
            "agent_response": f"Response {i}",
            "user_feedback": 1 if i % 2 == 0 else 2,
            "feedback_type": "rating",
            "timestamp": f"2024-01-{15 + i % 15:02d}T10:00:00Z",
        }
        for i in range(10)
    ]


@pytest.fixture
def correction_heavy_feedback() -> List[Dict[str, Any]]:
    """Feedback items with many corrections."""
    return [
        {
            "feedback_id": f"F{i:03d}",
            "source_agent": "causal_impact",
            "query": f"Query {i}",
            "agent_response": f"Response {i}",
            "user_feedback": f"Correction for query {i}",
            "feedback_type": "correction",
            "timestamp": f"2024-01-{15 + i % 15:02d}T10:00:00Z",
        }
        for i in range(8)
    ]


@pytest.fixture
def outcome_error_feedback() -> List[Dict[str, Any]]:
    """Feedback items with outcome errors."""
    return [
        {
            "feedback_id": f"F{i:03d}",
            "source_agent": "prediction_synthesizer",
            "query": f"Prediction query {i}",
            "agent_response": f"Prediction {i}",
            "user_feedback": {"actual": 100, "predicted": 150, "error": -50},
            "feedback_type": "outcome",
            "timestamp": f"2024-01-{15 + i % 15:02d}T10:00:00Z",
        }
        for i in range(5)
    ]


@pytest.fixture
def sample_detected_patterns() -> List[Dict[str, Any]]:
    """Sample detected patterns for testing."""
    return [
        {
            "pattern_id": "P1",
            "pattern_type": "accuracy_issue",
            "description": "Low average user ratings detected",
            "frequency": 10,
            "severity": "high",
            "affected_agents": ["explainer"],
            "example_feedback_ids": ["F001", "F002", "F003"],
            "root_cause_hypothesis": "Agent responses may not meet user expectations",
        },
        {
            "pattern_id": "P2",
            "pattern_type": "relevance_issue",
            "description": "Agent 'causal_impact' has high negative feedback rate",
            "frequency": 5,
            "severity": "medium",
            "affected_agents": ["causal_impact"],
            "example_feedback_ids": ["F004", "F005"],
            "root_cause_hypothesis": "Agent needs prompt updates",
        },
    ]


@pytest.fixture
def sample_recommendations() -> List[Dict[str, Any]]:
    """Sample learning recommendations for testing."""
    return [
        {
            "recommendation_id": "R1",
            "category": "data_update",
            "description": "Review and update training data for explainer",
            "affected_agents": ["explainer"],
            "expected_impact": "Improved accuracy and reduced errors",
            "implementation_effort": "medium",
            "priority": 1,
            "proposed_change": "Update knowledge base with corrected information",
        },
        {
            "recommendation_id": "R2",
            "category": "prompt_update",
            "description": "Improve response relevance for causal_impact",
            "affected_agents": ["causal_impact"],
            "expected_impact": "More relevant and focused responses",
            "implementation_effort": "low",
            "priority": 2,
            "proposed_change": "Update system prompts to better guide response generation",
        },
    ]


@pytest.fixture
def mock_feedback_store(sample_feedback_items):
    """Mock feedback store for testing."""
    store = AsyncMock()
    store.get_feedback = AsyncMock(return_value=sample_feedback_items)
    return store


@pytest.fixture
def mock_outcome_store():
    """Mock outcome store for testing."""
    store = AsyncMock()
    store.get_outcomes = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_knowledge_stores():
    """Mock knowledge stores for testing."""
    return {
        "baseline": AsyncMock(update=AsyncMock()),
        "agent_config": AsyncMock(update=AsyncMock()),
        "prompt": AsyncMock(update=AsyncMock()),
        "threshold": AsyncMock(update=AsyncMock()),
    }


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="""```json
{
  "patterns": [
    {
      "pattern_id": "P1",
      "pattern_type": "accuracy_issue",
      "description": "LLM detected accuracy pattern",
      "frequency": 5,
      "severity": "medium",
      "affected_agents": ["test_agent"],
      "example_feedback_ids": ["F001"],
      "root_cause_hypothesis": "LLM hypothesis"
    }
  ]
}
```"""
        )
    )
    return llm


@pytest.fixture
def base_state():
    """Base state for testing."""
    return {
        "batch_id": "test_batch_001",
        "time_range_start": "2024-01-01T00:00:00Z",
        "time_range_end": "2024-01-31T23:59:59Z",
        "focus_agents": None,
        "feedback_items": None,
        "feedback_summary": None,
        "detected_patterns": None,
        "pattern_clusters": None,
        "learning_recommendations": None,
        "priority_improvements": None,
        "proposed_updates": None,
        "applied_updates": None,
        "learning_summary": None,
        "metrics_before": None,
        "metrics_after": None,
        "collection_latency_ms": 0,
        "analysis_latency_ms": 0,
        "extraction_latency_ms": 0,
        "update_latency_ms": 0,
        "total_latency_ms": 0,
        "model_used": None,
        "errors": [],
        "warnings": [],
        "status": "pending",
    }


@pytest.fixture
def state_with_feedback(base_state, sample_feedback_items):
    """State with feedback items loaded."""
    return {
        **base_state,
        "feedback_items": sample_feedback_items,
        "feedback_summary": {
            "total_count": len(sample_feedback_items),
            "by_type": {"rating": 3, "correction": 1, "outcome": 2},
            "by_agent": {
                "causal_impact": 2,
                "gap_analyzer": 2,
                "prediction_synthesizer": 2,
            },
            "average_rating": 3.67,
        },
        "status": "analyzing",
    }


@pytest.fixture
def state_with_patterns(state_with_feedback, sample_detected_patterns):
    """State with patterns detected."""
    return {
        **state_with_feedback,
        "detected_patterns": sample_detected_patterns,
        "pattern_clusters": {
            "accuracy_issue": ["P1"],
            "relevance_issue": ["P2"],
        },
        "status": "extracting",
    }


@pytest.fixture
def state_with_recommendations(state_with_patterns, sample_recommendations):
    """State with recommendations generated."""
    return {
        **state_with_patterns,
        "learning_recommendations": sample_recommendations,
        "priority_improvements": [
            "Review and update training data for explainer",
            "Improve response relevance for causal_impact",
        ],
        "status": "updating",
    }
