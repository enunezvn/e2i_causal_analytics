"""Test fixtures for Explainer Agent tests."""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================================
# PROTOCOL MOCKS
# ============================================================================


class MockConversationStore:
    """Mock conversation store for testing."""

    def __init__(self, history: Optional[List[Dict[str, Any]]] = None):
        self._history = history or []

    async def get_recent(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return mock conversation history."""
        return self._history[:limit]


# ============================================================================
# ANALYSIS RESULT FIXTURES
# ============================================================================


@pytest.fixture
def sample_causal_analysis():
    """Sample causal impact analysis result."""
    return {
        "agent": "causal_impact",
        "analysis_type": "effect_estimation",
        "key_findings": {
            "treatment": "marketing_campaign",
            "outcome": "sales_uplift",
            "effect_estimate": 0.23,
            "confidence_interval": [0.15, 0.31],
            "p_value": 0.002,
        },
        "methodology": "doubly_robust",
        "sample_size": 5000,
        "status": "completed",
    }


@pytest.fixture
def sample_gap_analysis():
    """Sample gap analyzer result."""
    return {
        "agent": "gap_analyzer",
        "analysis_type": "opportunity_detection",
        "key_findings": {
            "gaps_identified": 3,
            "top_opportunity": {
                "region": "Northeast",
                "potential_value": 150000,
                "confidence": 0.85,
            },
            "total_opportunity_value": 450000,
        },
        "recommendations": [
            "Increase coverage in Northeast",
            "Target high-value HCPs",
        ],
        "status": "completed",
    }


@pytest.fixture
def sample_drift_analysis():
    """Sample drift monitor result."""
    return {
        "agent": "drift_monitor",
        "analysis_type": "drift_detection",
        "key_findings": {
            "data_drift_detected": True,
            "drift_score": 0.15,
            "affected_features": ["region", "hcp_specialty"],
            "severity": "moderate",
        },
        "recommendations": ["Retrain model", "Update feature pipeline"],
        "status": "completed",
    }


@pytest.fixture
def sample_prediction_result():
    """Sample prediction synthesizer result."""
    return {
        "agent": "prediction_synthesizer",
        "analysis_type": "ensemble_prediction",
        "key_findings": {
            "prediction": 0.78,
            "confidence": 0.92,
            "model_agreement": 0.85,
            "contributing_factors": ["prior_engagement", "hcp_tier", "region"],
        },
        "methodology": "weighted_ensemble",
        "status": "completed",
    }


@pytest.fixture
def sample_analysis_results(
    sample_causal_analysis, sample_gap_analysis, sample_prediction_result
):
    """Combined analysis results from multiple agents."""
    return [sample_causal_analysis, sample_gap_analysis, sample_prediction_result]


@pytest.fixture
def minimal_analysis_result():
    """Minimal analysis result for edge case testing."""
    return {
        "agent": "test_agent",
        "analysis_type": "test",
        "status": "completed",
    }


@pytest.fixture
def failed_analysis_result():
    """Failed analysis result for error handling."""
    return {
        "agent": "test_agent",
        "analysis_type": "test",
        "status": "failed",
        "error": "Analysis failed due to insufficient data",
    }


# ============================================================================
# STATE FIXTURES
# ============================================================================


@pytest.fixture
def base_explainer_state():
    """Base state for explainer testing."""
    return {
        "query": "What is driving sales performance?",
        "analysis_results": [],
        "user_expertise": "analyst",
        "output_format": "narrative",
        "focus_areas": None,
        "analysis_context": None,
        "user_context": None,
        "conversation_history": None,
        "extracted_insights": None,
        "narrative_structure": None,
        "key_themes": None,
        "executive_summary": None,
        "detailed_explanation": None,
        "narrative_sections": None,
        "visual_suggestions": None,
        "follow_up_questions": None,
        "related_analyses": None,
        "assembly_latency_ms": 0,
        "reasoning_latency_ms": 0,
        "generation_latency_ms": 0,
        "total_latency_ms": 0,
        "model_used": None,
        "errors": [],
        "warnings": [],
        "status": "pending",
    }


@pytest.fixture
def assembled_state(base_explainer_state, sample_analysis_results):
    """State after context assembly phase."""
    return {
        **base_explainer_state,
        "analysis_results": sample_analysis_results,
        "analysis_context": {
            "agents_involved": ["causal_impact", "gap_analyzer", "prediction_synthesizer"],
            "analysis_types": ["effect_estimation", "opportunity_detection", "ensemble_prediction"],
            "total_results": 3,
            "has_causal_findings": True,
            "has_predictions": True,
            "has_recommendations": True,
            "key_metrics": {
                "effect_estimate": 0.23,
                "opportunity_value": 450000,
                "prediction_confidence": 0.92,
            },
        },
        "assembly_latency_ms": 15,
        "status": "assembled",
    }


@pytest.fixture
def reasoned_state(assembled_state):
    """State after deep reasoning phase."""
    return {
        **assembled_state,
        "extracted_insights": [
            {
                "insight_id": "1",
                "category": "finding",
                "statement": "Marketing campaign shows 23% sales uplift with high confidence",
                "confidence": 0.89,
                "priority": 1,
                "actionability": "immediate",
                "source_agent": "causal_impact",
            },
            {
                "insight_id": "2",
                "category": "opportunity",
                "statement": "Northeast region presents $450K opportunity",
                "confidence": 0.85,
                "priority": 2,
                "actionability": "immediate",
                "source_agent": "gap_analyzer",
            },
        ],
        "narrative_structure": {
            "sections": ["overview", "causal_findings", "opportunities", "recommendations"],
            "emphasis": "actionable_insights",
            "detail_level": "balanced",
        },
        "key_themes": ["sales_uplift", "regional_opportunity", "high_confidence"],
        "reasoning_latency_ms": 25,
        "status": "reasoned",
    }


# ============================================================================
# MOCK FIXTURES
# ============================================================================


@pytest.fixture
def mock_conversation_store():
    """Mock conversation store with sample history."""
    history = [
        {
            "role": "user",
            "content": "Show me Q3 performance",
            "timestamp": "2024-01-01T10:00:00Z",
        },
        {
            "role": "assistant",
            "content": "Q3 showed 15% growth...",
            "timestamp": "2024-01-01T10:01:00Z",
        },
    ]
    return MockConversationStore(history)


@pytest.fixture
def mock_llm():
    """Mock LLM for testing LLM-enabled mode."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"insights": [], "narrative": "Test narrative"}'
        )
    )
    return llm


# ============================================================================
# EXPERTISE LEVEL FIXTURES
# ============================================================================


@pytest.fixture
def executive_state(base_explainer_state, sample_analysis_results):
    """State for executive audience."""
    return {
        **base_explainer_state,
        "analysis_results": sample_analysis_results,
        "user_expertise": "executive",
        "output_format": "brief",
    }


@pytest.fixture
def analyst_state(base_explainer_state, sample_analysis_results):
    """State for analyst audience."""
    return {
        **base_explainer_state,
        "analysis_results": sample_analysis_results,
        "user_expertise": "analyst",
        "output_format": "narrative",
    }


@pytest.fixture
def data_scientist_state(base_explainer_state, sample_analysis_results):
    """State for data scientist audience."""
    return {
        **base_explainer_state,
        "analysis_results": sample_analysis_results,
        "user_expertise": "data_scientist",
        "output_format": "structured",
    }
