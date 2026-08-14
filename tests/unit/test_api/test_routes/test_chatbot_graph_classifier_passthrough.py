"""orchestrator_node passthrough of 4-stage classifier observability.

Covers the chatbot-graph boundary: OrchestratorAgent output →
orchestrator_node result dict (routing_pattern / classification_latency_ms /
used_llm_layer on both the state update and its metadata).
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.api.routes.chatbot_graph import orchestrator_node

_ORCH_RESULT = {
    "response_text": "Causal analysis complete.",
    "response_confidence": 0.9,
    "agents_dispatched": ["causal_impact"],
    "successful_agents": ["causal_impact"],
    "failed_agents": [],
    "failure_details": [],
    "status": "completed",
    "classification": {
        "routing_pattern": "SINGLE_AGENT",
        "target_agents": ["causal_impact"],
        "confidence": 0.7,
        "classification_latency_ms": 2.9,
        "used_llm_layer": False,
    },
    "routing_pattern": "SINGLE_AGENT",
    "used_llm_layer": False,
    # #1582: which subsystem produced the dispatch plan for this turn.
    "routing_authority": "pipeline",
}


def _state(intent="causal_analysis"):
    return {
        "intent": intent,
        "query": "Why did TRx drop?",
        "session_id": "user-1~sess",
        "user_id": "user-1",
        "rag_context": [],
        "progress_steps": [],
        "metadata": {},
    }


def _mock_orchestrator(result=_ORCH_RESULT):
    orch = MagicMock()
    orch.run = AsyncMock(return_value=result)
    return orch


class TestOrchestratorNodeClassifierPassthrough:
    async def test_classifier_fields_reach_node_result(self):
        with patch(
            "src.api.routes.chatbot_graph.get_orchestrator",
            return_value=_mock_orchestrator(),
        ):
            result = await orchestrator_node(_state())

        assert result["routing_pattern"] == "SINGLE_AGENT"
        # The API-facing latency is the PIPELINE's own value from the
        # classification dump, not the node-total classification time.
        assert result["classification_latency_ms"] == 2.9
        assert result["used_llm_layer"] is False
        assert result["routing_authority"] == "pipeline"
        assert result["metadata"]["routing_pattern"] == "SINGLE_AGENT"
        assert result["metadata"]["classification_latency_ms"] == 2.9
        assert result["metadata"]["used_llm_layer"] is False
        assert result["metadata"]["routing_authority"] == "pipeline"

    async def test_abstaining_pattern_still_reports_legacy_authority(self):
        """#1582: the shape that motivated the field.

        An answered kpi_query turn carries the pipeline's abstention verbatim
        AND an explicit statement that legacy routing produced the plan, so the
        pair can no longer be read as one contradictory routing decision.
        """
        abstained = {
            **_ORCH_RESULT,
            "agents_dispatched": ["explainer"],
            "successful_agents": ["explainer"],
            "routing_pattern": "CLARIFICATION_NEEDED",
            "routing_authority": "legacy",
        }
        with patch(
            "src.api.routes.chatbot_graph.get_orchestrator",
            return_value=_mock_orchestrator(abstained),
        ):
            result = await orchestrator_node(_state())

        assert result["routing_pattern"] == "CLARIFICATION_NEEDED"
        assert result["routing_authority"] == "legacy"
        assert result["agents_dispatched"] == ["explainer"]
        assert result["metadata"]["routing_authority"] == "legacy"

    async def test_absent_classifier_fields_pass_none(self):
        bare = {
            k: v
            for k, v in _ORCH_RESULT.items()
            if k
            not in (
                "classification",
                "routing_pattern",
                "used_llm_layer",
                "routing_authority",
            )
        }
        with patch(
            "src.api.routes.chatbot_graph.get_orchestrator",
            return_value=_mock_orchestrator(bare),
        ):
            result = await orchestrator_node(_state())

        assert result["routing_pattern"] is None
        assert result["classification_latency_ms"] is None
        assert result["used_llm_layer"] is None
        assert result["routing_authority"] is None

    async def test_non_orchestrator_intent_untouched(self):
        with patch(
            "src.api.routes.chatbot_graph.get_orchestrator",
            return_value=_mock_orchestrator(),
        ) as get_orch:
            result = await orchestrator_node(_state(intent="greeting"))

        get_orch.assert_not_called()
        assert "routing_pattern" not in result
        assert "routing_authority" not in result
