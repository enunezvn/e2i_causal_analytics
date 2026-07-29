"""End-to-end tests for the 4-stage ClassificationPipeline."""

from src.agents.orchestrator.classifier import ClassificationPipeline
from src.agents.orchestrator.classifier.schemas import RoutingPattern


class TestClassificationPipeline:
    def setup_method(self):
        self.pipeline = ClassificationPipeline(llm_client=None, enable_llm_layer=False)

    async def test_single_domain_causal_single_agent(self):
        result = await self.pipeline.classify(
            "What caused the decline — was it due to fewer rep visits?"
        )
        assert result.routing_pattern == RoutingPattern.SINGLE_AGENT
        assert result.target_agents == ["causal_impact"]
        assert result.confidence >= 0.5

    async def test_gibberish_clarification_needed(self):
        result = await self.pipeline.classify("asdf qwerty zxcv")
        assert result.routing_pattern == RoutingPattern.CLARIFICATION_NEEDED
        assert result.target_agents == []

    async def test_independent_multi_domain_parallel(self):
        result = await self.pipeline.classify(
            "Compare TRx market share for Kisqali vs its competitors over the last 6 months, "
            "explain what's driving the difference, and recommend where to focus reps next quarter"
        )
        assert result.routing_pattern == RoutingPattern.PARALLEL_DELEGATION
        assert len(result.target_agents) >= 2

    async def test_dependent_multi_part_tool_composer(self):
        result = await self.pipeline.classify(
            "Which regions are underperforming on Remibrutinib conversion rate, and for those "
            "regions, what would be the ROI of shifting 20% more rep capacity there?"
        )
        assert result.routing_pattern == RoutingPattern.TOOL_COMPOSER
        assert result.target_agents == ["tool_composer"]
        assert len(result.sub_questions) == 2
        assert len(result.dependencies) >= 1

    async def test_stages_populated_for_logging(self):
        result = await self.pipeline.classify("What is the impact of rep visits on TRx?")
        assert result.stages is not None
        assert result.stages.features.raw_query
        assert result.stages.domain_mapping is not None
        assert result.stages.dependency_analysis is not None

    async def test_latency_measured(self):
        result = await self.pipeline.classify("What drove the TRx change?")
        assert result.classification_latency_ms > 0.0

    async def test_used_llm_layer_honest_false(self):
        """used_llm_layer must reflect an ACTUAL LLM invocation. With no
        client and the layer disabled it must be False even for complex
        queries that request escalation (the old code reported True whenever
        escalation was merely requested)."""
        result = await self.pipeline.classify(
            "Compare TRx for Kisqali vs competitors, explain the difference, predict next "
            "quarter, and design an experiment to test whether speaker programs help"
        )
        assert result.used_llm_layer is False

    async def test_followup_context_passthrough(self):
        result = await self.pipeline.classify(
            "What is the impact of rep visits on TRx?",
            is_followup=True,
            context_source="conversation_history",
        )
        assert result.is_followup is True
        assert result.context_source == "conversation_history"
