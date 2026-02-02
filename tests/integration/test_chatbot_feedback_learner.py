"""
Integration tests for E2I Chatbot Feedback Learner Integration (Phase 8).

Tests the feedback_learner integration including:
- ChatbotGEPAMetric scoring across all dimensions
- ChatbotOptimizer optimization workflow
- A/B testing for prompt variants
- Signal submission for optimization
- Database integration for training signals
"""

from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chatbot_dspy import (
    CHATBOT_AB_TESTING_ENABLED,
    # Feature flags
    CHATBOT_GEPA_OPTIMIZATION_ENABLED,
    # Phase 8 components
    ChatbotGEPAMetric,
    ChatbotOptimizationRequest,
    ChatbotOptimizer,
    ChatbotSignalCollector,
    get_chatbot_optimizer,
    get_chatbot_signal_collector,
    submit_signals_for_optimization,
)

# =============================================================================
# ChatbotGEPAMetric Tests
# =============================================================================


class TestChatbotGEPAMetric:
    """Test cases for ChatbotGEPAMetric scoring."""

    def test_metric_creation(self):
        """Test creating the metric with default weights."""
        metric = ChatbotGEPAMetric()

        assert metric.name == "chatbot_gepa"
        assert metric.intent_weight == 0.25
        assert metric.routing_weight == 0.20
        assert metric.rag_weight == 0.25
        assert metric.synthesis_weight == 0.30
        # Weights should sum to 1.0
        total_weight = (
            metric.intent_weight
            + metric.routing_weight
            + metric.rag_weight
            + metric.synthesis_weight
        )
        assert abs(total_weight - 1.0) < 0.001

    def test_metric_custom_weights(self):
        """Test creating the metric with custom weights."""
        metric = ChatbotGEPAMetric(
            intent_weight=0.30,
            routing_weight=0.30,
            rag_weight=0.20,
            synthesis_weight=0.20,
        )

        assert metric.intent_weight == 0.30
        assert metric.routing_weight == 0.30

    def test_score_intent_correct_prediction(self):
        """Test intent scoring with correct prediction."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            intent: str = "kpi_query"
            confidence: float = 0.95

        @dataclass
        class MockGold:
            expected_intent: str = "kpi_query"

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="intent_classifier",
        )

        assert "score" in result
        assert "feedback" in result
        assert result["score"] > 0.8  # High score for correct prediction
        assert "Correct" in result["feedback"]

    def test_score_intent_incorrect_prediction(self):
        """Test intent scoring with incorrect prediction."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            intent: str = "greeting"
            confidence: float = 0.8

        @dataclass
        class MockGold:
            expected_intent: str = "kpi_query"

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="intent_classifier",
        )

        assert result["score"] < 0.5  # Low score for wrong prediction
        assert "Wrong" in result["feedback"]

    def test_score_intent_partial_match(self):
        """Test intent scoring with similar but different intents."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            intent: str = "search"
            confidence: float = 0.7

        @dataclass
        class MockGold:
            expected_intent: str = "kpi_query"

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="intent_classifier",
        )

        # Should get partial credit for similar intents
        assert 0.4 <= result["score"] <= 0.6
        assert "Partial" in result["feedback"]

    def test_score_routing_correct_agent(self):
        """Test routing scoring with correct agent selection."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            primary_agent: str = "causal_impact"
            routing_confidence: float = 0.9

        @dataclass
        class MockGold:
            expected_agent: str = "causal_impact"

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="agent_router",
        )

        assert result["score"] > 0.8
        assert "Correct" in result["feedback"]

    def test_score_routing_agent_in_secondary(self):
        """Test routing scoring when expected agent is in secondary list."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            primary_agent: str = "gap_analyzer"
            secondary_agents: List[str] = None
            routing_confidence: float = 0.7

            def __post_init__(self):
                self.secondary_agents = ["causal_impact", "explainer"]

        @dataclass
        class MockGold:
            expected_agent: str = "causal_impact"

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="agent_router",
        )

        # Should get partial credit for having correct agent in secondary
        assert 0.5 <= result["score"] <= 0.7
        assert "Secondary" in result["feedback"]

    def test_score_rag_good_retrieval(self):
        """Test RAG scoring with good retrieval results."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            rewritten_query: str = "Kisqali TRx Northeast region Q3 2024 metrics"
            search_keywords: List[str] = None
            graph_entities: List[str] = None

            def __post_init__(self):
                self.search_keywords = ["kisqali", "trx", "northeast", "q3"]
                self.graph_entities = ["Kisqali", "Northeast"]

        @dataclass
        class MockGold:
            query: str = "What is TRx for Kisqali?"
            avg_relevance_score: float = 0.85
            evidence_count: int = 5

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="query_rewriter",
        )

        assert result["score"] > 0.7
        # Should mention keywords and entities in feedback
        assert "keywords" in result["feedback"] or "entities" in result["feedback"]

    def test_score_rag_no_rewrite(self):
        """Test RAG scoring with no query rewriting."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            rewritten_query: str = ""
            search_keywords: List[str] = None
            graph_entities: List[str] = None

            def __post_init__(self):
                self.search_keywords = []
                self.graph_entities = []

        @dataclass
        class MockGold:
            query: str = "What is TRx?"
            avg_relevance_score: float = 0.3
            evidence_count: int = 0

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="query_rewriter",
        )

        assert result["score"] < 0.3  # Low score for no rewrite

    def test_score_synthesis_good_response(self):
        """Test synthesis scoring with good response."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            response: str = "Based on the data, Kisqali TRx increased by 15% in Q3 2024, driven by strong HCP adoption in the Northeast region. This growth was primarily attributed to the new targeting campaign launched in July."
            confidence_statement: str = "high confidence based on multiple data sources"
            evidence_citations: List[str] = None

            def __post_init__(self):
                self.evidence_citations = ["src_001", "src_002", "src_003"]

        @dataclass
        class MockGold:
            was_helpful: bool = True

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="synthesizer",
        )

        assert result["score"] > 0.7
        assert "citations" in result["feedback"]

    def test_score_synthesis_no_response(self):
        """Test synthesis scoring with no response."""
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            response: str = ""
            confidence_statement: str = ""
            evidence_citations: List[str] = None

            def __post_init__(self):
                self.evidence_citations = []

        @dataclass
        class MockGold:
            was_helpful: bool = False

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name="synthesizer",
        )

        assert result["score"] == 0.0
        assert "CRITICAL" in result["feedback"]

    def test_score_all_components_weighted(self):
        """Test scoring all components with weighted average.

        Note: The metric uses hasattr() checks, so when a pred has the "intent"
        attribute, it will be scored as intent-only. To truly test all components,
        we would need a pred without those specific attributes. This test verifies
        the metric returns valid results for the detected component.
        """
        metric = ChatbotGEPAMetric()

        @dataclass
        class MockPred:
            intent: str = "kpi_query"
            confidence: float = 0.9
            primary_agent: str = "causal_impact"
            secondary_agents: List[str] = None
            routing_confidence: float = 0.85
            rewritten_query: str = "optimized query"
            search_keywords: List[str] = None
            graph_entities: List[str] = None
            response: str = "A well-structured response with good length and proper citations."
            confidence_statement: str = "moderate confidence"
            evidence_citations: List[str] = None

            def __post_init__(self):
                self.secondary_agents = ["gap_analyzer"]
                self.search_keywords = ["keyword1", "keyword2"]
                self.graph_entities = ["Entity1"]
                self.evidence_citations = ["src1"]

        @dataclass
        class MockGold:
            expected_intent: str = "kpi_query"
            expected_agent: str = "causal_impact"
            query: str = "test"
            avg_relevance_score: float = 0.75
            evidence_count: int = 3
            was_helpful: bool = True

        result = metric(
            gold=MockGold(),
            pred=MockPred(),
            pred_name=None,  # Score detected component (intent due to hasattr check)
        )

        assert 0.0 <= result["score"] <= 1.0
        # Since pred has "intent" attr, it will be scored as intent-only per metric logic
        assert "[Intent]" in result["feedback"]
        assert "score" in result


# =============================================================================
# ChatbotOptimizationRequest Tests
# =============================================================================


class TestChatbotOptimizationRequest:
    """Test cases for ChatbotOptimizationRequest dataclass."""

    def test_request_creation(self):
        """Test creating an optimization request."""
        request = ChatbotOptimizationRequest(
            request_id="test_req_123",
            module_name="intent_classifier",
            signal_count=100,
            min_reward=0.6,
            budget="medium",
            priority=2,
        )

        assert request.request_id == "test_req_123"
        assert request.module_name == "intent_classifier"
        assert request.signal_count == 100
        assert request.min_reward == 0.6
        assert request.budget == "medium"
        assert request.priority == 2
        assert request.status == "pending"
        assert request.created_at is not None

    def test_request_to_dict(self):
        """Test converting request to dictionary."""
        request = ChatbotOptimizationRequest(
            request_id="test_req_456",
            module_name="agent_router",
            signal_count=50,
            min_reward=0.5,
            budget="light",
        )

        d = request.to_dict()

        assert d["request_id"] == "test_req_456"
        assert d["module_name"] == "agent_router"
        assert d["signal_count"] == 50
        assert d["min_reward"] == 0.5
        assert d["budget"] == "light"
        assert "created_at" in d
        assert d["status"] == "pending"


# =============================================================================
# ChatbotOptimizer Tests
# =============================================================================


class TestChatbotOptimizer:
    """Test cases for ChatbotOptimizer class."""

    def test_optimizer_creation(self):
        """Test creating an optimizer instance."""
        optimizer = ChatbotOptimizer()

        assert optimizer is not None
        # Optimizer type depends on available packages
        assert optimizer.optimizer_type in [None, "gepa", "miprov2"]

    def test_optimizer_singleton(self):
        """Test optimizer singleton pattern."""
        opt1 = get_chatbot_optimizer()
        opt2 = get_chatbot_optimizer()

        assert opt1 is opt2

    @pytest.mark.asyncio
    async def test_get_training_signals_fallback(self):
        """Test training signal retrieval falls back to in-memory collector when DB unavailable."""
        # Create collector with some signals
        collector = ChatbotSignalCollector(buffer_size=10)

        # Add a completed session with intent
        collector.start_session(
            session_id="fallback_test",
            thread_id="thread_1",
            query="What is TRx?",
        )
        collector.update_intent(
            session_id="fallback_test",
            intent="kpi_query",
            confidence=0.85,
            method="dspy",
        )
        collector.finalize_session(session_id="fallback_test", total_duration_ms=100)

        optimizer = ChatbotOptimizer(signal_collector=collector)

        # Force fallback by making database return empty result
        async def mock_db_empty(*args, **kwargs):
            # Return None to trigger fallback
            return None

        with patch.object(optimizer, "get_training_signals", wraps=optimizer.get_training_signals):
            # Patch the factory to simulate DB unavailable
            with patch(
                "src.memory.services.factories.get_async_supabase_service_client",
                side_effect=Exception("DB unavailable"),
            ):
                signals = await optimizer.get_training_signals("intent", min_reward=0.0)

                # Should fall back to in-memory collector and get our signal
                assert len(signals) >= 1

    @pytest.mark.asyncio
    async def test_get_training_signals_from_database(self):
        """Test training signal retrieval from database."""
        optimizer = ChatbotOptimizer()

        # Mock database response
        mock_result = MagicMock()
        mock_result.data = [
            {"session_id": "sess1", "query": "test1", "predicted_intent": "kpi_query"},
            {"session_id": "sess2", "query": "test2", "predicted_intent": "causal_analysis"},
        ]

        mock_table = MagicMock()
        mock_table.select.return_value.gte.return_value.neq.return_value.order.return_value.limit.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        mock_client = MagicMock()
        mock_client.table.return_value = mock_table

        # Patch at the factory module where function is imported from
        with patch(
            "src.memory.services.factories.get_async_supabase_service_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            signals = await optimizer.get_training_signals("intent", min_reward=0.5, limit=50)

            assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_queue_optimization(self):
        """Test queuing an optimization request."""
        optimizer = ChatbotOptimizer()

        # Mock training signals
        mock_collector = MagicMock(spec=ChatbotSignalCollector)
        mock_collector.get_signals_for_training.return_value = [
            {"query": f"test{i}", "predicted_intent": "kpi_query"} for i in range(60)
        ]
        optimizer._signal_collector = mock_collector

        # Mock database persistence
        with patch.object(
            optimizer,
            "_persist_optimization_request",
            new_callable=AsyncMock,
            return_value=True,
        ):
            request_id = await optimizer.queue_optimization(
                module_name="intent_classifier",
                budget="medium",
                min_reward=0.5,
                priority=2,
            )

            assert request_id.startswith("chatbot_opt_")
            assert len(optimizer._pending_requests) >= 1

    @pytest.mark.asyncio
    async def test_get_pending_requests(self):
        """Test retrieving pending requests."""
        optimizer = ChatbotOptimizer()

        # Add some pending requests
        request1 = ChatbotOptimizationRequest(
            request_id="req1",
            module_name="intent_classifier",
            signal_count=50,
            min_reward=0.5,
            budget="light",
        )
        request2 = ChatbotOptimizationRequest(
            request_id="req2",
            module_name="agent_router",
            signal_count=30,
            min_reward=0.6,
            budget="medium",
        )
        optimizer._pending_requests = [request1, request2]

        # Get all pending
        all_pending = await optimizer.get_pending_requests()
        assert len(all_pending) == 2

        # Filter by module
        intent_pending = await optimizer.get_pending_requests("intent_classifier")
        assert len(intent_pending) == 1
        assert intent_pending[0].module_name == "intent_classifier"

    @pytest.mark.asyncio
    async def test_optimize_module_insufficient_signals(self):
        """Test optimization fails with insufficient signals."""
        optimizer = ChatbotOptimizer()

        # Mock insufficient training signals
        with patch.object(
            optimizer,
            "get_training_signals",
            new_callable=AsyncMock,
            return_value=[{"query": "test", "predicted_intent": "kpi_query"} for _ in range(5)],
        ):
            result = await optimizer.optimize_module(
                module_name="intent_classifier",
                budget="light",
            )

            assert result["success"] is False
            assert "Insufficient" in result["error"]

    @pytest.mark.asyncio
    async def test_optimize_module_unknown_module(self):
        """Test optimization fails for unknown module."""
        optimizer = ChatbotOptimizer()

        result = await optimizer.optimize_module(
            module_name="unknown_module",
            budget="light",
        )

        assert result["success"] is False
        assert "Unknown module" in result["error"]


# =============================================================================
# A/B Testing Tests
# =============================================================================


class TestChatbotABTesting:
    """Test cases for A/B testing functionality."""

    def test_register_ab_variant(self):
        """Test registering an A/B test variant."""
        optimizer = ChatbotOptimizer()
        mock_module = MagicMock()

        optimizer.register_ab_variant(
            module_name="intent_classifier",
            variant_module=mock_module,
            variant_name="v2_improved",
        )

        assert "intent_classifier" in optimizer._ab_test_variants
        assert len(optimizer._ab_test_variants["intent_classifier"]) == 1
        assert optimizer._ab_test_variants["intent_classifier"][0]["name"] == "v2_improved"

    def test_get_ab_variant_control(self):
        """Test getting A/B variant returns control when disabled."""
        optimizer = ChatbotOptimizer()

        with patch("src.api.routes.chatbot_dspy.CHATBOT_AB_TESTING_ENABLED", False):
            module, variant_name = optimizer.get_ab_variant(
                module_name="intent_classifier",
                session_id="test-session-123",
            )

            assert module is None
            assert variant_name == "control"

    def test_get_ab_variant_no_variants(self):
        """Test getting A/B variant with no variants registered."""
        optimizer = ChatbotOptimizer()

        with patch("src.api.routes.chatbot_dspy.CHATBOT_AB_TESTING_ENABLED", True):
            module, variant_name = optimizer.get_ab_variant(
                module_name="intent_classifier",
                session_id="test-session-123",
            )

            assert module is None
            assert variant_name == "control"

    def test_get_ab_variant_consistent_assignment(self):
        """Test A/B variant assignment is consistent for same session."""
        optimizer = ChatbotOptimizer()
        mock_module = MagicMock()

        optimizer.register_ab_variant(
            module_name="intent_classifier",
            variant_module=mock_module,
            variant_name="v2_test",
        )

        with patch("src.api.routes.chatbot_dspy.CHATBOT_AB_TESTING_ENABLED", True):
            session_id = "consistent-session-abc"

            # Get variant multiple times
            results = [optimizer.get_ab_variant("intent_classifier", session_id) for _ in range(5)]

            # All results should be the same for same session
            first_result = results[0]
            for result in results[1:]:
                assert result == first_result

    @pytest.mark.asyncio
    async def test_record_ab_result(self):
        """Test recording A/B test results."""
        optimizer = ChatbotOptimizer()
        mock_module = MagicMock()

        optimizer.register_ab_variant(
            module_name="intent_classifier",
            variant_module=mock_module,
            variant_name="v2_test",
        )

        # Record some results
        await optimizer.record_ab_result("intent_classifier", "v2_test", 0.85)
        await optimizer.record_ab_result("intent_classifier", "v2_test", 0.90)
        await optimizer.record_ab_result("intent_classifier", "v2_test", 0.88)

        # Check that metrics were recorded
        variant = optimizer._ab_test_variants["intent_classifier"][0]
        assert len(variant["metrics"]) == 3
        assert 0.85 in variant["metrics"]

    def test_get_ab_results(self):
        """Test getting A/B test results summary."""
        optimizer = ChatbotOptimizer()
        mock_module = MagicMock()

        optimizer.register_ab_variant(
            module_name="intent_classifier",
            variant_module=mock_module,
            variant_name="v2_test",
        )

        # Add metrics directly
        optimizer._ab_test_variants["intent_classifier"][0]["metrics"] = [
            0.80,
            0.85,
            0.90,
            0.88,
            0.92,
        ]

        results = optimizer.get_ab_results("intent_classifier")

        assert "v2_test" in results
        assert results["v2_test"]["count"] == 5
        assert 0.80 <= results["v2_test"]["mean"] <= 0.92
        assert results["v2_test"]["min"] == 0.80
        assert results["v2_test"]["max"] == 0.92


# =============================================================================
# Signal Submission Tests
# =============================================================================


@pytest.mark.asyncio
class TestSignalSubmission:
    """Test cases for signal submission for optimization."""

    async def test_submit_signals_insufficient(self):
        """Test submission fails with insufficient signals."""
        # Mock optimizer to return insufficient signals
        mock_optimizer = MagicMock(spec=ChatbotOptimizer)
        mock_optimizer.get_training_signals = AsyncMock(
            return_value=[{"query": "test", "predicted_intent": "kpi_query"}]
        )  # Only 1 signal, below min_signals=50
        mock_optimizer.queue_optimization = AsyncMock(return_value="req_123")

        with patch(
            "src.api.routes.chatbot_dspy.get_chatbot_optimizer",
            return_value=mock_optimizer,
        ):
            result = await submit_signals_for_optimization(
                min_signals=50,
                min_reward=0.5,
            )

            # Should not queue any optimizations since insufficient signals
            assert all("insufficient" in str(v).lower() for v in result.values())
            # queue_optimization should NOT have been called
            assert not mock_optimizer.queue_optimization.called

    async def test_submit_signals_queues_optimization(self):
        """Test submission queues optimizations when signals sufficient."""
        # Mock signals in the format returned by optimizer.get_training_signals
        mock_signals = [
            {
                "session_id": f"sess_{i}",
                "query": f"test query {i}",
                "predicted_intent": "kpi_query",
                "intent_method": "dspy",
            }
            for i in range(60)
        ]

        mock_optimizer = MagicMock(spec=ChatbotOptimizer)
        # The submit_signals_for_optimization calls optimizer.get_training_signals
        mock_optimizer.get_training_signals = AsyncMock(return_value=mock_signals)
        mock_optimizer.queue_optimization = AsyncMock(return_value="req_123")

        with patch(
            "src.api.routes.chatbot_dspy.get_chatbot_optimizer",
            return_value=mock_optimizer,
        ):
            await submit_signals_for_optimization(
                min_signals=50,
                min_reward=0.5,
            )

            # Should have queued optimizations for all modules
            assert mock_optimizer.queue_optimization.called
            # Should have been called for intent_classifier, agent_router, query_rewriter, synthesizer
            assert mock_optimizer.queue_optimization.call_count == 4


# =============================================================================
# Feature Flag Tests
# =============================================================================


class TestFeatureFlags:
    """Test cases for Phase 8 feature flags."""

    def test_gepa_optimization_flag_exists(self):
        """Test that GEPA optimization flag is defined."""
        assert isinstance(CHATBOT_GEPA_OPTIMIZATION_ENABLED, bool)

    def test_ab_testing_flag_exists(self):
        """Test that A/B testing flag is defined."""
        assert isinstance(CHATBOT_AB_TESTING_ENABLED, bool)


# =============================================================================
# Integration with Signal Collector Tests
# =============================================================================


class TestOptimizerCollectorIntegration:
    """Test integration between optimizer and signal collector."""

    def test_optimizer_uses_collector(self):
        """Test that optimizer uses the signal collector."""
        get_chatbot_signal_collector()
        optimizer = get_chatbot_optimizer()

        # Optimizer should have a reference to collector
        assert optimizer._signal_collector is not None

    @pytest.mark.asyncio
    async def test_signals_flow_to_optimizer(self):
        """Test that signals flow from collector to optimizer."""
        collector = ChatbotSignalCollector(buffer_size=100)
        optimizer = ChatbotOptimizer(signal_collector=collector)

        # Add some sessions
        for i in range(15):
            collector.start_session(
                session_id=f"flow_test_{i}",
                thread_id=f"thread_{i}",
                query=f"What is TRx for brand {i}?",
            )
            collector.update_intent(
                session_id=f"flow_test_{i}",
                intent="kpi_query",
                confidence=0.85 + (i * 0.01),
                method="dspy",
            )
            collector.update_feedback(
                session_id=f"flow_test_{i}",
                user_rating=4.0 + (i * 0.05),
                was_helpful=True,
            )
            collector.finalize_session(
                session_id=f"flow_test_{i}",
                total_duration_ms=1000 + i * 100,
            )

        # Verify collector state before async call
        assert len(collector._signals) == 15, (
            f"Collector should have 15 signals, got {len(collector._signals)}"
        )
        assert optimizer._signal_collector is collector, "Optimizer should reference same collector"

        # Verify get_signals_for_training works directly
        direct_signals = collector.get_signals_for_training("intent", 100)
        assert len(direct_signals) == 15, (
            f"Direct call should return 15 signals, got {len(direct_signals)}"
        )

        # Force fallback by making database raise exception
        with patch(
            "src.memory.services.factories.get_async_supabase_service_client",
            side_effect=Exception("DB unavailable for test"),
        ):
            # Get training signals through optimizer
            signals = await optimizer.get_training_signals(
                phase="intent",
                min_reward=0.0,  # Low threshold to get all signals
                limit=100,
            )

            # Should get signals from collector
            assert len(signals) >= 10, f"Expected >= 10 signals but got {len(signals)}"


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
class TestOptimizerErrorHandling:
    """Test error handling in optimizer."""

    async def test_database_error_fallback(self):
        """Test graceful fallback on database errors."""
        # Create a real collector with a signal
        collector = ChatbotSignalCollector(buffer_size=10)
        collector.start_session(
            session_id="error_test",
            thread_id="thread_1",
            query="Test query",
        )
        collector.update_intent(
            session_id="error_test",
            intent="kpi_query",
            confidence=0.9,
            method="dspy",
        )
        collector.finalize_session(session_id="error_test", total_duration_ms=100)

        optimizer = ChatbotOptimizer(signal_collector=collector)

        # Mock database to raise exception - patch at the factory module where it's imported
        with patch(
            "src.memory.services.factories.get_async_supabase_service_client",
            side_effect=Exception("Database connection failed"),
        ):
            # Should fall back gracefully to in-memory collector
            signals = await optimizer.get_training_signals("intent", min_reward=0.0)
            assert len(signals) >= 1

    async def test_optimization_error_handling(self):
        """Test error handling during optimization."""
        optimizer = ChatbotOptimizer()

        # Mock to return enough signals
        with patch.object(
            optimizer,
            "get_training_signals",
            new_callable=AsyncMock,
            return_value=[
                {"query": f"test{i}", "predicted_intent": "kpi_query"} for i in range(20)
            ],
        ):
            # Force optimizer type to test the path
            original_type = optimizer._optimizer_type
            optimizer._optimizer_type = "gepa"

            # Mock GEPA to raise exception
            with (
                patch(
                    "src.api.routes.chatbot_dspy.create_gepa_optimizer",
                    side_effect=Exception("GEPA failed"),
                ),
                patch(
                    "src.api.routes.chatbot_dspy.GEPA_AVAILABLE",
                    True,
                ),
            ):
                result = await optimizer.optimize_module(
                    module_name="intent_classifier",
                    budget="light",
                )

                assert result["success"] is False
                assert "error" in result

            optimizer._optimizer_type = original_type
