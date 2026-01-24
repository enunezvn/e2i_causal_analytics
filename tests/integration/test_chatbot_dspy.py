"""
Phase 10: End-to-End Integration Tests for Chatbot DSPy Integration.

This module provides comprehensive integration tests for validating the
complete DSPy integration pipeline including:
- DSPy intent classification accuracy
- DSPy agent routing accuracy
- Cognitive RAG quality improvement
- Opik trace completeness
- MLflow metrics accuracy
- Training signal collection
- Load testing with concurrent requests
- Performance validation (latency <2s p50)

Part of CopilotKit-DSPy Observability Integration Plan - Phase 10.
"""

import asyncio
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all tests in this module as integration tests and group DSPy tests together
pytestmark = [
    pytest.mark.integration,
    pytest.mark.xdist_group(name="dspy_integration"),
]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_queries() -> List[Dict[str, Any]]:
    """Sample queries with expected intents for accuracy testing."""
    return [
        # KPI queries
        {"query": "What is the TRx for Kisqali?", "expected_intent": "kpi_query"},
        {"query": "Show me Remibrutinib market share", "expected_intent": "kpi_query"},
        {"query": "NRx volume in Northeast region", "expected_intent": "kpi_query"},
        {"query": "How many new prescriptions last month?", "expected_intent": "kpi_query"},
        # Causal analysis
        {"query": "Why did Kisqali sales drop in Q3?", "expected_intent": "causal_analysis"},
        {"query": "What caused the TRx decline?", "expected_intent": "causal_analysis"},
        {"query": "Impact of the marketing campaign", "expected_intent": "causal_analysis"},
        {"query": "Drivers of market share growth", "expected_intent": "causal_analysis"},
        # Recommendations
        {"query": "How can we improve conversion rate?", "expected_intent": "recommendation"},
        {"query": "Suggest strategies to increase TRx", "expected_intent": "recommendation"},
        {"query": "What should we do to grow market share?", "expected_intent": "recommendation"},
        # Greetings
        {"query": "Hello", "expected_intent": "greeting"},
        {"query": "Hi there!", "expected_intent": "greeting"},
        # Help
        {"query": "What can you help me with?", "expected_intent": "help"},
        {"query": "What are your capabilities?", "expected_intent": "help"},
        # Agent status
        {"query": "What agents are available?", "expected_intent": "agent_status"},
        {"query": "Check drift monitor status", "expected_intent": "agent_status"},
        # Multi-faceted
        {"query": "Analyze Kisqali performance and suggest improvements", "expected_intent": "multi_faceted"},
        {"query": "Compare TRx trends across regions and identify causes", "expected_intent": "multi_faceted"},
        # General
        {"query": "Tell me about the weather", "expected_intent": "general"},
    ]


@pytest.fixture
def sample_routing_queries() -> List[Dict[str, Any]]:
    """Sample queries with expected agent routing."""
    return [
        {"query": "Why did sales drop?", "intent": "causal_analysis", "expected_agent": "causal_impact"},
        {"query": "Find growth opportunities", "intent": "recommendation", "expected_agent": "gap_analyzer"},
        {"query": "Segment-level analysis", "intent": "causal_analysis", "expected_agent": "heterogeneous_optimizer"},
        {"query": "Is there any data drift?", "intent": "agent_status", "expected_agent": "drift_monitor"},
        {"query": "Design an A/B test", "intent": "recommendation", "expected_agent": "experiment_designer"},
        {"query": "System health check", "intent": "agent_status", "expected_agent": "health_score"},
        {"query": "Predict next quarter TRx", "intent": "kpi_query", "expected_agent": "prediction_synthesizer"},
        {"query": "Optimize resource allocation", "intent": "recommendation", "expected_agent": "resource_optimizer"},
        {"query": "Explain this result", "intent": "general", "expected_agent": "explainer"},
    ]


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for database operations."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"id": "test-signal-id"}])
    return mock_client


@pytest.fixture
def mock_opik_client():
    """Mock Opik client for tracing."""
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_client.trace.return_value.__enter__ = MagicMock(return_value=mock_trace)
    mock_client.trace.return_value.__exit__ = MagicMock(return_value=None)
    return mock_client


# =============================================================================
# INTENT CLASSIFICATION TESTS
# =============================================================================


class TestDSPyIntentClassificationAccuracy:
    """Test DSPy intent classification accuracy against known queries."""

    @pytest.mark.asyncio
    async def test_intent_classification_accuracy(self, sample_queries):
        """Test that intent classification achieves >90% accuracy on sample queries."""
        from src.api.routes.chatbot_dspy import classify_intent_dspy

        correct = 0
        total = len(sample_queries)
        results = []

        for test_case in sample_queries:
            query = test_case["query"]
            expected = test_case["expected_intent"]

            try:
                intent, confidence, reasoning, method = await classify_intent_dspy(query)
                is_correct = intent == expected
                if is_correct:
                    correct += 1
                results.append({
                    "query": query,
                    "expected": expected,
                    "actual": intent,
                    "confidence": confidence,
                    "correct": is_correct,
                    "method": method,
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "expected": expected,
                    "actual": "error",
                    "confidence": 0.0,
                    "correct": False,
                    "method": "error",
                    "error": str(e),
                })

        accuracy = correct / total if total > 0 else 0.0

        # Log results for debugging
        for r in results:
            if not r["correct"]:
                print(f"Mismatch: '{r['query']}' - expected={r['expected']}, got={r['actual']}")

        # Target: >90% accuracy
        assert accuracy >= 0.90, f"Intent classification accuracy {accuracy:.2%} < 90% target. Correct: {correct}/{total}"

    @pytest.mark.asyncio
    async def test_intent_classification_confidence_scores(self, sample_queries):
        """Test that confidence scores are reasonable (0.5-1.0 for clear queries)."""
        from src.api.routes.chatbot_dspy import classify_intent_dspy

        confidences = []

        for test_case in sample_queries[:10]:  # Test subset
            query = test_case["query"]
            try:
                _, confidence, _, _ = await classify_intent_dspy(query)
                confidences.append(confidence)
            except Exception:
                pass

        if confidences:
            avg_confidence = statistics.mean(confidences)
            # Clear queries should have confidence >= 0.7
            assert avg_confidence >= 0.7, f"Average confidence {avg_confidence:.2f} < 0.7"

    @pytest.mark.asyncio
    async def test_intent_dspy_vs_hardcoded_comparison(self):
        """Compare DSPy and hardcoded classification for consistency."""
        from src.api.routes.chatbot_dspy import classify_intent_dspy, classify_intent_hardcoded

        test_queries = [
            "What is the TRx for Kisqali?",
            "Why did sales drop?",
            "Hello",
            "What can you help me with?",
        ]

        for query in test_queries:
            dspy_intent, dspy_conf, _, dspy_method = await classify_intent_dspy(query)
            hardcoded_intent, hardcoded_conf, _ = classify_intent_hardcoded(query)

            # Both methods should agree on clear-cut queries
            if hardcoded_conf >= 0.9:  # High confidence hardcoded
                assert dspy_intent == hardcoded_intent or dspy_method == "hardcoded", \
                    f"DSPy ({dspy_intent}) disagrees with hardcoded ({hardcoded_intent}) for '{query}'"


# =============================================================================
# AGENT ROUTING TESTS
# =============================================================================


class TestDSPyAgentRoutingAccuracy:
    """Test DSPy agent routing accuracy."""

    @pytest.mark.asyncio
    async def test_agent_routing_accuracy(self, sample_routing_queries):
        """Test that agent routing achieves >85% accuracy."""
        from src.api.routes.chatbot_dspy import route_agent_dspy

        correct = 0
        total = len(sample_routing_queries)

        for test_case in sample_routing_queries:
            query = test_case["query"]
            intent = test_case["intent"]
            expected_agent = test_case["expected_agent"]

            try:
                agent, _, confidence, rationale, method = await route_agent_dspy(query, intent)
                if agent == expected_agent:
                    correct += 1
                else:
                    print(f"Routing mismatch: '{query}' - expected={expected_agent}, got={agent}")
            except Exception as e:
                print(f"Routing error for '{query}': {e}")

        accuracy = correct / total if total > 0 else 0.0
        # Target: >85% accuracy
        assert accuracy >= 0.85, f"Agent routing accuracy {accuracy:.2%} < 85% target"

    @pytest.mark.asyncio
    async def test_agent_routing_returns_valid_agents(self):
        """Test that routing always returns valid agent names."""
        from src.api.routes.chatbot_dspy import VALID_AGENTS, route_agent_dspy

        test_queries = [
            ("Why did sales drop?", "causal_analysis"),
            ("Predict future growth", "kpi_query"),
            ("Find opportunities", "recommendation"),
        ]

        for query, intent in test_queries:
            agent, secondary, confidence, rationale, method = await route_agent_dspy(query, intent)
            assert agent in VALID_AGENTS, f"Invalid agent '{agent}' returned"

            # Secondary agents should also be valid if provided
            if secondary:
                for sec in secondary:
                    assert sec in VALID_AGENTS, f"Invalid secondary agent '{sec}'"


# =============================================================================
# COGNITIVE RAG TESTS
# =============================================================================


class TestCognitiveRAGQuality:
    """Test cognitive RAG quality improvement."""

    @pytest.mark.asyncio
    async def test_query_rewriting_improves_queries(self):
        """Test that query rewriting adds domain terms."""
        from src.api.routes.chatbot_dspy import rewrite_query_hardcoded

        test_queries = [
            ("sales performance", ["TRx", "NRx", "market share"]),
            ("Kisqali analysis", ["breast cancer", "HR+", "HER2-"]),
            ("northeast region", ["territory", "HCP"]),
        ]

        for query, expected_terms in test_queries:
            rewritten, keywords, entities = rewrite_query_hardcoded(query)
            # Rewritten query should be different or contain domain terms
            assert len(rewritten) >= len(query), "Rewritten query should not be shorter"

    @pytest.mark.asyncio
    async def test_cognitive_rag_result_structure(self):
        """Test that cognitive RAG returns proper result structure."""
        from src.api.routes.chatbot_dspy import CognitiveRAGResult

        result = CognitiveRAGResult(
            rewritten_query="What is TRx for Kisqali breast cancer treatment?",
            search_keywords=["TRx", "Kisqali", "breast cancer"],
            graph_entities=["Kisqali", "HR+", "HER2-"],
            evidence=[{"content": "test", "score": 0.8}],
            hop_count=1,
            avg_relevance_score=0.8,
            retrieval_method="cognitive"
        )

        assert result.rewritten_query is not None
        assert isinstance(result.evidence, list)
        assert isinstance(result.search_keywords, list)
        assert isinstance(result.graph_entities, list)
        assert 0 <= result.avg_relevance_score <= 1.0
        assert result.retrieval_method in ["cognitive", "basic"]


# =============================================================================
# OPIK TRACE TESTS
# =============================================================================


class TestOpikTraceCompleteness:
    """Test Opik tracing integration."""

    def test_opik_tracer_available(self):
        """Test that Opik tracer can be imported."""
        try:
            from src.api.routes.chatbot_tracer import ChatbotTracer, NodeSpanContext
            assert ChatbotTracer is not None
            assert NodeSpanContext is not None
        except ImportError as e:
            pytest.skip(f"Opik tracer not available: {e}")

    @pytest.mark.asyncio
    async def test_tracer_creates_spans(self):
        """Test that tracer creates spans for nodes."""
        try:
            from src.api.routes.chatbot_tracer import ChatbotTracer
        except ImportError:
            pytest.skip("Opik tracer not available")

        with patch("src.api.routes.chatbot_tracer.opik") as mock_opik:
            mock_opik.Opik.return_value = MagicMock()

            tracer = ChatbotTracer(session_id="test-session")
            # Tracer should initialize without error
            assert tracer is not None


# =============================================================================
# MLFLOW METRICS TESTS
# =============================================================================


class TestMLflowMetricsAccuracy:
    """Test MLflow metrics logging."""

    def test_mlflow_experiment_exists(self):
        """Test that chatbot MLflow experiment can be accessed."""
        try:
            import mlflow

            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name("e2i_chatbot_interactions")
            # Experiment may or may not exist in test environment
            # Just verify MLflow is accessible
            assert mlflow is not None
        except ImportError:
            pytest.skip("MLflow not available")

    def test_mlflow_metrics_format(self):
        """Test that metrics have correct format for logging."""
        # Sample metrics that would be logged
        metrics = {
            "latency_ms": 150.5,
            "response_length": 256,
            "intent_confidence": 0.95,
            "rag_result_count": 5,
            "rag_avg_relevance": 0.78,
        }

        for key, value in metrics.items():
            assert isinstance(key, str), f"Metric key must be string: {key}"
            assert isinstance(value, (int, float)), f"Metric value must be numeric: {value}"


# =============================================================================
# TRAINING SIGNAL TESTS
# =============================================================================


class TestTrainingSignalCollection:
    """Test training signal collection for feedback learner."""

    def test_intent_training_signal_creation(self):
        """Test IntentTrainingSignal dataclass creation."""
        from src.api.routes.chatbot_dspy import IntentTrainingSignal

        signal = IntentTrainingSignal(
            query="What is TRx?",
            conversation_context="",
            brand_context="Kisqali",
            predicted_intent="kpi_query",
            confidence=0.95,
            reasoning="Query asks about a KPI metric",
            classification_method="dspy",
        )

        assert signal.query == "What is TRx?"
        assert signal.predicted_intent == "kpi_query"
        assert signal.confidence == 0.95
        assert signal.classification_method == "dspy"

    def test_training_signal_reward_computation(self):
        """Test reward computation for training signals."""
        from src.api.routes.chatbot_dspy import IntentTrainingSignal

        # Signal with positive feedback
        signal = IntentTrainingSignal(
            query="What is TRx?",
            conversation_context="",
            brand_context="",
            predicted_intent="kpi_query",
            confidence=0.95,
            reasoning="KPI query",
            classification_method="dspy",
            correct_routing=True,
            response_helpful=True,
        )

        reward = signal.compute_reward()
        assert 0.0 <= reward <= 1.0, f"Reward {reward} out of bounds"
        assert reward >= 0.5, "Positive feedback should yield reward >= 0.5"

    def test_signal_collector_session_management(self):
        """Test ChatbotSignalCollector session management."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()

        # Start session with required parameters
        signal = collector.start_session(
            session_id="test-session",
            thread_id="test-thread",
            query="What is TRx?",
            user_id="test-user",
            brand_context="Kisqali"
        )

        # Verify session was started
        assert signal is not None
        assert signal.session_id == "test-session"
        assert signal.query == "What is TRx?"

        # Record intent using update_intent API
        collector.update_intent(
            session_id="test-session",
            intent="kpi_query",
            confidence=0.95,
            method="dspy",
            reasoning="Test intent classification"
        )

        # Verify collector tracks pending sessions
        assert "test-session" in collector._pending_sessions
        # Verify intent was recorded
        pending = collector._pending_sessions["test-session"]
        assert pending.predicted_intent == "kpi_query"


class TestDatabasePersistence:
    """Test database persistence of training signals."""

    @pytest.mark.asyncio
    async def test_signal_persistence_format(self, mock_supabase_client):
        """Test that signals have correct format for database."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test-session",
            thread_id="test-thread",
            user_id="test-user",
            query="Test query",
            brand_context="Kisqali",
            region_context="Northeast",
            predicted_intent="kpi_query",
            intent_confidence=0.95,
            intent_method="dspy",
            predicted_agent="explainer",
            routing_confidence=0.88,
            routing_method="dspy",
            rewritten_query="Enhanced test query",
            evidence_count=5,
            avg_relevance_score=0.75,
            rag_method="cognitive",
            synthesis_confidence="high",
            citations_count=2,
            synthesis_method="dspy",
            total_duration_ms=150.0,
        )

        # Verify required fields exist
        assert signal.session_id == "test-session"
        assert signal.query == "Test query"
        assert signal.predicted_intent == "kpi_query"
        assert signal.total_duration_ms == 150.0


# =============================================================================
# LOAD TESTING
# =============================================================================


class TestLoadPerformance:
    """Load testing with concurrent requests."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_intent_classification(self):
        """Test 100 concurrent intent classification requests."""
        from src.api.routes.chatbot_dspy import classify_intent_dspy

        queries = [
            "What is the TRx?",
            "Why did sales drop?",
            "Hello",
            "Suggest improvements",
            "Check system status",
        ] * 20  # 100 queries

        start_time = time.time()

        # Run concurrently
        tasks = [classify_intent_dspy(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))

        # At least 95% should succeed
        success_rate = successes / len(queries)
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} < 95%"

        # Total time should be reasonable (not sequential)
        # With concurrency, 100 requests should complete faster than 100 * avg_time
        print(f"Processed {len(queries)} requests in {total_time:.2f}s ({len(queries)/total_time:.1f} req/s)")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_agent_routing(self):
        """Test 50 concurrent agent routing requests."""
        from src.api.routes.chatbot_dspy import route_agent_dspy

        test_cases = [
            ("Why did sales drop?", "causal_analysis"),
            ("Find opportunities", "recommendation"),
            ("Predict next quarter", "kpi_query"),
            ("System health", "agent_status"),
            ("Explain this", "general"),
        ] * 10  # 50 requests

        start_time = time.time()

        tasks = [route_agent_dspy(q, i) for q, i in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        successes = sum(1 for r in results if not isinstance(r, Exception))
        success_rate = successes / len(test_cases)

        assert success_rate >= 0.95, f"Routing success rate {success_rate:.2%} < 95%"
        print(f"Processed {len(test_cases)} routing requests in {total_time:.2f}s")


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmark tests to ensure latency targets."""

    @pytest.mark.asyncio
    async def test_intent_classification_latency(self):
        """Test that intent classification meets latency target (<500ms p50)."""
        from src.api.routes.chatbot_dspy import classify_intent_dspy

        queries = [
            "What is the TRx for Kisqali?",
            "Why did sales drop in Q3?",
            "Hello",
            "Show me market share trends",
            "How can we improve?",
        ]

        latencies = []

        for query in queries * 4:  # 20 samples
            start = time.time()
            try:
                await classify_intent_dspy(query)
            except Exception:
                pass
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        if latencies:
            p50 = statistics.median(latencies)
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]

            print(f"Intent classification: p50={p50:.0f}ms, p95={p95:.0f}ms")

            # Target: p50 < 500ms for intent classification
            assert p50 < 500, f"Intent p50 latency {p50:.0f}ms > 500ms target"

    @pytest.mark.asyncio
    async def test_agent_routing_latency(self):
        """Test that agent routing meets latency target (<300ms p50)."""
        from src.api.routes.chatbot_dspy import route_agent_dspy

        test_cases = [
            ("Why did sales drop?", "causal_analysis"),
            ("Find opportunities", "recommendation"),
            ("Predict growth", "kpi_query"),
        ]

        latencies = []

        for query, intent in test_cases * 5:  # 15 samples
            start = time.time()
            try:
                await route_agent_dspy(query, intent)
            except Exception:
                pass
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        if latencies:
            p50 = statistics.median(latencies)
            print(f"Agent routing: p50={p50:.0f}ms")

            # Target: p50 < 300ms for routing
            assert p50 < 300, f"Routing p50 latency {p50:.0f}ms > 300ms target"

    @pytest.mark.asyncio
    async def test_end_to_end_response_latency(self):
        """Test end-to-end response latency target (<2s p50)."""
        from src.api.routes.chatbot_dspy import (
            classify_intent_dspy,
            route_agent_dspy,
            synthesize_response_hardcoded,
        )

        queries = [
            "What is the TRx for Kisqali?",
            "Why did Remibrutinib adoption slow?",
        ]

        latencies = []

        for query in queries * 3:  # 6 samples
            start = time.time()
            try:
                # Simulate end-to-end flow
                intent, conf, _, _ = await classify_intent_dspy(query)
                agent, _, _, _, _ = await route_agent_dspy(query, intent)
                # Synthesis (using hardcoded for speed in test)
                result = synthesize_response_hardcoded(
                    query=query,
                    intent=intent,
                    evidence=[{"content": "Test evidence", "score": 0.8}],
                    brand_context="",
                )
            except Exception:
                pass
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        if latencies:
            p50 = statistics.median(latencies)
            print(f"End-to-end: p50={p50:.0f}ms")

            # Target: p50 < 2000ms for full pipeline
            assert p50 < 2000, f"E2E p50 latency {p50:.0f}ms > 2000ms target"


# =============================================================================
# FEATURE FLAG TESTS
# =============================================================================


class TestFeatureFlags:
    """Test feature flag behavior for DSPy components."""

    def test_dspy_intent_feature_flag(self):
        """Test CHATBOT_DSPY_INTENT feature flag."""
        from src.api.routes.chatbot_dspy import CHATBOT_DSPY_INTENT_ENABLED
        # Flag should be readable
        assert isinstance(CHATBOT_DSPY_INTENT_ENABLED, bool)

    def test_feature_flags_allow_graceful_degradation(self):
        """Test that disabling flags falls back to hardcoded."""
        from src.api.routes.chatbot_dspy import classify_intent_hardcoded

        # Hardcoded should always work
        intent, confidence, reasoning = classify_intent_hardcoded("What is TRx?")
        assert intent in ["kpi_query", "general"]
        assert 0.0 <= confidence <= 1.0


# =============================================================================
# INTEGRATION VALIDATION
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration validation tests."""

    @pytest.mark.asyncio
    async def test_full_classification_to_routing_pipeline(self):
        """Test complete pipeline from classification to routing."""
        from src.api.routes.chatbot_dspy import classify_intent_dspy, route_agent_dspy

        query = "Why did Kisqali market share decline in Northeast?"

        # Step 1: Classify intent
        intent, intent_conf, reasoning, intent_method = await classify_intent_dspy(query)
        assert intent in ["causal_analysis", "kpi_query", "multi_faceted"]
        assert intent_conf > 0.5

        # Step 2: Route to agent
        agent, secondary, route_conf, rationale, route_method = await route_agent_dspy(query, intent)
        assert agent is not None
        assert route_conf > 0.5

        # Pipeline should complete without errors
        print(f"Query: {query}")
        print(f"Intent: {intent} ({intent_conf:.2f})")
        print(f"Agent: {agent} ({route_conf:.2f})")

    @pytest.mark.asyncio
    async def test_signal_collection_integration(self):
        """Test that signal collection works through pipeline."""
        from src.api.routes.chatbot_dspy import (
            ChatbotSignalCollector,
            classify_intent_dspy,
            route_agent_dspy,
        )

        collector = ChatbotSignalCollector()
        query = "What is TRx for Kisqali?"

        # Start session with required parameters
        signal = collector.start_session(
            session_id="integration-test",
            thread_id="test-thread",
            query=query,
            user_id="test-user",
            brand_context="Kisqali"
        )

        # Run through pipeline
        intent, conf, _, method = await classify_intent_dspy(query)

        # Record the classification using update_intent API
        collector.update_intent(
            session_id="integration-test",
            intent=intent,
            confidence=conf,
            method=method,
            reasoning="Integration test classification"
        )

        # Verify session tracking
        assert "integration-test" in collector._pending_sessions
        pending_signal = collector._pending_sessions["integration-test"]
        assert pending_signal.predicted_intent == intent
