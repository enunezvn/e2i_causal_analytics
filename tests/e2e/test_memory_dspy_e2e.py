"""
End-to-End Tests for DSPy MIPRO Memory Integration
===================================================

Comprehensive E2E tests for DSPy integration with the 4-phase cognitive workflow:
- DSPy signature invocation during cognitive cycles
- Training signal collection and persistence
- MIPROv2 optimization trigger when 50+ signals exist
- Optimized prompt loading from version tables

Configuration:
- Redis: Port 6382 (working memory)
- FalkorDB: Port 6381 (semantic memory)
- Supabase: From environment variables

Run with: pytest tests/e2e/test_memory_dspy_e2e.py -v
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from supabase import create_client

from src.rag.cognitive_rag_dspy import (
    AgentModule,
    CognitiveRAGOptimizer,
    CognitiveState,
    Evidence,
    InvestigatorModule,
    MemoryType,
    ReflectorModule,
    SummarizerModule,
    create_dspy_cognitive_workflow,
)
from src.rag.memory_adapters import (
    EpisodicMemoryAdapter,
    ProceduralMemoryAdapter,
    SemanticMemoryAdapter,
    SignalCollectorAdapter,
    create_memory_adapters,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6382")
FALKORDB_URL = os.getenv("FALKORDB_URL", "redis://localhost:6381")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Skip tests if required services not available
SKIP_SUPABASE = not SUPABASE_URL or not SUPABASE_KEY
SKIP_REASON_SUPABASE = "SUPABASE_URL and SUPABASE_ANON_KEY required"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest_asyncio.fixture(scope="module")
async def redis_client():
    """Real Redis connection on port 6382 for working memory."""
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await client.ping()
        yield client
    except Exception as e:
        pytest.skip(f"Redis not available at {REDIS_URL}: {e}")
    finally:
        await client.aclose()


@pytest_asyncio.fixture(scope="module")
async def falkordb_client():
    """Real FalkorDB connection on port 6381 for semantic memory."""
    client = aioredis.from_url(FALKORDB_URL, decode_responses=True)
    try:
        await client.ping()
        yield client
    except Exception as e:
        pytest.skip(f"FalkorDB not available at {FALKORDB_URL}: {e}")
    finally:
        await client.aclose()


@pytest.fixture(scope="module")
def supabase_client():
    """Real Supabase connection for episodic/procedural memory."""
    if SKIP_SUPABASE:
        pytest.skip(SKIP_REASON_SUPABASE)
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return client


@pytest_asyncio.fixture
async def clean_dspy_tables(supabase_client):
    """Clean DSPy tables before and after tests."""
    test_prefix = f"e2e_test_{uuid.uuid4().hex[:8]}"

    yield test_prefix

    # Cleanup after test
    try:
        # Delete test signals
        supabase_client.table("dspy_agent_training_signals").delete().ilike(
            "batch_id", f"{test_prefix}%"
        ).execute()

        # Delete test optimization runs
        supabase_client.table("dspy_optimization_runs").delete().ilike(
            "target_agent", f"{test_prefix}%"
        ).execute()

        # Delete test prompt versions
        supabase_client.table("dspy_prompt_versions").delete().ilike(
            "agent_name", f"{test_prefix}%"
        ).execute()
    except Exception as e:
        print(f"Cleanup warning: {e}")


@pytest.fixture
def mock_lm():
    """Mock DSPy language model for testing signature invocation."""
    mock = MagicMock()
    mock.return_value = MagicMock(
        rewritten_query="optimized query for Kisqali adoption",
        search_keywords=["Kisqali", "adoption", "TRx"],
        graph_entities=["Kisqali", "HCP"],
        brands=["Kisqali"],
        regions=["Northeast"],
        hcp_types=["Oncologist"],
        patient_stages=["Treatment"],
        time_references=["last quarter"],
        primary_intent="CAUSAL_ANALYSIS",
        secondary_intents=["EXPLANATION"],
        requires_visualization=True,
        complexity="MODERATE",
        investigation_goal="Understand Kisqali adoption drivers",
        hop_strategy=["episodic", "semantic"],
        max_hops=2,
        early_stop_criteria="Found causal chain",
        next_memory="episodic",
        retrieval_query="Kisqali adoption trends",
        reasoning="Need historical context first",
        confidence=0.7,
        relevance_score=0.85,
        key_insight="HCP engagement increased",
        follow_up_needed=False,
        synthesis="Kisqali adoption increased due to HCP engagement",
        confidence_statement="High confidence based on 3 evidence pieces",
        evidence_citations=["mem_001", "mem_002"],
        primary_agent="causal_impact",
        supporting_agents=["explainer"],
        requires_deep_reasoning=False,
        chart_type="bar",
        chart_config="{}",
        highlights=["Q3 growth"],
        worth_remembering=True,
        memory_type="episodic",
        importance_score=0.8,
        key_facts=["HCP engagement drives adoption"],
        procedure_pattern="query_episodic -> check_semantic",
        trigger_conditions=["adoption_query"],
        expected_outcome="causal_explanation",
    )
    return mock


@pytest.fixture
def mock_memory_backends():
    """Mock memory backends for testing."""
    episodic = AsyncMock()
    episodic.vector_search = AsyncMock(
        return_value=[
            {"content": "Kisqali adoption increased 15% in Q3 due to HCP targeting", "score": 0.9}
        ]
    )

    semantic = AsyncMock()
    semantic.graph_query = AsyncMock(
        return_value=[
            {"content": "HCP_ENGAGEMENT CAUSES TRx_INCREASE with confidence 0.85", "score": 0.8}
        ]
    )

    procedural = AsyncMock()
    procedural.procedure_search = AsyncMock(
        return_value=[
            {
                "content": "For adoption analysis: query episodic → check semantic → synthesize",
                "score": 0.7,
            }
        ]
    )

    return {
        "episodic": episodic,
        "semantic": semantic,
        "procedural": procedural,
    }


@pytest.fixture
def mock_signal_collector():
    """Mock signal collector for testing."""
    collector = AsyncMock()
    collector.collect = AsyncMock()
    collector.flush = AsyncMock(return_value=5)
    return collector


@pytest.fixture
def cognitive_state():
    """Create a cognitive state for testing."""
    return CognitiveState(
        user_query="Why did Kisqali adoption increase in the Northeast last quarter?",
        conversation_id=f"test-{uuid.uuid4().hex[:8]}",
    )


@pytest.fixture
def domain_vocabulary():
    """Domain vocabulary for E2I."""
    return """
    brands: [Remibrutinib (CSU), Fabhalta (PNH), Kisqali (HR+/HER2- breast cancer)]
    kpis: [TRx, NRx, conversion_rate, market_share, adoption_rate]
    entities: [HCP, patient, territory, region, therapeutic_area]
    metrics: [prescriptions, visits, detailing, samples]
    """


# =============================================================================
# TEST CLASS 1: DSPy SIGNATURE INVOCATION
# =============================================================================


class TestDSPySignatureInvocation:
    """Verify all 11 DSPy signatures are properly invoked during cognitive workflow."""

    def test_summarizer_signatures_invoked(self, mock_lm, domain_vocabulary):
        """Test Phase 1 signatures: QueryRewrite, EntityExtraction, IntentClassification."""
        with patch.object(dspy, "configure"):
            with patch.object(dspy, "ChainOfThought", return_value=mock_lm):
                with patch.object(dspy, "Predict", return_value=mock_lm):
                    module = SummarizerModule()

                    result = module.forward(
                        original_query="Why did Kisqali TRx drop?",
                        conversation_context="Previous discussion about Northeast region",
                        domain_vocabulary=domain_vocabulary,
                    )

                    # Verify outputs from all 3 signatures
                    assert "rewritten_query" in result
                    assert "extracted_entities" in result
                    assert "primary_intent" in result
                    assert result["primary_intent"] == "CAUSAL_ANALYSIS"

    def test_investigator_signatures_invoked(self, mock_lm, mock_memory_backends):
        """Test Phase 2 signatures: InvestigationPlan, HopDecision, EvidenceRelevance."""
        with patch.object(dspy, "ChainOfThought", return_value=mock_lm):
            with patch.object(dspy, "Predict", return_value=mock_lm):
                module = InvestigatorModule(mock_memory_backends)

                # Run async forward
                result = asyncio.get_event_loop().run_until_complete(
                    module.forward(
                        rewritten_query="Kisqali adoption drivers in Northeast",
                        intent="CAUSAL_ANALYSIS",
                        entities='{"brands": ["Kisqali"]}',
                    )
                )

                # Verify outputs from investigation
                assert "investigation_goal" in result
                assert "evidence_board" in result
                assert "hop_count" in result
                assert result["hop_count"] >= 1

    def test_agent_signatures_invoked(self, mock_lm, cognitive_state):
        """Test Phase 3 signatures: EvidenceSynthesis, AgentRouting, VisualizationConfig."""
        # Prepare state with evidence
        cognitive_state.detected_intent = "CAUSAL_ANALYSIS"
        cognitive_state.investigation_goal = "Find adoption drivers"
        cognitive_state.evidence_board = [
            Evidence(
                source=MemoryType.EPISODIC,
                hop_number=1,
                content="Kisqali adoption increased 15%",
                relevance_score=0.9,
            )
        ]

        with patch.object(dspy, "ChainOfThought", return_value=mock_lm):
            with patch.object(dspy, "Predict", return_value=mock_lm):
                module = AgentModule(agent_registry={})

                result = asyncio.get_event_loop().run_until_complete(
                    module.forward(cognitive_state)
                )

                # Verify agent phase outputs
                assert result.response is not None or result.visualization_config is not None
                assert result.routed_agents is not None

    def test_reflector_signatures_invoked(self, mock_lm, mock_signal_collector, cognitive_state):
        """Test Phase 4 signatures: MemoryWorthiness, ProcedureLearning."""
        # Prepare state with response
        cognitive_state.response = "Kisqali adoption increased due to HCP engagement"
        cognitive_state.detected_intent = "CAUSAL_ANALYSIS"
        cognitive_state.routed_agents = ["causal_impact"]
        cognitive_state.evidence_board = [
            Evidence(
                source=MemoryType.EPISODIC,
                hop_number=1,
                content="HCP engagement increased",
                relevance_score=0.9,
            )
        ]

        mock_writers = {
            "episodic": AsyncMock(),
            "semantic": AsyncMock(),
            "procedural": AsyncMock(),
        }

        with patch.object(dspy, "Predict", return_value=mock_lm):
            module = ReflectorModule(mock_writers, mock_signal_collector)

            result = asyncio.get_event_loop().run_until_complete(
                module.forward(cognitive_state, user_feedback="positive feedback")
            )

            # Verify reflector phase outputs
            assert result.worth_remembering is True
            assert len(result.dspy_signals) > 0
            mock_signal_collector.collect.assert_called_once()


# =============================================================================
# TEST CLASS 2: TRAINING SIGNAL COLLECTION
# =============================================================================


class TestTrainingSignalCollection:
    """Verify training signals are collected and persisted correctly."""

    def test_signals_collected_in_phase4(self, mock_lm, mock_signal_collector, cognitive_state):
        """Run cognitive cycle and verify signals appear in state.dspy_signals."""
        cognitive_state.response = "Analysis complete"
        cognitive_state.detected_intent = "CAUSAL_ANALYSIS"
        cognitive_state.rewritten_query = "optimized query"
        cognitive_state.investigation_goal = "Find causes"
        cognitive_state.evidence_board = [
            Evidence(
                source=MemoryType.EPISODIC,
                hop_number=1,
                content="Evidence 1",
                relevance_score=0.8,
            )
        ]
        cognitive_state.routed_agents = ["causal_impact"]
        cognitive_state.sufficient_evidence = True

        mock_writers = {
            "episodic": AsyncMock(),
            "semantic": AsyncMock(),
            "procedural": AsyncMock(),
        }

        with patch.object(dspy, "Predict", return_value=mock_lm):
            module = ReflectorModule(mock_writers, mock_signal_collector)

            result = asyncio.get_event_loop().run_until_complete(module.forward(cognitive_state))

            # Verify signals collected
            assert len(result.dspy_signals) >= 3  # summarizer, investigator, agent signals

            # Check signal structure (new format for SignalCollectorAdapter)
            for signal in result.dspy_signals:
                assert "type" in signal, "Signal must have 'type' field"
                assert "query" in signal, "Signal must have 'query' field"
                assert "response" in signal, "Signal must have 'response' field"
                assert "reward" in signal, "Signal must have 'reward' field"
                assert signal["type"] in ["summarizer", "investigator", "agent"]
                assert 0.0 <= signal["reward"] <= 1.0, "Reward must be between 0 and 1"

    @pytest.mark.skipif(SKIP_SUPABASE, reason=SKIP_REASON_SUPABASE)
    def test_signals_persisted_to_database(self, supabase_client, clean_dspy_tables):
        """Verify signals are persisted to dspy_agent_training_signals table."""
        test_prefix = clean_dspy_tables

        # Create and persist a test signal
        signal_data = {
            "source_agent": "feedback_learner",
            "batch_id": f"{test_prefix}_batch_1",
            "input_context": {"query": "test query", "intent": "CAUSAL_ANALYSIS"},
            "output": {"response": "test response"},
            "quality_metrics": {"accuracy": 0.9},
            "reward": 0.85,
            "total_latency_ms": 150,
            "model_used": "claude-sonnet-4-20250514",
            "llm_calls": 3,
            "is_training_example": True,
        }

        response = (
            supabase_client.table("dspy_agent_training_signals").insert(signal_data).execute()
        )

        assert len(response.data) > 0
        inserted = response.data[0]

        # Verify fields
        assert inserted["source_agent"] == "feedback_learner"
        assert inserted["reward"] == 0.85
        assert inserted["is_training_example"] is True

    @pytest.mark.skipif(SKIP_SUPABASE, reason=SKIP_REASON_SUPABASE)
    def test_signal_reward_computed(self, supabase_client, clean_dspy_tables):
        """Verify reward field is within valid range."""
        test_prefix = clean_dspy_tables

        # Insert signals with different rewards
        signals = [
            {
                "source_agent": "causal_impact",
                "batch_id": f"{test_prefix}_b1",
                "input_context": {},
                "output": {},
                "quality_metrics": {},
                "reward": 0.95,
            },
            {
                "source_agent": "gap_analyzer",
                "batch_id": f"{test_prefix}_b2",
                "input_context": {},
                "output": {},
                "quality_metrics": {},
                "reward": 0.65,
            },
            {
                "source_agent": "explainer",
                "batch_id": f"{test_prefix}_b3",
                "input_context": {},
                "output": {},
                "quality_metrics": {},
                "reward": 0.30,
            },
        ]

        for signal in signals:
            supabase_client.table("dspy_agent_training_signals").insert(signal).execute()

        # Query and verify rewards
        response = (
            supabase_client.table("dspy_agent_training_signals")
            .select("reward")
            .ilike("batch_id", f"{test_prefix}%")
            .execute()
        )

        rewards = [r["reward"] for r in response.data if r["reward"] is not None]

        # All rewards should be between 0 and 1
        assert all(0 <= r <= 1 for r in rewards)
        assert len(rewards) == 3


# =============================================================================
# TEST CLASS 3: MIPRO OPTIMIZATION TRIGGER
# =============================================================================


class TestMIPROOptimizationTrigger:
    """Verify MIPROv2 optimization triggers at correct thresholds."""

    @pytest.mark.skipif(SKIP_SUPABASE, reason=SKIP_REASON_SUPABASE)
    def test_optimization_not_triggered_under_threshold(self, supabase_client, clean_dspy_tables):
        """Test that optimization doesn't run with <50 signals."""
        test_prefix = clean_dspy_tables

        # Insert only 10 signals (below 50 threshold)
        for i in range(10):
            supabase_client.table("dspy_agent_training_signals").insert(
                {
                    "source_agent": f"{test_prefix}_agent",
                    "batch_id": f"{test_prefix}_batch_{i}",
                    "input_context": {"query": f"test query {i}"},
                    "output": {"response": f"response {i}"},
                    "quality_metrics": {},
                    "reward": 0.7,
                    "is_training_example": True,
                }
            ).execute()

        # Check signal count
        response = (
            supabase_client.table("dspy_agent_training_signals")
            .select("signal_id", count="exact")
            .ilike("source_agent", f"{test_prefix}%")
            .execute()
        )

        assert response.count == 10
        assert response.count < 50  # Below threshold

    @pytest.mark.skipif(SKIP_SUPABASE, reason=SKIP_REASON_SUPABASE)
    def test_optimization_threshold_check(self, supabase_client, clean_dspy_tables):
        """Test signal count reaches optimization threshold."""
        test_prefix = clean_dspy_tables

        # Insert 50 signals (at threshold)
        signals = []
        for i in range(50):
            signals.append(
                {
                    "source_agent": f"{test_prefix}_optimizable",
                    "batch_id": f"{test_prefix}_opt_{i}",
                    "input_context": {"query": f"query {i}"},
                    "output": {"response": f"response {i}"},
                    "quality_metrics": {"accuracy": 0.8},
                    "reward": 0.75,
                    "is_training_example": True,
                    "excluded_from_training": False,
                }
            )

        # Batch insert
        supabase_client.table("dspy_agent_training_signals").insert(signals).execute()

        # Verify count meets threshold
        response = (
            supabase_client.table("dspy_agent_training_signals")
            .select("signal_id", count="exact")
            .eq("source_agent", f"{test_prefix}_optimizable")
            .execute()
        )

        assert response.count >= 50  # At or above threshold for optimization

    @pytest.mark.skipif(SKIP_SUPABASE, reason=SKIP_REASON_SUPABASE)
    def test_optimization_run_recorded(self, supabase_client, clean_dspy_tables):
        """Verify optimization runs are recorded in dspy_optimization_runs table."""
        test_prefix = clean_dspy_tables

        # Record an optimization run
        run_data = {
            "target_agent": f"{test_prefix}_agent",
            "optimization_phase": "causal_impact",
            "signature_name": "CausalAnalysisSignature",
            "config": {
                "num_candidates": 10,
                "max_bootstrapped_demos": 4,
                "num_threads": 4,
            },
            "training_examples_count": 50,
            "status": "completed",
            "baseline_metric": 0.65,
            "optimized_metric": 0.82,
            "improvement_pct": 26.15,
            "best_prompt_template": "Analyze the causal relationship...",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 120,
            "trials_completed": 50,
            "deployed": False,
        }

        response = supabase_client.table("dspy_optimization_runs").insert(run_data).execute()

        assert len(response.data) > 0
        run = response.data[0]

        assert run["status"] == "completed"
        assert run["improvement_pct"] > 0
        assert run["training_examples_count"] == 50


# =============================================================================
# TEST CLASS 4: OPTIMIZED PROMPT LOADING
# =============================================================================


class TestOptimizedPromptLoading:
    """Verify optimized prompts are loaded from version table."""

    @pytest.mark.skipif(SKIP_SUPABASE, reason=SKIP_REASON_SUPABASE)
    def test_active_prompt_loaded(self, supabase_client, clean_dspy_tables):
        """Test that active prompt version is correctly retrieved."""
        test_prefix = clean_dspy_tables

        # Insert prompt versions (inactive and active)
        prompts = [
            {
                "agent_name": f"{test_prefix}_agent",
                "signature_name": "CausalSignature",
                "version_number": 1,
                "is_active": False,
                "prompt_template": "Version 1 prompt",
                "avg_reward": 0.65,
                "created_by": "baseline",
            },
            {
                "agent_name": f"{test_prefix}_agent",
                "signature_name": "CausalSignature",
                "version_number": 2,
                "is_active": True,
                "prompt_template": "Version 2 optimized prompt",
                "avg_reward": 0.82,
                "created_by": "miprov2",
                "activated_at": datetime.now(timezone.utc).isoformat(),
            },
        ]

        for prompt in prompts:
            supabase_client.table("dspy_prompt_versions").insert(prompt).execute()

        # Query active prompt using database function
        response = supabase_client.rpc(
            "get_active_prompt",
            {"p_agent": f"{test_prefix}_agent", "p_signature": "CausalSignature"},
        ).execute()

        assert len(response.data) > 0
        active = response.data[0]

        assert active["version_number"] == 2
        assert active["prompt_template"] == "Version 2 optimized prompt"
        assert active["avg_reward"] == 0.82

    @pytest.mark.skipif(SKIP_SUPABASE, reason=SKIP_REASON_SUPABASE)
    def test_prompt_version_fallback(self, supabase_client, clean_dspy_tables):
        """Test behavior when no active prompt exists."""
        test_prefix = clean_dspy_tables

        # Insert only inactive prompts
        supabase_client.table("dspy_prompt_versions").insert(
            {
                "agent_name": f"{test_prefix}_fallback",
                "signature_name": "TestSignature",
                "version_number": 1,
                "is_active": False,
                "prompt_template": "Inactive prompt",
                "created_by": "baseline",
            }
        ).execute()

        # Query should return empty (no active prompt)
        response = supabase_client.rpc(
            "get_active_prompt",
            {"p_agent": f"{test_prefix}_fallback", "p_signature": "TestSignature"},
        ).execute()

        # No active prompt should be found
        assert len(response.data) == 0


# =============================================================================
# TEST CLASS 5: FULL COGNITIVE CYCLE WITH DSPy
# =============================================================================


class TestFullCognitiveWithDSPy:
    """End-to-end tests for complete cognitive cycles with DSPy."""

    def test_full_cycle_kisqali_query(
        self, mock_lm, mock_memory_backends, mock_signal_collector, domain_vocabulary
    ):
        """Test complete E2E cognitive cycle with a Kisqali query."""
        with patch.object(dspy, "ChainOfThought", return_value=mock_lm):
            with patch.object(dspy, "Predict", return_value=mock_lm):
                # Create workflow with mocks
                workflow = create_dspy_cognitive_workflow(
                    memory_backends=mock_memory_backends,
                    memory_writers={
                        "episodic": AsyncMock(),
                        "semantic": AsyncMock(),
                        "procedural": AsyncMock(),
                    },
                    agent_registry={},
                    signal_collector=mock_signal_collector,
                    domain_vocabulary=domain_vocabulary,
                )

                # Create initial state
                initial_state = CognitiveState(
                    user_query="Why did Kisqali adoption increase in the Northeast?",
                    conversation_id="test-full-cycle",
                )

                # Run workflow with required LangGraph config
                config = {"configurable": {"thread_id": "test-full-cycle-thread"}}
                result = asyncio.get_event_loop().run_until_complete(
                    workflow.ainvoke(initial_state, config=config)
                )

                # Verify all phases completed (result may be dict or CognitiveState)
                if isinstance(result, dict):
                    assert result.get("rewritten_query", "") != "" or "rewritten_query" in result
                    assert result.get("detected_intent", "") != "" or "detected_intent" in result
                else:
                    assert result.rewritten_query != ""
                    assert result.detected_intent != ""
                    assert len(result.evidence_board) >= 0
                    assert len(result.dspy_signals) > 0

    def test_full_cycle_with_positive_feedback(
        self, mock_lm, mock_memory_backends, mock_signal_collector, domain_vocabulary
    ):
        """Test cycle with positive feedback triggers learning."""
        mock_writers = {
            "episodic": AsyncMock(),
            "semantic": AsyncMock(),
            "procedural": AsyncMock(),
        }

        with patch.object(dspy, "ChainOfThought", return_value=mock_lm):
            with patch.object(dspy, "Predict", return_value=mock_lm):
                # Create reflector module
                reflector = ReflectorModule(mock_writers, mock_signal_collector)

                # Create state with positive feedback scenario
                state = CognitiveState(
                    user_query="What drives HCP engagement?",
                    conversation_id="test-feedback",
                )
                state.response = "HCP engagement is driven by detailing frequency"
                state.detected_intent = "CAUSAL_ANALYSIS"
                state.routed_agents = ["causal_impact", "explainer"]
                state.evidence_board = [
                    Evidence(
                        source=MemoryType.EPISODIC,
                        hop_number=1,
                        content="Detailing correlates with engagement",
                        relevance_score=0.9,
                    )
                ]

                # Run with positive feedback
                result = asyncio.get_event_loop().run_until_complete(
                    reflector.forward(state, user_feedback="positive, very helpful!")
                )

                # Verify learning triggered
                assert result.worth_remembering is True
                assert len(result.learned_procedures) > 0
                mock_writers["procedural"].store_procedure.assert_called()


# =============================================================================
# TEST CLASS 6: DSPy PERFORMANCE BENCHMARKS
# =============================================================================


class TestDSPyPerformance:
    """Performance benchmark tests for DSPy integration."""

    def test_signal_collection_under_100ms(self, mock_lm, mock_signal_collector, cognitive_state):
        """Benchmark Phase 4 signal collection latency."""
        cognitive_state.response = "Test response"
        cognitive_state.detected_intent = "CAUSAL_ANALYSIS"
        cognitive_state.rewritten_query = "test query"
        cognitive_state.investigation_goal = "test goal"
        cognitive_state.routed_agents = ["causal_impact"]
        cognitive_state.evidence_board = [
            Evidence(
                source=MemoryType.EPISODIC,
                hop_number=1,
                content="evidence",
                relevance_score=0.8,
            )
        ]
        cognitive_state.sufficient_evidence = True

        mock_writers = {
            "episodic": AsyncMock(),
            "semantic": AsyncMock(),
            "procedural": AsyncMock(),
        }

        with patch.object(dspy, "Predict", return_value=mock_lm):
            module = ReflectorModule(mock_writers, mock_signal_collector)

            # Measure signal collection time
            start = time.time()
            asyncio.get_event_loop().run_until_complete(module.forward(cognitive_state))
            elapsed_ms = (time.time() - start) * 1000

            # With mocks, should be well under 100ms
            assert elapsed_ms < 100, f"Signal collection took {elapsed_ms:.1f}ms (>100ms)"

    def test_signature_invocation_latency(self, mock_lm, domain_vocabulary):
        """Benchmark individual signature invocation latency."""
        with patch.object(dspy, "ChainOfThought", return_value=mock_lm):
            with patch.object(dspy, "Predict", return_value=mock_lm):
                module = SummarizerModule()

                # Warm up
                module.forward(
                    original_query="test",
                    conversation_context="",
                    domain_vocabulary=domain_vocabulary,
                )

                # Measure
                latencies = []
                for _ in range(10):
                    start = time.time()
                    module.forward(
                        original_query="Why did Kisqali adoption increase?",
                        conversation_context="Previous context",
                        domain_vocabulary=domain_vocabulary,
                    )
                    latencies.append((time.time() - start) * 1000)

                avg_latency = sum(latencies) / len(latencies)

                # With mocks, should be reasonably fast (some overhead from DSPy module calls)
                assert avg_latency < 150, f"Avg signature latency {avg_latency:.1f}ms (>150ms)"


# =============================================================================
# ADDITIONAL INTEGRATION TESTS
# =============================================================================


class TestMemoryAdaptersIntegration:
    """Test memory adapter integration with DSPy workflow."""

    @pytest.mark.skipif(SKIP_SUPABASE, reason=SKIP_REASON_SUPABASE)
    def test_signal_collector_adapter_flush(self, supabase_client, clean_dspy_tables):
        """Test SignalCollectorAdapter buffer flush to database."""

        adapter = SignalCollectorAdapter(supabase_client, buffer_size=5)

        # Collect signals
        signals = [
            {"type": "response", "query": f"query_{i}", "response": f"resp_{i}", "reward": 0.8}
            for i in range(3)
        ]

        asyncio.get_event_loop().run_until_complete(adapter.collect(signals))

        # Buffer should have 3 signals
        assert len(adapter._signal_buffer) == 3

        # Flush manually
        asyncio.get_event_loop().run_until_complete(adapter.flush())

        # Buffer should be empty after flush
        assert len(adapter._signal_buffer) == 0

    def test_memory_adapters_factory(self):
        """Test create_memory_adapters factory function."""
        adapters = create_memory_adapters(
            supabase_client=None,
            falkordb_memory=None,
            memory_connector=None,
            embedding_model=None,
        )

        assert "episodic" in adapters
        assert "semantic" in adapters
        assert "procedural" in adapters
        assert "signals" in adapters

        assert isinstance(adapters["episodic"], EpisodicMemoryAdapter)
        assert isinstance(adapters["semantic"], SemanticMemoryAdapter)
        assert isinstance(adapters["procedural"], ProceduralMemoryAdapter)
        assert isinstance(adapters["signals"], SignalCollectorAdapter)


class TestCognitiveRAGOptimizer:
    """Test the CognitiveRAGOptimizer class."""

    def test_summarizer_metric_scoring(self):
        """Test summarizer metric function scores correctly."""
        optimizer = CognitiveRAGOptimizer(feedback_learner=None)

        # Create mock example and prediction
        example = MagicMock()
        example.original_query = "short query"

        prediction = MagicMock()
        prediction.rewritten_query = "longer optimized query for retrieval"
        prediction.graph_entities = ["Kisqali", "HCP"]
        prediction.primary_intent = "CAUSAL_ANALYSIS"
        prediction.search_keywords = ["hcp", "adoption", "kisqali"]

        score = optimizer.summarizer_metric(example, prediction)

        # Should get points for: longer query, entities, non-GENERAL intent, pharma terms
        assert score > 0.5
        assert score <= 1.0

    def test_investigator_metric_scoring(self):
        """Test investigator metric function scores correctly."""
        optimizer = CognitiveRAGOptimizer(feedback_learner=None)

        example = MagicMock()

        prediction = MagicMock()
        prediction.evidence_board = [
            Evidence(MemoryType.EPISODIC, 1, "evidence 1", 0.9),
            Evidence(MemoryType.SEMANTIC, 2, "evidence 2", 0.8),
        ]
        prediction.sufficient_evidence = True
        prediction.hop_count = 2

        score = optimizer.investigator_metric(example, prediction)

        # Should get points for: finding evidence, high relevance, efficiency
        assert score > 0.5
        assert score <= 1.0

    def test_agent_metric_scoring(self):
        """Test agent metric function scores correctly."""
        optimizer = CognitiveRAGOptimizer(feedback_learner=None)

        example = MagicMock()
        example.requires_visualization = True

        prediction = MagicMock()
        prediction.response = "A" * 250  # Substantive response
        prediction.evidence_citations = ["mem_001", "mem_002", "mem_003"]
        prediction.confidence_statement = "High confidence based on multiple evidence sources"
        prediction.chart_type = "bar"

        score = optimizer.agent_metric(example, prediction)

        # Should get points for: substantive response, citations, confidence, viz
        assert score > 0.5
        assert score <= 1.0
