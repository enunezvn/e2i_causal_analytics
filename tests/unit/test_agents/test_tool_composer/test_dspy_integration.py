"""
Tests for Tool Composer Agent DSPy Integration.

Tests the DSPy Hybrid role implementation including:
- Training signal dataclass
- Reward computation
- Signal collector
- Hybrid integration (prompt optimization)
- Singleton access patterns

Note: This module is marked to run sequentially (not in parallel) because
the dspy import has race conditions during parallel pytest-xdist execution.
"""

import pytest

# Mark entire module to run on same worker - prevents import race conditions
pytestmark = pytest.mark.xdist_group(name="dspy_integration")

from src.agents.tool_composer.dspy_integration import (
    DSPY_AVAILABLE,
    CompositionTrainingSignal,
    ToolComposerDSPyIntegration,
    ToolComposerSignalCollector,
    get_tool_composer_dspy_integration,
    get_tool_composer_signal_collector,
    reset_dspy_integration,
)


class TestCompositionTrainingSignal:
    """Tests for CompositionTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with defaults."""
        signal = CompositionTrainingSignal()

        assert signal.signal_id == ""
        assert signal.session_id == ""
        assert signal.query == ""
        assert signal.query_complexity == ""
        assert signal.entity_count == 0
        assert signal.domain_count == 0
        assert signal.sub_questions_count == 0
        assert signal.tools_planned == []
        assert signal.tools_succeeded == 0
        assert signal.tools_failed == 0
        assert signal.synthesis_confidence == 0.0
        assert signal.user_satisfaction is None
        assert signal.created_at is not None

    def test_custom_initialization(self):
        """Test signal with custom values."""
        signal = CompositionTrainingSignal(
            session_id="tc-session-123",
            query="Compare TRx trends and predict next quarter",
            query_complexity="complex",
            entity_count=4,
            domain_count=2,
        )

        assert signal.session_id == "tc-session-123"
        assert signal.query_complexity == "complex"
        assert signal.entity_count == 4
        assert signal.domain_count == 2

    def test_compute_reward_minimal(self):
        """Test reward with minimal data."""
        signal = CompositionTrainingSignal()
        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0
        assert reward < 0.5  # Minimal data should yield low reward

    def test_compute_reward_high_quality(self):
        """Test reward with high-quality composition."""
        signal = CompositionTrainingSignal(
            query_complexity="complex",
            domain_count=3,
            sub_questions_count=5,  # ~1.7 per domain
            decomposition_quality=0.9,
            tools_planned=["causal_effect", "gap_calc", "risk_scorer", "segment_ranker"],
            parallel_groups_count=2,
            tools_succeeded=4,
            tools_failed=0,
            synthesis_confidence=0.88,
            total_latency_ms=6000,  # Under 10s target
            user_satisfaction=5.0,
        )
        reward = signal.compute_reward()

        assert reward > 0.8  # High-quality should score well
        assert reward <= 1.0

    def test_compute_reward_execution_success(self):
        """Test that execution success affects reward."""
        # All tools succeeded
        signal_success = CompositionTrainingSignal(
            tools_planned=["tool1", "tool2", "tool3"],
            tools_succeeded=3,
            tools_failed=0,
        )

        # Some tools failed
        signal_failure = CompositionTrainingSignal(
            tools_planned=["tool1", "tool2", "tool3"],
            tools_succeeded=1,
            tools_failed=2,
        )

        assert signal_success.compute_reward() > signal_failure.compute_reward()

    def test_compute_reward_efficiency_impact(self):
        """Test that latency affects reward."""
        # Fast composition
        signal_fast = CompositionTrainingSignal(
            tools_planned=["tool1"],
            tools_succeeded=1,
            total_latency_ms=3000,
        )

        # Slow composition
        signal_slow = CompositionTrainingSignal(
            tools_planned=["tool1"],
            tools_succeeded=1,
            total_latency_ms=30000,
        )

        assert signal_fast.compute_reward() > signal_slow.compute_reward()

    def test_compute_reward_decomposition_quality_proxy(self):
        """Test decomposition quality proxy calculation."""
        # Ideal ratio: 1-2 sub-questions per domain
        signal_ideal = CompositionTrainingSignal(
            domain_count=3,
            sub_questions_count=5,  # ~1.7 per domain
        )

        # Too many sub-questions per domain
        signal_excessive = CompositionTrainingSignal(
            domain_count=2,
            sub_questions_count=10,  # 5 per domain
        )

        assert signal_ideal.compute_reward() > signal_excessive.compute_reward()

    def test_compute_reward_synthesis_confidence(self):
        """Test that synthesis confidence affects reward."""
        signal_confident = CompositionTrainingSignal(
            tools_planned=["tool1"],
            tools_succeeded=1,
            synthesis_confidence=0.95,
        )

        signal_uncertain = CompositionTrainingSignal(
            tools_planned=["tool1"],
            tools_succeeded=1,
            synthesis_confidence=0.3,
        )

        assert signal_confident.compute_reward() > signal_uncertain.compute_reward()

    def test_compute_reward_satisfaction_impact(self):
        """Test that user satisfaction affects reward."""
        signal_satisfied = CompositionTrainingSignal(
            tools_planned=["tool1"],
            tools_succeeded=1,
            user_satisfaction=5.0,
        )

        signal_unsatisfied = CompositionTrainingSignal(
            tools_planned=["tool1"],
            tools_succeeded=1,
            user_satisfaction=1.0,
        )

        assert signal_satisfied.compute_reward() > signal_unsatisfied.compute_reward()

    def test_to_dict_structure(self):
        """Test dictionary serialization structure."""
        signal = CompositionTrainingSignal(
            session_id="sess-123",
            query_complexity="moderate",
            tools_planned=["tool1", "tool2"],
            synthesis_confidence=0.85,
        )

        result = signal.to_dict()

        assert result["source_agent"] == "tool_composer"
        assert result["dspy_type"] == "hybrid"
        assert "input_context" in result
        assert "decomposition" in result
        assert "planning" in result
        assert "execution" in result
        assert "synthesis" in result
        assert "outcome" in result
        assert "reward" in result

    def test_to_dict_query_truncation(self):
        """Test long query truncation."""
        signal = CompositionTrainingSignal(query="A" * 600)
        result = signal.to_dict()

        assert len(result["input_context"]["query"]) <= 500

    def test_to_dict_tools_limit(self):
        """Test tools list limiting."""
        signal = CompositionTrainingSignal(tools_planned=[f"tool_{i}" for i in range(15)])
        result = signal.to_dict()

        assert len(result["planning"]["tools_planned"]) <= 10


class TestToolComposerSignalCollector:
    """Tests for ToolComposerSignalCollector."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = ToolComposerSignalCollector()

        assert collector._signals_buffer == []
        assert collector._buffer_limit == 100

    def test_collect_composition_signal(self):
        """Test signal collection initiation."""
        collector = ToolComposerSignalCollector()

        signal = collector.collect_composition_signal(
            session_id="tc-sess-123",
            query="Compare trends and predict",
            query_complexity="complex",
            entity_count=5,
            domain_count=3,
        )

        assert isinstance(signal, CompositionTrainingSignal)
        assert signal.session_id == "tc-sess-123"
        assert signal.query_complexity == "complex"
        assert signal.entity_count == 5

    def test_update_decomposition(self):
        """Test decomposition phase update."""
        collector = ToolComposerSignalCollector()
        signal = CompositionTrainingSignal()

        updated = collector.update_decomposition(
            signal,
            sub_questions_count=4,
            decomposition_method="llm",
            decomposition_quality=0.85,
        )

        assert updated.sub_questions_count == 4
        assert updated.decomposition_method == "llm"
        assert updated.decomposition_quality == 0.85

    def test_update_planning(self):
        """Test planning phase update."""
        collector = ToolComposerSignalCollector()
        signal = CompositionTrainingSignal()

        updated = collector.update_planning(
            signal,
            tools_planned=["causal_effect", "gap_calc", "risk_scorer"],
            parallel_groups_count=2,
            used_episodic=True,
        )

        assert len(updated.tools_planned) == 3
        assert updated.parallel_groups_count == 2
        assert updated.plan_used_episodic is True

    def test_update_execution(self):
        """Test execution phase update."""
        collector = ToolComposerSignalCollector()
        signal = CompositionTrainingSignal()

        updated = collector.update_execution(
            signal,
            tools_succeeded=3,
            tools_failed=0,
            execution_latency_ms=4500,
        )

        assert updated.tools_succeeded == 3
        assert updated.tools_failed == 0
        assert updated.total_execution_latency_ms == 4500

    def test_update_synthesis_adds_to_buffer(self):
        """Test synthesis update adds signal to buffer."""
        collector = ToolComposerSignalCollector()
        signal = CompositionTrainingSignal()

        assert len(collector._signals_buffer) == 0

        collector.update_synthesis(
            signal,
            synthesis_confidence=0.88,
            response_length=500,
            sources_cited_count=3,
            total_latency_ms=6000,
        )

        assert len(collector._signals_buffer) == 1
        assert signal.synthesis_confidence == 0.88
        assert signal.sources_cited_count == 3

    def test_buffer_limit_enforcement(self):
        """Test buffer respects limit."""
        collector = ToolComposerSignalCollector()
        collector._buffer_limit = 5

        for i in range(7):
            signal = CompositionTrainingSignal(session_id=f"tc-{i}")
            collector.update_synthesis(
                signal,
                synthesis_confidence=0.8,
                response_length=300,
                sources_cited_count=2,
                total_latency_ms=5000,
            )

        assert len(collector._signals_buffer) == 5
        assert collector._signals_buffer[0].session_id == "tc-2"

    def test_update_with_feedback(self):
        """Test delayed feedback update."""
        collector = ToolComposerSignalCollector()
        signal = CompositionTrainingSignal()

        updated = collector.update_with_feedback(
            signal,
            user_satisfaction=4.5,
            answer_quality=0.9,
        )

        assert updated.user_satisfaction == 4.5
        assert updated.answer_quality == 0.9

    def test_get_signals_for_training(self):
        """Test signal retrieval for training."""
        collector = ToolComposerSignalCollector()

        for i in range(3):
            signal = CompositionTrainingSignal(session_id=f"tc-{i}")
            collector.update_synthesis(
                signal,
                synthesis_confidence=0.8,
                response_length=300,
                sources_cited_count=2,
                total_latency_ms=5000,
            )

        all_signals = collector.get_signals_for_training(min_reward=0.0)
        assert len(all_signals) == 3

    def test_get_high_quality_examples(self):
        """Test retrieval of high-quality examples only."""
        collector = ToolComposerSignalCollector()

        # Create signals with varying quality
        for i, (succeeded, confidence) in enumerate([(3, 0.9), (1, 0.3), (3, 0.85)]):
            signal = CompositionTrainingSignal(
                session_id=f"tc-{i}",
                tools_planned=["t1", "t2", "t3"],
                tools_succeeded=succeeded,
                synthesis_confidence=confidence,
            )
            collector.update_synthesis(
                signal,
                synthesis_confidence=confidence,
                response_length=400,
                sources_cited_count=3,
                total_latency_ms=5000,
            )

        high_quality = collector.get_high_quality_examples(min_reward=0.6, limit=10)

        # Should get the higher quality ones
        assert len(high_quality) >= 1

    def test_clear_buffer(self):
        """Test buffer clearing."""
        collector = ToolComposerSignalCollector()

        signal = CompositionTrainingSignal()
        collector.update_synthesis(
            signal,
            synthesis_confidence=0.8,
            response_length=300,
            sources_cited_count=2,
            total_latency_ms=5000,
        )

        assert len(collector._signals_buffer) == 1

        collector.clear_buffer()

        assert len(collector._signals_buffer) == 0


class TestToolComposerDSPyIntegration:
    """Tests for ToolComposerDSPyIntegration (Hybrid role)."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test integration initializes correctly."""
        integration = ToolComposerDSPyIntegration()

        assert integration.dspy_type == "hybrid"
        assert integration._optimized_prompts == {}
        assert integration._optimization_requests == []

    @pytest.mark.asyncio
    async def test_get_optimized_decomposition_prompt_default(self):
        """Test returns default when no optimized prompt."""
        integration = ToolComposerDSPyIntegration()
        default = "Default decomposition prompt"

        result = await integration.get_optimized_decomposition_prompt(default)

        assert result == default

    @pytest.mark.asyncio
    async def test_get_optimized_decomposition_prompt_optimized(self):
        """Test returns optimized prompt when available."""
        integration = ToolComposerDSPyIntegration()
        optimized = "DSPy-optimized decomposition prompt"
        integration.update_optimized_prompt("decomposition", optimized)

        result = await integration.get_optimized_decomposition_prompt("default")

        assert result == optimized

    @pytest.mark.asyncio
    async def test_get_optimized_synthesis_prompt(self):
        """Test synthesis prompt retrieval."""
        integration = ToolComposerDSPyIntegration()
        optimized = "DSPy-optimized synthesis prompt"
        integration.update_optimized_prompt("synthesis", optimized)

        result = await integration.get_optimized_synthesis_prompt("default")

        assert result == optimized

    def test_update_optimized_prompt(self):
        """Test prompt update."""
        integration = ToolComposerDSPyIntegration()

        integration.update_optimized_prompt("decomposition", "new prompt")

        assert integration._optimized_prompts["decomposition"] == "new prompt"

    @pytest.mark.asyncio
    async def test_request_optimization(self):
        """Test optimization request."""
        integration = ToolComposerDSPyIntegration()

        request_id = await integration.request_optimization(
            signature_name="QueryDecompositionSignature",
            training_signals=[{"signal": "test"}],
            priority="high",
        )

        assert request_id.startswith("tc_opt_QueryDecompositionSignature_")
        assert len(integration._optimization_requests) == 1
        assert integration._optimization_requests[0]["status"] == "pending"

    def test_get_pending_requests(self):
        """Test pending request retrieval."""
        integration = ToolComposerDSPyIntegration()

        # Add a request
        integration._optimization_requests.append(
            {
                "request_id": "test_req",
                "status": "pending",
            }
        )

        pending = integration.get_pending_requests()
        assert len(pending) == 1

    def test_has_optimized_prompts(self):
        """Test optimized prompts check."""
        integration = ToolComposerDSPyIntegration()

        assert integration.has_optimized_prompts() is False

        integration.update_optimized_prompt("decomposition", "optimized")

        assert integration.has_optimized_prompts() is True


class TestSingletonAccess:
    """Tests for singleton access patterns."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_get_signal_collector_creates_singleton(self):
        """Test signal collector singleton creation."""
        collector1 = get_tool_composer_signal_collector()
        collector2 = get_tool_composer_signal_collector()

        assert collector1 is collector2

    def test_get_dspy_integration_creates_singleton(self):
        """Test DSPy integration singleton creation."""
        integration1 = get_tool_composer_dspy_integration()
        integration2 = get_tool_composer_dspy_integration()

        assert integration1 is integration2

    def test_reset_dspy_integration(self):
        """Test singleton reset."""
        collector1 = get_tool_composer_signal_collector()
        integration1 = get_tool_composer_dspy_integration()

        reset_dspy_integration()

        collector2 = get_tool_composer_signal_collector()
        integration2 = get_tool_composer_dspy_integration()

        assert collector1 is not collector2
        assert integration1 is not integration2


class TestDSPySignatures:
    """Tests for DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE reflects actual availability."""
        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_query_decomposition_signature(self):
        """Test QueryDecompositionSignature exists."""
        import dspy

        from src.agents.tool_composer.dspy_integration import QueryDecompositionSignature

        assert issubclass(QueryDecompositionSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_tool_mapping_signature(self):
        """Test ToolMappingSignature exists."""
        import dspy

        from src.agents.tool_composer.dspy_integration import ToolMappingSignature

        assert issubclass(ToolMappingSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_response_synthesis_signature(self):
        """Test ResponseSynthesisSignature exists."""
        import dspy

        from src.agents.tool_composer.dspy_integration import ResponseSynthesisSignature

        assert issubclass(ResponseSynthesisSignature, dspy.Signature)


class TestFullSignalLifecycle:
    """Integration tests for complete signal lifecycle."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_complete_signal_lifecycle(self):
        """Test full signal collection through all phases."""
        collector = get_tool_composer_signal_collector()

        # Phase 1: Initialize
        signal = collector.collect_composition_signal(
            session_id="lifecycle-tc-test",
            query="Compare TRx trends for Kisqali and predict Q4 performance",
            query_complexity="complex",
            entity_count=4,
            domain_count=2,
        )

        # Phase 2: Decomposition
        signal = collector.update_decomposition(
            signal,
            sub_questions_count=4,
            decomposition_method="llm",
            decomposition_quality=0.88,
        )

        # Phase 3: Planning
        signal = collector.update_planning(
            signal,
            tools_planned=["trend_analyzer", "causal_effect", "risk_scorer", "forecaster"],
            parallel_groups_count=2,
            used_episodic=False,
        )

        # Phase 4: Execution
        signal = collector.update_execution(
            signal,
            tools_succeeded=4,
            tools_failed=0,
            execution_latency_ms=5500,
        )

        # Phase 5: Synthesis (adds to buffer)
        signal = collector.update_synthesis(
            signal,
            synthesis_confidence=0.9,
            response_length=650,
            sources_cited_count=4,
            total_latency_ms=7000,
        )

        # Phase 6: Delayed feedback
        signal = collector.update_with_feedback(
            signal,
            user_satisfaction=5.0,
            answer_quality=0.92,
        )

        # Verify final state
        assert signal.session_id == "lifecycle-tc-test"
        assert signal.sub_questions_count == 4
        assert signal.tools_succeeded == 4
        assert signal.synthesis_confidence == 0.9
        assert signal.user_satisfaction == 5.0

        # Verify in buffer
        signals = collector.get_signals_for_training()
        assert len(signals) == 1

        # Verify reward is good
        reward = signal.compute_reward()
        assert reward > 0.7

    @pytest.mark.asyncio
    async def test_hybrid_prompt_optimization_flow(self):
        """Test the hybrid agent's prompt optimization flow."""
        integration = get_tool_composer_dspy_integration()

        # Initially no optimized prompts
        assert integration.has_optimized_prompts() is False

        # Get default prompts
        decomp_prompt = await integration.get_optimized_decomposition_prompt("default decomp")
        assert decomp_prompt == "default decomp"

        # Simulate receiving optimized prompts from feedback_learner
        integration.update_optimized_prompt("decomposition", "MIPROv2 optimized decomposition")
        integration.update_optimized_prompt("synthesis", "MIPROv2 optimized synthesis")

        # Now should have optimized prompts
        assert integration.has_optimized_prompts() is True

        # Get optimized prompts
        decomp_prompt = await integration.get_optimized_decomposition_prompt("default")
        synth_prompt = await integration.get_optimized_synthesis_prompt("default")

        assert decomp_prompt == "MIPROv2 optimized decomposition"
        assert synth_prompt == "MIPROv2 optimized synthesis"
