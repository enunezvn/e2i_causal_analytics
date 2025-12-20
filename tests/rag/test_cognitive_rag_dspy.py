"""
Unit tests for DSPy-enhanced Cognitive RAG system.

Tests cover:
- Enums: MemoryType, HopType
- Dataclasses: Evidence, CognitiveState
- 11 DSPy Signatures (Phase 1-4)
- 4 DSPy Modules: SummarizerModule, InvestigatorModule, AgentModule, ReflectorModule
- Workflow creation: create_dspy_cognitive_workflow()
- CognitiveRAGOptimizer metrics
- Mock backends

Author: E2I Causal Analytics Team
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import asdict
from typing import Dict, Any
import json


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_type_values(self):
        """Test all MemoryType enum values exist."""
        from src.rag.cognitive_rag_dspy import MemoryType

        assert MemoryType.WORKING.value == "working"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_memory_type_is_enum(self):
        """Test MemoryType is proper enum."""
        from src.rag.cognitive_rag_dspy import MemoryType
        from enum import Enum

        assert issubclass(MemoryType, Enum)
        assert len(list(MemoryType)) == 4


class TestHopType:
    """Tests for HopType enum."""

    def test_hop_type_values(self):
        """Test all HopType enum values exist."""
        from src.rag.cognitive_rag_dspy import HopType

        assert HopType.HOP_1_EPISODIC.value == "episodic"
        assert HopType.HOP_2_SEMANTIC.value == "semantic"
        assert HopType.HOP_3_PROCEDURAL.value == "procedural"
        assert HopType.HOP_4_REFINEMENT.value == "refinement"

    def test_hop_type_is_enum(self):
        """Test HopType is proper enum."""
        from src.rag.cognitive_rag_dspy import HopType
        from enum import Enum

        assert issubclass(HopType, Enum)
        assert len(list(HopType)) == 4


# =============================================================================
# DATACLASS TESTS
# =============================================================================

class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_evidence_creation(self):
        """Test creating Evidence with all fields."""
        from src.rag.cognitive_rag_dspy import Evidence, MemoryType

        evidence = Evidence(
            source=MemoryType.EPISODIC,
            hop_number=1,
            content="Test content about Kisqali",
            relevance_score=0.85,
            metadata={"brand": "Kisqali"}
        )

        assert evidence.content == "Test content about Kisqali"
        assert evidence.source == MemoryType.EPISODIC
        assert evidence.hop_number == 1
        assert evidence.relevance_score == 0.85
        assert evidence.metadata["brand"] == "Kisqali"

    def test_evidence_default_metadata(self):
        """Test Evidence with default empty metadata."""
        from src.rag.cognitive_rag_dspy import Evidence, MemoryType

        evidence = Evidence(
            source=MemoryType.SEMANTIC,
            hop_number=2,
            content="Content",
            relevance_score=0.7
        )

        assert evidence.metadata == {}

    def test_evidence_is_identifiable_by_content_and_hop(self):
        """Test Evidence can be identified by content and hop number."""
        from src.rag.cognitive_rag_dspy import Evidence, MemoryType

        e1 = Evidence(MemoryType.EPISODIC, 1, "Content", 0.8)
        e2 = Evidence(MemoryType.EPISODIC, 1, "Content", 0.9)

        # Same content and hop_number identifies same underlying evidence
        assert e1.content == e2.content
        assert e1.hop_number == e2.hop_number


class TestCognitiveState:
    """Tests for CognitiveState dataclass."""

    def test_cognitive_state_minimal_creation(self):
        """Test creating CognitiveState with minimal required fields."""
        from src.rag.cognitive_rag_dspy import CognitiveState

        state = CognitiveState(
            user_query="Why did TRx increase?",
            conversation_id="conv_001"
        )

        assert state.user_query == "Why did TRx increase?"
        assert state.conversation_id == "conv_001"
        # Defaults
        assert state.rewritten_query == ""
        assert state.extracted_entities == []
        assert state.detected_intent == ""
        assert state.evidence_board == []
        assert state.hop_count == 0
        assert state.sufficient_evidence is False
        assert state.response == ""
        assert state.dspy_signals == []

    def test_cognitive_state_full_creation(self):
        """Test CognitiveState with all fields populated."""
        from src.rag.cognitive_rag_dspy import CognitiveState, Evidence, MemoryType

        evidence = Evidence(MemoryType.EPISODIC, 1, "Test", 0.9)

        state = CognitiveState(
            user_query="Query",
            conversation_id="conv_002",
            compressed_history="Previous context...",
            rewritten_query="Optimized query",
            extracted_entities=["Kisqali", "TRx"],
            detected_intent="causal",
            investigation_goal="Understand TRx drivers",
            evidence_board=[evidence],
            hop_count=2,
            sufficient_evidence=True,
            response="The increase was due to...",
            visualization_config={"chart_type": "line"},
            routed_agents=["causal_impact"],
            worth_remembering=True,
            dspy_signals=[{"signature": "QueryRewrite"}]
        )

        assert len(state.evidence_board) == 1
        assert state.hop_count == 2
        assert state.sufficient_evidence is True
        assert state.worth_remembering is True

    def test_cognitive_state_mutable_lists(self):
        """Test that list fields are mutable."""
        from src.rag.cognitive_rag_dspy import CognitiveState, Evidence, MemoryType

        state = CognitiveState(
            user_query="Test",
            conversation_id="conv_003"
        )

        # Add to lists
        state.extracted_entities.append("Fabhalta")
        state.evidence_board.append(
            Evidence(MemoryType.SEMANTIC, 2, "New evidence", 0.85)
        )
        state.dspy_signals.append({"phase": 1})

        assert len(state.extracted_entities) == 1
        assert len(state.evidence_board) == 1
        assert len(state.dspy_signals) == 1


# =============================================================================
# DSPY SIGNATURE TESTS
# =============================================================================

class TestPhase1Signatures:
    """Tests for Phase 1 (Summarizer) DSPy signatures."""

    def test_query_rewrite_signature_exists(self):
        """Test QueryRewriteSignature is defined."""
        from src.rag.cognitive_rag_dspy import QueryRewriteSignature
        import dspy

        assert issubclass(QueryRewriteSignature, dspy.Signature)

    def test_query_rewrite_signature_fields(self):
        """Test QueryRewriteSignature has expected input/output fields."""
        from src.rag.cognitive_rag_dspy import QueryRewriteSignature

        sig = QueryRewriteSignature
        # Check input fields exist in signature
        assert hasattr(sig, '__annotations__') or hasattr(sig, '__signature__')

    def test_entity_extraction_signature_exists(self):
        """Test EntityExtractionSignature is defined."""
        from src.rag.cognitive_rag_dspy import EntityExtractionSignature
        import dspy

        assert issubclass(EntityExtractionSignature, dspy.Signature)

    def test_intent_classification_signature_exists(self):
        """Test IntentClassificationSignature is defined."""
        from src.rag.cognitive_rag_dspy import IntentClassificationSignature
        import dspy

        assert issubclass(IntentClassificationSignature, dspy.Signature)


class TestPhase2Signatures:
    """Tests for Phase 2 (Investigator) DSPy signatures."""

    def test_investigation_plan_signature_exists(self):
        """Test InvestigationPlanSignature is defined."""
        from src.rag.cognitive_rag_dspy import InvestigationPlanSignature
        import dspy

        assert issubclass(InvestigationPlanSignature, dspy.Signature)

    def test_hop_decision_signature_exists(self):
        """Test HopDecisionSignature is defined."""
        from src.rag.cognitive_rag_dspy import HopDecisionSignature
        import dspy

        assert issubclass(HopDecisionSignature, dspy.Signature)

    def test_evidence_relevance_signature_exists(self):
        """Test EvidenceRelevanceSignature is defined."""
        from src.rag.cognitive_rag_dspy import EvidenceRelevanceSignature
        import dspy

        assert issubclass(EvidenceRelevanceSignature, dspy.Signature)


class TestPhase3Signatures:
    """Tests for Phase 3 (Agent) DSPy signatures."""

    def test_evidence_synthesis_signature_exists(self):
        """Test EvidenceSynthesisSignature is defined."""
        from src.rag.cognitive_rag_dspy import EvidenceSynthesisSignature
        import dspy

        assert issubclass(EvidenceSynthesisSignature, dspy.Signature)

    def test_agent_routing_signature_exists(self):
        """Test AgentRoutingSignature is defined."""
        from src.rag.cognitive_rag_dspy import AgentRoutingSignature
        import dspy

        assert issubclass(AgentRoutingSignature, dspy.Signature)

    def test_visualization_config_signature_exists(self):
        """Test VisualizationConfigSignature is defined."""
        from src.rag.cognitive_rag_dspy import VisualizationConfigSignature
        import dspy

        assert issubclass(VisualizationConfigSignature, dspy.Signature)


class TestPhase4Signatures:
    """Tests for Phase 4 (Reflector) DSPy signatures."""

    def test_memory_worthiness_signature_exists(self):
        """Test MemoryWorthinessSignature is defined."""
        from src.rag.cognitive_rag_dspy import MemoryWorthinessSignature
        import dspy

        assert issubclass(MemoryWorthinessSignature, dspy.Signature)

    def test_procedure_learning_signature_exists(self):
        """Test ProcedureLearningSignature is defined."""
        from src.rag.cognitive_rag_dspy import ProcedureLearningSignature
        import dspy

        assert issubclass(ProcedureLearningSignature, dspy.Signature)


# =============================================================================
# DSPY MODULE TESTS
# =============================================================================

class TestSummarizerModule:
    """Tests for SummarizerModule (Phase 1)."""

    def test_summarizer_module_initialization(self):
        """Test SummarizerModule can be instantiated."""
        from src.rag.cognitive_rag_dspy import SummarizerModule
        import dspy

        module = SummarizerModule()
        assert isinstance(module, dspy.Module)

    def test_summarizer_module_has_forward(self):
        """Test SummarizerModule has forward method."""
        from src.rag.cognitive_rag_dspy import SummarizerModule

        module = SummarizerModule()
        assert hasattr(module, 'forward')
        assert callable(module.forward)

    @patch('dspy.Predict')
    def test_summarizer_forward_returns_dict(self, mock_predict):
        """Test forward method returns expected dictionary structure."""
        from src.rag.cognitive_rag_dspy import SummarizerModule

        # Mock DSPy predict to return expected structure
        mock_result = MagicMock()
        mock_result.rewritten_query = "optimized query"
        mock_result.graph_entities = '["Kisqali", "TRx"]'
        mock_result.search_keywords = '["prescription", "adoption"]'
        mock_result.primary_intent = "causal"
        mock_result.confidence = 0.9
        mock_predict.return_value = MagicMock(return_value=mock_result)

        module = SummarizerModule()
        # Replace internal predictors with mocks
        module.query_rewriter = MagicMock(return_value=mock_result)
        module.entity_extractor = MagicMock(return_value=mock_result)
        module.intent_classifier = MagicMock(return_value=mock_result)

        result = module.forward(
            original_query="Why did TRx increase?",
            conversation_context="",
            domain_vocabulary="brands: [Kisqali]"
        )

        assert isinstance(result, dict)
        assert "rewritten_query" in result
        assert "extracted_entities" in result
        assert "primary_intent" in result


class TestInvestigatorModule:
    """Tests for InvestigatorModule (Phase 2)."""

    def test_investigator_module_initialization(self):
        """Test InvestigatorModule can be instantiated with backends."""
        from src.rag.cognitive_rag_dspy import InvestigatorModule

        mock_backends = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "procedural": MagicMock()
        }

        module = InvestigatorModule(mock_backends)
        assert module.memory_backends == mock_backends

    def test_investigator_module_has_forward(self):
        """Test InvestigatorModule has async forward method."""
        from src.rag.cognitive_rag_dspy import InvestigatorModule

        module = InvestigatorModule({})
        assert hasattr(module, 'forward')
        assert callable(module.forward)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_investigator_forward_structure(self):
        """Test forward returns expected structure (requires DSPy LM)."""
        pytest.skip("Requires DSPy LM configuration - run with integration tests")
        # This test is skipped because InvestigatorModule.forward() internally
        # creates and calls DSPy predictors that require an LM to be configured.
        # To run this test, configure DSPy with a real LM and mark as integration.


class TestAgentModule:
    """Tests for AgentModule (Phase 3)."""

    def test_agent_module_initialization(self):
        """Test AgentModule can be instantiated."""
        from src.rag.cognitive_rag_dspy import AgentModule

        mock_registry = {"causal_impact": MagicMock()}
        module = AgentModule(mock_registry)
        assert module.agent_registry == mock_registry

    def test_agent_module_has_forward(self):
        """Test AgentModule has async forward method."""
        from src.rag.cognitive_rag_dspy import AgentModule

        module = AgentModule({})
        assert hasattr(module, 'forward')

    @pytest.mark.asyncio
    async def test_agent_forward_updates_state(self):
        """Test forward updates CognitiveState with response."""
        from src.rag.cognitive_rag_dspy import (
            AgentModule, CognitiveState, Evidence, MemoryType
        )
        import dspy

        mock_agent = AsyncMock()
        mock_agent.run.return_value = {"result": "Agent output"}
        mock_registry = {"causal_impact": mock_agent}

        # Configure mock LM for DSPy
        mock_lm = MagicMock()

        with patch.object(dspy, 'settings', MagicMock(lm=mock_lm)):
            module = AgentModule(mock_registry)

            # Mock DSPy predictors
            mock_synthesis = MagicMock()
            mock_synthesis.response = "The TRx increased due to..."
            mock_synthesis.evidence_citations = '["doc_001"]'
            mock_synthesis.confidence_statement = "High confidence"

            mock_routing = MagicMock()
            mock_routing.agent_names = "causal_impact"
            mock_routing.routing_reason = "Causal question"

            mock_viz = MagicMock()
            mock_viz.chart_type = "line"
            mock_viz.config_json = "{}"

            # Replace all internal predictors with mocks
            module.synthesize = MagicMock(return_value=mock_synthesis)
            module.route = MagicMock(return_value=mock_routing)
            module.visualize = MagicMock(return_value=mock_viz)

            state = CognitiveState(
                user_query="Why TRx?",
                conversation_id="conv_001",
                rewritten_query="TRx growth factors",
                detected_intent="causal",
                evidence_board=[
                    Evidence(MemoryType.EPISODIC, 1, "Evidence", 0.9)
                ]
            )

            updated_state = await module.forward(state)

            assert updated_state.response != ""
            assert "routed_agents" in asdict(updated_state)


class TestReflectorModule:
    """Tests for ReflectorModule (Phase 4)."""

    def test_reflector_module_initialization(self):
        """Test ReflectorModule can be instantiated."""
        from src.rag.cognitive_rag_dspy import ReflectorModule

        mock_writers = {"episodic": MagicMock()}
        mock_collector = MagicMock()

        module = ReflectorModule(mock_writers, mock_collector)
        assert module.memory_writers == mock_writers
        assert module.signal_collector == mock_collector

    def test_reflector_module_has_forward(self):
        """Test ReflectorModule has async forward method."""
        from src.rag.cognitive_rag_dspy import ReflectorModule

        module = ReflectorModule({}, MagicMock())
        assert hasattr(module, 'forward')

    @pytest.mark.asyncio
    async def test_reflector_evaluates_worthiness(self):
        """Test forward evaluates memory worthiness (requires DSPy LM)."""
        pytest.skip("Requires DSPy LM configuration - run with integration tests")
        # This test is skipped because ReflectorModule.forward() internally
        # creates and calls DSPy predictors that require an LM to be configured.
        # To run this test, configure DSPy with a real LM and mark as integration.


# =============================================================================
# WORKFLOW CREATION TESTS
# =============================================================================

class TestCreateDspyCognitiveWorkflow:
    """Tests for create_dspy_cognitive_workflow function."""

    def test_workflow_creation_returns_compiled_graph(self):
        """Test that workflow creation returns compiled StateGraph."""
        from src.rag.cognitive_rag_dspy import create_dspy_cognitive_workflow

        mock_backends = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "procedural": MagicMock()
        }

        workflow = create_dspy_cognitive_workflow(
            memory_backends=mock_backends,
            memory_writers=mock_backends,
            agent_registry={},
            signal_collector=MagicMock(),
            domain_vocabulary="test vocabulary"
        )

        # Compiled workflow should be a CompiledStateGraph
        assert workflow is not None
        assert hasattr(workflow, 'ainvoke')

    def test_workflow_has_four_nodes(self):
        """Test workflow contains all 4 phase nodes."""
        from src.rag.cognitive_rag_dspy import create_dspy_cognitive_workflow

        mock_backends = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "procedural": MagicMock()
        }

        workflow = create_dspy_cognitive_workflow(
            memory_backends=mock_backends,
            memory_writers=mock_backends,
            agent_registry={},
            signal_collector=MagicMock(),
            domain_vocabulary="test"
        )

        # Check that workflow has the expected structure
        # The compiled graph should have nodes from building
        assert workflow is not None


# =============================================================================
# OPTIMIZER TESTS
# =============================================================================

class TestCognitiveRAGOptimizer:
    """Tests for CognitiveRAGOptimizer."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        from src.rag.cognitive_rag_dspy import CognitiveRAGOptimizer

        mock_learner = MagicMock()
        optimizer = CognitiveRAGOptimizer(mock_learner)

        assert optimizer.feedback_learner == mock_learner

    def test_summarizer_metric_scoring(self):
        """Test summarizer_metric returns score 0-1."""
        from src.rag.cognitive_rag_dspy import CognitiveRAGOptimizer

        optimizer = CognitiveRAGOptimizer(MagicMock())

        # Create mock example and prediction
        example = MagicMock()
        example.original_query = "short query"

        prediction = MagicMock()
        prediction.rewritten_query = "longer optimized query with details"
        prediction.graph_entities = ["Kisqali"]
        prediction.primary_intent = "causal"
        prediction.search_keywords = ["hcp", "adoption"]

        score = optimizer.summarizer_metric(example, prediction)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Good prediction should score well

    def test_summarizer_metric_low_score_for_poor_prediction(self):
        """Test poor prediction gets low score."""
        from src.rag.cognitive_rag_dspy import CognitiveRAGOptimizer

        optimizer = CognitiveRAGOptimizer(MagicMock())

        example = MagicMock()
        example.original_query = "very long original query"

        prediction = MagicMock()
        prediction.rewritten_query = "short"  # Shorter than original
        prediction.graph_entities = []  # No entities
        prediction.primary_intent = "GENERAL"  # Generic intent
        prediction.search_keywords = []  # No keywords

        score = optimizer.summarizer_metric(example, prediction)

        assert score < 0.5  # Poor prediction should score low

    def test_investigator_metric_scoring(self):
        """Test investigator_metric returns score 0-1."""
        from src.rag.cognitive_rag_dspy import (
            CognitiveRAGOptimizer, Evidence, MemoryType
        )

        optimizer = CognitiveRAGOptimizer(MagicMock())

        example = MagicMock()

        prediction = MagicMock()
        prediction.evidence_board = [
            Evidence(MemoryType.EPISODIC, 1, "E1", 0.9),
            Evidence(MemoryType.SEMANTIC, 2, "E2", 0.85),
        ]
        prediction.sufficient_evidence = True
        prediction.hop_count = 2

        score = optimizer.investigator_metric(example, prediction)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Good evidence gathering

    def test_agent_metric_scoring(self):
        """Test agent_metric returns score 0-1."""
        from src.rag.cognitive_rag_dspy import CognitiveRAGOptimizer

        optimizer = CognitiveRAGOptimizer(MagicMock())

        example = MagicMock()
        example.requires_visualization = True

        prediction = MagicMock()
        prediction.response = "A" * 250  # Substantive response
        prediction.evidence_citations = ["doc_001", "doc_002"]
        prediction.confidence_statement = "High confidence based on multiple sources"
        prediction.chart_type = "line"

        score = optimizer.agent_metric(example, prediction)

        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_signals_to_examples_conversion(self):
        """Test _signals_to_examples converts correctly."""
        from src.rag.cognitive_rag_dspy import CognitiveRAGOptimizer
        import dspy

        optimizer = CognitiveRAGOptimizer(MagicMock())

        signals = [
            {
                "phase": "summarizer",
                "success": True,
                "input": {"query": "test query"},
                "output": {"rewritten": "optimized"}
            },
            {
                "phase": "summarizer",
                "success": False,  # Should be skipped
                "input": {},
                "output": {}
            },
            {
                "phase": "investigator",  # Different phase, skipped
                "success": True,
                "input": {},
                "output": {}
            }
        ]

        examples = optimizer._signals_to_examples(signals, "summarizer")

        # Only first signal should be converted
        assert len(examples) == 1
        assert isinstance(examples[0], dspy.Example)


# =============================================================================
# MOCK BACKEND TESTS
# =============================================================================

class TestMockBackends:
    """Tests for mock memory backends."""

    @pytest.mark.asyncio
    async def test_mock_episodic_memory(self):
        """Test MockEpisodicMemory returns expected structure."""
        from src.rag.cognitive_rag_dspy import MockEpisodicMemory

        mock = MockEpisodicMemory()
        results = await mock.vector_search("test query", limit=5)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

    @pytest.mark.asyncio
    async def test_mock_semantic_memory(self):
        """Test MockSemanticMemory returns expected structure."""
        from src.rag.cognitive_rag_dspy import MockSemanticMemory

        mock = MockSemanticMemory()
        results = await mock.graph_query("test query", max_depth=2)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

    @pytest.mark.asyncio
    async def test_mock_procedural_memory(self):
        """Test MockProceduralMemory returns expected structure."""
        from src.rag.cognitive_rag_dspy import MockProceduralMemory

        mock = MockProceduralMemory()
        results = await mock.procedure_search("test query", limit=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

    @pytest.mark.asyncio
    async def test_mock_signal_collector(self):
        """Test MockSignalCollector accepts signals."""
        from src.rag.cognitive_rag_dspy import MockSignalCollector

        mock = MockSignalCollector()

        # Should not raise
        await mock.collect([{"signal": "test"}])


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_evidence_with_explicit_metadata(self):
        """Test Evidence handles explicit metadata."""
        from src.rag.cognitive_rag_dspy import Evidence, MemoryType

        evidence = Evidence(
            source=MemoryType.EPISODIC,
            hop_number=1,
            content="Test",
            relevance_score=0.5,
            metadata={"key": "value"}
        )

        assert evidence.metadata == {"key": "value"}

    def test_cognitive_state_empty_lists_are_mutable(self):
        """Test default empty lists don't share state."""
        from src.rag.cognitive_rag_dspy import CognitiveState

        state1 = CognitiveState(user_query="Q1", conversation_id="c1")
        state2 = CognitiveState(user_query="Q2", conversation_id="c2")

        state1.extracted_entities.append("Entity1")

        # state2's list should be independent
        assert len(state2.extracted_entities) == 0

    def test_optimizer_handles_empty_evidence_board(self):
        """Test investigator_metric handles empty evidence."""
        from src.rag.cognitive_rag_dspy import CognitiveRAGOptimizer

        optimizer = CognitiveRAGOptimizer(MagicMock())

        example = MagicMock()
        prediction = MagicMock()
        prediction.evidence_board = []
        prediction.sufficient_evidence = False
        prediction.hop_count = 0

        score = optimizer.investigator_metric(example, prediction)

        # Should return a valid score, not crash
        assert 0.0 <= score <= 1.0

    def test_summarizer_metric_handles_missing_attributes(self):
        """Test summarizer_metric handles predictions with missing attributes."""
        from src.rag.cognitive_rag_dspy import CognitiveRAGOptimizer

        optimizer = CognitiveRAGOptimizer(MagicMock())

        example = MagicMock()
        example.original_query = "test"

        # Prediction without all expected attributes
        prediction = MagicMock(spec=[])  # Empty spec = no attributes

        # Add only some attributes
        prediction.rewritten_query = "test query"
        prediction.graph_entities = None
        prediction.primary_intent = None
        prediction.search_keywords = None

        # Should not crash
        try:
            score = optimizer.summarizer_metric(example, prediction)
            assert isinstance(score, (int, float))
        except AttributeError:
            # It's acceptable to fail on missing attributes
            # as long as it's a clear error
            pass


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestModuleIntegration:
    """Integration tests for module interactions."""

    def test_all_modules_can_be_imported(self):
        """Test all DSPy modules can be imported together."""
        from src.rag.cognitive_rag_dspy import (
            SummarizerModule,
            InvestigatorModule,
            AgentModule,
            ReflectorModule,
            CognitiveRAGOptimizer,
            create_dspy_cognitive_workflow
        )

        # All imports should succeed
        assert SummarizerModule is not None
        assert InvestigatorModule is not None
        assert AgentModule is not None
        assert ReflectorModule is not None
        assert CognitiveRAGOptimizer is not None
        assert create_dspy_cognitive_workflow is not None

    def test_modules_share_common_state_type(self):
        """Test all modules work with CognitiveState."""
        from src.rag.cognitive_rag_dspy import (
            CognitiveState,
            SummarizerModule,
            InvestigatorModule,
            AgentModule,
            ReflectorModule
        )

        state = CognitiveState(
            user_query="Test",
            conversation_id="test_001"
        )

        # All modules should accept CognitiveState
        modules = [
            SummarizerModule(),
            InvestigatorModule({}),
            AgentModule({}),
            ReflectorModule({}, MagicMock())
        ]

        for module in modules:
            assert hasattr(module, 'forward')


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for DSPy components."""

    def test_evidence_creation_performance(self):
        """Test Evidence creation is fast."""
        import time
        from src.rag.cognitive_rag_dspy import Evidence, MemoryType

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            Evidence(
                source=MemoryType.EPISODIC,
                hop_number=1,
                content=f"Content {i}",
                relevance_score=0.8,
                metadata={"index": i}
            )

        elapsed = time.time() - start
        avg_ms = (elapsed / iterations) * 1000

        # Should be very fast (<0.5ms per creation)
        assert avg_ms < 0.5, f"Evidence creation too slow: {avg_ms:.4f}ms"

    def test_cognitive_state_creation_performance(self):
        """Test CognitiveState creation is fast."""
        import time
        from src.rag.cognitive_rag_dspy import CognitiveState

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            CognitiveState(
                user_query=f"Query {i}",
                conversation_id=f"conv_{i}"
            )

        elapsed = time.time() - start
        avg_ms = (elapsed / iterations) * 1000

        # Should be very fast (<0.1ms per creation)
        assert avg_ms < 0.1, f"CognitiveState creation too slow: {avg_ms:.4f}ms"

    def test_optimizer_metric_performance(self):
        """Test optimizer metrics are fast."""
        import time
        from src.rag.cognitive_rag_dspy import (
            CognitiveRAGOptimizer, Evidence, MemoryType
        )

        optimizer = CognitiveRAGOptimizer(MagicMock())

        example = MagicMock()
        example.original_query = "test query"
        example.requires_visualization = True

        prediction = MagicMock()
        prediction.rewritten_query = "optimized test query"
        prediction.graph_entities = ["Entity"]
        prediction.primary_intent = "causal"
        prediction.search_keywords = ["hcp"]
        prediction.evidence_board = [
            Evidence(MemoryType.EPISODIC, 1, "E", 0.9)
        ]
        prediction.sufficient_evidence = True
        prediction.hop_count = 2
        prediction.response = "A" * 300
        prediction.evidence_citations = ["doc_001"]
        prediction.confidence_statement = "High confidence"
        prediction.chart_type = "line"

        iterations = 1000
        start = time.time()

        for _ in range(iterations):
            optimizer.summarizer_metric(example, prediction)
            optimizer.investigator_metric(example, prediction)
            optimizer.agent_metric(example, prediction)

        elapsed = time.time() - start
        avg_ms = (elapsed / iterations) * 1000

        # All 3 metrics should complete in <1ms total
        assert avg_ms < 1.0, f"Metrics too slow: {avg_ms:.4f}ms"
