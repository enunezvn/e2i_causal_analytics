"""
Unit tests for E2I Cognitive Integration Service.

Tests focus on:
- Pydantic models (CognitiveQueryInput, CognitiveQueryOutput, PhaseResult)
- CognitiveService main orchestration
- Phase implementations (summarizer, investigator, agent, reflector)
- Singleton and convenience functions
- Error handling and edge cases

All tests use mocked dependencies to avoid external services.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.memory.cognitive_integration import (
    CognitiveQueryInput,
    CognitiveQueryOutput,
    PhaseResult,
    CognitiveService,
    get_cognitive_service,
    process_cognitive_query,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_working_memory():
    """Create a mock RedisWorkingMemory."""
    wm = AsyncMock()
    wm.create_session = AsyncMock(return_value="session-123")
    wm.add_message = AsyncMock()
    wm.append_evidence = AsyncMock()
    return wm


@pytest.fixture
def mock_hybrid_search():
    """Create a mock for hybrid_search."""
    result = MagicMock()
    result.source = "episodic_memory"
    result.content = "Test evidence content about TRx trends"
    result.score = 0.85
    result.retrieval_method = "vector"
    result.metadata = {"timestamp": "2024-01-15"}
    return [result]


@pytest.fixture
def mock_graphiti_service():
    """Create a mock GraphitiService."""
    service = AsyncMock()
    result = MagicMock()
    result.episode_id = "episode-456"
    result.entities_extracted = [{"type": "Brand", "name": "Kisqali"}]
    result.relationships_extracted = [{"type": "AFFECTS", "target": "TRx"}]
    service.add_episode = AsyncMock(return_value=result)
    return service


@pytest.fixture
def cognitive_service(mock_working_memory):
    """Create a CognitiveService with mocked dependencies."""
    service = CognitiveService()
    service._working_memory = mock_working_memory
    return service


@pytest.fixture
def sample_query_input():
    """Create a sample CognitiveQueryInput."""
    return CognitiveQueryInput(
        query="Why is TRx declining for Kisqali in the Northeast?",
        session_id=None,
        user_id="user-123",
        brand="Kisqali",
        region="Northeast",
        include_evidence=True,
        max_hops=3,
    )


# ============================================================================
# PYDANTIC MODEL TESTS
# ============================================================================


class TestCognitiveQueryInput:
    """Tests for CognitiveQueryInput Pydantic model."""

    def test_valid_input(self):
        """Valid input should create model successfully."""
        input_data = CognitiveQueryInput(
            query="What is the TRx trend?",
            user_id="user-123",
        )
        assert input_data.query == "What is the TRx trend?"
        assert input_data.user_id == "user-123"
        assert input_data.include_evidence is True
        assert input_data.max_hops == 3

    def test_empty_query_raises_error(self):
        """Empty query should raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CognitiveQueryInput(query="")
        # Pydantic v2 uses "string_too_short" error type for min_length validation
        assert "string_too_short" in str(exc_info.value).lower()

    def test_max_hops_bounds(self):
        """max_hops should be constrained between 1 and 5."""
        # Valid range
        input_data = CognitiveQueryInput(query="test", max_hops=5)
        assert input_data.max_hops == 5

        # Below minimum
        with pytest.raises(ValidationError):
            CognitiveQueryInput(query="test", max_hops=0)

        # Above maximum
        with pytest.raises(ValidationError):
            CognitiveQueryInput(query="test", max_hops=6)

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        input_data = CognitiveQueryInput(query="test query")
        assert input_data.session_id is None
        assert input_data.user_id is None
        assert input_data.brand is None
        assert input_data.region is None


class TestCognitiveQueryOutput:
    """Tests for CognitiveQueryOutput Pydantic model."""

    def test_valid_output(self):
        """Valid output should create model successfully."""
        output = CognitiveQueryOutput(
            session_id="session-123",
            cycle_id="cycle-456",
            query="What is the TRx trend?",
            query_type="monitoring",
            agent_used="drift_monitor",
            response="TRx has been declining...",
            confidence=0.85,
            phases_completed=["summarizer", "investigator", "agent"],
            processing_time_ms=150.5,
            worth_remembering=True,
        )
        assert output.session_id == "session-123"
        assert output.confidence == 0.85
        assert len(output.phases_completed) == 3

    def test_output_with_evidence(self):
        """Output with evidence should include evidence list."""
        output = CognitiveQueryOutput(
            session_id="session-123",
            cycle_id="cycle-456",
            query="test",
            query_type="general",
            agent_used="orchestrator",
            response="response",
            confidence=0.5,
            evidence=[{"source": "memory", "content": "data"}],
            phases_completed=[],
            processing_time_ms=100.0,
            worth_remembering=False,
        )
        assert output.evidence is not None
        assert len(output.evidence) == 1


class TestPhaseResult:
    """Tests for PhaseResult Pydantic model."""

    def test_valid_phase_result(self):
        """Valid phase result should create model successfully."""
        result = PhaseResult(
            phase_name="summarizer",
            completed=True,
            duration_ms=25.5,
            outputs={"query_type": "causal", "entities": {"brands": ["Kisqali"]}},
        )
        assert result.phase_name == "summarizer"
        assert result.completed is True
        assert result.error is None

    def test_phase_result_with_error(self):
        """Phase result with error should include error message."""
        result = PhaseResult(
            phase_name="investigator",
            completed=False,
            duration_ms=100.0,
            error="Hybrid search timeout",
        )
        assert result.completed is False
        assert result.error == "Hybrid search timeout"


# ============================================================================
# COGNITIVE SERVICE TESTS
# ============================================================================


class TestCognitiveServiceInit:
    """Tests for CognitiveService initialization."""

    def test_init_creates_empty_working_memory(self):
        """Initialization should not create working memory yet."""
        service = CognitiveService()
        assert service._working_memory is None

    @pytest.mark.asyncio
    async def test_get_working_memory_creates_singleton(self):
        """get_working_memory should create and cache instance."""
        service = CognitiveService()

        with patch(
            "src.memory.cognitive_integration.get_working_memory"
        ) as mock_get_wm:
            mock_wm = MagicMock()
            mock_get_wm.return_value = mock_wm

            wm1 = await service.get_working_memory()
            wm2 = await service.get_working_memory()

            assert wm1 is wm2
            mock_get_wm.assert_called_once()


class TestSummarizerPhase:
    """Tests for _run_summarizer phase."""

    @pytest.mark.asyncio
    async def test_causal_query_type_detection(self, cognitive_service):
        """Causal keywords should be detected as causal query type."""
        result = await cognitive_service._run_summarizer(
            query="Why is TRx declining?",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert result["query_type"] == "causal"

    @pytest.mark.asyncio
    async def test_prediction_query_type_detection(self, cognitive_service):
        """Prediction keywords should be detected as prediction query type."""
        result = await cognitive_service._run_summarizer(
            query="What will TRx be next quarter?",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert result["query_type"] == "prediction"

    @pytest.mark.asyncio
    async def test_optimization_query_type_detection(self, cognitive_service):
        """Optimization keywords should be detected as optimization query type."""
        result = await cognitive_service._run_summarizer(
            query="How can we optimize resource allocation?",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert result["query_type"] == "optimization"

    @pytest.mark.asyncio
    async def test_comparison_query_type_detection(self, cognitive_service):
        """Comparison keywords should be detected as comparison query type."""
        result = await cognitive_service._run_summarizer(
            query="Compare TRx between regions",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert result["query_type"] == "comparison"

    @pytest.mark.asyncio
    async def test_monitoring_query_type_detection(self, cognitive_service):
        """Monitoring keywords should be detected as monitoring query type."""
        result = await cognitive_service._run_summarizer(
            query="What is the trend over time?",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert result["query_type"] == "monitoring"

    @pytest.mark.asyncio
    async def test_general_query_type_default(self, cognitive_service):
        """Unknown queries should default to general type."""
        result = await cognitive_service._run_summarizer(
            query="Show me the data",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert result["query_type"] == "general"

    @pytest.mark.asyncio
    async def test_brand_entity_extraction(self, cognitive_service):
        """Brand names should be extracted from query."""
        result = await cognitive_service._run_summarizer(
            query="What is Kisqali TRx?",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert "Kisqali" in result["entities"]["brands"]

    @pytest.mark.asyncio
    async def test_explicit_brand_included(self, cognitive_service):
        """Explicit brand parameter should be included in entities."""
        result = await cognitive_service._run_summarizer(
            query="What is the TRx?",
            session_id="session-123",
            brand="Fabhalta",
            region=None,
        )
        assert "Fabhalta" in result["entities"]["brands"]

    @pytest.mark.asyncio
    async def test_region_entity_extraction(self, cognitive_service):
        """Region names should be extracted from query."""
        result = await cognitive_service._run_summarizer(
            query="TRx in the northeast region",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert "northeast" in result["entities"]["regions"]

    @pytest.mark.asyncio
    async def test_kpi_entity_extraction(self, cognitive_service):
        """KPI names should be extracted from query."""
        result = await cognitive_service._run_summarizer(
            query="What is the TRx and NRx?",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert "TRX" in result["entities"]["kpis"]
        assert "NRX" in result["entities"]["kpis"]

    @pytest.mark.asyncio
    async def test_context_ready_flag(self, cognitive_service):
        """Result should include context_ready flag."""
        result = await cognitive_service._run_summarizer(
            query="test",
            session_id="session-123",
            brand=None,
            region=None,
        )
        assert result["context_ready"] is True


class TestInvestigatorPhase:
    """Tests for _run_investigator phase."""

    @pytest.mark.asyncio
    async def test_investigator_returns_evidence(self, cognitive_service, mock_hybrid_search):
        """Investigator should return evidence from hybrid search."""
        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            return_value=mock_hybrid_search,
        ):
            result = await cognitive_service._run_investigator(
                query="What is TRx trend?",
                query_type="monitoring",
                entities={"brands": ["Kisqali"], "regions": [], "kpis": ["TRX"]},
                brand="Kisqali",
                region=None,
                max_hops=3,
            )
            assert "evidence" in result
            assert len(result["evidence"]) > 0
            assert result["evidence"][0]["source"] == "episodic_memory"

    @pytest.mark.asyncio
    async def test_investigator_applies_brand_filter(self, cognitive_service):
        """Investigator should pass brand filter to hybrid search."""
        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_search:
            await cognitive_service._run_investigator(
                query="test",
                query_type="general",
                entities={"brands": [], "regions": [], "kpis": []},
                brand="Kisqali",
                region=None,
                max_hops=3,
            )
            call_kwargs = mock_search.call_args.kwargs
            assert call_kwargs.get("filters") == {"brand": "Kisqali"}

    @pytest.mark.asyncio
    async def test_investigator_handles_search_error(self, cognitive_service):
        """Investigator should handle hybrid search errors gracefully."""
        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            side_effect=Exception("Search timeout"),
        ):
            result = await cognitive_service._run_investigator(
                query="test",
                query_type="general",
                entities={"brands": [], "regions": [], "kpis": []},
                brand=None,
                region=None,
                max_hops=3,
            )
            assert result["evidence"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_investigator_scales_results_by_max_hops(self, cognitive_service):
        """Investigator should request more results for higher max_hops."""
        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_search:
            await cognitive_service._run_investigator(
                query="test",
                query_type="general",
                entities={"brands": [], "regions": [], "kpis": []},
                brand=None,
                region=None,
                max_hops=5,
            )
            call_kwargs = mock_search.call_args.kwargs
            assert call_kwargs.get("k") == 50  # 10 * 5


class TestAgentPhase:
    """Tests for _run_agent phase."""

    @pytest.mark.asyncio
    async def test_agent_routing_causal(self, cognitive_service):
        """Causal query type should route to causal_impact agent."""
        result = await cognitive_service._run_agent(
            query="Why is TRx declining?",
            query_type="causal",
            evidence=[{"source": "test", "content": "data", "score": 0.8}],
        )
        assert result["agent_used"] == "causal_impact"

    @pytest.mark.asyncio
    async def test_agent_routing_prediction(self, cognitive_service):
        """Prediction query type should route to prediction_synthesizer."""
        result = await cognitive_service._run_agent(
            query="What will TRx be?",
            query_type="prediction",
            evidence=[{"source": "test", "content": "data", "score": 0.8}],
        )
        assert result["agent_used"] == "prediction_synthesizer"

    @pytest.mark.asyncio
    async def test_agent_routing_optimization(self, cognitive_service):
        """Optimization query type should route to resource_optimizer."""
        result = await cognitive_service._run_agent(
            query="How to optimize?",
            query_type="optimization",
            evidence=[{"source": "test", "content": "data", "score": 0.8}],
        )
        assert result["agent_used"] == "resource_optimizer"

    @pytest.mark.asyncio
    async def test_agent_routing_comparison(self, cognitive_service):
        """Comparison query type should route to gap_analyzer."""
        result = await cognitive_service._run_agent(
            query="Compare regions",
            query_type="comparison",
            evidence=[{"source": "test", "content": "data", "score": 0.8}],
        )
        assert result["agent_used"] == "gap_analyzer"

    @pytest.mark.asyncio
    async def test_agent_routing_monitoring(self, cognitive_service):
        """Monitoring query type should route to drift_monitor."""
        result = await cognitive_service._run_agent(
            query="What is the trend?",
            query_type="monitoring",
            evidence=[{"source": "test", "content": "data", "score": 0.8}],
        )
        assert result["agent_used"] == "drift_monitor"

    @pytest.mark.asyncio
    async def test_agent_routing_default(self, cognitive_service):
        """Unknown query type should route to orchestrator."""
        result = await cognitive_service._run_agent(
            query="test",
            query_type="unknown",
            evidence=[{"source": "test", "content": "data", "score": 0.8}],
        )
        assert result["agent_used"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_agent_low_confidence_without_evidence(self, cognitive_service):
        """Agent should have low confidence without evidence."""
        result = await cognitive_service._run_agent(
            query="test",
            query_type="general",
            evidence=[],
        )
        assert result["confidence"] == 0.3

    @pytest.mark.asyncio
    async def test_agent_confidence_from_evidence_scores(self, cognitive_service):
        """Agent confidence should be derived from evidence scores."""
        evidence = [
            {"source": "test", "content": "data1", "score": 0.9},
            {"source": "test", "content": "data2", "score": 0.8},
        ]
        result = await cognitive_service._run_agent(
            query="test",
            query_type="general",
            evidence=evidence,
        )
        # Use approx for floating-point comparison: average of 0.9 and 0.8
        assert result["confidence"] == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_agent_confidence_capped_at_95(self, cognitive_service):
        """Agent confidence should be capped at 95%."""
        evidence = [
            {"source": "test", "content": "data", "score": 0.99},
        ]
        result = await cognitive_service._run_agent(
            query="test",
            query_type="general",
            evidence=evidence,
        )
        assert result["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_agent_visualization_config_causal(self, cognitive_service):
        """Causal queries should get sankey visualization config."""
        result = await cognitive_service._run_agent(
            query="Why?",
            query_type="causal",
            evidence=[{"source": "test", "content": "data", "score": 0.8}],
        )
        assert result["visualization_config"]["chart_type"] == "sankey"

    @pytest.mark.asyncio
    async def test_agent_visualization_config_prediction(self, cognitive_service):
        """Prediction queries should get line visualization config."""
        result = await cognitive_service._run_agent(
            query="What next?",
            query_type="prediction",
            evidence=[{"source": "test", "content": "data", "score": 0.8}],
        )
        assert result["visualization_config"]["chart_type"] == "line"


class TestReflectorPhase:
    """Tests for _run_reflector phase."""

    @pytest.mark.asyncio
    async def test_reflector_skips_low_confidence(self, cognitive_service):
        """Reflector should skip processing for low confidence."""
        with patch(
            "src.memory.cognitive_integration.insert_episodic_memory_with_text"
        ) as mock_insert:
            await cognitive_service._run_reflector(
                session_id="session-123",
                cycle_id="cycle-456",
                query="test",
                query_type="general",
                response="response",
                confidence=0.5,  # Below 0.6 threshold
                evidence=[],
                agent_used="orchestrator",
            )
            mock_insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflector_stores_episodic_memory(self, cognitive_service):
        """Reflector should store episodic memory for high confidence."""
        with patch(
            "src.memory.cognitive_integration.insert_episodic_memory_with_text",
            new_callable=AsyncMock,
        ) as mock_insert, patch(
            "src.memory.cognitive_integration.record_learning_signal",
            new_callable=AsyncMock,
        ), patch(
            "src.memory.cognitive_integration.get_graphiti_service",
            new_callable=AsyncMock,
        ):
            await cognitive_service._run_reflector(
                session_id="session-123",
                cycle_id="cycle-456",
                query="test query",
                query_type="causal",
                response="response",
                confidence=0.8,
                evidence=[],
                agent_used="causal_impact",
            )
            mock_insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflector_records_learning_signal(self, cognitive_service):
        """Reflector should record learning signal."""
        with patch(
            "src.memory.cognitive_integration.insert_episodic_memory_with_text",
            new_callable=AsyncMock,
        ), patch(
            "src.memory.cognitive_integration.record_learning_signal",
            new_callable=AsyncMock,
        ) as mock_signal, patch(
            "src.memory.cognitive_integration.get_graphiti_service",
            new_callable=AsyncMock,
        ):
            await cognitive_service._run_reflector(
                session_id="session-123",
                cycle_id="cycle-456",
                query="test",
                query_type="general",
                response="response",
                confidence=0.8,
                evidence=[],
                agent_used="orchestrator",
            )
            mock_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflector_stores_to_graphiti(
        self, cognitive_service, mock_graphiti_service
    ):
        """Reflector should store to Graphiti knowledge graph."""
        with patch(
            "src.memory.cognitive_integration.insert_episodic_memory_with_text",
            new_callable=AsyncMock,
        ), patch(
            "src.memory.cognitive_integration.record_learning_signal",
            new_callable=AsyncMock,
        ), patch(
            "src.memory.cognitive_integration.get_graphiti_service",
            new_callable=AsyncMock,
            return_value=mock_graphiti_service,
        ):
            await cognitive_service._run_reflector(
                session_id="session-123",
                cycle_id="cycle-456",
                query="test",
                query_type="causal",
                response="response",
                confidence=0.8,
                evidence=[],
                agent_used="causal_impact",
            )
            mock_graphiti_service.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflector_handles_errors_gracefully(self, cognitive_service):
        """Reflector should handle errors without raising."""
        with patch(
            "src.memory.cognitive_integration.insert_episodic_memory_with_text",
            new_callable=AsyncMock,
            side_effect=Exception("Database error"),
        ):
            # Should not raise
            await cognitive_service._run_reflector(
                session_id="session-123",
                cycle_id="cycle-456",
                query="test",
                query_type="general",
                response="response",
                confidence=0.8,
                evidence=[],
                agent_used="orchestrator",
            )


class TestStoreToGraphiti:
    """Tests for _store_to_graphiti method."""

    @pytest.mark.asyncio
    async def test_store_to_graphiti_builds_episode_content(
        self, cognitive_service, mock_graphiti_service
    ):
        """Graphiti episode content should include query and response."""
        with patch(
            "src.memory.cognitive_integration.get_graphiti_service",
            new_callable=AsyncMock,
            return_value=mock_graphiti_service,
        ):
            await cognitive_service._store_to_graphiti(
                session_id="session-123",
                cycle_id="cycle-456",
                query="Why is TRx declining?",
                query_type="causal",
                response="TRx is declining because...",
                confidence=0.85,
                agent_used="causal_impact",
            )
            call_kwargs = mock_graphiti_service.add_episode.call_args.kwargs
            assert "Why is TRx declining?" in call_kwargs["content"]
            assert "TRx is declining because" in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_store_to_graphiti_handles_failure(
        self, cognitive_service
    ):
        """Graphiti storage failure should not raise."""
        mock_service = AsyncMock()
        mock_service.add_episode = AsyncMock(side_effect=Exception("Graphiti error"))

        with patch(
            "src.memory.cognitive_integration.get_graphiti_service",
            new_callable=AsyncMock,
            return_value=mock_service,
        ):
            # Should not raise
            await cognitive_service._store_to_graphiti(
                session_id="session-123",
                cycle_id="cycle-456",
                query="test",
                query_type="general",
                response="response",
                confidence=0.8,
                agent_used="orchestrator",
            )


class TestProcessQuery:
    """Tests for process_query main orchestration."""

    @pytest.mark.asyncio
    async def test_process_query_full_cycle(self, cognitive_service, mock_working_memory):
        """process_query should execute full cognitive cycle."""
        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ), patch.object(
            cognitive_service, "_run_reflector", new_callable=AsyncMock
        ):
            input_data = CognitiveQueryInput(
                query="Why is TRx declining?",
                user_id="user-123",
            )
            result = await cognitive_service.process_query(input_data)

            assert result.session_id is not None
            assert result.cycle_id is not None
            assert result.query == "Why is TRx declining?"
            assert "summarizer" in result.phases_completed
            assert "investigator" in result.phases_completed
            assert "agent" in result.phases_completed

    @pytest.mark.asyncio
    async def test_process_query_creates_session(self, cognitive_service, mock_working_memory):
        """process_query should create session if not provided."""
        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ), patch.object(
            cognitive_service, "_run_reflector", new_callable=AsyncMock
        ):
            input_data = CognitiveQueryInput(
                query="test",
                session_id=None,
                user_id="user-123",
            )
            await cognitive_service.process_query(input_data)

            mock_working_memory.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_adds_messages(self, cognitive_service, mock_working_memory):
        """process_query should add user and assistant messages."""
        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ), patch.object(
            cognitive_service, "_run_reflector", new_callable=AsyncMock
        ):
            input_data = CognitiveQueryInput(
                query="test query",
                user_id="user-123",
            )
            await cognitive_service.process_query(input_data)

            # Check both user and assistant messages added
            assert mock_working_memory.add_message.call_count == 2

    @pytest.mark.asyncio
    async def test_process_query_stores_evidence(self, cognitive_service, mock_working_memory):
        """process_query should store evidence if include_evidence is True."""
        mock_result = MagicMock()
        mock_result.source = "test"
        mock_result.content = "evidence"
        mock_result.score = 0.8
        mock_result.retrieval_method = "vector"
        mock_result.metadata = {}

        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            return_value=[mock_result],
        ), patch.object(
            cognitive_service, "_run_reflector", new_callable=AsyncMock
        ):
            input_data = CognitiveQueryInput(
                query="test",
                include_evidence=True,
            )
            await cognitive_service.process_query(input_data)

            mock_working_memory.append_evidence.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_handles_error(self, cognitive_service, mock_working_memory):
        """process_query should handle errors gracefully."""
        mock_working_memory.create_session.side_effect = Exception("Redis error")

        input_data = CognitiveQueryInput(
            query="test",
            user_id="user-123",
        )
        result = await cognitive_service.process_query(input_data)

        assert result.query_type == "error"
        assert result.agent_used == "error_handler"
        assert result.confidence == 0.0
        assert "error" in result.response.lower()

    @pytest.mark.asyncio
    async def test_process_query_worth_remembering_threshold(
        self, cognitive_service, mock_working_memory
    ):
        """worth_remembering should be True for confidence > 0.6."""
        # High confidence evidence
        mock_result = MagicMock()
        mock_result.source = "test"
        mock_result.content = "evidence"
        mock_result.score = 0.9
        mock_result.retrieval_method = "vector"
        mock_result.metadata = {}

        with patch(
            "src.memory.cognitive_integration.hybrid_search",
            new_callable=AsyncMock,
            return_value=[mock_result],
        ), patch.object(
            cognitive_service, "_run_reflector", new_callable=AsyncMock
        ):
            input_data = CognitiveQueryInput(query="test")
            result = await cognitive_service.process_query(input_data)

            # With 0.9 score evidence, confidence should be ~0.9 (capped at 0.95)
            assert result.worth_remembering is True


# ============================================================================
# SINGLETON TESTS
# ============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_cognitive_service_returns_singleton(self):
        """get_cognitive_service should return same instance."""
        # Reset singleton
        import src.memory.cognitive_integration as module

        module._cognitive_service = None

        service1 = get_cognitive_service()
        service2 = get_cognitive_service()

        assert service1 is service2

    def test_get_cognitive_service_creates_new_instance(self):
        """get_cognitive_service should create instance if none exists."""
        import src.memory.cognitive_integration as module

        module._cognitive_service = None

        service = get_cognitive_service()

        assert service is not None
        assert isinstance(service, CognitiveService)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_process_cognitive_query_uses_service(self):
        """process_cognitive_query should use the singleton service."""
        with patch(
            "src.memory.cognitive_integration.get_cognitive_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.process_query = AsyncMock(
                return_value=CognitiveQueryOutput(
                    session_id="session-123",
                    cycle_id="cycle-456",
                    query="test",
                    query_type="general",
                    agent_used="orchestrator",
                    response="response",
                    confidence=0.5,
                    phases_completed=[],
                    processing_time_ms=100.0,
                    worth_remembering=False,
                )
            )
            mock_get_service.return_value = mock_service

            result = await process_cognitive_query(
                query="test query",
                user_id="user-123",
                brand="Kisqali",
            )

            mock_service.process_query.assert_called_once()
            assert result.query == "test"

    @pytest.mark.asyncio
    async def test_process_cognitive_query_passes_all_params(self):
        """process_cognitive_query should pass all parameters."""
        with patch(
            "src.memory.cognitive_integration.get_cognitive_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.process_query = AsyncMock(
                return_value=MagicMock(spec=CognitiveQueryOutput)
            )
            mock_get_service.return_value = mock_service

            await process_cognitive_query(
                query="test",
                session_id="existing-session",
                user_id="user-123",
                brand="Kisqali",
                region="Northeast",
                include_evidence=False,
            )

            call_args = mock_service.process_query.call_args[0][0]
            assert call_args.query == "test"
            assert call_args.session_id == "existing-session"
            assert call_args.user_id == "user-123"
            assert call_args.brand == "Kisqali"
            assert call_args.region == "Northeast"
            assert call_args.include_evidence is False
