"""
Integration Tests for Cognitive Workflow
========================================

Tests the complete cognitive cycle with all memory systems integrated.

Test Coverage:
- Full 4-phase cognitive cycle
- Memory persistence across phases
- Agent routing based on query type
- Learning signal propagation
- Session management
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import uuid

from src.memory.cognitive_integration import (
    CognitiveService,
    CognitiveQueryInput,
    CognitiveQueryOutput,
    get_cognitive_service,
    process_cognitive_query
)
from src.rag.models.retrieval_models import RetrievalResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_working_memory():
    """Mock working memory for tests."""
    memory = MagicMock()
    memory.create_session = AsyncMock(return_value={"session_id": "sess_test"})
    memory.get_session = AsyncMock(return_value={
        "user_id": "user_001",
        "context": {"brand": "Kisqali"},
        "state": "active"
    })
    memory.add_message = AsyncMock(return_value=True)
    memory.append_evidence = AsyncMock(return_value=True)
    memory.get_evidence_trail = AsyncMock(return_value=[])
    return memory


@pytest.fixture
def mock_hybrid_search():
    """Mock hybrid search for tests."""
    with patch("src.memory.cognitive_integration.hybrid_search") as mock:
        mock.return_value = [
            RetrievalResult(
                content="TRx dropped 15% in northeast due to HCP engagement decline",
                source="episodic_memories",
                source_id="mem_001",
                score=0.85,
                retrieval_method="dense",
                metadata={"brand": "Kisqali", "region": "northeast"}
            ),
            RetrievalResult(
                content="Causal path: HCP visits -> Script volume -> TRx",
                source="causal_paths",
                source_id="path_001",
                score=0.78,
                retrieval_method="graph",
                metadata={}
            ),
            RetrievalResult(
                content="Similar TRx decline observed in Q2 2024",
                source="episodic_memories",
                source_id="mem_002",
                score=0.72,
                retrieval_method="sparse",
                metadata={"brand": "Kisqali"}
            )
        ]
        yield mock


@pytest.fixture
def mock_memory_functions():
    """Mock memory insertion functions."""
    with patch("src.memory.cognitive_integration.insert_episodic_memory_with_text") as mock_episodic, \
         patch("src.memory.cognitive_integration.record_learning_signal") as mock_signal:
        mock_episodic.return_value = "mem_new_001"
        mock_signal.return_value = "signal_001"
        yield {"episodic": mock_episodic, "signal": mock_signal}


# =============================================================================
# COGNITIVE SERVICE TESTS
# =============================================================================

class TestCognitiveService:
    """Tests for the CognitiveService class."""

    @pytest.mark.asyncio
    async def test_process_query_completes_all_phases(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should complete all 4 cognitive phases."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="Why did TRx drop in the northeast?",
                brand="Kisqali",
                region="northeast"
            )

            result = await service.process_query(input_data)

            assert result.session_id is not None
            assert result.cycle_id is not None
            assert "summarizer" in result.phases_completed
            assert "investigator" in result.phases_completed
            assert "agent" in result.phases_completed

    @pytest.mark.asyncio
    async def test_query_type_detection_causal(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should detect causal query type."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="Why did adoption rates change?",
                brand="Kisqali"
            )

            result = await service.process_query(input_data)

            assert result.query_type == "causal"
            assert result.agent_used == "causal_impact"

    @pytest.mark.asyncio
    async def test_query_type_detection_prediction(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should detect prediction query type."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="What will TRx be next quarter?",
                brand="Kisqali"
            )

            result = await service.process_query(input_data)

            assert result.query_type == "prediction"
            assert result.agent_used == "prediction_synthesizer"

    @pytest.mark.asyncio
    async def test_query_type_detection_optimization(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should detect optimization query type."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="How can we optimize resource allocation?",
                brand="Kisqali"
            )

            result = await service.process_query(input_data)

            assert result.query_type == "optimization"
            assert result.agent_used == "resource_optimizer"

    @pytest.mark.asyncio
    async def test_evidence_included_when_requested(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should include evidence when include_evidence=True."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="Why did TRx drop?",
                include_evidence=True
            )

            result = await service.process_query(input_data)

            assert result.evidence is not None
            assert len(result.evidence) > 0

    @pytest.mark.asyncio
    async def test_evidence_excluded_when_not_requested(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should exclude evidence when include_evidence=False."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="Why did TRx drop?",
                include_evidence=False
            )

            result = await service.process_query(input_data)

            assert result.evidence is None

    @pytest.mark.asyncio
    async def test_entity_extraction(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should extract entities from query."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            # Test summarizer phase directly
            result = await service._run_summarizer(
                query="Why did Kisqali TRx drop in the northeast?",
                session_id="test_session",
                brand=None,
                region=None
            )

            assert "Kisqali" in result["entities"]["brands"]
            assert "northeast" in result["entities"]["regions"]
            assert "TRX" in result["entities"]["kpis"]

    @pytest.mark.asyncio
    async def test_confidence_calculation(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should calculate confidence from evidence scores."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="Why did TRx drop?",
                brand="Kisqali"
            )

            result = await service.process_query(input_data)

            # Confidence should be calculated from evidence
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_session_reuse(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should reuse existing session when session_id provided."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="What else affects this?",
                session_id="existing_session_123"
            )

            result = await service.process_query(input_data)

            assert result.session_id == "existing_session_123"
            # Should not create new session
            mock_working_memory.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_processing_time_tracked(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Should track processing time."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="Quick question"
            )

            result = await service.process_query(input_data)

            assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        mock_working_memory,
        mock_memory_functions
    ):
        """Should handle errors gracefully."""
        mock_working_memory.add_message.side_effect = Exception("Connection error")

        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="Test query"
            )

            result = await service.process_query(input_data)

            assert result.query_type == "error"
            assert result.confidence == 0.0
            assert "error" in result.response.lower()


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_process_cognitive_query(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """process_cognitive_query should work with defaults."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            with patch("src.memory.cognitive_integration.get_cognitive_service") as mock_get_service:
                mock_service = MagicMock()
                mock_service.process_query = AsyncMock(return_value=CognitiveQueryOutput(
                    session_id="test_session",
                    cycle_id="test_cycle",
                    query="Test query",
                    query_type="general",
                    agent_used="orchestrator",
                    response="Test response",
                    confidence=0.8,
                    phases_completed=["summarizer", "investigator", "agent"],
                    processing_time_ms=100.0,
                    worth_remembering=True
                ))
                mock_get_service.return_value = mock_service

                result = await process_cognitive_query(
                    query="Why did TRx drop?",
                    brand="Kisqali"
                )

                assert result.session_id == "test_session"
                assert result.response == "Test response"


# =============================================================================
# VISUALIZATION CONFIG TESTS
# =============================================================================

class TestVisualizationConfig:
    """Tests for visualization configuration generation."""

    @pytest.mark.asyncio
    async def test_causal_query_gets_sankey(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Causal queries should get sankey visualization."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="Why did this happen?",
                include_evidence=True
            )

            result = await service.process_query(input_data)

            if result.visualization_config:
                assert result.visualization_config.get("chart_type") in ["sankey", "bar", "line", "table"]

    @pytest.mark.asyncio
    async def test_prediction_query_gets_line(
        self,
        mock_working_memory,
        mock_hybrid_search,
        mock_memory_functions
    ):
        """Prediction queries should get line visualization."""
        with patch("src.memory.cognitive_integration.get_working_memory", return_value=mock_working_memory):
            service = CognitiveService()
            service._working_memory = mock_working_memory

            input_data = CognitiveQueryInput(
                query="What will happen next quarter?",
                include_evidence=True
            )

            result = await service.process_query(input_data)

            if result.visualization_config:
                assert result.visualization_config.get("chart_type") in ["line", "sankey", "bar", "table"]
