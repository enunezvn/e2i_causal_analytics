"""
Comprehensive tests for CausalRAG orchestrator.

Tests cover:
- Initialization with various retriever combinations
- Hybrid retrieval (vector, graph, KPI)
- Async retrieval with timing
- Cognitive search with DSPy workflow
- Error handling paths
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time

from src.rag.causal_rag import CausalRAG
from src.rag.models.retrieval_models import RetrievalResult, RetrievalContext
from src.rag.types import RetrievalSource


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockParsedQuery:
    """Mock ParsedQuery from NLP layer."""
    text: str = "Why did Kisqali adoption increase?"
    intent: Optional[Mock] = None
    entities: Optional[Mock] = None


@dataclass
class MockIntent:
    """Mock intent with value attribute."""
    value: str = "causal"


@dataclass
class MockEntities:
    """Mock entities with kpis attribute."""
    kpis: List[str] = field(default_factory=lambda: ["TRx", "market_share"])


def create_retrieval_result(
    content: str = "Test content",
    source: RetrievalSource = RetrievalSource.VECTOR,
    source_id: str = "test-123",
    score: float = 0.85,
    retrieval_method: str = "dense",
    metadata: Dict[str, Any] = None
) -> RetrievalResult:
    """Helper to create RetrievalResult instances."""
    # Map retrieval_method to RetrievalSource if not already set
    if source == RetrievalSource.VECTOR and retrieval_method != "dense":
        source_map = {
            "dense": RetrievalSource.VECTOR,
            "sparse": RetrievalSource.FULLTEXT,
            "bm25": RetrievalSource.FULLTEXT,
            "graph": RetrievalSource.GRAPH,
        }
        source = source_map.get(retrieval_method, RetrievalSource.VECTOR)

    return RetrievalResult(
        id=source_id,
        content=content,
        source=source,
        score=score,
        metadata={**(metadata or {}), "retrieval_method": retrieval_method}
    )


@pytest.fixture
def mock_vector_retriever():
    """Create mock vector retriever."""
    retriever = Mock()
    retriever.search.return_value = [
        create_retrieval_result(
            content="Vector result 1",
            source=RetrievalSource.VECTOR,
            source_id="vec-1",
            score=0.9,
            retrieval_method="dense"
        ),
        create_retrieval_result(
            content="Vector result 2",
            source=RetrievalSource.VECTOR,
            source_id="vec-2",
            score=0.8,
            retrieval_method="dense"
        ),
    ]
    return retriever


@pytest.fixture
def mock_graph_retriever():
    """Create mock graph retriever."""
    retriever = Mock()
    retriever.traverse.return_value = [
        create_retrieval_result(
            content="Causal path: Marketing -> HCP engagement -> Adoption",
            source=RetrievalSource.GRAPH,
            source_id="graph-1",
            score=0.85,
            retrieval_method="graph"
        ),
    ]
    return retriever


@pytest.fixture
def mock_kpi_retriever():
    """Create mock KPI retriever."""
    retriever = Mock()
    retriever.query.return_value = [
        create_retrieval_result(
            content="TRx increased 15% in Northeast region",
            source=RetrievalSource.FULLTEXT,
            source_id="kpi-1",
            score=0.95,
            retrieval_method="structured"
        ),
    ]
    return retriever


@pytest.fixture
def mock_reranker():
    """Create mock reranker."""
    reranker = Mock()
    # Reranker returns sorted results
    def rerank_fn(results, query, top_k=10):
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
    reranker.rerank.side_effect = rerank_fn
    return reranker


@pytest.fixture
def causal_rag(mock_vector_retriever, mock_graph_retriever, mock_kpi_retriever, mock_reranker):
    """Create CausalRAG with all retrievers."""
    return CausalRAG(
        vector_retriever=mock_vector_retriever,
        graph_retriever=mock_graph_retriever,
        kpi_retriever=mock_kpi_retriever,
        reranker=mock_reranker
    )


@pytest.fixture
def causal_query():
    """Create a causal intent query."""
    query = MockParsedQuery()
    query.intent = MockIntent(value="causal")
    query.entities = MockEntities(kpis=["TRx"])
    return query


@pytest.fixture
def simple_query():
    """Create a simple query without intent or entities."""
    return MockParsedQuery(text="What is the current TRx?")


# ============================================================================
# Initialization Tests
# ============================================================================


class TestCausalRAGInit:
    """Tests for CausalRAG initialization."""

    def test_init_with_all_components(
        self, mock_vector_retriever, mock_graph_retriever,
        mock_kpi_retriever, mock_reranker
    ):
        """Test initialization with all components."""
        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            kpi_retriever=mock_kpi_retriever,
            reranker=mock_reranker
        )

        assert rag.vector_retriever == mock_vector_retriever
        assert rag.graph_retriever == mock_graph_retriever
        assert rag.kpi_retriever == mock_kpi_retriever
        assert rag.reranker == mock_reranker

    def test_init_with_no_components(self):
        """Test initialization without any components."""
        rag = CausalRAG()

        assert rag.vector_retriever is None
        assert rag.graph_retriever is None
        assert rag.kpi_retriever is None
        assert rag.reranker is None

    def test_init_with_vector_only(self, mock_vector_retriever):
        """Test initialization with vector retriever only."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        assert rag.vector_retriever == mock_vector_retriever
        assert rag.graph_retriever is None
        assert rag.kpi_retriever is None
        assert rag.reranker is None

    def test_init_with_graph_only(self, mock_graph_retriever):
        """Test initialization with graph retriever only."""
        rag = CausalRAG(graph_retriever=mock_graph_retriever)

        assert rag.vector_retriever is None
        assert rag.graph_retriever == mock_graph_retriever

    def test_init_with_reranker_only(self, mock_reranker):
        """Test initialization with reranker only."""
        rag = CausalRAG(reranker=mock_reranker)

        assert rag.reranker == mock_reranker
        assert rag.vector_retriever is None


# ============================================================================
# Retrieve Method Tests
# ============================================================================


class TestRetrieve:
    """Tests for the retrieve() method."""

    def test_retrieve_vector_only(self, mock_vector_retriever, simple_query):
        """Test retrieve with only vector retriever."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        results = rag.retrieve(simple_query, top_k=5)

        mock_vector_retriever.search.assert_called_once()
        assert len(results) == 2  # Mock returns 2 results

    def test_retrieve_uses_query_text(self, mock_vector_retriever, simple_query):
        """Test that retrieve uses query.text for vector search."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        rag.retrieve(simple_query, top_k=5)

        call_args = mock_vector_retriever.search.call_args
        assert call_args[0][0] == simple_query.text

    def test_retrieve_string_query_fallback(self, mock_vector_retriever):
        """Test retrieve with string query (no text attribute)."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)
        query = "Simple string query"

        rag.retrieve(query, top_k=5)

        call_args = mock_vector_retriever.search.call_args
        assert call_args[0][0] == query

    def test_retrieve_graph_for_causal_intent(
        self, mock_vector_retriever, mock_graph_retriever, causal_query
    ):
        """Test that graph retriever is used for causal intent."""
        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever
        )

        results = rag.retrieve(causal_query, top_k=10)

        mock_graph_retriever.traverse.assert_called_once()
        # Should have results from both vector and graph
        assert len(results) == 3

    def test_retrieve_skips_graph_for_non_causal(
        self, mock_vector_retriever, mock_graph_retriever, simple_query
    ):
        """Test that graph retriever is skipped for non-causal queries."""
        simple_query.intent = MockIntent(value="descriptive")

        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever
        )

        rag.retrieve(simple_query, top_k=10)

        mock_graph_retriever.traverse.assert_not_called()

    def test_retrieve_skips_graph_without_intent(
        self, mock_vector_retriever, mock_graph_retriever, simple_query
    ):
        """Test that graph retriever is skipped when query has no intent."""
        simple_query.intent = None

        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever
        )

        rag.retrieve(simple_query, top_k=10)

        mock_graph_retriever.traverse.assert_not_called()

    def test_retrieve_kpi_with_kpis(
        self, mock_vector_retriever, mock_kpi_retriever, causal_query
    ):
        """Test KPI retrieval when query has KPIs."""
        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            kpi_retriever=mock_kpi_retriever
        )

        results = rag.retrieve(causal_query, top_k=10)

        mock_kpi_retriever.query.assert_called_once_with(["TRx"])
        assert len(results) == 3

    def test_retrieve_skips_kpi_without_kpis(
        self, mock_vector_retriever, mock_kpi_retriever, simple_query
    ):
        """Test KPI retrieval skipped when no KPIs in query."""
        simple_query.entities = MockEntities(kpis=[])

        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            kpi_retriever=mock_kpi_retriever
        )

        rag.retrieve(simple_query, top_k=10)

        mock_kpi_retriever.query.assert_not_called()

    def test_retrieve_skips_kpi_without_entities(
        self, mock_vector_retriever, mock_kpi_retriever, simple_query
    ):
        """Test KPI retrieval skipped when query has no entities."""
        simple_query.entities = None

        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            kpi_retriever=mock_kpi_retriever
        )

        rag.retrieve(simple_query, top_k=10)

        mock_kpi_retriever.query.assert_not_called()

    def test_retrieve_with_reranker(self, causal_rag, causal_query):
        """Test that reranker is called when provided."""
        results = causal_rag.retrieve(causal_query, top_k=5)

        causal_rag.reranker.rerank.assert_called_once()
        # Results should be sorted by score
        assert results[0].score >= results[-1].score

    def test_retrieve_without_reranker(
        self, mock_vector_retriever, mock_graph_retriever, causal_query
    ):
        """Test retrieve without reranker returns raw results."""
        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever
        )

        results = rag.retrieve(causal_query, top_k=10)

        # No reranking, just concatenated results
        assert len(results) == 3

    def test_retrieve_respects_top_k(self, mock_vector_retriever, simple_query):
        """Test that top_k limits results."""
        # Mock returns 5 results
        mock_vector_retriever.search.return_value = [
            create_retrieval_result(source_id=f"vec-{i}", score=0.9 - i*0.1)
            for i in range(5)
        ]

        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        results = rag.retrieve(simple_query, top_k=3)

        assert len(results) == 3

    def test_retrieve_no_retrievers(self, simple_query):
        """Test retrieve with no retrievers returns empty list."""
        rag = CausalRAG()

        results = rag.retrieve(simple_query)

        assert results == []

    def test_retrieve_empty_results(self, mock_vector_retriever, simple_query):
        """Test retrieve when vector search returns empty."""
        mock_vector_retriever.search.return_value = []

        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        results = rag.retrieve(simple_query)

        assert results == []

    def test_retrieve_full_hybrid(self, causal_rag, causal_query):
        """Test full hybrid retrieval with all sources."""
        results = causal_rag.retrieve(causal_query, top_k=10)

        # Should have combined results from all sources
        causal_rag.vector_retriever.search.assert_called_once()
        causal_rag.graph_retriever.traverse.assert_called_once()
        causal_rag.kpi_retriever.query.assert_called_once()
        causal_rag.reranker.rerank.assert_called_once()

        assert len(results) > 0

    def test_retrieve_with_config(self, mock_vector_retriever, simple_query):
        """Test retrieve with custom retrieval config."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        config = {"enable_sparse": True, "sparse_weight": 0.3}
        results = rag.retrieve(simple_query, retrieval_config=config)

        # Config is accepted but may not be used by all retrievers
        assert results is not None

    def test_retrieve_graph_entities_extraction(
        self, mock_graph_retriever, causal_query
    ):
        """Test that graph retriever receives entities from query."""
        rag = CausalRAG(graph_retriever=mock_graph_retriever)

        rag.retrieve(causal_query, top_k=10)

        call_kwargs = mock_graph_retriever.traverse.call_args[1]
        assert call_kwargs["relationship"] == "causal_path"


# ============================================================================
# Async Retrieve Tests
# ============================================================================


class TestRetrieveAsync:
    """Tests for the retrieve_async() method."""

    @pytest.mark.asyncio
    async def test_retrieve_async_returns_context(
        self, mock_vector_retriever, simple_query
    ):
        """Test that retrieve_async returns RetrievalContext."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        context = await rag.retrieve_async(simple_query, top_k=5)

        assert isinstance(context, RetrievalContext)
        assert context.query == simple_query
        assert context.total_retrieved == len(context.results)

    @pytest.mark.asyncio
    async def test_retrieve_async_timing(self, mock_vector_retriever, simple_query):
        """Test that retrieve_async records timing."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        context = await rag.retrieve_async(simple_query)

        assert context.retrieval_time_ms >= 0
        assert isinstance(context.retrieval_time_ms, float)

    @pytest.mark.asyncio
    async def test_retrieve_async_with_config(
        self, mock_vector_retriever, simple_query
    ):
        """Test retrieve_async with custom config."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        config = {"enable_sparse": True}
        context = await rag.retrieve_async(
            simple_query, top_k=5, retrieval_config=config
        )

        assert isinstance(context, RetrievalContext)

    @pytest.mark.asyncio
    async def test_retrieve_async_empty_results(
        self, mock_vector_retriever, simple_query
    ):
        """Test retrieve_async with empty results."""
        mock_vector_retriever.search.return_value = []
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        context = await rag.retrieve_async(simple_query)

        assert context.total_retrieved == 0
        assert context.results == []


# ============================================================================
# Cognitive Search Tests
# ============================================================================


class TestCognitiveSearch:
    """Tests for the cognitive_search() method."""

    @pytest.fixture
    def mock_cognitive_state(self):
        """Create mock cognitive state result."""
        state = Mock()
        state.response = "Kisqali adoption increased due to HCP engagement."
        state.evidence_board = []
        state.hop_count = 2
        state.visualization_config = {"chart_type": "bar"}
        state.routed_agents = ["explainer"]
        state.extracted_entities = ["Kisqali", "Northeast"]
        state.detected_intent = "causal"
        state.rewritten_query = "What drove Kisqali adoption in Northeast?"
        state.dspy_signals = []
        state.worth_remembering = True
        return state

    @pytest.mark.asyncio
    async def test_cognitive_search_success(self, mock_cognitive_state):
        """Test successful cognitive search."""
        import sys
        rag = CausalRAG()

        # Create mock dspy module
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None
        mock_dspy.LM.return_value = Mock()
        mock_dspy.configure = Mock()

        # Mock the cognitive workflow creator
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = mock_cognitive_state

        # Mock CognitiveState
        mock_cog_state_class = Mock(return_value=mock_cognitive_state)

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict(sys.modules, {'dspy': mock_dspy}):
                with patch(
                    'src.rag.cognitive_rag_dspy.create_dspy_cognitive_workflow',
                    return_value=mock_workflow
                ):
                    with patch(
                        'src.rag.cognitive_rag_dspy.CognitiveState',
                        mock_cog_state_class
                    ):
                        with patch(
                            'src.rag.cognitive_backends.get_cognitive_memory_backends'
                        ) as mock_backends:
                            mock_backends.return_value = {
                                "readers": {},
                                "writers": {},
                                "signal_collector": Mock()
                            }

                            result = await rag.cognitive_search(
                                query="Why did Kisqali adoption increase?",
                                conversation_id="test-123"
                            )

        assert result["response"] == mock_cognitive_state.response
        assert result["hop_count"] == 2
        assert result["intent"] == "causal"
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_cognitive_search_import_error(self):
        """Test cognitive search with import error - raises RuntimeError."""
        import sys
        rag = CausalRAG()

        # Mock dspy to cause ImportError when accessed
        with patch.dict(sys.modules, {'dspy': None}):
            # When dspy can't be imported, cognitive_search raises RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                await rag.cognitive_search(query="Test query")

            # Should indicate missing dependencies
            assert "additional dependencies" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cognitive_search_missing_api_key(self):
        """Test cognitive search without API key."""
        import sys
        import os
        rag = CausalRAG()

        # Create mock dspy module
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None

        # Remove ANTHROPIC_API_KEY
        original_key = os.environ.pop('ANTHROPIC_API_KEY', None)

        try:
            with patch.dict(sys.modules, {'dspy': mock_dspy}):
                result = await rag.cognitive_search(query="Test query")

                # Should return error response due to missing API key
                assert "error" in result or "Unable to complete" in result.get("response", "")
        finally:
            if original_key:
                os.environ['ANTHROPIC_API_KEY'] = original_key

    @pytest.mark.asyncio
    async def test_cognitive_search_general_exception(self):
        """Test cognitive search with general exception."""
        import sys
        rag = CausalRAG()

        # Create mock dspy module
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None
        mock_dspy.LM.return_value = Mock()
        mock_dspy.configure = Mock()

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict(sys.modules, {'dspy': mock_dspy}):
                with patch(
                    'src.rag.cognitive_backends.get_cognitive_memory_backends'
                ) as mock_backends:
                    mock_backends.side_effect = Exception("Backend connection failed")

                    result = await rag.cognitive_search(query="Test query")

        assert "error" in result
        assert result["hop_count"] == 0
        assert result["evidence"] == []

    @pytest.mark.asyncio
    async def test_cognitive_search_with_history(self, mock_cognitive_state):
        """Test cognitive search with conversation history."""
        import sys
        rag = CausalRAG()

        # Create mock dspy module
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None
        mock_dspy.LM.return_value = Mock()
        mock_dspy.configure = Mock()

        # Mock the cognitive workflow creator
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = mock_cognitive_state

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict(sys.modules, {'dspy': mock_dspy}):
                with patch(
                    'src.rag.cognitive_rag_dspy.create_dspy_cognitive_workflow',
                    return_value=mock_workflow
                ):
                    with patch(
                        'src.rag.cognitive_rag_dspy.CognitiveState'
                    ) as mock_cog_state:
                        mock_cog_state.return_value = mock_cognitive_state

                        with patch(
                            'src.rag.cognitive_backends.get_cognitive_memory_backends'
                        ) as mock_backends:
                            mock_backends.return_value = {
                                "readers": {},
                                "writers": {},
                                "signal_collector": Mock()
                            }

                            result = await rag.cognitive_search(
                                query="Why did adoption increase?",
                                conversation_id="session-456",
                                conversation_history="User asked about Kisqali trends."
                            )

        assert result["response"] is not None

    @pytest.mark.asyncio
    async def test_cognitive_search_with_agent_registry(self, mock_cognitive_state):
        """Test cognitive search with custom agent registry."""
        import sys
        rag = CausalRAG()

        agent_registry = {
            "explainer": {"name": "Explainer Agent"},
            "causal_impact": {"name": "Causal Impact Agent"}
        }

        # Create mock dspy module
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None
        mock_dspy.LM.return_value = Mock()
        mock_dspy.configure = Mock()

        # Mock the cognitive workflow creator
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = mock_cognitive_state

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict(sys.modules, {'dspy': mock_dspy}):
                with patch(
                    'src.rag.cognitive_rag_dspy.create_dspy_cognitive_workflow',
                    return_value=mock_workflow
                ) as mock_create_workflow:
                    with patch(
                        'src.rag.cognitive_rag_dspy.CognitiveState'
                    ) as mock_cog_state:
                        mock_cog_state.return_value = mock_cognitive_state

                        with patch(
                            'src.rag.cognitive_backends.get_cognitive_memory_backends'
                        ) as mock_backends:
                            mock_backends.return_value = {
                                "readers": {},
                                "writers": {},
                                "signal_collector": Mock()
                            }

                            result = await rag.cognitive_search(
                                query="Test query",
                                agent_registry=agent_registry
                            )

                            # Verify agent registry was passed
                            call_kwargs = mock_create_workflow.call_args[1]
                            assert call_kwargs["agent_registry"] == agent_registry

    @pytest.mark.asyncio
    async def test_cognitive_search_evidence_serialization(self):
        """Test that evidence objects are properly serialized."""
        import sys
        rag = CausalRAG()

        # Create mock evidence with dataclass
        @dataclass
        class MockEvidence:
            content: str
            source: str
            score: float

        mock_state = Mock()
        mock_state.response = "Test response"
        mock_state.evidence_board = [
            MockEvidence(content="Evidence 1", source="vector", score=0.9),
            {"content": "Evidence 2", "source": "graph"}  # dict-like evidence
        ]
        mock_state.hop_count = 1
        mock_state.visualization_config = {}
        mock_state.routed_agents = []
        mock_state.extracted_entities = []
        mock_state.detected_intent = "descriptive"
        mock_state.rewritten_query = "Test"
        mock_state.dspy_signals = []
        mock_state.worth_remembering = False

        # Create mock dspy module
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None
        mock_dspy.LM.return_value = Mock()
        mock_dspy.configure = Mock()

        # Mock the cognitive workflow creator
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = mock_state

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict(sys.modules, {'dspy': mock_dspy}):
                with patch(
                    'src.rag.cognitive_rag_dspy.create_dspy_cognitive_workflow',
                    return_value=mock_workflow
                ):
                    with patch(
                        'src.rag.cognitive_rag_dspy.CognitiveState'
                    ) as mock_cog_state:
                        mock_cog_state.return_value = mock_state

                        with patch(
                            'src.rag.cognitive_backends.get_cognitive_memory_backends'
                        ) as mock_backends:
                            mock_backends.return_value = {
                                "readers": {},
                                "writers": {},
                                "signal_collector": Mock()
                            }

                            result = await rag.cognitive_search(query="Test")

        # Evidence should be serialized to dicts
        assert len(result["evidence"]) == 2
        assert isinstance(result["evidence"][0], dict)

    @pytest.mark.asyncio
    async def test_cognitive_search_latency_tracking(self, mock_cognitive_state):
        """Test that latency is properly tracked."""
        import sys
        rag = CausalRAG()

        # Create mock dspy module
        mock_dspy = MagicMock()
        mock_dspy.settings.lm = None
        mock_dspy.LM.return_value = Mock()
        mock_dspy.configure = Mock()

        # Mock the cognitive workflow creator
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = mock_cognitive_state

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch.dict(sys.modules, {'dspy': mock_dspy}):
                with patch(
                    'src.rag.cognitive_rag_dspy.create_dspy_cognitive_workflow',
                    return_value=mock_workflow
                ):
                    with patch(
                        'src.rag.cognitive_rag_dspy.CognitiveState'
                    ) as mock_cog_state:
                        mock_cog_state.return_value = mock_cognitive_state

                        with patch(
                            'src.rag.cognitive_backends.get_cognitive_memory_backends'
                        ) as mock_backends:
                            mock_backends.return_value = {
                                "readers": {},
                                "writers": {},
                                "signal_collector": Mock()
                            }

                            result = await rag.cognitive_search(query="Test")

        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests for CausalRAG."""

    def test_retrieve_with_none_values(self):
        """Test retrieve handles None values gracefully."""
        rag = CausalRAG(
            vector_retriever=None,
            graph_retriever=None,
            kpi_retriever=None,
            reranker=None
        )

        results = rag.retrieve(MockParsedQuery(), top_k=10)

        assert results == []

    def test_retrieve_empty_query_text(self, mock_vector_retriever):
        """Test retrieve with empty query text."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)
        query = MockParsedQuery(text="")

        results = rag.retrieve(query)

        mock_vector_retriever.search.assert_called_once()

    def test_retrieve_large_top_k(self, mock_vector_retriever, simple_query):
        """Test retrieve with large top_k."""
        mock_vector_retriever.search.return_value = [
            create_retrieval_result(source_id=f"vec-{i}")
            for i in range(3)
        ]

        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        results = rag.retrieve(simple_query, top_k=1000)

        # Should return all available (only 3)
        assert len(results) == 3

    def test_retrieve_top_k_one(self, mock_vector_retriever, simple_query):
        """Test retrieve with top_k=1."""
        mock_vector_retriever.search.return_value = [
            create_retrieval_result(source_id=f"vec-{i}", score=0.9 - i*0.1)
            for i in range(5)
        ]

        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        results = rag.retrieve(simple_query, top_k=1)

        assert len(results) == 1

    def test_retrieve_query_without_text_attr(self, mock_vector_retriever):
        """Test retrieve when query object has no text attribute."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        class QueryNoText:
            pass

        query = QueryNoText()
        results = rag.retrieve(query)

        # Should fall back to str(query)
        mock_vector_retriever.search.assert_called_once()

    def test_retrieve_entities_without_kpis_attr(
        self, mock_vector_retriever, mock_kpi_retriever
    ):
        """Test retrieve when entities have no kpis attribute."""
        class EntitiesNoKpis:
            brands = ["Kisqali"]

        query = MockParsedQuery()
        query.entities = EntitiesNoKpis()

        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            kpi_retriever=mock_kpi_retriever
        )

        results = rag.retrieve(query)

        # KPI retriever should not be called
        mock_kpi_retriever.query.assert_not_called()

    def test_retrieve_reranker_not_called_with_empty_results(
        self, mock_vector_retriever, mock_reranker, simple_query
    ):
        """Test reranker is not called when results are empty."""
        mock_vector_retriever.search.return_value = []

        rag = CausalRAG(
            vector_retriever=mock_vector_retriever,
            reranker=mock_reranker
        )

        results = rag.retrieve(simple_query)

        mock_reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_async_no_retrievers(self):
        """Test retrieve_async with no retrievers."""
        rag = CausalRAG()

        context = await rag.retrieve_async(MockParsedQuery())

        assert context.total_retrieved == 0
        assert context.results == []


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration-style tests for CausalRAG."""

    def test_full_retrieval_pipeline(self):
        """Test complete retrieval pipeline with all components."""
        # Create mock components
        vector_ret = Mock()
        vector_ret.search.return_value = [
            create_retrieval_result(
                content="Kisqali market share increased 5%",
                source="embeddings",
                source_id="v1",
                score=0.88,
                retrieval_method="dense"
            )
        ]

        graph_ret = Mock()
        graph_ret.traverse.return_value = [
            create_retrieval_result(
                content="HCP engagement -> Adoption increase",
                source="causal_paths",
                source_id="g1",
                score=0.92,
                retrieval_method="graph"
            )
        ]

        kpi_ret = Mock()
        kpi_ret.query.return_value = [
            create_retrieval_result(
                content="TRx: 1,234 (+15%)",
                source="kpi_snapshots",
                source_id="k1",
                score=1.0,
                retrieval_method="structured"
            )
        ]

        reranker = Mock()
        reranker.rerank.side_effect = lambda r, q, top_k: sorted(
            r, key=lambda x: x.score, reverse=True
        )[:top_k]

        rag = CausalRAG(
            vector_retriever=vector_ret,
            graph_retriever=graph_ret,
            kpi_retriever=kpi_ret,
            reranker=reranker
        )

        query = MockParsedQuery(text="Why did Kisqali adoption increase?")
        query.intent = MockIntent(value="causal")
        query.entities = MockEntities(kpis=["TRx"])

        results = rag.retrieve(query, top_k=10)

        # All retrievers called
        vector_ret.search.assert_called_once()
        graph_ret.traverse.assert_called_once()
        kpi_ret.query.assert_called_once()
        reranker.rerank.assert_called_once()

        # Results combined and sorted
        assert len(results) == 3
        assert results[0].score >= results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_full_async_pipeline(self):
        """Test complete async retrieval pipeline."""
        vector_ret = Mock()
        vector_ret.search.return_value = [
            create_retrieval_result(score=0.9)
        ]

        rag = CausalRAG(vector_retriever=vector_ret)

        start = time.time()
        context = await rag.retrieve_async(MockParsedQuery())
        elapsed = time.time() - start

        assert context.total_retrieved == 1
        assert context.retrieval_time_ms >= 0
        # Timing should be roughly accurate
        assert context.retrieval_time_ms < (elapsed * 1000) + 100


# ============================================================================
# Config and Defaults Tests
# ============================================================================


class TestConfigAndDefaults:
    """Tests for configuration handling and defaults."""

    def test_default_top_k(self, mock_vector_retriever, simple_query):
        """Test default top_k value."""
        mock_vector_retriever.search.return_value = [
            create_retrieval_result(source_id=f"v{i}")
            for i in range(15)
        ]

        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        results = rag.retrieve(simple_query)  # No top_k specified

        assert len(results) == 10  # Default top_k is 10

    def test_retrieval_config_accepted(
        self, mock_vector_retriever, simple_query
    ):
        """Test that retrieval_config is accepted."""
        rag = CausalRAG(vector_retriever=mock_vector_retriever)

        config = {
            "enable_sparse": True,
            "sparse_weight": 0.3,
            "enable_graph": False
        }

        # Should not raise
        results = rag.retrieve(simple_query, retrieval_config=config)

        assert isinstance(results, list)
