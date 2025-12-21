"""
Integration tests for the complete RAG pipeline.

Tests cover end-to-end flows combining:
- QueryOptimizer → HybridRetriever → CrossEncoderReranker → InsightEnricher
- CausalRAG orchestrator with all components
- Memory integration with RAG pipeline
- Performance validation for SLA compliance

Author: E2I Causal Analytics Team
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import List, Dict, Any

from src.rag.models.retrieval_models import RetrievalResult
from src.rag.types import RetrievalSource


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_retrieval_result(
    content: str,
    source: str = "agent_activities",
    source_id: str = "id_001",
    score: float = 0.9,
    retrieval_method: str = "dense",
    metadata: Dict[str, Any] = None
):
    """Create a RetrievalResult for testing."""
    # Map old source strings to RetrievalSource enum
    source_map = {
        "agent_activities": RetrievalSource.VECTOR,
        "business_metrics": RetrievalSource.VECTOR,
        "causal_paths": RetrievalSource.FULLTEXT,
        "triggers": RetrievalSource.VECTOR,
        "conversations": RetrievalSource.VECTOR,
        "causal_graph": RetrievalSource.GRAPH,
        "test": RetrievalSource.VECTOR,
    }
    source_enum = source_map.get(source, RetrievalSource.VECTOR)

    result_metadata = metadata.copy() if metadata else {}
    result_metadata["retrieval_method"] = retrieval_method
    result_metadata["source_name"] = source

    return RetrievalResult(
        id=source_id,
        content=content,
        source=source_enum,
        score=score,
        metadata=result_metadata
    )


# =============================================================================
# RETRIEVER + RERANKER INTEGRATION TESTS
# =============================================================================

class TestRetrieverRerankerIntegration:
    """Tests for HybridRetriever + CrossEncoderReranker integration."""

    @pytest.fixture
    def mock_results(self):
        """Create mock retrieval results."""
        return [
            create_retrieval_result(
                content="TRx increased 15% due to territory expansion",
                source="agent_activities",
                source_id="act_001",
                score=0.85,
                metadata={"brand": "Kisqali"}
            ),
            create_retrieval_result(
                content="NRx growth driven by HCP targeting",
                source="causal_paths",
                source_id="path_002",
                score=0.75,
                metadata={"brand": "Kisqali"}
            )
        ]

    def test_reranker_accepts_retrieval_results(self, mock_results):
        """Test that reranker accepts RetrievalResult objects."""
        from src.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        query = "Why did TRx increase for Kisqali?"

        reranked = reranker.rerank(mock_results, query, top_k=2)

        assert len(reranked) == 2
        # Reranked results should have content
        assert all(hasattr(r, "content") for r in reranked)
        # Should have updated scores in metadata
        assert all("reranker_score" in r.metadata for r in reranked)

    def test_reranker_preserves_metadata(self, mock_results):
        """Test that reranker preserves document metadata."""
        from src.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(mock_results, "test query", top_k=2)

        # Metadata should be preserved (brand field)
        for result in reranked:
            assert "brand" in result.metadata or "original_score" in result.metadata

    def test_reranker_handles_empty_input(self):
        """Test that reranker handles empty input gracefully."""
        from src.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank([], "test query", top_k=5)

        assert reranked == []


# =============================================================================
# QUERY OPTIMIZER + RETRIEVER INTEGRATION TESTS
# =============================================================================

class TestQueryOptimizerRetrieverIntegration:
    """Tests for QueryOptimizer + HybridRetriever integration."""

    def test_optimizer_output_as_retriever_input(self):
        """Test that QueryOptimizer output works as retriever query."""
        from src.rag.query_optimizer import QueryOptimizer

        optimizer = QueryOptimizer()

        query = "TRx for Kisqali"
        expanded = optimizer.expand(query)

        # Expanded query should be string
        assert isinstance(expanded, str)
        # Should include original terms or expansions
        assert len(expanded) >= len(query)

    def test_temporal_context_added_to_query(self):
        """Test temporal context can be added to queries."""
        from src.rag.query_optimizer import QueryOptimizer

        optimizer = QueryOptimizer()
        query = "TRx trend"
        temporal_query = optimizer.add_temporal_context(query, "Q4 2024")

        assert "Q4 2024" in temporal_query
        assert query in temporal_query


# =============================================================================
# RERANKER + ENRICHER INTEGRATION TESTS
# =============================================================================

class TestRerankerEnricherIntegration:
    """Tests for CrossEncoderReranker + InsightEnricher integration."""

    @pytest.fixture
    def reranked_results(self):
        """Create reranked results for enricher tests."""
        return [
            create_retrieval_result(
                content="TRx increased 15% due to territory expansion",
                source="agent_activities",
                source_id="act_001",
                score=0.92,
                metadata={"brand": "Kisqali", "reranker_score": 0.92}
            ),
            create_retrieval_result(
                content="Market share grew from 12% to 18%",
                source="business_metrics",
                source_id="met_002",
                score=0.88,
                metadata={"brand": "Kisqali", "reranker_score": 0.88}
            )
        ]

    @pytest.mark.asyncio
    async def test_enricher_accepts_retrieval_results(self, reranked_results):
        """Test that enricher can process RetrievalResult list."""
        from src.rag.insight_enricher import InsightEnricher

        enricher = InsightEnricher()

        # Mock the LLM call
        with patch.object(enricher, "_generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "TRx increased significantly..."

            # Mock response parsing
            with patch.object(enricher, "_parse_response") as mock_parse:
                from src.rag.models.insight_models import EnrichedInsight
                mock_parse.return_value = EnrichedInsight(
                    summary="TRx increased due to territory expansion",
                    key_findings=["15% growth"],
                    confidence=0.85
                )

                result = await enricher.enrich(
                    retrieved=reranked_results,
                    query="Why did TRx increase?",
                    max_findings=5
                )

                assert result is not None
                assert hasattr(result, "summary")

    @pytest.mark.asyncio
    async def test_enricher_handles_empty_results(self):
        """Test that enricher handles empty input gracefully."""
        from src.rag.insight_enricher import InsightEnricher
        from src.rag.models.insight_models import EnrichedInsight

        enricher = InsightEnricher()

        result = await enricher.enrich(
            retrieved=[],
            query="test query",
            max_findings=5
        )

        # Should return an EnrichedInsight with empty/default data
        assert result is not None
        assert isinstance(result, EnrichedInsight)


# =============================================================================
# FULL PIPELINE INTEGRATION TESTS
# =============================================================================

class TestFullPipelineIntegration:
    """End-to-end tests for complete RAG pipeline."""

    @pytest.fixture
    def mock_retrieval_results(self):
        """Create mock retrieval results."""
        return [
            create_retrieval_result(
                content="Kisqali TRx increased 15% in Q4",
                source="business_metrics",
                source_id="met_001",
                score=0.9,
                metadata={"brand": "Kisqali", "kpi": "TRx"}
            ),
            create_retrieval_result(
                content="Territory expansion drove growth",
                source="agent_activities",
                source_id="act_001",
                score=0.85,
                metadata={"brand": "Kisqali"}
            ),
            create_retrieval_result(
                content="HCP targeting improved conversion",
                source="causal_paths",
                source_id="path_001",
                score=0.75,
                metadata={"brand": "Kisqali"}
            )
        ]

    def test_pipeline_query_to_reranked_results(self, mock_retrieval_results):
        """Test query processing through optimizer → retriever → reranker."""
        from src.rag.query_optimizer import QueryOptimizer
        from src.rag.reranker import CrossEncoderReranker

        # Step 1: Optimize query
        optimizer = QueryOptimizer()
        query = "Why did TRx increase for Kisqali?"
        expanded_query = optimizer.expand(query)

        assert isinstance(expanded_query, str)
        assert len(expanded_query) > 0

        # Step 2: Retriever would return results (mocked via fixture)
        # Step 3: Rerank results
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(mock_retrieval_results, expanded_query, top_k=2)

        assert len(reranked) == 2
        assert all(hasattr(r, "score") for r in reranked)

    def test_pipeline_preserves_metadata_throughout(self, mock_retrieval_results):
        """Test that metadata is preserved through all pipeline stages."""
        from src.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        query = "TRx for Kisqali"

        reranked = reranker.rerank(mock_retrieval_results, query, top_k=3)

        # All results should preserve source_name in metadata
        for result in reranked:
            assert result.metadata.get("source_name") in ["business_metrics", "agent_activities", "causal_paths"]
            assert hasattr(result, "metadata")


# =============================================================================
# CAUSAL RAG ORCHESTRATOR TESTS
# =============================================================================

class TestCausalRAGOrchestration:
    """Tests for CausalRAG orchestrator integration."""

    def test_causal_rag_retrieve_with_mock_components(self):
        """Test that CausalRAG.retrieve works with mock components."""
        from src.rag.causal_rag import CausalRAG

        # Create mock vector retriever
        mock_vector = MagicMock()
        mock_vector.search = MagicMock(return_value=[
            create_retrieval_result("TRx increased 15%", score=0.9)
        ])

        rag = CausalRAG(vector_retriever=mock_vector)

        # Create mock query
        mock_query = MagicMock()
        mock_query.text = "Why did TRx increase?"

        result = rag.retrieve(mock_query, top_k=5)

        assert isinstance(result, list)

    def test_causal_rag_handles_no_retrievers(self):
        """Test CausalRAG handles case with no retrievers configured."""
        from src.rag.causal_rag import CausalRAG

        rag = CausalRAG()  # No retrievers

        mock_query = MagicMock()
        mock_query.text = "Test query"

        result = rag.retrieve(mock_query)

        # Should return empty list, not raise
        assert isinstance(result, list)
        assert len(result) == 0


# =============================================================================
# MEMORY INTEGRATION TESTS
# =============================================================================

class TestMemoryRAGIntegration:
    """Tests for Memory Backend + RAG pipeline integration."""

    def test_episodic_memory_results_as_reranker_input(self):
        """Test that episodic memory results work with reranker."""
        from src.rag.reranker import CrossEncoderReranker

        # Simulate episodic memory search results
        memory_results = [
            create_retrieval_result(
                content="Previous conversation about TRx",
                source="conversations",
                source_id="conv_001",
                score=0.88,
                retrieval_method="dense",
                metadata={"turn_index": 5}
            )
        ]

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(memory_results, "TRx question", top_k=5)

        assert len(reranked) == 1
        assert reranked[0].metadata.get("source_name") == "conversations"

    def test_semantic_memory_results_as_reranker_input(self):
        """Test that semantic memory results work with reranker."""
        from src.rag.reranker import CrossEncoderReranker

        # Simulate semantic (graph) memory results
        graph_results = [
            create_retrieval_result(
                content="Causal relationship: HCP visits → TRx increase",
                source="causal_graph",
                source_id="edge_001",
                score=0.92,
                retrieval_method="graph",
                metadata={"relationship_type": "causes"}
            )
        ]

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(graph_results, "What causes TRx increase?", top_k=5)

        assert len(reranked) == 1
        assert "causes" in reranked[0].metadata.get("relationship_type", "")


# =============================================================================
# PERFORMANCE SLA TESTS
# =============================================================================

class TestPerformanceSLA:
    """Tests for performance SLA compliance."""

    def test_reranker_latency_under_100ms(self):
        """Test that reranker completes within 100ms for 10 documents."""
        from src.rag.reranker import CrossEncoderReranker

        docs = [
            create_retrieval_result(
                content=f"Document {i} content about pharmaceutical sales",
                source="test",
                source_id=f"id_{i}",
                score=0.9 - i * 0.05
            )
            for i in range(10)
        ]

        reranker = CrossEncoderReranker()

        start = time.time()
        reranker.rerank(docs, "test query", top_k=5)
        elapsed_ms = (time.time() - start) * 1000

        # P95 latency should be under 100ms
        assert elapsed_ms < 100, f"Reranker took {elapsed_ms:.2f}ms, exceeds 100ms SLA"

    def test_query_optimizer_latency_under_50ms(self):
        """Test that query optimization completes within 50ms."""
        from src.rag.query_optimizer import QueryOptimizer

        optimizer = QueryOptimizer()

        start = time.time()
        optimizer.expand("Test query about TRx and Kisqali market share")
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 50, f"Query optimizer took {elapsed_ms:.2f}ms, exceeds 50ms SLA"

    @pytest.mark.asyncio
    async def test_enricher_latency_with_mock_llm(self):
        """Test that enrichment completes quickly with mocked LLM."""
        from src.rag.insight_enricher import InsightEnricher
        from src.rag.models.insight_models import EnrichedInsight

        results = [
            create_retrieval_result(
                content=f"Content {i} about TRx metrics",
                source="test",
                source_id=f"id_{i}",
                score=0.9
            )
            for i in range(5)
        ]

        enricher = InsightEnricher()

        # Mock internal LLM calls
        with patch.object(enricher, "_generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Summary of insights..."
            with patch.object(enricher, "_parse_response") as mock_parse:
                mock_parse.return_value = EnrichedInsight(
                    summary="Test summary",
                    key_findings=[],
                    confidence=0.9
                )

                start = time.time()
                await enricher.enrich(results, "Test query", max_findings=5)
                elapsed_ms = (time.time() - start) * 1000

                # With mock LLM, should be fast
                assert elapsed_ms < 100, f"Enricher took {elapsed_ms:.2f}ms with mocked LLM"


# =============================================================================
# ERROR HANDLING INTEGRATION TESTS
# =============================================================================

class TestErrorHandlingIntegration:
    """Tests for error handling across integrated components."""

    def test_reranker_handles_single_result(self):
        """Test that reranker handles single result correctly."""
        from src.rag.reranker import CrossEncoderReranker

        single_result = [
            create_retrieval_result(
                content="Single document",
                source="test",
                source_id="id_001",
                score=0.9
            )
        ]

        reranker = CrossEncoderReranker()
        result = reranker.rerank(single_result, "test query", top_k=5)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_enricher_with_default_metadata(self):
        """Test that enricher handles results with default metadata."""
        from src.rag.insight_enricher import InsightEnricher
        from src.rag.models.insight_models import EnrichedInsight

        results = [
            create_retrieval_result(
                content="Content with minimal metadata",
                source="test",
                source_id="id_001",
                score=0.9,
                metadata={}  # Empty metadata
            )
        ]

        enricher = InsightEnricher()

        with patch.object(enricher, "_generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Summary..."
            with patch.object(enricher, "_parse_response") as mock_parse:
                mock_parse.return_value = EnrichedInsight(
                    summary="Test", key_findings=[], confidence=0.9
                )

                result = await enricher.enrich(results, "test query")
                assert result is not None

    def test_optimizer_handles_empty_query(self):
        """Test that optimizer handles empty query gracefully."""
        from src.rag.query_optimizer import QueryOptimizer

        optimizer = QueryOptimizer()

        result = optimizer.expand("")
        assert isinstance(result, str)


# =============================================================================
# COMPONENT INTEROPERABILITY TESTS
# =============================================================================

class TestComponentInteroperability:
    """Tests for component interoperability and type compatibility."""

    def test_retrieval_result_to_dict_conversion(self):
        """Test RetrievalResult can be converted to dict."""
        result = create_retrieval_result(
            content="Test content",
            source="agent_activities",
            source_id="act_001",
            score=0.9,
            metadata={"brand": "Kisqali"}
        )

        # Dataclass should have to_dict() method
        result_dict = result.to_dict()

        assert result_dict["content"] == "Test content"
        assert result_dict["source"] == RetrievalSource.VECTOR.value
        assert result_dict["metadata"]["brand"] == "Kisqali"
        assert result_dict["metadata"]["source_name"] == "agent_activities"

    def test_causal_rag_initialization_patterns(self):
        """Test various CausalRAG initialization patterns."""
        from src.rag.causal_rag import CausalRAG

        # No components
        rag1 = CausalRAG()
        assert rag1.vector_retriever is None

        # With vector retriever only
        mock_vector = MagicMock()
        rag2 = CausalRAG(vector_retriever=mock_vector)
        assert rag2.vector_retriever is not None

        # With reranker only
        mock_reranker = MagicMock()
        rag3 = CausalRAG(reranker=mock_reranker)
        assert rag3.reranker is not None


# =============================================================================
# MULTI-HOP RETRIEVAL INTEGRATION TESTS
# =============================================================================

class TestMultiHopRetrievalIntegration:
    """Tests for multi-hop retrieval integration with cognitive workflow."""

    def test_hop_results_accumulated_correctly(self):
        """Test that multi-hop results are accumulated correctly."""
        from src.rag.cognitive_rag_dspy import Evidence, MemoryType

        # Simulate hop 1 (episodic) results
        hop1_evidence = [
            Evidence(MemoryType.EPISODIC, 1, "First hop result", 0.9)
        ]

        # Simulate hop 2 (semantic) results
        hop2_evidence = [
            Evidence(MemoryType.SEMANTIC, 2, "Second hop result", 0.85)
        ]

        # Combine evidence from multiple hops
        all_evidence = hop1_evidence + hop2_evidence

        assert len(all_evidence) == 2
        assert all_evidence[0].hop_number == 1
        assert all_evidence[1].hop_number == 2
        assert all_evidence[0].source == MemoryType.EPISODIC
        assert all_evidence[1].source == MemoryType.SEMANTIC

    def test_evidence_deduplication(self):
        """Test that duplicate evidence is detected."""
        from src.rag.cognitive_rag_dspy import Evidence, MemoryType

        # Same content from different hops
        evidence1 = Evidence(MemoryType.EPISODIC, 1, "Duplicate content", 0.9)
        evidence2 = Evidence(MemoryType.SEMANTIC, 2, "Duplicate content", 0.85)
        evidence3 = Evidence(MemoryType.EPISODIC, 1, "Unique content", 0.8)

        all_evidence = [evidence1, evidence2, evidence3]

        # Deduplicate by content
        unique_contents = set(e.content for e in all_evidence)
        assert len(unique_contents) == 2  # "Duplicate content" and "Unique content"


# =============================================================================
# ENRICHED INSIGHT MODEL TESTS
# =============================================================================

class TestEnrichedInsightModel:
    """Tests for EnrichedInsight model integration."""

    def test_enriched_insight_creation(self):
        """Test EnrichedInsight model can be created."""
        from src.rag.models.insight_models import EnrichedInsight

        insight = EnrichedInsight(
            summary="TRx increased 15% in Q4 2024",
            key_findings=["Territory expansion", "HCP targeting improvement"],
            confidence=0.85
        )

        assert insight.summary == "TRx increased 15% in Q4 2024"
        assert len(insight.key_findings) == 2
        assert insight.confidence == 0.85

    def test_enriched_insight_with_supporting_evidence(self):
        """Test EnrichedInsight with supporting evidence."""
        from src.rag.models.insight_models import EnrichedInsight

        evidence = [
            create_retrieval_result("Evidence 1", score=0.9),
            create_retrieval_result("Evidence 2", score=0.85)
        ]

        insight = EnrichedInsight(
            summary="Summary",
            key_findings=["Finding 1"],
            supporting_evidence=evidence,
            confidence=0.9
        )

        assert len(insight.supporting_evidence) == 2
