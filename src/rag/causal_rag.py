"""
CausalRAG - Main orchestrator for graph-enhanced retrieval.

This is the primary entry point for RAG operations in E2I Causal Analytics.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from src.rag.models.retrieval_models import RetrievalResult, RetrievalContext
from src.rag.models.insight_models import EnrichedInsight


class CausalRAG:
    """
    Graph-enhanced retrieval for causal insights.

    Retrieval sources:
    1. Vector store (semantic similarity)
    2. Causal graph (path traversal)
    3. Structured queries (SQL for KPIs)

    CRITICAL: Only indexes operational data.
    NEVER indexes: clinical trials, medical literature, regulatory docs.
    """

    def __init__(
        self,
        vector_retriever=None,
        graph_retriever=None,
        kpi_retriever=None,
        reranker=None,
    ):
        """
        Initialize CausalRAG with retrieval components.

        Args:
            vector_retriever: Dense/sparse vector retriever
            graph_retriever: Causal graph traversal
            kpi_retriever: Structured KPI queries
            reranker: Cross-encoder reranker
        """
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.kpi_retriever = kpi_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query,  # ParsedQuery from NLP layer
        top_k: int = 10,
        retrieval_config: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid retrieval for a parsed query.

        Args:
            query: ParsedQuery from NLP layer
            top_k: Maximum results to return
            retrieval_config: Override default retrieval settings

        Returns:
            List of RetrievalResult ordered by relevance
        """
        all_results = []

        # 1. Semantic retrieval from vector store
        if self.vector_retriever:
            vector_results = self.vector_retriever.search(
                query.text if hasattr(query, 'text') else str(query),
                k=top_k
            )
            all_results.extend(vector_results)

        # 2. Graph-based retrieval for causal queries
        if self.graph_retriever and hasattr(query, 'intent'):
            # Only use graph retrieval for causal intent queries
            if query.intent and query.intent.value == "causal":
                graph_results = self.graph_retriever.traverse(
                    entities=query.entities if hasattr(query, 'entities') else [],
                    relationship="causal_path"
                )
                all_results.extend(graph_results)

        # 3. Structured retrieval for KPI queries
        if self.kpi_retriever and hasattr(query, 'entities'):
            if hasattr(query.entities, 'kpis') and query.entities.kpis:
                kpi_results = self.kpi_retriever.query(query.entities.kpis)
                all_results.extend(kpi_results)

        # 4. Rerank and deduplicate
        if self.reranker and all_results:
            return self.reranker.rerank(all_results, query, top_k=top_k)

        return all_results[:top_k]

    async def retrieve_async(
        self,
        query,
        top_k: int = 10,
        retrieval_config: Optional[Dict[str, Any]] = None,
    ) -> RetrievalContext:
        """
        Async version of retrieve with full context.

        Returns:
            RetrievalContext with results and metadata
        """
        import time
        start_time = time.time()

        results = self.retrieve(query, top_k, retrieval_config)

        elapsed_ms = (time.time() - start_time) * 1000

        return RetrievalContext(
            query=query,
            results=results,
            total_retrieved=len(results),
            retrieval_time_ms=elapsed_ms,
        )
