"""
CausalRAG - Retrieval-Augmented Generation for E2I Causal Analytics.

This module provides graph-enhanced retrieval for causal insights.
It is LIMITED to operational data only - NO medical/clinical content.

Retrieval sources:
1. Vector store (semantic similarity via pgvector)
2. Causal graph (NetworkX path traversal)
3. Structured queries (SQL for KPIs)

Key Components:
- CausalRAG: Main orchestrator
- HybridRetriever: Dense + Sparse + Graph retrieval
- CrossEncoderReranker: Result reranking
- QueryOptimizer: Domain-aware query expansion
- InsightEnricher: LLM-based synthesis
- ChunkProcessor: Agent output chunking
"""

from src.rag.causal_rag import CausalRAG
from src.rag.retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.query_optimizer import QueryOptimizer
from src.rag.insight_enricher import InsightEnricher
from src.rag.chunk_processor import ChunkProcessor

__all__ = [
    "CausalRAG",
    "HybridRetriever",
    "CrossEncoderReranker",
    "QueryOptimizer",
    "InsightEnricher",
    "ChunkProcessor",
]
