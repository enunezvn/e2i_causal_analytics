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

Cognitive RAG (DSPy-enhanced):
- CognitiveRAGWorkflow: 4-phase cognitive cycle with DSPy optimization
- create_production_cognitive_workflow: Factory for production backends
- Memory Adapters: Bridge real backends to DSPy workflow

RAGAS Evaluation (Opik-integrated):
- RAGASEvaluator: RAGAS metrics evaluation with Opik tracing
- RAGEvaluationPipeline: Batch evaluation pipeline
- OpikEvaluationTracer: Centralized Opik tracing
"""

from src.rag.causal_rag import CausalRAG
from src.rag.chunk_processor import ChunkProcessor

# Cognitive RAG exports
from src.rag.cognitive_rag_dspy import (
    CognitiveRAGOptimizer,
    CognitiveState,
    create_dspy_cognitive_workflow,
    create_production_cognitive_workflow,
)
from src.rag.config import RAGConfig
from src.rag.entity_extractor import EntityExtractor

# Evaluation exports
from src.rag.evaluation import (
    EvaluationResult,
    EvaluationSample,
    RAGASEvaluator,
    RAGEvaluationPipeline,
)
from src.rag.health_monitor import HealthMonitor
from src.rag.insight_enricher import InsightEnricher

# Memory adapters for DSPy integration
from src.rag.memory_adapters import (
    EpisodicMemoryAdapter,
    ProceduralMemoryAdapter,
    SemanticMemoryAdapter,
    SignalCollectorAdapter,
    create_memory_adapters,
)

# Opik integration (optional, with graceful fallback)
try:
    from src.rag.opik_integration import (
        CombinedEvaluationResult,
        OpikEvaluationTracer,
        log_ragas_scores_to_opik,
        log_rubric_scores_to_opik,
    )
    _OPIK_EXPORTS = [
        "OpikEvaluationTracer",
        "CombinedEvaluationResult",
        "log_ragas_scores_to_opik",
        "log_rubric_scores_to_opik",
    ]
except ImportError:
    _OPIK_EXPORTS = []

from src.rag.query_optimizer import QueryOptimizer
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import HybridRetriever
from src.rag.search_logger import SearchLogger

# Types
from src.rag.types import (
    BackendHealth,
    BackendStatus,
    ExtractedEntities,
    GraphPath,
    RetrievalResult,
    RetrievalSource,
    SearchStats,
)

__all__ = [
    # Core RAG
    "CausalRAG",
    "HybridRetriever",
    "CrossEncoderReranker",
    "QueryOptimizer",
    "InsightEnricher",
    "ChunkProcessor",
    "EntityExtractor",
    "HealthMonitor",
    "SearchLogger",
    "RAGConfig",
    # Types
    "RetrievalResult",
    "RetrievalSource",
    "ExtractedEntities",
    "BackendStatus",
    "BackendHealth",
    "SearchStats",
    "GraphPath",
    # Cognitive RAG
    "CognitiveState",
    "create_dspy_cognitive_workflow",
    "create_production_cognitive_workflow",
    "CognitiveRAGOptimizer",
    # Memory Adapters
    "EpisodicMemoryAdapter",
    "SemanticMemoryAdapter",
    "ProceduralMemoryAdapter",
    "SignalCollectorAdapter",
    "create_memory_adapters",
    # RAGAS Evaluation
    "RAGASEvaluator",
    "RAGEvaluationPipeline",
    "EvaluationResult",
    "EvaluationSample",
] + _OPIK_EXPORTS
