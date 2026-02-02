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

import logging

_logger = logging.getLogger(__name__)

# =============================================================================
# Core Types (always available - minimal dependencies)
# =============================================================================
from src.rag.config import RAGConfig

# =============================================================================
# Evaluation (minimal dependencies - needed for CI workflows)
# =============================================================================
from src.rag.evaluation import (
    EvaluationResult,
    EvaluationSample,
    RAGASEvaluator,
    RAGEvaluationPipeline,
)
from src.rag.types import (
    BackendHealth,
    BackendStatus,
    ExtractedEntities,
    GraphPath,
    RetrievalResult,
    RetrievalSource,
    SearchStats,
)

# =============================================================================
# Optional Components (may require additional dependencies)
# These are wrapped in try/except to allow minimal imports for CI workflows
# =============================================================================

# Core RAG components (require anthropic, networkx, etc.)
# Each import is wrapped separately so individual failures don't block others
_CORE_RAG_EXPORTS: list[str] = []

try:
    from src.rag.causal_rag import CausalRAG

    _CORE_RAG_EXPORTS.append("CausalRAG")
except ImportError as e:
    _logger.debug(f"CausalRAG not available: {e}")

try:
    from src.rag.chunk_processor import ChunkProcessor

    _CORE_RAG_EXPORTS.append("ChunkProcessor")
except ImportError as e:
    _logger.debug(f"ChunkProcessor not available: {e}")

try:
    from src.rag.entity_extractor import EntityExtractor

    _CORE_RAG_EXPORTS.append("EntityExtractor")
except ImportError as e:
    _logger.debug(f"EntityExtractor not available: {e}")

try:
    from src.rag.health_monitor import HealthMonitor

    _CORE_RAG_EXPORTS.append("HealthMonitor")
except ImportError as e:
    _logger.debug(f"HealthMonitor not available: {e}")

try:
    from src.rag.insight_enricher import InsightEnricher

    _CORE_RAG_EXPORTS.append("InsightEnricher")
except ImportError as e:
    _logger.debug(f"InsightEnricher not available: {e}")

try:
    from src.rag.query_optimizer import QueryOptimizer

    _CORE_RAG_EXPORTS.append("QueryOptimizer")
except ImportError as e:
    _logger.debug(f"QueryOptimizer not available: {e}")

try:
    from src.rag.reranker import CrossEncoderReranker

    _CORE_RAG_EXPORTS.append("CrossEncoderReranker")
except ImportError as e:
    _logger.debug(f"CrossEncoderReranker not available: {e}")

try:
    from src.rag.retriever import HybridRetriever

    _CORE_RAG_EXPORTS.append("HybridRetriever")
except ImportError as e:
    _logger.debug(f"HybridRetriever not available: {e}")

try:
    from src.rag.search_logger import SearchLogger

    _CORE_RAG_EXPORTS.append("SearchLogger")
except ImportError as e:
    _logger.debug(f"SearchLogger not available: {e}")

# Cognitive RAG exports (require dspy, langchain, etc.)
_COGNITIVE_EXPORTS: list[str] = []
try:
    from src.rag.cognitive_rag_dspy import (
        CognitiveRAGOptimizer,
        CognitiveState,
        create_dspy_cognitive_workflow,
        create_production_cognitive_workflow,
    )

    _COGNITIVE_EXPORTS = [
        "CognitiveState",
        "create_dspy_cognitive_workflow",
        "create_production_cognitive_workflow",
        "CognitiveRAGOptimizer",
    ]
except ImportError as e:
    _logger.debug(f"Cognitive RAG components not available: {e}")

# Memory adapters for DSPy integration
_MEMORY_EXPORTS: list[str] = []
try:
    from src.rag.memory_adapters import (
        EpisodicMemoryAdapter,
        ProceduralMemoryAdapter,
        SemanticMemoryAdapter,
        SignalCollectorAdapter,
        create_memory_adapters,
    )

    _MEMORY_EXPORTS = [
        "EpisodicMemoryAdapter",
        "SemanticMemoryAdapter",
        "ProceduralMemoryAdapter",
        "SignalCollectorAdapter",
        "create_memory_adapters",
    ]
except ImportError as e:
    _logger.debug(f"Memory adapters not available: {e}")

# Opik integration (optional, with graceful fallback)
_OPIK_EXPORTS: list[str] = []
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
except ImportError as e:
    _logger.debug(f"Opik integration not available: {e}")

__all__ = (
    [
        # Types (always available)
        "RetrievalResult",
        "RetrievalSource",
        "ExtractedEntities",
        "BackendStatus",
        "BackendHealth",
        "SearchStats",
        "GraphPath",
        "RAGConfig",
        # RAGAS Evaluation (always available - needed for CI)
        "RAGASEvaluator",
        "RAGEvaluationPipeline",
        "EvaluationResult",
        "EvaluationSample",
    ]
    + _CORE_RAG_EXPORTS
    + _COGNITIVE_EXPORTS
    + _MEMORY_EXPORTS
    + _OPIK_EXPORTS
)
