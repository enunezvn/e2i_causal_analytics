"""RAG dependencies for FastAPI and agent nodes.

Provides the HybridRetriever, embedding service, and entity extractor
used by the orchestrator's RAG context node.
"""

import logging
from typing import Any, Dict, Optional

from src.api.dependencies.falkordb_client import get_falkordb
from src.api.dependencies.supabase_client import get_supabase

logger = logging.getLogger(__name__)

_rag_deps: Optional[Dict[str, Any]] = None


async def get_rag_dependencies() -> Dict[str, Any]:
    """Get or create RAG dependency instances.

    Returns a dict with:
        - retriever: HybridRetriever (or None if backends unavailable)
        - embedding_service: OpenAIEmbeddingClient (or None)
        - entity_extractor: EntityExtractor (or None)
    """
    global _rag_deps

    if _rag_deps is not None:
        return _rag_deps

    supabase_client = get_supabase()
    falkordb_client = await get_falkordb()

    if not supabase_client or not falkordb_client:
        missing = []
        if not supabase_client:
            missing.append("Supabase")
        if not falkordb_client:
            missing.append("FalkorDB")
        logger.warning(f"RAG backends unavailable ({', '.join(missing)}) - retriever disabled")
        _rag_deps = {"retriever": None, "embedding_service": None, "entity_extractor": None}
        return _rag_deps

    try:
        from src.rag.config import EmbeddingConfig, RAGConfig
        from src.rag.embeddings import OpenAIEmbeddingClient
        from src.rag.entity_extractor import EntityExtractor
        from src.rag.hybrid_retriever import HybridRetriever

        config = RAGConfig.from_env()
        embedding_service = OpenAIEmbeddingClient(EmbeddingConfig.from_env())
        entity_extractor = EntityExtractor()
        retriever = HybridRetriever(
            supabase_client=supabase_client,
            falkordb_client=falkordb_client,
            config=config,
            embedding_service=embedding_service,
        )

        _rag_deps = {
            "retriever": retriever,
            "embedding_service": embedding_service,
            "entity_extractor": entity_extractor,
        }
        logger.info("RAG dependencies initialized successfully")
        return _rag_deps

    except Exception as e:
        logger.error(f"Failed to initialize RAG dependencies: {e}")
        _rag_deps = {"retriever": None, "embedding_service": None, "entity_extractor": None}
        return _rag_deps
