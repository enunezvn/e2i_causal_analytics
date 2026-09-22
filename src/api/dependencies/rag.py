"""RAG dependencies for FastAPI and agent nodes.

Provides the HybridRetriever, embedding service, and entity extractor
used by the orchestrator's RAG context node.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from src.api.dependencies.falkordb_client import get_falkordb
from src.api.dependencies.supabase_client import get_supabase

logger = logging.getLogger(__name__)

_rag_deps: Optional[Dict[str, Any]] = None
# Single-flight for the first build: a burst of first requests must not each
# construct the retriever, embedder and extractor (lane 2: the extractor's first
# build runs a bounded RxNav round). An asyncio.Lock binds to the loop that first
# contends it, so the lock is kept per running loop: a cold burst on a new loop
# (a fresh test loop after the singleton is reset) gets a fresh lock instead of
# "is bound to a different event loop".
_rag_deps_lock: Optional[asyncio.Lock] = None
_rag_deps_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _single_flight_lock() -> asyncio.Lock:
    global _rag_deps_lock, _rag_deps_lock_loop
    loop = asyncio.get_running_loop()
    if _rag_deps_lock is None or _rag_deps_lock_loop is not loop:
        # No await between the check and the assignment: race-free within one loop.
        _rag_deps_lock = asyncio.Lock()
        _rag_deps_lock_loop = loop
    return _rag_deps_lock


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

    async with _single_flight_lock():
        if _rag_deps is not None:  # built by the request that held the lock first
            return _rag_deps
        _rag_deps = await _build_rag_dependencies()
        return _rag_deps


async def _build_rag_dependencies() -> Dict[str, Any]:
    supabase_client = get_supabase()
    falkordb_client = await get_falkordb()

    if not supabase_client or not falkordb_client:
        missing = []
        if not supabase_client:
            missing.append("Supabase")
        if not falkordb_client:
            missing.append("FalkorDB")
        logger.warning(f"RAG backends unavailable ({', '.join(missing)}) - retriever disabled")
        return {"retriever": None, "embedding_service": None, "entity_extractor": None}

    try:
        from src.rag.config import EmbeddingConfig, RAGConfig
        from src.rag.embeddings import OpenAIEmbeddingClient
        from src.rag.entity_extractor import EntityExtractor
        from src.rag.hybrid_retriever import HybridRetriever

        config = RAGConfig.from_env()
        embedding_service = OpenAIEmbeddingClient(EmbeddingConfig.from_env())
        # Lane 2 (2026-09-22): from_default performs a bounded sync RxNav round at
        # first build; keep it off the event loop so other requests are not stalled.
        entity_extractor = await asyncio.to_thread(EntityExtractor)
        retriever = HybridRetriever(
            supabase_client=supabase_client,
            falkordb_client=falkordb_client,
            config=config,
            embedding_service=embedding_service,
        )

        deps = {
            "retriever": retriever,
            "embedding_service": embedding_service,
            "entity_extractor": entity_extractor,
        }
        logger.info("RAG dependencies initialized successfully")
        return deps

    except Exception as e:
        logger.error(f"Failed to initialize RAG dependencies: {e}")
        return {"retriever": None, "embedding_service": None, "entity_extractor": None}
