"""RAG dependencies for FastAPI and agent nodes.

Provides the HybridRetriever, embedding service, and entity extractor
used by the orchestrator's RAG context node.
"""

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Dict, Optional

from src.api.dependencies.falkordb_client import get_falkordb
from src.api.dependencies.supabase_client import get_supabase

logger = logging.getLogger(__name__)

_rag_deps: Optional[Dict[str, Any]] = None
# Single-flight for the first build: a burst of first requests must not each
# construct the retriever, embedder and extractor (lane 2: the extractor's first
# build runs a bounded RxNav round). The process runs more than one event loop
# (asyncio.run inside threadpool tools), so the flight is keyed on a thread lock
# and a concurrent Future that ANY loop can await; an asyncio.Lock would bind to
# one loop and let a second loop start its own build.
_build_guard = threading.Lock()
_build_future: Optional[concurrent.futures.Future] = None


async def get_rag_dependencies() -> Dict[str, Any]:
    """Get or create RAG dependency instances.

    Returns a dict with:
        - retriever: HybridRetriever (or None if backends unavailable)
        - embedding_service: OpenAIEmbeddingClient (or None)
        - entity_extractor: EntityExtractor (or None)
    """
    global _rag_deps, _build_future

    while True:
        if _rag_deps is not None:
            return _rag_deps

        with _build_guard:
            if _rag_deps is not None:
                return _rag_deps
            leader = _build_future is None
            if leader:
                _build_future = concurrent.futures.Future()
            flight = _build_future

        if leader:
            try:
                deps = await _build_rag_dependencies()
            except BaseException:
                # Cancelled or crashed leader: release the waiters to retry, and
                # let the next caller lead. Never hand them our CancelledError.
                with _build_guard:
                    _build_future = None
                flight.set_exception(RuntimeError("RAG dependency build aborted"))
                raise
            with _build_guard:
                _rag_deps = deps
                _build_future = None
            flight.set_result(deps)
            return deps

        try:
            # shield: a cancelled waiter must not cancel the shared flight.
            return await asyncio.shield(asyncio.wrap_future(flight))
        except asyncio.CancelledError:
            raise
        except RuntimeError:
            continue  # the leader aborted; go round again


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
