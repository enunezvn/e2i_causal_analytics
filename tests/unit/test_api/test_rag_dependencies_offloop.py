"""``get_rag_dependencies`` must build the EntityExtractor off the event loop.

Lane 2 (2026-09-22): ``EntityVocabulary.from_default`` performs a bounded sync
RxNav round at first build (up to ~6 calls x 2 s when RxNav is slow but under
its timeout). ``get_rag_dependencies`` is ``async def``; constructing the
extractor inline would stall every other request on the loop for that long.
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.dependencies import rag as rag_deps


@pytest.mark.asyncio
async def test_entity_extractor_is_constructed_off_the_main_thread(monkeypatch):
    seen: dict[str, bool] = {}

    def _recording_extractor():
        seen["on_main_thread"] = threading.current_thread() is threading.main_thread()
        return MagicMock(name="entity_extractor")

    monkeypatch.setattr(rag_deps, "_rag_deps", None)
    with (
        patch.object(rag_deps, "get_supabase", return_value=MagicMock(name="supabase")),
        patch.object(rag_deps, "get_falkordb", AsyncMock(return_value=MagicMock(name="falkordb"))),
        patch("src.rag.config.RAGConfig.from_env", return_value=MagicMock(name="rag_config")),
        patch("src.rag.config.EmbeddingConfig.from_env", return_value=MagicMock(name="emb_config")),
        patch("src.rag.embeddings.OpenAIEmbeddingClient", return_value=MagicMock(name="embedder")),
        patch("src.rag.hybrid_retriever.HybridRetriever", return_value=MagicMock(name="retriever")),
        patch("src.rag.entity_extractor.EntityExtractor", _recording_extractor),
    ):
        try:
            deps = await rag_deps.get_rag_dependencies()
        finally:
            rag_deps._rag_deps = None  # never leak the patched singleton

    assert deps["entity_extractor"] is not None
    assert seen["on_main_thread"] is False, (
        "EntityExtractor() ran on the event-loop thread; from_default's RxNav round "
        "would block every other request"
    )
