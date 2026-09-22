"""``_get_semantic_context`` must build the EntityExtractor off the event loop.

Lane 2 (2026-09-22): ``EntityVocabulary.from_default`` performs a bounded sync
RxNav round at first build. ``_get_semantic_context`` is ``async def`` and was
the first place in a cold explainer process to touch the lazy
``entity_extractor`` property, so that round ran on the loop thread.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agents.explainer.memory_hooks import ExplanationMemoryHooks


class _SemanticMemoryStub:
    def get_graph_stats(self):
        return {}

    def get_entity(self, *_args, **_kwargs):
        return None

    def get_relationships(self, *_args, **_kwargs):
        return []


@pytest.mark.asyncio
async def test_entity_extractor_is_constructed_off_the_loop_thread():
    seen: dict[str, int] = {}
    empty = SimpleNamespace(brands=[], kpis=[], agents=[], hcp_segments=[])

    def _recording_extractor():
        seen["constructed_on"] = threading.get_ident()
        return SimpleNamespace(extract=lambda _query: empty)

    hooks = ExplanationMemoryHooks()
    hooks._semantic_memory = _SemanticMemoryStub()
    loop_thread = threading.get_ident()
    with patch("src.rag.entity_extractor.EntityExtractor", _recording_extractor):
        context = await hooks._get_semantic_context("iptacopan TRx in the northeast")

    assert "constructed_on" in seen, "the extractor was never built"
    assert context["extraction_summary"]["brands_found"] == 0
    assert seen["constructed_on"] != loop_thread, (
        "EntityExtractor() ran on the event-loop thread; from_default's RxNav round "
        "would block every other request"
    )
    # The built instance is cached for the sync property too.
    assert hooks.entity_extractor is hooks._entity_extractor
