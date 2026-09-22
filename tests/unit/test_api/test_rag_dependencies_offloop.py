"""``get_rag_dependencies`` must build the EntityExtractor off the event loop.

Lane 2 (2026-09-22): ``EntityVocabulary.from_default`` performs a bounded sync
RxNav round at first build (up to 9 calls x 2 s when RxNav is slow but under
its timeout: two rxcui stages when the exact one misses, plus one related.json
call, per brand). ``get_rag_dependencies`` is ``async def``; constructing the
extractor inline would stall every other request on the loop for that long.
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.dependencies import rag as rag_deps


@pytest.mark.asyncio
async def test_entity_extractor_is_constructed_off_the_main_thread(monkeypatch):
    seen: dict[str, int] = {}

    def _recording_extractor():
        seen["constructed_on"] = threading.get_ident()
        return MagicMock(name="entity_extractor")

    # The loop's own thread, captured here rather than assumed to be the main one.
    loop_thread = threading.get_ident()

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
    assert seen["constructed_on"] != loop_thread, (
        "EntityExtractor() ran on the event-loop thread; from_default's RxNav round "
        "would block every other request"
    )


@pytest.mark.asyncio
async def test_a_burst_of_first_requests_builds_the_dependencies_once(monkeypatch):
    import asyncio

    factory_calls = {"n": 0}

    def _counting_extractor():
        factory_calls["n"] += 1
        return MagicMock(name="entity_extractor")

    monkeypatch.setattr(rag_deps, "_rag_deps", None)
    with (
        patch.object(rag_deps, "get_supabase", return_value=MagicMock(name="supabase")),
        patch.object(rag_deps, "get_falkordb", AsyncMock(return_value=MagicMock(name="falkordb"))),
        patch("src.rag.config.RAGConfig.from_env", return_value=MagicMock(name="rag_config")),
        patch("src.rag.config.EmbeddingConfig.from_env", return_value=MagicMock(name="emb_config")),
        patch("src.rag.embeddings.OpenAIEmbeddingClient", return_value=MagicMock(name="embedder")),
        patch("src.rag.hybrid_retriever.HybridRetriever", return_value=MagicMock(name="retriever")),
        patch("src.rag.entity_extractor.EntityExtractor", _counting_extractor),
    ):
        try:
            first, second = await asyncio.gather(
                rag_deps.get_rag_dependencies(), rag_deps.get_rag_dependencies()
            )
        finally:
            rag_deps._rag_deps = None

    assert factory_calls["n"] == 1, "two concurrent first requests must build once"
    assert first is second


def _burst_patches(counter):
    def _counting_extractor():
        counter["n"] += 1
        return MagicMock(name="entity_extractor")

    return (
        patch.object(rag_deps, "get_supabase", return_value=MagicMock(name="supabase")),
        patch.object(rag_deps, "get_falkordb", AsyncMock(return_value=MagicMock(name="falkordb"))),
        patch("src.rag.config.RAGConfig.from_env", return_value=MagicMock(name="rag_config")),
        patch("src.rag.config.EmbeddingConfig.from_env", return_value=MagicMock(name="emb_config")),
        patch("src.rag.embeddings.OpenAIEmbeddingClient", return_value=MagicMock(name="embedder")),
        patch("src.rag.hybrid_retriever.HybridRetriever", return_value=MagicMock(name="retriever")),
        patch("src.rag.entity_extractor.EntityExtractor", _counting_extractor),
    )


def test_the_single_flight_lock_survives_a_new_event_loop(monkeypatch):
    # Codex r2 LOW: a module-level asyncio.Lock binds to the loop that first
    # contends it; after the singleton is reset, a cold burst on a NEW loop
    # (function-scoped test loops, one per test) raised "bound to a different
    # event loop". Two bursts on two loops in one process must both build once.
    import asyncio
    from contextlib import ExitStack

    monkeypatch.setattr(rag_deps, "_rag_deps", None)

    async def _burst():
        return await asyncio.gather(
            rag_deps.get_rag_dependencies(), rag_deps.get_rag_dependencies()
        )

    for _loop_no in (1, 2):
        counter = {"n": 0}
        with ExitStack() as stack:
            for p in _burst_patches(counter):
                stack.enter_context(p)
            try:
                first, second = asyncio.run(_burst())
            finally:
                rag_deps._rag_deps = None
        assert counter["n"] == 1
        assert first is second


def test_two_concurrent_loops_share_one_cold_build(monkeypatch):
    # Codex r3 MED: the process runs more than one event loop (asyncio.run inside
    # threadpool tools), so single-flight must hold ACROSS loops, not only within
    # one. Loop A's leader is held mid-build while loop B goes cold: B must wait
    # for A's build and get the same object, not start a second one.
    import asyncio
    import threading
    from contextlib import ExitStack

    monkeypatch.setattr(rag_deps, "_rag_deps", None)
    counter = {"n": 0}
    release = threading.Event()
    leader_inside = threading.Event()

    async def _slow_falkordb():
        leader_inside.set()
        await asyncio.to_thread(release.wait, 5.0)
        return MagicMock(name="falkordb")

    results: dict[str, object] = {}

    def _run(tag):
        results[tag] = asyncio.run(rag_deps.get_rag_dependencies())

    with ExitStack() as stack:
        for p in _burst_patches(counter):
            stack.enter_context(p)
        stack.enter_context(patch.object(rag_deps, "get_falkordb", _slow_falkordb))
        a = threading.Thread(target=_run, args=("A",))
        b = threading.Thread(target=_run, args=("B",))
        try:
            a.start()
            assert leader_inside.wait(5.0), "loop A never entered the build"
            b.start()
            b.join(0.3)  # B either waits on A's build (correct) or starts its own (defect)
            release.set()
            a.join(5.0)
            b.join(5.0)
        finally:
            rag_deps._rag_deps = None

    assert not a.is_alive() and not b.is_alive()
    assert counter["n"] == 1, f"{counter['n']} cold builds across two concurrent loops"
    assert results["A"] is results["B"]


@pytest.mark.asyncio
async def test_a_cancelled_leader_hands_the_flight_to_a_waiter(monkeypatch):
    # The leader's request is cancelled mid-build (client disconnect): the waiter
    # must not inherit that cancellation; it retries, leads, and builds once.
    import asyncio

    monkeypatch.setattr(rag_deps, "_rag_deps", None)
    counter = {"n": 0}
    leader_inside = asyncio.Event()
    hold = asyncio.Event()

    async def _held_falkordb():
        if not leader_inside.is_set():
            leader_inside.set()
            await hold.wait()  # the leader parks here and is cancelled
        return MagicMock(name="falkordb")

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in _burst_patches(counter):
            stack.enter_context(p)
        stack.enter_context(patch.object(rag_deps, "get_falkordb", _held_falkordb))
        try:
            leader = asyncio.create_task(rag_deps.get_rag_dependencies())
            await leader_inside.wait()
            waiter = asyncio.create_task(rag_deps.get_rag_dependencies())
            await asyncio.sleep(0)  # let the waiter park on the flight
            leader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await leader
            deps = await asyncio.wait_for(waiter, 5.0)
        finally:
            rag_deps._rag_deps = None

    assert deps["entity_extractor"] is not None
    assert counter["n"] == 1
    assert rag_deps._build_future is None, "the flight slot must be clear when idle"
