"""F5: cognitive_rag_retrieve must score evidence rows CONCURRENTLY (asyncio.gather),
not in a serial await loop, and must keep the identical evidence set/order."""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.api.routes.chatbot_dspy import cognitive_rag_retrieve
from src.rag.models.retrieval_models import RetrievalResult

pytestmark = pytest.mark.asyncio  # asyncio_mode="auto" already applies; module marker is explicit


def _make_results(n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content=f"evidence row {i} about kisqali trx northeast",
            source="episodic_memories",
            source_id=f"row-{i}",
            score=0.5,  # RetrievalResult.score has ge=0,le=1 — 0.5 is valid
            retrieval_method="dense",
        )
        for i in range(n)
    ]


async def test_evidence_scoring_runs_concurrently_and_preserves_outcome():
    n = 5
    results = _make_results(n)

    # Concurrency recorder: every scorer call logs when it STARTS and when it
    # RETURNS. If the loop is serial, call k cannot start until call k-1 returns,
    # so max(active concurrent) == 1. With asyncio.gather all n are active at once.
    active = 0
    max_active = 0
    started_order: list[str] = []
    lock = asyncio.Lock()

    async def fake_score(investigation_goal, evidence_item, source_memory="episodic"):
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
            started_order.append(evidence_item[:30])
        # all-N-dispatched-before-any-resolve: hold every coroutine until the
        # whole batch has entered. If dispatch were serial this gather() deadlocks
        # the count and the timeout below trips -> red, proving non-concurrency.
        for _ in range(200):
            if max_active >= n:
                break
            await asyncio.sleep(0.001)
        # deterministic, content-derived score so the kept-set is reproducible:
        # 0.4 >= 0.3 -> every row kept.
        score = 0.4
        async with lock:
            active -= 1
        return (score, f"insight for {evidence_item[:20]}", False)

    # NOTE: accept entities=/**kwargs so this stub stays valid whether or not the
    # Phase-2 graph-wiring shard (which adds `entities=graph_entities or None` to
    # the real hybrid_search call) has already landed — otherwise, post-Phase-2,
    # cognitive_rag_retrieve would call this stub with an unexpected entities= kwarg.
    async def fake_hybrid_search(query, k, entities=None, kpi_name=None, filters=None, **kwargs):
        return results

    with (
        patch("src.api.routes.chatbot_dspy.rewrite_query_dspy") as mock_rewrite,
        patch("src.rag.retriever.hybrid_search", side_effect=fake_hybrid_search),
        patch("src.api.routes.chatbot_dspy.score_evidence_dspy", side_effect=fake_score),
    ):
        mock_rewrite.return_value = ("rewritten q", ["kisqali", "trx"], ["Kisqali"], "dspy")

        t0 = time.monotonic()
        out = await asyncio.wait_for(
            cognitive_rag_retrieve(
                query="TRx trend for Kisqali in Northeast",
                brand_context="",
                k=n,
                collect_signal=False,
            ),
            timeout=5.0,
        )
        elapsed = time.monotonic() - t0

    # (1) CONCURRENCY: all n scorers were active simultaneously (serial loop => 1).
    assert max_active == n, f"expected {n} concurrent scorers, saw max {max_active} (serial loop?)"
    # (2) all n dispatched before any resolved (gather, not pipelined awaits)
    assert len(started_order) == n
    # (3) OUTCOME UNCHANGED: every row scored 0.4 >= 0.3 -> all kept, in input order
    assert [e["source_id"] for e in out.evidence] == [f"row-{i}" for i in range(n)]
    assert all(e["relevance_score"] == 0.4 for e in out.evidence)
    assert out.avg_relevance_score == pytest.approx(0.4)
    # (4) ADVISORY wall-clock sanity bound (not the discriminator — the rigorous
    # concurrency check is max_active == n above; this just guards against a gross
    # regression and is set well below the 5.0s wait_for timeout to avoid CI flake).
    assert elapsed < 4.0
