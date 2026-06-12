"""#883 read-side deferral: gap_analyzer readers vs the search RPC's missing raw_content.

PR #884 fixed gap_analyzer's episodic WRITE (migration 071 + outcome remap) and
documented — but deferred — the read-side gap: ``search_episodic_memory``
(database/memory/035, re-verified live 2026-06-12: the RETURNS TABLE has no
``raw_content`` column) returns rows WITHOUT ``raw_content``, while
``_get_episodic_context`` post-filters on ``result.get("raw_content", {})``.
Net effect on main: the moment the production caller
(``gap_detector._get_memory_context``) passes brand/metrics/segments — which it
ALWAYS does — every just-written row is dropped and the episodic context is
permanently empty. ``get_historical_roi_data`` / ``get_opportunity_benchmarks``
returned rows but WITHOUT the ROI payload (raw_content) their declared purpose
("with ROI outcomes" / "calibrate ROI estimates") requires, and their
brand/metric/segment parameters only seeded the embedding text instead of
filtering.

Fix under test (option (a), the #886 cohort_constructor precedent): hydrate
``raw_content`` by memory_id via ONE batched PK select
(``episodic_memory.hydrate_raw_content`` — the live column holds JSON-string
scalars because the insert path json.dumps's it; verified live: 628/628 rows
``jsonb_typeof = 'string'``), move the brand filter SERVER-side onto the RPC's
existing ``filter_brand`` param (the write populates the brand column via
``e2i_refs``), and post-filter metrics/segments on the hydrated content with
over-fetch-then-trim. The migration-073 alternative (extend the RPC's return
shape) was rejected on live data: returning the JSON-string scalar would hand
every one of the ~20 RPC consumers a ``str`` whose ``.get`` post-filters would
AttributeError-and-swallow into ``[]`` — the migration fixes nobody without the
same per-consumer parse-back this helper centralizes.

RED on main @ 59b4067a (quoted in each test): the brand/metrics/segments call
returns ``[]`` for a row written seconds earlier; the ROI/benchmark readers
return rows lacking ``raw_content`` and ignore their declared filters.

Each test inserts uniquely-marked rows and deletes them afterwards
(non-polluting). Gated like the other faithful real-DB tests; run with the
shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_gap_analyzer_raw_content_reads_883c.py'
"""

import os
import uuid

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB memory-read test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]

_GAP_QUERY = "Find TRx opportunity gaps and ROI for remibrutinib by region segment"


async def _store_gap_row(session_id: str, marker: str) -> str | None:
    """Seed one real gap episodic row through the #884-fixed write path."""
    from src.agents.gap_analyzer.memory_hooks import GapAnalyzerMemoryHooks

    hooks = GapAnalyzerMemoryHooks()
    result = {
        "prioritized_opportunities": [{"opportunity_id": marker}],
        "total_addressable_value": 1_250_000,
        "quick_wins": [{"opportunity_id": marker}],
        "strategic_bets": [],
        "confidence": 0.82,
        "executive_summary": f"TRx gap concentrated in northeast region ({marker})",
        "key_insights": ["northeast underperforms benchmark by 12%"],
    }
    state = {
        "query": f"{_GAP_QUERY} ({marker})",
        "brand": "remibrutinib",
        "metrics": ["trx_rate"],
        "segments": ["region"],
        "status": "completed",
    }
    return await hooks.store_gap_analysis(
        session_id=session_id,
        result=result,
        state=state,
        region="northeast",
    )


def _cleanup_episodic(memory_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("memory_id", memory_id).execute()


@pytest.mark.asyncio
async def test_episodic_context_with_brand_metrics_segments_returns_hydrated_row():
    """RED on main: ``_get_episodic_context(query, brand=..., metrics=...,
    segments=...)`` returned ``[]`` for a row stored seconds earlier — the
    post-filter read ``raw_content`` off rows that never carry it (the RPC's
    TABLE shape has no such column), so EVERY row failed the brand check.
    This is the exact call shape of the production caller
    (gap_detector._get_memory_context), i.e. the prod episodic context was
    permanently empty. GREEN: the row comes back, raw_content hydrated to the
    real stored dict."""
    from src.agents.gap_analyzer.memory_hooks import GapAnalyzerMemoryHooks

    session_id = str(uuid.uuid4())
    marker = f"883c-gap-read-{uuid.uuid4()}"

    memory_id = await _store_gap_row(session_id, marker)
    assert memory_id, "store_gap_analysis failed — cannot exercise the filtered read"

    hooks = GapAnalyzerMemoryHooks()
    try:
        results = await hooks._get_episodic_context(
            query=f"{_GAP_QUERY} ({marker})",
            brand="remibrutinib",
            metrics=["trx_rate"],
            segments=["region"],
        )
        by_id = {str(r.get("memory_id")): r for r in results}
        assert str(memory_id) in by_id, (
            "_get_episodic_context(brand/metrics/segments) dropped the just-written "
            "row — the raw_content post-filter ran against rows the search RPC "
            "returns WITHOUT raw_content (#883 read-side deferral, PR #884 body)"
        )
        row = by_id[str(memory_id)]
        assert isinstance(row.get("raw_content"), dict), (
            "returned row must carry the HYDRATED raw_content dict (the jsonb "
            "column holds a JSON-string scalar; hydration must parse it back)"
        )
        assert row["raw_content"].get("brand") == "remibrutinib"
        assert row["raw_content"].get("total_addressable_value") == 1_250_000

        # The filters must be REAL in both directions: a non-matching brand
        # (server-side filter_brand) and non-overlapping metrics (client-side
        # on hydrated content) must exclude the row — hydration alone that
        # ignores the declared filters would be a fabricated pass.
        wrong_brand = await hooks._get_episodic_context(
            query=f"{_GAP_QUERY} ({marker})",
            brand="fabhalta",
        )
        assert str(memory_id) not in {str(r.get("memory_id")) for r in wrong_brand}, (
            "brand filter is not actually filtering (wrong brand still returned)"
        )
        wrong_metric = await hooks._get_episodic_context(
            query=f"{_GAP_QUERY} ({marker})",
            brand="remibrutinib",
            metrics=["nbrx_share"],
        )
        assert str(memory_id) not in {str(r.get("memory_id")) for r in wrong_metric}, (
            "metrics filter is not actually filtering (non-overlapping metric still returned)"
        )
    finally:
        _cleanup_episodic(memory_id)


@pytest.mark.asyncio
async def test_historical_roi_data_hydrates_payload_and_filters_metric():
    """RED on main: rows came back WITHOUT raw_content — useless for the
    declared purpose ("historical gap analyses with ROI outcomes" feeding DSPy
    training), and ``metric=`` only seeded the embedding text (a non-matching
    metric still returned the row). GREEN: hydrated ROI payload + the declared
    brand (server-side) / metric (hydrated-content) filters are real."""
    from src.agents.gap_analyzer.memory_hooks import GapAnalyzerMemoryHooks

    session_id = str(uuid.uuid4())
    marker = f"883c-gap-roi-{uuid.uuid4()}"

    memory_id = await _store_gap_row(session_id, marker)
    assert memory_id, "store_gap_analysis failed — cannot exercise get_historical_roi_data"

    hooks = GapAnalyzerMemoryHooks()
    try:
        rows = await hooks.get_historical_roi_data(brand="remibrutinib", metric="trx_rate")
        by_id = {str(r.get("memory_id")): r for r in rows}
        assert str(memory_id) in by_id, (
            "get_historical_roi_data did not return the just-written row for its "
            "matching brand+metric"
        )
        row = by_id[str(memory_id)]
        assert isinstance(row.get("raw_content"), dict), (
            "ROI reader must return the hydrated raw_content payload — without it "
            "there ARE no 'ROI outcomes' in the result (the RPC returns none)"
        )
        assert row["raw_content"].get("total_addressable_value") == 1_250_000

        missing_metric = await hooks.get_historical_roi_data(
            brand="remibrutinib", metric="nbrx_share"
        )
        assert str(memory_id) not in {str(r.get("memory_id")) for r in missing_metric}, (
            "declared metric filter is not actually filtering"
        )
    finally:
        _cleanup_episodic(memory_id)


@pytest.mark.asyncio
async def test_opportunity_benchmarks_hydrates_payload_and_filters():
    """RED on main: same two defects as the ROI reader — no raw_content (so no
    benchmark data to 'calibrate ROI estimates' with) and the required
    segment/metric params never filtered. GREEN: hydrated payload, real
    filters in both directions."""
    from src.agents.gap_analyzer.memory_hooks import GapAnalyzerMemoryHooks

    session_id = str(uuid.uuid4())
    marker = f"883c-gap-bench-{uuid.uuid4()}"

    memory_id = await _store_gap_row(session_id, marker)
    assert memory_id, "store_gap_analysis failed — cannot exercise get_opportunity_benchmarks"

    hooks = GapAnalyzerMemoryHooks()
    try:
        rows = await hooks.get_opportunity_benchmarks(segment="region", metric="trx_rate")
        by_id = {str(r.get("memory_id")): r for r in rows}
        assert str(memory_id) in by_id, (
            "get_opportunity_benchmarks did not return the just-written row for "
            "its matching segment+metric"
        )
        assert isinstance(by_id[str(memory_id)].get("raw_content"), dict), (
            "benchmark reader must return the hydrated raw_content payload"
        )

        wrong_segment = await hooks.get_opportunity_benchmarks(
            segment="specialty", metric="trx_rate"
        )
        assert str(memory_id) not in {str(r.get("memory_id")) for r in wrong_segment}, (
            "declared segment filter is not actually filtering"
        )
    finally:
        _cleanup_episodic(memory_id)


@pytest.mark.asyncio
@pytest.mark.timeout(300)  # 8 real-embedding inserts; measured well under 60s, x3+ headroom
async def test_content_filter_overfetch_window_survives_decoy_starvation():
    """codex R1 (MED): a FIXED small over-fetch (limit*3) starves the
    post-filter when high-similarity non-matching rows fill the fetched
    window — the matching row sits just below it and the reader returns []
    even though a match EXISTS. RED quoted against the limit*3 window:
    6 decoys whose embedded text is an exact copy of the query out-ranked
    the 1 matching row; ``_get_episodic_context(limit=1, metrics=...)``
    fetched 3 candidates (all decoys) -> []. GREEN: the bounded candidate
    window (``content_filter_fetch_limit``: >=50, capped 100) reaches past
    the decoys.

    Determinism: decoys embed the EXACT search query text (similarity ~1.0);
    the matching row's embedded text carries extra tokens so it ranks below
    every decoy.
    """
    from src.agents.gap_analyzer.memory_hooks import GapAnalyzerMemoryHooks

    session_id = str(uuid.uuid4())
    marker = f"883c-gap-starve-{uuid.uuid4().hex[:12]}"
    query = f"TRx opportunity window starvation probe ({marker})"

    from src.agents.gap_analyzer.memory_hooks import GapAnalyzerMemoryHooks as _H

    async def _seed(metrics: list, embed_text: str) -> str | None:
        hooks = _H()
        result = {
            "prioritized_opportunities": [],
            "total_addressable_value": 10_000,
            "quick_wins": [],
            "strategic_bets": [],
            "confidence": 0.5,
            "executive_summary": embed_text,
            "key_insights": [],
        }
        state = {
            "query": embed_text,
            "brand": "remibrutinib",
            "metrics": metrics,
            "segments": ["region"],
            "status": "completed",
        }
        return await hooks.store_gap_analysis(
            session_id=session_id, result=result, state=state, region="northeast"
        )

    seeded: list = []
    try:
        # 6 decoys: same brand, NON-matching metric, embedded text == query.
        for _ in range(6):
            mid = await _seed(metrics=["nbrx_share"], embed_text=query)
            assert mid, "decoy seed failed"
            seeded.append(mid)
        # 1 match: the wanted metric, embedded text ranks BELOW the decoys.
        match_id = await _seed(
            metrics=["trx_rate"],
            embed_text=f"{query} additional benchmark detail northeast specialty mix",
        )
        assert match_id, "matching seed failed"
        seeded.append(match_id)

        hooks = GapAnalyzerMemoryHooks()
        results = await hooks._get_episodic_context(
            query=query,
            brand="remibrutinib",
            metrics=["trx_rate"],
            limit=1,
        )
        assert str(match_id) in {str(r.get("memory_id")) for r in results}, (
            "the matching row exists but was starved out of the over-fetch "
            "window by higher-similarity non-matching rows (codex R1 MED: "
            "fixed limit*3 window)"
        )
    finally:
        for mid in seeded:
            _cleanup_episodic(mid)
