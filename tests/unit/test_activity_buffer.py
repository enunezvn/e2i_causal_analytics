"""Unit tests for the bounded activity aggregation buffer (no DB, no mocks —
pure data structure). OOM-safety is the point: the buffer must CAP distinct
buckets and drop (counting drops) rather than grow."""

from src.api.middleware.activity_tracking import ActivityBuffer

UID = "11111111-1111-1111-1111-111111111111"


def test_record_aggregates_same_bucket():
    buf = ActivityBuffer(max_buckets=10, flush_interval_s=9999, flush_threshold=9999)
    for _ in range(5):
        buf.record(UID, "a@x.com", "causal", "GET", "2026-07-11T15:00:00+00:00")
    rows = buf.drain()
    assert len(rows) == 1
    assert rows[0]["request_count"] == 5
    assert rows[0]["endpoint_group"] == "causal"
    assert rows[0]["user_id"] == UID


def test_distinct_buckets_are_separate_rows():
    buf = ActivityBuffer(max_buckets=10, flush_interval_s=9999, flush_threshold=9999)
    buf.record(UID, "a@x.com", "causal", "GET", "2026-07-11T15:00:00+00:00")
    buf.record(UID, "a@x.com", "kpis", "GET", "2026-07-11T15:00:00+00:00")
    buf.record(UID, "a@x.com", "causal", "POST", "2026-07-11T15:00:00+00:00")
    assert len(buf.drain()) == 3


def test_cap_drops_new_buckets_but_still_counts_existing():
    buf = ActivityBuffer(max_buckets=2, flush_interval_s=9999, flush_threshold=9999)
    buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:00+00:00")
    buf.record(UID, "a@x.com", "g2", "GET", "2026-07-11T15:00:00+00:00")
    # new bucket beyond cap -> dropped
    buf.record(UID, "a@x.com", "g3", "GET", "2026-07-11T15:00:00+00:00")
    # existing bucket still increments at cap
    buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:00+00:00")
    assert buf.dropped == 1
    rows = {r["endpoint_group"]: r["request_count"] for r in buf.drain()}
    assert rows == {"g1": 2, "g2": 1}


def test_flush_threshold_and_drain_resets():
    buf = ActivityBuffer(max_buckets=100, flush_interval_s=9999, flush_threshold=2)
    assert buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:00+00:00") is False
    assert buf.record(UID, "a@x.com", "g2", "GET", "2026-07-11T15:00:00+00:00") is True
    assert buf.drain() != []
    assert buf.drain() == []  # drained


def test_time_based_flush(monkeypatch):
    import src.api.middleware.activity_tracking as at

    t = {"now": 1000.0}
    monkeypatch.setattr(at.time, "monotonic", lambda: t["now"])
    buf = ActivityBuffer(max_buckets=100, flush_interval_s=30.0, flush_threshold=9999)
    assert buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:00+00:00") is False
    t["now"] = 1031.0
    assert buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:01+00:00") is True


def test_schedule_flush_keeps_strong_reference_until_done():
    """asyncio.create_task results are GC-eligible if unreferenced — a dropped
    flush task would silently lose activity rows (fail-open hides it). The
    scheduler must hold each in-flight task and release it on completion."""
    import asyncio

    from src.api.middleware import activity_tracking as at

    async def _run():
        task = at.schedule_flush([])  # empty rows: flush_rows returns fast
        assert task in at._INFLIGHT_FLUSHES
        await task
        await asyncio.sleep(0)  # let the done-callback run
        assert task not in at._INFLIGHT_FLUSHES

    asyncio.run(_run())
