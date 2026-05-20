"""RED-first tests for issue #404: cluster-wide alert_id dedup via Redis SETNX.

Goal
----
Replace the per-process LRU dedup at the alert-latency emission site
with a cluster-wide SETNX-with-TTL claim so that under multi-worker
uvicorn / multi-pod K8s, the SAME ``alert_id`` observed by N workers
produces ONE metric sample (not N).

Boundary
--------
Tests substitute the Redis client (one level deeper than the function
under test) and the MLflow emission boundary so the REAL exception-
swallowing logic inside ``record_alert_latency_cluster`` executes.
This mirrors the `[[feedback-test-must-exercise-real-catch-not-mock]]`
contract.

Contracts
---------
1. Same ``alert_id`` claimed via SETNX returns "winner" once → emit
   exactly ONE sample regardless of N workers.
2. Redis transport error (``RedisConnectionError`` /
   ``RedisTimeoutError`` / builtin ``ConnectionError`` /
   ``TimeoutError``) → fall back to per-process LRU + emit
   ``e2i.sentinel.dedup_redis_unavailable_total`` counter.
3. Missing ``alert_id`` (back-compat with pre-stamping alerts) →
   no dedup, every observation emits.
4. ``publish_at`` validation (missing / non-numeric / negative skew)
   matches the existing sync helper's behavior.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src.mlops import lifecycle_monitoring as lm


@pytest.fixture(autouse=True)
def reset_alert_latency_dedup() -> None:
    """Clear the per-process LRU before every test."""
    lm._reset_alert_latency_dedup()


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):
    """Capture MLflow metric emissions at the public boundary.

    Patches the mid-level shim ``_emit_mlflow_metric`` (one level
    above the executor dispatch) so the real exception-swallowing in
    the production helper still runs.
    """
    captured: list[tuple[str, float, dict]] = []

    def _emit(metric_name: str, value: float, tags: dict | None = None) -> None:
        captured.append((metric_name, float(value), dict(tags or {})))

    monkeypatch.setattr(lm, "_emit_mlflow_metric", _emit)
    return captured


# ---------------------------------------------------------------------
# Contract 1 — cluster-wide dedup via Redis SETNX
# ---------------------------------------------------------------------


async def test_record_alert_latency_cluster_emits_when_redis_claim_succeeds(
    fake_mlflow,
) -> None:
    """When SETNX returns truthy (key did not exist; this worker won the
    race), the latency metric MUST be emitted."""
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)

    async def _factory() -> Any:
        return fake_redis

    publish_at = int(time.time() * 1000) - 30
    await lm.record_alert_latency_cluster(
        payload={
            "publish_at": publish_at,
            "alert_id": "alert-cluster-winner",
            "brands": ["kisqali"],
        },
        redis_factory=_factory,
    )

    latencies = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS]
    assert len(latencies) == 1
    assert latencies[0][2].get("brand") == "kisqali"

    # SETNX MUST be called with nx=True + a positive TTL.
    fake_redis.set.assert_awaited_once()
    call = fake_redis.set.await_args
    assert call.kwargs.get("nx") is True
    ttl_sec = call.kwargs.get("ex")
    assert isinstance(ttl_sec, int) and ttl_sec > 0


async def test_record_alert_latency_cluster_skips_when_redis_claim_refused(
    fake_mlflow,
) -> None:
    """When SETNX returns falsy (key already existed; another worker
    won the race), this worker MUST skip emission — cluster-wide
    at-most-one sample per alert_id."""
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=None)

    async def _factory() -> Any:
        return fake_redis

    await lm.record_alert_latency_cluster(
        payload={
            "publish_at": int(time.time() * 1000) - 10,
            "alert_id": "alert-cluster-loser",
            "brands": ["kisqali"],
        },
        redis_factory=_factory,
    )

    latencies = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS]
    assert len(latencies) == 0, "loser of SETNX race MUST NOT emit latency sample"


async def test_record_alert_latency_cluster_multi_worker_emits_exactly_once(
    fake_mlflow,
) -> None:
    """Simulate N workers calling the helper for the same alert_id.
    A REAL Redis would return truthy to exactly ONE caller. With our
    fake configured to return True on the first call and None on
    subsequent calls, exactly ONE worker MUST emit."""
    call_count = 0

    async def _set(key: str, value: str, *, nx: bool, ex: int) -> Any:
        nonlocal call_count
        call_count += 1
        return True if call_count == 1 else None

    fake_redis = AsyncMock()
    fake_redis.set = _set

    async def _factory() -> Any:
        return fake_redis

    alert_id = "alert-multi-worker-race"
    payload = {
        "publish_at": int(time.time() * 1000) - 5,
        "alert_id": alert_id,
        "brands": ["kisqali"],
    }

    # Simulate 5 workers racing on the same alert_id.
    for _ in range(5):
        await lm.record_alert_latency_cluster(
            payload=dict(payload),
            redis_factory=_factory,
        )

    latencies = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS]
    assert len(latencies) == 1, (
        f"5 workers racing on the same alert_id MUST emit exactly 1 sample "
        f"(got {len(latencies)}); cluster-wide dedup is broken"
    )


async def test_record_alert_latency_cluster_uses_alert_id_in_key(
    fake_mlflow,
) -> None:
    """The Redis key MUST be namespaced + include the alert_id so distinct
    alerts get distinct keys (no false dedup across alerts)."""
    captured_keys: list[str] = []

    async def _set(key: str, value: str, *, nx: bool, ex: int) -> Any:
        captured_keys.append(key)
        return True

    fake_redis = AsyncMock()
    fake_redis.set = _set

    async def _factory() -> Any:
        return fake_redis

    payload_a = {
        "publish_at": 0,
        "alert_id": "alert-a",
        "brands": ["kisqali"],
    }
    payload_b = {
        "publish_at": 0,
        "alert_id": "alert-b",
        "brands": ["kisqali"],
    }
    await lm.record_alert_latency_cluster(payload=payload_a, redis_factory=_factory)
    await lm.record_alert_latency_cluster(payload=payload_b, redis_factory=_factory)

    assert len(captured_keys) == 2
    assert "alert-a" in captured_keys[0]
    assert "alert-b" in captured_keys[1]
    # Namespace prefix so we don't collide with unrelated keys.
    assert captured_keys[0].startswith(lm._ALERT_LATENCY_DEDUP_KEY_PREFIX)
    assert captured_keys[1].startswith(lm._ALERT_LATENCY_DEDUP_KEY_PREFIX)


# ---------------------------------------------------------------------
# Contract 2 — Redis-down fallback to per-process LRU
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_class",
    [
        RedisConnectionError,
        RedisTimeoutError,
        ConnectionError,  # builtin
        TimeoutError,  # builtin
    ],
)
async def test_record_alert_latency_cluster_falls_back_to_lru_on_redis_error(
    fake_mlflow, exc_class: type[BaseException]
) -> None:
    """When SETNX raises any transport-error class (REDIS-PY's
    ConnectionError/TimeoutError OR the builtin classes), the helper
    MUST fall back to per-process LRU dedup so the latency metric
    keeps flowing in degraded mode.

    This pins the catch-tuple correctness pinpointed in memory
    `feat_378_reanalysis_enqueue_close_20260519`: redis-py's
    ConnectionError/TimeoutError do NOT inherit from the builtin
    classes, so the catch tuple MUST include both."""

    async def _set(key: str, value: str, *, nx: bool, ex: int) -> Any:
        raise exc_class("simulated redis down")

    fake_redis = AsyncMock()
    fake_redis.set = _set

    async def _factory() -> Any:
        return fake_redis

    payload = {
        "publish_at": int(time.time() * 1000) - 10,
        "alert_id": f"alert-fallback-{exc_class.__name__}",
        "brands": ["kisqali"],
    }

    # First call falls through to LRU and emits.
    await lm.record_alert_latency_cluster(
        payload=dict(payload),
        redis_factory=_factory,
    )
    # Second call hits the LRU dedup (same alert_id) and does NOT emit.
    await lm.record_alert_latency_cluster(
        payload=dict(payload),
        redis_factory=_factory,
    )

    latencies = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS]
    assert len(latencies) == 1, (
        f"with Redis down ({exc_class.__name__}), LRU fallback MUST still "
        f"emit exactly 1 sample for duplicate alert_id observations"
    )

    # Counter MUST be emitted on Redis fallback so the dashboard surfaces
    # the degraded mode.
    counters = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DEDUP_REDIS_UNAVAILABLE]
    assert len(counters) >= 1, (
        f"Redis-down fallback MUST emit {lm.METRIC_ALERT_DEDUP_REDIS_UNAVAILABLE} "
        f"counter so operators can see the degraded-mode rate"
    )


async def test_record_alert_latency_cluster_falls_back_when_factory_raises(
    fake_mlflow,
) -> None:
    """If the Redis factory itself raises a transport error (e.g.,
    cached client is gone, reconnect fails), the helper MUST fall back
    to per-process LRU."""

    async def _factory_that_raises() -> Any:
        raise RedisConnectionError("redis factory simulated down")

    payload = {
        "publish_at": int(time.time() * 1000) - 10,
        "alert_id": "alert-factory-fail",
        "brands": ["kisqali"],
    }

    await lm.record_alert_latency_cluster(
        payload=dict(payload),
        redis_factory=_factory_that_raises,
    )

    latencies = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS]
    assert len(latencies) == 1, "factory-raise MUST fall through to LRU and emit once"


async def test_record_alert_latency_cluster_programming_error_propagates(
    fake_mlflow,
) -> None:
    """Programming errors (TypeError on bad payload shape) MUST propagate
    — they signal a bug, not a transport blip. The catch tuple is NARROW
    on transport classes only."""

    async def _set(key: str, value: str, *, nx: bool, ex: int) -> Any:
        raise TypeError("simulated programming error — bad value shape")

    fake_redis = AsyncMock()
    fake_redis.set = _set

    async def _factory() -> Any:
        return fake_redis

    payload = {
        "publish_at": int(time.time() * 1000) - 10,
        "alert_id": "alert-prog-error",
        "brands": ["kisqali"],
    }

    with pytest.raises(TypeError, match="programming error"):
        await lm.record_alert_latency_cluster(
            payload=dict(payload),
            redis_factory=_factory,
        )


# ---------------------------------------------------------------------
# Contract 3 — back-compat: missing alert_id → no dedup, always emit
# ---------------------------------------------------------------------


async def test_record_alert_latency_cluster_no_alert_id_emits_every_time(
    fake_mlflow,
) -> None:
    """Back-compat with payloads published BEFORE alert_id stamping
    landed: no ``alert_id`` field → no dedup, every observation emits.
    Redis MUST NOT be consulted at all in this path."""
    factory_called = False

    async def _factory() -> Any:
        nonlocal factory_called
        factory_called = True
        return AsyncMock()

    publish_at = int(time.time() * 1000) - 10
    await lm.record_alert_latency_cluster(
        payload={"publish_at": publish_at, "brands": ["kisqali"]},
        redis_factory=_factory,
    )
    await lm.record_alert_latency_cluster(
        payload={"publish_at": publish_at, "brands": ["kisqali"]},
        redis_factory=_factory,
    )
    await lm.record_alert_latency_cluster(
        payload={"publish_at": publish_at, "brands": ["kisqali"]},
        redis_factory=_factory,
    )

    latencies = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS]
    assert len(latencies) == 3
    assert factory_called is False, (
        "no alert_id → no dedup needed; Redis MUST NOT be consulted to avoid "
        "wasted round-trips on legacy payloads"
    )


# ---------------------------------------------------------------------
# Contract 4 — publish_at validation (parity with sync helper)
# ---------------------------------------------------------------------


async def test_record_alert_latency_cluster_skips_when_publish_at_missing(
    fake_mlflow,
) -> None:
    """Missing publish_at → skip emission (same as sync helper)."""
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)

    async def _factory() -> Any:
        return fake_redis

    await lm.record_alert_latency_cluster(
        payload={"alert_id": "x", "brands": ["kisqali"]},
        redis_factory=_factory,
    )
    latencies = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS]
    assert len(latencies) == 0
    # SETNX MUST NOT have been issued — short-circuit before Redis.
    fake_redis.set.assert_not_awaited()


async def test_record_alert_latency_cluster_skips_when_publish_at_non_numeric(
    fake_mlflow,
) -> None:
    """Non-numeric publish_at → skip emission."""
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)

    async def _factory() -> Any:
        return fake_redis

    for bad in ["not-a-number", None, [1, 2, 3], True]:
        await lm.record_alert_latency_cluster(
            payload={"publish_at": bad, "alert_id": "x", "brands": ["kisqali"]},
            redis_factory=_factory,
        )
    latencies = [m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS]
    assert len(latencies) == 0


async def test_record_alert_latency_cluster_clamps_negative_to_zero(
    fake_mlflow,
) -> None:
    """Clock-skew clamp: publish_at > now → delta = 0.0 (don't poison
    dashboards with negative latencies)."""
    fake_redis = AsyncMock()
    fake_redis.set = AsyncMock(return_value=True)

    async def _factory() -> Any:
        return fake_redis

    publish_at_future = int(time.time() * 1000) + 60_000
    await lm.record_alert_latency_cluster(
        payload={
            "publish_at": publish_at_future,
            "alert_id": "clock-skew",
            "brands": ["kisqali"],
        },
        redis_factory=_factory,
    )
    latency = next(m for m in fake_mlflow if m[0] == lm.METRIC_ALERT_DELIVERY_LATENCY_MS)
    assert latency[1] == 0.0


# ---------------------------------------------------------------------
# Contract bonus — TTL choice rationale pinned in code
# ---------------------------------------------------------------------


def test_dedup_ttl_pinned_for_alert_lifetime() -> None:
    """Pin the TTL so that the rationale (alert lifetime + slack) is
    documented in test and any change is a deliberate decision."""
    # TTL >= 30s so transient SSE consumer reconnects don't double-emit.
    # TTL <= 300s so an alert_id collision after 5min (functionally
    # impossible with uuid4 but still) doesn't permanently shadow.
    assert 30 <= lm._ALERT_LATENCY_DEDUP_TTL_SEC <= 300


def test_dedup_key_prefix_namespaced() -> None:
    """Pin the key prefix to a clear namespace so the dedup keys don't
    collide with any other Redis usage."""
    assert lm._ALERT_LATENCY_DEDUP_KEY_PREFIX.startswith("e2i:")
    assert "alert" in lm._ALERT_LATENCY_DEDUP_KEY_PREFIX


def test_dedup_unavailable_counter_metric_name() -> None:
    """Pin the metric name for the Redis-down counter so dashboards
    can rely on a stable identifier."""
    assert lm.METRIC_ALERT_DEDUP_REDIS_UNAVAILABLE == ("e2i.sentinel.dedup_redis_unavailable_total")
