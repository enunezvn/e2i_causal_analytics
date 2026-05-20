"""RED-first tests for box 3: Redis pub/sub alert delivery latency.

Goal
----
Measure the wall-clock delta between when a sentinel handler publishes
an alert to the ``e2i:alerts`` Redis channel and when the SSE
consumer receives it. Embed ``publish_at: <epoch_ms>`` in the alert
payload at the publisher; capture ``received_at: <epoch_ms>`` at the
consumer (the ``AlertBridge`` in ``src.api.routes.staleness_alerts``);
compute the delta and emit to MLflow as
``e2i.sentinel.alert_delivery_latency_ms``.

Boundary
--------
Two distinct surfaces:

1. **Publisher contract** — ``src.tasks.sentinel_actions.publish_alert``
   MUST stamp every payload with ``publish_at`` (an integer epoch ms).
   Tests use a fake Redis to capture the published JSON and assert the
   field is present + monotonic.
2. **Consumer contract** — given an alert with ``publish_at``, the
   consumer's recorded ``compute_alert_latency`` helper MUST emit a
   ``e2i.sentinel.alert_delivery_latency_ms`` MLflow metric whose
   value is ``now_ms - publish_at`` (>= 0).

Both pieces compose end-to-end: publish + receive in the same test
yields a small positive latency band.

Optional-instrumentation guard: missing publish_at MUST cause the
helper to skip emission silently (don't crash the consumer's hot path).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from src.mlops import lifecycle_monitoring as lm

# ---------------------------------------------------------------------
# Box 3.a — publisher stamps payload with publish_at
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_alert_stamps_publish_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """``publish_alert(payload)`` MUST stamp ``payload['publish_at']``
    with an integer epoch-ms BEFORE serializing to JSON onto Redis."""
    captured: List[str] = []

    fake_redis = AsyncMock()

    async def _capture(channel: str, data: str) -> int:
        captured.append(data)
        return 1

    fake_redis.publish = _capture

    monkeypatch.setattr("src.memory.services.factories.get_redis_client", lambda: fake_redis)

    from src.tasks.sentinel_actions import publish_alert

    payload: Dict[str, Any] = {"type": "staleness_alert", "brands": ["kisqali"]}
    before_ms = int(time.time() * 1000)
    await publish_alert(payload)
    after_ms = int(time.time() * 1000)

    assert len(captured) == 1
    serialized = json.loads(captured[0])
    assert "publish_at" in serialized, (
        "publish_alert MUST stamp publish_at on the outgoing payload "
        "(needed by alert-delivery latency monitor; see box 3)"
    )
    pub_at = serialized["publish_at"]
    assert isinstance(pub_at, int)
    assert before_ms <= pub_at <= after_ms


# ---------------------------------------------------------------------
# Box 3.b — consumer computes delta + emits MLflow histogram
# ---------------------------------------------------------------------


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):
    """Capture MLflow metric emissions for the latency tests."""
    captured: list[tuple[str, float, dict]] = []

    def _emit(metric_name: str, value: float, tags: dict | None = None) -> None:
        captured.append((metric_name, float(value), dict(tags or {})))

    monkeypatch.setattr(lm, "_emit_mlflow_metric", _emit)
    return captured


def test_record_alert_latency_emits_mlflow_metric(fake_mlflow) -> None:
    """``record_alert_latency`` MUST compute ``now_ms - publish_at`` and
    emit ``e2i.sentinel.alert_delivery_latency_ms`` to MLflow."""
    publish_at = int(time.time() * 1000) - 50  # 50ms ago
    lm.record_alert_latency(payload={"publish_at": publish_at, "brands": ["kisqali"]})

    latencies = [m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms"]
    assert len(latencies) == 1
    name, value, tags = latencies[0]
    # Value should be approximately the publish-time-delta, with some slack
    # for the time elapsed during the test.
    assert value >= 50.0
    assert value < 5000.0  # generous upper bound for slow CI


def test_record_alert_latency_passes_brand_tag(fake_mlflow) -> None:
    """If payload carries brands, the latency metric MUST be brand-tagged
    so per-brand dashboards can chart divergence.

    For multi-brand alerts (brands=["all"] or list of two), tag the FIRST
    brand for v1; document this as a known limitation. Closing fully
    requires per-brand fanout which is V2 follow-up.
    """
    publish_at = int(time.time() * 1000) - 10
    lm.record_alert_latency(payload={"publish_at": publish_at, "brands": ["remibrutinib"]})

    latency = next(m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms")
    assert latency[2].get("brand") == "remibrutinib"


def test_record_alert_latency_skips_silently_when_publish_at_missing(
    fake_mlflow,
) -> None:
    """Missing ``publish_at`` MUST cause the helper to skip emission
    (don't crash the consumer's hot path). Backward-compatible with
    alerts published BEFORE the publish_at stamping landed.
    """
    lm.record_alert_latency(payload={"brands": ["kisqali"]})
    latencies = [m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms"]
    assert len(latencies) == 0


def test_record_alert_latency_skips_when_publish_at_non_numeric(fake_mlflow) -> None:
    """A malformed ``publish_at`` field (string, list, etc.) MUST cause
    the helper to skip rather than crash."""
    lm.record_alert_latency(payload={"publish_at": "not-a-number", "brands": ["kisqali"]})
    lm.record_alert_latency(payload={"publish_at": None, "brands": ["kisqali"]})
    lm.record_alert_latency(payload={"publish_at": [1, 2, 3], "brands": ["kisqali"]})
    latencies = [m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms"]
    assert len(latencies) == 0


def test_record_alert_latency_clamps_negative_to_zero(fake_mlflow) -> None:
    """Clock skew can produce ``publish_at > now``. Clamp delta to >= 0
    so the metric doesn't poison downstream dashboards with negative
    latencies. Document this as the clamp behaviour.
    """
    publish_at = int(time.time() * 1000) + 60_000  # 60s in future (skew)
    lm.record_alert_latency(payload={"publish_at": publish_at, "brands": ["kisqali"]})
    latency = next(m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms")
    assert latency[1] == 0.0


# ---------------------------------------------------------------------
# Box 3.c — end-to-end compose (publish → receive → emit)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_publish_to_record_latency(
    monkeypatch: pytest.MonkeyPatch, fake_mlflow
) -> None:
    """Compose publish + consume in one test to assert latency band is
    sane (smaller than 1 second when both run in the same process).

    Codex audit premise (d): latency metric uses publish_at→receive
    timestamp delta, not single-point measurement.
    """
    captured_payloads: List[str] = []

    fake_redis = AsyncMock()

    async def _capture(channel: str, data: str) -> int:
        captured_payloads.append(data)
        return 1

    fake_redis.publish = _capture

    monkeypatch.setattr("src.memory.services.factories.get_redis_client", lambda: fake_redis)

    from src.tasks.sentinel_actions import publish_alert

    # 1) Publish.
    await publish_alert({"type": "staleness_alert", "brands": ["kisqali"]})

    # 2) Simulate consumer receive after a brief delay.
    await asyncio.sleep(0.005)  # 5ms

    received = json.loads(captured_payloads[0])
    lm.record_alert_latency(payload=received)

    latency = next(m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms")
    # We slept 5ms, so latency >= 5ms; pad upper bound generously for CI.
    assert latency[1] >= 5.0
    assert latency[1] < 5000.0
    assert latency[2].get("brand") == "kisqali"
