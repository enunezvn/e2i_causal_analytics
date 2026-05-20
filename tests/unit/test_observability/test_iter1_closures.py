"""Tests for codex iter-0 findings closure (iter-1 deltas).

Closures (per codex iter-0 audit on PR #391 monitoring slice):

* **H1**: consumer-dependency-trace — covered by
  ``test_dashboard_manifest.py``.
* **H2**: Redis latency subscriber-weighted — closed via (a)
  publisher-side ``record_alert_published`` counter so dashboards can
  detect "no subscribers connected" as ``publish_count > 0 AND
  latency_samples == 0``, AND (b) consumer-side LRU dedup keyed on
  ``payload['alert_id']`` so multiple SSE subscribers receiving the
  same publish emit at-most-one sample.
* **M1**: brand namespacing leaks — closed via (a) ``_coerce_brand_tags``
  multi-brand → ``_multi_`` bucket (cross-brand → ``all``), and (b)
  per-brand fanout in ``Consolidator.run`` brand=None sweeps.
* **M2**: sync MLflow in async paths — closed via single-thread
  background ``ThreadPoolExecutor`` so the producer-side call is
  fire-and-forget.
* **M3**: cascade propagation depth off-by-one — closed via
  ``propagation_depth = max(0, depth - 1)`` in
  ``cascade_invalidate``.
"""

from __future__ import annotations

import pytest

from src.mlops import lifecycle_monitoring as lm

# ---------------------------------------------------------------------
# H2 — Redis pub/sub alert delivery dedup
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_alert_latency_dedup() -> None:
    """Clear the per-process LRU before every test so per-test
    alert_id reuse doesn't accidentally suppress emission."""
    lm._reset_alert_latency_dedup()


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):
    """Capture MLflow metric emissions."""
    captured: list[tuple[str, float, dict]] = []

    def _emit(metric_name: str, value: float, tags: dict | None = None) -> None:
        captured.append((metric_name, float(value), dict(tags or {})))

    monkeypatch.setattr(lm, "_emit_mlflow_metric", _emit)
    return captured


def test_record_alert_published_emits_publish_count(fake_mlflow) -> None:
    """``record_alert_published`` MUST emit one
    ``e2i.sentinel.alert_publish_count`` per call so dashboards can
    detect 'no subscribers connected' (publish>0, latency==0)."""
    lm.record_alert_published(payload={"brands": ["kisqali"]})
    pubs = [m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_publish_count"]
    assert len(pubs) == 1
    assert pubs[0][1] == 1.0
    assert pubs[0][2].get("brand") == "kisqali"


def test_record_alert_latency_dedups_on_alert_id(fake_mlflow) -> None:
    """When the same ``alert_id`` is recorded twice (e.g., two SSE
    subscribers receiving the same publish), the second call MUST
    skip emission — bounded at-most-one sample per (process,
    alert_id)."""
    alert_id = "alert-abc-123"
    payload = {
        "publish_at": 0,
        "alert_id": alert_id,
        "brands": ["kisqali"],
    }
    lm.record_alert_latency(payload=dict(payload))
    lm.record_alert_latency(payload=dict(payload))  # same alert_id

    latencies = [m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms"]
    assert len(latencies) == 1, "subsequent SSE subscriber receiving same publish must dedup"


def test_record_alert_latency_distinct_alert_ids_emit_separately(fake_mlflow) -> None:
    """Distinct ``alert_id`` values MUST each emit (no false dedup)."""
    lm.record_alert_latency(payload={"publish_at": 0, "alert_id": "a1", "brands": ["kisqali"]})
    lm.record_alert_latency(payload={"publish_at": 0, "alert_id": "a2", "brands": ["kisqali"]})

    latencies = [m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms"]
    assert len(latencies) == 2


def test_record_alert_latency_no_alert_id_emits_every_time(fake_mlflow) -> None:
    """Back-compat: alerts published BEFORE the alert_id stamping
    landed (no ``alert_id`` field) MUST emit on every observation.
    The dedup is opt-in via alert_id."""
    lm.record_alert_latency(payload={"publish_at": 0, "brands": ["kisqali"]})
    lm.record_alert_latency(payload={"publish_at": 0, "brands": ["kisqali"]})
    lm.record_alert_latency(payload={"publish_at": 0, "brands": ["kisqali"]})

    latencies = [m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms"]
    assert len(latencies) == 3


# ---------------------------------------------------------------------
# M1 — brand fanout
# ---------------------------------------------------------------------


def test_coerce_brand_tags_single_brand(fake_mlflow) -> None:
    """Single-brand payload → brand=<brand>."""
    tags = lm._coerce_brand_tags({"brands": ["kisqali"]})
    assert tags == {"brand": "kisqali"}


def test_coerce_brand_tags_two_distinct_brands(fake_mlflow) -> None:
    """Two-or-more distinct brands → brand=_multi_."""
    tags = lm._coerce_brand_tags({"brands": ["kisqali", "remibrutinib"]})
    assert tags == {"brand": "_multi_"}


def test_coerce_brand_tags_cross_brand_all(fake_mlflow) -> None:
    """Cross-brand (``all`` in brands) → brand=all."""
    tags = lm._coerce_brand_tags({"brands": ["all"]})
    assert tags == {"brand": "all"}
    tags = lm._coerce_brand_tags({"brands": ["kisqali", "all"]})
    assert tags == {"brand": "all"}


def test_coerce_brand_tags_empty_returns_empty(fake_mlflow) -> None:
    """Missing / empty / non-list brands → no brand tag."""
    assert lm._coerce_brand_tags({}) == {}
    assert lm._coerce_brand_tags({"brands": []}) == {}
    assert lm._coerce_brand_tags({"brands": None}) == {}
    assert lm._coerce_brand_tags({"brands": "kisqali"}) == {}  # not a list


def test_record_alert_latency_multi_brand_uses_multi_tag(fake_mlflow) -> None:
    """Multi-brand alert MUST emit one latency sample tagged
    ``brand="_multi_"`` (not just the first brand)."""
    lm.record_alert_latency(
        payload={
            "publish_at": 0,
            "alert_id": "multi-a1",
            "brands": ["kisqali", "remibrutinib"],
        }
    )
    latency = next(m for m in fake_mlflow if m[0] == "e2i.sentinel.alert_delivery_latency_ms")
    assert latency[2].get("brand") == "_multi_"


# ---------------------------------------------------------------------
# M2 — background MLflow executor (codex iter-0 M2 closure)
# ---------------------------------------------------------------------


def test_mlflow_executor_is_lazy_singleton() -> None:
    """Two calls to ``_get_mlflow_executor`` return the SAME instance."""
    e1 = lm._get_mlflow_executor()
    e2 = lm._get_mlflow_executor()
    assert e1 is e2


def test_mlflow_executor_uses_single_thread_for_ordering() -> None:
    """The executor MUST be single-thread so MLflow metric ordering
    is preserved across emissions (helps with debugging)."""
    executor = lm._get_mlflow_executor()
    assert executor._max_workers == 1
