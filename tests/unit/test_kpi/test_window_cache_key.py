"""Part 2: the requested window is part of the cache context.

Two calls with different windows must NOT collide in cache. We assert this at
the seam where `KPICalculator.calculate` builds its `cache_context` and hands it
to the cache: a fake cache records the context kwargs it is queried/set with, so
we can confirm `_window` is present and distinguishes windows -- without needing
a live Redis.
"""

from typing import Any

import pytest

from src.kpi.calculator import KPICalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


class _RecordingCache:
    """Always-enabled fake cache; records context kwargs of get/set."""

    enabled = True

    def __init__(self):
        self.get_contexts: list[dict[str, Any]] = []
        self.set_contexts: list[dict[str, Any]] = []
        self._store: dict[str, KPIResult] = {}

    def _key(self, kpi_id: str, context: dict[str, Any]) -> str:
        return kpi_id + "|" + repr(sorted(context.items(), key=lambda kv: kv[0]))

    def get(self, kpi_id: str, **context: Any) -> KPIResult | None:
        self.get_contexts.append(context)
        hit = self._store.get(self._key(kpi_id, context))
        if hit is None:
            return None
        # Mirror the real cache, which reconstructs the result with cached=True.
        return hit.model_copy(update={"cached": True})

    def set(self, result: KPIResult, ttl: int | None = None, **context: Any) -> bool:
        self.set_contexts.append(context)
        self._store[self._key(result.kpi_id, context)] = result
        return True


class _StubRegistry:
    def __init__(self, kpi: KPIMetadata):
        self._kpi = kpi

    def get(self, kpi_id: str) -> KPIMetadata | None:
        return self._kpi if kpi_id == self._kpi.id else None


def _kpi() -> KPIMetadata:
    return KPIMetadata(
        id="WS3-BI-006",
        name="New Prescriptions (NRx)",
        definition="d",
        formula="f",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS3_BUSINESS,
        windowable="clean",
    )


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


def _make_calc(cache: _RecordingCache) -> KPICalculator:
    calc = KPICalculator(registry=_StubRegistry(_kpi()), cache=cache)
    # Make _calculate_kpi cheap & deterministic (no real calculators/DB).
    calc._calculate_kpi = lambda kpi, context: KPIResult(  # type: ignore[method-assign]
        kpi_id=kpi.id, value=1.0, status=KPIStatus.UNKNOWN, cached=False, error=None
    )
    return calc


def test_window_is_part_of_cache_context():
    cache = _RecordingCache()
    calc = _make_calc(cache)
    w = {"start": "2025-01-01", "end": "2025-04-01"}
    calc.calculate("WS3-BI-006", context={"brand": "Kisqali", "window": w})
    assert cache.set_contexts, "expected a cache set"
    ctx = cache.set_contexts[-1]
    assert "_window" in ctx
    assert ctx["_window"] == ("2025-01-01", "2025-04-01")


def test_no_window_records_none_window_key():
    cache = _RecordingCache()
    calc = _make_calc(cache)
    calc.calculate("WS3-BI-006", context={"brand": "Kisqali"})
    ctx = cache.set_contexts[-1]
    assert "_window" in ctx
    assert ctx["_window"] is None


def test_different_windows_do_not_collide():
    cache = _RecordingCache()
    calc = _make_calc(cache)
    w1 = {"start": "2025-01-01", "end": "2025-04-01"}
    w2 = {"start": "2025-04-01", "end": "2025-07-01"}

    # First window: miss -> compute -> set.
    r1 = calc.calculate("WS3-BI-006", context={"brand": "Kisqali", "window": w1})
    assert r1.value == 1.0
    # Second, different window: must NOT hit the w1 entry (distinct cache key).
    before_sets = len(cache.set_contexts)
    calc.calculate("WS3-BI-006", context={"brand": "Kisqali", "window": w2})
    # A new set happened -> it was a miss, not a collision with w1.
    assert len(cache.set_contexts) == before_sets + 1
    # Re-requesting w1 should now be a cache HIT (cached flag True).
    r1_again = calc.calculate("WS3-BI-006", context={"brand": "Kisqali", "window": w1})
    assert r1_again.cached is True
