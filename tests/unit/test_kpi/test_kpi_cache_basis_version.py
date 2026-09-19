"""A cache entry written under an older substrate contract can never be served
under the new one (canonical TRx lane, codex r1 HIGH / Task 10A).

Why this exists: ``KPICache._make_key`` keys only on ``kpi_id`` plus context, and
``KPICalculator.calculate`` serves a hit before computing, while the route stamps
``measure_basis`` from the LIVE registry. So a Redis entry written by the old
containers -- when WS3-BI-005 was an event count over treatment_events -- would be
served for up to its TTL under the canonical basis, wearing the new basis label.
Folding a contract version into the key makes every pre-lane entry unreachable.

The third test deliberately checks the CAPABILITY (both the read and the write
actually carry the version, observed through a fake cache) rather than a source
-text proxy: counting ``**cache_context`` occurrences is satisfiable while a cache
site quietly builds its own unversioned context.
"""

from typing import Any

import pytest

from src.kpi.cache import KPICache
from src.kpi.calculator import KPICalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)
from src.kpi.volume_family import KPI_CACHE_BASIS_VERSION


class _RecordingCache:
    """Always-enabled fake cache; records the context kwargs of get/set."""

    enabled = True

    def __init__(self) -> None:
        self.get_contexts: list[dict[str, Any]] = []
        self.set_contexts: list[dict[str, Any]] = []

    def get(self, kpi_id: str, **context: Any) -> KPIResult | None:
        self.get_contexts.append(context)
        return None

    def set(self, result: KPIResult, ttl: int | None = None, **context: Any) -> bool:
        self.set_contexts.append(context)
        return True


class _StubRegistry:
    def __init__(self, kpi: KPIMetadata) -> None:
        self._kpi = kpi

    def get(self, kpi_id: str) -> KPIMetadata | None:
        return self._kpi if kpi_id == self._kpi.id else None


def _kpi() -> KPIMetadata:
    return KPIMetadata(
        id="WS3-BI-005",
        name="Total Prescriptions (TRx)",
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


def test_the_cache_context_carries_the_basis_version():
    ctx = KPICalculator._cache_context(
        {"brand": "Kisqali", "window": {"start": "2026-06-01", "end": "2026-08-31"}}, True
    )
    assert ctx["_basis"] == KPI_CACHE_BASIS_VERSION
    assert ctx["_window"] == ("2026-06-01", "2026-08-31")
    assert ctx["_include_synthetic"] is True
    assert ctx["brand"] == "Kisqali"


def test_a_pre_lane_key_can_never_match_a_post_lane_key():
    cache = KPICache.__new__(KPICache)
    pre_lane = cache._make_key("WS3-BI-005", brand="Kisqali", _include_synthetic=True, _window=None)
    post_lane = cache._make_key(
        "WS3-BI-005", **KPICalculator._cache_context({"brand": "Kisqali"}, True)
    )
    assert pre_lane != post_lane
    assert KPI_CACHE_BASIS_VERSION in post_lane


def test_every_cache_read_and_write_actually_carries_the_version():
    """Observed at the seam, not counted in the source.

    ``calculate`` must version the key it READS as well as the one it WRITES: a
    versioned write with an unversioned read still serves the pre-lane entry.
    """
    cache = _RecordingCache()
    calc = KPICalculator(registry=_StubRegistry(_kpi()), cache=cache)
    calc._calculate_kpi = lambda kpi, context: KPIResult(  # type: ignore[method-assign]
        kpi_id=kpi.id, value=1.0, status=KPIStatus.UNKNOWN, cached=False, error=None
    )

    calc.calculate("WS3-BI-005", context={"brand": "Kisqali"})

    assert cache.get_contexts, "expected a cache read"
    assert cache.set_contexts, "expected a cache write"
    for where, contexts in (("read", cache.get_contexts), ("write", cache.set_contexts)):
        for ctx in contexts:
            assert ctx.get("_basis") == KPI_CACHE_BASIS_VERSION, (
                f"a cache {where} used an unversioned context: {ctx}"
            )


def test_the_version_is_a_non_empty_scalar_that_lands_in_the_key():
    """A blank or None version would key identically to the pre-lane entries."""
    assert isinstance(KPI_CACHE_BASIS_VERSION, str)
    assert KPI_CACHE_BASIS_VERSION.strip()
    assert ":" not in KPI_CACHE_BASIS_VERSION, "the key separator would split the version"
