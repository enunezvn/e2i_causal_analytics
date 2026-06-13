"""Provenance + cache-isolation tests for KPI synthetic-visibility demo mode.

Locks the two fixes the codex review required:
1. KPICalculator stamps `metadata['include_synthetic']` and keys the cache on the
   synthetic mode, so a value computed in demo mode can NEVER be served after the
   flag is unset (and vice versa) — the reversible gate holds.
2. `_result_to_response` surfaces `data_source='synthetic'` so the FE badges the
   figure rather than reading it as real-world data.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.kpi.calculator import KPICalculator, KPICalculatorBase
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    KPIThreshold,
    Workstream,
)
from src.kpi.registry import KPIRegistry

_FLAG = "E2I_KPI_INCLUDE_SYNTHETIC"


class _FixedCalculator(KPICalculatorBase):
    def calculate(self, kpi, context=None):
        return KPIResult(kpi_id=kpi.id, value=0.77, status=KPIStatus.GOOD)

    def supports(self, kpi) -> bool:
        return True


class _RecordingCache:
    """A cache stub that is 'enabled' and records the kwargs of every get/set so
    the test can assert the synthetic-mode discriminator lands in the key."""

    def __init__(self):
        self.enabled = True
        self.get_calls: list[dict] = []
        self.set_calls: list[dict] = []

    def get(self, kpi_id, **context):
        self.get_calls.append(context)
        return None  # force a real calculation

    def set(self, result, ttl=None, **context):
        self.set_calls.append(context)
        return True


def _kpi() -> KPIMetadata:
    return KPIMetadata(
        id="WS1-MP-001",
        name="Model ROC-AUC",
        definition="t",
        formula="t",
        calculation_type=CalculationType.DERIVED,
        workstream=Workstream.WS1_DATA_QUALITY,
        threshold=KPIThreshold(target=0.85, warning=0.70, critical=0.50),
    )


@pytest.fixture
def calc():
    registry = Mock(spec=KPIRegistry)
    registry.get.return_value = _kpi()
    cache = _RecordingCache()
    c = KPICalculator(registry=registry, cache=cache, router=Mock())
    c.register_calculator(Workstream.WS1_DATA_QUALITY, _FixedCalculator())
    return c, cache


def test_cache_key_carries_synthetic_discriminator_off(calc, monkeypatch):
    c, cache = calc
    monkeypatch.delenv(_FLAG, raising=False)
    result = c.calculate("WS1-MP-001")
    assert result.metadata["include_synthetic"] is False
    # The cache get + set both saw the mode discriminator (False).
    assert cache.get_calls[0].get("_include_synthetic") is False
    assert cache.set_calls[0].get("_include_synthetic") is False


def test_cache_key_carries_synthetic_discriminator_on(calc, monkeypatch):
    c, cache = calc
    monkeypatch.setenv(_FLAG, "true")
    result = c.calculate("WS1-MP-001")
    assert result.metadata["include_synthetic"] is True
    assert cache.get_calls[0].get("_include_synthetic") is True
    assert cache.set_calls[0].get("_include_synthetic") is True


def test_demo_and_prod_modes_use_distinct_cache_keys(monkeypatch):
    """The same kpi/context must land on DIFFERENT cache keys per mode, so a
    synthetic value is never served once the flag is off (the reversibility bug)."""
    from src.kpi.cache import KPICache

    cache = KPICache.__new__(KPICache)  # bypass Redis connect; we only need _make_key
    cache.KEY_PREFIX = KPICache.KEY_PREFIX
    off = cache._make_key("WS1-MP-001", brand="All", _include_synthetic=False)
    on = cache._make_key("WS1-MP-001", brand="All", _include_synthetic=True)
    assert off != on


def test_result_to_response_maps_data_source(monkeypatch):
    from src.api.routes.kpi import _result_to_response

    synth = KPIResult(
        kpi_id="WS1-MP-001",
        value=0.77,
        status=KPIStatus.GOOD,
        metadata={"include_synthetic": True},
    )
    real = KPIResult(
        kpi_id="WS1-MP-001",
        value=0.77,
        status=KPIStatus.GOOD,
        metadata={"include_synthetic": False},
    )
    assert _result_to_response(synth).data_source == "synthetic"
    assert _result_to_response(real).data_source == "database"
    # Absent metadata defaults to the production label, never "synthetic".
    bare = KPIResult(kpi_id="WS1-MP-001", value=0.77, status=KPIStatus.GOOD)
    assert _result_to_response(bare).data_source == "database"
