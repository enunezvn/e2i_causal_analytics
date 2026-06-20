"""Part 3: generic window provenance stamping on KPIResult.

`KPICalculator._stamp_window(result, kpi, window)` sets window provenance from
the KPI's `windowable` classification and the requested window, for ALL KPIs
(registered-calculator path and default path alike):

- clean / needs_care + window  -> status "applied", requested == applied == window
- not_applicable    + window  -> status "not_applicable", applied None, value kept
- any KPI, no window           -> defaults left untouched ("default")

We test the pure helper directly, then assert calculate() routes through it for
both a clean and a not_applicable KPI.
"""


import pytest

from src.kpi.calculator import KPICalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)

_WINDOW = {"start": "2025-01-01", "end": "2025-04-01"}


def _kpi(kpi_id: str, windowable: str) -> KPIMetadata:
    return KPIMetadata(
        id=kpi_id,
        name=f"name-{kpi_id}",
        definition="d",
        formula="f",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS3_BUSINESS,
        windowable=windowable,
    )


def _calc() -> KPICalculator:
    # No real cache/registry needed for the pure-helper tests.
    return KPICalculator()


# ---- pure helper -----------------------------------------------------------


@pytest.mark.parametrize("windowable", ["clean", "needs_care"])
def test_stamp_clean_applies_window(windowable):
    r = KPIResult(kpi_id="WS3-BI-006", value=3394.0, status=KPIStatus.UNKNOWN)
    out = _calc()._stamp_window(r, _kpi("WS3-BI-006", windowable), _WINDOW)
    assert out.window_status == "applied"
    assert out.window_requested == _WINDOW
    assert out.window_applied == _WINDOW
    assert out.value == 3394.0  # value untouched


def test_stamp_not_applicable_ignores_window_but_keeps_value():
    r = KPIResult(kpi_id="WS3-BI-004", value=0.42, status=KPIStatus.GOOD)
    out = _calc()._stamp_window(r, _kpi("WS3-BI-004", "not_applicable"), _WINDOW)
    assert out.window_status == "not_applicable"
    assert out.window_requested == _WINDOW
    assert out.window_applied is None
    assert out.value == 0.42  # still computed


def test_stamp_no_window_leaves_defaults():
    r = KPIResult(kpi_id="WS3-BI-006", value=1.0, status=KPIStatus.UNKNOWN)
    out = _calc()._stamp_window(r, _kpi("WS3-BI-006", "clean"), None)
    assert out.window_status == "default"
    assert out.window_requested is None
    assert out.window_applied is None


# ---- wired through calculate() ---------------------------------------------


class _AlwaysOffCache:
    enabled = True

    def get(self, kpi_id, **context):
        return None

    def set(self, result, ttl=None, **context):
        return True


class _StubRegistry:
    def __init__(self, kpi: KPIMetadata):
        self._kpi = kpi

    def get(self, kpi_id: str) -> KPIMetadata | None:
        return self._kpi if kpi_id == self._kpi.id else None


def _wired_calc(kpi: KPIMetadata, value: float) -> KPICalculator:
    calc = KPICalculator(registry=_StubRegistry(kpi), cache=_AlwaysOffCache())
    calc._calculate_kpi = lambda k, ctx: KPIResult(  # type: ignore[method-assign]
        kpi_id=k.id, value=value, status=KPIStatus.GOOD, cached=False, error=None
    )
    return calc


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


def test_calculate_stamps_applied_for_clean_kpi():
    kpi = _kpi("WS3-BI-006", "clean")
    calc = _wired_calc(kpi, 3394.0)
    res = calc.calculate("WS3-BI-006", context={"brand": "Kisqali", "window": _WINDOW})
    assert res.window_status == "applied"
    assert res.window_applied == _WINDOW
    assert res.value == 3394.0


def test_calculate_stamps_not_applicable_but_computes_value():
    kpi = _kpi("WS3-BI-004", "not_applicable")
    calc = _wired_calc(kpi, 0.42)
    res = calc.calculate("WS3-BI-004", context={"brand": "Kisqali", "window": _WINDOW})
    assert res.window_status == "not_applicable"
    assert res.window_applied is None
    assert res.value == 0.42


def test_calculate_stamps_not_applicable_for_roi():
    """WS3-BI-010 ROI has windowable=not_applicable: window is recorded but not applied."""
    kpi = _kpi("WS3-BI-010", "not_applicable")
    calc = _wired_calc(kpi, 2.5)
    res = calc.calculate("WS3-BI-010", context={"brand": "Kisqali", "window": _WINDOW})
    assert res.window_status == "not_applicable"
    assert res.window_requested == _WINDOW
    assert res.window_applied is None
    assert res.value == 2.5  # ROI still computed, just not windowed


def test_calculate_no_window_is_default():
    kpi = _kpi("WS3-BI-006", "clean")
    calc = _wired_calc(kpi, 1.0)
    res = calc.calculate("WS3-BI-006", context={"brand": "Kisqali"})
    assert res.window_status == "default"
    assert res.window_requested is None
