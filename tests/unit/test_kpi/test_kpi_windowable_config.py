"""
Tests for per-KPI windowable flag and window block in KPI definitions.

TDD: these tests are written BEFORE the implementation; they should fail initially
and pass once KPIMetadata, the loader, and kpi_definitions.yaml are updated.
"""

import pytest

from src.kpi.registry import KPIRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset KPI registry singleton between tests."""
    KPIRegistry.reset()
    yield
    KPIRegistry.reset()


@pytest.fixture
def registry():
    return KPIRegistry()


class TestWindowableFields:
    def test_nrx_windowable_clean_and_column(self, registry):
        """WS3-BI-006 NRx must be windowable=clean with window.column=event_date."""
        kpi = registry.get("WS3-BI-006")
        assert kpi is not None, "WS3-BI-006 not found in registry"
        assert kpi.windowable == "clean"
        assert kpi.window is not None
        assert kpi.window["column"] == "event_date"

    def test_trx_share_windowable_clean(self, registry):
        """WS3-BI-008 TRx Share gained windowed SQL in migration 111; clean over event_date."""
        kpi = registry.get("WS3-BI-008")
        assert kpi is not None, "WS3-BI-008 not found in registry"
        assert kpi.windowable == "clean"
        assert kpi.window is not None
        assert kpi.window["column"] == "event_date"

    def test_conversion_rate_windowable_clean(self, registry):
        """WS3-BI-009 Conversion Rate gained windowed SQL in migration 111; the
        window bounds trigger_timestamp (which triggers count), never the 30-day
        trigger->Rx conversion horizon."""
        kpi = registry.get("WS3-BI-009")
        assert kpi is not None, "WS3-BI-009 not found in registry"
        assert kpi.windowable == "clean"
        assert kpi.window is not None
        assert kpi.window["column"] == "trigger_timestamp"

    def test_roi_not_applicable(self, registry):
        """WS3-BI-010 ROI has no working windowed SQL; must be not_applicable."""
        kpi = registry.get("WS3-BI-010")
        assert kpi is not None, "WS3-BI-010 not found in registry"
        assert kpi.windowable == "not_applicable"

    def test_snapshot_kpi_not_applicable(self, registry):
        """WS1-MP-001 ROC-AUC (snapshot/ML KPI) must be windowable=not_applicable."""
        kpi = registry.get("WS1-MP-001")
        assert kpi is not None, "WS1-MP-001 not found in registry"
        assert kpi.windowable == "not_applicable"

    def test_clean_kpi_count_equals_5(self, registry):
        """Exactly 5 KPIs (TRx/NRx/NBRx + TRx Share/Conversion Rate, mig 111)
        must have windowable='clean'."""
        clean_kpis = [kpi for kpi in registry.get_all() if kpi.windowable == "clean"]
        ids = sorted(k.id for k in clean_kpis)
        assert len(clean_kpis) == 5, f"Expected 5 clean KPIs, got {len(clean_kpis)}. IDs: {ids}"
        assert ids == [
            "WS3-BI-005",
            "WS3-BI-006",
            "WS3-BI-007",
            "WS3-BI-008",
            "WS3-BI-009",
        ], f"Expected TRx/NRx/NBRx/TRx-Share/Conversion only, got: {ids}"
