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

    def test_trx_share_legs(self, registry):
        """WS3-BI-008 TRx Share must have window.legs == ['brand_rx', 'category']."""
        kpi = registry.get("WS3-BI-008")
        assert kpi is not None, "WS3-BI-008 not found in registry"
        assert kpi.windowable == "clean"
        assert kpi.window is not None
        assert kpi.window["legs"] == ["brand_rx", "category"]

    def test_conversion_rate_look_forward_days(self, registry):
        """WS3-BI-009 Conversion Rate must have window.look_forward_days == 30."""
        kpi = registry.get("WS3-BI-009")
        assert kpi is not None, "WS3-BI-009 not found in registry"
        assert kpi.windowable == "clean"
        assert kpi.window is not None
        assert kpi.window["look_forward_days"] == 30

    def test_snapshot_kpi_not_applicable(self, registry):
        """WS1-MP-001 ROC-AUC (snapshot/ML KPI) must be windowable=not_applicable."""
        kpi = registry.get("WS1-MP-001")
        assert kpi is not None, "WS1-MP-001 not found in registry"
        assert kpi.windowable == "not_applicable"

    def test_clean_kpi_count_equals_18(self, registry):
        """Exactly 18 KPIs in the config must have windowable='clean'."""
        clean_kpis = [kpi for kpi in registry.get_all() if kpi.windowable == "clean"]
        ids = sorted(k.id for k in clean_kpis)
        assert len(clean_kpis) == 18, (
            f"Expected 18 clean KPIs, got {len(clean_kpis)}. IDs: {ids}"
        )
