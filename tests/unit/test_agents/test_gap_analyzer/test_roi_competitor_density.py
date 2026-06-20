"""Task 8: gap-analysis surfaces curated competitor density on each ROI estimate
(count + label + names), case-insensitive on the brand, and fail-open. This is
INFORMATIONAL only — it does NOT change the ROI value or the prioritizer ranking
(covered by the unchanged test_roi_calculator.py)."""

from __future__ import annotations

import pytest

from src.agents.gap_analyzer.nodes.roi_calculator import ROICalculatorNode


@pytest.mark.unit
@pytest.mark.parametrize("brand", ["kisqali", "Kisqali", "KISQALI"])
def test_competitor_density_case_insensitive(brand):
    d = ROICalculatorNode._competitor_density(brand)
    assert d["competitor_products_count"] >= 1
    assert d["competitor_density_label"] in {"limited", "moderate", "crowded"}
    assert any(
        "palbociclib" in c.lower() or "abemaciclib" in c.lower()
        for c in d["competitor_drug_names"]
    )


@pytest.mark.unit
def test_competitor_density_unknown_brand_is_empty():
    d = ROICalculatorNode._competitor_density("NotABrand")
    assert d == {
        "competitor_products_count": 0,
        "competitor_density_label": "unknown",
        "competitor_drug_names": [],
    }


@pytest.mark.unit
def test_competitor_density_none_brand_is_empty():
    d = ROICalculatorNode._competitor_density(None)
    assert d["competitor_products_count"] == 0
    assert d["competitor_density_label"] == "unknown"
    assert d["competitor_drug_names"] == []


@pytest.mark.unit
def test_competitor_density_fail_open(monkeypatch):
    import src.services.clinical_context.brand_map as bm

    def _boom(_key):
        raise RuntimeError("brand map error")

    monkeypatch.setattr(bm, "resolve_brand_profile", _boom)
    # A valid brand key resolves, but resolution raises -> fail-open to empty.
    d = ROICalculatorNode._competitor_density("Kisqali")
    assert d["competitor_products_count"] == 0
    assert d["competitor_density_label"] == "unknown"
