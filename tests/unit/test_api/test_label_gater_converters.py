"""Converter guard (codex HIGH#4): the off-label flag must survive the agent->API
conversion or the UI can never show it. Asserts _convert_policies / _convert_opportunities
carry off_label / off_label_reason / label_verdict / label_evidence_confirmed."""

import pytest


@pytest.mark.unit
def test_convert_policies_carries_off_label_fields():
    from src.api.routes.segments import _convert_policies

    out = _convert_policies(
        [
            {
                "segment": "prior_antihistamine_therapy=False",
                "current_treatment_rate": 0.5,
                "recommended_treatment_rate": 0.7,
                "expected_incremental_outcome": 50.0,
                "confidence": 0.8,
                "off_label": True,
                "off_label_reason": "treatment-naive; label requires prior H1-antihistamine failure",
                "label_verdict": "off_label",
                "label_evidence_confirmed": True,
            }
        ]
    )
    assert len(out) == 1
    p = out[0]
    assert p.off_label is True
    assert p.label_verdict == "off_label"
    assert p.off_label_reason
    assert p.label_evidence_confirmed is True


@pytest.mark.unit
def test_convert_policies_off_label_absent_defaults_none():
    from src.api.routes.segments import _convert_policies

    out = _convert_policies(
        [
            {
                "segment": "region=Northeast",
                "current_treatment_rate": 0.5,
                "recommended_treatment_rate": 0.6,
                "expected_incremental_outcome": 10.0,
                "confidence": 0.7,
            }
        ]
    )
    assert out[0].off_label is None and out[0].label_verdict is None


@pytest.mark.unit
def test_convert_opportunities_carries_off_label_fields():
    from src.api.routes.gaps import _convert_opportunities

    out = _convert_opportunities(
        [
            {
                "rank": 1,
                "gap": {
                    "gap_id": "specialty_Dermatology_trx",
                    "metric": "trx",
                    "segment": "hr_status",
                    "segment_value": "negative",
                    "current_value": 1.0,
                    "target_value": 2.0,
                    "gap_size": 1.0,
                    "gap_percentage": 50.0,
                    "gap_type": "vs_target",
                },
                "roi_estimate": {
                    "gap_id": "specialty_Dermatology_trx",
                    "estimated_revenue_impact": 100.0,
                    "estimated_cost_to_close": 10.0,
                    "expected_roi": 9.0,
                    "risk_adjusted_roi": 7.0,
                    "payback_period_months": 6,
                    "attribution_level": "partial",
                    "attribution_rate": 0.5,
                    "confidence": 0.7,
                    "off_label": True,
                    "off_label_reason": "HR-negative is outside the HR+/HER2- indication",
                    "label_verdict": "off_label",
                    "label_evidence_confirmed": True,
                },
                "recommended_action": "x",
                "implementation_difficulty": "medium",
                "time_to_impact": "3-6 months",
            }
        ]
    )
    assert len(out) == 1
    roi = out[0].roi_estimate
    assert roi.off_label is True
    assert roi.label_verdict == "off_label"
    assert roi.off_label_reason


@pytest.mark.unit
def test_convert_opportunities_carries_competitor_density_fields():
    """#1056: the surface-only competitor-density fields the ROI node writes onto
    each estimate must serialize through the API ROIEstimate. They were silently
    dropped because the Pydantic response model never declared them, so the FE
    could never display them."""
    from src.api.routes.gaps import _convert_opportunities

    out = _convert_opportunities(
        [
            {
                "rank": 1,
                "gap": {
                    "gap_id": "region_Northeast_trx",
                    "metric": "trx",
                    "segment": "region",
                    "segment_value": "Northeast",
                    "current_value": 85.0,
                    "target_value": 100.0,
                    "gap_size": 15.0,
                    "gap_percentage": 15.0,
                    "gap_type": "vs_target",
                },
                "roi_estimate": {
                    "gap_id": "region_Northeast_trx",
                    "estimated_revenue_impact": 500000.0,
                    "estimated_cost_to_close": 100000.0,
                    "expected_roi": 5.0,
                    "risk_adjusted_roi": 4.0,
                    "payback_period_months": 6,
                    "attribution_level": "partial",
                    "attribution_rate": 0.7,
                    "confidence": 0.8,
                    "competitor_products_count": 3,
                    "competitor_density_label": "moderate",
                    "competitor_drug_names": ["Verzenio", "Ibrance", "Kisqali"],
                },
                "recommended_action": "Increase coverage",
                "implementation_difficulty": "low",
                "time_to_impact": "3-6 months",
            }
        ]
    )
    assert len(out) == 1
    roi = out[0].roi_estimate
    assert roi.competitor_products_count == 3
    assert roi.competitor_density_label == "moderate"
    assert roi.competitor_drug_names == ["Verzenio", "Ibrance", "Kisqali"]


@pytest.mark.unit
def test_convert_opportunities_competitor_density_absent_is_none():
    """Honest empty state: when the ROI node wrote no density, the API fields are
    None (never fabricated)."""
    from src.api.routes.gaps import _convert_opportunities

    out = _convert_opportunities(
        [
            {
                "rank": 1,
                "gap": {
                    "gap_id": "g",
                    "metric": "trx",
                    "segment": "region",
                    "segment_value": "X",
                    "current_value": 1.0,
                    "target_value": 2.0,
                    "gap_size": 1.0,
                    "gap_percentage": 50.0,
                    "gap_type": "vs_target",
                },
                "roi_estimate": {
                    "gap_id": "g",
                    "estimated_revenue_impact": 1.0,
                    "estimated_cost_to_close": 1.0,
                    "expected_roi": 1.0,
                    "risk_adjusted_roi": 1.0,
                    "payback_period_months": 6,
                    "attribution_level": "partial",
                    "attribution_rate": 0.5,
                    "confidence": 0.7,
                },
                "recommended_action": "x",
                "implementation_difficulty": "low",
                "time_to_impact": "3-6 months",
            }
        ]
    )
    roi = out[0].roi_estimate
    assert roi.competitor_products_count is None
    assert roi.competitor_density_label is None
    assert roi.competitor_drug_names is None
