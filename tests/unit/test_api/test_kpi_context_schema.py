"""Schema contract for KPICalculationContext.therapy_line (migration 105: patient-segment
KPI breakdowns). ``segment`` already existed; this covers its new sibling field and the
combined segment+therapy_line round-trip."""

from __future__ import annotations

import pytest

from src.api.schemas.kpi import KPICalculationContext


@pytest.mark.unit
def test_therapy_line_round_trips():
    ctx = KPICalculationContext(therapy_line="2")
    assert ctx.therapy_line == "2"


@pytest.mark.unit
def test_therapy_line_defaults_to_none():
    ctx = KPICalculationContext()
    assert ctx.therapy_line is None
    assert ctx.segment is None


@pytest.mark.unit
def test_segment_and_therapy_line_coexist_on_the_model():
    """Both fields can be set on the model at once -- mutual exclusivity is a routing
    contract enforced by BusinessImpactCalculator._resolve_windowed_call (segment takes
    precedence), not a schema-level validation constraint."""
    ctx = KPICalculationContext(segment="high_severity", therapy_line="3")
    assert ctx.segment == "high_severity"
    assert ctx.therapy_line == "3"
