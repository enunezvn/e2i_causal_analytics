"""Registration invariants for the KPI route's calculator wiring.

Previously `get_kpi_calculator()` registered NO per-workstream calculators, so
every /api/kpis grid/single/batch KPI fell through to the unimplemented generic
table path and returned honest None+error ("Not yet computed"). The calculators
were deliberately kept unregistered while they still held fabricating placeholder
defaults (#421/#439). Now that they FAIL-LOUD on missing data, they are
registered. These tests lock that wiring + the workstream<->calculator mapping.
"""

from __future__ import annotations

import pytest

from src.api.routes.kpi import _WORKSTREAM_CALCULATORS, get_kpi_calculator
from src.kpi.calculators.brand_specific import BrandSpecificCalculator
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.calculators.causal_metrics import CausalMetricsCalculator
from src.kpi.calculators.data_quality import DataQualityCalculator
from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIThreshold,
    Workstream,
)

_EXPECTED = {
    Workstream.WS1_DATA_QUALITY: DataQualityCalculator,
    Workstream.WS1_MODEL_PERFORMANCE: ModelPerformanceCalculator,
    Workstream.WS2_TRIGGERS: TriggerPerformanceCalculator,
    Workstream.WS3_BUSINESS: BusinessImpactCalculator,
    Workstream.BRAND_SPECIFIC: BrandSpecificCalculator,
    Workstream.CAUSAL_METRICS: CausalMetricsCalculator,
}


def test_every_workstream_has_a_registered_calculator():
    # No workstream may be left unmapped (else its KPIs silently fall to the
    # unimplemented generic path = honest error, never a value).
    assert set(_WORKSTREAM_CALCULATORS.keys()) == set(Workstream)
    assert _WORKSTREAM_CALCULATORS == _EXPECTED


def test_get_kpi_calculator_registers_all_six(monkeypatch):
    # Registration must not require a live DB; the calculators resolve their
    # client lazily and are not invoked at registration time.
    monkeypatch.setattr("src.api.routes.kpi.get_supabase", lambda: None)
    calc = get_kpi_calculator()
    assert set(calc._calculators.keys()) == set(Workstream)
    for ws, cls in _EXPECTED.items():
        assert isinstance(calc._calculators[ws], cls)


@pytest.mark.parametrize("workstream,calculator_cls", list(_EXPECTED.items()))
def test_registered_calculator_supports_its_own_workstream(workstream, calculator_cls):
    # The map must be coherent: each calculator's supports() accepts a KPI in the
    # workstream it is registered under (otherwise _calculate_kpi falls through).
    kpi = KPIMetadata(
        id="X-001",
        name="probe",
        definition="probe",
        formula="probe",
        calculation_type=CalculationType.DERIVED,
        workstream=workstream,
        threshold=KPIThreshold(target=0.5, warning=0.3, critical=0.1),
    )
    assert calculator_cls().supports(kpi) is True
