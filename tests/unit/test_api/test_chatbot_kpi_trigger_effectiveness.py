"""kpi_calculate_tool wiring for the trigger-effectiveness KPIs (#1360).

The ruling: trigger precision (WS2-TR-001), acceptance rate (WS2-TR-004),
override rate (WS2-TR-006) and trigger funnel conversion (WS2-TR-009) are
chat-KPI-path KPIs — computable BY NAME through ``kpi_calculate_tool`` with
brand / region / trigger_type / window axes. Pure unit coverage (no DB): the
tool must thread ``trigger_type`` into the calculator context, guard it against
non-trigger KPIs (never a silent dead-key drop), disclose the real reporting
windows, and surface the funnel stage counts.
"""

import pytest

from src.kpi.models import KPIResult, KPIStatus
from src.kpi.registry import get_registry

TRIGGER_KPI_IDS = ["WS2-TR-001", "WS2-TR-004", "WS2-TR-006", "WS2-TR-009"]


@pytest.mark.unit
def test_kpi_calculate_input_declares_trigger_type_axis():
    from src.api.routes.chatbot_tools import KpiCalculateInput

    assert "trigger_type" in KpiCalculateInput.model_fields
    field = KpiCalculateInput.model_fields["trigger_type"]
    assert field.default is None  # optional filter


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_passes_trigger_type_into_context(monkeypatch):
    """trigger_type must reach the calculator under the key it reads —
    ``context['trigger_type']`` (TriggerPerformanceCalculator routes the
    migration-118 effectiveness family on it)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["kpi_id"] = kpi_id
            captured["context"] = context
            return KPIResult(kpi_id=kpi_id, value=0.44, status=KPIStatus.GOOD)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "acceptance rate", "brand": "Kisqali", "trigger_type": "adherence_risk"}
    )
    assert resp["success"] is True
    assert captured["kpi_id"] == "WS2-TR-004"
    assert captured["context"]["trigger_type"] == "adherence_risk"
    assert captured["context"]["brand"] == "Kisqali"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trigger_type_on_non_trigger_kpi_fails_honestly(monkeypatch):
    """A trigger_type filter on a KPI that cannot honor it (e.g. TRx) must
    error BEFORE touching the calculator — a context key no calculator reads
    would silently drop the filter while the response implied it applied (the
    dead-'territory'-key incident)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    calls: list = []

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            calls.append(kpi_id)
            return KPIResult(kpi_id=kpi_id, value=1.0, status=KPIStatus.UNKNOWN)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "TRx", "brand": "Kisqali", "trigger_type": "adherence_risk"}
    )
    assert resp["success"] is False
    assert "trigger_type" in resp["error"]
    assert calls == []  # never reached the calculator


@pytest.mark.unit
def test_reporting_windows_cover_the_trigger_effectiveness_family():
    """All four ruled KPIs disclose their registry-verified default window;
    precision's is the LAGGED matured cohort (migration 113), not a plain
    trailing 30 days."""
    from src.api.routes.chatbot_tools import KPI_REPORTING_WINDOWS

    for kpi_id in TRIGGER_KPI_IDS:
        assert kpi_id in KPI_REPORTING_WINDOWS, f"{kpi_id} missing a reporting window"
    assert "30 days" in KPI_REPORTING_WINDOWS["WS2-TR-001"]
    assert "before" in KPI_REPORTING_WINDOWS["WS2-TR-001"], (
        "precision's default window is the matured cohort ending 30 days before "
        "the trigger-data frontier — describing it as a plain trailing window "
        "would misstate the figure"
    )
    for kpi_id in ("WS2-TR-004", "WS2-TR-006", "WS2-TR-009"):
        assert "trigger" in KPI_REPORTING_WINDOWS[kpi_id]


@pytest.mark.unit
def test_semantic_notes_disambiguate_precision_and_define_the_funnel():
    """WS2-TR-001 'precision' must be flagged as NBA-trigger precision (not
    deployed-model telemetry — the bench-0024 health_score confusion), and the
    funnel note must pin the headline to actioned/delivered."""
    from src.api.routes.chatbot_tools import KPI_SEMANTIC_NOTES

    assert "WS2-TR-001" in KPI_SEMANTIC_NOTES
    note_001 = KPI_SEMANTIC_NOTES["WS2-TR-001"].lower()
    assert "model" in note_001  # explicitly NOT model telemetry
    assert "WS2-TR-009" in KPI_SEMANTIC_NOTES
    note_009 = KPI_SEMANTIC_NOTES["WS2-TR-009"].lower()
    assert "actioned" in note_009 and "delivered" in note_009


@pytest.mark.unit
def test_kpi_result_to_response_surfaces_funnel_stages():
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS2-TR-009")
    assert kpi is not None, "WS2-TR-009 must be defined in the KPI registry"
    stages = {
        "delivered": 33023,
        "viewed": 9351,
        "accepted": 18120,
        "actioned": 7471,
        "outcome": 1830,
    }
    result = KPIResult(
        kpi_id="WS2-TR-009",
        value=0.2262,
        status=KPIStatus.INFORMATIONAL,
        metadata={"include_synthetic": True, "context": {"funnel_stages": stages}},
    )
    resp = _kpi_result_to_response(kpi, result)
    assert resp["success"] is True
    assert resp["funnel_stages"] == stages
    assert resp["data_source"] == "synthetic"


@pytest.mark.unit
def test_kpi_result_to_response_omits_funnel_stages_when_absent():
    """Non-funnel KPIs must not grow a funnel_stages key."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS2-TR-004")
    assert kpi is not None
    result = KPIResult(kpi_id="WS2-TR-004", value=0.55, status=KPIStatus.GOOD, metadata={})
    resp = _kpi_result_to_response(kpi, result)
    assert "funnel_stages" not in resp
