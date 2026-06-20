"""WS-ENGINE: the chatbot ``kpi_calculate_tool`` bridges to the KPI engine.

Pure unit coverage (no DB, no mocks — real ``KPIResult``/``KPIMetadata`` objects):
- the KPI name resolves to its definition (e.g. NBRx -> WS3-BI-007), and
- a ``KPIResult`` maps onto the tool response, badging synthetic provenance.

The end-to-end "value computes from the real substrate" is covered by the live
integration test ``tests/integration/test_chatbot_kpi_tool_live.py``.
"""

import pytest

from src.kpi.models import KPIResult, KPIStatus
from src.kpi.registry import get_registry


@pytest.mark.unit
def test_recognize_kpi_resolves_nbrx():
    from src.services.kpi_resolution import recognize_kpi

    kpi = recognize_kpi("what is the NBRx for Kisqali")
    assert kpi is not None and kpi.id == "WS3-BI-007"


@pytest.mark.unit
def test_kpi_result_to_response_value_badges_synthetic():
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-007")
    assert kpi is not None
    result = KPIResult(
        kpi_id="WS3-BI-007",
        value=3298.0,
        status=KPIStatus.UNKNOWN,
        metadata={"include_synthetic": True},
    )
    resp = _kpi_result_to_response(kpi, result, brand="Kisqali")
    assert resp == {
        "success": True,
        "query_type": "kpi_calculate",
        "kpi_id": "WS3-BI-007",
        "kpi_name": kpi.name,
        "value": 3298.0,
        "status": "unknown",  # KPIResult uses_enum_values -> status is the str value
        "data_source": "synthetic",
        "brand": "Kisqali",
        "region": None,
        # No custom window requested -> engine default; provenance reflects that.
        "window_requested": None,
        "window_applied": None,
        "window_status": "default",
        # NBRx counts first-brand Rx over the registry's fixed rolling 30-day
        # window (migrations 044/066); disclosing it stops the chatbot from
        # presenting a 30-day figure as a user-requested "past 3 months".
        # Included ONLY because window_status == "default".
        "reporting_window": "rolling last 30 days",
    }


@pytest.mark.unit
def test_kpi_result_to_response_database_when_not_synthetic():
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-005")  # TRx
    assert kpi is not None
    result = KPIResult(kpi_id="WS3-BI-005", value=10.0, status=KPIStatus.UNKNOWN, metadata={})
    resp = _kpi_result_to_response(kpi, result)
    assert resp["success"] is True
    assert resp["data_source"] == "database"
    # Volume KPIs disclose the real (fixed) reporting window.
    assert resp["reporting_window"] == "rolling last 30 days"


@pytest.mark.unit
def test_kpi_result_to_response_omits_window_for_unverified_kpi():
    """A KPI whose window we have NOT verified against the registry must NOT
    carry a fabricated ``reporting_window`` -- honest absence over a guessed
    period. ROI (WS3-BI-010) is not in the verified-window map."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-010")  # ROI
    assert kpi is not None
    result = KPIResult(kpi_id="WS3-BI-010", value=1.5, status=KPIStatus.UNKNOWN, metadata={})
    resp = _kpi_result_to_response(kpi, result)
    assert "reporting_window" not in resp


@pytest.mark.unit
def test_kpi_result_to_response_error_passthrough():
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-007")
    result = KPIResult(kpi_id="WS3-BI-007", value=None, status=KPIStatus.UNKNOWN, error="boom")
    resp = _kpi_result_to_response(kpi, result)
    assert resp["success"] is False
    assert resp["error"] == "boom"
    assert "value" not in resp


@pytest.mark.unit
def test_kpi_result_to_response_echoes_brand_and_window_provenance():
    """A custom window that was honored: the response echoes the brand and the
    engine's window provenance, and DROPS the stale fixed-window note (the answer
    no longer covers the default rolling window)."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-007")  # NBRx
    assert kpi is not None
    applied = {"start": "2025-01-01T00:00:00+00:00", "end": "2025-04-01T00:00:00+00:00"}
    result = KPIResult(
        kpi_id="WS3-BI-007",
        value=3394.0,
        status=KPIStatus.UNKNOWN,
        error=None,
        window_requested=applied,
        window_applied=applied,
        window_status="applied",
    )
    resp = _kpi_result_to_response(kpi, result, brand="Kisqali", region="West")
    assert resp["success"] is True
    assert resp["brand"] == "Kisqali"
    assert resp["region"] == "West"
    assert resp["window_status"] == "applied"
    assert resp["window_applied"] == applied
    assert resp["window_requested"] == applied
    # Stale "rolling last 30 days" note must NOT contradict the applied window.
    assert "reporting_window" not in resp


@pytest.mark.unit
def test_kpi_result_to_response_not_applicable_window():
    """A KPI with no time dimension reports window_status='not_applicable' and,
    again, omits the fixed-window note."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-007")
    assert kpi is not None
    result = KPIResult(
        kpi_id="WS3-BI-007",
        value=1.5,
        status=KPIStatus.UNKNOWN,
        window_requested={"start": "2025-01-01T00:00:00+00:00", "end": "2025-04-01T00:00:00+00:00"},
        window_applied=None,
        window_status="not_applicable",
    )
    resp = _kpi_result_to_response(kpi, result)
    assert resp["window_status"] == "not_applicable"
    assert resp["window_applied"] is None
    assert "reporting_window" not in resp


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_bad_window_fails_without_calculator(monkeypatch):
    """An unparseable window is a user-input error: the tool fails fast with a
    hint and NEVER reaches the calculator (no DB)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    def _boom():  # pragma: no cover - must never be called
        raise AssertionError("calculator must not be constructed on a parse error")

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", _boom, raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "NBRx", "brand": "Kisqali", "window": "not a date"}
    )
    assert resp["success"] is False
    assert "hint" in resp
    assert resp["query_type"] == "kpi_calculate"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_passes_window_into_context(monkeypatch):
    """A valid window is parsed and threaded into the calculator context as a
    {'start','end'} dict alongside brand/territory."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["kpi_id"] = kpi_id
            captured["context"] = context
            return KPIResult(
                kpi_id=kpi_id,
                value=42.0,
                status=KPIStatus.UNKNOWN,
                window_status="applied",
                window_applied=context.get("window") if context else None,
            )

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "NBRx", "brand": "Kisqali", "window": "last 3 months"}
    )
    assert resp["success"] is True
    ctx = captured["context"]
    assert ctx["brand"] == "Kisqali"
    assert isinstance(ctx.get("window"), dict)
    assert set(ctx["window"].keys()) == {"start", "end"}
