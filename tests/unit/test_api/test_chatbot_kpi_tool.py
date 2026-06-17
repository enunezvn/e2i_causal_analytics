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
    resp = _kpi_result_to_response(kpi, result)
    assert resp == {
        "success": True,
        "query_type": "kpi_calculate",
        "kpi_id": "WS3-BI-007",
        "kpi_name": kpi.name,
        "value": 3298.0,
        "status": "unknown",  # KPIResult uses_enum_values -> status is the str value
        "data_source": "synthetic",
        # NBRx counts first-brand Rx over the registry's fixed rolling 30-day
        # window (migrations 044/066); disclosing it stops the chatbot from
        # presenting a 30-day figure as a user-requested "past 3 months".
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
