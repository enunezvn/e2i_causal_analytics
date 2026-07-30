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
        # NBRx counts first-brand Rx over the registry's frontier-anchored
        # 30-day window (migration 089: ends at the latest prescription date,
        # NOT wall-clock now -- the substrate is calendar-fixed by design).
        # Disclosing it stops the chatbot from presenting the figure as
        # "the last 30 calendar days". Included ONLY because
        # window_status == "default".
        "reporting_window": "most recent 30 days of prescription data",
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
    # Volume KPIs disclose the real (frontier-anchored) reporting window.
    assert resp["reporting_window"] == "most recent 30 days of prescription data"


@pytest.mark.unit
def test_kpi_result_to_response_omits_window_for_unverified_kpi():
    """A KPI whose window we have NOT verified against the registry must NOT
    carry a fabricated ``reporting_window`` -- honest absence over a guessed
    period. ROI (WS3-BI-010) is a two-source probe (business_metrics /
    agent_activities frontiers diverge) so it stays out of the map."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-010")  # ROI
    assert kpi is not None
    result = KPIResult(kpi_id="WS3-BI-010", value=1.5, status=KPIStatus.UNKNOWN, metadata={})
    resp = _kpi_result_to_response(kpi, result)
    assert "reporting_window" not in resp


@pytest.mark.unit
def test_kpi_result_to_response_surfaces_data_through():
    """The frontier-anchored rows (089) return a ``data_through`` column; the
    calculator stashes it in metadata context and the chatbot response must
    surface it so the answer cites the real as-of date."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-007")
    assert kpi is not None
    result = KPIResult(
        kpi_id="WS3-BI-007",
        value=9.0,
        status=KPIStatus.UNKNOWN,
        metadata={
            "include_synthetic": True,
            "context": {"brand": "Kisqali", "data_through": "2025-04-23"},
        },
    )
    resp = _kpi_result_to_response(kpi, result, brand="Kisqali")
    assert resp["data_through"] == "2025-04-23"


@pytest.mark.unit
def test_kpi_result_to_response_omits_data_through_when_absent():
    """No data_through in the engine metadata (windowed variants, pre-089
    rows) -> no key. Honest absence, never a fabricated date."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-007")
    assert kpi is not None
    result = KPIResult(
        kpi_id="WS3-BI-007",
        value=9.0,
        status=KPIStatus.UNKNOWN,
        metadata={"context": {"brand": "Kisqali"}},
    )
    resp = _kpi_result_to_response(kpi, result, brand="Kisqali")
    assert "data_through" not in resp


@pytest.mark.unit
def test_reporting_window_covers_frontier_anchored_ws3_family():
    """TRx share and conversion rate were frontier-anchored by 089 alongside
    the volumes, so their windows are now verified and must be disclosed --
    with the DOMAIN the frontier belongs to (share: prescriptions; conversion:
    triggers). #1360 added the trigger-effectiveness family (089/113/118):
    acceptance/override/funnel disclose the trigger-data frontier window;
    precision's is the LAGGED matured cohort (migration 113), so its note must
    say the window ends BEFORE the frontier, not at it."""
    from src.api.routes.chatbot_tools import KPI_REPORTING_WINDOWS

    assert KPI_REPORTING_WINDOWS == {
        "WS3-BI-005": "most recent 30 days of prescription data",
        "WS3-BI-006": "most recent 30 days of prescription data",
        "WS3-BI-007": "most recent 30 days of prescription data",
        "WS3-BI-008": "most recent 30 days of prescription data",
        "WS3-BI-009": "most recent 30 days of trigger data",
        "WS2-TR-001": (
            "30-day trigger cohort ending 30 days before the trigger-data "
            "frontier (the conversion window must mature)"
        ),
        "WS2-TR-004": "most recent 30 days of trigger data",
        "WS2-TR-006": "most recent 30 days of trigger data",
        "WS2-TR-009": "most recent 30 days of trigger data",
    }


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
    {'start','end'} dict alongside brand/region."""
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_passes_region_into_context(monkeypatch):
    """A region filter must reach the calculator under the key it actually reads
    -- ``context['region']`` (business_impact/trigger_performance/data_quality).
    A dead 'territory' key would silently drop the filter, running the
    region-agnostic query while the response still echoed the region."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["context"] = context
            return KPIResult(kpi_id=kpi_id, value=7.0, status=KPIStatus.UNKNOWN)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "NBRx", "brand": "Kisqali", "region": "northeast"}
    )
    assert resp["success"] is True
    ctx = captured["context"]
    assert ctx["region"] == "northeast"
    assert "territory" not in ctx  # the dead key must be gone
    assert resp["region"] == "northeast"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_passes_segment_into_context(monkeypatch):
    """A severity-tier filter must reach the calculator under ``context['segment']``
    (migration 105 -- BusinessImpactCalculator._resolve_windowed_call routes on it)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["context"] = context
            return KPIResult(kpi_id=kpi_id, value=855.0, status=KPIStatus.UNKNOWN)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "NRx", "brand": "Remibrutinib", "segment": "low_severity"}
    )
    assert resp["success"] is True
    ctx = captured["context"]
    assert ctx["segment"] == "low_severity"
    assert "therapy_line" not in ctx


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_passes_therapy_line_into_context(monkeypatch):
    """A line-of-therapy filter must reach the calculator under
    ``context['therapy_line']`` (migration 105). Line 0 is a real, commonly-populated
    bucket -- the tool threads it with a truthy check on the (non-empty) string, so
    "0" is included, mirroring how the base compute core guards with ``is not None``."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["context"] = context
            return KPIResult(kpi_id=kpi_id, value=822.0, status=KPIStatus.UNKNOWN)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "NRx", "brand": "Remibrutinib", "therapy_line": "0"}
    )
    assert resp["success"] is True
    ctx = captured["context"]
    assert ctx["therapy_line"] == "0"
    assert "segment" not in ctx


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_passes_biologic_into_context(monkeypatch):
    """A biologic-status filter must reach the calculator under
    ``context['biologic']`` (migration 108 -- Remibrutinib-only axis)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["context"] = context
            return KPIResult(kpi_id=kpi_id, value=1258.0, status=KPIStatus.UNKNOWN)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "NRx", "brand": "Remibrutinib", "biologic": "experienced"}
    )
    assert resp["success"] is True
    ctx = captured["context"]
    assert ctx["biologic"] == "experienced"
    assert "ige_tier" not in ctx and "segment" not in ctx


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_passes_ige_tier_into_context(monkeypatch):
    """An IgE-tertile filter must reach the calculator under
    ``context['ige_tier']`` (migration 108 -- Remibrutinib-only axis)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["context"] = context
            return KPIResult(kpi_id=kpi_id, value=1060.0, status=KPIStatus.UNKNOWN)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "NRx", "brand": "Remibrutinib", "ige_tier": "low"}
    )
    assert resp["success"] is True
    ctx = captured["context"]
    assert ctx["ige_tier"] == "low"
    assert "biologic" not in ctx and "segment" not in ctx


@pytest.mark.unit
def test_kpi_result_to_response_attaches_share_semantic_note():
    """WS3-BI-008 responses must carry the honest share basis: the denominator
    is the tracked portfolio's prescriptions (Fabhalta/Kisqali/Remibrutinib),
    NOT an external market — the 2026-07-18 session review caught the chatbot
    attributing the share complement to Xolair/Dupixent, which are not in the
    data model at all."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-008")
    assert kpi is not None
    result = KPIResult(kpi_id="WS3-BI-008", value=0.3338, status=KPIStatus.GOOD, metadata={})
    resp = _kpi_result_to_response(kpi, result, brand="Remibrutinib")
    assert "tracked portfolio" in resp["semantic_note"]
    assert "Xolair" in resp["semantic_note"]


@pytest.mark.unit
def test_kpi_result_to_response_no_semantic_note_for_other_kpis():
    """The note is WS3-BI-008-specific; other KPIs must not carry it."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-005")
    assert kpi is not None
    result = KPIResult(kpi_id="WS3-BI-005", value=10.0, status=KPIStatus.UNKNOWN, metadata={})
    resp = _kpi_result_to_response(kpi, result)
    assert "semantic_note" not in resp


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_segment_window_reach_conversion_context(monkeypatch):
    """A per-segment windowed conversion ask must thread segment AND window into
    the calculator context (the session_1784387374342 gap: the tool accepted the
    params but the calculator dropped them; the routing itself is covered by
    tests/unit/test_kpi/test_conversion_rate_routing.py)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["kpi_id"] = kpi_id
            captured["context"] = context
            return KPIResult(kpi_id=kpi_id, value=0.6779, status=KPIStatus.GOOD)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {
            "kpi_name": "conversion rate",
            "brand": "Remibrutinib",
            "segment": "high_severity",
            "window": "2025-07-14 to 2026-07-13",
        }
    )
    assert resp["success"] is True
    assert captured["kpi_id"] == "WS3-BI-009"
    ctx = captured["context"]
    assert ctx["brand"] == "Remibrutinib"
    assert ctx["segment"] == "high_severity"
    assert "window" in ctx and ctx["window"]["start"] and ctx["window"]["end"]
