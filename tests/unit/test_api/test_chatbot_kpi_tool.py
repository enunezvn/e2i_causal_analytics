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
    from src.api.routes.chatbot_tools import _kpi_result_to_response, _measure_basis_for_kpi

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
        # #1640: what the figure MEASURES, derived from KPIMetadata.tables, so an
        # event-ledger count is never silently compared with a business_metrics
        # level. WS3-BI-007 (NBRx) counts first-brand Rx over treatment_events.
        "measure_basis": _measure_basis_for_kpi(kpi),
        "brand": "Kisqali",
        "region": None,
        # #1538: no region requested -> provenance defaults.
        "region_status": "default",
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
        "WS2-TR-005": "most recent 30 days of trigger data",
        "WS2-TR-006": "most recent 30 days of trigger data",
        "WS2-TR-009": "most recent 30 days of trigger data",
    }


@pytest.mark.unit
def test_ws2_tr005_payload_carries_window_metadata_1713():
    """#1713 platform half: WS2-TR-005 (False Alert Rate) must disclose the
    same window metadata as its sibling trigger-substrate KPI WS2-TR-006
    (Override Rate). The 2026-08-19 full eval (turn 4.6) certified the
    asymmetry -- Override Rate's payload carried ``reporting_window`` and
    ``data_through`` while False Alert Rate's carried neither -- as the
    substrate of a recurring defect: Override's window label was asserted in
    prose for a metric whose payload declared no window. Migration 089
    registers WS2-TR-005 with the exact same frontier-anchored 30-day trigger
    window and ``data_through`` output column as WS2-TR-006, so the disclosure
    is registry-verified, not guessed."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS2-TR-005")
    assert kpi is not None
    result = KPIResult(
        kpi_id="WS2-TR-005",
        value=0.083,
        status=KPIStatus.WARNING,
        metadata={"context": {"data_through": "2026-08-17"}, "lower_is_better": True},
    )
    resp = _kpi_result_to_response(kpi, result)
    assert resp["reporting_window"] == "most recent 30 days of trigger data"
    assert resp["data_through"] == "2026-08-17"


@pytest.mark.unit
def test_kpi_result_to_response_surfaces_direction_1713():
    """Direction glosses ("above/below threshold") are only checkable when the
    payload names the metric's polarity. The 2026-08-19 eval turn 4.6 wrote
    "flagged warning (below healthy threshold)" for WS2-TR-005 -- inverted: it
    is lower-is-better, so warning means ABOVE its threshold. The calculator
    already evaluates status with a ``lower_is_better`` flag and stashes it in
    ``KPIResult.metadata``; the response must surface that same flag (the
    polarity the status was actually computed WITH) as ``direction``."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi_005 = get_registry().get("WS2-TR-005")
    assert kpi_005 is not None
    result = KPIResult(
        kpi_id="WS2-TR-005",
        value=0.083,
        status=KPIStatus.WARNING,
        metadata={"context": {}, "lower_is_better": True},
    )
    resp = _kpi_result_to_response(kpi_005, result)
    assert resp["direction"] == "lower_is_better"

    kpi_004 = get_registry().get("WS2-TR-004")
    assert kpi_004 is not None
    result = KPIResult(
        kpi_id="WS2-TR-004",
        value=0.55,
        status=KPIStatus.GOOD,
        metadata={"context": {}, "lower_is_better": False},
    )
    resp = _kpi_result_to_response(kpi_004, result)
    assert resp["direction"] == "higher_is_better"


@pytest.mark.unit
def test_kpi_result_to_response_omits_direction_when_polarity_unknown_1713():
    """A calculator that did not stash ``lower_is_better`` (e.g. the WS3
    business-impact family) must yield NO ``direction`` field -- honest
    absence over asserting a polarity the status evaluation never used."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-007")
    assert kpi is not None
    result = KPIResult(kpi_id="WS3-BI-007", value=9.0, status=KPIStatus.UNKNOWN, metadata={})
    resp = _kpi_result_to_response(kpi, result)
    assert "direction" not in resp


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
        # #1538: the region echo is provenance-based — the engine attests the
        # scoped variant ran, and the echo carries the APPLIED enum label
        # (lowercase), not the caller's display-form argument.
        region_requested="west",
        region_applied="west",
        region_status="applied",
    )
    resp = _kpi_result_to_response(kpi, result, brand="Kisqali", region="west")
    assert resp["success"] is True
    assert resp["brand"] == "Kisqali"
    assert resp["region"] == "west"
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
            # #1538: mirror the real engine, which stamps region provenance
            # when the routing seam selects a region variant — the tool's echo
            # is provenance-based, not a blind copy of the argument.
            return KPIResult(
                kpi_id=kpi_id,
                value=7.0,
                status=KPIStatus.UNKNOWN,
                region_requested="northeast",
                region_applied="northeast",
                region_status="applied",
            )

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "NBRx", "brand": "Kisqali", "region": "northeast"}
    )
    assert resp["success"] is True
    ctx = captured["context"]
    assert ctx["region"] == "northeast"
    assert "territory" not in ctx  # the dead key must be gone
    assert resp["region"] == "northeast"
    assert resp["region_status"] == "applied"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_normalizes_brand_casing(monkeypatch):
    """The input schema promises case-insensitive brands (codex #1534 iter-1
    finding 1): 'kisqali' must reach the calculator as the real enum label
    'Kisqali' — brand/region are enum columns, and #1534's scoped ROI query
    (migration 125) matches ``brand::text = $1`` exactly, so a raw lowercase
    brand would 0-row the scoped query and fail loud on a resolvable ask.
    The response echo must carry the RESOLVED label (truthful echo)."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    captured: dict = {}

    class _FakeCalc:
        def calculate(self, kpi_id, context=None):
            captured["context"] = context
            return KPIResult(kpi_id=kpi_id, value=1.9, status=KPIStatus.UNKNOWN)

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: _FakeCalc(), raising=False)

    resp = await kpi_calculate_tool.ainvoke({"kpi_name": "ROI", "brand": "kisqali"})
    assert resp["success"] is True
    assert captured["context"]["brand"] == "Kisqali"
    assert resp["brand"] == "Kisqali"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kpi_calculate_tool_unknown_brand_fails_before_calculator(monkeypatch):
    """An unmappable brand can never match an enum row — fail fast with the
    known-brand list BEFORE touching the calculator (the ``_query_kpis``
    #1501 precedent), never a misleading 'no rows for that scope' error."""
    import src.api.routes.kpi as kpi_route
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    def _boom():
        raise AssertionError("calculator must not be constructed for an unknown brand")

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", _boom, raising=False)

    resp = await kpi_calculate_tool.ainvoke({"kpi_name": "ROI", "brand": "Acme"})
    assert resp["success"] is False
    assert "Acme" in resp["error"]
    assert "Kisqali" in resp["error"]  # the known-brand list guides the retry


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


@pytest.mark.unit
def test_kpi_result_to_response_surfaces_roi_temporal_band():
    """#1532: WS3-BI-010 stashes ``temporal_variability_band`` into the
    calculator context (the ``funnel_stages`` seam, #1360) and the mapper must
    copy it up so the synthesizer can present the per-slice 12-month range."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-010")
    assert kpi is not None
    band = {
        "semantics": "range of monthly values over the trailing 12 months — temporal variability",
        "window": "trailing 12 months ending at the business_metrics data frontier",
        "min_n": 6,
        "slices": [
            {
                "metric_name": "market_share",
                "brand": "Kisqali",
                "region": "northeast",
                "n": 12,
                "band": {"roi_min": 1.1, "roi_max": 1.9, "roi_mean": 1.5, "roi_stddev": 0.2},
                "band_suppressed": False,
            }
        ],
    }
    result = KPIResult(
        kpi_id="WS3-BI-010",
        value=1.52,
        status=KPIStatus.INFORMATIONAL,
        metadata={"context": {"temporal_variability_band": band}},
    )
    resp = _kpi_result_to_response(kpi, result)
    assert resp["success"] is True
    assert resp["temporal_variability_band"] == band


@pytest.mark.unit
def test_kpi_result_to_response_band_only_for_roi_kpi():
    """Codex iter-1 finding 2 (2026-08-10): the band is an ROI-only estimand.
    A non-ROI result whose metadata context carries the key (e.g. a cached
    entry polluted by the pre-fix shared-context leak) must NOT present an
    ROI band beside a TRx/NRx figure — the mapper gates on the KPI id."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-005")
    assert kpi is not None
    result = KPIResult(
        kpi_id="WS3-BI-005",
        value=125000.0,
        status=KPIStatus.INFORMATIONAL,
        metadata={"context": {"temporal_variability_band": {"min_n": 6, "slices": []}}},
    )
    resp = _kpi_result_to_response(kpi, result)
    assert "temporal_variability_band" not in resp


@pytest.mark.unit
def test_kpi_result_to_response_band_absent_when_not_stashed():
    """Honest absence: no band in the calculator context (agent_activities
    fallback answered, band query failed, or real-mode zero slices) -> no
    band key in the response. Applies to every other KPI too."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    for kpi_id, value in (("WS3-BI-010", 1.52), ("WS3-BI-005", 10.0)):
        kpi = get_registry().get(kpi_id)
        assert kpi is not None
        result = KPIResult(kpi_id=kpi_id, value=value, status=KPIStatus.UNKNOWN, metadata={})
        resp = _kpi_result_to_response(kpi, result)
        assert "temporal_variability_band" not in resp


@pytest.mark.unit
def test_roi_response_carries_no_interval_keys_regression_1527():
    """#1527 regression, executable: monthly data gives n=1 per slice in the
    30-day headline window, so NO conditioning scheme can produce an interval
    there — the ROI response must never grow confidence_interval / ci_lower /
    ci_upper keys, band present or not (the band is a DIFFERENT estimand)."""
    import json as _json

    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-010")
    assert kpi is not None
    band = {
        "semantics": "range of monthly values over the trailing 12 months — temporal variability",
        "window": "trailing 12 months ending at the business_metrics data frontier",
        "min_n": 6,
        "slices": [],
    }
    result = KPIResult(
        kpi_id="WS3-BI-010",
        value=1.52,
        status=KPIStatus.INFORMATIONAL,
        metadata={"context": {"temporal_variability_band": band, "data_through": "2026-08-03"}},
    )
    resp = _kpi_result_to_response(kpi, result)
    dumped = _json.dumps(resp).lower()
    for forbidden in ("confidence_interval", "ci_lower", "ci_upper"):
        assert forbidden not in dumped, f"ROI response contains forbidden key {forbidden!r}"


@pytest.mark.unit
def test_roi_semantic_note_pins_band_meaning():
    """#1532 acceptance criterion 2: the WS3-BI-010 semantic note must state
    the band is the range of monthly values over the past 12 months and must
    not describe it as a confidence interval."""
    from src.api.routes.chatbot_tools import _kpi_result_to_response

    kpi = get_registry().get("WS3-BI-010")
    assert kpi is not None
    result = KPIResult(kpi_id="WS3-BI-010", value=1.52, status=KPIStatus.INFORMATIONAL, metadata={})
    resp = _kpi_result_to_response(kpi, result)
    note = resp.get("semantic_note", "")
    assert "12 months" in note
    assert "point estimate" in note
    assert "confidence interval" not in note.lower().replace("not a confidence interval", "")
