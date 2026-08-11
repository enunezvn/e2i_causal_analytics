"""#1538: truthful region provenance on KPIResult.

Region-scoped CURRENT values exist for a fixed set of calculators (migrations
077/078/113/118/125). Every OTHER calculator silently keeps its global or
portfolio value when the context carries a region — so a consumer (the chat
chart tools, the copilot response echo) could caption a global figure as
regional. The fix mirrors the window-provenance idiom:

* each routing seam stamps ``context["_region_routed"] = True`` at the exact
  point it selects a region-scoped query variant (truth by construction — no
  static capability registry to drift), and
* ``KPICalculator.calculate()`` stamps ``region_requested`` /
  ``region_applied`` / ``region_status`` ("default" | "applied" |
  "not_applicable") onto every result, exactly like ``_stamp_window``.

Cache: the serializer hand-picks fields, so the three region fields must
round-trip, and a region-keyed cache hit whose entry PREDATES this feature
(``region_status == "default"`` while a region was requested) must be
recomputed, never served as-is.
"""

from typing import Any

import pytest

from src.kpi.calculator import KPICalculator
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.calculators.data_quality import DataQualityCalculator
from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


def _kpi(kpi_id: str = "WS3-BI-005") -> KPIMetadata:
    return KPIMetadata(
        id=kpi_id,
        name=f"name-{kpi_id}",
        definition="d",
        formula="f",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS3_BUSINESS,
    )


# ---- model fields -----------------------------------------------------------


@pytest.mark.unit
def test_kpiresult_region_fields_default():
    r = KPIResult(kpi_id="WS3-BI-005", value=1.0, status=KPIStatus.UNKNOWN)
    assert r.region_requested is None
    assert r.region_applied is None
    assert r.region_status == "default"


# ---- pure helper ------------------------------------------------------------


@pytest.mark.unit
def test_stamp_region_no_region_leaves_defaults():
    r = KPIResult(kpi_id="WS3-BI-005", value=1.0, status=KPIStatus.UNKNOWN)
    out = KPICalculator._stamp_region(r, {"brand": "Kisqali"})
    assert out.region_status == "default"
    assert out.region_requested is None
    assert out.region_applied is None


@pytest.mark.unit
def test_stamp_region_applied_when_seam_routed():
    r = KPIResult(kpi_id="WS3-BI-005", value=249.0, status=KPIStatus.UNKNOWN)
    out = KPICalculator._stamp_region(
        r, {"brand": "Kisqali", "region": "northeast", "_region_routed": True}
    )
    assert out.region_status == "applied"
    assert out.region_requested == "northeast"
    assert out.region_applied == "northeast"
    assert out.value == 249.0  # value untouched


@pytest.mark.unit
def test_stamp_region_not_applicable_keeps_value():
    """A calculator with no region variant keeps its global value — the stamp
    says so honestly instead of letting the caller caption it as regional."""
    r = KPIResult(kpi_id="WS1-MP-001", value=0.87, status=KPIStatus.GOOD)
    out = KPICalculator._stamp_region(r, {"region": "northeast"})
    assert out.region_status == "not_applicable"
    assert out.region_requested == "northeast"
    assert out.region_applied is None
    assert out.value == 0.87  # still the (global) computed value


# ---- wired through calculate() ----------------------------------------------


class _AlwaysMissCache:
    enabled = True

    def get(self, kpi_id, **context):
        return None

    def set(self, result, ttl=None, **context):
        return True


class _StubRegistry:
    def __init__(self, kpi: KPIMetadata):
        self._kpi = kpi

    def get(self, kpi_id: str) -> KPIMetadata | None:
        return self._kpi if kpi_id == self._kpi.id else None


def _wired_calc(kpi: KPIMetadata, value: float, *, routes_region: bool) -> KPICalculator:
    calc = KPICalculator(registry=_StubRegistry(kpi), cache=_AlwaysMissCache())

    def _fake_calculate(k, ctx):
        if routes_region and ctx.get("region"):
            ctx["_region_routed"] = True
        return KPIResult(kpi_id=k.id, value=value, status=KPIStatus.GOOD, cached=False)

    calc._calculate_kpi = _fake_calculate  # type: ignore[method-assign]
    return calc


@pytest.mark.unit
def test_calculate_stamps_applied_for_region_routing_calculator():
    calc = _wired_calc(_kpi("WS3-BI-005"), 249.0, routes_region=True)
    res = calc.calculate("WS3-BI-005", context={"brand": "Kisqali", "region": "northeast"})
    assert res.region_status == "applied"
    assert res.region_applied == "northeast"
    assert res.value == 249.0


@pytest.mark.unit
def test_calculate_stamps_not_applicable_for_global_only_calculator():
    calc = _wired_calc(_kpi("WS1-MP-001"), 0.87, routes_region=False)
    res = calc.calculate("WS1-MP-001", context={"region": "northeast"})
    assert res.region_status == "not_applicable"
    assert res.region_requested == "northeast"
    assert res.region_applied is None
    assert res.value == 0.87


@pytest.mark.unit
def test_calculate_no_region_is_default():
    calc = _wired_calc(_kpi("WS3-BI-005"), 1.0, routes_region=True)
    res = calc.calculate("WS3-BI-005", context={"brand": "Kisqali"})
    assert res.region_status == "default"
    assert res.region_requested is None


@pytest.mark.unit
def test_calculate_error_result_still_echoes_requested_region():
    """Errored results echo what was asked; consumers read `error` first."""
    kpi = _kpi("WS3-BI-009")
    calc = KPICalculator(registry=_StubRegistry(kpi), cache=_AlwaysMissCache())
    calc._calculate_kpi = lambda k, ctx: KPIResult(  # type: ignore[method-assign]
        kpi_id=k.id, value=None, status=KPIStatus.UNKNOWN, error="boom"
    )
    res = calc.calculate("WS3-BI-009", context={"brand": "Kisqali", "region": "northeast"})
    assert res.error == "boom"
    assert res.region_requested == "northeast"
    assert res.region_applied is None


# ---- routing seams stamp at the decision point ------------------------------


class _Resp:
    def __init__(self, data):
        self.data = data


class _Exec:
    def __init__(self, data):
        self._d = data

    def execute(self):
        return _Resp(self._d)


class _StubClient:
    def __init__(self, row):
        self.row = row
        self.calls: list[dict[str, Any]] = []

    def rpc(self, name, payload):
        self.calls.append(payload)
        return _Exec([self.row])


@pytest.mark.unit
def test_trx_region_routes_and_stamps():
    client = _StubClient({"trx": 249})
    calc = BusinessImpactCalculator(db_client=client)
    ctx = {"brand": "Kisqali", "region": "northeast"}
    value = calc._calc_trx(ctx)
    assert value == 249.0
    assert client.calls[0]["query_id"] == "business_impact_trx_region"
    assert client.calls[0]["params"] == ["Kisqali", "northeast"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_trx_windowed_region_routes_and_stamps():
    client = _StubClient({"trx": 100})
    calc = BusinessImpactCalculator(db_client=client)
    ctx = {"brand": "Kisqali", "region": "northeast", "window": {"start": "S", "end": "E"}}
    calc._calc_trx(ctx)
    assert client.calls[0]["query_id"] == "business_impact_trx_windowed_region"
    assert client.calls[0]["params"] == ["Kisqali", "northeast", "S", "E"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_trx_segment_precedence_drops_region_without_stamp():
    """Axis precedence silently drops region (4-param cap) — the marker must
    NOT be set, so the stamp downstream says not_applicable, truthfully."""
    client = _StubClient({"trx": 80})
    calc = BusinessImpactCalculator(db_client=client)
    ctx = {"brand": "Kisqali", "region": "northeast", "segment": "high_severity"}
    calc._calc_trx(ctx)
    assert "_region" not in client.calls[0]["query_id"]
    assert ctx.get("_region_routed") is None


@pytest.mark.unit
def test_trx_no_region_no_stamp():
    client = _StubClient({"trx": 500})
    calc = BusinessImpactCalculator(db_client=client)
    ctx = {"brand": "Kisqali"}
    calc._calc_trx(ctx)
    assert ctx.get("_region_routed") is None


@pytest.mark.unit
def test_conversion_rate_region_alone_stamps():
    client = _StubClient({"conversion_rate": 0.31})
    calc = BusinessImpactCalculator(db_client=client)
    ctx = {"region": "northeast"}
    value = calc._calc_conversion_rate(ctx)
    assert value == 0.31
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_region"
    assert client.calls[0]["params"] == ["northeast"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_roi_region_scoped_read_stamps():
    client = _StubClient({"avg_roi": 3.2})
    calc = BusinessImpactCalculator(db_client=client)
    ctx = {"brand": "Kisqali", "region": "northeast"}
    value = calc._calc_roi(ctx)
    assert value == 3.2
    assert client.calls[0]["query_id"] == "business_impact_roi_business_metrics_scoped"
    assert client.calls[0]["params"] == ["Kisqali", "northeast"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_roi_no_region_no_stamp():
    client = _StubClient({"avg_roi": 3.2})
    calc = BusinessImpactCalculator(db_client=client)
    ctx: dict[str, Any] = {}
    calc._calc_roi(ctx)
    assert ctx.get("_region_routed") is None


@pytest.mark.unit
def test_data_quality_region_scoped_stamps():
    ctx = {"region": "midwest"}
    query_id, params = DataQualityCalculator._region_scoped(
        "data_quality_source_coverage_patients", ctx, ["Kisqali"]
    )
    assert query_id == "data_quality_source_coverage_patients_region"
    assert params == ["midwest"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_data_quality_no_region_no_stamp():
    ctx = {"brand": "Kisqali"}
    query_id, params = DataQualityCalculator._region_scoped(
        "data_quality_source_coverage_patients", ctx, ["Kisqali"]
    )
    assert query_id == "data_quality_source_coverage_patients"
    assert ctx.get("_region_routed") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "context,expected_query,expected_params",
    [
        (
            {"region": "west"},
            "trigger_performance_precision_region",
            ["west"],
        ),
        (
            {"brand": "Kisqali", "region": "west"},
            "trigger_performance_precision_brand_region",
            ["Kisqali", "west"],
        ),
    ],
)
def test_trigger_scoped_region_stamps(context, expected_query, expected_params):
    query_id, params = TriggerPerformanceCalculator._scoped(
        "trigger_performance_precision", context
    )
    assert query_id == expected_query
    assert params == expected_params
    assert context.get("_region_routed") is True


@pytest.mark.unit
def test_trigger_scoped_brand_only_no_stamp():
    ctx = {"brand": "Kisqali"}
    TriggerPerformanceCalculator._scoped("trigger_performance_precision", ctx)
    assert ctx.get("_region_routed") is None


@pytest.mark.unit
def test_trigger_effectiveness_region_stamps_plain_and_windowed():
    plain_ctx = {"brand": "Kisqali", "region": "south", "trigger_type": None}
    query_id, params = TriggerPerformanceCalculator._effectiveness_scoped(
        "acceptance_rate", plain_ctx
    )
    assert params[1] == "south"
    assert plain_ctx.get("_region_routed") is True

    windowed_ctx = {
        "brand": "Kisqali",
        "region": "south",
        "window": {"start": "S", "end": "E"},
    }
    query_id, params = TriggerPerformanceCalculator._effectiveness_scoped(
        "acceptance_rate", windowed_ctx
    )
    assert "windowed_region" in query_id
    assert windowed_ctx.get("_region_routed") is True


@pytest.mark.unit
def test_trigger_effectiveness_no_region_no_stamp():
    ctx = {"brand": "Kisqali", "trigger_type": "adherence_risk"}
    TriggerPerformanceCalculator._effectiveness_scoped("acceptance_rate", ctx)
    assert ctx.get("_region_routed") is None


# ---- cache round-trip + stale-entry guard -----------------------------------


class _FakeRedis:
    """Minimal in-memory stand-in implementing the two ops KPICache uses."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True


@pytest.mark.unit
def test_cache_round_trips_region_provenance():
    from src.kpi.cache import KPICache

    cache = KPICache(redis_url=None)
    cache._redis = _FakeRedis()  # type: ignore[assignment]
    cache._enabled = True
    result = KPIResult(
        kpi_id="WS3-BI-005",
        value=249.0,
        status=KPIStatus.UNKNOWN,
        region_requested="northeast",
        region_applied="northeast",
        region_status="applied",
    )
    assert cache.set(result, ttl=60, brand="Kisqali", region="northeast")
    cached = cache.get("WS3-BI-005", brand="Kisqali", region="northeast")
    assert cached is not None
    assert cached.region_requested == "northeast"
    assert cached.region_applied == "northeast"
    assert cached.region_status == "applied"


class _StaleEntryCache:
    """Serves a PRE-#1538 entry: region-keyed, but no region provenance."""

    enabled = True

    def __init__(self):
        self.gets = 0

    def get(self, kpi_id, **context):
        self.gets += 1
        # What a pre-feature serialized entry deserializes to: defaults.
        return KPIResult(kpi_id=kpi_id, value=999.0, status=KPIStatus.UNKNOWN, cached=True)

    def set(self, result, ttl=None, **context):
        return True


@pytest.mark.unit
def test_region_request_ignores_pre_feature_cache_entry():
    """A cached entry that cannot attest its region provenance must be
    recomputed — serving it would put an unattested value under a region ask."""
    kpi = _kpi("WS3-BI-005")
    calc = KPICalculator(registry=_StubRegistry(kpi), cache=_StaleEntryCache())
    recomputed: dict[str, Any] = {}

    def _fake_calculate(k, ctx):
        recomputed["yes"] = True
        ctx["_region_routed"] = True
        return KPIResult(kpi_id=k.id, value=249.0, status=KPIStatus.GOOD)

    calc._calculate_kpi = _fake_calculate  # type: ignore[method-assign]
    res = calc.calculate("WS3-BI-005", context={"brand": "Kisqali", "region": "northeast"})
    assert recomputed.get("yes") is True
    assert res.value == 249.0
    assert res.region_status == "applied"


@pytest.mark.unit
def test_no_region_request_still_serves_cache():
    """The guard is region-scoped only: global asks keep their cache hits."""
    kpi = _kpi("WS3-BI-005")
    cache = _StaleEntryCache()
    calc = KPICalculator(registry=_StubRegistry(kpi), cache=cache)
    calc._calculate_kpi = lambda k, ctx: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("must not recompute on a cache hit")
    )
    res = calc.calculate("WS3-BI-005", context={"brand": "Kisqali"})
    assert res.value == 999.0
    assert cache.gets == 1


# ---- REST response mapping ---------------------------------------------------


@pytest.mark.unit
def test_result_to_response_carries_region_provenance():
    from src.api.routes.kpi import _result_to_response

    result = KPIResult(
        kpi_id="WS3-BI-005",
        value=249.0,
        status=KPIStatus.UNKNOWN,
        region_requested="northeast",
        region_applied="northeast",
        region_status="applied",
    )
    resp = _result_to_response(result)
    assert resp.region_requested == "northeast"
    assert resp.region_applied == "northeast"
    assert resp.region_status == "applied"


@pytest.mark.unit
def test_result_to_response_region_defaults():
    from src.api.routes.kpi import _result_to_response

    result = KPIResult(kpi_id="WS3-BI-005", value=1.0, status=KPIStatus.UNKNOWN)
    resp = _result_to_response(result)
    assert resp.region_requested is None
    assert resp.region_applied is None
    assert resp.region_status == "default"
