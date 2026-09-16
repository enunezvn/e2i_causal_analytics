"""kpi_calculate_tool patient-axis allowlist (#1911; #1901 item 4j).

The tool accepts ``segment`` / ``therapy_line`` / ``biologic`` / ``ige_tier``
for EVERY KPI, but only ``BusinessImpactCalculator`` (WS3-BI-005..009) and
``CausalMetricsCalculator`` (CM-002, ``segment`` only) read them. For any other
KPI the key was written into the calculator context and ignored: the query ran
brand-global, no provenance marker exists for the patient axes (``calculator.py``
stamps region only) and the answer read as segment-scoped -- "Kisqali trigger
precision for high-severity patients" came back as the brand-global figure with
nothing telling the model the tier was not applied.

The gate refuses those combinations BEFORE any calculator is built (the #1360
``trigger_type`` precedent), naming the KPIs that do serve the axis. Pure unit
coverage, no DB: the fakes below stand in for ``get_kpi_calculator``; the
allowlist itself is pinned to the real calculators' query binding with a
recording client so it cannot drift from the code.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from src.kpi.models import KPIResult, KPIStatus
from src.kpi.registry import get_registry

# One probe value per axis. Distinctive enough to be recognised inside a bound
# parameter list (the derivation test looks for the VALUE, not a query id).
_AXIS_PROBE: Dict[str, str] = {
    "segment": "high_severity",
    "therapy_line": "3",
    "biologic": "naive",
    "ige_tier": "high",
}

# The human-readable spec: the KPIs whose calculator BINDS each axis to a query.
# Kept literal here on purpose (a reviewer can read it) and pinned three ways
# below -- to the tool's constant and to the calculators' actual binding.
# TRx Share is NOT served on any patient axis: every patient is on one tracked
# brand, so a per-bucket portfolio share mixes indications (and is always 100% on
# the Remibrutinib-only biologic/IgE axes) -- the 2026-09-16
# session_1789548670222_fcscf3u incident. The calculator refuses it too.
#
# ⚠ THE SHARE KPI AND THE VOLUME TRIO BOTH MOVED (#2114, owner #11). The lane makes
# business_metrics canonical and puts the patient panel on WS3-BI-011..014, so the
# KPIs whose calculators BIND a patient axis are the PANEL trio, not the canonical
# 005/006/007 -- `refuse_patient_axis` now turns those away. The share with no
# patient-axis breakdown is panel WS3-BI-014, and it is in none of the four
# measured sets, so owner #11 holds by MEASUREMENT here rather than by assertion.
_VOLUME = frozenset({"WS3-BI-011", "WS3-BI-012", "WS3-BI-013"})
_SERVED: Dict[str, frozenset] = {
    # _resolve_windowed_call (005..007) + _calc_conversion_rate (009, migration
    # 111) + _calc_cate (CM-002 binds ml_predictions.segment_assignment, same
    # low/medium/high_severity label space -- verified on the prod substrate).
    "segment": _VOLUME | {"WS3-BI-009", "CM-002"},
    "therapy_line": _VOLUME | {"WS3-BI-009"},
    # WS3-BI-009 REFUSES these two itself (triggers carry no biologic/IgE
    # dimension), so it does not bind them and stays outside the set.
    "biologic": _VOLUME,
    "ige_tier": _VOLUME,
}

_ALL_KPI_IDS: List[str] = [k.id for k in get_registry().get_all()]


class _RaisingCalc:
    """Any attribute access means the gate let the call through to a calculator."""

    def __init__(self, reached: Dict[str, Any]) -> None:
        self._reached = reached

    def __getattr__(self, name: str) -> Any:
        self._reached["attr"] = name
        raise AssertionError("calculator reached: the tool gate let the call through")


class _CapturingCalc:
    def __init__(self, captured: Dict[str, Any], value: float = 0.42) -> None:
        self._captured = captured
        self._value = value

    def calculate(self, kpi_id: str, context: Any = None) -> KPIResult:
        self._captured["kpi_id"] = kpi_id
        self._captured["context"] = context
        return KPIResult(kpi_id=kpi_id, value=self._value, status=KPIStatus.INFORMATIONAL)


class _NoQueries:
    """Sentinel client: any attribute access means a guard moved after its query."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"routing test must not touch the DB client ({name})")


class _RecordingClient:
    """Fake supabase client: records every kpi_query RPC payload, returns no rows."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def rpc(self, name: str, payload: Dict[str, Any]) -> Any:
        self.calls.append((name, dict(payload)))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))


def _install(monkeypatch: pytest.MonkeyPatch, calc: Any) -> None:
    import src.api.routes.kpi as kpi_route

    monkeypatch.setattr(kpi_route, "get_kpi_calculator", lambda: calc, raising=False)


# =============================================================================
# THE GATE: every registry KPI x every patient axis
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("axis", sorted(_AXIS_PROBE))
@pytest.mark.parametrize("kpi_id", _ALL_KPI_IDS)
async def test_patient_axis_is_served_or_refused_before_any_calculator(monkeypatch, axis, kpi_id):
    """Served KPI: the axis reaches ``context`` under the key the calculator
    reads (routing unchanged). Any other KPI: an honest error naming the axis,
    the KPI and the KPIs that serve the axis, and the calculator is NEVER
    constructed -- red on main for every unserved KPI (the fake is reached)."""
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    kpi = get_registry().get(kpi_id)
    assert kpi is not None
    value = _AXIS_PROBE[axis]
    # Remibrutinib so the calculator-side biologic/IgE brand guard is never the
    # reason for a refusal here (that guard is covered separately below).
    ask = {"kpi_name": kpi.name, "brand": "Remibrutinib", axis: value}

    if kpi_id in _SERVED[axis]:
        captured: Dict[str, Any] = {}
        _install(monkeypatch, _CapturingCalc(captured))
        resp = await kpi_calculate_tool.ainvoke(ask)
        assert resp["success"] is True, resp
        assert captured["kpi_id"] == kpi_id
        assert captured["context"][axis] == value
        assert captured["context"]["brand"] == "Remibrutinib"
        return

    reached: Dict[str, Any] = {}
    _install(monkeypatch, _RaisingCalc(reached))
    resp = await kpi_calculate_tool.ainvoke(ask)
    assert reached == {}, f"{kpi_id} with {axis}: the calculator was reached ({reached})"
    assert resp["success"] is False
    assert resp["query_type"] == "kpi_calculate"
    assert resp["kpi_id"] == kpi_id and resp["kpi_name"] == kpi.name
    assert axis in resp["error"] and kpi.name in resp["error"]
    for served_id in _SERVED[axis]:
        served = get_registry().get(served_id)
        assert served is not None
        assert served.name in resp["error"], (served_id, resp["error"])
    # #1565: a question / next step, not a dead end.
    assert kpi.name in resp["hint"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refusal_text_is_the_documented_contract(monkeypatch):
    """The motivating ask (#1911): trigger precision by severity tier."""
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    reached: Dict[str, Any] = {}
    _install(monkeypatch, _RaisingCalc(reached))
    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "trigger precision", "brand": "Kisqali", "segment": "high_severity"}
    )
    assert reached == {}
    assert resp["success"] is False and resp["kpi_id"] == "WS2-TR-001"
    assert resp["error"].startswith("segment (severity tier) applies only to ")
    assert resp["error"].endswith(", not Trigger Precision.")
    assert "Trigger Precision without the severity tier filter" in resp["hint"]
    assert "by severity tier" in resp["hint"]
    # The served list is REGISTRY order, so the volume KPIs lead and CATE closes.
    # #2114: the volume KPI that leads is now PANEL TRx — the canonical 005 no longer
    # binds a patient axis, so its name is absent from the served list and .index()
    # raised. The ORDERING PROPERTY is unchanged and still asserted; only the name of
    # the KPI that leads moved with the contract.
    assert resp["error"].index("Observed Rx Events - Patient Panel TRx (TRx Panel)") < resp[
        "error"
    ].index("Conditional ATE (CATE)")


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("axis", sorted(_AXIS_PROBE))
async def test_trx_share_by_patient_axis_refusal_points_to_the_within_brand_mix(monkeypatch, axis):
    """The motivating ask (2026-09-16): "Remibrutinib TRx Share by severity tier
    and biologic status". The refusal must say WHY and offer a working next step.

    ⚠ REWRITTEN TO OWNER DECISION #14 (#2114), not reverted to #2137's wording. This
    ask resolves to CANONICAL share WS3-BI-008, which after this lane reads
    business_metrics at brand x region x calendar month (measured:
    `registry.get("WS3-BI-008").tables == ['business_metrics']`). #2137's reason --
    "every patient is on exactly one tracked brand" -- is a statement about
    patient_journeys and is now LITERALLY FALSE of 008, so reusing it would make the
    refusal lie about the data it read. The reasoning followed the SUBSTRATE, not the
    id.

    The user's need did not move with the sentence, though: canonical TRx 005 no
    longer serves patient axes either, so this ask hit a dead stop where main gave a
    working redirect. Owner #14 gives 008 its OWN substrate-true reason and sends it
    to the SAME destination as panel share 014 -- panel TRx WS3-BI-011, the only TRx
    carrying a patient axis. Two reasons, one destination.

    The assertions are kept as strong as they were: it still refuses, the refusal
    still names a reason, and the hint still names the TRx to ask for."""
    from src.api.routes.chatbot_tools import _PATIENT_AXIS_LABELS, kpi_calculate_tool

    reached: Dict[str, Any] = {}
    _install(monkeypatch, _RaisingCalc(reached))
    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "TRx share", "brand": "Remibrutinib", axis: _AXIS_PROBE[axis]}
    )
    assert reached == {}
    assert resp["success"] is False and resp["kpi_id"] == "WS3-BI-008"
    label = _PATIENT_AXIS_LABELS[axis]
    # 008's OWN reason — true of business_metrics, not the panel sentence.
    assert "no patient dimension" in resp["error"], resp["error"]
    assert "one tracked brand" not in resp["error"], "008 must not reuse the panel reason"
    # ...and the same destination as 014: panel TRx, the only TRx carrying an axis.
    assert f"Observed Rx Events - Patient Panel TRx (TRx Panel) by {label}" in resp["hint"]
    assert "within-brand mix" in resp["hint"]


# =============================================================================
# NEIGHBOURING GATES ARE UNTOUCHED
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_calculator_side_brand_guard_for_biologic_axes_still_fires(monkeypatch):
    """The Remibrutinib-only rule lives in BusinessImpactCalculator
    (_guard_brand_scoped_axis) and fires BEFORE any query. Two gates refuse
    Kisqali + biologic, and WHICH ONE speaks depends on the KPI -- so this pins
    the ORDER on both sides of the lane's contract flip:

    * PANEL TRx/NRx/NBRx SERVE the axis, so the tool gate passes them through
      and the BRAND guard is what refuses Kisqali (it was canonical TRx that
      did this before #2114 moved the patient panel to WS3-BI-011..013).
    * CANONICAL TRx no longer serves the axis, so the TOOL gate refuses first
      and no calculator is ever built -- the brand guard is not reached, and
      asserting the brand word absent would say nothing about which gate ran.

    Measured, not assumed: the brand guard fires for all three panel volumes on
    both Remibrutinib-only axes, and for Remibrutinib each axis BINDS into its
    own query (the positive control that proves the probe can see success).
    """
    from src.api.routes.chatbot_tools import kpi_calculate_tool
    from src.kpi.calculators.business_impact import BusinessImpactCalculator

    calc = BusinessImpactCalculator(db_client=_NoQueries())

    class _RealBI:
        def calculate(self, kpi_id: str, context: Any = None) -> KPIResult:
            kpi = get_registry().get(kpi_id)
            assert kpi is not None
            return calc.calculate(kpi, context)

    panel = {"TRx Panel": "trx", "NRx Panel": "nrx", "NBRx Panel": "nbrx"}

    _install(monkeypatch, _RealBI())
    for kpi_name in panel:
        for axis in ("biologic", "ige_tier"):
            resp = await kpi_calculate_tool.ainvoke(
                {"kpi_name": kpi_name, "brand": "Kisqali", axis: _AXIS_PROBE[axis]}
            )
            assert resp["success"] is False, (kpi_name, axis, resp)
            assert "Remibrutinib" in resp["error"], (kpi_name, axis, resp)
            assert "applies only to" not in resp["error"]  # not the axis gate

    # Positive control: for Remibrutinib each axis is BOUND into its own query.
    bound: List[Tuple[str, List[Any]]] = []

    def _record(query_id: str, params: List[Any]) -> None:
        bound.append((query_id, list(params)))
        raise RuntimeError("stop before the DB")

    monkeypatch.setattr(calc, "_execute_query", _record)
    for kpi_name, stem in panel.items():
        for axis in ("biologic", "ige_tier"):
            bound.clear()
            resp = await kpi_calculate_tool.ainvoke(
                {"kpi_name": kpi_name, "brand": "Remibrutinib", axis: _AXIS_PROBE[axis]}
            )
            assert resp["success"] is False and "stop before the DB" in resp["error"]
            # The base id, not the ``_include_synthetic`` twin: the swap happens
            # INSIDE ``_execute_query`` (which this test replaces), and the unit
            # tree's autouse fixture pins both synthetic flags off. A probe run
            # outside pytest picks the repo-root .env up and records the twin.
            assert bound == [
                (
                    f"business_impact_{stem}_{axis}",
                    ["Remibrutinib", _AXIS_PROBE[axis]],
                )
            ], (kpi_name, axis, bound)

    # The other side of the flip: canonical TRx is refused by the TOOL gate, so
    # no calculator is built at all. `_RaisingCalc` records any attribute touch,
    # which is a stronger discriminator than the absence of a phrase.
    reached: Dict[str, Any] = {}
    _install(monkeypatch, _RaisingCalc(reached))
    for axis in ("biologic", "ige_tier"):
        resp = await kpi_calculate_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Kisqali", axis: _AXIS_PROBE[axis]}
        )
        assert reached == {}, (axis, reached)
        assert resp["success"] is False and resp["kpi_id"] == "WS3-BI-005"
        assert "applies only to" in resp["error"], resp  # the axis gate spoke


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trigger_type_gate_is_untouched_and_ordered_first(monkeypatch):
    """#1360 gate unchanged: trigger_type on TRx still errors with ITS message
    (no patient-axis text), and a trigger KPI with a patient axis gets the
    patient-axis refusal, never a trigger_type one."""
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    reached: Dict[str, Any] = {}
    _install(monkeypatch, _RaisingCalc(reached))

    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "TRx", "brand": "Kisqali", "trigger_type": "adherence_risk"}
    )
    assert reached == {} and resp["success"] is False
    assert "trigger_type" in resp["error"] and "applies only to" not in resp["error"]

    resp = await kpi_calculate_tool.ainvoke(
        {
            "kpi_name": "acceptance rate",
            "brand": "Kisqali",
            "trigger_type": "adherence_risk",
            "therapy_line": "1",
        }
    )
    assert reached == {} and resp["success"] is False
    assert resp["error"].startswith("therapy_line (line of therapy) applies only to ")

    captured: Dict[str, Any] = {}
    _install(monkeypatch, _CapturingCalc(captured))
    resp = await kpi_calculate_tool.ainvoke(
        {"kpi_name": "acceptance rate", "brand": "Kisqali", "trigger_type": "adherence_risk"}
    )
    assert resp["success"] is True and captured["context"]["trigger_type"] == "adherence_risk"


# =============================================================================
# THE ALLOWLIST IS DERIVED FROM THE CALCULATORS, NOT ASSERTED BY HAND
# =============================================================================


@pytest.mark.unit
def test_allowlist_equals_the_kpis_whose_calculator_binds_the_axis():
    """For every registry KPI, run its real per-workstream calculator (the same
    classes get_kpi_calculator registers) against a recording client with the
    axis in context; the axis is SERVED iff its value is bound into a kpi_query
    parameter list. That set must equal the tool's allowlist AND the literal
    spec above -- the #1903 lesson: a hand-maintained claim went stale against
    a migration once already."""
    from src.api.routes.chatbot_tools import _PATIENT_AXIS_KPI_IDS
    from src.api.routes.kpi import _WORKSTREAM_CALCULATORS

    assert set(_PATIENT_AXIS_KPI_IDS) == set(_AXIS_PROBE)
    derived: Dict[str, set] = {axis: set() for axis in _AXIS_PROBE}
    refused_by_calculator: Dict[str, set] = {axis: set() for axis in _AXIS_PROBE}

    for kpi in get_registry().get_all():
        calculator_cls = _WORKSTREAM_CALCULATORS[kpi.workstream]
        for axis, value in _AXIS_PROBE.items():
            client = _RecordingClient()
            calc = calculator_cls(db_client=client)
            # Model-performance KPIs may construct an MLflow client (a ~100 s
            # retry loop against an unreachable tracking URI); the calculator
            # REMEMBERS a failure, so pre-set it and stay hermetic.
            if hasattr(calc, "_mlflow_client_error"):
                calc._mlflow_client_error = "mlflow_client_unavailable"
            assert calc.supports(kpi), (kpi.id, calculator_cls.__name__)
            result = calc.calculate(kpi, {"brand": "Remibrutinib", axis: value})
            params = [p for name, payload in client.calls for p in payload.get("params", [])]
            if value in params:
                derived[axis].add(kpi.id)
            elif result.error and "does not support" in result.error:
                refused_by_calculator[axis].add(kpi.id)

    for axis in _AXIS_PROBE:
        assert derived[axis] == set(_SERVED[axis]), axis
        assert derived[axis] == set(_PATIENT_AXIS_KPI_IDS[axis]), axis
    # WS3-BI-009 reads biologic / ige_tier only to refuse them (honest error,
    # nothing bound) -- documented here so the exclusion is visibly deliberate.
    assert "WS3-BI-009" in refused_by_calculator["biologic"]
    assert "WS3-BI-009" in refused_by_calculator["ige_tier"]


# =============================================================================
# THE TOOL'S OWN PROSE (reaches the LLM as the tool description)
# =============================================================================


@pytest.mark.unit
def test_axis_field_descriptions_and_docstring_name_the_served_kpis():
    from src.api.routes.chatbot_tools import KpiCalculateInput, kpi_calculate_tool

    doc = kpi_calculate_tool.coroutine.__doc__ or ""
    for axis in _AXIS_PROBE:
        desc = KpiCalculateInput.model_fields[axis].description or ""
        assert "TRx" in desc and "NBRx" in desc, axis
        assert "never silently dropped" in desc, axis
        assert f"{axis}:" in doc, axis
    for axis in ("segment", "therapy_line"):
        assert "conversion rate" in KpiCalculateInput.model_fields[axis].description.lower()
    for axis in ("biologic", "ige_tier"):
        desc = KpiCalculateInput.model_fields[axis].description.lower()
        assert "conversion rate" not in desc, axis
        assert "remibrutinib only" in desc, axis
    assert "CATE" in KpiCalculateInput.model_fields["segment"].description
    assert "CATE" not in KpiCalculateInput.model_fields["therapy_line"].description
    # #1910 window wording is untouched (it was just certified).
    assert "A window composes with any ONE" in doc
