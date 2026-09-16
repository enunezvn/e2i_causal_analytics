"""Declared axis capability must match what the calculators ACTUALLY do (#2150).

Stage 1 of the 13-owner consolidation: these tests encode the contract before any
surface is rewired, so the spec is reviewable before the metadata exists. They are
RED BY COLLECTION until ``src.kpi.capability_policy`` and
``KPIMetadata.axis_capability`` land — that is the intended red.

⭐ THE EXPECTED SIDE COMES FROM THE CALCULATORS, NEVER FROM THE METADATA.
Deriving both the implementation and the expected answers from one declaration
would conceal wrong metadata: the suite would agree with itself no matter what the
platform does. So "served" here is measured the way
``tests/unit/test_api/test_chatbot_kpi_axis_gate_1911.py`` measures it — run the
REAL per-workstream calculator against a recording client and look for the axis
VALUE inside a bound ``kpi_query`` parameter list.

⭐ POSITIVE EVIDENCE ONLY. "Served" is never inferred from the absence of an error
string. The ``_GUARD_MARKERS`` lesson in this lane: a marker-phrase list scored a
real dead end as REACHABLE because a brand refusal matched none of its phrases. A
refusal can be worded anything; a bound parameter cannot be faked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from src.kpi.registry import get_registry

#: One probe value per axis, distinctive enough to be recognised in a parameter
#: list (we look for the VALUE, not a query id).
_AXIS_PROBE: Dict[str, str] = {
    "segment": "high_severity",
    "therapy_line": "3",
    "biologic": "naive",
    "ige_tier": "high",
}
#: Remibrutinib is the only brand carrying the biologic / IgE columns, so the
#: measurement brand must be one for which those axes CAN bind.
_MEASURE_BRAND = "Remibrutinib"


class _RecordingClient:
    """Records every kpi_query RPC payload and returns no rows.

    Returning no rows rather than raising is deliberate: a fake that raises on any
    attribute access cannot tell "a guard refused this" from "the query ran and
    found nothing", and a probe that cannot tell those apart cannot answer whether
    an axis is served.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def rpc(self, name: str, payload: Dict[str, Any]) -> Any:
        self.calls.append((name, dict(payload)))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))


def _bound_axes_by_kpi() -> Dict[str, set]:
    """MEASURED: axis -> {kpi_id whose calculator binds that axis into a query}."""
    from src.api.routes.kpi import _WORKSTREAM_CALCULATORS

    derived: Dict[str, set] = {axis: set() for axis in _AXIS_PROBE}
    for kpi in get_registry().get_all():
        calculator_cls = _WORKSTREAM_CALCULATORS[kpi.workstream]
        for axis, value in _AXIS_PROBE.items():
            client = _RecordingClient()
            calc = calculator_cls(db_client=client)
            # Model-performance calculators otherwise build an MLflow client and
            # retry for ~100 s against an unreachable tracking URI.
            if hasattr(calc, "_mlflow_client_error"):
                calc._mlflow_client_error = "mlflow_client_unavailable"
            calc.calculate(kpi, {"brand": _MEASURE_BRAND, axis: value})
            params = [p for _name, payload in client.calls for p in payload.get("params", [])]
            if value in params:
                derived[axis].add(kpi.id)
    return derived


# =============================================================================
# 1.1 — conformance: declared capability == measured behaviour
# =============================================================================


@pytest.mark.unit
def test_declared_capability_matches_what_the_calculators_bind_today():
    """Every combination served today must be declared served, and vice versa.

    This is the migration proof: no currently-served combination may silently
    disappear behind the new declaration, and none may silently appear.
    """
    from src.kpi.capability_policy import serves

    measured = _bound_axes_by_kpi()
    disagreements = []
    for kpi in get_registry().get_all():
        for axis in _AXIS_PROBE:
            served_now = kpi.id in measured[axis]
            declared = serves(kpi.id, axis, brand=_MEASURE_BRAND)
            if served_now != declared:
                disagreements.append(
                    f"{kpi.id}/{axis}: calculators say served={served_now}, "
                    f"declaration says {declared}"
                )
    assert not disagreements, (
        "declared capability disagrees with measured behaviour:\n" + "\n".join(disagreements)
    )


@pytest.mark.unit
def test_the_measurement_itself_found_something():
    """Positive control for the probe above.

    If the recording client stopped recording, or the probe values stopped being
    bound, every KPI would read as "not served" and the conformance test would pass
    against a declaration of all-False. Assert the measurement is non-empty and
    contains the panel trio before trusting any comparison built on it.
    """
    measured = _bound_axes_by_kpi()
    for axis in _AXIS_PROBE:
        assert measured[axis], f"{axis}: the probe bound nothing at all"
    assert {"WS3-BI-011", "WS3-BI-012", "WS3-BI-013"} <= measured["segment"]


# =============================================================================
# 1.2 — the declaration must be able to EXPRESS the platform's real rules
# =============================================================================


@pytest.mark.unit
def test_capability_is_declared_on_the_registry_metadata():
    """The one source is the registry, so the metadata must carry it — and the
    loader must actually parse it. ``registry._parse_kpi`` builds ``KPIMetadata``
    from an explicit ``data.get(...)`` per field (verified at 0f4916916), so a new
    YAML key is silently DROPPED unless the loader changes too. This asserts the
    parsed object, never the YAML text, so a loader that ignores the key fails.
    """
    kpi = get_registry().get("WS3-BI-011")
    assert kpi is not None
    assert kpi.axis_capability is not None, "WS3-BI-011 carries no parsed axis capability"
    assert "segment" in kpi.axis_capability.axes


@pytest.mark.unit
def test_an_axis_can_be_served_but_refused_UNDER_A_WINDOW():
    """A flat ``patient_axes`` list cannot express this, which is why it is not one.

    WS3-BI-014 serves ``region`` but refuses it once a window is present; a list of
    supported axes has nowhere to put that, so the declaration would have to lie in
    one direction or the other.
    """
    from src.kpi.capability_policy import serves

    assert serves("WS3-BI-014", "region", window=False) is True
    assert serves("WS3-BI-014", "region", window=True) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "brand,expected", [("Remibrutinib", True), ("Kisqali", False), ("Fabhalta", False)]
)
def test_brand_requirement_is_per_axis_not_per_kpi(brand, expected):
    """biologic / ige_tier exist for Remibrutinib only, while segment / therapy_line
    serve every brand — on the SAME KPI. A per-KPI brand flag cannot say that."""
    from src.kpi.capability_policy import serves

    assert serves("WS3-BI-011", "biologic", brand=brand) is expected
    assert serves("WS3-BI-011", "segment", brand=brand) is True


@pytest.mark.unit
def test_the_redirect_is_resolved_by_capability_and_round_trips():
    """The lane's four plan defects were all a destination named from an ID MAP
    instead of from the destination's capability. So the redirect is not stored as
    an id: it is whatever the evaluator finds that CAN serve the ask, and following
    it must succeed.
    """
    from src.kpi.capability_policy import redirect_for, serves

    target = redirect_for("WS3-BI-005", "segment", brand="Remibrutinib")
    assert target is not None, "a refusal that can be answered elsewhere must say where"
    assert serves(target, "segment", brand="Remibrutinib") is True, (
        f"redirect names {target}, which does not serve the same ask — a DEAD END"
    )


@pytest.mark.unit
def test_an_unserveable_ask_names_no_destination():
    """The terminal case must stay terminal: biologic for Kisqali is served by NO
    KPI, so naming one would be dead end 4 all over again."""
    from src.kpi.capability_policy import redirect_for

    assert redirect_for("WS3-BI-005", "biologic", brand="Kisqali") is None


@pytest.mark.unit
def test_the_two_authored_share_reasons_stay_DISTINCT():
    """Owner #14 ruled the canonical share and the panel share get DIFFERENT
    reasons, because each is true of its own substrate: 008 reads business_metrics
    (no patient dimension at all), 014 reads treatment_events (one tracked brand per
    patient). Flattening them into one generated sentence would restate for 008 the
    thing owner #14 rejected, so the policy must READ the authored reasons rather
    than compose them.
    """
    from src.kpi.capability_policy import reason_for

    canonical = reason_for("WS3-BI-008", "segment")
    panel = reason_for("WS3-BI-014", "segment")
    assert canonical and panel
    assert canonical != panel, "the two share reasons were flattened into one"
    assert "business_metrics" in canonical


# =============================================================================
# POSITIVE CONTROL ON CONFIG LOADING — the silent-empty failure mode
# =============================================================================


@pytest.mark.unit
def test_the_derived_sets_are_not_silently_empty():
    """A mis-resolved config path turns the whole capability system OFF, quietly.

    ``registry._load_definitions`` WARNS instead of raising when the YAML is not
    found, so every derived set would come back EMPTY — and empty is
    indistinguishable from "nothing serves these axes" at every surface
    downstream: an empty allowlist refuses every axis, an empty coverage set never
    probes, and EVERY test that asserts a refusal still passes. Nothing else in
    this file would fail, because they all assert that things are refused.

    So assert the positive: the sets have members, and the members are the ones
    the platform is built on. Same shape as the probe's own control above, pointed
    at config loading instead.
    """
    from src.kpi.capability_policy import axis_kpi_ids, trailing_coverage_kpi_ids

    panel = {"WS3-BI-011", "WS3-BI-012", "WS3-BI-013"}
    assert get_registry().get_all(), "the registry loaded NOTHING — config path?"
    assert panel <= axis_kpi_ids("segment"), "the axis allowlist lost the panel trio"
    assert trailing_coverage_kpi_ids() == panel, (
        "the trailing-coverage set is not the panel trio — empty means the probe "
        "silently never runs"
    )


@pytest.mark.unit
def test_every_event_grain_kpi_declares_its_additivity():
    """Completeness, so a MISSING declaration is loud at test time rather than a
    silent absence of a probe at runtime (where it fails closed by design)."""
    from src.kpi.capability_policy import event_grain_kpi_ids

    undeclared = sorted(
        kpi_id for kpi_id in event_grain_kpi_ids() if get_registry().get(kpi_id).additive is None
    )
    assert not undeclared, f"event-grain KPIs with no declared additivity: {undeclared}"
