"""IF A REFUSAL NAMES A NEXT STEP, FOLLOWING IT MUST SUCCEED (#2114).

THREE DEAD-END REDIRECTS WERE SHIPPED IN ONE DAY, every one of them text that read
correctly and led nowhere:

1. WS3-BI-014 refused a patient axis and told the user to ask for TRx Panel —
   which was not in the tool's allowlist, so following it refused again.
2. WS3-BI-008 refused and (briefly) pointed at canonical TRx WS3-BI-005, which by
   then no longer served patient axes: the premise "the within-brand mix lives
   where the share that refused it lives" had been falsified by the allowlist
   commit made an hour earlier.
3. The generic canonical refusal pointed WS3-BI-008's user at WS3-BI-014 — the
   panel SHARE, which refuses patient axes itself. Visible only in a before/after
   diff, because the sentence was perfectly well-formed.

A FOURTH, found by codex r13 and reproduced here: the redirect was BRAND-BLIND.
``biologic`` / ``ige_tier`` exist only for Remibrutinib, so canonical TRx refusing
"Kisqali by biologic status" and naming panel TRx sent the user to a KPI that
refuses the same ask for the same brand. On main that ask got the brand guard's
complete answer in ONE hop, because canonical TRx served the axis then — the lane
turned a true one-hop refusal into a two-hop redirect whose first hop cannot work.

Each was caught by a human following the hint by hand. None could have been caught
by an assertion on the message TEXT, and every one of those messages had a test.

⚠ THIS GUARD IS DELIBERATELY NOT SHARE-ONLY. Defect 3 came from the GENERIC
canonical path, so a share-scoped guard would have missed the very case that
proved the guard was needed. The invariant is general, so the test is general.

⚠ THE CLIENT RETURNS NO ROWS RATHER THAN RAISING. A fake that raises on any
attribute access cannot distinguish "an axis guard refused this" from "the query
ran and found nothing" — and a probe that cannot tell a guard from a datasource
cannot answer whether a redirect works. That mistake produced a false DEAD END
reading during this lane; the no-rows client is what separates the two.

⚠⚠ SUCCESS IS POSITIVE EVIDENCE, NOT THE ABSENCE OF A PHRASE. The first version of
this guard scored a destination as reachable when its error matched none of three
hand-listed marker phrases ("does not support", "is not defined by", "applies only
to"). A brand refusal says "biologic-status breakdown is not available for Kisqali",
which matches none of them — so THE GUARD WOULD HAVE SCORED DEFECT 4 AS A SUCCESS,
and adding the brand dimension alone would not have fixed that. It now requires the
destination to BIND the requested axis value into a recorded query: the same class
as a probe with no positive control, caught by codex r13 (MEDIUM 3).
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.registry import get_registry
from src.kpi.share_axis import PATIENT_AXES, brand_scoped_axis_refusal

#: KPIs that refuse a patient axis AND name a next step. Both families, per the
#: lesson above: the canonical volume KPIs (generic refusal) and the panel share.
_REFUSING_KPI_IDS = ("WS3-BI-005", "WS3-BI-006", "WS3-BI-007", "WS3-BI-008", "WS3-BI-014")

#: axis -> a probe value distinctive enough to be recognised in a bound parameter list.
_AXES: Tuple[Tuple[str, str], ...] = (
    ("segment", "high_severity"),
    ("therapy_line", "3"),
    ("biologic", "naive"),
    ("ige_tier", "high"),
)

#: Every brand the DGP knows, plus the no-brand ask. Kisqali/Fabhalta are the
#: brands for which the biologic/IgE columns are NULL by design.
_BRANDS: Tuple[Optional[str], ...] = ("Remibrutinib", "Kisqali", "Fabhalta", None)


class _RecordingNoRows:
    """Returns no rows instead of raising, and records every kpi_query it is asked
    to run — so a guard refusal, a datasource miss and a real binding are three
    distinguishable outcomes rather than two."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def rpc(self, name: str, payload: Any = None) -> Any:
        if isinstance(payload, dict):
            self.calls.append(dict(payload))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: self


def _context(brand: Optional[str], axis: str, value: str) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {axis: value}
    if brand is not None:
        ctx["brand"] = brand
    return ctx


def _calculate(kpi_id: str, ctx: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """(error text, recorded queries) for one ask against one KPI."""
    client = _RecordingNoRows()
    calc = BusinessImpactCalculator(db_client=client)
    kpi = get_registry().get(kpi_id)
    assert kpi is not None, kpi_id
    result = calc.calculate(kpi, dict(ctx))
    return str(result.error or ""), client.calls


def _named_targets(message: str, refusing_id: str) -> List[str]:
    """Every OTHER KPI the refusal names, by id or by registry name."""
    targets = {m for m in re.findall(r"WS3-BI-\d{3}", message) if m != refusing_id}
    for kpi in get_registry().get_all():
        if kpi.id != refusing_id and kpi.name and kpi.name in message:
            targets.add(kpi.id)
    return sorted(targets)


def _bound_the_axis(calls: List[Dict[str, Any]], value: str) -> bool:
    """POSITIVE evidence that the ask reached a query carrying the requested axis."""
    return any(
        any(str(param) == value for param in (call.get("params") or []))
        for call in calls
    )


@pytest.mark.parametrize("brand", _BRANDS, ids=lambda b: str(b))
@pytest.mark.parametrize("axis,value", _AXES, ids=lambda v: str(v))
@pytest.mark.parametrize("kpi_id", _REFUSING_KPI_IDS)
def test_every_redirect_named_in_a_refusal_leads_somewhere(
    kpi_id: str, axis: str, value: str, brand: Optional[str]
) -> None:
    """THE ROUND TRIP, not the wording: read the target OUT of the refusal and follow it
    IN THE SAME CONTEXT — same brand, same axis, same value.

    Reading the target out of the message rather than asserting a name is what makes
    this a guard instead of a restatement: it follows whatever the code actually
    said, so it keeps working when the destination legitimately changes and fails
    when the destination stops being able to answer.
    """
    ctx = _context(brand, axis, value)
    message, _ = _calculate(kpi_id, ctx)
    assert message, f"{kpi_id} did not refuse {axis} for brand={brand}"

    targets = _named_targets(message, kpi_id)
    dead_ends: Dict[str, str] = {}
    for target_id in targets:
        target_error, target_calls = _calculate(target_id, ctx)
        if _bound_the_axis(target_calls, value):
            continue
        # A destination that refuses for a MISSING PREREQUISITE the source ask also
        # lacked is not a dead end: the user supplies the brand and the redirect
        # works. The defect this guard exists for is the opposite — a COMPLETE ask
        # whose destination still cannot answer it. So the carve-out is allowed
        # only when the ask carried no brand, and only for that reason.
        if brand is None and "no brand specified" in target_error:
            continue
        dead_ends[target_id] = target_error or "(no error, but no query bound the axis)"

    assert not dead_ends, (
        f"{kpi_id} refused {axis}={value} for brand={brand} and named a next step that "
        f"CANNOT answer the same ask — a DEAD END: {dead_ends}\n  source said: {message}"
    )


@pytest.mark.parametrize("brand", _BRANDS, ids=lambda b: str(b))
@pytest.mark.parametrize("axis,value", _AXES, ids=lambda v: str(v))
@pytest.mark.parametrize("kpi_id", _REFUSING_KPI_IDS)
def test_a_serveable_axis_is_always_given_a_next_step(
    kpi_id: str, axis: str, value: str, brand: Optional[str]
) -> None:
    """The other half of the invariant: when the ask IS answerable somewhere, the
    refusal must say where. Silence would be honest but useless, and it is what
    the pre-#14 canonical share did — a dead stop where main gave a next step.

    Skipped exactly where no destination exists: a brand-scoped axis asked for a
    brand whose columns are NULL by design. That condition comes from the SAME
    helper the calculators guard with, so this test cannot disagree with the
    product about which asks are serveable.
    """
    if brand_scoped_axis_refusal(axis, brand) is not None:
        pytest.skip(f"{axis} is not serveable for brand={brand} by any KPI")

    message, _ = _calculate(kpi_id, _context(brand, axis, value))
    assert message, f"{kpi_id} did not refuse {axis} for brand={brand}"

    label = dict(PATIENT_AXES)[axis]
    if axis not in message and label not in message:
        # Not the axis refusal at all: a PREREQUISITE is missing and is refused
        # first (the share KPIs need a brand before any axis is considered). The
        # next-step invariant does not apply — but the case is not waved through:
        # the message must name the prerequisite, or it is a third possibility we
        # have not accounted for.
        assert "no brand specified" in message, (
            f"{kpi_id} refused {axis}={value} for brand={brand} with a message about "
            f"neither the axis nor a missing prerequisite: {message}"
        )
        return

    assert _named_targets(message, kpi_id), (
        f"{kpi_id} refused {axis}={value} for brand={brand} without naming any next "
        f"step, although one exists: {message}"
    )


@pytest.mark.parametrize("brand", ("Kisqali", "Fabhalta", None), ids=lambda b: str(b))
@pytest.mark.parametrize("axis,value", (("biologic", "naive"), ("ige_tier", "high")))
@pytest.mark.parametrize("kpi_id", _REFUSING_KPI_IDS)
def test_an_impossible_axis_states_the_brand_limit_instead_of_a_destination(
    kpi_id: str, axis: str, value: str, brand: Optional[str]
) -> None:
    """DEFECT 4, pinned from the other side: the terminal case must be TERMINAL.

    biologic / IgE exist only for Remibrutinib, so for any other brand no KPI can
    answer — there is no next step to name, and naming one is the defect. The
    refusal must instead say what the limit is, naming the brand that does carry
    the axis, so the user learns the real constraint in ONE hop (which is what
    main did before this lane moved the axes to the panel KPIs).
    """
    message, calls = _calculate(kpi_id, _context(brand, axis, value))
    assert calls == [], f"{kpi_id} queried the DB for an impossible ask: {calls}"

    # TERMINAL means: no destination is advertised...
    assert not _named_targets(message, kpi_id), (
        f"{kpi_id} refused {axis} for brand={brand} and STILL named a KPI to try — "
        f"there is none that can answer it for this brand: {message}"
    )
    # ...and the message says what is missing. Either wording is truthful: the axis
    # limit (naming the brand that does carry it), or — when the ask supplied no
    # brand at all — the missing brand itself, which is the more fundamental gap.
    assert "Remibrutinib" in message or "no brand specified" in message, (
        f"{kpi_id} refused {axis} for brand={brand} without naming either the brand "
        f"that carries the axis or the missing brand: {message}"
    )
