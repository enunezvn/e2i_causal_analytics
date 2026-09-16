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
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.registry import get_registry

#: KPIs that refuse a patient axis AND name a next step. Both families, per the
#: lesson above: the canonical volume KPIs (generic refusal) and the panel share.
_REFUSING_KPI_IDS = ("WS3-BI-005", "WS3-BI-006", "WS3-BI-007", "WS3-BI-008", "WS3-BI-014")

_AXIS = ("segment", "high_severity")

#: A guard refused it. Anything else (including "no data") means the ask CLEARED
#: the guards and reached the query, which is what a working redirect looks like.
_GUARD_MARKERS = ("does not support", "is not defined by", "applies only to")


class _NoRows:
    """Returns no rows instead of raising, so a guard is distinguishable from the DB."""

    def rpc(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: self


def _refuse(kpi_id: str) -> str:
    calc = BusinessImpactCalculator(db_client=_NoRows())
    kpi = get_registry().get(kpi_id)
    assert kpi is not None, kpi_id
    result = calc.calculate(kpi, {"brand": "Remibrutinib", _AXIS[0]: _AXIS[1]})
    assert result.error, f"{kpi_id} did not refuse the {_AXIS[0]} axis"
    return str(result.error)


def _named_targets(message: str, refusing_id: str) -> List[str]:
    """Every OTHER KPI the refusal names, by id or by registry name."""
    targets = {m for m in re.findall(r"WS3-BI-\d{3}", message) if m != refusing_id}
    for kpi in get_registry().get_all():
        if kpi.id != refusing_id and kpi.name and kpi.name in message:
            targets.add(kpi.id)
    return sorted(targets)


@pytest.mark.parametrize("kpi_id", _REFUSING_KPI_IDS)
def test_every_redirect_named_in_a_refusal_leads_somewhere(kpi_id: str) -> None:
    """THE ROUND TRIP, not the wording: read the target OUT of the refusal and follow it.

    Reading the target out of the message rather than asserting a name is what makes
    this a guard instead of a restatement — it follows whatever the code actually
    said, so it keeps working when the destination legitimately changes and fails
    when the destination stops being able to answer."""
    message = _refuse(kpi_id)
    targets = _named_targets(message, kpi_id)
    assert targets, f"{kpi_id} refused without naming any next step: {message}"

    calc = BusinessImpactCalculator(db_client=_NoRows())
    failures: Dict[str, str] = {}
    for target_id in targets:
        target = get_registry().get(target_id)
        assert target is not None, target_id
        result = calc.calculate(target, {"brand": "Remibrutinib", _AXIS[0]: _AXIS[1]})
        error = str(result.error or "")
        if any(marker in error for marker in _GUARD_MARKERS):
            failures[target_id] = error

    assert not failures, (
        f"{kpi_id}'s refusal sends the user to a KPI that also refuses the "
        f"{_AXIS[0]} axis — a DEAD END: {failures}"
    )
