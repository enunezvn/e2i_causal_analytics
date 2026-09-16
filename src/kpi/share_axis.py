"""Why TRx Share has no patient-axis breakdown, stated once for every surface.

Every patient in ``patient_journeys`` is on exactly ONE tracked brand, so "the
brand's share of the portfolio's prescriptions among patients in bucket B" divides
one indication's prescriptions by those of other indications' patients who happen to
carry the same label (severity tier, line of therapy). On the Remibrutinib-only
biologic-status / IgE-tier axes the bucket holds only the brand's own patients, so
the "share" is always 100%. Chat session_1789548670222_fcscf3u (2026-09-16) served
both. The answer a "share by tier" ask is after is the brand's own TRx by the axis,
whose buckets sum to the brand total: each bucket's fraction is the within-brand mix.

``BusinessImpactCalculator._calc_trx_share`` (the /api/kpis path) and
``kpi_calculate_tool`` (the chat path) both refuse with this wording.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

TRX_SHARE_KPI_ID = "WS3-BI-008"
TRX_KPI_ID = "WS3-BI-005"
#: Registry name of TRX_KPI_ID (pinned by tests/unit/test_kpi/test_share_axis.py).
TRX_NAME = "Total Prescriptions (TRx)"

#: (context key, human label), in the calculator's axis precedence order.
PATIENT_AXES: Tuple[Tuple[str, str], ...] = (
    ("segment", "severity tier"),
    ("therapy_line", "line of therapy"),
    ("biologic", "biologic status"),
    ("ige_tier", "IgE tier"),
)

#: Axes populated for one brand only, where the share is identically 100%.
_BRAND_ONLY_AXES = frozenset({"biologic", "ige_tier"})


def requested_patient_axis(context: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """The first patient axis set in ``context``. ``is not None``, not truthiness:
    therapy line 0 is a real bucket."""
    return next(((key, label) for key, label in PATIENT_AXES if context.get(key) is not None), None)


def share_axis_reason(axis: str, label: str) -> str:
    """Why the share is undefined on this axis."""
    reason = (
        f"Each patient is on one tracked brand, so a portfolio share within a {label} "
        f"bucket divides this brand's prescriptions by other indications' patients who "
        f"share the label."
    )
    if axis in _BRAND_ONLY_AXES:
        reason += " On this axis every patient is the brand's own, so it would be always 100%."
    return reason


def share_axis_next_step(trx_name: str, label: str) -> str:
    """The answer to ask for instead."""
    return (
        f"Ask for {trx_name} by {label}: its buckets sum to the brand total, so each "
        f"bucket's fraction of that total is the within-brand mix."
    )
