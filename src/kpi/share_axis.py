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

# ⚠ RE-KEYED TO THE PANEL KPIs BY OWNER DECISION #11 (#2114 lane merge).
# #2137 wrote this module against the pre-lane substrate, where TRx Share was the
# canonical WS3-BI-008 and the answer to redirect to was canonical TRx WS3-BI-005.
# The lane makes business_metrics canonical and moves the patient panel to
# WS3-BI-011..014, so the KPI whose share has no patient-axis breakdown is now the
# PANEL share WS3-BI-014, and the within-brand mix the user actually wants lives on
# PANEL TRx WS3-BI-011. The reasoning in this module's docstring is unchanged and
# was always about the panel substrate -- only the ids it names had to move.
#
# BOTH SURFACES READ THESE CONSTANTS -- `_calc_trx_share` (/api/kpis) and
# `kpi_calculate_tool` (chat) -- so re-keying here moves them together and they
# cannot drift apart. That is why the ids are not hardcoded at either call site.
TRX_SHARE_KPI_ID = "WS3-BI-014"
TRX_KPI_ID = "WS3-BI-011"
#: Registry name of TRX_KPI_ID (pinned by tests/unit/test_kpi/test_share_axis.py).
TRX_NAME = "Observed Rx Events - Patient Panel TRx (TRx Panel)"

#: ⚠ TWO SHARES, TWO REASONS, ONE DESTINATION — OWNER DECISION #14 (#2114).
#:
#: This module's reason is a statement about the PATIENT-PANEL substrate: "every
#: patient is on exactly one tracked brand, so a share within a bucket divides one
#: indication's prescriptions by another's". On main that was true of WS3-BI-008,
#: which read the event ledger. After this lane 008 reads business_metrics at
#: brand x region x calendar month (MEASURED: `registry.get("WS3-BI-008").tables ==
#: ['business_metrics']`), where the patient is not the unit at all -- so the
#: sentence is now LITERALLY FALSE of 008 and true only of panel share 014.
#: Re-keying the constants to 014 was therefore correct: THE REASONING FOLLOWED THE
#: SUBSTRATE, NOT THE ID.
#:
#: But the user's need did not move with the sentence. "TRx share by severity tier"
#: resolves to canonical 008, and canonical TRx 005 no longer serves patient axes
#: either, so that ask hit a dead stop where main gave a working next step. Owner
#: #14: give 008 its OWN reason -- true of its own substrate -- and send both shares
#: to the SAME destination, panel TRx WS3-BI-011, whose buckets sum to the brand
#: total. Two reasons, one destination; the false sentence is not reused.
CANONICAL_SHARE_KPI_ID = "WS3-BI-008"

#: share KPI -> (TRx KPI id, TRx registry name) that answers the ask instead.
#: BOTH point at the PANEL TRx: it is the only TRx that carries a patient axis.
SHARE_REDIRECTS: dict[str, Tuple[str, str]] = {
    CANONICAL_SHARE_KPI_ID: (TRX_KPI_ID, TRX_NAME),
    TRX_SHARE_KPI_ID: (TRX_KPI_ID, TRX_NAME),
}


def share_axis_reason_for(kpi_id: str, axis: str, label: str) -> str:
    """The reason THIS share KPI has no breakdown on ``axis`` -- true of ITS substrate.

    The panel share gets the patient_journeys argument; the canonical share gets the
    one that is true of business_metrics. Reusing the panel sentence for 008 would
    state something false about the data it reads, which is what owner #14 rejected.
    """
    if kpi_id == CANONICAL_SHARE_KPI_ID:
        return (
            f"Canonical TRx Share is a portfolio share computed from business_metrics at "
            f"brand x region x calendar month, which carries no patient dimension to split "
            f"by {label}."
        )
    return share_axis_reason(axis, label)


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
