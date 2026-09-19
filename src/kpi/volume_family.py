"""The two Rx-volume KPI families and the one map between them (canonical TRx lane).

Owner decision 2026-09-15: TRx, NRx, NBRx and TRx Share (WS3-BI-005..008) read
the CANONICAL monthly ``business_metrics`` series. The ``treatment_events``
prescription counts they used to compute are a different quantity and become
their own KPIs — "Observed Rx Events - Patient Panel" (WS3-BI-011..014) — which
keep everything that needs event grain: the patient-axis splits, the claims-lag
nowcast, segmented history and the 089 data-frontier windows.

Deliberately dependency-free (like ``src.kpi.measure_basis``): the chat tools,
the measure-basis SSOT and the calculators all import it.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

CANONICAL_TO_PANEL: Mapping[str, str] = MappingProxyType(
    {
        "WS3-BI-005": "WS3-BI-011",  # TRx -> TRx Panel
        "WS3-BI-006": "WS3-BI-012",  # NRx -> NRx Panel
        "WS3-BI-007": "WS3-BI-013",  # NBRx -> NBRx Panel
        "WS3-BI-008": "WS3-BI-014",  # TRx Share -> TRx Share Panel
    }
)
PANEL_TO_CANONICAL: Mapping[str, str] = MappingProxyType(
    {panel: canonical for canonical, panel in CANONICAL_TO_PANEL.items()}
)
CANONICAL_VOLUME_KPI_IDS: frozenset[str] = frozenset(CANONICAL_TO_PANEL)
PANEL_VOLUME_KPI_IDS: frozenset[str] = frozenset(PANEL_TO_CANONICAL)
#: The cumulative event-count panel KPIs (the share is a ratio, not additive).
PANEL_RX_COUNT_KPI_IDS: frozenset[str] = frozenset({"WS3-BI-011", "WS3-BI-012", "WS3-BI-013"})

#: Patient axes as calculator context keys — served only at event grain.
PATIENT_AXES: tuple[str, ...] = ("segment", "therapy_line", "biologic", "ige_tier")

#: Measured, not modeled (read-only DB reads, 2026-09-15). Quoted by every note
#: that fences a panel figure off a canonical one.
#: scale_note_597_provenance.txt: Kisqali prescription rows, event_date >= max(event_date) - 30
MEASURED_SCALE_NOTE = (
    "measured 2026-09-15, Kisqali canonical TRx for 2026-08 was 800,349 "
    "(business_metrics, brand x region x calendar month) against 597 patient-panel "
    "prescription events from 2026-08-14 through 2026-09-14 inclusive "
    "(treatment_events), about 1,300x"
)

#: ONE NULL-dimension rule for every canonical volume reader (codex r2 HIGH).
#: business_metrics.brand and .region are nullable
#: (database/core/e2i_ml_complete_v3_schema.sql:647-648).
#: A row with no brand cannot sit in any brand headline, brand share or brand history
#: cell, and a row with no region cannot sit in any region cell. Counting it in the
#: national total but in none of its own cells would make a headline differ from its
#: history and would move every share denominator. So a row counts only when BOTH are
#: present: the migration-143 statements (headline, share, windowed, monthly series)
#: embed DIMENSIONED_ROW_SQL, and the history handler applies has_dimensions.
#: Measured 2026-09-15: 0 NULL brand or region among trx, nrx and market_share rows,
#: so the rule changes no current number; it fixes the latent disagreement.
DIMENSIONED_ROW_SQL = "brand IS NOT NULL AND region IS NOT NULL"


def has_dimensions(row: Mapping[str, Any]) -> bool:
    """The Python twin of DIMENSIONED_ROW_SQL (the brand/region enums cannot hold '')."""
    return row.get("brand") not in (None, "") and row.get("region") not in (None, "")


#: Substrate-contract version folded into every KPI cache key (codex r1 HIGH, Task 10A).
#: Bump it whenever a KPI id changes WHAT IT MEASURES, so a Redis entry written under
#: the old meaning can never be served under the new one. The canonical TRx lane moves
#: WS3-BI-005..008 from treatment_events counts to the business_metrics series, and the
#: API stamps measure_basis from the live registry -- so without this an entry written
#: by the pre-deploy containers would be served, for up to its TTL, wearing the new
#: basis label. Read at deploy time by prove_state.sh (Task 30) to bind the app tier.
KPI_CACHE_BASIS_VERSION = "canonical-trx-2026-09-15"
