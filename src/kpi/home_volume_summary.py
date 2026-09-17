"""Canonical volume fields for the Home KPI summary (canonical TRx lane).

The Home volume tiles (TRx / NRx / NBRx / TRx share) read the canonical monthly
business_metrics series, whose period is the latest COMPLETE calendar month.
``data_through`` stays the event frontier (hcp_reach / conversion tiles and the
empty-tile label need it); ``volume_data_through`` and ``volume_period`` state
the volume tiles' month. A failed read degrades to None — never a guessed month.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Optional

from src.kpi.synthetic_mode import resolve_kpi_query_id

logger = logging.getLogger(__name__)

VOLUME_PERIOD_QUERY_ID = "canonical_volume_trx"


def volume_tile_fields(client: Any, data_through: Optional[str]) -> Dict[str, Optional[str]]:
    """``{data_through, volume_data_through, volume_period}`` for the Home summary.

    The month is READ from the statement the tiles themselves run (its
    ``data_month`` / ``data_through`` columns), never derived from the clock: the
    canonical frontier lags the calendar by a variable amount, so a label built
    from ``CURRENT_DATE`` would be confidently wrong and indistinguishable from a
    real one. Unknown stays None.
    """
    fields: Dict[str, Optional[str]] = {
        "data_through": data_through,
        "volume_data_through": None,
        "volume_period": None,
    }
    if client is None:
        return fields
    try:
        rows = (
            client.rpc(
                "kpi_query",
                {"query_id": resolve_kpi_query_id(VOLUME_PERIOD_QUERY_ID), "params": [None]},
            )
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001 - supplementary period label only
        logger.warning("[home] canonical volume period query failed: %s", exc)
        return fields
    row = rows[0] if rows else {}
    if row.get("data_through") is not None:
        fields["volume_data_through"] = str(row["data_through"])[:10]
    if row.get("data_month") is not None:
        fields["volume_period"] = date.fromisoformat(str(row["data_month"])[:10]).strftime("%B %Y")
    return fields
