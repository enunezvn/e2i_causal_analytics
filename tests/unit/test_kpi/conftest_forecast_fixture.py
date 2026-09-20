"""Loader for the real canonical TRx fixture (Lane B, #2115).

The fixture is a PROD read, not a synthesised curve: every assertion about model
accuracy in this suite is therefore a statement about the series the tool will
actually forecast.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "forecast" / "canonical_trx_monthly.csv"
)


def load_canonical_trx() -> Dict[str, List[Tuple[date, float]]]:
    out: Dict[str, List[Tuple[date, float]]] = {}
    with FIXTURE.open() as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith("#"))
        for row in rows:
            out.setdefault(row["brand"], []).append(
                (date.fromisoformat(row["month"]), float(row["value"]))
            )
    for series in out.values():
        series.sort()
    return out


def series_values(brand: str = "Kisqali") -> List[float]:
    return [v for _, v in load_canonical_trx()[brand]]
