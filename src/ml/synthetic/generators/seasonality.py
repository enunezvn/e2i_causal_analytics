"""Deterministic calendar-month seasonality for the monthly Rx-volume series.

Canonical TRx lane (owner decision 2026-09-15): ``business_metrics`` ``value``
for metric types trx / nrx / nbrx carries a multiplicative calendar-month
profile. Properties (pinned by tests/unit/test_synthetic/test_seasonality.py):

* amplitude +-8% (max |deviation| = 800 bp), annual mean EXACTLY 1.0 — stored
  as integer basis points so the identity is sum(bp) == 0, not a float;
* January is the unique trough (commercial-plan deductible reset and prior-auth
  re-verification), February stays below 1, a mild summer dip (Jun-Aug),
  December is the unique peak (deductibles met, year-end fills);
* keyed on the CALENDAR month of ``metric_date`` — never on the generator's
  ``month_idx`` (which depends on ``trend_origin``), so a single-date frontier
  cohort and the frozen-base regeneration agree on every month;
* applied to ``value`` only (targets stay on the market-size trend line) and
  draws from NO seeded generator, so metric_ids, targets and every seeded
  column reproduce byte-for-byte and a reseed stays an in-place upsert on metric_id.

Why not ``DGPConfig.include_seasonality``: that flag configures the patient-level
causal DGPs (``DGPType.TIME_SERIES``) and is read by nothing;
``BusinessMetricsGenerator`` takes a ``GeneratorConfig``. A toggle here would let
the Monday cron, the reseed and the gap arbiter disagree — the profile is
unconditional by design.
"""

from __future__ import annotations

from datetime import date
from types import MappingProxyType
from typing import Mapping

#: Deviation from 1.0 in basis points, keyed by calendar month (1 = January).
SEASONAL_DEVIATION_BP: Mapping[int, int] = MappingProxyType(
    {
        1: -800,  # January: deductible reset / prior-auth re-verification -> the trough
        2: -400,  # February: most patients still below deductible
        3: 100,
        4: 100,
        5: 100,
        6: -100,  # mild summer dip begins (fewer office visits)
        7: -300,
        8: -200,
        9: 100,
        10: 300,
        11: 300,
        12: 800,  # December: deductibles met, year-end fills -> the peak
    }
)

#: Metric types the profile applies to. Everything else is identity.
SEASONAL_METRIC_TYPES: frozenset[str] = frozenset({"trx", "nrx", "nbrx"})


def seasonal_factor(metric_type: str, metric_date: date) -> float:
    """Multiplicative calendar factor for one row's ``value`` (1.0 when not seasonal)."""
    if metric_type not in SEASONAL_METRIC_TYPES:
        return 1.0
    return 1.0 + SEASONAL_DEVIATION_BP[metric_date.month] / 10_000
