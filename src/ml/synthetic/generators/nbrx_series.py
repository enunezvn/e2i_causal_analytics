"""NBRx monthly series for ``business_metrics`` — generated BESIDE the frozen stream.

Canonical TRx lane (owner decision 2026-09-15): NBRx becomes a canonical
business_metrics KPI, and no ``nbrx`` metric_type existed. Adding it to
``BusinessMetricsGenerator.METRIC_CONFIGS`` is unworkable (measured by reading
the generator): every cell draws 7 variates + 6 metric_id bytes in
``dates x brands x regions x metric_types`` order, so a sixth metric type would
shift every later metric_id and target, change ``combos_per_date`` 60 -> 72
(the frozen base would shrink from 163 to 136 months) and clip the 61-row
monthly cohort — the literal DB fingerprints and the reseed's target-drift
refusal would both fail.

Instead this pass reads an ALREADY-GENERATED frame and emits one nbrx row per
trx (month, brand, region) cell:

* its own RNG per calendar month, ``default_rng([42, 7007, year, month])``,
  drawing the same 7 variates for EVERY brand x region cell in fixed order (a
  cell absent from the input still consumes its draws), so a month's values
  are a pure function of the month — the frozen-base regeneration and the
  Monday cron cohort agree;
* content-addressed ids ``nbrx_<YYYYMM>_<brand>_<region>`` (no RNG), disjoint
  from ``metric_<hex>`` base ids and ``m<YYMM>_<NNNN>`` cohort ids;
* ``data_split`` copied from the sibling trx row;
* the same value model shape as the generator (market-size REGION_FACTORS,
  2013-01 trend origin, brand x region execution factor + planted events,
  calendar seasonality), value-only terms RNG-free.

The input frame is never mutated and its RNG is never touched.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import Brand, RegionEnum
from . import seasonality
from .business_metrics_generator import BusinessMetricsGenerator

NBRX_METRIC = "nbrx"
NBRX_ID_PREFIX = "nbrx"
#: Must equal frontier_append.BM_TREND_ORIGIN (pinned by test; not imported to
#: avoid a generators -> frontier_append import cycle).
NBRX_TREND_ORIGIN = date(2013, 1, 1)
NBRX_BASE_SEED = 42
NBRX_SEED_TAG = 7007  # WS3-BI-007
NBRX_MODEL: Dict[str, Any] = {
    "base_values": {"Remibrutinib": 1200.0, "Fabhalta": 600.0, "Kisqali": 4000.0},
    "volatility": 0.20,
    "trend": 0.03,
}
BRANDS: Tuple[str, ...] = tuple(b.value for b in Brand if b.value not in ("competitor", "other"))
REGIONS: Tuple[str, ...] = tuple(r.value for r in RegionEnum)

_COLUMNS = [
    "metric_id",
    "metric_date",
    "metric_type",
    "metric_name",
    "brand",
    "region",
    "value",
    "target",
    "achievement_rate",
    "year_over_year_change",
    "month_over_month_change",
    "roi",
    "statistical_significance",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "sample_size",
    "data_split",
]


def nbrx_metric_id(metric_date: date, brand: str, region: str) -> str:
    return (
        f"{NBRX_ID_PREFIX}_{metric_date.year:04d}{metric_date.month:02d}_{brand.lower()}_{region}"
    )


def nbrx_month_rng(metric_date: date) -> np.random.Generator:
    return np.random.default_rng(
        [NBRX_BASE_SEED, NBRX_SEED_TAG, metric_date.year, metric_date.month]
    )


def generate_nbrx_rows(bm_frame: pd.DataFrame) -> pd.DataFrame:
    """One nbrx row per trx (metric_date, brand, region) cell of ``bm_frame``."""
    trx = bm_frame[bm_frame["metric_type"] == "trx"]
    base_values = NBRX_MODEL["base_values"]
    vol = float(NBRX_MODEL["volatility"])
    trend = float(NBRX_MODEL["trend"])
    records: List[Dict[str, Any]] = []
    for metric_date_raw, month_rows in trx.groupby("metric_date", sort=True):
        d = date.fromisoformat(str(metric_date_raw)[:10])
        split_by_cell = {
            (str(r.brand), str(r.region)): r.data_split for r in month_rows.itertuples()
        }
        rng = nbrx_month_rng(d)
        month_idx = (d.year - NBRX_TREND_ORIGIN.year) * 12 + (d.month - NBRX_TREND_ORIGIN.month)
        trend_factor = 1 + trend * month_idx
        for brand in BRANDS:
            for region in REGIONS:
                # Draw for EVERY cell first so an absent cell never shifts another.
                noise = rng.normal(0, vol)
                target_multiplier = 1 + rng.uniform(0.05, 0.15)
                yoy = trend * 12 + rng.normal(0, 0.05)
                mom = trend + rng.normal(0, 0.02)
                roi = 2.5 + rng.normal(0, 0.5)
                sample_size = int(rng.integers(500, 5000))
                stat_sig = rng.uniform(0.001, 0.10)
                if (brand, region) not in split_by_cell:
                    continue
                base_value = base_values[brand]
                region_factor = BusinessMetricsGenerator.REGION_FACTORS.get(region, 1.0)
                value = max(
                    0.0,
                    base_value
                    * region_factor
                    * trend_factor
                    * (1 + noise)
                    * BusinessMetricsGenerator.brand_region_factor(brand, region, NBRX_METRIC, d)
                    * seasonality.seasonal_factor(NBRX_METRIC, d),
                )
                target = base_value * region_factor * trend_factor * target_multiplier
                std_error = value * vol / np.sqrt(sample_size)
                rounded_value, rounded_target = round(value, 2), round(target, 2)
                records.append(
                    {
                        "metric_id": nbrx_metric_id(d, brand, region),
                        "metric_date": d.isoformat(),
                        "metric_type": NBRX_METRIC,
                        "metric_name": NBRX_METRIC,
                        "brand": brand,
                        "region": region,
                        "value": rounded_value,
                        "target": rounded_target,
                        "achievement_rate": round(
                            rounded_value / rounded_target if rounded_target > 0 else 0, 3
                        ),
                        "year_over_year_change": round(yoy, 3),
                        "month_over_month_change": round(mom, 3),
                        "roi": round(roi, 2),
                        "statistical_significance": round(stat_sig, 3),
                        "confidence_interval_lower": round(value - 1.96 * std_error, 2),
                        "confidence_interval_upper": round(value + 1.96 * std_error, 2),
                        "sample_size": sample_size,
                        "data_split": split_by_cell[(brand, region)],
                    }
                )
    return pd.DataFrame(records, columns=_COLUMNS)


def with_nbrx(bm_frame: pd.DataFrame) -> pd.DataFrame:
    """The ONE seam every entrypoint uses: ``bm_frame`` with its nbrx rows appended.

    The input is never mutated and its rows keep their order, so the cohort re-key
    and the frozen-base identity are untouched (canonical TRx lane, codex r1).
    Applying it twice would append duplicate content-addressed ids, so a frame that
    already carries nbrx rows is refused."""
    already = bm_frame["metric_name"] == NBRX_METRIC
    if already.any():
        raise ValueError(
            f"bm_frame already carries nbrx rows ({int(already.sum())}); "
            "with_nbrx must be applied exactly once"
        )
    return pd.concat([bm_frame, generate_nbrx_rows(bm_frame)], ignore_index=True)
