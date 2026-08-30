"""
Business Metrics Generator.

Generates synthetic business metrics for Gap Analyzer agent.
Produces time-series metrics per brand/region combination.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Brand, RegionEnum
from .base import BaseGenerator, GeneratorConfig


@dataclass(frozen=True)
class BrandRegionEvent:
    """A calendar-anchored STEP shock on ``value`` only (#1833).

    From ``start`` (a month start) onward every row of ``brand`` x ``region``
    whose metric_type is in ``metric_types`` is multiplied by ``factor``. Steps
    never revert: while the gap analyzer's 90-day current/prior windows
    straddle ``start`` the shock is a *temporal* story, afterwards it persists
    as a *level* story. Events on the same (brand, region, metric) compound.
    """

    brand: str
    region: str
    metric_types: Tuple[str, ...]
    start: date
    factor: float
    label: str


class BusinessMetricsGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for business metrics time-series data.

    Creates metrics for each brand/region combination over time:
    - TRx (total prescriptions)
    - NRx (new prescriptions)
    - market_share
    - conversion_rate
    - hcp_engagement_score

    Metrics include targets, achievement rates, and statistical measures.

    Trend indexing (#1566 D1): each record's value/target scale by
    ``trend_factor = 1 + trend * month_idx``. By default month_idx is the
    date's POSITION in the current run's date list, so a single-date run
    (e.g. a monthly frontier cohort) always gets month_idx=0. Setting
    ``config.trend_origin`` switches month_idx to absolute calendar months
    since that origin, keeping short runs on the same trend line as a longer
    frozen base. ``trend_origin=None`` preserves the positional behavior
    byte-for-byte.

    Derived-field caveat (#1566 D2, deferred — documented, not changed):
    ``year_over_year_change``, ``month_over_month_change`` and ``roi`` are
    MODEL-PARAMETER DRAWS, not measurements over the generated series:
    ``yoy = trend*12 + N(0, 0.05)``, ``mom = trend + N(0, 0.02)``, and
    ``roi`` is a level draw (``2.5 + N(0, 0.5)`` for trx/nrx, else
    ``1.5 + N(0, 0.3)``). No spend column exists anywhere in the schema, so
    ROI has no arithmetic basis in the data at all. Consumers must not
    reconcile these stamps against month-over-month arithmetic on ``value``
    (that inconsistency is exactly what #1566 observed). Closed-form
    derivation from the value series is deferred (issue #1566, D2 in the
    local plans backlog).

    Brand x region geography (#1833). ``value`` (never ``target``) carries two
    deterministic, RNG-free terms so each brand has its OWN weakest region
    and the gap analyzer can rank it (before #1833 the only regional term was
    the market-size ``REGION_FACTORS`` on both value and target, which
    cancels in every gap, leaving i.i.d. noise — all three brands' #1 gap was
    "west" by coincidence). Planted matrix (``BRAND_REGION_PERFORMANCE``,
    execution factors on value; market-size-weighted mean 1.0 per brand so
    national scale is unchanged: Kisqali 0.997 / Fabhalta 0.996 /
    Remibrutinib 0.998)::

                       northeast  south  midwest  west    weakest (planted)
        Kisqali          1.09     0.97    0.86    1.04    midwest
        Fabhalta         1.03     0.86    0.98    1.10    south
        Remibrutinib     1.00     1.08    1.04    0.88    west

    Anchored step events (``BRAND_REGION_EVENTS``; trx + nrx + market_share;
    value only; never revert; compound) with their true effects::

        Kisqali/midwest      x0.88 from 2026-05-01, x0.85 from 2026-10-01
                             (compounded 0.748; x0.86 execution = 0.643 of
                             the market-size line from 2026-10)
        Fabhalta/south       x0.88 from 2026-06-01, x0.85 from 2026-11-01
        Remibrutinib/west    x0.88 from 2026-06-01, x0.85 from 2026-11-01

    Measured national TRx effect (regenerated vs the pre-#1833 DB, 2026-08):
    Kisqali x0.970, Fabhalta x0.975, Remibrutinib x0.980; all months x0.996 /
    x0.995 / x0.998. The pre-#1833 frozen base (seed 42, n=10000, start
    2013-01-01) reproduces the DB byte-for-byte on every non-value column,
    so the reseed is an in-place upsert on ``metric_id``
    (``scripts/reseed_business_metrics_aggregate.py``).

    Empirical arbiter (``scripts/gap_arbiter_1833.py``, real connector /
    benchmark-store / gap_detector / ROI / prioritizer code over the
    regenerated frame, production request defaults): the #1 opportunity is
    the planted region for all three brands at the 2026-08-30 frontier and at
    each of the next six monthly positions (also on the 5th/15th of the
    month). Winning gap types: Kisqali midwest market_share vs_benchmark
    (20-30%, ROI 62-88x), Fabhalta south market_share vs_benchmark (10-32%,
    ROI 11-43x), Remibrutinib west trx vs_target (13%) then market_share
    vs_target (8.5-25%, ROI 5.7-25x); the best non-planted opportunity never
    exceeds ROI 15x (northeast trx temporal noise). Two measured facts shape
    the design: (1) the production benchmark store is UN-windowed (all-history
    per-region means / P75 / P90), so under the 2%/month trx trend the
    current 90-day level sits ~1.6x above every historical bar and a level
    factor cannot surface on trx — it surfaces through market_share (trend
    0.005/month) and, while the 90+90-day windows straddle a step, as a
    temporal gap; (2) the matrix alone and the events alone each FAIL
    Remibrutinib (0 gaps at two positions, then noise regions win); both
    together pass, and the second step re-arms the temporal story once the
    first is absorbed into both windows.
    """

    # Metric configurations by type
    METRIC_CONFIGS: Dict[str, Dict] = {
        "trx": {
            "description": "Total Prescriptions",
            "base_values": {"Remibrutinib": 15000, "Fabhalta": 8000, "Kisqali": 50000},
            "volatility": 0.15,
            "trend": 0.02,  # 2% monthly growth
        },
        "nrx": {
            "description": "New Prescriptions",
            "base_values": {"Remibrutinib": 3000, "Fabhalta": 1500, "Kisqali": 10000},
            "volatility": 0.20,
            "trend": 0.03,
        },
        "market_share": {
            "description": "Market Share Percentage",
            "base_values": {"Remibrutinib": 0.12, "Fabhalta": 0.08, "Kisqali": 0.25},
            "volatility": 0.05,
            "trend": 0.005,
        },
        "conversion_rate": {
            "description": "HCP Conversion Rate",
            "base_values": {"Remibrutinib": 0.15, "Fabhalta": 0.12, "Kisqali": 0.22},
            "volatility": 0.10,
            "trend": 0.01,
        },
        "hcp_engagement_score": {
            "description": "HCP Engagement Score (0-10)",
            "base_values": {"Remibrutinib": 6.5, "Fabhalta": 5.8, "Kisqali": 7.2},
            "volatility": 0.08,
            "trend": 0.005,
        },
    }

    # Regional MARKET-SIZE factors — brand-independent, applied to BOTH value
    # and target (so they cancel in every gap the analyzer computes). This is
    # the region-richness ordering scripts/backfill_segment_engagement.py mirrors.
    REGION_FACTORS: Dict[str, float] = {
        "northeast": 1.15,
        "south": 0.95,
        "midwest": 0.90,
        "west": 1.00,
    }

    # #1833: brand x region EXECUTION factors, applied to ``value`` ONLY (targets
    # stay on the market-size trend line). Deterministic lookup — consumes no
    # RNG draws, so metric_ids / dates / targets / every other column reproduce
    # byte-identically and a reseed is an in-place upsert on metric_id.
    # Each row is market-size-weighted mean 1.0 (+-1%) under REGION_FACTORS so
    # per-brand national scale is unchanged (pinned by test). The weakest
    # region differs per brand — that is the planted per-brand geography:
    #   Kisqali      (HR+/HER2- breast cancer): midwest   (community-oncology
    #                 pathway adherence lag; NE academic centers over-index)
    #   Fabhalta     (PNH, rare disease):       south     (thin hematology
    #                 referral network; west/NE centers of excellence over-index)
    #   Remibrutinib (CSU):                     west      (late Kaiser/IDN
    #                 formulary access; south allergy-practice density over-index)
    BRAND_REGION_PERFORMANCE: Dict[str, Dict[str, float]] = {
        "Kisqali": {"northeast": 1.09, "south": 0.97, "midwest": 0.86, "west": 1.04},
        "Fabhalta": {"northeast": 1.03, "south": 0.86, "midwest": 0.98, "west": 1.10},
        "Remibrutinib": {"northeast": 1.00, "south": 1.08, "midwest": 1.04, "west": 0.88},
    }

    # #1833: calendar-anchored STEP shocks on ``value`` only (see
    # BrandRegionEvent). A level factor cancels in the temporal gap (both 90-day
    # windows carry it) and the production benchmark store aggregates ALL
    # history (un-windowed), so with the +2%/month trend the current window is
    # always above the historical target/P75/P90 bars — the anchored events are
    # what actually reach the gap analyzer's ranking. Two staggered steps per
    # brand keep a temporal shortfall visible across ~7 consecutive monthly
    # frontier positions (a single step is straddled by the 90+90-day windows
    # for only ~6 months). True effects are documented per event; the arbiter
    # (scripts/gap_arbiter_1833.py) is what tuned the magnitudes.
    # First steps are -12% (a step in a region that is <=25% of national moves
    # the brand's frontier-month national scale by <=3.0%, the #1640 substrate
    # note's tolerance); the later compounding steps are -15%.
    BRAND_REGION_EVENTS: Tuple[BrandRegionEvent, ...] = (
        BrandRegionEvent(
            brand="Kisqali",
            region="midwest",
            metric_types=("trx", "nrx", "market_share"),
            start=date(2026, 5, 1),
            factor=0.88,
            label="midwest IDN/PBM formulary exclusion (step -12% from 2026-05)",
        ),
        BrandRegionEvent(
            brand="Kisqali",
            region="midwest",
            metric_types=("trx", "nrx", "market_share"),
            start=date(2026, 10, 1),
            factor=0.85,
            label="oral-SERD competitor launch in midwest community oncology (step -15% from 2026-10)",
        ),
        BrandRegionEvent(
            brand="Fabhalta",
            region="south",
            metric_types=("trx", "nrx", "market_share"),
            start=date(2026, 6, 1),
            factor=0.88,
            label="south PNH center-of-excellence referral pathway loss (step -12% from 2026-06)",
        ),
        BrandRegionEvent(
            brand="Fabhalta",
            region="south",
            metric_types=("trx", "nrx", "market_share"),
            start=date(2026, 11, 1),
            factor=0.85,
            label="south Medicaid prior-authorization tightening (step -15% from 2026-11)",
        ),
        BrandRegionEvent(
            brand="Remibrutinib",
            region="west",
            metric_types=("trx", "nrx", "market_share"),
            start=date(2026, 6, 1),
            factor=0.88,
            label="west Kaiser/IDN formulary step-edit (step -12% from 2026-06)",
        ),
        BrandRegionEvent(
            brand="Remibrutinib",
            region="west",
            metric_types=("trx", "nrx", "market_share"),
            start=date(2026, 11, 1),
            factor=0.85,
            label="west biologic competitor copay program (step -15% from 2026-11)",
        ),
    )

    @classmethod
    def brand_region_factor(
        cls, brand: str, region: str, metric_type: str, metric_date: date
    ) -> float:
        """The deterministic value-only multiplier for one row (#1833).

        Persistent execution factor times every planted step event active on
        ``metric_date`` for this brand/region/metric. Unknown brands or
        regions (e.g. ``competitor``) are identity. NO RNG.
        """
        factor = cls.BRAND_REGION_PERFORMANCE.get(brand, {}).get(region, 1.0)
        for event in cls.BRAND_REGION_EVENTS:
            if (
                event.brand == brand
                and event.region == region
                and metric_type in event.metric_types
                and metric_date >= event.start
            ):
                factor *= event.factor
        return factor

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "business_metrics"

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the business metrics generator.

        Args:
            config: Generator configuration.
        """
        super().__init__(config)

    def generate(self) -> pd.DataFrame:
        """
        Generate business metrics time-series.

        Returns:
            DataFrame with business metrics matching schema.
        """
        n = self.config.n_records
        self._log(f"Generating {n} business metrics records")

        # Canonical v1.1: lowercase gap-connector metric_name set ONLY.
        # NO title-case alias rows (those are get_kpi_summary RPC response keys
        # over treatment_events, not business_metrics.metric_name filters).
        brands = [b.value for b in Brand if b.value not in ("competitor", "other")]
        regions = [r.value for r in RegionEnum]
        # METRIC_CONFIGS already carries the full lowercase set incl. nrx; the
        # guard below is a defensive no-op so the connector key set is complete
        # even if a future edit drops nrx from METRIC_CONFIGS.
        metric_types = list(self.METRIC_CONFIGS.keys())
        if "nrx" not in metric_types:
            metric_types.append("nrx")

        # Number of combinations per time point (+1 headroom so df.head(n)
        # trimming does not clip a whole brand/region/metric combo off a date).
        combos_per_date = len(brands) * len(regions) * len(metric_types)
        n_dates = max(1, n // (combos_per_date + 1))

        # Generate date range
        dates = self._generate_date_range(n_dates)

        # #1566 D1: month_idx drives trend_factor = 1 + trend * month_idx. With
        # trend_origin set, use absolute calendar months from that origin so a
        # single-date run stays on the same trend line as a longer base run;
        # None keeps the positional index (byte-identical legacy behavior).
        origin = self.config.trend_origin

        records = []
        for metric_date in dates:
            if origin is not None:
                month_idx = (metric_date.year - origin.year) * 12 + (
                    metric_date.month - origin.month
                )
            else:
                month_idx = dates.index(metric_date)
            for brand in brands:
                for region in regions:
                    for metric_type in metric_types:
                        record = self._generate_metric_record(
                            metric_date=metric_date,
                            brand=brand,
                            region=region,
                            metric_type=metric_type,
                            month_idx=month_idx,
                        )
                        records.append(record)

        df = pd.DataFrame(records)

        # Assign data splits based on dates
        df["data_split"] = self._assign_splits(df["metric_date"].astype(str).tolist())

        # Trim to requested size
        if len(df) > n:
            df = df.head(n)

        self._log(f"Generated {len(df)} business metrics records")
        return df

    # The GeneratorConfig default start (base.py) — the 2022 staleness root.
    _STALE_DEFAULT_START = date(2022, 1, 1)

    def _generate_date_range(self, n_months: int) -> List[date]:
        """Generate monthly date range.

        Recency: when the config still carries the stale 2022 default start,
        anchor the window to end at the current run month so the KPI 30-day
        window sees rows (full per-run rolling-window stamping is delivered by
        Shard 04; this only guarantees the column is populated, not 2022). An
        explicitly-provided non-default start_date is honored as-is.
        """
        if self.config.start_date == self._STALE_DEFAULT_START:
            from datetime import timedelta  # noqa: F401  (kept for clarity/parity)

            today = date.today()
            anchor = date(today.year, today.month, 1)
            # Walk back (n_months - 1) months from the current month so the
            # forward walk below ends at the current month.
            start = anchor
            for _ in range(max(0, n_months - 1)):
                start = (
                    date(start.year - 1, 12, 1)
                    if start.month == 1
                    else date(start.year, start.month - 1, 1)
                )
        else:
            start = self.config.start_date

        dates = []
        current = date(start.year, start.month, 1)

        for _ in range(n_months):
            dates.append(current)
            # Move to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        return dates

    def _generate_metric_record(
        self,
        metric_date: date,
        brand: str,
        region: str,
        metric_type: str,
        month_idx: int,
    ) -> Dict:
        """
        Generate a single metric record.

        Args:
            metric_date: Date of the metric.
            brand: Brand name.
            region: Geographic region.
            metric_type: Type of metric.
            month_idx: Index for trend calculation.

        Returns:
            Dictionary with metric data.
        """
        # Tolerate a metric_type appended by the nrx guard but absent from
        # METRIC_CONFIGS (defensive): fall back to the trx prescription model.
        config = self.METRIC_CONFIGS.get(metric_type) or self.METRIC_CONFIGS["trx"]
        base_value = config["base_values"].get(brand, config["base_values"]["Kisqali"] * 0.5)
        volatility = config["volatility"]
        trend = config["trend"]

        # Apply regional adjustment
        region_factor = self.REGION_FACTORS.get(region, 1.0)

        # Calculate value with trend and noise. #1833: the brand x region
        # execution factor and anchored events multiply VALUE only (targets
        # below stay on the market-size line) and consume no RNG.
        trend_factor = 1 + (trend * month_idx)
        noise = self._rng.normal(0, volatility)
        value = (
            base_value
            * region_factor
            * trend_factor
            * (1 + noise)
            * self.brand_region_factor(brand, region, metric_type, metric_date)
        )

        # Ensure non-negative values
        value = max(0, value)

        # For percentage metrics, cap at 1.0
        if metric_type in ("market_share", "conversion_rate"):
            value = min(value, 1.0)

        # For engagement score, cap at 10
        if metric_type == "hcp_engagement_score":
            value = min(value, 10.0)

        # Generate target (typically 5-15% above current trailing average)
        target_multiplier = 1 + self._rng.uniform(0.05, 0.15)
        target = base_value * region_factor * trend_factor * target_multiplier

        # Calculate achievement rate
        achievement_rate = value / target if target > 0 else 0

        # Calculate YoY change (simulated)
        yoy_change = trend * 12 + self._rng.normal(0, 0.05)

        # Calculate MoM change
        mom_change = trend + self._rng.normal(0, 0.02)

        # Calculate ROI (for prescription metrics)
        if metric_type in ("trx", "nrx"):
            roi = 2.5 + self._rng.normal(0, 0.5)
        else:
            roi = 1.5 + self._rng.normal(0, 0.3)

        # Statistical measures
        sample_size = self._rng.integers(500, 5000)
        std_error = value * volatility / np.sqrt(sample_size)

        confidence_interval_lower = value - 1.96 * std_error
        confidence_interval_upper = value + 1.96 * std_error

        # P-value simulation (most are significant)
        stat_sig = self._rng.uniform(0.001, 0.10)

        # Use seeded RNG for reproducible metric_id
        metric_id_hex = "".join(format(self._rng.integers(0, 256), "02x") for _ in range(6))

        # Round value and target first
        rounded_value = round(value, 2)
        rounded_target = round(target, 2)

        # Calculate achievement rate from rounded values
        achievement_rate = rounded_value / rounded_target if rounded_target > 0 else 0

        return {
            "metric_id": f"metric_{metric_id_hex}",
            "metric_date": metric_date.isoformat(),
            "metric_type": metric_type,
            # Canonical v1.1: gap_analyzer queries business_metrics on
            # metric_name == lowercase connector key (business_metric.py:79
            # .eq("metric_name", kpi_name)). Emit the key, not the description.
            "metric_name": metric_type,
            "brand": brand,
            "region": region,
            "value": rounded_value,
            "target": rounded_target,
            "achievement_rate": round(achievement_rate, 3),
            "year_over_year_change": round(yoy_change, 3),
            "month_over_month_change": round(mom_change, 3),
            "roi": round(roi, 2),
            "statistical_significance": round(stat_sig, 3),
            "confidence_interval_lower": round(confidence_interval_lower, 2),
            "confidence_interval_upper": round(confidence_interval_upper, 2),
            "sample_size": int(sample_size),
        }
