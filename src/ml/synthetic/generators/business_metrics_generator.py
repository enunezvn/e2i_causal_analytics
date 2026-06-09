"""
Business Metrics Generator.

Generates synthetic business metrics for Gap Analyzer agent.
Produces time-series metrics per brand/region combination.
"""

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Brand, RegionEnum
from .base import BaseGenerator, GeneratorConfig


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

    # Regional adjustment factors
    REGION_FACTORS: Dict[str, float] = {
        "northeast": 1.15,
        "south": 0.95,
        "midwest": 0.90,
        "west": 1.00,
    }

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

        records = []
        for metric_date in dates:
            for brand in brands:
                for region in regions:
                    for metric_type in metric_types:
                        record = self._generate_metric_record(
                            metric_date=metric_date,
                            brand=brand,
                            region=region,
                            metric_type=metric_type,
                            month_idx=dates.index(metric_date),
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

        # Calculate value with trend and noise
        trend_factor = 1 + (trend * month_idx)
        noise = self._rng.normal(0, volatility)
        value = base_value * region_factor * trend_factor * (1 + noise)

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
