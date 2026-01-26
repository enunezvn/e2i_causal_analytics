"""
Business Metrics Generator.

Generates synthetic business metrics for Gap Analyzer agent.
Produces time-series metrics per brand/region combination.
"""

from typing import Dict, List, Optional
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .base import BaseGenerator, GeneratorConfig
from ..config import Brand, RegionEnum


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

        # Calculate how many time points we need
        brands = [b.value for b in Brand if b.value not in ("competitor", "other")]
        regions = [r.value for r in RegionEnum]
        metric_types = list(self.METRIC_CONFIGS.keys())

        # Number of combinations per time point
        combos_per_date = len(brands) * len(regions) * len(metric_types)
        n_dates = max(1, n // combos_per_date)

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

    def _generate_date_range(self, n_months: int) -> List[date]:
        """Generate monthly date range."""
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
        config = self.METRIC_CONFIGS[metric_type]
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
        metric_id_hex = "".join(
            format(self._rng.integers(0, 256), "02x") for _ in range(6)
        )

        # Round value and target first
        rounded_value = round(value, 2)
        rounded_target = round(target, 2)

        # Calculate achievement rate from rounded values
        achievement_rate = rounded_value / rounded_target if rounded_target > 0 else 0

        return {
            "metric_id": f"metric_{metric_id_hex}",
            "metric_date": metric_date.isoformat(),
            "metric_type": metric_type,
            "metric_name": config["description"],
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
