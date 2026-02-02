"""
Benchmark Store for Gap Analyzer.

Production benchmark store that retrieves targets, peer benchmarks,
and top decile performance from the business_metrics table.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkStore:
    """
    Production benchmark store using business_metrics table.

    Replaces MockBenchmarkStore for production use.
    Provides targets, peer benchmarks, and top decile metrics.
    """

    def __init__(self, supabase_client=None):
        """
        Initialize store with optional Supabase client.

        Args:
            supabase_client: Optional Supabase client
        """
        self._repository = None
        self._client = supabase_client

    @property
    def repository(self):
        """Lazy-load BusinessMetricRepository."""
        if self._repository is None:
            from src.repositories.business_metric import BusinessMetricRepository

            self._repository = BusinessMetricRepository(self._client)
        return self._repository

    async def get_targets(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
    ) -> pd.DataFrame:
        """
        Get target values from business_metrics.target column.

        Args:
            brand: Brand name
            metrics: List of KPI names
            segments: List of segment dimensions

        Returns:
            DataFrame with target values indexed by segment
        """
        try:
            # Get latest snapshot which includes targets
            snapshot = await self.repository.get_latest_snapshot(brand)

            if not snapshot:
                logger.warning(f"No targets found for brand={brand}")
                return pd.DataFrame()

            # Build target data for requested metrics
            data = []
            for metric in metrics:
                if metric in snapshot:
                    metric_data = snapshot[metric]
                    row = {
                        "metric": metric,
                        "target": metric_data.get("target", 0),
                        "current_value": metric_data.get("value", 0),
                        "achievement_rate": metric_data.get("achievement_rate", 0),
                    }
                    data.append(row)

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Pivot to get metrics as columns for target values
            if not df.empty and "metric" in df.columns:
                df_pivot = df.pivot_table(
                    columns="metric",
                    values="target",
                    aggfunc="first",
                )
                # Flatten to single row DataFrame
                result = pd.DataFrame([df_pivot.to_dict()])
                return result

            return df

        except Exception as e:
            logger.error(f"Failed to get targets: {e}")
            return pd.DataFrame()

    async def get_peer_benchmarks(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
    ) -> pd.DataFrame:
        """
        Get peer benchmark data aggregated across regions.

        Calculates mean, median, P75, and P90 across all regions
        for each metric to provide peer comparison benchmarks.

        Args:
            brand: Brand name
            metrics: List of KPI names
            segments: List of segment dimensions

        Returns:
            DataFrame with peer benchmark statistics
        """
        try:
            # Get all data across regions for comparison
            all_data = []
            regions = ["Northeast", "Southeast", "Midwest", "West", "National"]

            for region in regions:
                records = await self.repository.get_by_region(
                    region=region,
                    brand=brand,
                    limit=500,
                )

                for record in records:
                    metric_name = (
                        record.get("metric_name")
                        if isinstance(record, dict)
                        else getattr(record, "metric_name", None)
                    )
                    if metric_name in metrics:
                        value = (
                            record.get("value")
                            if isinstance(record, dict)
                            else getattr(record, "value", None)
                        )
                        if value is not None:
                            all_data.append(
                                {
                                    "metric": metric_name,
                                    "region": region,
                                    "value": value,
                                }
                            )

            if not all_data:
                logger.warning(f"No peer benchmark data found for brand={brand}")
                return pd.DataFrame()

            df = pd.DataFrame(all_data)

            # Calculate peer statistics per metric
            benchmarks = []
            for metric in metrics:
                metric_values = df[df["metric"] == metric]["value"]
                if not metric_values.empty:
                    benchmarks.append(
                        {
                            "metric": metric,
                            "mean": metric_values.mean(),
                            "median": metric_values.median(),
                            "p75": metric_values.quantile(0.75),
                            "p90": metric_values.quantile(0.90),
                            "min": metric_values.min(),
                            "max": metric_values.max(),
                        }
                    )

            return pd.DataFrame(benchmarks)

        except Exception as e:
            logger.error(f"Failed to get peer benchmarks: {e}")
            return pd.DataFrame()

    async def get_top_decile(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
    ) -> pd.DataFrame:
        """
        Calculate top decile (P90) performance across all regions.

        Top decile represents best-in-class performance benchmarks.

        Args:
            brand: Brand name
            metrics: List of KPI names
            segments: List of segment dimensions

        Returns:
            DataFrame with top decile values
        """
        try:
            # Get peer benchmarks which already calculate P90
            peer_df = await self.get_peer_benchmarks(brand, metrics, segments)

            if peer_df.empty:
                return pd.DataFrame()

            # Extract P90 values as top decile
            top_decile = []
            for _, row in peer_df.iterrows():
                top_decile.append(
                    {
                        "metric": row["metric"],
                        "top_decile_value": row.get("p90", 0),
                        "peer_mean": row.get("mean", 0),
                        "gap_to_top": row.get("p90", 0) - row.get("mean", 0),
                    }
                )

            return pd.DataFrame(top_decile)

        except Exception as e:
            logger.error(f"Failed to get top decile: {e}")
            return pd.DataFrame()

    async def get_benchmark_summary(
        self,
        brand: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of benchmark data availability.

        Args:
            brand: Brand name

        Returns:
            Dict with benchmark summary info
        """
        try:
            snapshot = await self.repository.get_latest_snapshot(brand)
            achievement = await self.repository.get_achievement_summary(brand)
            roi = await self.repository.get_roi_summary(brand)

            return {
                "brand": brand,
                "total_metrics": len(snapshot),
                "metrics_with_targets": sum(
                    1 for m in snapshot.values() if m.get("target") is not None
                ),
                "avg_achievement": achievement.get("avg_achievement", 0),
                "metrics_at_target": achievement.get("metrics_at_target", 0),
                "metrics_below_target": achievement.get("metrics_below_target", 0),
                "avg_roi": roi.get("avg_roi", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get benchmark summary: {e}")
            return {"brand": brand, "error": str(e)}
