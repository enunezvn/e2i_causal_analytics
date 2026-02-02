"""
Supabase Data Connector for Gap Analyzer.

Production data connector that fetches performance data from the business_metrics
table via BusinessMetricRepository.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SupabaseDataConnector:
    """
    Production data connector using BusinessMetricRepository.

    Replaces MockDataConnector for production use.
    Fetches real pharmaceutical KPI data from Supabase.
    """

    def __init__(self, supabase_client=None):
        """
        Initialize connector with optional Supabase client.

        Args:
            supabase_client: Optional Supabase client. If not provided,
                           will be loaded lazily from the repository.
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

    async def fetch_performance_data(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
        time_period: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fetch current period performance data from business_metrics.

        Args:
            brand: Brand name (e.g., 'Remibrutinib', 'Fabhalta', 'Kisqali')
            metrics: List of KPI names to fetch
            segments: List of segment dimensions (e.g., ['region', 'specialty'])
            time_period: Time period string (e.g., 'Q4_2024', 'YTD')
            filters: Optional additional filters

        Returns:
            DataFrame with performance data indexed by segment
        """
        try:
            # Calculate date range from time_period
            start_date, end_date = self._parse_time_period(time_period)

            # Fetch data for each metric
            all_data = []
            for metric in metrics:
                records = await self.repository.get_time_series(
                    kpi_name=metric,
                    brand=brand,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Convert to records with metric column
                for record in records:
                    row = {
                        "metric": metric,
                        "value": record.get("value")
                        if isinstance(record, dict)
                        else getattr(record, "value", None),
                        "target": record.get("target")
                        if isinstance(record, dict)
                        else getattr(record, "target", None),
                        "date": record.get("metric_date")
                        if isinstance(record, dict)
                        else getattr(record, "metric_date", None),
                    }

                    # Add segment columns if available
                    for segment in segments:
                        seg_value = (
                            record.get(segment)
                            if isinstance(record, dict)
                            else getattr(record, segment, None)
                        )
                        if seg_value:
                            row[segment] = seg_value

                    all_data.append(row)

            if not all_data:
                logger.warning(f"No data found for brand={brand}, metrics={metrics}")
                return pd.DataFrame()

            df = pd.DataFrame(all_data)

            # Pivot to get metrics as columns
            if not df.empty and "metric" in df.columns and "value" in df.columns:
                # Get segment columns
                segment_cols = [s for s in segments if s in df.columns]
                if segment_cols:
                    df_pivot = df.pivot_table(
                        index=segment_cols,
                        columns="metric",
                        values="value",
                        aggfunc="mean",  # Average if multiple values per segment
                    ).reset_index()
                    return df_pivot

            return df

        except Exception as e:
            logger.error(f"Failed to fetch performance data: {e}")
            return pd.DataFrame()

    async def fetch_prior_period(
        self,
        brand: str,
        metrics: List[str],
        segments: List[str],
        time_period: str,
    ) -> pd.DataFrame:
        """
        Fetch prior period data for comparison.

        Args:
            brand: Brand name
            metrics: List of KPI names
            segments: List of segment dimensions
            time_period: Current time period (prior will be calculated)

        Returns:
            DataFrame with prior period data
        """
        try:
            # Calculate prior period date range
            start_date, end_date = self._parse_time_period(time_period)
            period_days = (
                datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)
            ).days

            # Shift dates back by period length (YoY comparison)
            prior_end = datetime.fromisoformat(start_date) - timedelta(days=1)
            prior_start = prior_end - timedelta(days=period_days)

            prior_period = f"{prior_start.strftime('%Y-%m-%d')}_{prior_end.strftime('%Y-%m-%d')}"

            # Fetch prior period using same logic
            return await self.fetch_performance_data(
                brand=brand,
                metrics=metrics,
                segments=segments,
                time_period=prior_period,
                filters=None,
            )

        except Exception as e:
            logger.error(f"Failed to fetch prior period data: {e}")
            return pd.DataFrame()

    async def health_check(self) -> bool:
        """
        Verify database connectivity.

        Returns:
            True if database is accessible
        """
        try:
            # Try to get a simple snapshot to verify connectivity
            await self.repository.get_latest_snapshot("Remibrutinib")
            return True  # If no exception, we're connected
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def _parse_time_period(self, time_period: str) -> tuple:
        """
        Parse time period string to date range.

        Args:
            time_period: Period string like 'Q4_2024', 'YTD', '2024-01-01_2024-03-31'

        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format
        """
        today = datetime.now()

        # Handle direct date range format
        if "_" in time_period and len(time_period) == 21:  # YYYY-MM-DD_YYYY-MM-DD
            parts = time_period.split("_")
            return parts[0], parts[1]

        # Handle quarter format (Q1_2024, Q2_2024, etc.)
        if time_period.startswith("Q") and "_" in time_period:
            parts = time_period.split("_")
            quarter = int(parts[0][1])
            year = int(parts[1])

            quarter_starts = {
                1: f"{year}-01-01",
                2: f"{year}-04-01",
                3: f"{year}-07-01",
                4: f"{year}-10-01",
            }
            quarter_ends = {
                1: f"{year}-03-31",
                2: f"{year}-06-30",
                3: f"{year}-09-30",
                4: f"{year}-12-31",
            }
            return quarter_starts[quarter], quarter_ends[quarter]

        # Handle YTD
        if time_period.upper() == "YTD":
            start = f"{today.year}-01-01"
            end = today.strftime("%Y-%m-%d")
            return start, end

        # Handle MTD (month to date)
        if time_period.upper() == "MTD":
            start = f"{today.year}-{today.month:02d}-01"
            end = today.strftime("%Y-%m-%d")
            return start, end

        # Default: last 90 days
        logger.warning(f"Unknown time period format: {time_period}, using last 90 days")
        end = today.strftime("%Y-%m-%d")
        start = (today - timedelta(days=90)).strftime("%Y-%m-%d")
        return start, end
