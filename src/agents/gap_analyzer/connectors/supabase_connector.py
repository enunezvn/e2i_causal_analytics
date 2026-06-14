"""
Supabase Data Connector for Gap Analyzer.

Production data connector that fetches performance data from the business_metrics
table via BusinessMetricRepository.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from src.memory.services.factories import ServiceConnectionError

logger = logging.getLogger(__name__)


class SupabaseDataConnector:
    """
    Production data connector using BusinessMetricRepository.

    Replaces MockDataConnector for production use.
    Fetches real pharmaceutical KPI data from Supabase.
    """

    def __init__(self, supabase_client=None, include_synthetic: bool = False):
        """
        Initialize connector with optional Supabase client.

        Args:
            supabase_client: Optional Supabase client. If not provided, it is
                resolved lazily and fail-closed via ``get_async_supabase_client()``
                on first use (the #845 family: a client-less repo silently no-ops,
                so we resolve a real client instead of returning []).
            include_synthetic: When True, repository reads opt in to synthetic rows
                (the validation layer; #851). Default False keeps real-mode isolation.
        """
        self._repository = None
        self._client = supabase_client
        self._client_resolved = supabase_client is not None
        self.include_synthetic = include_synthetic

    async def _ensure_repository(self):
        """Resolve a real async Supabase client (fail-closed) and build the repo.

        The factory constructs this connector WITHOUT a client. A client-less
        ``BusinessMetricRepository`` guards every method with ``if not self.client:
        return []`` — i.e. it silently returns no data (the #845 fail-OPEN family).
        We instead resolve the service-role async client lazily; if Supabase is
        unconfigured, ``get_async_supabase_client()`` RAISES (fail-closed), which is
        surfaced as an error by the agent rather than fabricated as "no gaps".
        """
        from src.repositories.business_metric import BusinessMetricRepository

        if self._repository is not None:
            return self._repository

        if not self._client_resolved:
            from src.memory.services.factories import get_async_supabase_client

            # Raises ServiceConnectionError when Supabase is unconfigured (fail-closed).
            self._client = await get_async_supabase_client()
            self._client_resolved = True

        self._repository = BusinessMetricRepository(self._client)
        return self._repository

    @property
    def repository(self):
        """Return a BusinessMetricRepository bound to an ALREADY-RESOLVED client.

        FAIL-CLOSED: this sync property must NEVER construct a client-less repo. If no
        client was injected and none has been resolved yet, building ``Repository(None)``
        here would cache a silent-no-op repo that the async fetch path then reuses
        (every read returns []), re-opening the exact #845 fail-OPEN hole. Async callers
        must go through ``await self._ensure_repository()`` (which resolves the client
        fail-closed); this property only serves an injected/resolved client.
        """
        if self._repository is not None:
            return self._repository
        if self._client is None:
            raise ServiceConnectionError(
                "Supabase",
                "SupabaseDataConnector.repository accessed before a client was resolved; "
                "use the async path (await connector.fetch_*) so the client is resolved "
                "fail-closed",
            )
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
        period_role: Literal["current", "prior"] = "current",
    ) -> pd.DataFrame:
        """
        Fetch current period performance data from business_metrics.

        Args:
            brand: Brand name (e.g., 'Remibrutinib', 'Fabhalta', 'Kisqali')
            metrics: List of KPI names to fetch
            segments: List of segment dimensions (e.g., ['region', 'specialty'])
            time_period: Time period string (e.g., 'Q4_2024', 'YTD')
            filters: Optional additional filters
            period_role: ``"current"`` (default) or ``"prior"`` — labels the empty-fetch
                log so a benign YoY-window miss is not reported as a current-period
                alarm (#929 observability). See the empty-fetch branch below.

        Returns:
            DataFrame with performance data indexed by segment
        """
        # FAIL-CLOSED: NO broad `except Exception → empty frame` here. Any read failure
        # (ServiceConnectionError, postgrest.APIError, httpx timeouts/transport errors,
        # or any unexpected client error) must PROPAGATE so the gap_detector node
        # records it (status='failed') rather than laundering it into "no data / no
        # gaps" (#845/#851 fail-OPEN family). An EMPTY frame here means genuinely no
        # matching rows, never a swallowed error.
        start_date, end_date = self._parse_time_period(time_period)

        repository = await self._ensure_repository()

        # Fetch data for each metric
        all_data = []
        for metric in metrics:
            records = await repository.get_time_series(
                kpi_name=metric,
                brand=brand,
                start_date=start_date,
                end_date=end_date,
                include_synthetic=self.include_synthetic,
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
            # #929 observability: distinguish the CURRENT fetch from the PRIOR (YoY)
            # fetch. ``fetch_prior_period`` shifts the window back by one period
            # length; for any wide ``time_period`` that shifted window predates the
            # data, so the prior fetch is empty on EVERY successful run. Emitting the
            # same "No data found" WARNING for it reads like fabrication when it is
            # the expected, benign case — so the prior miss is a labelled INFO and a
            # genuine current-period miss stays a WARNING.
            if period_role == "prior":
                logger.info(
                    "No prior-period (YoY) data for brand=%s metrics=%s window=%s..%s "
                    "— expected when the comparison window predates available data; "
                    "not an error.",
                    brand,
                    metrics,
                    start_date,
                    end_date,
                )
            else:
                logger.warning(
                    "No current-period data found for brand=%s metrics=%s window=%s..%s",
                    brand,
                    metrics,
                    start_date,
                    end_date,
                )
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
        # FAIL-CLOSED: no broad swallow — read errors propagate (see
        # fetch_performance_data). Calculate prior period date range.
        start_date, end_date = self._parse_time_period(time_period)
        period_days = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days

        # Shift dates back by period length (YoY comparison)
        prior_end = datetime.fromisoformat(start_date) - timedelta(days=1)
        prior_start = prior_end - timedelta(days=period_days)

        prior_period = f"{prior_start.strftime('%Y-%m-%d')}_{prior_end.strftime('%Y-%m-%d')}"

        # Fetch prior period using same logic (propagates read errors). Tag the role
        # so an empty YoY window is logged as a benign INFO, not a false alarm (#929).
        return await self.fetch_performance_data(
            brand=brand,
            metrics=metrics,
            segments=segments,
            time_period=prior_period,
            filters=None,
            period_role="prior",
        )

    async def health_check(self) -> bool:
        """
        Verify database connectivity.

        Returns:
            True if database is accessible
        """
        try:
            # Try to get a simple snapshot to verify connectivity
            repository = await self._ensure_repository()
            await repository.get_latest_snapshot(
                "Remibrutinib", include_synthetic=self.include_synthetic
            )
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
