"""
Supabase Data Connector for Gap Analyzer.

Production data connector that fetches performance data from the business_metrics
table via BusinessMetricRepository.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from src.memory.services.factories import ServiceConnectionError
from src.utils.gap_time_period import ResolvedTimePeriod, resolve_time_period

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
        period: Optional[ResolvedTimePeriod] = None,
    ) -> pd.DataFrame:
        """
        Fetch current period performance data from business_metrics.

        Args:
            brand: Brand name (e.g., 'Remibrutinib', 'Fabhalta', 'Kisqali')
            metrics: List of KPI names to fetch
            segments: List of segment dimensions (e.g., ['region', 'specialty'])
            time_period: Time period label (e.g., 'current_quarter', 'Q3_2026', 'YTD')
            filters: Optional additional filters
            period_role: ``"current"`` (default) or ``"prior"`` — labels the empty-fetch
                log so a benign YoY-window miss is not reported as a current-period
                alarm (#929 observability). See the empty-fetch branch below.
            period: #1834 — the window ALREADY resolved by the caller (gap_detector
                resolves once, up front, and reports it in state). When given, its
                ``period_start``/``period_end`` are used verbatim so the reported
                window and the queried window cannot diverge (e.g. across a midnight
                clock flip between resolution and this read). When omitted (direct
                callers), ``time_period`` is resolved here.

        Returns:
            DataFrame with performance data indexed by segment
        """
        # FAIL-CLOSED: NO broad `except Exception → empty frame` here. Any read failure
        # (ServiceConnectionError, postgrest.APIError, httpx timeouts/transport errors,
        # or any unexpected client error) must PROPAGATE so the gap_detector node
        # records it (status='failed') rather than laundering it into "no data / no
        # gaps" (#845/#851 fail-OPEN family). An EMPTY frame here means genuinely no
        # matching rows, never a swallowed error.
        if period is not None:
            start_date = period.period_start.isoformat()
            end_date = period.period_end.isoformat()
        else:
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
        period: Optional[ResolvedTimePeriod] = None,
    ) -> pd.DataFrame:
        """
        Fetch prior period data for comparison.

        Args:
            brand: Brand name
            metrics: List of KPI names
            segments: List of segment dimensions
            time_period: Current time period label (the prior window is derived)
            period: #1834 — the caller's already-resolved window; when given, its
                ``prior_start``/``prior_end`` are used verbatim instead of resolving
                ``time_period`` again (see ``fetch_performance_data``).

        Returns:
            DataFrame with prior period data
        """
        # FAIL-CLOSED: no broad swallow — read errors propagate (see
        # fetch_performance_data), and an unparseable ``time_period`` raises
        # ``TimePeriodError`` (a ValueError) from the shared grammar before any read.
        #
        # #1834: the prior window comes from the grammar, not a day-count shift.
        # ``business_metrics`` rows sit on the 1st of each month, so shifting a
        # quarter-to-date window (Jul 1–Aug 30) back by its day count produced a
        # prior of May 2–Jun 30 with ONE monthly row instead of three. Calendar
        # quarters now compare against the preceding FULL quarter, MTD against the
        # preceding full month, YTD against the same span of the previous year, and
        # explicit ranges keep their length-shift aligned to the monthly grain
        # (see src.utils.gap_time_period for the exact rules).
        resolved = period if period is not None else resolve_time_period(time_period)
        # The prior window travels as an EXPLICIT range — absolute, clock-independent.
        prior_period = f"{resolved.prior_start.isoformat()}_{resolved.prior_end.isoformat()}"

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

    def _parse_time_period(self, time_period: str) -> Tuple[str, str]:
        """
        Resolve a ``time_period`` label to the CURRENT window's inclusive date range.

        Thin shim over the shared grammar (``src.utils.gap_time_period``) — the same
        one the API request model validates against and the gap_detector node
        surfaces in state. Before #1834 this method had its own four-form parser and
        a silent "last 90 days" default that swallowed the request DEFAULT
        ``current_quarter`` (measured on prod 2026-08-30: every persisted analysis
        compared Jun 1–Aug 30 against Mar 2–May 31 under a "current quarter" label).

        Args:
            time_period: 'current_quarter', 'previous_quarter'/'last_quarter',
                'Q3_2026', '2026-Q3', 'YTD', 'MTD' or '2026-07-01_2026-08-30'

        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format

        Raises:
            TimePeriodError: (a ValueError) for any other form — no fallback window.
        """
        resolved = resolve_time_period(time_period)
        return resolved.period_start.isoformat(), resolved.period_end.isoformat()
