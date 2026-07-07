"""
Business Metric Repository.

Handles KPI snapshots and metric queries.
"""

import logging
from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class BusinessMetricRepository(BaseRepository):
    """
    Repository for business_metrics table.

    Supports:
    - KPI value queries
    - Time series retrieval
    - Brand/region filtering

    Table schema:
    - metric_id (PK)
    - metric_date (DATE)
    - metric_name (VARCHAR)
    - brand (brand_type)
    - value, target, achievement_rate, roi, etc.
    """

    table_name = "business_metrics"
    id_column = "metric_id"  # live PK (#894: .eq("id") was a latent 42703)
    model_class = None  # Set to BusinessMetric model when available
    HAS_PROVENANCE = True  # business_metrics carries is_synthetic (Shard 01)

    async def get_by_kpi(
        self,
        kpi_name: str,
        brand: Optional[str] = None,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get metrics for a specific KPI.

        Args:
            kpi_name: KPI identifier (metric_name in table)
            brand: Optional brand filter
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of BusinessMetric records
        """
        filters = {"metric_name": kpi_name}
        if brand:
            filters["brand"] = brand
        return await self.get_many(
            filters=filters, limit=limit, include_synthetic=include_synthetic
        )

    async def get_time_series(
        self,
        kpi_name: str,
        brand: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get time series data for a KPI.

        Args:
            kpi_name: KPI identifier (metric_name in table)
            brand: Brand name
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format)
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Time-ordered list of metrics (ascending by date)
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("metric_name", kpi_name)
            .eq("brand", brand)
        )

        if start_date:
            query = query.gte("metric_date", start_date)
        if end_date:
            query = query.lte("metric_date", end_date)

        from src.repositories.provenance import apply_provenance_filter

        query = apply_provenance_filter(query, include_synthetic)

        # Order by date ascending for time series
        result = await query.order("metric_date", desc=False).limit(1000).execute()

        return [self._to_model(row) for row in result.data]

    async def query_metrics(
        self,
        filters: Optional[Dict[str, Any]] = None,
        since: Optional[str] = None,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get metrics by equality filters with an optional date lower bound.

        Unlike ``get_many`` this supports a ``metric_date >= since`` window and
        returns rows newest-first, which "last N days" readers (chatbot KPI
        queries) need.

        Args:
            filters: Column-value equality filters (brand, region, metric_name)
            since: Optional inclusive lower bound for metric_date (YYYY-MM-DD)
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of BusinessMetric records, newest first
        """
        if not self.client:
            return []

        query = self.client.table(self.table_name).select("*")

        for column, value in (filters or {}).items():
            query = query.eq(column, value)

        if since:
            query = query.gte("metric_date", since)

        from src.repositories.provenance import apply_provenance_filter

        query = apply_provenance_filter(query, include_synthetic)

        result = await query.order("metric_date", desc=True).limit(limit).execute()

        return [self._to_model(row) for row in result.data]

    async def get_latest_snapshot(
        self,
        brand: str,
        include_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Get the latest snapshot of all KPIs for a brand.

        Returns the most recent value for each metric_name.

        Args:
            brand: Brand name
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Dict of metric_name to {value, target, achievement_rate, date}
        """
        if not self.client:
            return {}

        from src.repositories.provenance import apply_provenance_filter

        # Get all metrics for brand, ordered by name then date DESC
        # This ensures for each metric_name, the first row is the latest
        query = (
            self.client.table(self.table_name)
            .select("metric_name, metric_date, value, target, achievement_rate, roi")
            .eq("brand", brand)
        )
        query = apply_provenance_filter(query, include_synthetic)
        result = await (
            query.order("metric_name").order("metric_date", desc=True).limit(5000).execute()
        )

        if not result.data:
            return {}

        # Deduplicate: keep only the first (latest) row per metric_name
        snapshot = {}
        seen_metrics = set()

        for row in result.data:
            metric_name = row.get("metric_name")
            if metric_name and metric_name not in seen_metrics:
                snapshot[metric_name] = {
                    "value": row.get("value"),
                    "target": row.get("target"),
                    "achievement_rate": row.get("achievement_rate"),
                    "roi": row.get("roi"),
                    "date": row.get("metric_date"),
                }
                seen_metrics.add(metric_name)

        return snapshot

    async def get_by_region(
        self,
        region: str,
        brand: Optional[str] = None,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get metrics filtered by region.

        Args:
            region: Region identifier
            brand: Optional brand filter
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of BusinessMetric records
        """
        filters = {"region": region}
        if brand:
            filters["brand"] = brand
        return await self.get_many(
            filters=filters, limit=limit, include_synthetic=include_synthetic
        )

    async def get_by_region_paged(
        self,
        region: str,
        brand: Optional[str] = None,
        include_synthetic: bool = False,
        columns: str = "*",
        page_size: int = 5000,
        max_pages: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Fetch EVERY row for ``(region[, brand])`` by paging to exhaustion.

        ``get_by_region`` (via ``get_many``) reads a SINGLE ``.limit()`` window with no
        ``ORDER BY``. A caller that AGGREGATES those rows — ``BenchmarkStore``'s per-
        segment means — therefore computes the mean over a truncated, arbitrarily-ordered
        sample once a ``(brand, region)`` slice exceeds the window, biasing the P75/P90
        benchmark (#931, the per-VALUE sibling of the #929 segment-NAME drop). This pages
        PK-ordered ``.range()`` windows until a short page proves the slice exhausted — the
        same blessed idiom as the #929 ``get_distinct_values`` fix and
        ``dispatcher._resolve_gap_inputs`` (#874 R2). PK (``id_column``) ordering makes
        OFFSET paging deterministic (no skip/duplicate across pages, even under concurrent
        writes). Paging is **cap-agnostic**: each iteration advances by the rows actually
        returned and stops only on an EMPTY page, so it stays correct even if PostgREST
        caps a response below ``page_size`` (``db-max-rows`` differs per environment — a
        ``page_size``-stride + short-page terminator would silently drop the capped tail).
        ``max_pages`` is a runaway guard that WARNs — a bounded result is never silently
        truncated.

        Operational errors PROPAGATE (no broad swallow): ``region``/``brand`` are real
        columns, so there is no 42703-unsupported-column case here (unlike
        ``get_distinct_values``); any failure surfaces rather than fabricating "no rows".

        Args:
            region: Segment value to filter on (``region`` enum).
            brand: Optional brand filter.
            include_synthetic: When True, do not exclude synthetic rows (opt-in).
            columns: PostgREST select list (default ``"*"``); callers may narrow it to
                the columns they aggregate to cut transfer (e.g. ``"metric_name,value"``).
            page_size: Rows per ``.range()`` window (cap-agnostic; need not match
                PostgREST ``db-max-rows``).
            max_pages: Runaway guard on the number of windows paged; warns if hit.

        Returns:
            Raw row dicts for the whole slice (the caller reads them dict-style, mirroring
            ``get_by_region`` whose ``model_class`` is None).
        """
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")

        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        rows: List[Dict[str, Any]] = []
        exhausted = False
        offset = 0
        for _page in range(max_pages):
            query = self.client.table(self.table_name).select(columns).eq("region", region)
            if brand:
                query = query.eq("brand", brand)
            query = apply_provenance_filter(query, include_synthetic)
            # PK-ordered .range() window — see the completeness note above.
            query = query.order(self.id_column).range(offset, offset + page_size - 1)
            result = await query.execute()

            page_rows = result.data or []
            # Cap-agnostic termination: advance by the rows ACTUALLY returned and stop
            # only on an EMPTY page. PostgREST may cap a response below page_size
            # (db-max-rows differs per environment — CI's fresh DB, a future prod config);
            # advancing by page_size and stopping on a short page would then SKIP the
            # capped tail and silently truncate. Advancing by len(page_rows) tiles the
            # slice for ANY cap, and an empty page is the only proof of exhaustion.
            if not page_rows:
                exhausted = True
                break
            rows.extend(page_rows)
            offset += len(page_rows)

        if not exhausted:
            logger.warning(
                "business_metrics get_by_region_paged for region=%s brand=%s hit the "
                "max_pages=%d page bound (page_size=%d) before exhausting the slice; rows "
                "beyond it are omitted from the aggregate.",
                region,
                brand,
                max_pages,
                page_size,
            )
        return rows

    async def get_distinct_values(
        self,
        column: str,
        brand: Optional[str] = None,
        include_synthetic: bool = False,
        page_size: int = 5000,
        max_pages: int = 1000,
    ) -> List[str]:
        """
        Discover the distinct values of a column present in the data (data-driven).

        Replaces hardcoded value lists (e.g. title-case regions) in the benchmark
        store. The live ``region`` enum is lowercase (``northeast``/``south``/…), so a
        hardcoded ``Northeast`` matches nothing; discovering actual values makes the
        store case- and value-agnostic for ANY segment column (#851). If ``column`` is
        not a real column on ``business_metrics`` (e.g. ``specialty``/``hcp_tier``,
        which this table does not carry), the query returns no rows → ``[]``, and the
        gap math then yields no gaps for that segment rather than fabricating any.

        **Completeness (#929).** A single ``.limit()`` window with no ``ORDER BY``
        SILENTLY drops any distinct value whose rows fall outside that arbitrary
        window once a brand exceeds the window size (Remibrutinib has 7289 rows /
        4 regions; the old ``.limit(5000)`` read omitted ``west`` entirely, so the
        cross-segment P75/P90 benchmark was computed over only 3 of 4 regions). We
        therefore page through ``.range()`` windows **ordered by the PK** until an
        EMPTY page proves the slice is exhausted — the same blessed idiom the
        gap-orchestration probe uses (``dispatcher._resolve_gap_inputs``, #874 R2)
        and the crystallizer candidate scan (#694). PK ordering makes OFFSET paging
        deterministic (no skip/duplicate across pages, even under concurrent writes).

        Paging is **cap-agnostic** (#938): each iteration advances by the rows actually
        returned and stops only on an EMPTY page, so it stays correct even if PostgREST
        caps a response below ``page_size`` (``db-max-rows`` differs per environment — a
        ``page_size``-stride + short-page terminator would silently drop the capped tail).
        Paging is bounded by ``max_pages`` purely as a runaway guard — for any real brand
        the slice is exhausted in a couple of pages; hitting the bound emits a WARNING so a
        truncated result is **never silent** (failing closed on the bound would be worse —
        MORE data must never disable a bindable substrate; #874 R2).

        Args:
            column: The column to extract distinct values from (e.g. "region").
            brand: Optional brand filter.
            include_synthetic: When True, do not exclude synthetic rows (opt-in).
            page_size: Rows per ``.range()`` window (cap-agnostic; need not match
                PostgREST ``db-max-rows``).
            max_pages: Runaway guard on the number of windows paged; warns if hit.

        Returns:
            Sorted list of distinct non-null values as strings.
        """
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")

        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        values: set[str] = set()
        exhausted = False
        offset = 0
        for page in range(max_pages):
            try:
                query = self.client.table(self.table_name).select(column)
                if brand:
                    query = query.eq("brand", brand)
                query = apply_provenance_filter(query, include_synthetic)
                # PK-ordered .range() window — see the completeness note above.
                query = query.order(self.id_column).range(offset, offset + page_size - 1)
                result = await query.execute()
            except Exception as e:
                # ONLY swallow the specific "undefined column" case (PostgREST 42703)
                # AND only on the FIRST page: a missing column raises 42703 on the very
                # first ``.select()``, so a requested segment like specialty/hcp_tier
                # that this table does not carry always fails soft to [] here and the
                # caller treats it as an unsupported segment. A 42703 surfacing AFTER a
                # page has succeeded is anomalous; it — and EVERY other failure
                # (connection, auth, provenance-filter, any operational error) — MUST
                # surface rather than return a silently-PARTIAL set. Laundering it would
                # re-open the #845/#851 fail-OPEN hole ("no benchmark data" == "DB is down").
                # The ``raise`` discards any ``values`` accumulated on prior pages — the
                # caller receives the exception, never a partial distinct set.
                if getattr(e, "code", None) == "42703" and page == 0:
                    return []
                raise

            rows = result.data or []
            # Cap-agnostic termination (#938): advance by the rows ACTUALLY returned and
            # stop only on an EMPTY page. PostgREST may cap a response below page_size
            # (db-max-rows differs per environment — CI's fresh DB, a future prod config);
            # advancing by page_size and stopping on a short page would then SKIP the
            # capped tail and silently drop distinct values. Advancing by len(rows) tiles
            # the slice for ANY cap, and an empty page is the only proof of exhaustion.
            if not rows:
                exhausted = True
                break
            for row in rows:
                val = row.get(column) if isinstance(row, dict) else getattr(row, column, None)
                if val:
                    values.add(str(val))
            offset += len(rows)

        if not exhausted:
            logger.warning(
                "business_metrics distinct '%s' scan for brand=%s hit the "
                "max_pages=%d page bound (page_size=%d) before exhausting the slice; "
                "distinct values beyond it may be missed.",
                column,
                brand,
                max_pages,
                page_size,
            )
        return sorted(values)

    async def get_distinct_regions(
        self,
        brand: Optional[str] = None,
        include_synthetic: bool = False,
        page_size: int = 5000,
        max_pages: int = 1000,
    ) -> List[str]:
        """Convenience wrapper: distinct ``region`` values (see get_distinct_values)."""
        return await self.get_distinct_values(
            "region",
            brand=brand,
            include_synthetic=include_synthetic,
            page_size=page_size,
            max_pages=max_pages,
        )

    async def get_achievement_summary(
        self,
        brand: str,
        include_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Get achievement rate summary for a brand.

        Args:
            brand: Brand name
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Dict with summary statistics:
            - avg_achievement: Average achievement rate
            - metrics_at_target: Count of metrics at or above target
            - metrics_below_target: Count of metrics below target
            - total_metrics: Total unique metrics
        """
        if not self.client:
            return {
                "avg_achievement": 0,
                "metrics_at_target": 0,
                "metrics_below_target": 0,
                "total_metrics": 0,
            }

        # Get latest snapshot first
        snapshot = await self.get_latest_snapshot(brand, include_synthetic=include_synthetic)

        if not snapshot:
            return {
                "avg_achievement": 0,
                "metrics_at_target": 0,
                "metrics_below_target": 0,
                "total_metrics": 0,
            }

        # Calculate summary
        achievements = [
            m["achievement_rate"]
            for m in snapshot.values()
            if m.get("achievement_rate") is not None
        ]

        at_target = sum(1 for a in achievements if a >= 1.0)
        below_target = sum(1 for a in achievements if a < 1.0)

        return {
            "avg_achievement": sum(achievements) / len(achievements) if achievements else 0,
            "metrics_at_target": at_target,
            "metrics_below_target": below_target,
            "total_metrics": len(snapshot),
        }

    async def get_roi_summary(
        self,
        brand: str,
        include_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Get ROI summary for a brand across all metrics.

        Args:
            brand: Brand name
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Dict with ROI statistics
        """
        if not self.client:
            return {"avg_roi": 0, "max_roi": 0, "min_roi": 0, "total_value": 0}

        snapshot = await self.get_latest_snapshot(brand, include_synthetic=include_synthetic)

        if not snapshot:
            return {"avg_roi": 0, "max_roi": 0, "min_roi": 0, "total_value": 0}

        rois = [m["roi"] for m in snapshot.values() if m.get("roi") is not None]
        values = [m["value"] for m in snapshot.values() if m.get("value") is not None]

        return {
            "avg_roi": sum(rois) / len(rois) if rois else 0,
            "max_roi": max(rois) if rois else 0,
            "min_roi": min(rois) if rois else 0,
            "total_value": sum(values) if values else 0,
        }
