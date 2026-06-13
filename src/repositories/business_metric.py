"""
Business Metric Repository.

Handles KPI snapshots and metric queries.
"""

from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository


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

    async def get_distinct_values(
        self,
        column: str,
        brand: Optional[str] = None,
        include_synthetic: bool = False,
        limit: int = 5000,
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

        Args:
            column: The column to extract distinct values from (e.g. "region").
            brand: Optional brand filter.
            include_synthetic: When True, do not exclude synthetic rows (opt-in).
            limit: Cap on scanned rows for distinct extraction.

        Returns:
            Sorted list of distinct non-null values as strings.
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        try:
            query = self.client.table(self.table_name).select(column)
            if brand:
                query = query.eq("brand", brand)
            query = apply_provenance_filter(query, include_synthetic)
            result = await query.limit(limit).execute()
        except Exception as e:
            # ONLY swallow the specific "undefined column" case (PostgREST 42703):
            # a requested segment like specialty/hcp_tier that this table does not
            # carry → fail soft to [] so the caller treats it as an unsupported
            # segment. EVERY OTHER failure (connection, auth, provenance-filter, any
            # operational error) MUST surface — laundering it into [] would re-open
            # the #845/#851 fail-OPEN hole ("no benchmark data" == "DB is down").
            if getattr(e, "code", None) == "42703":
                return []
            raise

        values = set()
        for row in result.data or []:
            val = row.get(column) if isinstance(row, dict) else getattr(row, column, None)
            if val:
                values.add(str(val))
        return sorted(values)

    async def get_distinct_regions(
        self,
        brand: Optional[str] = None,
        include_synthetic: bool = False,
        limit: int = 5000,
    ) -> List[str]:
        """Convenience wrapper: distinct ``region`` values (see get_distinct_values)."""
        return await self.get_distinct_values(
            "region", brand=brand, include_synthetic=include_synthetic, limit=limit
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
