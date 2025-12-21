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
    model_class = None  # Set to BusinessMetric model when available

    async def get_by_kpi(
        self,
        kpi_name: str,
        brand: Optional[str] = None,
        limit: int = 100,
    ) -> List:
        """
        Get metrics for a specific KPI.

        Args:
            kpi_name: KPI identifier (metric_name in table)
            brand: Optional brand filter
            limit: Maximum records

        Returns:
            List of BusinessMetric records
        """
        filters = {"metric_name": kpi_name}
        if brand:
            filters["brand"] = brand
        return await self.get_many(filters=filters, limit=limit)

    async def get_time_series(
        self,
        kpi_name: str,
        brand: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List:
        """
        Get time series data for a KPI.

        Args:
            kpi_name: KPI identifier (metric_name in table)
            brand: Brand name
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format)

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

        # Order by date ascending for time series
        result = await query.order("metric_date", desc=False).limit(1000).execute()

        return [self._to_model(row) for row in result.data]

    async def get_latest_snapshot(
        self,
        brand: str,
    ) -> Dict[str, Any]:
        """
        Get the latest snapshot of all KPIs for a brand.

        Returns the most recent value for each metric_name.

        Args:
            brand: Brand name

        Returns:
            Dict of metric_name to {value, target, achievement_rate, date}
        """
        if not self.client:
            return {}

        # Get all metrics for brand, ordered by name then date DESC
        # This ensures for each metric_name, the first row is the latest
        result = await (
            self.client.table(self.table_name)
            .select("metric_name, metric_date, value, target, achievement_rate, roi")
            .eq("brand", brand)
            .order("metric_name")
            .order("metric_date", desc=True)
            .limit(5000)
            .execute()
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
    ) -> List:
        """
        Get metrics filtered by region.

        Args:
            region: Region identifier
            brand: Optional brand filter
            limit: Maximum records

        Returns:
            List of BusinessMetric records
        """
        filters = {"region": region}
        if brand:
            filters["brand"] = brand
        return await self.get_many(filters=filters, limit=limit)

    async def get_achievement_summary(
        self,
        brand: str,
    ) -> Dict[str, Any]:
        """
        Get achievement rate summary for a brand.

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
        snapshot = await self.get_latest_snapshot(brand)

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
    ) -> Dict[str, Any]:
        """
        Get ROI summary for a brand across all metrics.

        Returns:
            Dict with ROI statistics
        """
        if not self.client:
            return {"avg_roi": 0, "max_roi": 0, "min_roi": 0, "total_value": 0}

        snapshot = await self.get_latest_snapshot(brand)

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
