"""
Business Metric Repository.

Handles KPI snapshots and metric queries.
"""

from typing import List, Optional, Dict, Any
from src.repositories.base import BaseRepository


class BusinessMetricRepository(BaseRepository):
    """
    Repository for business_metrics table.

    Supports:
    - KPI value queries
    - Time series retrieval
    - Brand/region filtering
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
            kpi_name: KPI identifier
            brand: Optional brand filter
            limit: Maximum records

        Returns:
            List of BusinessMetric records
        """
        filters = {"kpi_name": kpi_name}
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
            kpi_name: KPI identifier
            brand: Brand name
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)

        Returns:
            Time-ordered list of metrics
        """
        # TODO: Implement with date range filtering
        return await self.get_many(
            filters={"kpi_name": kpi_name, "brand": brand},
            limit=1000,
        )

    async def get_latest_snapshot(
        self,
        brand: str,
    ) -> Dict[str, Any]:
        """
        Get the latest snapshot of all KPIs for a brand.

        Args:
            brand: Brand name

        Returns:
            Dict of KPI name to latest value
        """
        # TODO: Implement with distinct on and order by
        return {}
