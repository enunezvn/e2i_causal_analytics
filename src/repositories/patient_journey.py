"""
Patient Journey Repository.

Handles patient journey data with source tracking and ML split awareness.
"""

from typing import Any, Dict, List, Optional

from src.repositories.base import SplitAwareRepository


class PatientJourneyRepository(SplitAwareRepository):
    """
    Repository for patient_journeys table.

    Supports:
    - Multi-source data tracking
    - Cross-source match queries
    - Split-aware ML data access
    """

    table_name = "patient_journeys"
    model_class = None  # Set to PatientJourney model when available

    async def get_by_brand(
        self,
        brand: str,
        split: Optional[str] = None,
        limit: int = 1000,
    ) -> List:
        """
        Get patient journeys for a specific brand.

        Args:
            brand: Brand name (e.g., 'Kisqali', 'Fabhalta', 'Remibrutinib')
            split: Optional ML split filter
            limit: Maximum records

        Returns:
            List of PatientJourney records
        """
        return await self.get_many(
            filters={"brand": brand},
            split=split,
            limit=limit,
        )

    async def get_cross_source_matches(
        self,
        brand: str,
        limit: int = 1000,
    ) -> List:
        """
        Get patients with multi-source data.

        These are patients where source_stacking_flag = True,
        indicating data from multiple sources has been matched.

        Args:
            brand: Brand name
            limit: Maximum records

        Returns:
            List of patients with cross-source data
        """
        return await self.get_many(
            filters={
                "brand": brand,
                "source_stacking_flag": True,
            },
            limit=limit,
        )

    async def get_by_journey_stage(
        self,
        brand: str,
        stage: str,
        split: Optional[str] = None,
    ) -> List:
        """
        Get patients at a specific journey stage.

        Args:
            brand: Brand name
            stage: Journey stage (e.g., 'awareness', 'consideration', 'conversion')
            split: Optional ML split filter

        Returns:
            List of PatientJourney records
        """
        return await self.get_many(
            filters={
                "brand": brand,
                "journey_stage": stage,
            },
            split=split,
        )

    async def get_data_freshness(self, brand: str) -> Dict[str, Any]:
        """
        Calculate data freshness metrics for a brand.

        Queries patient_journeys.data_lag_hours for:
        - avg_lag_hours: Average data lag in hours
        - max_lag_hours: Maximum data lag
        - stale_count: Records with lag > 24 hours (threshold for "stale")
        - total_records: Total records evaluated

        Args:
            brand: Brand name

        Returns:
            Dict with avg_lag_hours, max_lag_hours, stale_count, total_records
        """
        if not self.client:
            return {
                "avg_lag_hours": 0,
                "max_lag_hours": 0,
                "stale_count": 0,
                "total_records": 0,
            }

        # Query patient_journeys for data_lag_hours filtered by brand
        result = await (
            self.client.table(self.table_name)
            .select("data_lag_hours")
            .eq("brand", brand)
            .not_.is_("data_lag_hours", "null")
            .limit(10000)
            .execute()
        )

        if not result.data:
            return {
                "avg_lag_hours": 0,
                "max_lag_hours": 0,
                "stale_count": 0,
                "total_records": 0,
            }

        # Aggregate in Python
        lag_hours = [row.get("data_lag_hours", 0) or 0 for row in result.data]
        stale_threshold = 24  # Hours - consider data stale if > 24 hours old

        return {
            "avg_lag_hours": sum(lag_hours) / len(lag_hours) if lag_hours else 0,
            "max_lag_hours": max(lag_hours) if lag_hours else 0,
            "stale_count": sum(1 for h in lag_hours if h > stale_threshold),
            "total_records": len(lag_hours),
        }

    async def get_freshness_by_source(
        self,
        brand: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get data freshness metrics grouped by data source.

        Args:
            brand: Optional brand filter

        Returns:
            Dict mapping data_source to freshness metrics
        """
        if not self.client:
            return {}

        query = (
            self.client.table(self.table_name)
            .select("data_source, data_lag_hours")
            .not_.is_("data_lag_hours", "null")
        )

        if brand:
            query = query.eq("brand", brand)

        result = await query.limit(10000).execute()

        if not result.data:
            return {}

        # Group by data_source
        source_data: Dict[str, List[int]] = {}
        for row in result.data:
            source = row.get("data_source") or "unknown"
            lag = row.get("data_lag_hours", 0) or 0
            if source not in source_data:
                source_data[source] = []
            source_data[source].append(lag)

        # Compute metrics per source
        stale_threshold = 24
        metrics = {}
        for source, lags in source_data.items():
            metrics[source] = {
                "avg_lag_hours": sum(lags) / len(lags) if lags else 0,
                "max_lag_hours": max(lags) if lags else 0,
                "stale_count": sum(1 for h in lags if h > stale_threshold),
                "record_count": len(lags),
            }

        return metrics

    async def get_journey_stage_distribution(
        self,
        brand: str,
    ) -> Dict[str, int]:
        """
        Get distribution of patients across journey stages.

        Args:
            brand: Brand name

        Returns:
            Dict mapping journey_stage to count
        """
        if not self.client:
            return {}

        result = await (
            self.client.table(self.table_name)
            .select("journey_stage")
            .eq("brand", brand)
            .limit(10000)
            .execute()
        )

        if not result.data:
            return {}

        # Count by stage
        stage_counts: Dict[str, int] = {}
        for row in result.data:
            stage = row.get("journey_stage") or "unknown"
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        return stage_counts

    async def get_source_stacking_metrics(
        self,
        brand: str,
    ) -> Dict[str, Any]:
        """
        Get cross-source matching and stacking metrics.

        Supports WS1 Cross-source Match and Stacking Lift KPIs.

        Args:
            brand: Brand name

        Returns:
            Dict with stacking metrics:
            - total_patients: Total patient journeys
            - stacked_patients: Patients with multiple sources
            - stacking_rate: Percentage with source stacking
            - avg_match_confidence: Average cross-source match confidence
        """
        if not self.client:
            return {
                "total_patients": 0,
                "stacked_patients": 0,
                "stacking_rate": 0.0,
                "avg_match_confidence": 0.0,
            }

        result = await (
            self.client.table(self.table_name)
            .select("source_stacking_flag, source_match_confidence")
            .eq("brand", brand)
            .limit(10000)
            .execute()
        )

        if not result.data:
            return {
                "total_patients": 0,
                "stacked_patients": 0,
                "stacking_rate": 0.0,
                "avg_match_confidence": 0.0,
            }

        total = len(result.data)
        stacked = sum(1 for row in result.data if row.get("source_stacking_flag") is True)
        confidences = [
            float(row.get("source_match_confidence"))
            for row in result.data
            if row.get("source_match_confidence") is not None
        ]

        return {
            "total_patients": total,
            "stacked_patients": stacked,
            "stacking_rate": stacked / total if total > 0 else 0.0,
            "avg_match_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        }

    async def get_by_data_source(
        self,
        data_source: str,
        brand: Optional[str] = None,
        limit: int = 1000,
    ) -> List:
        """
        Get patient journeys from a specific data source.

        Args:
            data_source: Data source identifier
            brand: Optional brand filter
            limit: Maximum records

        Returns:
            List of PatientJourney records
        """
        filters = {"data_source": data_source}
        if brand:
            filters["brand"] = brand
        return await self.get_many(filters=filters, limit=limit)
