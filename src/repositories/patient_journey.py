"""
Patient Journey Repository.

Handles patient journey data with source tracking and ML split awareness.
"""

from typing import List, Optional, Dict, Any
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

        Returns:
            Dict with avg_lag_hours, max_lag_hours, stale_count
        """
        # TODO: Implement with raw SQL query for aggregations
        return {
            "avg_lag_hours": 0,
            "max_lag_hours": 0,
            "stale_count": 0,
        }
