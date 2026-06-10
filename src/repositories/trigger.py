"""
Trigger Repository.

Handles HCP triggers with change tracking.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository


class TriggerRepository(BaseRepository):
    """
    Repository for triggers table.

    Supports:
    - HCP trigger queries
    - Change tracking (WS2 Change-Fail Rate)
    - Reason filtering

    Table schema:
    - trigger_id (PK)
    - trigger_timestamp (TIMESTAMPTZ)
    - patient_id, hcp_id
    - brand (brand_type)
    - delivery_status, acceptance_status
    - change_type, change_failed (WS2 Change-Fail Rate KPI)
    """

    table_name = "triggers"
    model_class = None  # Set to Trigger model when available
    HAS_PROVENANCE = True  # triggers carries is_synthetic (Shard 01)

    async def get_by_hcp(
        self,
        hcp_id: str,
        limit: int = 100,
    ) -> List:
        """
        Get triggers for a specific HCP.

        Args:
            hcp_id: Healthcare provider identifier
            limit: Maximum records

        Returns:
            List of Trigger records
        """
        return await self.get_many(
            filters={"hcp_id": hcp_id},
            limit=limit,
        )

    async def get_by_brand(
        self,
        brand: str,
        limit: int = 1000,
    ) -> List:
        """
        Get triggers for a specific brand.

        Args:
            brand: Brand name
            limit: Maximum records

        Returns:
            List of Trigger records
        """
        return await self.get_many(
            filters={"brand": brand},
            limit=limit,
        )

    async def get_recent_triggers(
        self,
        days: int = 7,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get triggers from the last N days.

        Args:
            days: Number of days to look back
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Recent triggers ordered by timestamp descending
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        # Calculate the cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        query = (
            self.client.table(self.table_name)
            .select("*")
            .gte("trigger_timestamp", cutoff_date.isoformat())
        )
        query = apply_provenance_filter(query, include_synthetic)
        result = await query.order("trigger_timestamp", desc=True).limit(limit).execute()

        return [self._to_model(row) for row in result.data]

    async def get_by_patient(
        self,
        patient_id: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get triggers for a specific patient.

        Args:
            patient_id: Patient identifier
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of Trigger records ordered by timestamp descending
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        query = self.client.table(self.table_name).select("*").eq("patient_id", patient_id)
        query = apply_provenance_filter(query, include_synthetic)
        result = await query.order("trigger_timestamp", desc=True).limit(limit).execute()

        return [self._to_model(row) for row in result.data]

    async def get_change_fail_rate(
        self,
        brand: Optional[str] = None,
        days: int = 30,
        include_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate the Change-Fail Rate (WS2 KPI).

        Change-Fail Rate = (triggers with change_failed=True) / (total triggers with changes)

        Args:
            brand: Optional brand filter
            days: Number of days to look back
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Dict with change_fail_rate, total_changes, failed_changes
        """
        if not self.client:
            return {
                "change_fail_rate": 0.0,
                "total_changes": 0,
                "failed_changes": 0,
            }

        from src.repositories.provenance import apply_provenance_filter

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Build query for triggers with changes
        query = (
            self.client.table(self.table_name)
            .select("change_type, change_failed")
            .gte("trigger_timestamp", cutoff_date.isoformat())
            .not_.is_("change_type", "null")  # Only triggers with changes
        )

        if brand:
            query = query.eq("brand", brand)

        query = apply_provenance_filter(query, include_synthetic)
        result = await query.limit(10000).execute()

        if not result.data:
            return {
                "change_fail_rate": 0.0,
                "total_changes": 0,
                "failed_changes": 0,
            }

        total_changes = len(result.data)
        failed_changes = sum(1 for row in result.data if row.get("change_failed") is True)

        return {
            "change_fail_rate": failed_changes / total_changes if total_changes > 0 else 0.0,
            "total_changes": total_changes,
            "failed_changes": failed_changes,
        }

    async def get_trigger_acceptance_rate(
        self,
        brand: Optional[str] = None,
        days: int = 30,
        include_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate trigger acceptance rate.

        Acceptance Rate = (accepted triggers) / (delivered triggers)

        Args:
            brand: Optional brand filter
            days: Number of days to look back
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Dict with acceptance_rate, total_delivered, total_accepted
        """
        if not self.client:
            return {
                "acceptance_rate": 0.0,
                "total_delivered": 0,
                "total_accepted": 0,
            }

        from src.repositories.provenance import apply_provenance_filter

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        query = (
            self.client.table(self.table_name)
            .select("delivery_status, acceptance_status")
            .gte("trigger_timestamp", cutoff_date.isoformat())
            .eq("delivery_status", "delivered")
        )

        if brand:
            query = query.eq("brand", brand)

        query = apply_provenance_filter(query, include_synthetic)
        result = await query.limit(10000).execute()

        if not result.data:
            return {
                "acceptance_rate": 0.0,
                "total_delivered": 0,
                "total_accepted": 0,
            }

        total_delivered = len(result.data)
        total_accepted = sum(1 for row in result.data if row.get("acceptance_status") == "accepted")

        return {
            "acceptance_rate": total_accepted / total_delivered if total_delivered > 0 else 0.0,
            "total_delivered": total_delivered,
            "total_accepted": total_accepted,
        }

    async def get_triggers_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        brand: Optional[str] = None,
        limit: int = 5000,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get triggers within a date range.

        Args:
            start_date: Start of range
            end_date: End of range
            brand: Optional brand filter
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Triggers within range, ordered by timestamp
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        query = (
            self.client.table(self.table_name)
            .select("*")
            .gte("trigger_timestamp", start_date.isoformat())
            .lte("trigger_timestamp", end_date.isoformat())
        )

        if brand:
            query = query.eq("brand", brand)

        query = apply_provenance_filter(query, include_synthetic)
        result = await query.order("trigger_timestamp", desc=True).limit(limit).execute()

        return [self._to_model(row) for row in result.data]
