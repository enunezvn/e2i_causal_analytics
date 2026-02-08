"""
Repository for storing and retrieving data quality reports.

Handles CRUD operations for ml_data_quality_reports table.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from .base import BaseRepository

logger = logging.getLogger(__name__)


class DataQualityReportRepository(BaseRepository):
    """Repository for ml_data_quality_reports table."""

    table_name = "ml_data_quality_reports"
    model_class = None  # Using dict directly

    async def store_result(
        self,
        result_dict: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Store a DataQualityResult in the database.

        Args:
            result_dict: Dictionary from DataQualityResult.to_dict()

        Returns:
            Stored record with database-generated fields
        """
        if not self.client:
            logger.warning("No Supabase client, skipping DB store")
            return None

        try:
            # Map Python types to database types
            db_record = {
                "id": result_dict.get("id"),
                "report_name": result_dict.get("report_name"),
                "expectation_suite_name": result_dict.get("expectation_suite_name"),
                "table_name": result_dict.get("table_name"),
                "brand": result_dict.get("brand"),
                "region": result_dict.get("region"),
                "overall_status": result_dict.get("overall_status"),
                "expectations_evaluated": result_dict.get("expectations_evaluated", 0),
                "expectations_passed": result_dict.get("expectations_passed", 0),
                "expectations_failed": result_dict.get("expectations_failed", 0),
                "success_rate": result_dict.get("success_rate"),
                "failed_expectations": result_dict.get("failed_expectations", []),
                "completeness_score": result_dict.get("completeness_score"),
                "validity_score": result_dict.get("validity_score"),
                "uniqueness_score": result_dict.get("uniqueness_score"),
                "consistency_score": result_dict.get("consistency_score"),
                "timeliness_score": result_dict.get("timeliness_score"),
                "accuracy_score": result_dict.get("accuracy_score"),
                "leakage_detected": result_dict.get("leakage_detected", False),
                "data_split": result_dict.get("data_split"),
                "training_run_id": result_dict.get("training_run_id"),
                "duration_seconds": (
                    result_dict.get("validation_time_ms", 0) // 1000
                    if result_dict.get("validation_time_ms")
                    else None
                ),
            }

            # Remove None values for optional fields
            db_record = {k: v for k, v in db_record.items() if v is not None}

            result = await self.client.table(self.table_name).insert(db_record).execute()

            if result.data:
                logger.info(f"Stored DQ report: {db_record.get('report_name')}")
                return cast(Dict[str, Any], result.data[0])
            return None

        except Exception as e:
            logger.error(f"Failed to store DQ report: {e}", exc_info=True)
            return None

    async def get_latest_for_table(
        self,
        table_name: str,
        data_split: Optional[str] = None,
        limit: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get the latest data quality reports for a table.

        Args:
            table_name: Source table name
            data_split: Optional data split filter
            limit: Number of reports to return

        Returns:
            List of report records
        """
        if not self.client:
            return []

        try:
            query = self.client.table(self.table_name).select("*").eq("table_name", table_name)

            if data_split:
                query = query.eq("data_split", data_split)

            query = query.order("run_at", desc=True).limit(limit)
            result = await query.execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Failed to get DQ reports: {e}", exc_info=True)
            return []

    async def get_by_training_run(
        self,
        training_run_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all data quality reports for a training run.

        Args:
            training_run_id: Training run UUID

        Returns:
            List of report records
        """
        if not self.client:
            return []

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("training_run_id", training_run_id)
                .order("run_at", desc=True)
                .execute()
            )

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Failed to get DQ reports for run: {e}", exc_info=True)
            return []

    async def get_failed_reports(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get failed data quality reports.

        Args:
            since: Get failures since this datetime
            limit: Maximum reports to return

        Returns:
            List of failed report records
        """
        if not self.client:
            return []

        try:
            query = self.client.table(self.table_name).select("*").eq("overall_status", "failed")

            if since:
                query = query.gte("run_at", since.isoformat())

            query = query.order("run_at", desc=True).limit(limit)
            result = await query.execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Failed to get failed reports: {e}", exc_info=True)
            return []

    async def check_data_quality_gate(
        self,
        table_name: str,
    ) -> bool:
        """
        Check if data quality gate passes for a table.

        Returns True if the latest report for the table passed
        and no leakage was detected.

        Args:
            table_name: Table to check

        Returns:
            True if gate passes, False otherwise
        """
        reports = await self.get_latest_for_table(table_name, data_split="train")

        if not reports:
            return False  # No report = fail gate

        latest = reports[0]
        return latest.get("overall_status") == "passed" and not latest.get(
            "leakage_detected", False
        )


# Singleton instance
_repository: Optional[DataQualityReportRepository] = None


def get_data_quality_report_repository(
    supabase_client=None,
) -> DataQualityReportRepository:
    """Get the data quality report repository.

    Args:
        supabase_client: Optional Supabase client

    Returns:
        DataQualityReportRepository instance
    """
    global _repository
    if _repository is None:
        _repository = DataQualityReportRepository(supabase_client)
    elif supabase_client is not None:
        _repository.client = supabase_client
    return _repository
