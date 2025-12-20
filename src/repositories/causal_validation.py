"""
Causal Validation Repository.

Handles persistence of refutation test results and validation gate decisions.

Version: 4.3
Database: causal_validations table (010_causal_validation_tables.sql)
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import json
import logging

from src.repositories.base import BaseRepository
from src.causal_engine import (
    RefutationSuite,
    RefutationResult,
    RefutationTestType,
    RefutationStatus,
    GateDecision,
)

logger = logging.getLogger(__name__)


class CausalValidationRepository(BaseRepository):
    """
    Repository for causal_validations table.

    Supports:
    - Saving RefutationSuite results (bulk insert)
    - Querying validation status for estimates
    - Gate decision lookups
    - Validation history queries

    Database Schema (causal_validations):
    - validation_id: UUID PRIMARY KEY
    - estimate_id: UUID NOT NULL
    - estimate_source: VARCHAR(50) DEFAULT 'causal_paths'
    - test_type: refutation_test_type ENUM
    - status: validation_status ENUM
    - original_effect: DECIMAL(12,6)
    - refuted_effect: DECIMAL(12,6)
    - p_value: DECIMAL(6,5)
    - delta_percent: DECIMAL(8,4)
    - confidence_score: DECIMAL(4,3)
    - gate_decision: gate_decision ENUM
    - test_config: JSONB
    - details_json: JSONB
    - agent_activity_id: VARCHAR(100)
    - brand: VARCHAR(50)
    - treatment_variable: VARCHAR(100)
    - outcome_variable: VARCHAR(100)
    - data_split: VARCHAR(20)
    """

    table_name = "causal_validations"
    model_class = None  # Using raw dicts for now

    async def save_suite(
        self,
        suite: RefutationSuite,
        estimate_id: str,
        estimate_source: str = "causal_paths",
        agent_activity_id: Optional[str] = None,
        data_split: Optional[str] = None,
    ) -> List[str]:
        """
        Save all test results from a RefutationSuite.

        Creates one row per test in the causal_validations table.

        Args:
            suite: RefutationSuite with test results
            estimate_id: UUID of the causal estimate being validated
            estimate_source: Source table ('causal_paths' or 'ml_experiments')
            agent_activity_id: Optional FK to agent_activities
            data_split: Optional ML split (train, validation, test, holdout)

        Returns:
            List of created validation_id UUIDs
        """
        if not self.client:
            logger.warning("No Supabase client, skipping validation persistence")
            return []

        rows = []
        for test in suite.tests:
            row = self._test_to_row(
                test=test,
                suite=suite,
                estimate_id=estimate_id,
                estimate_source=estimate_source,
                agent_activity_id=agent_activity_id,
                data_split=data_split,
            )
            rows.append(row)

        try:
            result = await self.client.table(self.table_name).insert(rows).execute()
            validation_ids = [row["validation_id"] for row in result.data]
            logger.info(
                f"Saved {len(validation_ids)} validation records for estimate {estimate_id}"
            )
            return validation_ids
        except Exception as e:
            logger.error(f"Failed to save validation suite: {e}")
            return []

    async def save_single_test(
        self,
        test: RefutationResult,
        estimate_id: str,
        gate_decision: GateDecision,
        confidence_score: float,
        estimate_source: str = "causal_paths",
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        brand: Optional[str] = None,
        agent_activity_id: Optional[str] = None,
        data_split: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save a single refutation test result.

        Args:
            test: RefutationResult to save
            estimate_id: UUID of the causal estimate
            gate_decision: Aggregate gate decision
            confidence_score: Suite confidence score
            estimate_source: Source table
            treatment: Treatment variable name
            outcome: Outcome variable name
            brand: Brand context
            agent_activity_id: Optional FK to agent_activities
            data_split: Optional ML split

        Returns:
            Created validation_id or None on failure
        """
        if not self.client:
            return None

        row = {
            "estimate_id": estimate_id,
            "estimate_source": estimate_source,
            "test_type": test.test_name.value,
            "status": test.status.value,
            "original_effect": test.original_effect,
            "refuted_effect": test.refuted_effect,
            "p_value": test.p_value,
            "delta_percent": test.delta_percent,
            "confidence_score": confidence_score,
            "gate_decision": gate_decision.value,
            "test_config": json.dumps(test.details.get("config", {})),
            "details_json": json.dumps(test.details),
            "agent_activity_id": agent_activity_id,
            "brand": brand,
            "treatment_variable": treatment,
            "outcome_variable": outcome,
            "data_split": data_split,
        }

        try:
            result = await self.client.table(self.table_name).insert(row).execute()
            return result.data[0]["validation_id"] if result.data else None
        except Exception as e:
            logger.error(f"Failed to save validation test: {e}")
            return None

    async def get_validations_for_estimate(
        self,
        estimate_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get all validation records for a causal estimate.

        Args:
            estimate_id: UUID of the causal estimate
            limit: Maximum records to return

        Returns:
            List of validation records
        """
        return await self.get_many(
            filters={"estimate_id": estimate_id},
            limit=limit,
        )

    async def get_gate_decision(self, estimate_id: str) -> Optional[str]:
        """
        Get the aggregate gate decision for an estimate.

        Priority: block > review > proceed

        Args:
            estimate_id: UUID of the causal estimate

        Returns:
            Gate decision string or None if no validations exist
        """
        validations = await self.get_validations_for_estimate(estimate_id)
        if not validations:
            return None

        # Priority: block > review > proceed
        has_block = any(v.get("gate_decision") == "block" for v in validations)
        has_review = any(v.get("gate_decision") == "review" for v in validations)

        if has_block:
            return "block"
        elif has_review:
            return "review"
        else:
            return "proceed"

    async def get_failed_tests(
        self,
        estimate_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get failed test results for an estimate.

        Args:
            estimate_id: UUID of the causal estimate

        Returns:
            List of failed validation records
        """
        if not self.client:
            return []

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("estimate_id", estimate_id)
                .eq("status", "failed")
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get failed tests: {e}")
            return []

    async def get_blocked_estimates(
        self,
        brand: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get all blocked causal estimates.

        Args:
            brand: Optional brand filter
            limit: Maximum records

        Returns:
            List of blocked estimates with failed test details
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("gate_decision", "block")
                .limit(limit)
            )
            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get blocked estimates: {e}")
            return []

    async def get_validation_summary(
        self,
        estimate_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get aggregated validation summary for an estimate.

        Uses v_validation_summary view if available, otherwise calculates.

        Args:
            estimate_id: UUID of the causal estimate

        Returns:
            Summary dict with counts and gate decision
        """
        validations = await self.get_validations_for_estimate(estimate_id)
        if not validations:
            return None

        passed = sum(1 for v in validations if v.get("status") == "passed")
        failed = sum(1 for v in validations if v.get("status") == "failed")
        warning = sum(1 for v in validations if v.get("status") == "warning")
        skipped = sum(1 for v in validations if v.get("status") == "skipped")

        confidence_scores = [
            v.get("confidence_score", 0)
            for v in validations
            if v.get("confidence_score") is not None
        ]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        return {
            "estimate_id": estimate_id,
            "total_tests": len(validations),
            "passed_count": passed,
            "failed_count": failed,
            "warning_count": warning,
            "skipped_count": skipped,
            "avg_confidence": avg_confidence,
            "final_gate": await self.get_gate_decision(estimate_id),
            "failed_tests": [
                v.get("test_type")
                for v in validations
                if v.get("status") == "failed"
            ],
        }

    async def can_use_estimate(self, estimate_id: str) -> bool:
        """
        Check if a causal estimate can be used (not blocked).

        Args:
            estimate_id: UUID of the causal estimate

        Returns:
            True if estimate can be used (proceed or review with approval)
        """
        gate = await self.get_gate_decision(estimate_id)
        return gate != "block" if gate else True  # Allow if no validations

    async def get_recent_validations(
        self,
        brand: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get recent validation records.

        Args:
            brand: Optional brand filter
            limit: Maximum records

        Returns:
            List of recent validation records, newest first
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
            )
            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get recent validations: {e}")
            return []

    def _test_to_row(
        self,
        test: RefutationResult,
        suite: RefutationSuite,
        estimate_id: str,
        estimate_source: str,
        agent_activity_id: Optional[str] = None,
        data_split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert RefutationResult to database row.

        Args:
            test: Individual test result
            suite: Parent RefutationSuite for context
            estimate_id: UUID of the causal estimate
            estimate_source: Source table
            agent_activity_id: Optional FK
            data_split: Optional ML split

        Returns:
            Dict ready for database insertion
        """
        return {
            "estimate_id": estimate_id,
            "estimate_source": estimate_source,
            "test_type": test.test_name.value,
            "status": test.status.value,
            "original_effect": test.original_effect,
            "refuted_effect": test.refuted_effect,
            "p_value": test.p_value,
            "delta_percent": test.delta_percent,
            "confidence_score": suite.confidence_score,
            "gate_decision": suite.gate_decision.value,
            "test_config": json.dumps({
                "execution_time_ms": test.execution_time_ms,
            }),
            "details_json": json.dumps(test.details),
            "agent_activity_id": agent_activity_id,
            "brand": suite.brand,
            "treatment_variable": suite.treatment_variable,
            "outcome_variable": suite.outcome_variable,
            "data_split": data_split,
        }
