"""
Expert Review Repository.

Handles persistence of domain expert reviews for causal DAG validation.

Version: 4.3
Database: expert_reviews table (010_causal_validation_tables.sql)
"""

from typing import List, Optional, Dict, Any
from datetime import date, timedelta
import json
import logging

from src.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ExpertReviewRepository(BaseRepository):
    """
    Repository for expert_reviews table.

    Supports:
    - Creating and updating expert reviews
    - Checking DAG approval status
    - Querying pending reviews for admin UI
    - Managing review validity periods

    Database Schema (expert_reviews):
    - review_id: UUID PRIMARY KEY
    - review_type: expert_review_type ENUM
    - dag_version_hash: VARCHAR(64)
    - reviewer_id: VARCHAR(100) NOT NULL
    - reviewer_name: VARCHAR(200)
    - reviewer_role: VARCHAR(100)
    - reviewer_email: VARCHAR(200)
    - approval_status: VARCHAR(30) DEFAULT 'pending'
    - checklist_json: JSONB
    - comments_json: JSONB
    - concerns_raised: TEXT[]
    - conditions: TEXT
    - brand: VARCHAR(50)
    - analysis_context: TEXT
    - treatment_variable: VARCHAR(100)
    - outcome_variable: VARCHAR(100)
    - valid_from: DATE
    - valid_until: DATE
    - created_at: TIMESTAMPTZ
    - updated_at: TIMESTAMPTZ
    - approved_at: TIMESTAMPTZ
    - related_validation_ids: UUID[]
    - supersedes_review_id: UUID
    """

    table_name = "expert_reviews"
    model_class = None  # Using raw dicts

    # Default validity period for expert reviews (90 days = quarterly)
    DEFAULT_VALIDITY_DAYS = 90

    async def create_review(
        self,
        reviewer_id: str,
        review_type: str,
        dag_version_hash: Optional[str] = None,
        reviewer_name: Optional[str] = None,
        reviewer_role: Optional[str] = None,
        reviewer_email: Optional[str] = None,
        brand: Optional[str] = None,
        treatment_variable: Optional[str] = None,
        outcome_variable: Optional[str] = None,
        analysis_context: Optional[str] = None,
        checklist: Optional[Dict[str, Any]] = None,
        related_validation_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Create a new expert review request.

        Args:
            reviewer_id: User ID of the reviewer
            review_type: Type of review ('initial_dag', 'quarterly_audit', 'methodology_change', 'edge_dispute')
            dag_version_hash: SHA256 hash of the DAG being reviewed
            reviewer_name: Display name of reviewer
            reviewer_role: Role (commercial_ops, medical_affairs, data_science, etc.)
            reviewer_email: Contact email
            brand: Brand context
            treatment_variable: Treatment being analyzed
            outcome_variable: Outcome being measured
            analysis_context: Description of what analysis this covers
            checklist: Initial checklist items (to be completed during review)
            related_validation_ids: Related causal_validations records

        Returns:
            Created review_id or None on failure
        """
        if not self.client:
            logger.warning("No Supabase client, skipping review creation")
            return None

        row = {
            "reviewer_id": reviewer_id,
            "review_type": review_type,
            "dag_version_hash": dag_version_hash,
            "reviewer_name": reviewer_name,
            "reviewer_role": reviewer_role,
            "reviewer_email": reviewer_email,
            "approval_status": "pending",
            "brand": brand,
            "treatment_variable": treatment_variable,
            "outcome_variable": outcome_variable,
            "analysis_context": analysis_context,
            "checklist_json": json.dumps(checklist) if checklist else None,
            "related_validation_ids": related_validation_ids,
        }

        # Remove None values
        row = {k: v for k, v in row.items() if v is not None}

        try:
            result = await self.client.table(self.table_name).insert(row).execute()
            review_id = result.data[0]["review_id"] if result.data else None
            logger.info(f"Created expert review {review_id} for DAG {dag_version_hash}")
            return review_id
        except Exception as e:
            logger.error(f"Failed to create expert review: {e}")
            return None

    async def submit_review(
        self,
        review_id: str,
        approval_status: str,
        checklist: Dict[str, Any],
        comments: Optional[Dict[str, Any]] = None,
        concerns_raised: Optional[List[str]] = None,
        conditions: Optional[str] = None,
        validity_days: int = DEFAULT_VALIDITY_DAYS,
    ) -> bool:
        """
        Submit a completed expert review.

        Args:
            review_id: UUID of the review to complete
            approval_status: 'approved' or 'rejected'
            checklist: Completed checklist with responses
            comments: Reviewer notes and feedback
            concerns_raised: List of specific concerns
            conditions: Any conditions on approval
            validity_days: Days until review expires (default 90)

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        if approval_status not in ("approved", "rejected"):
            logger.error(f"Invalid approval_status: {approval_status}")
            return False

        update_data = {
            "approval_status": approval_status,
            "checklist_json": json.dumps(checklist),
            "comments_json": json.dumps(comments) if comments else None,
            "concerns_raised": concerns_raised,
            "conditions": conditions,
        }

        if approval_status == "approved":
            valid_until = date.today() + timedelta(days=validity_days)
            update_data["valid_from"] = date.today().isoformat()
            update_data["valid_until"] = valid_until.isoformat()
            update_data["approved_at"] = "now()"

        # Remove None values
        update_data = {k: v for k, v in update_data.items() if v is not None}

        try:
            await (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("review_id", review_id)
                .execute()
            )
            logger.info(f"Submitted review {review_id} with status {approval_status}")
            return True
        except Exception as e:
            logger.error(f"Failed to submit review {review_id}: {e}")
            return False

    async def is_dag_approved(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
    ) -> bool:
        """
        Check if a DAG has an active expert approval.

        Args:
            dag_hash: SHA256 hash of the DAG structure
            brand: Optional brand filter

        Returns:
            True if DAG has active approval, False otherwise
        """
        if not self.client:
            # Default to allowing when no client (development mode)
            logger.warning("No Supabase client, assuming DAG is approved")
            return True

        try:
            query = (
                self.client.table(self.table_name)
                .select("review_id")
                .eq("dag_version_hash", dag_hash)
                .eq("approval_status", "approved")
                .gte("valid_until", date.today().isoformat())
            )

            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Failed to check DAG approval: {e}")
            return False

    async def get_dag_approval(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the active approval record for a DAG.

        Args:
            dag_hash: SHA256 hash of the DAG structure
            brand: Optional brand filter

        Returns:
            Active approval record or None if not approved
        """
        if not self.client:
            return None

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("dag_version_hash", dag_hash)
                .eq("approval_status", "approved")
                .gte("valid_until", date.today().isoformat())
                .order("approved_at", desc=True)
                .limit(1)
            )

            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get DAG approval: {e}")
            return None

    async def get_pending_reviews(
        self,
        brand: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get pending reviews for admin UI.

        Args:
            brand: Optional brand filter
            reviewer_id: Optional filter by assigned reviewer
            limit: Maximum records to return

        Returns:
            List of pending review records, oldest first
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("approval_status", "pending")
                .order("created_at", desc=False)
                .limit(limit)
            )

            if brand:
                query = query.eq("brand", brand)
            if reviewer_id:
                query = query.eq("reviewer_id", reviewer_id)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get pending reviews: {e}")
            return []

    async def get_expiring_reviews(
        self,
        days_until_expiry: int = 14,
        brand: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get reviews expiring within the specified days.

        Used for proactive renewal notifications.

        Args:
            days_until_expiry: Number of days until expiration
            brand: Optional brand filter

        Returns:
            List of soon-to-expire review records
        """
        if not self.client:
            return []

        expiry_date = (date.today() + timedelta(days=days_until_expiry)).isoformat()

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("approval_status", "approved")
                .gte("valid_until", date.today().isoformat())
                .lte("valid_until", expiry_date)
                .order("valid_until", desc=False)
            )

            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get expiring reviews: {e}")
            return []

    async def get_reviews_for_dag(
        self,
        dag_hash: str,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get all reviews for a specific DAG version.

        Args:
            dag_hash: SHA256 hash of the DAG structure
            include_expired: Whether to include expired reviews

        Returns:
            List of review records for the DAG
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("dag_version_hash", dag_hash)
                .order("created_at", desc=True)
            )

            if not include_expired:
                query = query.or_(
                    f"valid_until.gte.{date.today().isoformat()},valid_until.is.null"
                )

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get reviews for DAG: {e}")
            return []

    async def renew_review(
        self,
        original_review_id: str,
        reviewer_id: str,
        reviewer_name: Optional[str] = None,
        reviewer_role: Optional[str] = None,
        reviewer_email: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a renewal review that supersedes an existing one.

        Args:
            original_review_id: UUID of the review to renew
            reviewer_id: User ID of the new reviewer
            reviewer_name: Display name
            reviewer_role: Role
            reviewer_email: Contact email

        Returns:
            New review_id or None on failure
        """
        if not self.client:
            return None

        # Get the original review
        original = await self.get_by_id(original_review_id)
        if not original:
            logger.error(f"Original review {original_review_id} not found")
            return None

        # Create renewal review with context from original
        row = {
            "reviewer_id": reviewer_id,
            "review_type": "quarterly_audit",
            "dag_version_hash": original.get("dag_version_hash"),
            "reviewer_name": reviewer_name,
            "reviewer_role": reviewer_role,
            "reviewer_email": reviewer_email,
            "approval_status": "pending",
            "brand": original.get("brand"),
            "treatment_variable": original.get("treatment_variable"),
            "outcome_variable": original.get("outcome_variable"),
            "analysis_context": f"Renewal of review {original_review_id}",
            "supersedes_review_id": original_review_id,
        }

        # Remove None values
        row = {k: v for k, v in row.items() if v is not None}

        try:
            result = await self.client.table(self.table_name).insert(row).execute()
            review_id = result.data[0]["review_id"] if result.data else None
            logger.info(f"Created renewal review {review_id} superseding {original_review_id}")
            return review_id
        except Exception as e:
            logger.error(f"Failed to create renewal review: {e}")
            return None

    async def get_review_summary(
        self,
        brand: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get summary statistics of expert reviews.

        Args:
            brand: Optional brand filter

        Returns:
            Summary dict with counts by status
        """
        if not self.client:
            return {
                "pending": 0,
                "approved": 0,
                "rejected": 0,
                "expired": 0,
                "expiring_soon": 0,
            }

        try:
            # Get all reviews for counting
            query = self.client.table(self.table_name).select("approval_status, valid_until")
            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            reviews = result.data or []

            today = date.today()
            soon = today + timedelta(days=14)

            pending = 0
            approved = 0
            rejected = 0
            expired = 0
            expiring_soon = 0

            for r in reviews:
                status = r.get("approval_status")
                valid_until = r.get("valid_until")

                if status == "pending":
                    pending += 1
                elif status == "rejected":
                    rejected += 1
                elif status == "approved":
                    if valid_until:
                        exp_date = date.fromisoformat(valid_until)
                        if exp_date < today:
                            expired += 1
                        elif exp_date <= soon:
                            expiring_soon += 1
                            approved += 1
                        else:
                            approved += 1
                    else:
                        approved += 1

            return {
                "pending": pending,
                "approved": approved,
                "rejected": rejected,
                "expired": expired,
                "expiring_soon": expiring_soon,
            }
        except Exception as e:
            logger.error(f"Failed to get review summary: {e}")
            return {
                "pending": 0,
                "approved": 0,
                "rejected": 0,
                "expired": 0,
                "expiring_soon": 0,
            }
