"""
Expert Review Gate for Causal DAG Validation.

Provides workflow gating based on domain expert approval status of causal DAGs.
Integrates with ExpertReviewRepository for persistence.

Version: 4.3
"""

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, Optional

from src.repositories.expert_review import ExpertReviewRepository

logger = logging.getLogger(__name__)


class ReviewGateDecision(Enum):
    """Expert review gate decisions."""

    PROCEED = "proceed"  # DAG has active approval
    PENDING_REVIEW = "pending_review"  # Review request created, awaiting approval
    RENEWAL_REQUIRED = "renewal_required"  # Approval expiring soon, needs renewal
    BLOCKED = "blocked"  # No approval and no pending review


@dataclass
class ReviewGateResult:
    """Result of expert review gate check."""

    decision: ReviewGateDecision
    dag_hash: str
    is_approved: bool
    review_id: Optional[str] = None
    approved_at: Optional[str] = None
    valid_until: Optional[str] = None
    days_until_expiry: Optional[int] = None
    reviewer_name: Optional[str] = None
    message: str = ""
    requires_action: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "dag_hash": self.dag_hash,
            "is_approved": self.is_approved,
            "review_id": self.review_id,
            "approved_at": self.approved_at,
            "valid_until": self.valid_until,
            "days_until_expiry": self.days_until_expiry,
            "reviewer_name": self.reviewer_name,
            "message": self.message,
            "requires_action": self.requires_action,
        }


class ExpertReviewGate:
    """
    Workflow gate for expert review of causal DAGs.

    Usage:
        gate = ExpertReviewGate(expert_review_repo)
        result = await gate.check_approval(dag_hash, brand)

        if result.decision == ReviewGateDecision.PROCEED:
            # Continue with causal analysis
            pass
        elif result.decision == ReviewGateDecision.PENDING_REVIEW:
            # Analysis blocked until review complete
            pass
    """

    # Default renewal warning threshold (days before expiry)
    RENEWAL_WARNING_DAYS = 14

    def __init__(
        self,
        repository: Optional[ExpertReviewRepository] = None,
        renewal_warning_days: int = RENEWAL_WARNING_DAYS,
        auto_create_review: bool = True,
    ):
        """
        Initialize expert review gate.

        Args:
            repository: ExpertReviewRepository instance
            renewal_warning_days: Days before expiry to warn about renewal
            auto_create_review: Whether to auto-create review requests for new DAGs
        """
        self.repository = repository
        self.renewal_warning_days = renewal_warning_days
        self.auto_create_review = auto_create_review

    async def check_approval(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        requester_id: Optional[str] = None,
        analysis_context: Optional[str] = None,
    ) -> ReviewGateResult:
        """
        Check if a DAG has expert approval and is valid.

        Args:
            dag_hash: SHA256 hash of the DAG structure
            brand: Brand context for filtering
            treatment: Treatment variable name
            outcome: Outcome variable name
            requester_id: User ID requesting the analysis
            analysis_context: Description of the analysis

        Returns:
            ReviewGateResult with decision and metadata
        """
        if not self.repository:
            # No repository means no gating (development mode)
            logger.warning("No repository configured, bypassing expert review gate")
            return ReviewGateResult(
                decision=ReviewGateDecision.PROCEED,
                dag_hash=dag_hash,
                is_approved=True,
                message="Expert review gate bypassed (no repository)",
            )

        # Check for active approval
        approval = await self.repository.get_dag_approval(dag_hash, brand)

        if approval:
            # DAG has active approval - check expiry
            valid_until = approval.get("valid_until")
            days_until_expiry = None

            if valid_until:
                try:
                    expiry_date = date.fromisoformat(valid_until)
                    days_until_expiry = (expiry_date - date.today()).days
                except (ValueError, TypeError):
                    pass

            # Check if renewal warning needed
            if days_until_expiry is not None and days_until_expiry <= self.renewal_warning_days:
                return ReviewGateResult(
                    decision=ReviewGateDecision.RENEWAL_REQUIRED,
                    dag_hash=dag_hash,
                    is_approved=True,
                    review_id=approval.get("review_id"),
                    approved_at=approval.get("approved_at"),
                    valid_until=valid_until,
                    days_until_expiry=days_until_expiry,
                    reviewer_name=approval.get("reviewer_name"),
                    message=f"DAG approval expiring in {days_until_expiry} days. Renewal required.",
                    requires_action=True,
                )

            # Approval is valid and not expiring soon
            return ReviewGateResult(
                decision=ReviewGateDecision.PROCEED,
                dag_hash=dag_hash,
                is_approved=True,
                review_id=approval.get("review_id"),
                approved_at=approval.get("approved_at"),
                valid_until=valid_until,
                days_until_expiry=days_until_expiry,
                reviewer_name=approval.get("reviewer_name"),
                message="DAG has active expert approval",
            )

        # No active approval - check for pending review
        pending_reviews = await self.repository.get_reviews_for_dag(dag_hash, include_expired=False)
        pending = [r for r in pending_reviews if r.get("approval_status") == "pending"]

        if pending:
            # Review already pending
            return ReviewGateResult(
                decision=ReviewGateDecision.PENDING_REVIEW,
                dag_hash=dag_hash,
                is_approved=False,
                review_id=pending[0].get("review_id"),
                message="DAG review pending expert approval",
                requires_action=True,
            )

        # No approval and no pending review
        if self.auto_create_review and requester_id:
            # Auto-create review request
            review_id = await self.repository.create_review(
                reviewer_id=requester_id,
                review_type="initial_dag",
                dag_version_hash=dag_hash,
                brand=brand,
                treatment_variable=treatment,
                outcome_variable=outcome,
                analysis_context=analysis_context,
            )

            if review_id:
                return ReviewGateResult(
                    decision=ReviewGateDecision.PENDING_REVIEW,
                    dag_hash=dag_hash,
                    is_approved=False,
                    review_id=review_id,
                    message="New DAG detected. Expert review request created.",
                    requires_action=True,
                )

        # Blocked - no approval and couldn't create review
        return ReviewGateResult(
            decision=ReviewGateDecision.BLOCKED,
            dag_hash=dag_hash,
            is_approved=False,
            message="DAG requires expert approval before analysis can proceed",
            requires_action=True,
        )

    async def can_proceed(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
        allow_pending: bool = False,
        allow_expiring: bool = True,
    ) -> bool:
        """
        Simple check if analysis can proceed.

        Args:
            dag_hash: SHA256 hash of the DAG
            brand: Brand context
            allow_pending: If True, allow proceeding with pending reviews
            allow_expiring: If True, allow proceeding with expiring approvals

        Returns:
            True if analysis can proceed
        """
        result = await self.check_approval(dag_hash, brand)

        if result.decision == ReviewGateDecision.PROCEED:
            return True
        elif result.decision == ReviewGateDecision.RENEWAL_REQUIRED and allow_expiring:
            return True
        elif result.decision == ReviewGateDecision.PENDING_REVIEW and allow_pending:
            return True

        return False

    async def request_renewal(
        self,
        dag_hash: str,
        requester_id: str,
        brand: Optional[str] = None,
        requester_name: Optional[str] = None,
        requester_email: Optional[str] = None,
    ) -> Optional[str]:
        """
        Request renewal of an expiring DAG approval.

        Args:
            dag_hash: SHA256 hash of the DAG
            requester_id: User ID requesting renewal
            brand: Brand context
            requester_name: Display name
            requester_email: Contact email

        Returns:
            New review_id or None on failure
        """
        if not self.repository:
            return None

        # Find the existing approval to renew
        approval = await self.repository.get_dag_approval(dag_hash, brand)

        if not approval:
            logger.warning(f"No existing approval found for DAG {dag_hash}")
            return None

        # Create renewal review
        review_id = approval.get("review_id")
        if review_id is None:
            logger.warning(f"No review_id found in approval for DAG {dag_hash}")
            return None
        return await self.repository.renew_review(
            original_review_id=review_id,
            reviewer_id=requester_id,
            reviewer_name=requester_name,
            reviewer_email=requester_email,
        )

    async def get_pending_review_count(
        self,
        brand: Optional[str] = None,
    ) -> int:
        """
        Get count of pending reviews for monitoring.

        Args:
            brand: Optional brand filter

        Returns:
            Count of pending reviews
        """
        if not self.repository:
            return 0

        pending = await self.repository.get_pending_reviews(brand=brand)
        return len(pending)

    async def get_expiring_dag_count(
        self,
        days: int = 14,
        brand: Optional[str] = None,
    ) -> int:
        """
        Get count of DAG approvals expiring soon.

        Args:
            days: Days threshold for expiry warning
            brand: Optional brand filter

        Returns:
            Count of expiring approvals
        """
        if not self.repository:
            return 0

        expiring = await self.repository.get_expiring_reviews(days, brand)
        return len(expiring)

    async def get_gate_status(
        self,
        brand: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get overall gate status for monitoring dashboard.

        Args:
            brand: Optional brand filter

        Returns:
            Dict with gate health metrics
        """
        if not self.repository:
            return {
                "healthy": True,
                "pending_reviews": 0,
                "expiring_soon": 0,
                "total_approved": 0,
                "message": "Expert review gate not configured",
            }

        summary = await self.repository.get_review_summary(brand)

        pending = summary.get("pending", 0)
        expiring = summary.get("expiring_soon", 0)
        approved = summary.get("approved", 0)

        # Gate is unhealthy if many reviews pending or expiring
        healthy = pending < 5 and expiring < 3

        return {
            "healthy": healthy,
            "pending_reviews": pending,
            "expiring_soon": expiring,
            "total_approved": approved,
            "total_rejected": summary.get("rejected", 0),
            "total_expired": summary.get("expired", 0),
            "message": (
                "Gate healthy"
                if healthy
                else "Attention needed: pending reviews or expiring approvals"
            ),
        }


# Convenience function for integration with causal workflow
async def check_dag_approval(
    dag_hash: str,
    brand: Optional[str] = None,
    repository: Optional[ExpertReviewRepository] = None,
) -> ReviewGateResult:
    """
    Check DAG approval status (standalone function).

    Args:
        dag_hash: SHA256 hash of the DAG
        brand: Brand context
        repository: ExpertReviewRepository instance

    Returns:
        ReviewGateResult with decision
    """
    gate = ExpertReviewGate(repository=repository)
    return await gate.check_approval(dag_hash, brand)
