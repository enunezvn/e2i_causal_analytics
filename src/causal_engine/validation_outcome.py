"""
E2I Causal Engine - Validation Outcome
Version: 4.3
Purpose: Validation outcome schema for Feedback Learner integration

This module defines the ValidationOutcome dataclass that wraps RefutationSuite
results for consumption by the Feedback Learner agent (Tier 5).

Phase 4 of Causal Validation Protocol:
- Connect Feedback Learner to validation outcomes
- Enable learning from validation failures

Reference: docs/E2I_Causal_Validation_Protocol.html
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .refutation_runner import RefutationSuite, GateDecision


class ValidationOutcomeType(str, Enum):
    """Type of validation outcome for pattern classification."""

    PASSED = "passed"                    # All tests passed, proceed
    FAILED_CRITICAL = "failed_critical"  # Critical test failed (placebo/sensitivity)
    FAILED_MULTIPLE = "failed_multiple"  # Multiple tests failed
    NEEDS_REVIEW = "needs_review"        # Borderline results, expert review needed
    BLOCKED = "blocked"                  # Blocked by gate decision


class FailureCategory(str, Enum):
    """Category of validation failure for learning."""

    INSUFFICIENT_SAMPLE = "insufficient_sample"      # Data subset/bootstrap failed
    UNOBSERVED_CONFOUNDING = "unobserved_confounding"  # Sensitivity analysis failed
    SPURIOUS_CORRELATION = "spurious_correlation"    # Placebo treatment detected effect
    MODEL_MISSPECIFICATION = "model_misspecification"  # Random common cause failed
    EFFECT_INSTABILITY = "effect_instability"        # Bootstrap variance too high
    UNKNOWN = "unknown"


@dataclass
class ValidationFailurePattern:
    """Structured failure pattern for learning.

    This captures the essence of why a validation failed,
    enabling the Feedback Learner to identify patterns and
    the Experiment Designer to avoid similar mistakes.
    """

    category: FailureCategory
    test_name: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    original_effect: float
    refuted_effect: float
    delta_percent: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "test_name": self.test_name,
            "description": self.description,
            "severity": self.severity,
            "original_effect": self.original_effect,
            "refuted_effect": self.refuted_effect,
            "delta_percent": self.delta_percent,
            "recommendation": self.recommendation,
        }


@dataclass
class ValidationOutcome:
    """Complete validation outcome for Feedback Learner consumption.

    This wraps the RefutationSuite with additional metadata and
    extracted failure patterns for systematic learning.

    Attributes:
        outcome_id: Unique identifier for this outcome
        outcome_type: Classification of outcome (passed/failed/review/blocked)
        timestamp: When the validation occurred
        estimate_id: ID of the causal estimate being validated
        treatment_variable: Treatment variable name
        outcome_variable: Outcome variable name
        brand: Brand context
        gate_decision: Decision from RefutationSuite (proceed/review/block)
        confidence_score: Confidence score from RefutationSuite (0-1)
        tests_passed: Number of tests that passed
        tests_failed: Number of tests that failed
        tests_total: Total number of tests run
        failure_patterns: Extracted patterns from failed tests
        raw_suite: Original RefutationSuite data (serialized)
        agent_context: Additional context from the calling agent
        dag_hash: DAG version hash for reproducibility
        sample_size: Sample size used in analysis
        effect_size: Estimated effect size
    """

    outcome_id: str
    outcome_type: ValidationOutcomeType
    timestamp: str
    estimate_id: Optional[str] = None
    treatment_variable: Optional[str] = None
    outcome_variable: Optional[str] = None
    brand: Optional[str] = None
    gate_decision: str = "unknown"
    confidence_score: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    failure_patterns: List[ValidationFailurePattern] = field(default_factory=list)
    raw_suite: Dict[str, Any] = field(default_factory=dict)
    agent_context: Dict[str, Any] = field(default_factory=dict)
    dag_hash: Optional[str] = None
    sample_size: Optional[int] = None
    effect_size: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "outcome_id": self.outcome_id,
            "outcome_type": self.outcome_type.value,
            "timestamp": self.timestamp,
            "estimate_id": self.estimate_id,
            "treatment_variable": self.treatment_variable,
            "outcome_variable": self.outcome_variable,
            "brand": self.brand,
            "gate_decision": self.gate_decision,
            "confidence_score": self.confidence_score,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_total": self.tests_total,
            "failure_patterns": [p.to_dict() for p in self.failure_patterns],
            "raw_suite": self.raw_suite,
            "agent_context": self.agent_context,
            "dag_hash": self.dag_hash,
            "sample_size": self.sample_size,
            "effect_size": self.effect_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationOutcome":
        """Create ValidationOutcome from dictionary."""
        patterns = [
            ValidationFailurePattern(
                category=FailureCategory(p["category"]),
                test_name=p["test_name"],
                description=p["description"],
                severity=p["severity"],
                original_effect=p["original_effect"],
                refuted_effect=p["refuted_effect"],
                delta_percent=p["delta_percent"],
                recommendation=p["recommendation"],
            )
            for p in data.get("failure_patterns", [])
        ]

        return cls(
            outcome_id=data["outcome_id"],
            outcome_type=ValidationOutcomeType(data["outcome_type"]),
            timestamp=data["timestamp"],
            estimate_id=data.get("estimate_id"),
            treatment_variable=data.get("treatment_variable"),
            outcome_variable=data.get("outcome_variable"),
            brand=data.get("brand"),
            gate_decision=data.get("gate_decision", "unknown"),
            confidence_score=data.get("confidence_score", 0.0),
            tests_passed=data.get("tests_passed", 0),
            tests_failed=data.get("tests_failed", 0),
            tests_total=data.get("tests_total", 0),
            failure_patterns=patterns,
            raw_suite=data.get("raw_suite", {}),
            agent_context=data.get("agent_context", {}),
            dag_hash=data.get("dag_hash"),
            sample_size=data.get("sample_size"),
            effect_size=data.get("effect_size"),
        )

    def get_learning_summary(self) -> str:
        """Generate a summary for learning purposes."""
        if self.outcome_type == ValidationOutcomeType.PASSED:
            return (
                f"Validation passed for {self.treatment_variable} → {self.outcome_variable} "
                f"with confidence {self.confidence_score:.2f}"
            )

        patterns_summary = ", ".join(
            f"{p.category.value}: {p.description}" for p in self.failure_patterns[:3]
        )

        return (
            f"Validation {self.outcome_type.value} for {self.treatment_variable} → "
            f"{self.outcome_variable}: {patterns_summary}"
        )


def extract_failure_patterns(
    suite: "RefutationSuite",
) -> List[ValidationFailurePattern]:
    """Extract failure patterns from a RefutationSuite.

    This analyzes failed tests and categorizes them into
    learnable patterns for the Feedback Learner.

    Args:
        suite: RefutationSuite with validation results

    Returns:
        List of ValidationFailurePattern objects
    """
    from .refutation_runner import RefutationStatus, RefutationTestType

    patterns = []

    for test in suite.tests:
        if test.status not in (RefutationStatus.FAILED, RefutationStatus.WARNING):
            continue

        # Categorize the failure
        category, severity, recommendation = _categorize_failure(test)

        patterns.append(ValidationFailurePattern(
            category=category,
            test_name=test.test_name.value,
            description=_describe_failure(test),
            severity=severity,
            original_effect=test.original_effect,
            refuted_effect=test.refuted_effect,
            delta_percent=test.delta_percent,
            recommendation=recommendation,
        ))

    return patterns


def _categorize_failure(test) -> tuple:
    """Categorize a test failure into a learning category."""
    from .refutation_runner import RefutationTestType

    test_name = test.test_name
    delta = abs(test.delta_percent)

    if test_name == RefutationTestType.PLACEBO_TREATMENT:
        # Placebo treatment should show no effect
        return (
            FailureCategory.SPURIOUS_CORRELATION,
            "critical" if delta > 50 else "high",
            "Detected effect on placebo treatment. Check for spurious correlations "
            "or confounding variables not included in the model."
        )

    elif test_name == RefutationTestType.SENSITIVITY_E_VALUE:
        return (
            FailureCategory.UNOBSERVED_CONFOUNDING,
            "critical" if test.details.get("e_value", 0) < 1.5 else "high",
            "Effect is sensitive to unobserved confounding. Consider collecting "
            "additional covariates or using instrumental variables."
        )

    elif test_name == RefutationTestType.RANDOM_COMMON_CAUSE:
        return (
            FailureCategory.MODEL_MISSPECIFICATION,
            "high" if delta > 30 else "medium",
            "Effect is sensitive to random common causes. Review the causal DAG "
            "for missing confounders or incorrect causal assumptions."
        )

    elif test_name == RefutationTestType.DATA_SUBSET:
        return (
            FailureCategory.INSUFFICIENT_SAMPLE,
            "medium",
            "Effect is unstable across data subsets. Consider increasing sample size "
            "or stratifying analysis by key segments."
        )

    elif test_name == RefutationTestType.BOOTSTRAP:
        return (
            FailureCategory.EFFECT_INSTABILITY,
            "medium" if delta > 20 else "low",
            "High variance in bootstrap estimates. Increase sample size or "
            "investigate heterogeneous treatment effects."
        )

    return (
        FailureCategory.UNKNOWN,
        "medium",
        "Unknown failure pattern. Review test results manually."
    )


def _describe_failure(test) -> str:
    """Generate a human-readable description of the failure."""
    from .refutation_runner import RefutationTestType

    test_name = test.test_name
    delta = test.delta_percent

    if test_name == RefutationTestType.PLACEBO_TREATMENT:
        return f"Placebo treatment showed {abs(delta):.1f}% of original effect"

    elif test_name == RefutationTestType.SENSITIVITY_E_VALUE:
        e_value = test.details.get("e_value", "N/A")
        return f"E-value of {e_value} indicates sensitivity to unmeasured confounding"

    elif test_name == RefutationTestType.RANDOM_COMMON_CAUSE:
        return f"Random common cause changed effect by {abs(delta):.1f}%"

    elif test_name == RefutationTestType.DATA_SUBSET:
        return f"Effect varied by {abs(delta):.1f}% across data subsets"

    elif test_name == RefutationTestType.BOOTSTRAP:
        return f"Bootstrap variance: effect changed by {abs(delta):.1f}%"

    return f"Test {test_name.value} failed with {abs(delta):.1f}% change"


def create_validation_outcome(
    suite: "RefutationSuite",
    agent_context: Optional[Dict[str, Any]] = None,
    dag_hash: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> ValidationOutcome:
    """Create a ValidationOutcome from a RefutationSuite.

    This is the main entry point for converting validation results
    into a format suitable for the Feedback Learner.

    Args:
        suite: RefutationSuite with validation results
        agent_context: Optional context from the calling agent
        dag_hash: Optional DAG version hash
        sample_size: Optional sample size used in analysis

    Returns:
        ValidationOutcome for Feedback Learner consumption
    """
    from .refutation_runner import GateDecision
    import uuid

    # Determine outcome type based on gate decision and test results
    if suite.gate_decision == GateDecision.PROCEED:
        outcome_type = ValidationOutcomeType.PASSED
    elif suite.gate_decision == GateDecision.BLOCK:
        # Check if critical test failed
        critical_failed = any(
            t.test_name.value in ("placebo_treatment", "sensitivity_e_value")
            and t.status.value == "failed"
            for t in suite.tests
        )
        if critical_failed:
            outcome_type = ValidationOutcomeType.FAILED_CRITICAL
        else:
            outcome_type = ValidationOutcomeType.FAILED_MULTIPLE
    elif suite.gate_decision == GateDecision.REVIEW:
        outcome_type = ValidationOutcomeType.NEEDS_REVIEW
    else:
        outcome_type = ValidationOutcomeType.BLOCKED

    # Extract failure patterns
    failure_patterns = extract_failure_patterns(suite)

    # Count tests
    tests_passed = suite.tests_passed
    tests_failed = suite.tests_failed
    tests_total = len(suite.tests)

    # Serialize the raw suite
    raw_suite = {
        "passed": suite.passed,
        "confidence_score": suite.confidence_score,
        "gate_decision": suite.gate_decision.value,
        "total_execution_time_ms": suite.total_execution_time_ms,
        "tests": [t.to_dict() for t in suite.tests],
    }

    # Get effect size from first test (all should have same original_effect)
    effect_size = suite.tests[0].original_effect if suite.tests else None

    return ValidationOutcome(
        outcome_id=f"vo_{uuid.uuid4().hex[:12]}",
        outcome_type=outcome_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        estimate_id=suite.estimate_id,
        treatment_variable=suite.treatment_variable,
        outcome_variable=suite.outcome_variable,
        brand=suite.brand,
        gate_decision=suite.gate_decision.value,
        confidence_score=suite.confidence_score,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_total=tests_total,
        failure_patterns=failure_patterns,
        raw_suite=raw_suite,
        agent_context=agent_context or {},
        dag_hash=dag_hash,
        sample_size=sample_size,
        effect_size=effect_size,
    )
