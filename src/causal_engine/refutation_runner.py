"""
E2I Causal Engine - RefutationRunner
Version: 4.3
Purpose: DoWhy-based refutation testing for causal estimate validation

This module implements the Causal Validation Protocol's primary validation tier:
- 5 refutation tests (placebo, random_common_cause, data_subset, bootstrap, sensitivity)
- Configurable thresholds for pass/fail criteria
- Gate decision logic (proceed, review, block)
- Database persistence integration
- Opik tracing for per-test observability

Reference: docs/E2I_Causal_Validation_Protocol.html
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np

# Structured fail-closed error for refutation failures (F-014, #416)
from src.causal_engine.errors import RefutationError

# Opik tracing for causal validation observability
from src.mlops.opik_connector import get_opik_connector

# Conditional DoWhy import for graceful degradation
try:
    from dowhy import CausalModel

    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    CausalModel = None

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _require_p_value(refutation: Any, test_name: str, original_effect: float) -> float:
    """Extract ``p_value`` from a DoWhy refutation result without silent defaulting.

    Codex iter-2 H4 (#416): the previous ``refutation.refutation_result.get(
    "p_value", 0.5)`` pattern silently inserted ``0.5`` (which passes the
    placebo threshold) when the refuter did not expose a p-value. This is a
    placeholder evidence value, exactly the kind of silent-wrong this PR is
    closing. Fail-closed instead.

    Raises:
        RefutationError: if ``p_value`` is missing or non-finite.
    """
    pv = refutation.refutation_result.get("p_value") if refutation.refutation_result else None
    if pv is None:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"DoWhy {test_name} refuter did not return a p_value; refusing to "
            "substitute a placeholder (e.g., 0.5).",
            details={
                "test_name": test_name,
                "original_effect": original_effect,
                "reason": "missing_p_value",
            },
        )
    try:
        pv_float = float(pv)
    except (TypeError, ValueError) as exc:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"DoWhy {test_name} refuter returned non-numeric p_value: {pv!r}.",
            details={
                "test_name": test_name,
                "original_effect": original_effect,
                "reason": "non_numeric_p_value",
                "p_value_raw": repr(pv),
            },
            original_error=exc,
        ) from exc
    if not np.isfinite(pv_float):
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"DoWhy {test_name} refuter returned non-finite p_value: {pv_float}.",
            details={
                "test_name": test_name,
                "original_effect": original_effect,
                "reason": "non_finite_p_value",
            },
        )
    return pv_float


# ============================================================================
# ENUMS (aligned with database/ml/010_causal_validation_tables.sql)
# ============================================================================


class RefutationStatus(str, Enum):
    """Status of individual refutation test.

    Aligned with database ENUM: validation_status
    """

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class GateDecision(str, Enum):
    """Aggregate decision from RefutationSuite.

    Aligned with database ENUM: gate_decision
    """

    PROCEED = "proceed"  # Confidence >= 0.7, all critical tests passed
    REVIEW = "review"  # Confidence 0.5-0.7, requires expert review
    BLOCK = "block"  # Confidence < 0.5 or critical test failed


class RefutationTestType(str, Enum):
    """Types of refutation tests.

    Aligned with database ENUM: refutation_test_type
    """

    PLACEBO_TREATMENT = "placebo_treatment"
    RANDOM_COMMON_CAUSE = "random_common_cause"
    DATA_SUBSET = "data_subset"
    BOOTSTRAP = "bootstrap"
    SENSITIVITY_E_VALUE = "sensitivity_e_value"


# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass
class RefutationResult:
    """Result of a single refutation test.

    Attributes:
        test_name: Type of refutation test run
        status: Pass/fail/warning/skipped status
        original_effect: Original causal effect estimate (ATE)
        refuted_effect: Effect after refutation manipulation
        p_value: Statistical significance (if applicable)
        delta_percent: Percentage change from original effect
        details: Additional test-specific information
        execution_time_ms: Time taken to run this test
    """

    test_name: RefutationTestType
    status: RefutationStatus
    original_effect: float
    refuted_effect: float
    p_value: Optional[float] = None
    delta_percent: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name.value,
            "status": self.status.value,
            "original_effect": self.original_effect,
            "refuted_effect": self.refuted_effect,
            "p_value": self.p_value,
            "delta_percent": self.delta_percent,
            "details": self.details,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class RefutationSuite:
    """Complete refutation analysis results.

    Attributes:
        passed: Whether the estimate is NOT blocked (gate is PROCEED or REVIEW).
            NOTE: this is "not blocked", NOT "majority of tests passed" — a
            REVIEW-band result has passed=True but is only borderline-robust
            (see ``needs_review``). Consumers that need true robustness must
            check ``gate_decision == PROCEED`` / ``needs_review``, not ``passed``.
        confidence_score: Weighted confidence score (0-1)
        tests: List of individual test results
        gate_decision: Aggregate decision (proceed/review/block)
        total_execution_time_ms: Total time for all tests
        estimate_id: UUID of the causal estimate being validated
        treatment_variable: Treatment variable name
        outcome_variable: Outcome variable name
        brand: Brand context (optional)
    """

    passed: bool
    confidence_score: float
    tests: List[RefutationResult]
    gate_decision: GateDecision
    total_execution_time_ms: float = 0.0
    estimate_id: Optional[str] = None
    treatment_variable: Optional[str] = None
    outcome_variable: Optional[str] = None
    brand: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def tests_passed(self) -> int:
        """Count of passed tests."""
        return sum(1 for t in self.tests if t.status == RefutationStatus.PASSED)

    @property
    def tests_failed(self) -> int:
        """Count of failed tests."""
        return sum(1 for t in self.tests if t.status == RefutationStatus.FAILED)

    @property
    def tests_warning(self) -> int:
        """Count of warning tests."""
        return sum(1 for t in self.tests if t.status == RefutationStatus.WARNING)

    @property
    def total_tests(self) -> int:
        """Total number of tests run (excluding skipped)."""
        return sum(1 for t in self.tests if t.status != RefutationStatus.SKIPPED)

    @property
    def needs_review(self) -> bool:
        """True when the gate decision is REVIEW (borderline-robust, NOT 'passed').

        Distinct from ``passed`` (= not blocked): a REVIEW result is "valid to
        use with caution" but must NOT be surfaced as robust/validated without
        an expert-review caveat (H2).
        """
        return self.gate_decision == GateDecision.REVIEW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "needs_review": self.needs_review,
            "confidence_score": self.confidence_score,
            "tests": [t.to_dict() for t in self.tests],
            "gate_decision": self.gate_decision.value,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_warning": self.tests_warning,
            "total_tests": self.total_tests,
            "total_execution_time_ms": self.total_execution_time_ms,
            "estimate_id": self.estimate_id,
            "treatment_variable": self.treatment_variable,
            "outcome_variable": self.outcome_variable,
            "brand": self.brand,
            "created_at": self.created_at,
        }

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy RefutationResults format for backward compatibility.

        Maps to the existing state.RefutationResults TypedDict.

        Contract: individual_tests MUST be Dict with test names as keys:
        - placebo_treatment
        - random_common_cause
        - data_subset
        - unobserved_common_cause (maps from sensitivity_e_value)
        """
        # Build Dict with test names as keys (contract requirement)
        individual_tests: Dict[str, Dict[str, Any]] = {}
        for t in self.tests:
            # Map test name to contract key
            # Note: sensitivity_e_value maps to unobserved_common_cause per contract
            key = t.test_name.value
            if key == "sensitivity_e_value":
                key = "unobserved_common_cause"
            elif key == "bootstrap":
                # Bootstrap is additional, not in original contract
                # Keep as-is for backward compatibility
                key = "bootstrap"

            individual_tests[key] = {
                "test_name": t.test_name.value,
                "passed": t.status == RefutationStatus.PASSED,
                "new_effect": t.refuted_effect,
                "original_effect": t.original_effect,
                "p_value": t.p_value or 0.0,
                "details": t.details.get("message", ""),
            }

        return {
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "total_tests": self.total_tests,
            "overall_robust": self.passed,
            "individual_tests": individual_tests,
            "confidence_adjustment": self.confidence_score,
            "gate_decision": self.gate_decision.value,
            # H2: distinct signal so a REVIEW-band result is not consumed as robust.
            "needs_review": self.needs_review,
        }


# ============================================================================
# REFUTATION RUNNER
# ============================================================================


class RefutationRunner:
    """Orchestrates DoWhy refutation tests for causal estimate validation.

    This class implements the Causal Validation Protocol's primary tier:
    automated refutation testing for every causal estimate.

    Example:
        ```python
        runner = RefutationRunner()
        suite = runner.run_all_tests(
            causal_model=model,
            identified_estimand=estimand,
            estimate=estimate,
            data=df,
            treatment="hcp_engagement",
            outcome="conversion_rate"
        )

        if suite.gate_decision == GateDecision.BLOCK:
            raise ValidationError("Causal estimate failed refutation")
        ```

    Attributes:
        config: Test configuration (num_simulations, thresholds, etc.)
        thresholds: Pass/fail thresholds for each test type
    """

    # Default configuration for each test type.
    #
    # #622 prod-latency tuning. The previous defaults (placebo 100, bootstrap
    # 500, and random_common_cause at DoWhy's own internal default of 100 — it
    # had NO ``num_simulations`` key, so the runner never passed one) made the
    # full suite ~610 DoWhy re-estimations. MEASURED on the synthetic fixture
    # (DoWhy 0.14 / EconML 0.16, see #622): each re-estimation is ~0.05s for the
    # linear refit but ~3.1s when the energy-score selector picks
    # CausalForestDML, so the suite ran ~33s (OLS) to ~35-60 min (causal_forest)
    # per query, far over the node's documented SLA.
    #
    # We lower the per-refuter simulation counts to the smallest values that
    # still answer each refuter's question meaningfully. The cuts trade a small
    # amount of statistical precision for a large latency win, and each refuter
    # only renders a coarse PASS/WARNING/FAIL decision (e.g. "did the effect
    # move >20%?", "is the placebo p-value >0.05?"), which does not need the
    # high-resolution null distributions the old defaults produced.
    #
    #   * placebo_treatment 100 -> 30: the placebo p-value is a permutation
    #     p-value with resolution ~1/(n+1); 30 perms gives ~0.032 resolution,
    #     comfortably finer than the 0.05 pass threshold while ~3x cheaper.
    #   * random_common_cause add num_simulations=20: previously unbounded at
    #     DoWhy's internal default 100. Each sim adds an independent random
    #     confounder and re-estimates; the decision is whether the mean effect
    #     shifts >20%. By the CLT the SE of that mean shrinks as 1/sqrt(n);
    #     20 sims keeps the SE ~0.22x of a single draw — ample for a coarse
    #     stability gate — at ~5x lower cost than 100.
    #   * bootstrap 500 -> 50: a 50-resample percentile CI has acceptable
    #     coverage error for the CI-width-ratio stability check (the refuter
    #     compares bootstrap CI width vs original CI width, a ratio, not a
    #     high-precision interval). This is the single biggest latency item
    #     (MEASURED ~22s at 500 on OLS), cut ~10x.
    #   * data_subset 10 -> 5: each subset is an independent fit on 80% of rows;
    #     5 subsets is enough to gauge cross-subset consistency for a coarse
    #     coverage gate, halving the cost.
    #
    # Callers needing full statistical rigor (e.g. an offline / slow-tests run)
    # can still pass a richer ``config`` to ``RefutationRunner`` /
    # ``RefutationNode`` (merged per-key onto these defaults). The Tier 1-5
    # smoke harness already passes an even-smaller bounded config via
    # ``parameters.refutation_config`` (#606), which still wins because it is
    # merged on top of these defaults.
    DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
        "placebo_treatment": {
            "enabled": True,
            "num_simulations": 30,
            "critical": True,  # Failure blocks estimate
        },
        "random_common_cause": {
            "enabled": True,
            "effect_strength": 0.1,
            "num_simulations": 20,
            "critical": True,
        },
        "data_subset": {
            "enabled": True,
            "subset_fraction": 0.8,
            "num_subsets": 5,
            "critical": False,
        },
        "bootstrap": {
            "enabled": True,
            "num_bootstraps": 50,
            "critical": False,
        },
        "sensitivity_e_value": {
            "enabled": True,
            "e_value_threshold": 2.0,
            "critical": True,
        },
    }

    # Thresholds for determining pass/fail/warning
    PASS_THRESHOLDS: Dict[str, Dict[str, float]] = {
        "placebo_p_value": {
            "pass": 0.05,  # Placebo effect p-value must be > 0.05
            "warning": 0.10,  # Warning if 0.05 < p < 0.10
        },
        "common_cause_delta": {
            "pass": 0.20,  # Effect change must be < 20%
            "warning": 0.30,  # Warning if 20% < delta < 30%
        },
        "subset_ci_coverage": {
            "pass": 0.80,  # 80% of subsets must contain original effect
            "warning": 0.70,
        },
        "bootstrap_ci_ratio": {
            "pass": 0.50,  # Bootstrap CI must not be > 50% wider than original
            "warning": 0.75,
        },
        "e_value_min": {
            "pass": 2.0,  # E-value must be >= 2.0
            "warning": 1.5,
        },
    }

    # Gate decision thresholds
    GATE_THRESHOLDS = {
        "proceed": 0.70,  # Confidence >= 0.70 → proceed
        "review": 0.50,  # Confidence 0.50-0.70 → review
        # Below 0.50 → block
    }

    def __init__(
        self,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize RefutationRunner.

        Args:
            config: Custom test configuration (merged with DEFAULT_CONFIG)
            thresholds: Custom pass thresholds (merged with PASS_THRESHOLDS)
        """
        # Use deep copy to prevent mutation of class-level defaults
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = copy.deepcopy(value)

        self.thresholds = copy.deepcopy(self.PASS_THRESHOLDS)
        if thresholds:
            for key, value in thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key].update(value)
                else:
                    self.thresholds[key] = copy.deepcopy(value)

    def run_all_tests(
        self,
        original_effect: float,
        original_ci: Tuple[float, float],
        data: Optional["pd.DataFrame"] = None,
        causal_model: Optional[Any] = None,
        identified_estimand: Optional[Any] = None,
        estimate: Optional[Any] = None,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        brand: Optional[str] = None,
        estimate_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> RefutationSuite:
        """Run all enabled refutation tests with Opik tracing.

        Args:
            original_effect: The ATE to validate
            original_ci: Confidence interval (lower, upper)
            data: DataFrame with treatment/outcome data (for DoWhy)
            causal_model: DoWhy CausalModel instance (optional)
            identified_estimand: DoWhy estimand (optional)
            estimate: DoWhy estimate object (optional)
            treatment: Treatment variable name
            outcome: Outcome variable name
            brand: Brand context for logging
            estimate_id: UUID for database linking
            trace_id: Opik trace ID for correlation (optional)

        Returns:
            RefutationSuite with all test results and gate decision
        """
        import time

        start_time = time.time()

        tests: List[RefutationResult] = []

        # Determine if we can use DoWhy or need mock mode
        use_dowhy = (
            DOWHY_AVAILABLE
            and causal_model is not None
            and identified_estimand is not None
            and estimate is not None
        )

        if not use_dowhy:
            # F-014 fail-closed: no silent mock fallback. Tests below raise
            # ``RefutationError`` when ``causal_model is None``. This log line
            # remains as a diagnostic only — execution will not proceed past
            # the first per-test mock fallback because those have been deleted.
            logger.warning(
                "Refutation invoked without a real CausalModel "
                "(DOWHY_AVAILABLE=%s, causal_model=%s, identified_estimand=%s, "
                "estimate=%s) — each test will raise RefutationError; the agent "
                "refutation node should reconstruct CausalModel before calling this.",
                DOWHY_AVAILABLE,
                causal_model is not None,
                identified_estimand is not None,
                estimate is not None,
            )

        # Get Opik connector for tracing
        opik = get_opik_connector()

        # Run each enabled test with Opik tracing
        if self.config["placebo_treatment"]["enabled"]:
            test_result = self._run_test_with_tracing(
                test_name="placebo_treatment",
                test_func=self._run_placebo_test,
                opik=opik,
                trace_id=trace_id,
                estimate_id=estimate_id,
                original_effect=original_effect,
                causal_model=causal_model,
                identified_estimand=identified_estimand,
                estimate=estimate,
                use_dowhy=use_dowhy,
            )
            tests.append(test_result)

        if self.config["random_common_cause"]["enabled"]:
            test_result = self._run_test_with_tracing(
                test_name="random_common_cause",
                test_func=self._run_random_common_cause_test,
                opik=opik,
                trace_id=trace_id,
                estimate_id=estimate_id,
                original_effect=original_effect,
                causal_model=causal_model,
                identified_estimand=identified_estimand,
                estimate=estimate,
                use_dowhy=use_dowhy,
            )
            tests.append(test_result)

        if self.config["data_subset"]["enabled"]:
            test_result = self._run_test_with_tracing(
                test_name="data_subset",
                test_func=self._run_data_subset_test,
                opik=opik,
                trace_id=trace_id,
                estimate_id=estimate_id,
                original_effect=original_effect,
                original_ci=original_ci,
                causal_model=causal_model,
                identified_estimand=identified_estimand,
                estimate=estimate,
                use_dowhy=use_dowhy,
            )
            tests.append(test_result)

        if self.config["bootstrap"]["enabled"]:
            test_result = self._run_test_with_tracing(
                test_name="bootstrap",
                test_func=self._run_bootstrap_test,
                opik=opik,
                trace_id=trace_id,
                estimate_id=estimate_id,
                original_effect=original_effect,
                original_ci=original_ci,
                causal_model=causal_model,
                identified_estimand=identified_estimand,
                estimate=estimate,
                use_dowhy=use_dowhy,
            )
            tests.append(test_result)

        if self.config["sensitivity_e_value"]["enabled"]:
            # H3: the E-value needs a STANDARDIZED effect, so compute the outcome
            # SD from the passthrough data and hand it to the sensitivity test.
            outcome_std: Optional[float] = None
            if data is not None and outcome is not None:
                try:
                    if outcome in getattr(data, "columns", []):
                        outcome_std = float(np.std(data[outcome].to_numpy(dtype=float)))
                except Exception:  # noqa: BLE001 - missing/non-numeric outcome → no standardization
                    outcome_std = None
            test_result = self._run_test_with_tracing(
                test_name="sensitivity_e_value",
                test_func=self._run_sensitivity_test,
                opik=opik,
                trace_id=trace_id,
                estimate_id=estimate_id,
                original_effect=original_effect,
                original_ci=original_ci,
                outcome_std=outcome_std,
            )
            tests.append(test_result)

        total_time = (time.time() - start_time) * 1000

        # Calculate confidence score and gate decision
        confidence_score = self._calculate_confidence_score(tests)
        gate_decision = self._determine_gate_decision(tests, confidence_score)
        overall_passed = gate_decision != GateDecision.BLOCK

        suite = RefutationSuite(
            passed=overall_passed,
            confidence_score=confidence_score,
            tests=tests,
            gate_decision=gate_decision,
            total_execution_time_ms=total_time,
            estimate_id=estimate_id,
            treatment_variable=treatment,
            outcome_variable=outcome,
            brand=brand,
        )

        # Log suite-level metrics to Opik
        try:
            opik.log_metric(
                name="refutation_confidence_score",
                value=confidence_score,
                metadata={
                    "gate_decision": gate_decision.value,
                    "tests_passed": suite.tests_passed,
                    "tests_total": suite.total_tests,
                    "treatment": treatment,
                    "outcome": outcome,
                    "brand": brand,
                    "estimate_id": estimate_id,
                },
            )
        except Exception as metric_error:
            logger.debug(f"Failed to log Opik metric: {metric_error}")

        logger.info(
            f"Refutation suite completed: {suite.tests_passed}/{suite.total_tests} passed, "
            f"confidence={confidence_score:.2f}, gate={gate_decision.value}"
        )

        return suite

    def _run_test_with_tracing(
        self,
        test_name: str,
        test_func,
        opik,
        trace_id: Optional[str] = None,
        estimate_id: Optional[str] = None,
        **kwargs,
    ) -> RefutationResult:
        """Run a single refutation test with Opik span tracing.

        Args:
            test_name: Name of the test (e.g., "placebo_treatment")
            test_func: The test function to execute
            opik: OpikConnector instance
            trace_id: Parent trace ID for correlation
            estimate_id: Estimate ID for logging
            **kwargs: Arguments to pass to the test function

        Returns:
            RefutationResult from the test
        """
        import time

        span_start = time.time()

        try:
            # Execute the test
            result: RefutationResult = cast(RefutationResult, test_func(**kwargs))

            # Log span to Opik
            span_duration_ms = (time.time() - span_start) * 1000
            try:
                opik.log_span(
                    name=f"refutation_{test_name}",
                    span_type="tool",
                    input_data={
                        "test_name": test_name,
                        "original_effect": kwargs.get("original_effect"),
                        "use_dowhy": kwargs.get("use_dowhy", False),
                    },
                    output_data={
                        "status": result.status.value,
                        "refuted_effect": result.refuted_effect,
                        "p_value": result.p_value,
                        "delta_percent": result.delta_percent,
                    },
                    metadata={
                        "test_name": test_name,
                        "estimate_id": estimate_id,
                        "trace_id": trace_id,
                        "critical": self.config.get(test_name, {}).get("critical", False),
                    },
                    duration_ms=span_duration_ms,
                    tags=["causal_validation", "refutation", test_name],
                )
            except Exception as span_error:
                logger.debug(f"Failed to log Opik span for {test_name}: {span_error}")

            return result

        except Exception as e:
            # Log error span
            span_duration_ms = (time.time() - span_start) * 1000
            try:
                opik.log_span(
                    name=f"refutation_{test_name}",
                    span_type="tool",
                    input_data={"test_name": test_name},
                    output_data={"error": str(e)},
                    metadata={
                        "test_name": test_name,
                        "estimate_id": estimate_id,
                        "trace_id": trace_id,
                        "error_type": type(e).__name__,
                    },
                    duration_ms=span_duration_ms,
                    status="error",
                    tags=["causal_validation", "refutation", test_name, "error"],
                )
            except Exception as span_error:
                logger.debug(f"Failed to log error span for {test_name}: {span_error}")
            raise

    def _run_placebo_test(
        self,
        original_effect: float,
        causal_model: Optional[Any],
        identified_estimand: Optional[Any],
        estimate: Optional[Any],
        use_dowhy: bool,
    ) -> RefutationResult:
        """Run placebo treatment refutation test.

        Replaces the treatment with random noise. If the effect disappears
        (p-value > 0.05), the original effect is likely causal.
        """
        import time

        start_time = time.time()

        test_name = RefutationTestType.PLACEBO_TREATMENT

        if use_dowhy and causal_model is not None:
            try:
                refutation = causal_model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="placebo_treatment_refuter",
                    placebo_type="permute",
                    num_simulations=self.config["placebo_treatment"]["num_simulations"],
                )
                refuted_effect = float(refutation.new_effect)
                # Iter-2 codex H4: p_value must come from real refuter output;
                # no silent default that would auto-pass the placebo threshold.
                p_value = _require_p_value(refutation, "placebo_treatment", original_effect)
            except RefutationError:
                raise  # re-raise structured errors as-is
            except Exception as e:
                # F-014 fail-closed: no silent mock fallback. Caller (agent
                # refutation node) catches RefutationError and surfaces to chat.
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    f"DoWhy placebo_treatment refuter failed: {e}",
                    details={
                        "test_name": "placebo_treatment",
                        "original_effect": original_effect,
                    },
                    original_error=e,
                ) from e
        else:
            # F-014 fail-closed: ``use_dowhy=False`` reaches here only when
            # the agent caller did NOT reconstruct CausalModel. The new agent
            # path (``refutation.py``) raises ``RefutationError`` BEFORE
            # invoking run_all_tests in that scenario. This branch remains as
            # a defense-in-depth for any non-agent caller (e.g.,
            # ``run_refutation_suite`` convenience function) that still
            # invokes with ``causal_model=None``.
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                "Placebo test requires a real DoWhy CausalModel; caller passed causal_model=None.",
                details={
                    "test_name": "placebo_treatment",
                    "dowhy_available": DOWHY_AVAILABLE,
                    "original_effect": original_effect,
                },
            )

        # Determine status based on thresholds
        # For placebo: we want p-value > threshold (placebo effect not significant)
        if p_value >= self.thresholds["placebo_p_value"]["pass"]:
            status = RefutationStatus.PASSED
            message = "Placebo treatment shows no significant effect (as expected)"
        elif p_value >= self.thresholds["placebo_p_value"]["warning"]:
            status = RefutationStatus.WARNING
            message = "Borderline placebo effect detected, recommend review"
        else:
            status = RefutationStatus.FAILED
            message = "WARNING: Placebo treatment shows significant effect"

        delta_percent = (
            abs(refuted_effect - original_effect) / max(abs(original_effect), 1e-10) * 100
        )
        execution_time = (time.time() - start_time) * 1000

        details: Dict[str, Any] = {
            "message": message,
            "num_simulations": self.config["placebo_treatment"]["num_simulations"],
        }

        return RefutationResult(
            test_name=test_name,
            status=status,
            original_effect=original_effect,
            refuted_effect=refuted_effect,
            p_value=p_value,
            delta_percent=delta_percent,
            details=details,
            execution_time_ms=execution_time,
        )

    def _run_random_common_cause_test(
        self,
        original_effect: float,
        causal_model: Optional[Any],
        identified_estimand: Optional[Any],
        estimate: Optional[Any],
        use_dowhy: bool,
    ) -> RefutationResult:
        """Run random common cause refutation test.

        Adds a random variable as a common cause. If the effect changes
        significantly, unmeasured confounding may be present.
        """
        import time

        start_time = time.time()

        test_name = RefutationTestType.RANDOM_COMMON_CAUSE

        if use_dowhy and causal_model is not None:
            try:
                # Pass num_simulations ONLY when configured, so prod (no key set)
                # keeps DoWhy's own default exactly. DoWhy defaults to 100
                # simulations here; each re-adds a random common cause and
                # re-estimates (~1.4s on this fixture -> ~140s), which dominates
                # the whole pipeline. Callers (e.g. the Tier 1-5 smoke harness)
                # can bound it via ``refutation_config`` like the other tests. (#606)
                _rcc_cfg = self.config["random_common_cause"]
                _rcc_kwargs: Dict[str, Any] = {
                    "method_name": "random_common_cause",
                    "effect_strength_on_treatment": _rcc_cfg["effect_strength"],
                    "effect_strength_on_outcome": _rcc_cfg["effect_strength"],
                }
                if "num_simulations" in _rcc_cfg:
                    _rcc_kwargs["num_simulations"] = _rcc_cfg["num_simulations"]
                refutation = causal_model.refute_estimate(
                    identified_estimand,
                    estimate,
                    **_rcc_kwargs,
                )
                refuted_effect = float(refutation.new_effect)
                # Iter-2 codex H4: p_value must come from real refuter output.
                p_value = _require_p_value(refutation, "random_common_cause", original_effect)
            except RefutationError:
                raise
            except Exception as e:
                # F-014 fail-closed: no silent mock fallback.
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    f"DoWhy random_common_cause refuter failed: {e}",
                    details={
                        "test_name": "random_common_cause",
                        "original_effect": original_effect,
                    },
                    original_error=e,
                ) from e
        else:
            # F-014 fail-closed: defense-in-depth for legacy non-agent callers.
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                "random_common_cause test requires a real DoWhy CausalModel; "
                "caller passed causal_model=None.",
                details={
                    "test_name": "random_common_cause",
                    "dowhy_available": DOWHY_AVAILABLE,
                    "original_effect": original_effect,
                },
            )

        # Calculate delta percentage
        delta_percent = (
            abs(refuted_effect - original_effect) / max(abs(original_effect), 1e-10) * 100
        )

        # Determine status: effect should remain stable
        if delta_percent <= self.thresholds["common_cause_delta"]["pass"] * 100:
            status = RefutationStatus.PASSED
            message = "Effect remains stable when adding random common cause"
        elif delta_percent <= self.thresholds["common_cause_delta"]["warning"] * 100:
            status = RefutationStatus.WARNING
            message = "Effect somewhat sensitive to random confounders"
        else:
            status = RefutationStatus.FAILED
            message = "WARNING: Effect highly sensitive to random confounders"

        execution_time = (time.time() - start_time) * 1000

        return RefutationResult(
            test_name=test_name,
            status=status,
            original_effect=original_effect,
            refuted_effect=refuted_effect,
            p_value=p_value,
            delta_percent=delta_percent,
            details={
                "message": message,
                "effect_strength": self.config["random_common_cause"]["effect_strength"],
            },
            execution_time_ms=execution_time,
        )

    def _run_data_subset_test(
        self,
        original_effect: float,
        original_ci: Tuple[float, float],
        causal_model: Optional[Any],
        identified_estimand: Optional[Any],
        estimate: Optional[Any],
        use_dowhy: bool,
    ) -> RefutationResult:
        """Run data subset validation test.

        Tests effect on random subsets. If effect varies significantly
        across subsets, it may not be robust.
        """
        import time

        start_time = time.time()

        test_name = RefutationTestType.DATA_SUBSET

        if use_dowhy and causal_model is not None:
            try:
                refutation = causal_model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="data_subset_refuter",
                    subset_fraction=self.config["data_subset"]["subset_fraction"],
                    num_simulations=self.config["data_subset"]["num_subsets"],
                )
                refuted_effect = float(refutation.new_effect)
                # Iter-2 codex H4: p_value must come from real refuter output.
                p_value = _require_p_value(refutation, "data_subset", original_effect)
                # Iter-6 codex H-iter5-2: the data_subset test answers
                # "is the effect consistent across data subsets?". That is
                # a DISTRIBUTIONAL question requiring per-subset effects.
                # DoWhy >= 0.10 does not expose ``subset_effects`` in
                # refutation_result by default; collapsing the question to
                # "is the single aggregated mean within original CI?" is a
                # silent substitution, not the same answer. Mark the test
                # SKIPPED when raw subset effects are unavailable instead
                # of fabricating a coverage signal.
                subset_effects = refutation.refutation_result.get("subset_effects", [])
                if not subset_effects:
                    execution_time = (time.time() - start_time) * 1000
                    return RefutationResult(
                        test_name=test_name,
                        status=RefutationStatus.SKIPPED,
                        original_effect=original_effect,
                        refuted_effect=refuted_effect,
                        p_value=p_value,
                        delta_percent=0.0,
                        details={
                            "message": (
                                "Data-subset distributional check skipped: "
                                "DoWhy refutation_result did not expose 'subset_effects'. "
                                "Single-point coverage would not answer the consistency "
                                "question. Mark SKIPPED rather than fabricate."
                            ),
                            "ci_coverage_available": False,
                            "subset_fraction": self.config["data_subset"]["subset_fraction"],
                            "num_subsets": self.config["data_subset"]["num_subsets"],
                        },
                        execution_time_ms=execution_time,
                    )
                ci_coverage = self._calculate_ci_coverage(subset_effects, original_ci)
            except RefutationError:
                raise
            except Exception as e:
                # F-014 fail-closed: no silent mock fallback.
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    f"DoWhy data_subset refuter failed: {e}",
                    details={
                        "test_name": "data_subset",
                        "original_effect": original_effect,
                    },
                    original_error=e,
                ) from e
        else:
            # F-014 fail-closed: defense-in-depth for legacy non-agent callers.
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                "data_subset test requires a real DoWhy CausalModel; "
                "caller passed causal_model=None.",
                details={
                    "test_name": "data_subset",
                    "dowhy_available": DOWHY_AVAILABLE,
                    "original_effect": original_effect,
                },
            )

        delta_percent = (
            abs(refuted_effect - original_effect) / max(abs(original_effect), 1e-10) * 100
        )

        # Determine status based on CI coverage
        if ci_coverage >= self.thresholds["subset_ci_coverage"]["pass"]:
            status = RefutationStatus.PASSED
            message = f"Effect consistent across {int(ci_coverage * 100)}% of data subsets"
        elif ci_coverage >= self.thresholds["subset_ci_coverage"]["warning"]:
            status = RefutationStatus.WARNING
            message = f"Effect varies in {int((1 - ci_coverage) * 100)}% of subsets"
        else:
            status = RefutationStatus.FAILED
            message = f"WARNING: Effect inconsistent across data subsets ({int(ci_coverage * 100)}% coverage)"

        execution_time = (time.time() - start_time) * 1000

        return RefutationResult(
            test_name=test_name,
            status=status,
            original_effect=original_effect,
            refuted_effect=refuted_effect,
            p_value=p_value,
            delta_percent=delta_percent,
            details={
                "message": message,
                "ci_coverage": ci_coverage,
                "subset_fraction": self.config["data_subset"]["subset_fraction"],
                "num_subsets": self.config["data_subset"]["num_subsets"],
            },
            execution_time_ms=execution_time,
        )

    def _run_bootstrap_test(
        self,
        original_effect: float,
        original_ci: Tuple[float, float],
        causal_model: Optional[Any],
        identified_estimand: Optional[Any],
        estimate: Optional[Any],
        use_dowhy: bool,
    ) -> RefutationResult:
        """Run bootstrap stability test.

        Tests effect stability via bootstrap resampling.
        """
        import time

        start_time = time.time()

        test_name = RefutationTestType.BOOTSTRAP

        if use_dowhy and causal_model is not None:
            try:
                refutation = causal_model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="bootstrap_refuter",
                    num_simulations=self.config["bootstrap"]["num_bootstraps"],
                )
                # DoWhy's BootstrapRefuter exposes the bootstrapped mean via
                # ``new_effect`` and ``p_value`` via ``refutation_result``.
                # Older / stub variants may also expose ``bootstrap_estimates``.
                # Iter-6 codex H-iter5-3: the bootstrap test answers
                # "what is the variance / stability of the effect under
                # resampling?". That is a DISTRIBUTIONAL question requiring
                # per-bootstrap effects. When DoWhy does not expose
                # ``bootstrap_estimates``, delta-based stability against
                # original_ci is a different question (point-in-interval
                # check, not variance). Mark SKIPPED rather than substitute.
                bootstrap_effects = refutation.refutation_result.get("bootstrap_estimates", [])
                # Iter-2 codex H4: p_value must come from real refuter output.
                p_value = _require_p_value(refutation, "bootstrap", original_effect)
                if not bootstrap_effects:
                    refuted_effect = float(refutation.new_effect)
                    execution_time = (time.time() - start_time) * 1000
                    return RefutationResult(
                        test_name=test_name,
                        status=RefutationStatus.SKIPPED,
                        original_effect=original_effect,
                        refuted_effect=refuted_effect,
                        p_value=p_value,
                        delta_percent=0.0,
                        details={
                            "message": (
                                "Bootstrap variance check skipped: "
                                "DoWhy refutation_result did not expose 'bootstrap_estimates'. "
                                "Delta-vs-original-CI would not answer the variance "
                                "question. Mark SKIPPED rather than fabricate."
                            ),
                            "bootstrap_ci_available": False,
                            "num_bootstraps": self.config["bootstrap"]["num_bootstraps"],
                        },
                        execution_time_ms=execution_time,
                    )
                refuted_effect = float(np.mean(bootstrap_effects))
                bootstrap_ci = (
                    float(np.percentile(bootstrap_effects, 2.5)),
                    float(np.percentile(bootstrap_effects, 97.5)),
                )
            except RefutationError:
                raise
            except Exception as e:
                # F-014 fail-closed: no silent mock fallback.
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    f"DoWhy bootstrap refuter failed: {e}",
                    details={
                        "test_name": "bootstrap",
                        "original_effect": original_effect,
                    },
                    original_error=e,
                ) from e
        else:
            # F-014 fail-closed: defense-in-depth for legacy non-agent callers.
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                "bootstrap test requires a real DoWhy CausalModel; "
                "caller passed causal_model=None.",
                details={
                    "test_name": "bootstrap",
                    "dowhy_available": DOWHY_AVAILABLE,
                    "original_effect": original_effect,
                },
            )

        # Iter-6 codex H-iter5-3: by this point ``bootstrap_ci_available`` is
        # guaranteed True (the ``not bootstrap_effects`` branch above returned
        # SKIPPED early). We compute CI ratio from real bootstrap percentiles.
        delta_percent = (
            abs(refuted_effect - original_effect) / max(abs(original_effect), 1e-10) * 100
        )
        original_ci_width = original_ci[1] - original_ci[0]
        bootstrap_ci_width = bootstrap_ci[1] - bootstrap_ci[0]
        ci_ratio = bootstrap_ci_width / max(original_ci_width, 1e-10)

        if ci_ratio <= self.thresholds["bootstrap_ci_ratio"]["pass"]:
            status = RefutationStatus.PASSED
            message = f"Effect stable across {self.config['bootstrap']['num_bootstraps']} bootstrap samples"
        elif ci_ratio <= self.thresholds["bootstrap_ci_ratio"]["warning"]:
            status = RefutationStatus.WARNING
            message = "Bootstrap CI moderately wider than original"
        else:
            status = RefutationStatus.FAILED
            message = "WARNING: High variance in bootstrap estimates"

        execution_time = (time.time() - start_time) * 1000

        return RefutationResult(
            test_name=test_name,
            status=status,
            original_effect=original_effect,
            refuted_effect=refuted_effect,
            p_value=p_value,
            delta_percent=delta_percent,
            details={
                "message": message,
                "bootstrap_ci": bootstrap_ci,
                "ci_ratio": ci_ratio,
                "bootstrap_ci_available": True,
                "num_bootstraps": self.config["bootstrap"]["num_bootstraps"],
            },
            execution_time_ms=execution_time,
        )

    def _run_sensitivity_test(
        self,
        original_effect: float,
        original_ci: Tuple[float, float],
        outcome_std: Optional[float] = None,
    ) -> RefutationResult:
        """Run E-value sensitivity analysis.

        Calculates the E-value to assess robustness to unmeasured confounding.
        Based on VanderWeele & Ding (2017).

        H3 fix: the Chinn(2000)/VanderWeele-Ding ``RR ≈ exp(0.91·d)`` approximation
        requires a STANDARDIZED mean difference d. The effect/CI arrive in native
        outcome units, so they MUST be divided by the outcome SD first — otherwise
        the E-value is scale-dependent (near 1 on a 0–1 outcome, exploding on a
        dollar/count outcome) and ``sensitivity_e_value`` is a critical gate, so a
        meaningless number can hard-BLOCK or wave through depending only on units.
        """
        import time

        start_time = time.time()

        test_name = RefutationTestType.SENSITIVITY_E_VALUE

        # Calculate E-value using VanderWeele formula
        # E-value = RR + sqrt(RR * (RR - 1)) where RR is the relative risk
        abs_effect = abs(original_effect)
        ci_bound = min(abs(original_ci[0]), abs(original_ci[1]))

        # H3: standardize the effect + CI bound by the outcome SD before the
        # exp(0.91·d) step (d must be a standardized mean difference). Guard a
        # non-positive / missing SD (constant outcome or no data passthrough) —
        # in that degenerate case we cannot standardize and flag it in details.
        standardized = False
        if outcome_std is not None and np.isfinite(outcome_std) and outcome_std > 0:
            abs_effect = abs_effect / outcome_std
            ci_bound = ci_bound / outcome_std
            standardized = True

        # Approximate risk ratio from the (now standardized) effect.
        rr = np.exp(0.91 * abs_effect)
        e_value = rr + np.sqrt(rr * (rr - 1)) if rr > 1 else 1.0

        # E-value for CI bound (more conservative)
        rr_ci = np.exp(0.91 * ci_bound)
        e_value_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1)) if rr_ci > 1 else 1.0

        threshold = self.thresholds["e_value_min"]["pass"]
        warning_threshold = self.thresholds["e_value_min"]["warning"]

        # Determine status
        if e_value >= threshold:
            status = RefutationStatus.PASSED
            message = f"E-value {e_value:.2f} indicates robustness to unmeasured confounding"
            strength = "strong" if e_value >= 3.0 else "moderate"
        elif e_value >= warning_threshold:
            status = RefutationStatus.WARNING
            message = f"E-value {e_value:.2f} suggests moderate sensitivity to confounding"
            strength = "weak"
        else:
            status = RefutationStatus.FAILED
            message = (
                f"WARNING: Low E-value {e_value:.2f} indicates high sensitivity to confounding"
            )
            strength = "very_weak"

        execution_time = (time.time() - start_time) * 1000

        return RefutationResult(
            test_name=test_name,
            status=status,
            original_effect=original_effect,
            refuted_effect=original_effect,  # E-value doesn't produce refuted effect
            p_value=None,  # Not applicable for E-value
            delta_percent=0.0,
            details={
                "message": message,
                "e_value": e_value,
                "e_value_ci": e_value_ci,
                "threshold": threshold,
                "confounder_strength": strength,
                # H3: surface whether the effect was standardized + the SD used,
                # so a scale-dependent (unstandardized) E-value is not mistaken
                # for a comparable one.
                "standardized": standardized,
                "outcome_std": outcome_std,
            },
            execution_time_ms=execution_time,
        )

    # ========================================================================
    # F-014 (#416): The previous ``_mock_*`` methods that simulated placebo,
    # random_common_cause, data_subset, and bootstrap tests via seeded random
    # noise have been DELETED. The agent refutation node now reconstructs a
    # real DoWhy ``CausalModel`` (via
    # ``src/agents/causal_impact/nodes/refutation.py::_reconstruct_dowhy_artifacts``)
    # before invoking ``run_all_tests``, and the per-test methods above
    # raise ``RefutationError`` when ``causal_model is None`` so no caller
    # can silently dispatch to mock paths.
    #
    # Per ``CLAUDE.md`` §"CRITICAL — Anti-Mocking & Verification Discipline":
    # mock surfaces with zero non-test production consumers must be DELETED,
    # not LABELED. Consumer grep at commit time verified that the only
    # external consumers were the per-test fallbacks in this file (now
    # replaced with ``RefutationError`` raises) and the test fixtures in
    # ``tests/unit/test_causal_engine/test_refutation_runner.py`` (also
    # updated in this PR to test the structured-error path).
    # ========================================================================

    def _calculate_ci_coverage(
        self, subset_effects: List[float], original_ci: Tuple[float, float]
    ) -> float:
        """Calculate what fraction of subset effects fall within original CI.

        Args:
            subset_effects: List of per-subset effect estimates from a data-subset
                refuter. MUST be non-empty; the caller is responsible for handling
                the empty case (no silent default — see F-014 #416).
            original_ci: Original confidence interval (lower, upper).

        Returns:
            Fraction in [0, 1].

        Raises:
            ValueError: if ``subset_effects`` is empty. This is intentional:
                a silent ``0.9`` default would mask the fact that the refuter
                returned no per-subset data. Callers must either get real
                subset effects from the refuter, or compute coverage via a
                single-point check at the call site.
        """
        if not subset_effects:
            raise ValueError(
                "_calculate_ci_coverage requires non-empty subset_effects; "
                "the caller must handle the empty case explicitly (e.g., "
                "single-point CI check) instead of relying on a silent default."
            )
        count_in_ci = sum(1 for e in subset_effects if original_ci[0] <= e <= original_ci[1])
        return count_in_ci / len(subset_effects)

    # ========================================================================
    # SCORING AND GATE DECISION
    # ========================================================================

    def _calculate_confidence_score(self, tests: List[RefutationResult]) -> float:
        """Calculate weighted confidence score from all tests.

        Weights:
        - Critical tests (placebo, random_common_cause, sensitivity): 0.25 each
        - Non-critical tests (data_subset, bootstrap): 0.125 each

        Args:
            tests: List of test results

        Returns:
            Confidence score between 0 and 1
        """
        if not tests:
            return 0.0

        weights = {
            RefutationTestType.PLACEBO_TREATMENT: 0.25,
            RefutationTestType.RANDOM_COMMON_CAUSE: 0.25,
            RefutationTestType.SENSITIVITY_E_VALUE: 0.25,
            RefutationTestType.DATA_SUBSET: 0.125,
            RefutationTestType.BOOTSTRAP: 0.125,
        }

        status_scores = {
            RefutationStatus.PASSED: 1.0,
            RefutationStatus.WARNING: 0.6,
            RefutationStatus.FAILED: 0.0,
            RefutationStatus.SKIPPED: 0.5,  # Neutral
        }

        total_weight = 0.0
        weighted_score = 0.0

        for test in tests:
            weight = weights.get(test.test_name, 0.1)
            score = status_scores.get(test.status, 0.5)
            weighted_score += weight * score
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_score / total_weight

    def _determine_gate_decision(
        self, tests: List[RefutationResult], confidence_score: float
    ) -> GateDecision:
        """Determine gate decision based on test results and confidence.

        Rules:
        1. If any CRITICAL test FAILED → BLOCK
        2. If confidence >= 0.70 → PROCEED
        3. If confidence >= 0.50 → REVIEW
        4. Otherwise → BLOCK

        Args:
            tests: List of test results
            confidence_score: Weighted confidence score

        Returns:
            Gate decision (proceed, review, or block)
        """
        # Check for critical test failures
        critical_tests = {
            RefutationTestType.PLACEBO_TREATMENT,
            RefutationTestType.RANDOM_COMMON_CAUSE,
            RefutationTestType.SENSITIVITY_E_VALUE,
        }

        for test in tests:
            if test.test_name in critical_tests and test.status == RefutationStatus.FAILED:
                logger.warning(f"Critical test {test.test_name.value} failed → BLOCK")
                return GateDecision.BLOCK

        # Apply confidence thresholds
        if confidence_score >= self.GATE_THRESHOLDS["proceed"]:
            return GateDecision.PROCEED
        elif confidence_score >= self.GATE_THRESHOLDS["review"]:
            return GateDecision.REVIEW
        else:
            return GateDecision.BLOCK


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def run_refutation_suite(
    original_effect: float,
    original_ci: Tuple[float, float],
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    brand: Optional[str] = None,
    config: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    causal_model: Optional[Any] = None,
    identified_estimand: Optional[Any] = None,
    estimate: Optional[Any] = None,
    data: Optional[Any] = None,
    estimate_id: Optional[str] = None,
) -> RefutationSuite:
    """Convenience function to run refutation suite.

    Iter-4 codex H3 (#416): the keyword-only model artifacts are OPTIONAL
    in the signature (preserving the iter-0 positional contract for
    ``original_effect`` / ``original_ci`` / ``treatment`` / ``outcome`` /
    ``brand`` / ``config``) — but they are FUNCTIONALLY REQUIRED. If any of
    ``causal_model`` / ``identified_estimand`` / ``estimate`` is None, this
    function fail-closes with ``RefutationError`` (not ``TypeError``). This
    preserves call compatibility (legacy callers still bind their args
    correctly) while still rejecting the silent-mock dispatch that F-014
    closed.

    Args:
        original_effect: ATE to validate
        original_ci: Confidence interval
        treatment: Treatment variable name (logging only)
        outcome: Outcome variable name (logging only)
        brand: Brand context (logging only)
        config: Custom test configuration
        causal_model: DoWhy CausalModel instance (REQUIRED at runtime).
            None raises RefutationError, not TypeError.
        identified_estimand: DoWhy identified estimand (REQUIRED at runtime).
        estimate: DoWhy estimate object (REQUIRED at runtime).
        data: DataFrame used for the estimate (passed to refuters)
        estimate_id: Estimate ID for persistence

    Returns:
        RefutationSuite with results

    Raises:
        RefutationError: when any model artifact is missing, or when
            refuters fail / a per-test placeholder would be required.
    """
    if causal_model is None or identified_estimand is None or estimate is None:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "run_refutation_suite requires a real DoWhy CausalModel + identified_estimand "
            "+ estimate; F-014 closed the silent-mock fallback that previously dispatched "
            "to _mock_* paths when these were None.",
            details={
                "reason": "missing_model_artifacts",
                "has_causal_model": causal_model is not None,
                "has_identified_estimand": identified_estimand is not None,
                "has_estimate": estimate is not None,
                "treatment": treatment,
                "outcome": outcome,
            },
        )
    runner = RefutationRunner(config=config)
    return runner.run_all_tests(
        original_effect=original_effect,
        original_ci=original_ci,
        causal_model=causal_model,
        identified_estimand=identified_estimand,
        estimate=estimate,
        treatment=treatment,
        outcome=outcome,
        brand=brand,
        data=data,
        estimate_id=estimate_id,
    )


def is_estimate_valid(suite: RefutationSuite) -> bool:
    """Check if estimate passed validation (not blocked).

    Args:
        suite: Refutation suite results

    Returns:
        True if estimate can be used (proceed or review)
    """
    return suite.gate_decision != GateDecision.BLOCK
