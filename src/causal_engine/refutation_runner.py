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

References (#1979):
- docs/decisions/adr-017-discovery-corroboration-and-honest-provenance.md
- docs/lineage/causal_dag_lineage.html (governance section)
"""

from __future__ import annotations

import copy
import logging
import time
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
# REAL NON-CRITICAL EVIDENCE (lane 1, spec §4.1)
# ============================================================================
# DoWhy 0.14's data_subset / bootstrap refuters compute per-sample effects and
# keep only their mean and a p-value on the CausalRefutation object, so the two
# distributional tests below could never score and were recorded SKIPPED on
# every live run (96/96 measured 2026-09-08). The loops below re-fit the SAME
# reported estimator with the SAME public calls DoWhy's ``_refute_once`` uses
# and keep every effect. Reference interval: ``original_ci`` from the
# estimation node (the reported interval) -- never the reconstruction's own.

_MIN_SUBSET_RESAMPLES = 3
_MIN_BOOTSTRAP_RESAMPLES = 10


def _refit_effect_on(new_data: Any, identified_estimand: Any, estimate: Any) -> float:
    """Re-fit the reported estimator on ``new_data`` and return its effect.

    The four calls are the public estimator API DoWhy 0.14's own
    ``data_subset_refuter._refute_once`` / ``bootstrap_refuter._refute_once``
    use; nothing here substitutes a different model.
    """
    new_estimator = estimate.estimator.get_new_estimator_object(identified_estimand)
    fit_params = getattr(new_estimator, "_fit_params", None) or {}
    new_estimator.fit(
        new_data,
        effect_modifier_names=estimate.estimator._effect_modifier_names,
        **fit_params,
    )
    new_effect = new_estimator.estimate_effect(
        new_data,
        control_value=estimate.control_value,
        treatment_value=estimate.treatment_value,
        target_units=estimate.estimator._target_units,
    )
    return float(new_effect.value)


def _refutation_frame(causal_model: Any, test_name: str, original_effect: float) -> Any:
    """The frame the CausalModel was built on (DoWhy stores it as ``_data``)."""
    frame = getattr(causal_model, "_data", None)
    if frame is None or not hasattr(frame, "sample") or not hasattr(frame, "columns"):
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"{test_name} needs the CausalModel's DataFrame (``_data``) to resample; "
            "the model exposes none.",
            details={
                "test_name": test_name,
                "original_effect": original_effect,
                "reason": "refutation_frame_missing",
            },
        )
    return frame


def _resample_effects(
    *,
    kind: str,
    frame: Any,
    identified_estimand: Any,
    estimate: Any,
    requested: int,
    rng: np.random.Generator,
    deadline: Optional[float],
    subset_fraction: float = 0.8,
) -> Tuple[List[float], bool]:
    """Run up to ``requested`` re-fits, stopping at the cooperative deadline.

    ``kind`` is ``"subset"`` (``frame.sample(frac=subset_fraction)``) or
    ``"bootstrap"`` (row resample WITH replacement, same size; no confounder
    noise -- DoWhy's default bootstrap refuter also perturbs the chosen
    covariates, which answers a measurement-error question, not the variance
    question this test scores). Each draw is seeded from ``rng`` so a seeded
    caller reproduces its evidence. Returns ``(effects, stopped_for_budget)``.
    """
    effects: List[float] = []
    for _ in range(max(1, int(requested))):
        if deadline is not None and time.monotonic() >= deadline:
            return effects, True
        seed = int(rng.integers(0, 2**31 - 1))
        if kind == "subset":
            new_data = frame.sample(frac=subset_fraction, random_state=seed)
        else:
            new_data = frame.sample(n=len(frame), replace=True, random_state=seed)
        effects.append(_refit_effect_on(new_data, identified_estimand, estimate))
    return effects, False


def _significance_p_value(
    estimate: Any, effects: List[float], test_name: str, original_effect: float
) -> float:
    """p-value of the reported estimate under the resample distribution --
    DoWhy's own ``test_significance`` (the refuters' p-value), kept real."""
    try:
        from dowhy.causal_refuter import test_significance
    except ImportError as ie:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "DoWhy import failed while scoring resample evidence.",
            details={"test_name": test_name, "reason": "dowhy_import_failed"},
            original_error=ie,
        ) from ie
    result = test_significance(estimate, np.asarray(effects, dtype=float))
    pv = result.get("p_value") if isinstance(result, dict) else None
    if pv is None or not np.isfinite(float(pv)):
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"{test_name} significance test returned no finite p_value; refusing to "
            "substitute a placeholder.",
            details={
                "test_name": test_name,
                "original_effect": original_effect,
                "reason": "missing_p_value",
            },
        )
    return float(pv)


def _require_finite_effects(effects: List[float], test_name: str, original_effect: float) -> None:
    """Fail closed on a NaN / inf re-fit (spec §5: an anomaly inside the loop is
    treated like an exception). A non-finite effect would otherwise be SCORED:
    DoWhy's percentile test counts NaN as "below the estimate", np.percentile
    poisons the bootstrap interval, and coverage silently drops one sample."""
    for i, e in enumerate(effects):
        if not np.isfinite(e):
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                f"{test_name} re-fit #{i} returned a non-finite effect ({e!r}); refusing "
                "to score a distribution that contains it.",
                details={
                    "test_name": test_name,
                    "original_effect": original_effect,
                    "reason": "non_finite_resample_effect",
                    "resamples_completed": len(effects),
                    "first_non_finite_index": i,
                },
            )


def _budget_skip_result(
    test_name: RefutationTestType,
    original_effect: float,
    completed: int,
    requested: int,
    minimum: int,
    stopped: bool,
    config_details: Dict[str, Any],
    execution_time_ms: float = 0.0,
) -> RefutationResult:
    """Honest SKIPPED when fewer than ``minimum`` re-fits completed.

    ``stopped`` says WHY: the deadline stopped the loop (``time_budget``, same
    ``reason`` / ``message`` contract as the #1419 pre-start skip) or the loop
    ran to completion because the configured count is below the minimum
    (``config_below_minimum``) -- a skip must not blame the budget when the
    budget was never hit.
    """
    name = test_name.value
    if stopped:
        reason = (
            "time_budget — non-critical test stopped before its minimum resample "
            "count; the critical gates decide the suite"
        )
        message = (
            f"{name} skipped: {completed}/{requested} resamples completed before the "
            f"compute deadline (minimum {minimum}); non-critical, degraded honestly"
        )
    else:
        reason = (
            "config_below_minimum — requested resample count is below the test's "
            "minimum; the critical gates decide the suite"
        )
        message = (
            f"{name} skipped: {completed}/{requested} resamples requested, below the "
            f"minimum {minimum}; non-critical, degraded honestly"
        )
    return RefutationResult(
        test_name=test_name,
        status=RefutationStatus.SKIPPED,
        original_effect=original_effect,
        refuted_effect=original_effect,
        details={
            "reason": reason,
            "message": message,
            "resamples_completed": completed,
            "resamples_requested": requested,
            "stopped_for_budget": stopped,
            **config_details,
        },
        execution_time_ms=execution_time_ms,
    )


def _degenerate_skip_result(
    test_name: RefutationTestType,
    original_effect: float,
    effects: List[float],
    requested: int,
    stopped: bool,
    config_details: Dict[str, Any],
    execution_time_ms: float = 0.0,
) -> RefutationResult:
    """Honest SKIPPED when every re-fit returned the SAME effect (owner decision
    2026-09-09). A zero-variance distribution cannot be scored (DoWhy's normal
    test divides by its standard deviation) and a constant re-fit is not
    evidence of instability: an estimator that ignores its data fails the
    CRITICAL placebo test, which decides the suite. Never a placeholder p-value,
    never a fail-closed halt from a non-critical test."""
    name = test_name.value
    return RefutationResult(
        test_name=test_name,
        status=RefutationStatus.SKIPPED,
        original_effect=original_effect,
        refuted_effect=float(effects[0]),
        details={
            "reason": (
                "degenerate_resample_distribution — every re-fit returned the same "
                "effect; a zero-variance distribution cannot be scored; the critical "
                "gates decide the suite"
            ),
            "message": (
                f"{name} skipped: {len(effects)} re-fits all returned "
                f"{float(effects[0]):.6g}; non-critical, degraded honestly"
            ),
            "resample_effects": [float(e) for e in effects],
            "resamples_completed": len(effects),
            "resamples_requested": requested,
            "stopped_for_budget": stopped,
            **config_details,
        },
        execution_time_ms=execution_time_ms,
    )


def _degenerate_ci_skip_result(
    test_name: RefutationTestType,
    original_effect: float,
    original_ci: Tuple[float, float],
    config_details: Dict[str, Any],
    execution_time_ms: float = 0.0,
) -> RefutationResult:
    """Honest SKIPPED, decided BEFORE any re-fit, when the reported interval has
    no width: coverage of a point and a width ratio against ~0 cannot be scored
    and would blame the estimate for an upstream degenerate interval."""
    name = test_name.value
    return RefutationResult(
        test_name=test_name,
        status=RefutationStatus.SKIPPED,
        original_effect=original_effect,
        refuted_effect=original_effect,
        details={
            "reason": (
                "original_ci_degenerate — the reported interval has no width, so "
                "coverage / width ratio cannot be scored; the critical gates decide "
                "the suite"
            ),
            "message": (
                f"{name} skipped: original_ci={tuple(original_ci)!r} has width "
                f"{float(original_ci[1] - original_ci[0]):.6g}; no re-fit was run; "
                "non-critical, degraded honestly"
            ),
            "original_ci": (float(original_ci[0]), float(original_ci[1])),
            "resamples_completed": 0,
            "stopped_for_budget": False,
            **config_details,
        },
        execution_time_ms=execution_time_ms,
    )


def _resample_seed_for(estimate_id: Optional[str]) -> Optional[int]:
    """Stable 31-bit seed from the estimate id (``None`` → unseeded, as before).

    The first 8 hex digits of the digest are 32 bits (measured max 4294943764
    over the live ids, 2026-09-09); the mask keeps the promise in this docstring.
    """
    if not estimate_id:
        return None
    import hashlib

    return int(hashlib.sha256(str(estimate_id).encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


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

    # PROCEED requires confidence >= 0.70 and NO critical test FAILED; a critical
    # test in WARNING still permits PROCEED (see _determine_gate_decision).
    PROCEED = "proceed"  # Confidence >= 0.70 and no critical test FAILED
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

        #1249: ``skipped_tests`` ({contract_key: reason}, always present)
        records WHY a test was SKIPPED. Skipped entries stay out of
        ``individual_tests``/``total_tests`` (#1219 — no fake-FAILED rows).
        """
        # Build Dict with test names as keys (contract requirement)
        individual_tests: Dict[str, Dict[str, Any]] = {}
        skipped_tests: Dict[str, str] = {}
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

            # SKIPPED (not-applicable) tests are OMITTED from individual_tests:
            # this legacy dict is pass/fail-shaped (``passed: bool``), so a
            # skipped test — e.g. the E-value on a randomized design — would
            # surface as a FAILED row in the FE while ``total_tests`` already
            # excludes it. Absent key → consumers three-state to None (audit
            # chain) / not-narrated (interpretation), the honest rendering for
            # "did not gate". #1249: the skip *reason* survives in
            # ``skipped_tests`` (same contract-key space) so downstream
            # consumers can distinguish "skipped because X" from "never
            # attempted".
            if t.status == RefutationStatus.SKIPPED:
                skipped_tests[key] = t.details.get("message", "")
                continue

            individual_tests[key] = {
                "test_name": t.test_name.value,
                "passed": t.status == RefutationStatus.PASSED,
                # #1867: the three-state verdict (passed/warning/failed) — the
                # two-state ``passed`` collapses WARNING into FAILED for display.
                "status": t.status.value,
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
            # #1249: opt-in observability — {contract_key: reason} for tests
            # that were SKIPPED (not applicable / data unavailable). Always
            # present ({} when nothing skipped) so consumers read it without
            # KeyError. Never feeds individual_tests/total_tests.
            "skipped_tests": skipped_tests,
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
            # ratio = bootstrap_width / original_width. A stable estimate's
            # bootstrap interval is about as wide as its analytic one (ratio
            # ≈ 1; measured 1.01 and 0.81 on two live pairs, 2026-09-08). The
            # pre-lane-1 values 0.50 / 0.75 contradicted the comment beside them
            # ("must not be > 50% wider") and would have failed nearly every
            # real run; they never scored because the test was always SKIPPED.
            "pass": 1.50,  # bootstrap CI at most 50% wider than the original
            "warning": 1.75,
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
        deadline: Optional[float] = None,
        per_refit_hint: Optional[float] = None,
        per_refit_hint_heavy: Optional[float] = None,
        randomized_design: bool = False,
        outcome_std: Optional[float] = None,
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
            deadline: Absolute ``time.monotonic()`` second by which the suite
                must be done. Each refuter re-fits the estimator many times and
                CANNOT be cancelled once started in a worker thread, so when a
                deadline is given we skip any refuter whose ESTIMATED cost
                (per-refit time observed so far x its simulation count) would
                run past it, and fail-closed with a ``RefutationError``. This
                lets a timed-out run return cleanly instead of orphaning compute
                past the caller's hard wall-clock cap. ``None`` (default) =
                unbounded (no behavior change for existing callers).
            per_refit_hint: Optional a-priori per-refit cost (seconds), e.g. the
                wall-time of the single estimator reconstruction fit the caller
                already did (≈ one refit). Used to GATE the FIRST refuter too —
                without it the first refuter would have no cost estimate and run
                unconditionally, so a single pathologically-slow first refuter
                (e.g. a non-converging propensity at ~40s/refit x 30 sims) could
                still straddle the hard cap and orphan one thread. Only used
                when no refuter has run yet; ignored once a real per-refit time
                is observed.
            per_refit_hint_heavy: Optional a-priori cost (seconds) of an
                INFERENCE-BEARING refit — i.e. the reconstruction fit's
                wall-time. #1419 measured (live 5k subsample): placebo /
                random_common_cause / data_subset sims are point refits at
                ~1.6-2.6 s, but a bootstrap sim re-runs the full inference
                machinery at ~11.7 s ≈ the reconstruction fit. Gating bootstrap
                on the cheap observed per-refit would START a run that
                overshoots the deadline ~5x and orphans the worker thread —
                so bootstrap gates on ``max(observed, heavy)`` when this hint
                is provided. ``None`` keeps the observed/hint behavior for
                every test (non-agent callers unchanged).
            randomized_design: DESIGN declaration that the treatment assignment
                is genuinely randomized (threaded from the dataset spec by the
                caller — never inferred from an empty discovered backdoor, which
                would fail-open observational questions). When True the
                ``sensitivity_e_value`` test is reported as SKIPPED/not
                applicable (its threat model — unmeasured confounding of
                assignment — is excluded by construction) with the computed
                E-value kept in details for information. The four
                data-perturbation refuters (placebo, random_common_cause,
                data_subset, bootstrap) still run and still gate.
            outcome_std: Caller-supplied outcome SD for the e-value's effect
                standardization. #1419: the agent node passes the SUBSAMPLE as
                ``data`` but the e-value standardizes the FULL-frame reported
                effect — a scale-sensitive critical gate must be standardized
                on the same frame as the effect it gates, so the node computes
                this on the full frame BEFORE subsampling. ``None`` (default)
                keeps the data-derived SD for existing callers.

        Returns:
            RefutationSuite with all test results and gate decision
        """
        start_time = time.time()

        # --- Cooperative time budget (orphan-fix) ------------------------- #
        # Track per-refit cost so we can estimate whether the NEXT refuter will
        # fit before ``deadline``. ``skipped_for_budget`` records refuters we
        # refused to start; a non-empty list at the end => fail-closed.
        _budget = {"sim_time": 0.0, "sims": 0}
        skipped_for_budget: List[str] = []

        def _sims_for(name: str) -> int:
            cfg = self.config.get(name, {})
            val = cfg.get("num_simulations") or cfg.get("num_bootstraps") or cfg.get("num_subsets")
            try:
                return max(1, int(val)) if val is not None else 1
            except (TypeError, ValueError):
                return 1

        def _budget_allows(n_sims: int, heavy: bool = False) -> bool:
            """True if this refuter may run under the deadline.

            ``heavy=True`` marks an inference-bearing refuter (bootstrap): its
            per-sim cost is ~the reconstruction fit, not the cheap point refit
            the observed average reflects (#1419 measured ~5x gap) — gate it
            on ``max(observed, per_refit_hint_heavy)`` so a run that cannot
            finish is never started.
            """
            if deadline is None:
                return True
            now = time.monotonic()
            if now >= deadline:
                return False
            if _budget["sims"] > 0:
                # Use the per-refit cost observed from refuters already run.
                per_refit = _budget["sim_time"] / _budget["sims"]
            elif per_refit_hint is not None and per_refit_hint > 0:
                # No refuter has run yet, but the caller measured the single
                # estimator reconstruction fit (≈ one refit) — gate even the
                # FIRST refuter with it so a slow first fit cannot run
                # unconditionally and orphan past the hard cap.
                per_refit = per_refit_hint
            elif heavy and per_refit_hint_heavy is not None and per_refit_hint_heavy > 0:
                per_refit = per_refit_hint_heavy
            else:
                # No observation and no hint (non-agent callers that pass a bare
                # deadline) — allow the first refuter to run and calibrate.
                return True
            if heavy and per_refit_hint_heavy is not None and per_refit_hint_heavy > 0:
                per_refit = max(per_refit, per_refit_hint_heavy)
            return now + per_refit * max(1, n_sims) <= deadline

        def _record(n_sims: int, elapsed: float) -> None:
            _budget["sim_time"] += elapsed
            _budget["sims"] += max(1, n_sims)

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

        # Run each enabled test with Opik tracing. Before each refuter we check
        # the cooperative time budget: if its estimated cost would run past the
        # deadline we SKIP it (recording the skip) rather than start work that
        # cannot be cancelled and would orphan the worker thread past the cap.
        #
        # #1419 ordering: CRITICAL gates first, cheapest first, so a deadline
        # that dies mid-suite costs non-critical evidence, not the gate itself.
        # The analytic e-value leads (no refits, ~ms — the per-refit cost model
        # does not apply, so it is gated on bare ``now < deadline`` and its
        # elapsed time is NOT recorded: feeding its ~ms into the observed
        # per-refit average would collapse the average and disarm the orphan
        # guard for every refit-based test after it). Then the two refit-based
        # criticals (placebo, random_common_cause), then the non-critical
        # data_subset and bootstrap, which the #1419 skip policy degrades to
        # honest SKIPPED results when the budget runs out.
        if self.config["sensitivity_e_value"]["enabled"]:
            if deadline is None or time.monotonic() < deadline:
                # H3: the E-value needs a STANDARDIZED effect. A caller-supplied
                # ``outcome_std`` wins (#1419: ``data`` may be the refutation
                # SUBSAMPLE while the gated effect is the FULL-frame estimate);
                # otherwise compute the SD from the passthrough data.
                evalue_outcome_std: Optional[float] = outcome_std
                if evalue_outcome_std is None and data is not None and outcome is not None:
                    try:
                        if outcome in getattr(data, "columns", []):
                            evalue_outcome_std = float(np.std(data[outcome].to_numpy(dtype=float)))
                    except Exception:  # noqa: BLE001 - missing/non-numeric outcome → no standardization
                        evalue_outcome_std = None
                test_result = self._run_test_with_tracing(
                    test_name="sensitivity_e_value",
                    test_func=self._run_sensitivity_test,
                    opik=opik,
                    trace_id=trace_id,
                    estimate_id=estimate_id,
                    original_effect=original_effect,
                    original_ci=original_ci,
                    outcome_std=evalue_outcome_std,
                    randomized_design=randomized_design,
                )
                tests.append(test_result)
            else:
                skipped_for_budget.append("sensitivity_e_value")

        if self.config["placebo_treatment"]["enabled"]:
            _n = _sims_for("placebo_treatment")
            if _budget_allows(_n):
                _t0 = time.monotonic()
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
                _record(_n, time.monotonic() - _t0)
            else:
                skipped_for_budget.append("placebo_treatment")

        if self.config["random_common_cause"]["enabled"]:
            _n = _sims_for("random_common_cause")
            if _budget_allows(_n):
                _t0 = time.monotonic()
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
                _record(_n, time.monotonic() - _t0)
            else:
                skipped_for_budget.append("random_common_cause")

        if self.config["data_subset"]["enabled"]:
            _n = _sims_for("data_subset")
            if _budget_allows(_n):
                _t0 = time.monotonic()
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
                _record(_n, time.monotonic() - _t0)
            else:
                skipped_for_budget.append("data_subset")

        if self.config["bootstrap"]["enabled"]:
            _n = _sims_for("bootstrap")
            if _budget_allows(_n, heavy=True):
                _t0 = time.monotonic()
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
                _record(_n, time.monotonic() - _t0)
            else:
                skipped_for_budget.append("bootstrap")

        # #1419 skip policy. A budget-skipped CRITICAL gate still fails the
        # suite closed — an estimate whose placebo test never ran must not be
        # presented as validated. Budget-skipped NON-critical tests degrade to
        # honest SKIPPED results (reason in ``details`` → persisted via
        # ``details_json``; surfaced via to_legacy_format()['skipped_tests'],
        # the #1249 channel) and the suite completes on the evidence that ran —
        # instead of the pre-#1419 behavior where ANY skip raised and the whole
        # turn failed closed with `ran 0 test(s)`. Criticality reads from the
        # merged config (custom per-test dicts update DEFAULT_CONFIG, which
        # carries the ``critical`` flags).
        if skipped_for_budget:
            ran = [
                t.test_name.value if hasattr(t.test_name, "value") else str(t.test_name)
                for t in tests
            ]
            critical_skipped = [
                name
                for name in skipped_for_budget
                if self.config.get(name, {}).get(
                    "critical", self.DEFAULT_CONFIG.get(name, {}).get("critical", False)
                )
            ]
            if critical_skipped:
                raise RefutationError(
                    "Refutation exceeded its time budget: ran "
                    f"{len(tests)} test(s) {ran}, skipped {skipped_for_budget} "
                    f"(critical: {critical_skipped}) to avoid orphaning compute "
                    "past the worker's wall-clock cap. A suite whose critical "
                    "gates never ran cannot validate the estimate.",
                    details={
                        "reason": "time_budget_exceeded",
                        "ran": ran,
                        "skipped": skipped_for_budget,
                        "critical_skipped": critical_skipped,
                    },
                )
            for name in skipped_for_budget:
                tests.append(
                    RefutationResult(
                        test_name=RefutationTestType(name),
                        status=RefutationStatus.SKIPPED,
                        original_effect=original_effect,
                        refuted_effect=original_effect,
                        details={
                            "reason": (
                                "time_budget — non-critical test skipped to avoid "
                                "orphaning compute past the cooperative deadline; "
                                "the critical gates completed and decide the suite"
                            ),
                            "message": (
                                f"{name} skipped: estimated cost would run past the "
                                "compute deadline (non-critical, degraded honestly)"
                            ),
                        },
                    )
                )
            logger.info(
                "Refutation budget skipped non-critical test(s) %s after running %s.",
                skipped_for_budget,
                ran,
            )

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
        *,
        deadline: Optional[float] = None,
        resample_seed: Optional[int] = None,
    ) -> RefutationResult:
        """Data-subset consistency test on REAL per-subset evidence (spec §4.1).

        Re-fits the reported estimator on ``num_subsets`` random subsets of
        ``subset_fraction`` of the model's frame and scores the SHARE of subset
        effects that fall inside ``original_ci`` (the estimation node's reported
        interval). Stops at ``deadline`` between re-fits; below
        ``_MIN_SUBSET_RESAMPLES`` completed it returns an honest SKIPPED.
        """
        import time

        start_time = time.time()
        test_name = RefutationTestType.DATA_SUBSET

        if not (use_dowhy and causal_model is not None):
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

        cfg = self.config["data_subset"]
        requested = int(cfg["num_subsets"])
        subset_fraction = float(cfg["subset_fraction"])
        config_details = {"subset_fraction": subset_fraction, "num_subsets": requested}
        frame = _refutation_frame(causal_model, "data_subset", original_effect)
        if original_ci[1] - original_ci[0] <= 0:
            return _degenerate_ci_skip_result(
                test_name,
                original_effect,
                original_ci,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        rng = np.random.default_rng(resample_seed)
        try:
            subset_effects, stopped = _resample_effects(
                kind="subset",
                frame=frame,
                identified_estimand=identified_estimand,
                estimate=estimate,
                requested=requested,
                rng=rng,
                deadline=deadline,
                subset_fraction=subset_fraction,
            )
        except RefutationError:
            raise
        except Exception as e:
            # F-014 fail-closed: no silent mock fallback.
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                f"data_subset re-fit failed: {e}",
                details={"test_name": "data_subset", "original_effect": original_effect},
                original_error=e,
            ) from e

        _require_finite_effects(subset_effects, "data_subset", original_effect)
        if len(subset_effects) < _MIN_SUBSET_RESAMPLES:
            return _budget_skip_result(
                test_name,
                original_effect,
                len(subset_effects),
                requested,
                _MIN_SUBSET_RESAMPLES,
                stopped,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # "Every re-fit returned the same effect" is tested EXACTLY (max == min):
        # np.std of n identical floats is not 0.0 for most n (twelve 0.15s give
        # 2.8e-17, measured 2026-09-09), which would let DoWhy's normal test
        # score a constant series with a meaningless p-value.
        if float(np.ptp(subset_effects)) == 0.0:
            return _degenerate_skip_result(
                test_name,
                original_effect,
                subset_effects,
                requested,
                stopped,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        refuted_effect = float(np.mean(subset_effects))
        p_value = _significance_p_value(estimate, subset_effects, "data_subset", original_effect)
        ci_coverage = self._calculate_ci_coverage(subset_effects, original_ci)
        delta_percent = (
            abs(refuted_effect - original_effect) / max(abs(original_effect), 1e-10) * 100
        )

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
                "subset_effects": [float(e) for e in subset_effects],
                "resamples_completed": len(subset_effects),
                "resamples_requested": requested,
                "stopped_for_budget": stopped,
                **config_details,
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
        *,
        deadline: Optional[float] = None,
        resample_seed: Optional[int] = None,
    ) -> RefutationResult:
        """Bootstrap stability test on REAL per-resample evidence (spec §4.1).

        Re-fits the reported estimator on ``num_bootstraps`` row resamples (with
        replacement, same size) of the model's frame; the 2.5th–97.5th
        percentile width of the resample effects is compared with the width of
        ``original_ci``. Thresholds: pass ≤ 1.5×, warning ≤ 1.75×, else failed
        (``PASS_THRESHOLDS["bootstrap_ci_ratio"]``). Stops at ``deadline``
        between re-fits; below ``_MIN_BOOTSTRAP_RESAMPLES`` it returns SKIPPED.
        """
        import time

        start_time = time.time()
        test_name = RefutationTestType.BOOTSTRAP

        if not (use_dowhy and causal_model is not None):
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

        requested = int(self.config["bootstrap"]["num_bootstraps"])
        config_details = {"num_bootstraps": requested}
        frame = _refutation_frame(causal_model, "bootstrap", original_effect)
        if original_ci[1] - original_ci[0] <= 0:
            return _degenerate_ci_skip_result(
                test_name,
                original_effect,
                original_ci,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        rng = np.random.default_rng(resample_seed)
        try:
            bootstrap_effects, stopped = _resample_effects(
                kind="bootstrap",
                frame=frame,
                identified_estimand=identified_estimand,
                estimate=estimate,
                requested=requested,
                rng=rng,
                deadline=deadline,
            )
        except RefutationError:
            raise
        except Exception as e:
            raise RefutationError(
                "Refutation analysis unavailable for this query, retry without refutation. "
                f"bootstrap re-fit failed: {e}",
                details={"test_name": "bootstrap", "original_effect": original_effect},
                original_error=e,
            ) from e

        _require_finite_effects(bootstrap_effects, "bootstrap", original_effect)
        if len(bootstrap_effects) < _MIN_BOOTSTRAP_RESAMPLES:
            return _budget_skip_result(
                test_name,
                original_effect,
                len(bootstrap_effects),
                requested,
                _MIN_BOOTSTRAP_RESAMPLES,
                stopped,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Exact degeneracy check (max == min); see _run_data_subset_test.
        if float(np.ptp(bootstrap_effects)) == 0.0:
            return _degenerate_skip_result(
                test_name,
                original_effect,
                bootstrap_effects,
                requested,
                stopped,
                config_details,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        refuted_effect = float(np.mean(bootstrap_effects))
        p_value = _significance_p_value(estimate, bootstrap_effects, "bootstrap", original_effect)
        bootstrap_ci = (
            float(np.percentile(bootstrap_effects, 2.5)),
            float(np.percentile(bootstrap_effects, 97.5)),
        )
        delta_percent = (
            abs(refuted_effect - original_effect) / max(abs(original_effect), 1e-10) * 100
        )
        original_ci_width = original_ci[1] - original_ci[0]
        bootstrap_ci_width = bootstrap_ci[1] - bootstrap_ci[0]
        ci_ratio = bootstrap_ci_width / max(original_ci_width, 1e-10)

        if ci_ratio <= self.thresholds["bootstrap_ci_ratio"]["pass"]:
            status = RefutationStatus.PASSED
            message = f"Effect stable across {len(bootstrap_effects)} bootstrap samples"
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
                "bootstrap_effects": [float(e) for e in bootstrap_effects],
                "resamples_completed": len(bootstrap_effects),
                "resamples_requested": requested,
                "stopped_for_budget": stopped,
                **config_details,
            },
            execution_time_ms=execution_time,
        )

    def _run_sensitivity_test(
        self,
        original_effect: float,
        original_ci: Tuple[float, float],
        outcome_std: Optional[float] = None,
        randomized_design: bool = False,
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

        ``randomized_design=True`` (a DESIGN declaration threaded from the dataset
        spec — never inferred from an empty discovered backdoor) marks the E-value
        NOT APPLICABLE as a gate: unmeasured confounding of assignment is excluded
        by randomization, so the test returns status=SKIPPED (no confidence weight,
        never a critical failure) while still computing the numbers into details
        for information. Verified live: the honest 8pp nba_triggers RCT effect has
        E-value(CI) < 2.0 and was hard-BLOCKed by this gate (PR #1217 follow-up).
        """
        import time

        start_time = time.time()

        test_name = RefutationTestType.SENSITIVITY_E_VALUE

        # Calculate E-value using VanderWeele formula
        # E-value = RR + sqrt(RR * (RR - 1)) where RR is the relative risk
        abs_effect = abs(original_effect)
        # M-stat1 null-crossing guard: a CI that straddles (or touches) 0 is
        # statistically indistinguishable from the null, so the conservative
        # E-value-for-CI must collapse to 1.0 instead of min(|lo|,|hi|).
        ci_straddles_null = original_ci[0] <= 0.0 <= original_ci[1]
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

        # E-value for CI bound (more conservative). M-stat1: a null-crossing CI
        # collapses this to the null.
        if ci_straddles_null:
            e_value_ci = 1.0
        else:
            rr_ci = np.exp(0.91 * ci_bound)
            e_value_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1)) if rr_ci > 1 else 1.0

        threshold = self.thresholds["e_value_min"]["pass"]
        warning_threshold = self.thresholds["e_value_min"]["warning"]

        # M-stat2: the PASS/WARNING/FAILED decision uses the CONSERVATIVE CI
        # E-value (e_value_ci), not the point estimate. A strong point effect
        # with a wide / null-crossing CI must not wave the gate through.
        if randomized_design:
            # Randomized design: the E-value's threat model (unmeasured
            # confounding of assignment) is excluded by construction, so the
            # gate is NOT APPLICABLE — SKIPPED carries no confidence weight and
            # never trips the critical-failure BLOCK. The computed numbers stay
            # in details as information, not as evidence for/against validity.
            status = RefutationStatus.SKIPPED
            message = (
                "not applicable: randomized design — treatment assignment is "
                "exogenous by construction, so the unmeasured-confounding gate "
                f"does not apply; E-value (CI bound) {e_value_ci:.2f} reported "
                "for information only"
            )
            strength = "not_applicable_randomized"
        elif e_value_ci >= threshold:
            status = RefutationStatus.PASSED
            message = (
                f"E-value (CI bound) {e_value_ci:.2f} indicates robustness to "
                "unmeasured confounding"
            )
            strength = "strong" if e_value_ci >= 3.0 else "moderate"
        elif e_value_ci >= warning_threshold:
            status = RefutationStatus.WARNING
            message = (
                f"E-value (CI bound) {e_value_ci:.2f} suggests moderate sensitivity to confounding"
            )
            strength = "weak"
        else:
            status = RefutationStatus.FAILED
            message = (
                f"WARNING: Low E-value (CI bound) {e_value_ci:.2f} indicates high "
                "sensitivity to confounding"
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
                # Randomized designs report the E-value as information only —
                # the unmeasured-confounding gate does not apply to them.
                "gate_applicable": not randomized_design,
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
        }

        total_weight = 0.0
        weighted_score = 0.0

        for test in tests:
            # SKIPPED tests carry no evidence either way: EXCLUDE them from the
            # average rather than padding at a neutral 0.5 (which would dilute
            # genuinely-passed evidence toward the REVIEW band).
            if test.status == RefutationStatus.SKIPPED:
                continue
            weight = weights.get(test.test_name, 0.1)
            score = status_scores.get(test.status, 0.0)
            weighted_score += weight * score
            total_weight += weight

        if total_weight == 0:
            # No non-skipped evidence (all tests skipped) -> fail closed, not 0.5.
            return 0.0

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
