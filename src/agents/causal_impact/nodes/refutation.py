"""Refutation Node - Robustness testing for causal estimates.

Runs multiple refutation tests to validate causal effect.

Version: 4.3
Integration: Uses RefutationRunner from src.causal_engine for DoWhy-based validation
Persistence: Uses CausalValidationRepository for database storage

Phase 4 Integration:
- Logs ValidationOutcome to Feedback Learner for learning from failures
- Creates failure patterns for ExperimentKnowledgeStore queries

Anti-Mocking (F-014 fix, #416):
- Reconstructs DoWhy CausalModel + identified_estimand + estimate from
  estimation_data passthrough BEFORE calling ``RefutationRunner.run_all_tests``.
- Fail-closed: raises ``RefutationError`` when DoWhy is unavailable OR when
  reconstruction fails. NEVER dispatches to the deleted ``_mock_*`` paths.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, cast

from src.agents.causal_impact.state import (
    CausalImpactState,
    RefutationResults,
)
from src.causal_engine import (
    DOWHY_AVAILABLE,
    GateDecision,
    RefutationError,
    RefutationRunner,
    RefutationSuite,
    # Phase 4: ValidationOutcome for Feedback Learner integration
    create_validation_outcome,
    log_validation_outcome,
)
from src.repositories.causal_validation import CausalValidationRepository

logger = logging.getLogger(__name__)


# Map agent EstimationResult.method values (and selected_estimator type-values)
# to DoWhy backdoor method_name. Iter-2 codex H3: the refutation rebuild MUST
# use the SAME estimator that produced the reported ATE — otherwise refuters
# critique a linear_regression estimate while the chat UI displays a
# CausalForestDML one.
_SELECTOR_TO_DOWHY_METHOD = {
    # Energy-score selector estimator_type.value
    "causal_forest": "backdoor.econml.dml.CausalForestDML",
    "linear_dml": "backdoor.econml.dml.LinearDML",
    "drlearner": "backdoor.econml.dr.DRLearner",
    "ols": "backdoor.linear_regression",
    # EstimationResult.method (legacy + new labels)
    "CausalForestDML": "backdoor.econml.dml.CausalForestDML",
    "LinearDML": "backdoor.econml.dml.LinearDML",
    "linear_regression": "backdoor.linear_regression",
    "propensity_score_weighting": "backdoor.propensity_score_weighting",
}


def _resolve_dowhy_method(estimation_result: Dict[str, Any]) -> str:
    """Resolve the DoWhy method_name to refute against.

    Prefers the energy-score-selected estimator (selected_estimator field
    when present, from the EnergyScoreSelector) and falls back to the
    legacy ``method`` label. Unknown labels raise RefutationError instead
    of silently defaulting to linear_regression (which would refute a
    DIFFERENT estimate than the one reported — codex iter-2 H3).
    """
    candidate = estimation_result.get("selected_estimator") or estimation_result.get("method")
    if not candidate:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "EstimationResult is missing both 'selected_estimator' and 'method' "
            "fields; cannot determine which DoWhy estimator to refute.",
            details={"reason": "missing_estimator_label"},
        )
    method = _SELECTOR_TO_DOWHY_METHOD.get(candidate)
    if method is None:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"Unknown estimator label '{candidate}' has no DoWhy method mapping. "
            "Refusing to silently default to backdoor.linear_regression which "
            "would refute a different estimate than the one reported.",
            details={
                "reason": "unmapped_estimator_label",
                "estimator_label": candidate,
                "known_labels": sorted(_SELECTOR_TO_DOWHY_METHOD.keys()),
            },
        )
    return method


# Iter-6 codex H-iter5-1: tolerance for reconstructed-vs-reported ATE.
# DoWhy will re-fit the model with default EconML parameters, which may
# differ from the energy-score wrapper's tuned hyperparameters. Without
# a tolerance check we'd be refuting a DIFFERENT estimate than the one
# reported to chat. The tolerance is relative (20% of |reported_ate|) +
# additive (0.1) — the additive floor avoids spurious failures on
# near-zero ATEs where relative tolerance is unstable.
_DOWHY_RECONSTRUCTION_REL_TOL = 0.20
_DOWHY_RECONSTRUCTION_ABS_TOL = 0.10


def _reconstruct_dowhy_artifacts(
    *,
    data: Any,
    treatment: str,
    outcome: str,
    common_causes: List[str],
    estimation_result: Dict[str, Any],
) -> Tuple[Any, Any, Any]:
    """Reconstruct DoWhy CausalModel, identified_estimand, and estimate.

    The agent state does not persist the live DoWhy model from the estimation
    node (would require serialization of fitted EconML wrappers). Instead,
    we re-build the model in-place using the same DAG inputs (treatment,
    outcome, common_causes) and the persisted data passthrough.

    Codex iter-2 H3 (#416): the rebuilt estimate uses the SAME estimator
    method that produced the reported ATE (resolved via
    ``_resolve_dowhy_method``). Earlier iter-1 used a hardcoded
    ``backdoor.linear_regression`` which meant refuters critiqued a
    different estimate than the one reported to chat — silent-wrong.

    This is NOT a mock — it constructs a real DoWhy CausalModel which runs
    real refutation methods (placebo_treatment_refuter, random_common_cause,
    data_subset_refuter, bootstrap_refuter) against real EconML estimators
    matching the reported method.

    Args:
        data: pandas DataFrame with treatment, outcome, and common-cause columns
        treatment: treatment variable name
        outcome: outcome variable name
        common_causes: list of confounder column names
        estimation_result: EstimationResult dict carrying ``selected_estimator``
            (preferred) or ``method`` (fallback) — used to resolve which
            DoWhy estimator backend to instantiate.

    Returns:
        Tuple of (causal_model, identified_estimand, estimate)

    Raises:
        RefutationError: when DoWhy is unavailable OR reconstruction fails.
            Surfaces to chat as "Refutation analysis unavailable for this
            query, retry without refutation".
    """
    if not DOWHY_AVAILABLE:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "DoWhy library is not installed in this environment.",
            details={
                "reason": "dowhy_not_available",
                "treatment": treatment,
                "outcome": outcome,
            },
        )

    if data is None:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "Estimation data passthrough is missing — cannot reconstruct CausalModel.",
            details={
                "reason": "estimation_data_missing",
                "treatment": treatment,
                "outcome": outcome,
            },
        )

    if not hasattr(data, "columns"):
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "Estimation data is not a DataFrame.",
            details={
                "reason": "estimation_data_not_dataframe",
                "data_type": type(data).__name__,
            },
        )

    # Validate columns
    columns = set(data.columns)
    missing_cols: List[str] = []
    if treatment not in columns:
        missing_cols.append(treatment)
    if outcome not in columns:
        missing_cols.append(outcome)
    for cc in common_causes:
        if cc not in columns:
            missing_cols.append(cc)
    if missing_cols:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"Estimation data is missing required columns: {missing_cols}.",
            details={
                "reason": "missing_columns",
                "missing": missing_cols,
                "available_columns": list(data.columns),
                "treatment": treatment,
                "outcome": outcome,
                "common_causes": common_causes,
            },
        )

    try:
        from dowhy import CausalModel  # type: ignore[import-not-found]
    except ImportError as ie:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "DoWhy import failed at reconstruction time.",
            details={"reason": "dowhy_import_failed"},
            original_error=ie,
        ) from ie

    # Resolve the DoWhy method matching the reported estimator (codex H3).
    dowhy_method = _resolve_dowhy_method(estimation_result)

    try:
        # Forest-based CATE estimators (CausalForestDML, DRLearner) REQUIRE
        # effect modifiers X — econml raises "does not support X=None" without
        # them. The estimation path fits these with X=W=features
        # (segment_cate.py: model.fit(outcome, treatment, X=X_clean, W=X_clean)),
        # so mirror that here by reusing the confounders as effect modifiers.
        # When there are no confounders we pass None and let the fail-closed
        # wrapper surface the genuine "needs X" error rather than fabricating.
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes,
            effect_modifiers=common_causes if common_causes else None,
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # Build the estimate using the SAME method that produced the reported
        # ATE (resolved above). Refuters now critique the actual reported
        # estimate, not a separately-fitted linear regression.
        # DoWhy 0.14 + EconML 0.16: for a string econml method_name, DoWhy's
        # EconML wrapper does ``estimator_class(**kwargs["init_params"])`` with a
        # *direct* key access (dowhy/causal_estimators/econml.py), so omitting
        # method_params raises ``KeyError: 'init_params'`` and the whole
        # reconstruction fails (this regression was latent because test_agents
        # never ran in CI — #583).
        #
        # Mirror the two production-estimator init params that materially affect
        # the reconstructed ATE (src/causal_engine/energy_score/estimator_selector.py:
        # ``discrete_treatment=is_binary`` with ``is_binary = len(unique(treatment)) == 2``,
        # and ``random_state=rs`` defaulting to 42). Without discrete_treatment a
        # binary 0/1 treatment is modeled as CONTINUOUS, systematically
        # under-estimating the ATE and tripping the reconstructed-vs-reported
        # tolerance check below; without a fixed random_state the forest ATE
        # varies run-to-run, making that check flaky near its boundary. (#583
        # follow-up: caught by slow-tests on test_repository_failure_handled,
        # ATE 0.3968 vs reported 0.5000 > 0.1 tol.) These mirror the estimator
        # that produced the reported ATE, so refuters critique the SAME model.
        treatment_is_binary = data[treatment].nunique() == 2
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=dowhy_method,
            method_params={
                "init_params": {
                    "discrete_treatment": treatment_is_binary,
                    "random_state": 42,
                },
                "fit_params": {},
            },
            test_significance=False,
        )
    except Exception as exc:  # noqa: BLE001 — fail-closed wrapper
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"DoWhy CausalModel reconstruction failed for method={dowhy_method!r}: {exc}",
            details={
                "reason": "dowhy_reconstruction_failed",
                "dowhy_method": dowhy_method,
                "treatment": treatment,
                "outcome": outcome,
                "common_causes": common_causes,
            },
            original_error=exc,
        ) from exc

    # Iter-6 codex H-iter5-1 (#416): verify the reconstructed estimate's ATE
    # matches the reported ATE within tolerance. DoWhy re-fits with default
    # EconML hyperparameters which may not match the energy-score wrapper's
    # tuned settings; if the reconstructed estimate diverges too far from
    # the reported one, refuters would silently critique a DIFFERENT model
    # than the one whose ATE is on screen.
    reported_ate_raw = estimation_result.get("ate")
    if reported_ate_raw is None:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            "EstimationResult is missing 'ate'; cannot verify reconstructed estimate "
            "matches the reported one.",
            details={"reason": "missing_reported_ate"},
        )
    try:
        reported_ate = float(reported_ate_raw)
        reconstructed_ate = float(getattr(estimate, "value", None) or estimate.value)
    except (TypeError, ValueError, AttributeError) as ate_exc:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"Reported or reconstructed ATE is non-numeric: "
            f"reported={reported_ate_raw!r}.",
            details={
                "reason": "ate_mismatch_non_numeric",
                "reported_ate": repr(reported_ate_raw),
            },
            original_error=ate_exc,
        ) from ate_exc

    tolerance = max(
        abs(reported_ate) * _DOWHY_RECONSTRUCTION_REL_TOL,
        _DOWHY_RECONSTRUCTION_ABS_TOL,
    )
    if abs(reconstructed_ate - reported_ate) > tolerance:
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"Reconstructed DoWhy estimate (ATE={reconstructed_ate:.4f}) diverges from "
            f"reported ATE ({reported_ate:.4f}) beyond tolerance ({tolerance:.4f}). "
            "Refuting a different estimate than the one on screen would be silent-wrong.",
            details={
                "reason": "reconstructed_ate_mismatch",
                "reported_ate": reported_ate,
                "reconstructed_ate": reconstructed_ate,
                "tolerance": tolerance,
                "dowhy_method": dowhy_method,
                "rel_tolerance": _DOWHY_RECONSTRUCTION_REL_TOL,
                "abs_tolerance": _DOWHY_RECONSTRUCTION_ABS_TOL,
            },
        )

    return model, identified_estimand, estimate


class RefutationNode:
    """Runs refutation tests on causal estimates.

    Performance target: <15s
    Type: Standard (computation-heavy)

    This node integrates with the Causal Validation Protocol by:
    1. Using RefutationRunner for 5 standard refutation tests
    2. Applying gate decision logic (proceed/review/block)
    3. Persisting results to causal_validations table
    4. Providing legacy format for backward compatibility

    Attributes:
        runner: RefutationRunner instance for test execution
        validation_repo: Repository for persisting validation results
        config: Custom configuration for tests (optional)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        validation_repo: Optional[CausalValidationRepository] = None,
    ):
        """Initialize refutation node.

        Args:
            config: Custom test configuration (merged with defaults)
            thresholds: Custom pass/fail thresholds
            validation_repo: Repository for database persistence (optional)
        """
        self.runner = RefutationRunner(config=config, thresholds=thresholds)
        self.validation_repo = validation_repo
        logger.info(f"RefutationNode initialized (DoWhy available: {DOWHY_AVAILABLE})")

    async def execute(self, state: CausalImpactState) -> Dict:
        """Run refutation tests.

        Args:
            state: Current workflow state with estimation_result

        Returns:
            Updated state with refutation_results and gate_decision
        """
        start_time = time.time()

        try:
            # Get estimation result
            estimation_result = state.get("estimation_result")
            if not estimation_result:
                raise ValueError("Estimation result not found in state")

            # Iter-3 codex H1 (#416): the previous default of
            # ``original_ate +/- 0.1`` when ate_ci_lower / ate_ci_upper were
            # missing fabricated uncertainty that fed directly into data_subset
            # and bootstrap pass/review/block scoring — same silent-evidence
            # class as iter-1 H4. Now we require both bounds present, numeric,
            # finite, and ordered. Anything else fail-closes via
            # ``RefutationError`` (caught below and surfaced to chat).
            original_ate = estimation_result["ate"]
            ate_ci_lower_raw = estimation_result.get("ate_ci_lower")
            ate_ci_upper_raw = estimation_result.get("ate_ci_upper")
            if ate_ci_lower_raw is None or ate_ci_upper_raw is None:
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    "EstimationResult is missing ate_ci_lower / ate_ci_upper; refusing to "
                    "fabricate a +/- 0.1 default CI which would feed silent-wrong evidence "
                    "into data_subset and bootstrap scoring.",
                    details={
                        "reason": "missing_ci_bounds",
                        "has_ate_ci_lower": ate_ci_lower_raw is not None,
                        "has_ate_ci_upper": ate_ci_upper_raw is not None,
                        "ate": original_ate,
                    },
                )
            try:
                ate_ci_lower = float(ate_ci_lower_raw)
                ate_ci_upper = float(ate_ci_upper_raw)
            except (TypeError, ValueError) as ci_exc:
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    "EstimationResult CI bounds are non-numeric: "
                    f"ate_ci_lower={ate_ci_lower_raw!r}, ate_ci_upper={ate_ci_upper_raw!r}.",
                    details={
                        "reason": "non_numeric_ci_bounds",
                        "ate_ci_lower": repr(ate_ci_lower_raw),
                        "ate_ci_upper": repr(ate_ci_upper_raw),
                    },
                    original_error=ci_exc,
                ) from ci_exc
            if not (math.isfinite(ate_ci_lower) and math.isfinite(ate_ci_upper)):
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    f"EstimationResult CI bounds are non-finite: "
                    f"ate_ci_lower={ate_ci_lower}, ate_ci_upper={ate_ci_upper}.",
                    details={
                        "reason": "non_finite_ci_bounds",
                        "ate_ci_lower": ate_ci_lower,
                        "ate_ci_upper": ate_ci_upper,
                    },
                )
            if ate_ci_lower >= ate_ci_upper:
                # Iter-5 codex H-iter4-2 (#416): reject zero-width AND
                # out-of-order CIs. A successful estimator emitting
                # (0.0, 0.0) or (0.1, 0.1) passes the previous strict-greater
                # check but is functionally a degenerate (no-uncertainty) CI;
                # downstream data_subset / bootstrap scoring treats this as
                # "always covered" which is silent-wrong evidence.
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    f"EstimationResult CI bounds are degenerate or out of order: "
                    f"ate_ci_lower={ate_ci_lower} >= ate_ci_upper={ate_ci_upper}.",
                    details={
                        "reason": "ci_bounds_degenerate_or_out_of_order",
                        "ate_ci_lower": ate_ci_lower,
                        "ate_ci_upper": ate_ci_upper,
                    },
                )
            # Iter-5 codex H-iter4-2 (#416): refuse to refute an ATE that
            # lies outside its own reported CI — internally inconsistent.
            try:
                _ate_float = float(original_ate)
            except (TypeError, ValueError) as ate_exc:
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    f"EstimationResult.ate is non-numeric: {original_ate!r}.",
                    details={"reason": "non_numeric_ate", "ate_raw": repr(original_ate)},
                    original_error=ate_exc,
                ) from ate_exc
            if not (ate_ci_lower <= _ate_float <= ate_ci_upper):
                raise RefutationError(
                    "Refutation analysis unavailable for this query, retry without refutation. "
                    f"EstimationResult ATE={_ate_float} is outside its own CI "
                    f"[{ate_ci_lower}, {ate_ci_upper}].",
                    details={
                        "reason": "ate_outside_ci",
                        "ate": _ate_float,
                        "ate_ci_lower": ate_ci_lower,
                        "ate_ci_upper": ate_ci_upper,
                    },
                )
            original_ci = (ate_ci_lower, ate_ci_upper)

            # Get context for logging
            treatment = state.get("treatment_var", "unknown_treatment")
            outcome = state.get("outcome_var", "unknown_outcome")
            brand = state.get("brand")
            query_id = state.get("query_id", "")

            logger.info(
                f"Running refutation suite for {treatment} → {outcome} "
                f"(ATE={original_ate:.4f}, CI=[{ate_ci_lower:.4f}, {ate_ci_upper:.4f}])"
            )

            # Get data passthrough from estimation node
            # This enables proper refutation tests on actual data
            estimation_data = state.get("estimation_data")
            if estimation_data is not None and hasattr(estimation_data, "shape"):
                logger.debug(
                    f"Using estimation data for refutation (shape: {estimation_data.shape})"  # type: ignore[union-attr]
                )

            # F-014 fix (#416): reconstruct CausalModel from estimation_data
            # so refutation runs REAL DoWhy refuters (placebo, random_common_cause,
            # data_subset, bootstrap) — NOT the deleted ``_mock_*`` paths.
            # Iter-2 (codex H3): rebuild uses the SAME estimator (resolved
            # from estimation_result.selected_estimator / .method) that
            # produced the reported ATE — not a hardcoded linear regression.
            # Fail-closed: ``RefutationError`` propagates to caller's except block.
            common_causes = cast(
                List[str],
                state.get("confounders") or estimation_result.get("covariates_adjusted") or [],
            )
            causal_model, identified_estimand, estimate = _reconstruct_dowhy_artifacts(
                data=estimation_data,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
                estimation_result=cast(Dict[str, Any], estimation_result),
            )

            # Run all refutation tests
            suite: RefutationSuite = self.runner.run_all_tests(
                original_effect=original_ate,
                original_ci=original_ci,
                treatment=treatment,
                outcome=outcome,
                brand=brand,
                estimate_id=query_id,
                # Data passthrough from estimation node (enables DoWhy-based refutation)
                data=estimation_data,
                causal_model=causal_model,
                identified_estimand=identified_estimand,
                estimate=estimate,
            )

            # Convert to legacy format for backward compatibility
            refutation_results = cast(RefutationResults, suite.to_legacy_format())

            # Persist validation results to database
            validation_ids = []
            if self.validation_repo and query_id:
                try:
                    validation_ids = await self.validation_repo.save_suite(
                        suite=suite,
                        estimate_id=query_id,
                        estimate_source="causal_paths",
                        agent_activity_id=cast(Optional[str], state.get("agent_activity_id")),
                        data_split=cast(Optional[str], state.get("data_split")),
                    )
                    logger.info(
                        f"Persisted {len(validation_ids)} validation records for estimate {query_id}"
                    )
                except Exception as persist_error:
                    logger.warning(f"Failed to persist validation results: {persist_error}")

            # Phase 4: Log ValidationOutcome for Feedback Learner integration
            validation_outcome_id = None
            try:
                validation_outcome = create_validation_outcome(
                    suite=suite,
                    agent_context={
                        "agent": "causal_impact",
                        "node": "refutation",
                        "query_id": query_id,
                        "agent_activity_id": state.get("agent_activity_id"),
                    },
                    dag_hash=cast(Optional[str], state.get("dag_hash")),
                    sample_size=cast(Optional[int], estimation_result.get("n_samples")),
                )
                validation_outcome_id = await log_validation_outcome(validation_outcome)
                logger.info(
                    f"Logged validation outcome {validation_outcome_id} for Feedback Learner: "
                    f"{validation_outcome.outcome_type.value}"
                )
            except Exception as learner_error:
                logger.warning(
                    f"Failed to log validation outcome for Feedback Learner: {learner_error}"
                )

            latency_ms = (time.time() - start_time) * 1000

            # Determine next phase based on gate decision
            if suite.gate_decision == GateDecision.BLOCK:
                logger.warning(
                    f"Refutation BLOCKED estimate: confidence={suite.confidence_score:.2f}, "
                    f"tests_passed={suite.tests_passed}/{suite.total_tests}"
                )
                next_phase = "failed"
                status = "failed"
                error_message = self._format_block_reason(suite)
            elif suite.gate_decision == GateDecision.REVIEW:
                logger.info(
                    f"Refutation requires REVIEW: confidence={suite.confidence_score:.2f}, "
                    f"tests_passed={suite.tests_passed}/{suite.total_tests}"
                )
                next_phase = "analyzing_sensitivity"
                status = state.get("status", "in_progress")
                error_message = None
            else:
                logger.info(
                    f"Refutation PASSED: confidence={suite.confidence_score:.2f}, "
                    f"tests_passed={suite.tests_passed}/{suite.total_tests}"
                )
                next_phase = "analyzing_sensitivity"
                status = state.get("status", "in_progress")
                error_message = None

            result = {
                **state,
                "refutation_results": refutation_results,
                "refutation_latency_ms": latency_ms,
                "current_phase": next_phase,
                # Extended fields for validation protocol
                "refutation_suite": suite.to_dict(),
                "gate_decision": suite.gate_decision.value,
                "refutation_confidence": suite.confidence_score,
                # Persistence tracking
                "validation_ids": validation_ids,
                # Phase 4: Feedback Learner tracking
                "validation_outcome_id": validation_outcome_id,
            }

            if status == "failed":
                result["status"] = status
                result["error_message"] = error_message

            return result

        except RefutationError as re:
            # F-014 fail-closed: structured error surfaces to chat UI as
            # "Refutation analysis unavailable for this query, retry without refutation"
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Refutation fail-closed (no mock fallback): {re.message}",
                extra={"details": re.details},
            )
            return {
                **state,
                "refutation_error": re.message,
                "refutation_error_details": re.details,
                "refutation_latency_ms": latency_ms,
                "status": "failed",
                "error_message": re.message,
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Refutation failed: {e}", exc_info=True)
            return {
                **state,
                "refutation_error": str(e),
                "refutation_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Refutation failed: {e}",
            }

    def _format_block_reason(self, suite: RefutationSuite) -> str:
        """Format a human-readable reason for blocking the estimate.

        Args:
            suite: RefutationSuite with test results

        Returns:
            Formatted error message
        """
        failed_tests = [t.test_name.value for t in suite.tests if t.status.value == "failed"]

        if failed_tests:
            tests_str = ", ".join(failed_tests)
            return (
                f"Causal estimate blocked by validation protocol. "
                f"Failed tests: {tests_str}. "
                f"Confidence score: {suite.confidence_score:.2f}. "
                f"Requires expert review or alternative estimation method."
            )
        else:
            return (
                f"Causal estimate blocked due to low confidence score "
                f"({suite.confidence_score:.2f} < 0.50 threshold). "
                f"Consider additional data or alternative methods."
            )


# Standalone function for LangGraph integration
async def refute_causal_estimate(
    state: CausalImpactState,
    validation_repo: Optional[CausalValidationRepository] = None,
) -> Dict:
    """Run refutation tests (standalone function).

    Args:
        state: Current workflow state
        validation_repo: Optional repository for persistence

    Returns:
        Updated state with refutation_results
    """
    node = RefutationNode(validation_repo=validation_repo)
    return await node.execute(state)
