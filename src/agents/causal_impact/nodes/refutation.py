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


def _reconstruct_dowhy_artifacts(
    *,
    data: Any,
    treatment: str,
    outcome: str,
    common_causes: List[str],
) -> Tuple[Any, Any, Any]:
    """Reconstruct DoWhy CausalModel, identified_estimand, and estimate.

    The agent state does not persist the live DoWhy model from the estimation
    node (would require serialization of fitted EconML wrappers). Instead,
    we re-build the model in-place using the same DAG inputs (treatment,
    outcome, common_causes) and the persisted data passthrough.

    This is NOT a mock — it constructs a real DoWhy CausalModel which runs
    real refutation methods (placebo_treatment_refuter, random_common_cause,
    data_subset_refuter, bootstrap_refuter) against real EconML estimators.

    Args:
        data: pandas DataFrame with treatment, outcome, and common-cause columns
        treatment: treatment variable name
        outcome: outcome variable name
        common_causes: list of confounder column names

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

    try:
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes,
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # Use backdoor.linear_regression as the lightweight estimator for the
        # refutation run. Real estimation (with energy-score-selected estimator)
        # has already happened in EstimationNode; this rebuild only provides
        # the DoWhy object graph for refutation method dispatch.
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=False,
        )
    except Exception as exc:  # noqa: BLE001 — fail-closed wrapper
        raise RefutationError(
            "Refutation analysis unavailable for this query, retry without refutation. "
            f"DoWhy CausalModel reconstruction failed: {exc}",
            details={
                "reason": "dowhy_reconstruction_failed",
                "treatment": treatment,
                "outcome": outcome,
                "common_causes": common_causes,
            },
            original_error=exc,
        ) from exc

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

            original_ate = estimation_result["ate"]
            ate_ci_lower = estimation_result.get("ate_ci_lower", original_ate - 0.1)
            ate_ci_upper = estimation_result.get("ate_ci_upper", original_ate + 0.1)
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
