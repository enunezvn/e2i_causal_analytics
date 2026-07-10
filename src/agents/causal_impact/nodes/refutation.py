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

import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

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
    ValidationOutcome,
    # Phase 4: ValidationOutcome for Feedback Learner integration
    create_validation_outcome,
    log_validation_outcome_with_status,
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


def _reconstruction_nuisance_init_params(
    dowhy_method: str, *, discrete_treatment: bool
) -> Dict[str, Any]:
    """Nuisance models that make the reconstructed estimate REPRODUCE the reported ATE.

    The reconstructed-vs-reported tolerance guard (``_DOWHY_RECONSTRUCTION_*_TOL``)
    only means anything if the reconstruction fits the SAME estimator the reported
    ATE came from. For ``LinearDML`` that means the SAME nuisance models as
    production's ``LinearDMLWrapper`` (``src/causal_engine/energy_score/
    estimator_selector.py``): RandomForest outcome + treatment models.

    A PRIOR version substituted scaled-LINEAR nuisance here (StandardScaler +
    LinearRegression / LogisticRegressionCV) to dodge an lbfgs-grind that only
    afflicts econml's *default* logistic propensity on mixed-scale covariates. But
    production never used that default — it uses RandomForest — so the linear
    substitution refit a DIFFERENT model. It happened to agree on the 9-covariate
    ``patient_journeys`` frame (ATE 0.1891 vs 0.1895), but on nonlinear data it
    diverges hard: on ``hcp_adoption`` (``peer_influence_score -> adopted``,
    adjusting ``centrality_z``) the RF nuisance gives ATE 0.2033 while the scaled-
    linear substitution gives 0.0248 — a 0.18 gap that tripped the tolerance guard
    and FAIL-CLOSED refutation (the analyst saw "no refutation test results").
    RandomForest is scale-invariant and fast (no lbfgs grind — the reason the
    linear substitution existed does not apply to it), so mirroring production is
    both correct-by-construction AND within the time budget.

    Applied ONLY to ``LinearDML``. Returns ``{}`` (leave econml defaults) for
    every other method:
      * ``CausalForestDML`` — forest nuisance is scale-invariant (no lbfgs grind).
      * ``DRLearner`` — its reconstruction is NOT yet validated against the
        selector's GradientBoosting nuisance, so we do NOT ship an unvalidated
        numeric change here. A DRLearner-winner run that diverges still
        fails-closed cleanly (PR #1028), as it did before. (Follow-up.)
      * plain ``linear_regression`` / IPW — no iterative nuisance to converge.
    """
    if "LinearDML" in dowhy_method:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Mirror production's LinearDMLWrapper nuisance EXACTLY (same class + params)
        # so the reconstructed ATE reproduces the reported one by construction.
        def _rf_regressor() -> Any:
            return RandomForestRegressor(
                n_estimators=50,
                min_samples_leaf=5,
                min_impurity_decrease=1e-7,
                random_state=42,
            )

        return {
            "model_y": _rf_regressor(),
            "model_t": (
                RandomForestClassifier(
                    n_estimators=50,
                    min_samples_leaf=5,
                    min_impurity_decrease=1e-7,
                    random_state=42,
                )
                if discrete_treatment
                else _rf_regressor()
            ),
        }
    return {}


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
        # Mirror production's treatment preprocessing (estimation.py:170-174): a
        # non-integer (continuous) treatment is binarized at its MEDIAN before
        # the estimator is fit, so the reported ATE is the effect of the
        # BINARIZED treatment. Reconstruct on the same transform so refuters
        # critique the SAME estimand. Without this, a continuous treatment would
        # be reconstructed as continuous — a different model whose ATE could
        # coincidentally land within tolerance, refuting a different estimate
        # than the one on screen (silent-wrong). #583 follow-up (codex HIGH).
        # Use the SAME NumPy ops as production (estimation.py:170-172) so the
        # integer-vs-continuous decision and the median split are byte-identical
        # (incl. how a NaN treatment fails — there it raises in estimation
        # before any estimate exists; here it fail-closes via the wrapper).
        treatment_arr = data[treatment].to_numpy()
        if not np.array_equal(treatment_arr, treatment_arr.astype(int)):
            data = data.copy()
            data[treatment] = (treatment_arr > np.median(treatment_arr)).astype(int)

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
        # Mirror the production-estimator init params that materially affect the
        # reconstructed ATE (src/causal_engine/energy_score/estimator_selector.py:
        # ``discrete_treatment=is_binary`` and ``random_state=rs`` defaulting to
        # 42). Without discrete_treatment a binary 0/1 treatment is modeled as
        # CONTINUOUS, systematically under-estimating the ATE and tripping the
        # reconstructed-vs-reported tolerance check below; without a fixed
        # random_state the forest ATE varies run-to-run, making that check flaky
        # near its boundary. (#583 follow-up: caught by slow-tests on
        # test_repository_failure_handled, ATE 0.3968 vs 0.5000 > 0.1 tol; with
        # these params the reconstruction lands at 0.4219, within tolerance and
        # deterministic.) These mirror the estimator that produced the reported
        # ATE, so refuters critique the SAME model.
        #
        # init_params are estimator-specific: DRLearner is inherently a
        # discrete-treatment learner and its econml-0.16 __init__ does NOT accept
        # a ``discrete_treatment`` kwarg (passing it raises TypeError), so we add
        # ``discrete_treatment`` only for CausalForestDML / LinearDML. All three
        # accept ``random_state``. (Continuous treatments that production binarizes
        # internally will reconstruct here as non-binary; the ATE-tolerance guard
        # below then fails closed rather than refuting a mis-specified model — a
        # fail-safe, not silent-wrong.)
        _raw_seed = estimation_result.get("random_state")
        _discrete = bool(data[treatment].nunique() == 2)
        init_params: Dict[str, Any] = {
            "random_state": 42 if _raw_seed is None else int(_raw_seed),
        }
        if "DRLearner" not in dowhy_method:
            init_params["discrete_treatment"] = _discrete
        # Reconstruct with the SAME nuisance models production used (RandomForest
        # for LinearDML) so the reconstructed ATE reproduces the reported one and
        # the tolerance guard validates the ACTUAL estimate — not a differently-
        # fit model. RandomForest is scale-invariant + fast, so refutation still
        # re-fits ~45x within the time budget (see helper docstring for the
        # scaled-linear substitution this replaced and why it diverged).
        init_params.update(
            _reconstruction_nuisance_init_params(dowhy_method, discrete_treatment=_discrete)
        )
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=dowhy_method,
            method_params={"init_params": init_params, "fit_params": {}},
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

    Performance target (#622, reconciled to MEASURED reality):
      The node runs the REAL DoWhy refutation suite, which re-fits the SAME
      estimator that produced the reported ATE many times (placebo + bootstrap
      + random_common_cause + data_subset). Latency is therefore dominated by
      that estimator's per-re-estimation cost, NOT a single flat budget. The
      old "<15s" comment was aspirational and never met — the suite was ~610
      DoWhy re-estimations (~33s on OLS to ~35-60 min on CausalForestDML).
      With the lowered ``RefutationRunner.DEFAULT_CONFIG`` sim counts (#622:
      placebo 30, random_common_cause 20, bootstrap 50, data_subset 5) and the
      energy-score fast-estimator tiebreak, the realistic SLA is:
        * linear estimators (ols / linear regression): < ~15s
        * meta-learners (S/T/X-learner, DRLearner): < ~60s
        * forest/ensemble DML (causal_forest, ortho_forest) when GENUINELY
          selected (lowest energy, not a tie): a few minutes — bounded, not
          unbounded. The tiebreak avoids picking these on a tie.
      Callers that need sub-second feedback (smoke tests) pass an even smaller
      bounded ``parameters.refutation_config`` (#606), which is merged on top
      of these defaults.
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
        expert_review_gate: Optional[Any] = None,
    ):
        """Initialize refutation node.

        Args:
            config: Custom test configuration (merged with defaults)
            thresholds: Custom pass/fail thresholds
            validation_repo: Repository for database persistence (optional)
            expert_review_gate: ExpertReviewGate consulted on a REVIEW-band gate
                (H2). When None, a no-repository gate is constructed lazily and
                bypasses gracefully (development mode); a real repository-backed
                gate creates/looks up the DAG approval in production.
        """
        self.runner = RefutationRunner(config=config, thresholds=thresholds)
        self.validation_repo = validation_repo
        self.expert_review_gate = expert_review_gate
        logger.info(f"RefutationNode initialized (DoWhy available: {DOWHY_AVAILABLE})")

    async def _consult_review_gate(
        self,
        state: CausalImpactState,
        suite: "RefutationSuite",
        validation_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """On a REVIEW- or BLOCK-band gate, consult the ExpertReviewGate.

        Routes both borderline-robust (REVIEW) and failed-robustness (BLOCK)
        estimates to the expert-review queue so a human can adjudicate. Emits a
        band-specific caveat and returns the created/looked-up ``review_id``. The
        consult is best-effort (a missing repository bypasses with a logged
        warning; a gate error degrades without breaking the node). ``needs_review``
        is set by the caller from ``suite.needs_review`` (REVIEW only), so a BLOCK
        row is queued without being mislabelled valid-but-needs-review.

        Args:
            validation_ids: causal_validations row ids persisted for THIS
                refutation run — forwarded so the queued review row links to
                its statistical evidence (mig 097).
        """
        from src.causal_engine.expert_review_gate import ExpertReviewGate

        gate = self.expert_review_gate or ExpertReviewGate()
        treatment = state.get("treatment_var", "unknown_treatment")
        outcome = state.get("outcome_var", "unknown_outcome")
        brand = state.get("brand")
        # FIX: read the DAG hash from the key the graph builder actually writes
        # (`dag_version_hash`, graph_builder.py). The old `dag_hash` key was never
        # populated, so every auto-created review row was keyed on "" — which broke
        # the approval round-trip and left the queue effectively empty.
        dag_hash = str(state.get("dag_version_hash") or "")
        requester_id = state.get("query_id") or "causal_impact_agent"
        is_block = suite.gate_decision == GateDecision.BLOCK

        expert_review_decision: Optional[str] = None
        review_id: Optional[str] = None
        active_approval_note = ""
        try:
            review_result = await gate.check_approval(
                dag_hash=dag_hash,
                brand=brand,
                treatment=treatment,
                outcome=outcome,
                requester_id=requester_id,
                # check_approval expects a STRING description (Optional[str]),
                # not a dict — summarise the refutation verdict for the audit row.
                analysis_context=(
                    f"confidence={suite.confidence_score:.2f}, gate={suite.gate_decision.value}"
                ),
                # 097: the graph snapshot makes the queued row renderable in the
                # review UI; the validation ids link its statistical evidence.
                dag_structure=state.get("causal_graph"),
                related_validation_ids=validation_ids,
            )
            expert_review_decision = review_result.decision.value
            review_id = review_result.review_id
            # HITL visibility: when this DAG already holds an ACTIVE approval,
            # say so — reviewer + validity — while making explicit that the
            # approval vouches the DAG STRUCTURE, not this estimate's
            # statistical robustness (structure sign-off never upgrades a
            # borderline/failed estimate to validated).
            if getattr(review_result, "is_approved", False):
                reviewer = getattr(review_result, "reviewer_name", None)
                valid_until = getattr(review_result, "valid_until", None)
                active_approval_note = (
                    " The DAG structure was expert-approved"
                    + (f" by {reviewer}" if reviewer else "")
                    + (f" (valid until {valid_until})" if valid_until else "")
                    + "; that approval covers the DAG structure, not this"
                    " estimate's statistical robustness."
                )
        except Exception as gate_err:  # noqa: BLE001 - gate must never break the node
            logger.warning(
                f"ExpertReviewGate consult failed (degrading to needs_review): {gate_err}"
            )

        if is_block:
            caveat = (
                f"Refutation gate is BLOCK (failed robustness, "
                f"confidence={suite.confidence_score:.2f}). This estimate did not pass "
                f"and has been routed to expert review for adjudication."
            )
        else:
            caveat = (
                f"Refutation gate is REVIEW (borderline robust, "
                f"confidence={suite.confidence_score:.2f}). This estimate needs expert "
                f"review before it is used as a validated result."
            )
        caveat += active_approval_note
        # `needs_review` is intentionally NOT returned here: the caller's result
        # dict sets it from ``suite.needs_review`` (True only for REVIEW), so a
        # BLOCK row is queued without being surfaced as valid-but-needs-review.
        return {
            "expert_review_decision": expert_review_decision,
            "review_caveat": caveat,
            "expert_review_id": review_id,
        }

    async def _log_validation_outcome_signal(
        self, validation_outcome: "ValidationOutcome"
    ) -> Optional[str]:
        """Persist the Feedback-Learner ValidationOutcome, fail-closed on non-durable writes (H11).

        Returns the outcome id ONLY when the write was DURABLE. On a DEGRADED
        (ephemeral in-memory fallback) write we log a WARNING and return None so a
        non-durable signal is never propagated to agent state as a persisted id,
        and we do NOT emit a confident success line.
        """
        try:
            result = await log_validation_outcome_with_status(validation_outcome)
        except Exception as learner_error:  # noqa: BLE001 - persistence must never break the node
            logger.warning(
                f"Failed to log validation outcome for Feedback Learner: {learner_error}"
            )
            return None

        if result.degraded or not result.persisted:
            logger.warning(
                "Validation outcome %s for Feedback Learner was written to a DEGRADED "
                "(non-durable, backend=%s) store — dropping the Feedback-Learner signal; "
                "it will be lost on restart.",
                result.outcome_id,
                result.backend,
                extra={"metric": "refutation_validation_outcome_degraded"},
            )
            return None

        logger.info(
            f"Logged validation outcome {result.outcome_id} for Feedback Learner: "
            f"{validation_outcome.outcome_type.value}"
        )
        return result.outcome_id

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
            # Cooperative compute deadline (orphan-fix): the offloaded refutation
            # suite runs in a worker thread that the API task's asyncio.wait_for
            # CANNOT cancel (Python can't force-kill a thread), so a timed-out run
            # would otherwise keep grinding refits and orphan a CPU core. When a
            # deadline is set we (a) refuse to even start reconstruction if it has
            # already passed, and (b) hand it to run_all_tests so the suite skips
            # refuters that would run past it and fails-closed cleanly.
            deadline = cast(Optional[float], state.get("compute_deadline"))
            if deadline is not None and time.monotonic() >= deadline:
                raise RefutationError(
                    "Compute budget exhausted before refutation could start; failing closed "
                    "rather than orphaning refutation compute past the worker's wall-clock cap.",
                    details={"reason": "time_budget_exceeded_pre_refutation"},
                )
            # Offload the CPU-bound DoWhy model reconstruction + refutation suite
            # (placebo / random_common_cause / data_subset / bootstrap each
            # re-estimate the effect many times) to threads so the gunicorn worker's
            # event loop stays responsive. A blocking multi-minute suite trips
            # gunicorn's --timeout and the worker is KILLED mid-run, orphaning the
            # async job. These calls are pure compute (no async clients — the
            # supabase client is used elsewhere on the main loop), so threading is
            # loop-safe.
            # Time the reconstruction — it fits the SAME estimator once, so its
            # wall-time is a good a-priori per-refit cost. We hand it to
            # run_all_tests as ``per_refit_hint`` so even the FIRST refuter is
            # gated against the deadline (otherwise a single slow first refuter
            # could run unconditionally and orphan past the hard cap).
            _recon_t0 = time.monotonic()
            causal_model, identified_estimand, estimate = await asyncio.to_thread(
                _reconstruct_dowhy_artifacts,
                data=estimation_data,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
                estimation_result=cast(Dict[str, Any], estimation_result),
            )
            per_refit_hint = time.monotonic() - _recon_t0

            # Run all refutation tests
            suite: RefutationSuite = await asyncio.to_thread(
                self.runner.run_all_tests,
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
                deadline=deadline,
                per_refit_hint=per_refit_hint,
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
            validation_outcome = create_validation_outcome(
                suite=suite,
                agent_context={
                    "agent": "causal_impact",
                    "node": "refutation",
                    "query_id": query_id,
                    "agent_activity_id": state.get("agent_activity_id"),
                },
                dag_hash=cast(Optional[str], state.get("dag_version_hash")),
                sample_size=cast(Optional[int], estimation_result.get("n_samples")),
            )
            validation_outcome_id = await self._log_validation_outcome_signal(validation_outcome)

            latency_ms = (time.time() - start_time) * 1000

            # Determine next phase based on gate decision
            review_fields: Dict[str, Any] = {}
            if suite.gate_decision == GateDecision.BLOCK:
                logger.warning(
                    f"Refutation BLOCKED estimate: confidence={suite.confidence_score:.2f}, "
                    f"tests_passed={suite.tests_passed}/{suite.total_tests}"
                )
                next_phase = "failed"
                status = "failed"
                error_message = self._format_block_reason(suite)
                # Route BLOCKED (failed-robustness) estimates to the expert-review
                # queue too, so a human can adjudicate or override the failure. The
                # estimate still surfaces as failed (needs_review stays False from
                # suite.needs_review); the queued row carries the gate=block context.
                review_fields = await self._consult_review_gate(
                    state, suite, validation_ids=validation_ids
                )
            elif suite.gate_decision == GateDecision.REVIEW:
                logger.info(
                    f"Refutation requires REVIEW: confidence={suite.confidence_score:.2f}, "
                    f"tests_passed={suite.tests_passed}/{suite.total_tests}"
                )
                next_phase = "analyzing_sensitivity"
                status = state.get("status", "in_progress")
                error_message = None
                # H2: consult the ExpertReviewGate and flag needs_review + caveat
                # so a REVIEW band is NOT surfaced/persisted as robust/validated.
                review_fields = await self._consult_review_gate(
                    state, suite, validation_ids=validation_ids
                )
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
                # H2: distinct REVIEW signal (default False for PROCEED/BLOCK)
                "needs_review": suite.needs_review,
                # Persistence tracking
                "validation_ids": validation_ids,
                # Phase 4: Feedback Learner tracking
                "validation_outcome_id": validation_outcome_id,
                # REVIEW-band caveat / expert-review decision (empty otherwise)
                **review_fields,
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


async def _build_expert_review_gate() -> Optional[Any]:
    """Build a repo-backed ExpertReviewGate (auto_create_review=True), or None.

    R6-F2 C2: the REVIEW band on the agent path creates a ``pending``
    ``expert_reviews`` row keyed by ``dag_version_hash`` so a human can resolve
    it (the consumer endpoints + admin UI were built first, Phases A/B). R5
    flipped ``auto_create_review`` to default False (fail-closed, no orphan rows
    before a consumer existed); we re-enable it EXPLICITLY here now that the
    consumer is in place.

    Best-effort / graceful-degrade — but ONLY for the missing-config signal: in
    dev/test (or any environment without a Supabase service-role key)
    ``get_async_supabase_client`` raises ``ServiceConnectionError`` — we catch
    THAT specific class and return None so ``_consult_review_gate`` falls back to
    a bare ``ExpertReviewGate()`` (bypass to PROCEED-with-warning) and still flags
    ``needs_review`` via the H2 caveat. A missing Supabase must NEVER crash the
    node.

    FIX A (codex HIGH): any OTHER exception (a transient/unexpected prod Supabase
    failure, a bug) PROPAGATES (fail-loud / fail-closed). Collapsing every error
    into the same ``return None`` -> bare-gate bypass would silently self-bypass
    the review gate in prod on an unexpected error and surface a REVIEW-band
    estimate as approved. A real error must surface, not proceed as approved.
    """
    from src.memory.services.factories import (
        ServiceConnectionError,
        get_async_supabase_client,
    )

    try:
        client = await get_async_supabase_client()
    except ServiceConnectionError as exc:
        # ONLY the missing-config / connection-unavailable signal degrades.
        logger.warning("Expert review gate unavailable (degrading to needs_review only): %s", exc)
        return None

    from src.causal_engine.expert_review_gate import ExpertReviewGate
    from src.repositories.expert_review import ExpertReviewRepository

    return ExpertReviewGate(
        repository=ExpertReviewRepository(supabase_client=client),
        auto_create_review=True,  # R5 default is False; re-enable here explicitly
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
    # Callers may tune refutation rigor via ``parameters.refutation_config``
    # (merged onto RefutationRunner.DEFAULT_CONFIG, per-key). This lets a smoke
    # harness run the REAL refutation suite with fewer simulations — the full
    # suite is ~610 dowhy re-estimations (placebo 100 + bootstrap 500 + subset
    # 10), i.e. ~10-60 min depending on estimator, which no per-agent CI budget
    # can hold. Omitted in prod -> None -> full DEFAULT_CONFIG (unchanged). (#606)
    refutation_config = (state.get("parameters") or {}).get("refutation_config")
    # R6-F2 C2: wire a repo-backed ExpertReviewGate so a REVIEW band creates a
    # `pending` expert_reviews row a human can resolve. Best-effort: None in
    # dev/test (no Supabase) -> the node bypasses to PROCEED-with-warning and
    # still flags needs_review. REVIEW never hard-blocks the agent run.
    expert_review_gate = await _build_expert_review_gate()
    node = RefutationNode(
        config=refutation_config,
        validation_repo=validation_repo,
        expert_review_gate=expert_review_gate,
    )
    return await node.execute(state)
