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
from src.repositories.causal_validation import (
    CAUSAL_QUERY_ESTIMATE_SOURCE,
    CausalValidationRepository,
    derive_causal_path_estimate_id,
    derive_query_estimate_id,
)

logger = logging.getLogger(__name__)

# Gate → validation_status transition map (#1352 item 3; demotion mechanics are
# this lane's documented call — see _persist_suite_and_promote):
#   PROCEED  : pending/needs_review → validated  (the sole promotion)
#   REVIEW   : pending → needs_review  (a borderline re-run never DOWNGRADES an
#              already-validated path — earlier passed evidence stands until a
#              BLOCK verdict actually contradicts it)
#   BLOCK    : pending/needs_review/validated → refuted  (a real failed suite
#              demotes even a validated path; the contradicting evidence rows
#              are persisted alongside — never deleted, so migration 119's
#              deliberately-unguarded evidence DELETE surface stays untouched)
_GATE_STATUS_TRANSITIONS: Dict[GateDecision, Tuple[str, Tuple[str, ...]]] = {
    GateDecision.PROCEED: ("validated", ("pending", "needs_review")),
    GateDecision.REVIEW: ("needs_review", ("pending",)),
    GateDecision.BLOCK: ("refuted", ("pending", "needs_review", "validated")),
}


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


def _effective_reconstruction_common_causes(
    common_causes: List[str], estimation_result: Dict[str, Any]
) -> List[str]:
    """#1188: the columns the DoWhy reconstruction must condition on so the
    refuters critique the SAME model whose ATE is on screen.

    An efficiency run (randomized design, baselines as variance-reduction
    controls) fits the selected covariate estimator with the baselines as X=W
    while ``confounders``/``covariates_adjusted`` stay honestly empty.
    Rebuilding that estimator with no columns would refute a DIFFERENT
    (unadjusted) model — or fail closed on the ATE-mismatch guard below. So
    for an efficiency run whose selected estimator is covariate-based, thread
    the baselines into the reconstruction. An OLS-selected efficiency run
    reported the UNADJUSTED anchor, so it reconstructs with no columns
    (adding baselines would fit an ANCOVA — a different model, off by the
    chance-imbalance correction). Inside the DoWhy graph the baselines act as
    adjustment columns for MODEL FIDELITY only; adjusting for pre-treatment
    covariates of a randomized treatment stays unbiased, and the response
    labeling (adjustment_type='efficiency', covariates_adjusted=[]) is
    untouched.
    """
    if estimation_result.get("adjustment_type") != "efficiency":
        return common_causes
    baselines = list(estimation_result.get("baseline_covariates_adjusted") or [])
    if not baselines:
        return common_causes
    method = _resolve_dowhy_method(estimation_result)
    if method in ("backdoor.linear_regression", "backdoor.propensity_score_weighting"):
        return common_causes
    return baselines


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

    Applied to ``LinearDML`` and ``DRLearner`` (#1188 codex iter-1 MED — the
    DR wrapper now uses GradientBoosting nuisances + a
    ``StatsModelsLinearRegression`` final stage for honest ATE inference, so
    the rebuild mirrors those EXACT models; econml's DR defaults would refit a
    different surface and trip the tolerance guard). Returns ``{}`` (leave
    econml defaults) for every other method:
      * ``CausalForestDML`` — forest nuisance is scale-invariant (no lbfgs grind).
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
    if "DRLearner" in dowhy_method:
        from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        # Mirror production's DRLearnerWrapper models EXACTLY
        # (src/causal_engine/energy_score/estimator_selector.py).
        return {
            "model_regression": GradientBoostingRegressor(n_estimators=50, random_state=42),
            "model_propensity": GradientBoostingClassifier(n_estimators=50, random_state=42),
            "model_final": StatsModelsLinearRegression(),
        }
    return {}


def _subsample_for_refutation(
    data: Any, treatment: str, outcome: str
) -> Tuple[Any, Dict[str, Any]]:
    """#1419: deterministic stratified subsample of the estimation passthrough.

    Full-frame refutation can never fit any enforceable budget on the live
    substrate (measured 2026-07-31: ~23 s per refit x ~105 configured sims on
    the 37,371-row conversion frame). Refutation therefore reconstructs and
    refutes on the SAME deterministic stratified subsample family as #1392
    selection — the identical helper (treatment x outcome-bin strata,
    content-derived seed, cap ``SELECTION_MAX_ROWS_DEFAULT``), so there is a
    single subsampling source and zero drift. Measured on the live frame the
    5,000-row subsample preserves the marginals almost exactly (treat share
    0.4849 -> 0.4848, outcome mean 0.1836 -> 0.1836) and drops per-sim cost to
    ~2.1-2.6 s. The reported ATE/CI stays the FULL-frame fit (#1392 contract);
    the reconstructed-vs-reported tolerance guard below covers the
    subsample-vs-full drift (observed live: 0.0404 vs 0.0352).

    A continuous treatment is binarized at the FULL frame's median BEFORE the
    draw (same NumPy ops as estimation.py's preprocessing): subsampling first
    would shift the split to the subsample's median — a different estimand.
    Reconstruction's integer check then passes the 0/1 column through
    unchanged.

    Bad passthroughs (None / non-DataFrame / missing columns) return
    unchanged: reconstruction owns the fail-closed messaging for those.
    Unexpected coercion failures (e.g. a non-numeric treatment the integer
    check chokes on) likewise fall back to the full frame — subsampling is an
    optimization and must never bypass the structured fail-closed paths with a
    raw dtype error; the downstream budget gates own any resulting skip.

    Returns ``(frame, disclosure)`` where disclosure carries
    ``refutation_subsampled`` / ``refutation_n_rows`` /
    ``refutation_n_rows_total`` — stamped onto every suite test's ``details``
    so each persisted evidence row (``details_json``) records
    validated-on-subsample.
    """
    if (
        data is None
        or not hasattr(data, "columns")
        or treatment not in data.columns
        or outcome not in data.columns
    ):
        n = int(data.shape[0]) if hasattr(data, "shape") else 0
        return data, {
            "refutation_subsampled": False,
            "refutation_n_rows": n,
            "refutation_n_rows_total": n,
        }

    from src.causal_engine.energy_score.estimator_selector import (
        SELECTION_MAX_ROWS_DEFAULT,
        _stratified_subsample_indices,
    )

    n_total = int(len(data))
    if n_total <= SELECTION_MAX_ROWS_DEFAULT:
        return data, {
            "refutation_subsampled": False,
            "refutation_n_rows": n_total,
            "refutation_n_rows_total": n_total,
        }

    try:
        frame = data
        treatment_arr = frame[treatment].to_numpy()
        if not np.array_equal(treatment_arr, treatment_arr.astype(int)):
            frame = frame.copy()
            frame[treatment] = (treatment_arr > np.median(treatment_arr)).astype(int)

        indices = _stratified_subsample_indices(
            frame[treatment].to_numpy(),
            frame[outcome].to_numpy(),
            SELECTION_MAX_ROWS_DEFAULT,
        )
    except Exception as exc:  # noqa: BLE001 - unexpected dtypes/content
        logger.warning(
            "Refutation subsample failed (%s: %s) — falling back to the full "
            "%d-row frame; downstream budget gates own any resulting "
            "fail-closed skip (#1419).",
            type(exc).__name__,
            exc,
            n_total,
        )
        return data, {
            "refutation_subsampled": False,
            "refutation_n_rows": n_total,
            "refutation_n_rows_total": n_total,
        }
    return frame.iloc[indices], {
        "refutation_subsampled": True,
        "refutation_n_rows": int(len(indices)),
        "refutation_n_rows_total": n_total,
    }


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

        # #1417: the estimation node fits on a one-hot-encoded design matrix
        # (_encode_categorical_covariates), so the reported ATE conditions on
        # the ENCODED columns. The passthrough frame reaches this node RAW;
        # handing DoWhy raw categorical names makes its EconML wrapper select
        # effect modifiers that no longer exist after DoWhy's own internal
        # dummification — KeyError "['delivery_channel', ...] not in index"
        # (live 2026-07-31). Encode ONLY the adjustment-set columns with the
        # SAME estimation helper (single source, zero transform drift) and
        # condition on the encoded names; encoding the whole passthrough would
        # trip the cardinality guard on identifier columns (hcp_id) that are
        # not confounders. The guard still fires when an identifier IS in the
        # adjustment set — that stays fail-closed by design.
        effective_common_causes = common_causes
        if common_causes:
            from src.agents.causal_impact.nodes.estimation import (
                _encode_categorical_covariates,
            )

            encoded_covariates = _encode_categorical_covariates(data[common_causes])
            if list(encoded_covariates.columns) != list(common_causes):
                data = data[[treatment, outcome]].join(encoded_covariates)
                effective_common_causes = list(encoded_covariates.columns)

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
            common_causes=effective_common_causes,
            effect_modifiers=effective_common_causes if effective_common_causes else None,
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
        causal_path_repo: Optional[Any] = None,
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
            causal_path_repo: ``CausalPathRepository`` (or protocol-compatible)
                used by the #1352 sole-promoter wiring — resolving the run's
                linked ``causal_paths`` row and conditionally moving its
                ``validation_status`` AFTER passed evidence is persisted under
                ``derive_causal_path_estimate_id(path_id)``. ``None`` disables
                promotion (evidence persistence still runs when
                ``validation_repo`` is wired).
        """
        self.runner = RefutationRunner(config=config, thresholds=thresholds)
        self.validation_repo = validation_repo
        self.expert_review_gate = expert_review_gate
        self.causal_path_repo = causal_path_repo
        logger.info(f"RefutationNode initialized (DoWhy available: {DOWHY_AVAILABLE})")

    # ------------------------------------------------------------------ #1352
    async def _resolve_linked_path(self, state: CausalImpactState) -> Optional[Dict[str, Any]]:
        """Resolve the REAL ``causal_paths`` row this run is tied to, or None.

        Linkage sources, in order:

        1. An explicit ``state['causal_path_id']`` (analyst-supplied dispatch
           parameter / API caller) — authoritative when it resolves to a REAL
           row; a synthetic row is REFUSED (a real run cannot validate a DGP
           fiction) and a missing id logs a warning (never silently re-matched).
        2. Auto-match: the UNIQUE real row whose (start_node, end_node[, brand])
           equals this run's (treatment_var, outcome_var[, brand]). Ambiguity
           (≥2 rows) binds nothing — promoting several paths off one run would
           overclaim; the candidates are logged for the operator.
        """
        repo = self.causal_path_repo
        if repo is None:
            return None

        explicit = state.get("causal_path_id")
        if explicit:
            row = await repo.get_path_row(str(explicit))
            if row is None:
                logger.warning(
                    "refutation promoter: explicit causal_path_id %r does not exist; "
                    "run stays unlinked.",
                    explicit,
                )
                return None
            if row.get("is_synthetic"):
                logger.warning(
                    "refutation promoter: explicit causal_path_id %r is a SYNTHETIC "
                    "path; a real run never promotes DGP rows — run stays unlinked.",
                    explicit,
                )
                return None
            return cast(Dict[str, Any], row)

        treatment = state.get("treatment_var")
        outcome = state.get("outcome_var")
        if not treatment or not outcome:
            return None
        rows = await repo.find_real_paths_for_pair(
            treatment=str(treatment),
            outcome=str(outcome),
            brand=cast(Optional[str], state.get("brand")),
        )
        if not rows:
            return None
        if len(rows) > 1:
            logger.warning(
                "refutation promoter: %d real causal_paths rows match (%s -> %s, brand=%s); "
                "ambiguous linkage binds nothing (candidates: %s).",
                len(rows),
                treatment,
                outcome,
                state.get("brand"),
                [r.get("path_id") for r in rows],
            )
            return None
        return cast(Dict[str, Any], rows[0])

    async def _persist_suite_and_promote(
        self, state: CausalImpactState, suite: "RefutationSuite"
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Persist the suite's evidence and — as SOLE promoter — move the linked
        real path's ``validation_status`` (#1352 item 3).

        Contract (mirrors migration 119's enforcement, but never RELIES on the
        trigger being installed — the order is enforced here):

        * linked run  → evidence rows land under
          ``derive_causal_path_estimate_id(path_id)`` with
          ``estimate_source='causal_paths'`` (the id the mig-119 evidence gate
          looks up), and ONLY AFTER a successful persist does the gate-mapped
          status transition fire (see ``_GATE_STATUS_TRANSITIONS``).
        * unlinked run → evidence rows land under the query-derived uuid
          (``derive_query_estimate_id``) with
          ``estimate_source='causal_impact_query'`` — per-run history that can
          never accidentally bless a path.
        * a synthetic-FIXTURE run (``data_source='synthetic'``, the dev/test
          path) persists only unlinked evidence and NEVER promotes: promoting a
          real path off fixture data would be fabricated validation.

        Returns ``(validation_ids, promotion_info)``; ``promotion_info`` is
        ``{}`` unless a status transition actually happened. Best-effort
        throughout: any persistence/promotion failure degrades with a logged
        warning and never breaks the analysis — but a promotion can NEVER
        happen without its evidence persisted first.
        """
        query_id = str(state.get("query_id") or "")
        if not self.validation_repo or not query_id:
            return [], {}

        synthetic_fixture_run = state.get("data_source") == "synthetic"

        linked_row: Optional[Dict[str, Any]] = None
        if not synthetic_fixture_run:
            try:
                linked_row = await self._resolve_linked_path(state)
            except Exception as link_err:  # noqa: BLE001 - linkage is best-effort
                logger.warning(
                    "refutation promoter: path-linkage resolution failed (%s); "
                    "persisting unlinked evidence only.",
                    link_err,
                )
                linked_row = None

        if linked_row is not None:
            path_id = str(linked_row.get("path_id"))
            estimate_id = derive_causal_path_estimate_id(path_id)
            estimate_source = "causal_paths"
        else:
            path_id = ""
            estimate_id = derive_query_estimate_id(query_id)
            estimate_source = CAUSAL_QUERY_ESTIMATE_SOURCE

        validation_ids: List[str] = []
        try:
            validation_ids = await self.validation_repo.save_suite(
                suite=suite,
                estimate_id=estimate_id,
                estimate_source=estimate_source,
                agent_activity_id=cast(Optional[str], state.get("agent_activity_id")),
                data_split=cast(Optional[str], state.get("data_split")),
            )
            logger.info(
                "Persisted %d validation records under estimate %s (%s).",
                len(validation_ids),
                estimate_id,
                estimate_source,
            )
        except Exception as persist_error:  # noqa: BLE001 - never break the analysis
            logger.warning(f"Failed to persist validation results: {persist_error}")

        # Promotion: evidence FIRST (above), status flip second — and only when
        # the evidence rows actually landed (mig-119's gate would reject a
        # 'validated' claim without them; we enforce the same order without
        # depending on the trigger).
        if linked_row is None or self.causal_path_repo is None or not validation_ids:
            return validation_ids, {}

        transition = _GATE_STATUS_TRANSITIONS.get(suite.gate_decision)
        if transition is None:  # pragma: no cover - enum is closed
            return validation_ids, {}
        new_status, allowed_current = transition
        try:
            moved = await self.causal_path_repo.set_validation_status(
                path_id, new_status, allowed_current
            )
        except Exception as promote_err:  # noqa: BLE001 - never break the analysis
            logger.warning(
                "refutation promoter: status transition %s -> %s for path %s failed (%s); "
                "evidence is persisted, status unchanged.",
                allowed_current,
                new_status,
                path_id,
                promote_err,
            )
            return validation_ids, {}
        if not moved:
            logger.info(
                "refutation promoter: path %s not in %s; no transition to %s "
                "(concurrent writer or operator adjudication wins).",
                path_id,
                allowed_current,
                new_status,
            )
            return validation_ids, {}
        logger.info(
            "refutation promoter: causal_paths.%s moved to '%s' (gate=%s) with %d "
            "evidence rows under %s.",
            path_id,
            new_status,
            suite.gate_decision.value,
            len(validation_ids),
            estimate_id,
        )
        return validation_ids, {
            "path_id": path_id,
            "new_status": new_status,
            "gate_decision": suite.gate_decision.value,
            "estimate_id": estimate_id,
        }

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
            # #1188: on an efficiency (RCT baseline) run the reported estimator
            # conditioned on the baselines even though the confounder set is
            # honestly empty — the reconstruction must condition on the SAME
            # columns or the refuters critique a different model.
            common_causes = _effective_reconstruction_common_causes(
                common_causes, cast(Dict[str, Any], estimation_result)
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
            # #1419: reconstruct + refute on the deterministic stratified
            # subsample (same helper as #1392 selection) — full-frame refutation
            # measures ~23 s/refit x ~105 sims on the live substrate and can
            # never fit any enforceable budget. See _subsample_for_refutation.
            refutation_data, data_disclosure = _subsample_for_refutation(
                estimation_data, treatment, outcome
            )
            # The e-value standardizes the FULL-frame reported effect, but the
            # runner's ``data`` below is the subsample — hand it the full
            # frame's outcome SD so the scale-sensitive critical gate is
            # standardized on the same frame as the effect it gates. ``None``
            # lets the runner fall back to its data-derived SD.
            outcome_std_full: Optional[float] = None
            try:
                if (
                    estimation_data is not None
                    and hasattr(estimation_data, "columns")
                    and outcome in estimation_data.columns
                ):
                    outcome_std_full = float(np.std(estimation_data[outcome].to_numpy(dtype=float)))
            except Exception:  # noqa: BLE001 - non-numeric outcome → runner fallback
                outcome_std_full = None
            if data_disclosure["refutation_subsampled"]:
                logger.info(
                    "Refutation subsampled estimation data %s -> %s rows (#1419); "
                    "reported ATE/CI remain the full-frame fit.",
                    data_disclosure["refutation_n_rows_total"],
                    data_disclosure["refutation_n_rows"],
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
                data=refutation_data,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
                estimation_result=cast(Dict[str, Any], estimation_result),
            )
            per_refit_hint = time.monotonic() - _recon_t0
            # #1419: the reconstruction fit is NOT a faithful per-sim cost for
            # the POINT-refit refuters — on the live 5k subsample it measures
            # ~9-11.5 s while a placebo/rcc/subset SIM measures ~1.6-2.6 s
            # (recon includes model build + identification + the inference
            # machinery those sims skip). Gating on the ~5x-inflated recon hint
            # would budget-skip placebo (30 x 11.5 s = 345 s "needed") even
            # though the real work fits. When a deadline is set, calibrate the
            # hint with one throwaway 1-sim placebo refute (~one true sim's
            # cost; result discarded, not evidence). The calibration itself is
            # gated on the conservative recon hint so it cannot orphan past
            # the cap; on any failure the recon hint stands (conservative
            # fallback — never a fabricated cheap hint). The recon wall-time
            # is preserved as ``per_refit_hint_heavy``: a BOOTSTRAP sim
            # re-runs the full inference machinery and measures ~11.7 s ≈ the
            # recon fit, so the runner gates bootstrap on the heavy cost —
            # gating it on the cheap observed per-refit would start a ~5x
            # longer run than estimated and orphan the worker thread.
            per_refit_hint_heavy = per_refit_hint
            if deadline is not None and time.monotonic() + per_refit_hint <= deadline:
                try:
                    _cal_t0 = time.monotonic()
                    await asyncio.to_thread(
                        causal_model.refute_estimate,
                        identified_estimand,
                        estimate,
                        method_name="placebo_treatment_refuter",
                        placebo_type="permute",
                        num_simulations=1,
                    )
                    per_refit_hint = time.monotonic() - _cal_t0
                except Exception as cal_err:  # noqa: BLE001 - keep conservative hint
                    logger.debug(
                        "per-refit calibration failed (%s); keeping the "
                        "conservative reconstruction-time hint.",
                        cal_err,
                    )

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
                data=refutation_data,
                causal_model=causal_model,
                identified_estimand=identified_estimand,
                estimate=estimate,
                deadline=deadline,
                per_refit_hint=per_refit_hint,
                per_refit_hint_heavy=per_refit_hint_heavy,
                outcome_std=outcome_std_full,
                # DESIGN declaration from the API layer (dataset spec): a
                # genuinely randomized treatment reports the E-value as
                # information instead of an unmeasured-confounding BLOCK gate.
                # Fail-closed: absent/False keeps the full observational gate.
                randomized_design=bool(state.get("randomized_design")),
            )

            # #1419: stamp subsample provenance onto every test's details so
            # each persisted evidence row (details_json) records
            # validated-on-subsample — never presented as a full-frame result.
            for _suite_test in suite.tests:
                _suite_test.details.update(data_disclosure)

            # Convert to legacy format for backward compatibility
            refutation_results = cast(RefutationResults, suite.to_legacy_format())

            # Persist validation results + SOLE-promoter path transition
            # (#1352 item 3, extracted to _persist_suite_and_promote): linked
            # runs write path-linked evidence (derive_causal_path_estimate_id)
            # then move the real row's validation_status per the gate; unlinked
            # runs write per-run history under the query-derived uuid (the old
            # ``estimate_id=query_id`` write ALWAYS failed the uuid cast — half
            # of #1352's "causal_validations never populated").
            validation_ids, causal_path_promotion = await self._persist_suite_and_promote(
                state, suite
            )

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
                # #1352 item 3: the sole-promoter transition applied to a
                # linked real causal_paths row ({} when unlinked/no transition)
                "causal_path_promotion": causal_path_promotion,
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
    # #1352 item 3: the production graph adds this function as a bare node, so
    # ``validation_repo`` was ALWAYS None here and evidence persistence never
    # even ran on the agent path (the other half of "causal_validations never
    # populated"). Build the validation + causal-path repositories lazily from
    # the async client; a missing-config environment degrades to None (no
    # persistence, no promotion) exactly like the review gate.
    causal_path_repo = None
    if validation_repo is None:
        validation_repo, causal_path_repo = await _build_persistence_repos()
    node = RefutationNode(
        config=refutation_config,
        validation_repo=validation_repo,
        expert_review_gate=expert_review_gate,
        causal_path_repo=causal_path_repo,
    )
    return await node.execute(state)


async def _build_persistence_repos() -> Tuple[Optional[CausalValidationRepository], Optional[Any]]:
    """Lazily build the validation + causal-path repositories, or (None, None).

    Same degrade contract as ``_build_expert_review_gate``: ONLY the
    missing-config signal (``ServiceConnectionError``) degrades — dev/test
    without a Supabase runs the node with no persistence and no promotion.
    Unlike the review gate (whose silent bypass would surface a REVIEW-band
    estimate as approved), persistence here is already best-effort inside the
    node (every write failure logs a warning and the analysis proceeds), so a
    broader failure while BUILDING the clients is also degraded with a warning
    rather than failing the analysis — evidence loss is observable in logs and
    can never flip a path's status (no evidence ⇒ no promotion, by order).
    """
    from src.memory.services.factories import (
        ServiceConnectionError,
        get_async_supabase_client,
    )

    try:
        client = await get_async_supabase_client()
    except ServiceConnectionError as exc:
        logger.warning("Validation persistence unavailable (no Supabase config): %s", exc)
        return None, None
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort by contract
        logger.warning("Validation persistence unavailable (client build failed): %s", exc)
        return None, None

    from src.repositories.causal_path import CausalPathRepository

    return (
        CausalValidationRepository(supabase_client=client),
        CausalPathRepository(supabase_client=client),
    )
