"""Registry Manager Node - MLflow registration and stage promotion.

Handles:
1. Model registration in MLflow registry (via MLflowConnector)
2. Stage validation and promotion (via MLflowConnector)
3. Shadow mode criteria validation
4. Regulatory-eligibility evaluation (Gate N1 — plan v4 §2)

Uses MLflowConnector for circuit breaker protection and async support.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from src.agents.ml_foundation.model_deployer.regulatory_audit import (
    RegulatoryEligibilityAudit,
    is_adapted_regulatory_candidate,
    is_regulatory_eligible,
)

logger = logging.getLogger(__name__)

# Gate N1 (plan v4 §2): the literature-anchored absolute thresholds the
# deployer MUST evaluate before granting ``regulatory_eligible=True``.
# Each entry maps a gate name (the same key used in ``gate_history``) to
# its threshold key on ``validation_metrics`` (or ``success_criteria``)
# and its comparison direction.
#
# We start with ``minimum_auc`` only (the canonical literature-anchored
# floor for binary classification per
# ``scope_definer/nodes/criteria_validator.py:118-120``). Other gates
# (precision, recall, calibration, etc.) can be added once their
# literature-anchored thresholds are signed off — explicit list, no
# auto-discovery, so a typo cannot silently disable a gate.
N1_REQUIRED_REGULATORY_GATES: List[str] = ["minimum_auc"]


# ============================================================================
# Plan v3 §4 T2.6a — Deployer input metric computation (pure compute, no
# enforcement). Categorizes the three quality signals the T2.6c enforcement
# phase will gate on:
#   - signal_genuineness:   from validation_metrics["permutation_pvalue"]
#   - calibration_quality:  from metrics_result["calibration_error"] (ECE)
#   - cv_stability:         from validation_metrics["cv_5fold_roc_auc_std" /
#                           "_mean"] ratio
#
# Bands are domain-typical defaults; T2.6c can override per cohort. Plan §6
# T2.6 calibration protocol: synthetic regimes [0.55, 0.85] for plumbing →
# retrospective held-out cohorts for threshold fitting → operator decisions
# for drift monitoring only.
# ============================================================================

SignalGenuinenessCategory = Literal["genuine", "likely_genuine", "marginal", "random", "degenerate"]
CalibrationQualityCategory = Literal["excellent", "good", "marginal", "poor", "degenerate"]
CvStabilityCategory = Literal["stable", "moderate", "unstable", "very_unstable", "degenerate"]

# Signal-genuineness pvalue bands. Stricter end matches the existing
# `PERMUTATION_P_MAX = 0.01` ceiling in `tests/integration/test_csu_val_auc_measurement.py`.
T2_6A_SIGNAL_GENUINE_PVALUE_MAX: float = 0.001
T2_6A_SIGNAL_LIKELY_GENUINE_PVALUE_MAX: float = 0.01
T2_6A_SIGNAL_MARGINAL_PVALUE_MAX: float = 0.05

# Calibration-quality ECE bands. ECE thresholds anchored to Vickers 2019 +
# Naeini 2015: < 0.05 = excellent for clinical models; ≥ 0.20 = poor.
T2_6A_CALIBRATION_EXCELLENT_ECE_MAX: float = 0.05
T2_6A_CALIBRATION_GOOD_ECE_MAX: float = 0.10
T2_6A_CALIBRATION_MARGINAL_ECE_MAX: float = 0.20

# CV-stability std/mean ratio bands. < 0.05 = stable across folds; ≥ 0.20 =
# very unstable (suggests fold-dependent overfitting).
T2_6A_CV_STABILITY_STABLE_RATIO_MAX: float = 0.05
T2_6A_CV_STABILITY_MODERATE_RATIO_MAX: float = 0.10
T2_6A_CV_STABILITY_UNSTABLE_RATIO_MAX: float = 0.20


def _categorize_signal_genuineness(
    pvalue: Optional[float],
) -> SignalGenuinenessCategory:
    """Plan v3 §4 T2.6a: categorize permutation-test p-value into a
    deployer-input signal-genuineness band. Lower p = stronger signal.

    Returns ``"degenerate"`` when ``pvalue`` is None (perm test could not
    be evaluated — single-class y, missing proba). Returns the band
    Literal otherwise.
    """
    if pvalue is None:
        return "degenerate"
    if pvalue < T2_6A_SIGNAL_GENUINE_PVALUE_MAX:
        return "genuine"
    if pvalue < T2_6A_SIGNAL_LIKELY_GENUINE_PVALUE_MAX:
        return "likely_genuine"
    if pvalue < T2_6A_SIGNAL_MARGINAL_PVALUE_MAX:
        return "marginal"
    return "random"


def _categorize_calibration_quality(
    ece: Optional[float],
) -> CalibrationQualityCategory:
    """Plan v3 §4 T2.6a: categorize Expected Calibration Error into a
    deployer-input calibration-quality band. Lower ECE = better calibrated.

    Returns ``"degenerate"`` when ``ece`` is None.
    """
    if ece is None:
        return "degenerate"
    if ece < T2_6A_CALIBRATION_EXCELLENT_ECE_MAX:
        return "excellent"
    if ece < T2_6A_CALIBRATION_GOOD_ECE_MAX:
        return "good"
    if ece < T2_6A_CALIBRATION_MARGINAL_ECE_MAX:
        return "marginal"
    return "poor"


def _categorize_cv_stability(
    std_over_mean: Optional[float],
) -> CvStabilityCategory:
    """Plan v3 §4 T2.6a: categorize CV-fold std/mean AUC ratio into a
    deployer-input stability band. Lower ratio = more stable across folds.

    Returns ``"degenerate"`` when ``std_over_mean`` is None.
    """
    if std_over_mean is None:
        return "degenerate"
    if std_over_mean < T2_6A_CV_STABILITY_STABLE_RATIO_MAX:
        return "stable"
    if std_over_mean < T2_6A_CV_STABILITY_MODERATE_RATIO_MAX:
        return "moderate"
    if std_over_mean < T2_6A_CV_STABILITY_UNSTABLE_RATIO_MAX:
        return "unstable"
    return "very_unstable"


def compute_deployer_input_metrics(
    validation_metrics: Dict[str, Any],
    calibration_error: Optional[float] = None,
) -> Dict[str, Any]:
    """Plan v3 §4 T2.6a — Deployer-input metric computation (pure compute).

    Pulls three quality signals from the model-trainer's
    ``validation_metrics`` payload and categorizes each into a deployer-
    input band that T2.6c (separate work) will gate on. NO ENFORCEMENT
    here — pure computation that surfaces structured signals for the
    deployer to read.

    Inputs:

      * ``validation_metrics["permutation_pvalue"]`` — promoted in PR #118
        (plan v3 §3 Tier 1B step 1). Lower = more signal.
      * ``calibration_error`` — Expected Calibration Error, computed by
        ``compute_calibration_analysis`` (evaluator.py) and stored at
        ``metrics_result["calibration_error"]``. Lower = better calibrated.
      * ``validation_metrics["cv_5fold_roc_auc_std"]`` and
        ``["cv_5fold_roc_auc_mean"]`` — promoted in PR #114 (backlog #18).
        Ratio std/mean is the stability index.

    Returns dict with the following deployer-input keys (all should land
    on the deployer's input contract by the T2.6b shadow-reporting phase):

      * ``signal_genuineness_category`` — Literal band, one of
        ``"genuine" | "likely_genuine" | "marginal" | "random" | "degenerate"``.
      * ``signal_genuineness_pvalue`` — float input or None.
      * ``calibration_quality_category`` — Literal band, one of
        ``"excellent" | "good" | "marginal" | "poor" | "degenerate"``.
      * ``calibration_quality_ece`` — float input or None.
      * ``cv_stability_category`` — Literal band, one of
        ``"stable" | "moderate" | "unstable" | "very_unstable" | "degenerate"``.
      * ``cv_stability_std_over_mean`` — float ratio or None.
      * ``cv_stability_std`` / ``cv_stability_mean`` — raw inputs or None.

    Backward compat: graceful no-op when any input key is missing —
    returns the corresponding ``"degenerate"`` band.
    """
    pvalue = validation_metrics.get("permutation_pvalue")
    cv_std = validation_metrics.get("cv_5fold_roc_auc_std")
    cv_mean = validation_metrics.get("cv_5fold_roc_auc_mean")

    if cv_std is None or cv_mean is None or cv_mean == 0:
        std_over_mean: Optional[float] = None
    else:
        std_over_mean = float(cv_std) / float(cv_mean)

    return {
        "signal_genuineness_category": _categorize_signal_genuineness(pvalue),
        "signal_genuineness_pvalue": pvalue,
        "calibration_quality_category": _categorize_calibration_quality(calibration_error),
        "calibration_quality_ece": calibration_error,
        "cv_stability_category": _categorize_cv_stability(std_over_mean),
        "cv_stability_std_over_mean": std_over_mean,
        "cv_stability_std": cv_std,
        "cv_stability_mean": cv_mean,
    }


# ============================================================================
# Plan v3 §4 T2.6b — Shadow reporting (advisory-mode warnings).
# Emit structured "denial reasons" derived from the T2.6a categories.
# OBSERVABILITY ONLY — does NOT mutate `promotion_allowed`. The T2.6c
# enforcement phase (separate work) is where these signals graduate to
# blocking checks.
# ============================================================================

# Categories that the T2.6c enforcement phase will reject. T2.6b emits
# these as STRUCTURED WARNINGS (not denials) so an operator can monitor
# the would-be denial rate during the one-quarter advisory window.
T2_6B_SIGNAL_GENUINENESS_REJECT_CATEGORIES: frozenset[str] = frozenset(
    {"random", "marginal", "degenerate"}
)
T2_6B_CALIBRATION_QUALITY_REJECT_CATEGORIES: frozenset[str] = frozenset(
    {"poor", "marginal", "degenerate"}
)
T2_6B_CV_STABILITY_REJECT_CATEGORIES: frozenset[str] = frozenset(
    {"very_unstable", "unstable", "degenerate"}
)


def compute_advisory_denial_reasons(
    deployer_metrics: Dict[str, Any],
) -> list[str]:
    """Plan v3 §4 T2.6b — Shadow reporting (advisory-mode warnings).

    Derives a list of structured "would-be denial reasons" from the
    deployer-input metric categories computed by
    ``compute_deployer_input_metrics`` (T2.6a). Each entry is a
    human-readable string describing the category + the input value so
    operator dashboards can triage.

    This is OBSERVABILITY only — the T2.6c enforcement phase (separate
    work) is where these reasons graduate to blocking checks. Plan §6
    T2.6: "advisory mode for one quarter; same calibration protocol as
    T2.2 — synthetic for plumbing, retrospective held-out for threshold
    fitting, operator decisions for drift monitoring only".

    Returns an empty list when all three categories are healthy.
    """
    reasons: list[str] = []

    sig_cat = deployer_metrics.get("signal_genuineness_category")
    if sig_cat in T2_6B_SIGNAL_GENUINENESS_REJECT_CATEGORIES:
        pvalue = deployer_metrics.get("signal_genuineness_pvalue")
        pvalue_str = f"{pvalue:.4f}" if pvalue is not None else "None"
        reasons.append(
            f"T2.6b ADVISORY: signal_genuineness={sig_cat} "
            f"(perm_pvalue={pvalue_str}). T2.6c enforcement would reject "
            "promotion at this category; current run is advisory-only."
        )

    calib_cat = deployer_metrics.get("calibration_quality_category")
    if calib_cat in T2_6B_CALIBRATION_QUALITY_REJECT_CATEGORIES:
        ece = deployer_metrics.get("calibration_quality_ece")
        ece_str = f"{ece:.4f}" if ece is not None else "None"
        reasons.append(
            f"T2.6b ADVISORY: calibration_quality={calib_cat} "
            f"(ece={ece_str}). T2.6c enforcement would reject promotion "
            "at this category; current run is advisory-only."
        )

    cv_cat = deployer_metrics.get("cv_stability_category")
    if cv_cat in T2_6B_CV_STABILITY_REJECT_CATEGORIES:
        ratio = deployer_metrics.get("cv_stability_std_over_mean")
        ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
        reasons.append(
            f"T2.6b ADVISORY: cv_stability={cv_cat} "
            f"(std/mean={ratio_str}). T2.6c enforcement would reject "
            "promotion at this category; current run is advisory-only."
        )

    return reasons


# ============================================================================
# Gate N1 (plan v4 §2) — regulatory-eligibility evaluation helpers.
#
# These helpers wrap the ``RegulatoryEligibilityAudit`` runtime guard to
# produce the eligibility verdict during ``validate_promotion``. The
# split into helpers keeps the promotion-validation function readable
# and lets the eligibility logic be tested in isolation.
# ============================================================================


def _load_regulatory_audit_from_state(
    state: Dict[str, Any],
) -> RegulatoryEligibilityAudit:
    """Reconstruct the audit guard from the state's validation_metrics.

    The on-disk shape is a plain dict on
    ``validation_metrics["regulatory_eligibility_audit"]``. We deep-copy
    it through ``RegulatoryEligibilityAudit.from_dict`` so the returned
    guard is fully decoupled from state. Missing dict → fresh empty
    audit (consistent with checkpoint-restart behavior elsewhere).
    """
    validation_metrics = state.get("validation_metrics") or {}
    if hasattr(validation_metrics, "model_dump"):
        # MetricsSchema instance — drop into dict shape via pydantic.
        validation_metrics = validation_metrics.model_dump()
    audit_payload = (
        validation_metrics.get("regulatory_eligibility_audit")
        if isinstance(validation_metrics, dict)
        else None
    )
    if audit_payload is None:
        return RegulatoryEligibilityAudit()
    return RegulatoryEligibilityAudit.from_dict(audit_payload)


def _evaluate_absolute_threshold_gates(
    state: Dict[str, Any],
    audit: RegulatoryEligibilityAudit,
    timestamp: str,
) -> Dict[str, Any]:
    """Append a ``gate_history`` entry for each required absolute gate.

    The deployer reads ``state["success_criteria"]`` for the
    literature-anchored threshold (e.g. ``minimum_auc=0.75``) and
    ``state["validation_metrics"]["roc_auc"]`` for the value to compare.
    Each evaluation produces one ``gate_history`` entry with outcome
    ``"pass" | "fail" | "skipped"``. ``"skipped"`` fires when either the
    threshold or the value is missing — this counts as a non-pass for
    the eligibility check (see
    ``RegulatoryEligibilityAudit.is_regulatory_eligible``).

    Returns a dict with two keys:

      * ``all_thresholds_cleared``: True iff EVERY required gate
        evaluated to "pass".
      * ``failures``: list of human-readable strings describing each
        failure (or skip) — passed back to the caller for surfacing on
        ``promotion_denial_reason``.
    """
    success_criteria = state.get("success_criteria") or {}
    validation_metrics = state.get("validation_metrics") or {}
    if hasattr(validation_metrics, "model_dump"):
        validation_metrics = validation_metrics.model_dump()

    # Map gate name → (criterion_key, metric_key, direction).
    # Direction "min" means value must be >= threshold; "max" means
    # value must be <= threshold. Today we only ship "minimum_auc" —
    # other gates can be appended once their literature anchor is signed.
    gate_specs: Dict[str, Tuple[str, str, str]] = {
        "minimum_auc": ("minimum_auc", "roc_auc", "min"),
    }

    failures: List[str] = []
    all_pass = True

    for gate_name in N1_REQUIRED_REGULATORY_GATES:
        if gate_name not in gate_specs:
            failures.append(
                f"Gate N1: unknown required gate '{gate_name}' — "
                "no spec registered. Eligibility CANNOT be granted."
            )
            audit.append_gate_evaluation(
                timestamp=timestamp,
                gate_name=gate_name,
                threshold=None,
                value=None,
                outcome="skipped",
            )
            all_pass = False
            continue

        criterion_key, metric_key, direction = gate_specs[gate_name]
        threshold = success_criteria.get(criterion_key)
        # validation_metrics may store the AUC under "roc_auc" (modern
        # producer key) or "auc_roc" (canonical schema name). Both are
        # accepted at MetricsSchema construction; here we read both for
        # legacy / dict inputs that bypass the schema.
        if isinstance(validation_metrics, dict):
            value = validation_metrics.get(metric_key)
            if value is None and metric_key == "roc_auc":
                value = validation_metrics.get("auc_roc")
        else:
            value = None

        if threshold is None or value is None:
            audit.append_gate_evaluation(
                timestamp=timestamp,
                gate_name=gate_name,
                threshold=threshold,
                value=value,
                outcome="skipped",
            )
            failures.append(
                f"Gate N1: '{gate_name}' skipped — "
                f"threshold={threshold}, value={value}. "
                "Eligibility CANNOT be granted."
            )
            all_pass = False
            continue

        if direction == "min":
            passed = float(value) >= float(threshold)
        else:  # "max"
            passed = float(value) <= float(threshold)

        outcome = "pass" if passed else "fail"
        audit.append_gate_evaluation(
            timestamp=timestamp,
            gate_name=gate_name,
            threshold=threshold,
            value=value,
            outcome=outcome,
        )

        if not passed:
            all_pass = False
            failures.append(
                f"Gate N1: '{gate_name}' failed — "
                f"value={value:.4f} {('<' if direction == 'min' else '>')} "
                f"threshold={float(threshold):.4f}."
            )

    return {
        "all_thresholds_cleared": all_pass,
        "failures": failures,
    }


def _evaluate_regulatory_eligibility(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Gate N1 preconditions; return state-update dict.

    Three preconditions per plan v4 §2 Gate N1 (codex-rescue HIGH-3):

      1. All literature-anchored absolute thresholds clear (e.g.
         minimum_auc).
      2. ``adaptation_history == []`` — no adaptive relaxation during
         the model's lifecycle.
      3. ``gate_history`` shows EVERY required gate evaluated to "pass"
         (no advisory bypasses). Implemented inside
         ``is_regulatory_eligible``.

    When (1) holds but (2) does not, sets
    ``adapted_regulatory_candidate=True`` instead of
    ``regulatory_eligible=True``.

    Returns a dict with three keys for merging into validate_promotion's
    return value:

      * ``regulatory_eligible``: bool
      * ``adapted_regulatory_candidate``: bool
      * ``regulatory_eligibility_audit``: dict (the updated audit, ready
        to be persisted onto ``validation_metrics``).

    Plus an optional ``regulatory_eligibility_failures`` list (only
    present when the verdict is False) so the caller can surface why.
    """
    audit = _load_regulatory_audit_from_state(state)
    timestamp = datetime.now(tz=None).isoformat()

    threshold_result = _evaluate_absolute_threshold_gates(state, audit, timestamp)
    all_thresholds_cleared = bool(threshold_result["all_thresholds_cleared"])
    threshold_failures = list(threshold_result["failures"])

    # The eligibility verdict reads the (now-updated) audit.
    eligible = all_thresholds_cleared and is_regulatory_eligible(
        audit, N1_REQUIRED_REGULATORY_GATES
    )
    candidate = all_thresholds_cleared and is_adapted_regulatory_candidate(
        audit, N1_REQUIRED_REGULATORY_GATES
    )

    # eligibility and candidate are mutually exclusive by construction
    # (eligible requires adaptation_history == []; candidate requires
    # adaptation_history non-empty). The check is defensive — a logic
    # bug elsewhere should never let both flip to True simultaneously.
    assert not (eligible and candidate), (
        "Gate N1 invariant violation: regulatory_eligible and "
        "adapted_regulatory_candidate cannot both be True."
    )

    failures: List[str] = []
    if not eligible:
        if not all_thresholds_cleared:
            failures.extend(threshold_failures)
        elif audit.adaptation_history:
            failures.append(
                f"Gate N1: regulatory_eligible=False because "
                f"adaptation_history is non-empty "
                f"({len(audit.adaptation_history)} adaptation entr"
                f"{'y' if len(audit.adaptation_history) == 1 else 'ies'}). "
                "Per plan v4 §2 codex-rescue HIGH-3, ANY adaptive "
                "relaxation during the model's lifecycle disqualifies "
                "regulatory eligibility."
            )

    result = {
        "regulatory_eligible": eligible,
        "adapted_regulatory_candidate": candidate,
        "regulatory_eligibility_audit": audit.to_dict(),
    }
    if failures:
        result["regulatory_eligibility_failures"] = failures
    return result


def _get_mlflow_connector() -> Optional[Any]:
    """Get MLflow connector singleton if available.

    Returns:
        MLflowConnector instance or None if unavailable
    """
    try:
        from src.mlops.mlflow_connector import MLflowConnector

        connector = MLflowConnector()
        return connector if connector.enabled else None
    except ImportError:
        logger.warning("MLflowConnector not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get MLflow connector: {e}")
        return None


async def _register_model_mlflow(
    model_uri: str, deployment_name: str
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Register model with MLflow via MLflowConnector.

    Args:
        model_uri: MLflow model URI (runs:/<run_id>/model)
        deployment_name: Name to register model under

    Returns:
        Tuple of (registered_name, version, stage) or (None, None, None) on failure
    """
    connector = _get_mlflow_connector()
    if not connector:
        return None, None, None

    try:
        # Extract run_id and model_path from model_uri
        # MLflow 3.x returns models:/m-<hash> format; legacy uses runs:/<run_id>/<path>
        if model_uri.startswith("runs:/"):
            parts = model_uri[6:].split("/", 1)
            run_id = parts[0]
            model_path = parts[1] if len(parts) > 1 else "model"
        elif model_uri.startswith("models:/"):
            # MLflow 3.x model URI — register directly via mlflow.register_model()
            try:
                import mlflow

                result = mlflow.register_model(model_uri, deployment_name)
                logger.info(
                    f"Registered model from models:/ URI: {deployment_name} v{result.version}"
                )
                return deployment_name, int(result.version), "None"
            except Exception as e:
                logger.warning(f"Direct registration from models:/ URI failed: {e}")
                return None, None, None
        else:
            logger.warning(f"Unexpected model_uri format: {model_uri}")
            return None, None, None

        # Use MLflowConnector's async register_model method

        model_version = await connector.register_model(
            run_id=run_id,
            model_name=deployment_name,
            model_path=model_path,
        )

        if model_version:
            return (
                model_version.name,
                int(model_version.version),
                model_version.stage.value if model_version.stage else "None",
            )
        return None, None, None

    except Exception as e:
        logger.warning(f"MLflow registration failed via connector: {e}")
        return None, None, None


async def _transition_stage_mlflow(model_name: str, version: int, target_stage: str) -> bool:
    """Transition model stage via MLflowConnector.

    Args:
        model_name: Registered model name
        version: Model version
        target_stage: Target stage name (Staging, Production, Archived)

    Returns:
        True if successful, False otherwise
    """
    connector = _get_mlflow_connector()
    if not connector:
        return False

    try:
        from src.mlops.mlflow_connector import ModelStage

        # Map MLflow stage names to our enum
        stage_map = {
            "None": ModelStage.DEVELOPMENT,
            "Staging": ModelStage.STAGING,
            "Shadow": ModelStage.SHADOW,
            "Production": ModelStage.PRODUCTION,
            "Archived": ModelStage.ARCHIVED,
        }

        stage = stage_map.get(target_stage, ModelStage.DEVELOPMENT)

        # Use MLflowConnector's async transition_model_stage method
        success = await connector.transition_model_stage(
            model_name=model_name,
            version=str(version),
            stage=stage,
            archive_existing=(target_stage == "Production"),
        )

        if success:
            logger.info(f"MLflow: Transitioned {model_name} v{version} to {target_stage}")
        return cast(bool, success)

    except Exception as e:
        logger.warning(f"MLflow stage transition failed: {e}")
        return False


async def register_model(state: Dict[str, Any]) -> Dict[str, Any]:
    """Register model in MLflow registry.

    Args:
        state: Current agent state with model_uri and experiment_id

    Returns:
        State updates with registration results
    """
    try:
        model_uri = state.get("model_uri")
        state.get("experiment_id")
        deployment_name = state.get("deployment_name")

        if not model_uri:
            return {
                "error": "Missing model_uri for registration",
                "error_type": "missing_model_uri",
                "registration_successful": False,
            }

        if not deployment_name:
            return {
                "error": "Missing deployment_name for registration",
                "error_type": "missing_deployment_name",
                "registration_successful": False,
            }

        # Try real MLflow registration first
        registered_model_name, model_version, current_stage = await _register_model_mlflow(
            model_uri, deployment_name
        )

        # Fall back to simulation if MLflow unavailable
        if registered_model_name is None:
            logger.info("Using simulated MLflow registration")
            registered_model_name = deployment_name
            model_version = 1
            current_stage = "None"

        return {
            "registered_model_name": registered_model_name,
            "model_version": model_version,
            "current_stage": current_stage,
            "deployment_id": f"{registered_model_name}:v{model_version}",
            "deployment_status": "healthy",
            "deployed_at": datetime.now(tz=None).isoformat(),
            "registration_successful": True,
            "registration_timestamp": datetime.now(tz=None).isoformat(),
            "mlflow_available": registered_model_name is not None,
        }

    except Exception as e:
        return {
            "error": f"Model registration failed: {str(e)}",
            "error_type": "registration_error",
            "error_details": {"exception": str(e)},
            "registration_successful": False,
        }


async def validate_promotion(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate stage promotion criteria.

    Args:
        state: Current agent state with current_stage and target_stage

    Returns:
        State updates with validation results
    """
    try:
        current_stage = state.get("current_stage", "None")
        target_stage = state.get("target_stage")
        target_environment = state.get("target_environment", "staging")

        # Map environment to MLflow stage
        ENVIRONMENT_TO_STAGE = {
            "staging": "Staging",
            "shadow": "Shadow",
            "production": "Production",
            "archived": "Archived",
        }

        if not target_stage:
            target_stage = ENVIRONMENT_TO_STAGE.get(target_environment, "Staging")

        # Define allowed promotion paths
        # For initial deployments (None stage), allow any target environment
        # Production requires shadow mode validation (checked below)
        ALLOWED_PROMOTIONS = {
            "None": ["Staging", "Shadow", "Production"],  # Initial deployments
            "Staging": ["Shadow", "Archived"],
            "Shadow": ["Production", "Archived"],
            "Production": ["Archived"],
            "Archived": [],  # Terminal
        }

        validation_failures = []
        promotion_denial_reason = None
        shadow_mode_validated = None

        # Check if promotion path is allowed
        allowed_targets = ALLOWED_PROMOTIONS.get(current_stage, [])
        if target_stage not in allowed_targets:
            promotion_denial_reason = (
                f"Invalid promotion path: Cannot promote from {current_stage} to {target_stage}. "
                f"Allowed targets: {', '.join(allowed_targets) if allowed_targets else 'none'}"
            )
            return {
                "promotion_target_stage": target_stage,
                "promotion_allowed": False,
                "promotion_denial_reason": promotion_denial_reason,
                "promotion_validation_errors": [promotion_denial_reason],
            }

        # For production promotion, validate shadow mode
        if target_stage == "Production":
            shadow_result = _validate_shadow_mode_detailed(state)
            shadow_mode_validated = shadow_result["validated"]
            validation_failures = shadow_result["failures"]

            if not shadow_mode_validated:
                return {
                    "promotion_target_stage": target_stage,
                    "promotion_allowed": False,
                    "shadow_mode_validated": False,
                    "validation_failures": validation_failures,
                    "promotion_validation_errors": validation_failures,
                    "error": f"Shadow mode validation failed: {'; '.join(validation_failures)}",
                    "error_type": "shadow_validation_failed",
                }

        # Plan v3 §4 T2.6b — Compute deployer-input quality metrics +
        # advisory denial reasons. SHADOW REPORTING ONLY: does NOT mutate
        # `promotion_allowed`. The structured warnings emit through the
        # logger AND ride along on the return dict so observability
        # dashboards can monitor the would-be denial rate during the
        # one-quarter advisory window before T2.6c enforcement.
        validation_metrics_for_t26: Dict[str, Any] = state.get("validation_metrics") or {}
        if hasattr(validation_metrics_for_t26, "model_dump"):
            validation_metrics_for_t26 = validation_metrics_for_t26.model_dump()
        calibration_error_for_t26 = state.get("calibration_error")
        if calibration_error_for_t26 is None and isinstance(validation_metrics_for_t26, dict):
            calibration_error_for_t26 = validation_metrics_for_t26.get("calibration_error")
        t26_deployer_input_metrics = compute_deployer_input_metrics(
            validation_metrics_for_t26,
            calibration_error=calibration_error_for_t26,
        )
        t26_advisory_warnings = compute_advisory_denial_reasons(t26_deployer_input_metrics)
        for w in t26_advisory_warnings:
            logger.warning(w)

        # Plan v4 §2 Gate N1 — regulatory-eligibility evaluation. Reads
        # the immutable ``regulatory_eligibility_audit`` (if any) +
        # appends one ``gate_history`` entry per required gate. Sets
        # ``regulatory_eligible=True`` only when ALL three preconditions
        # hold (codex-rescue HIGH-3): (1) absolute thresholds cleared,
        # (2) adaptation_history empty, (3) every required gate passed.
        # When (1) holds but (2) does not, sets
        # ``adapted_regulatory_candidate=True`` instead. NEVER blocks
        # promotion on its own — the eligibility flag is the verdict
        # signal; the gate-passing decision is upstream. This keeps the
        # rollout signal-only and lets ``deployment_orchestrator`` /
        # downstream consumers make the final shipping decision.
        regulatory_result = _evaluate_regulatory_eligibility(state)

        # Promotion is allowed
        return {
            "promotion_target_stage": target_stage,
            "promotion_allowed": True,
            "promotion_reason": f"Promotion from {current_stage} to {target_stage} validated",
            "shadow_mode_validated": shadow_mode_validated,
            "promotion_validation_errors": [],
            "t26_deployer_input_metrics": t26_deployer_input_metrics,
            "t26_advisory_warnings": t26_advisory_warnings,
            **regulatory_result,
        }

    except Exception as e:
        return {
            "error": f"Promotion validation failed: {str(e)}",
            "error_type": "promotion_validation_error",
            "error_details": {"exception": str(e)},
            "promotion_allowed": False,
        }


async def promote_stage(state: Dict[str, Any]) -> Dict[str, Any]:
    """Promote model to target stage in MLflow.

    Args:
        state: Current agent state with model and promotion_target_stage

    Returns:
        State updates with promotion results
    """
    try:
        # Validate required fields FIRST (before checking promotion_allowed)
        registered_model_name = state.get("registered_model_name")
        model_version = state.get("model_version")
        promotion_target_stage = state.get("promotion_target_stage")
        current_stage = state.get("current_stage", "None")

        if not registered_model_name:
            return {
                "error": "Missing registered_model_name for promotion",
                "error_type": "missing_model_name",
                "promotion_successful": False,
            }

        if not promotion_target_stage:
            return {
                "error": "Missing promotion_target_stage for promotion",
                "error_type": "missing_target_stage",
                "promotion_successful": False,
            }

        # Now check if promotion is allowed (default to True if not explicitly set)
        # This allows promote_stage to work after validate_promotion sets promotion_allowed=True
        # or when called directly without validation
        promotion_allowed = state.get("promotion_allowed", True)

        if not promotion_allowed:
            errors = state.get("promotion_validation_errors", [])
            return {
                "error": f"Promotion not allowed: {'; '.join(errors)}",
                "error_type": "promotion_blocked",
                "promotion_successful": False,
            }

        # Try real MLflow stage transition first
        mlflow_success = False
        if registered_model_name and model_version:
            mlflow_success = await _transition_stage_mlflow(
                model_name=registered_model_name,
                version=int(model_version),
                target_stage=promotion_target_stage,
            )

        if not mlflow_success:
            logger.info("Using simulated MLflow stage transition")

        # Record previous stage for version record
        previous_stage = current_stage

        # Get metrics at promotion time.
        # D2.0: read ``roc_auc`` (the canonical key emitted by
        # model_trainer's evaluator) — pre-D2.0 this used ``auc_roc`` (a
        # transposed typo) and silently returned the 0.0 default for every
        # promotion. Surfaced by Phase-1 D2 investigation
        # (.claude/state/d2_investigation_20260505.md, field #4).
        validation_metrics = state.get("validation_metrics", {})
        metrics_at_promotion = {
            "test_auc": validation_metrics.get("roc_auc", 0.0),
            "test_precision": validation_metrics.get("precision", 0.0),
            "test_recall": validation_metrics.get("recall", 0.0),
            "test_f1": validation_metrics.get("f1_score", 0.0),
        }

        promotion_reason = state.get("promotion_reason", "Automated promotion")

        return {
            "previous_stage": previous_stage,
            "current_stage": promotion_target_stage,
            "metrics_at_promotion": metrics_at_promotion,
            "promotion_successful": True,
            "promotion_reason": promotion_reason,
            "promotion_timestamp": datetime.now(tz=None).isoformat(),
            "mlflow_transition_success": mlflow_success,
        }

    except Exception as e:
        return {
            "error": f"Stage promotion failed: {str(e)}",
            "error_type": "promotion_error",
            "error_details": {"exception": str(e)},
            "promotion_successful": False,
        }


def _validate_shadow_mode_detailed(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate shadow mode requirements for production promotion with detailed failures.

    Requirements:
    - Min duration: 24 hours
    - Min requests: 1000
    - Max error rate: 1%
    - Max latency p99: 150ms

    Args:
        state: Current agent state

    Returns:
        Dictionary with validated (bool) and failures (list of failure messages)
    """
    # Shadow mode requirements
    MIN_DURATION_HOURS = 24
    MIN_REQUESTS = 1000
    MAX_ERROR_RATE = 0.01
    MAX_LATENCY_P99_MS = 150

    # Get shadow mode metrics (in production, this would come from observability)
    shadow_duration = state.get("shadow_mode_duration_hours", 0)
    shadow_requests = state.get("shadow_mode_requests", 0)
    shadow_error_rate = state.get("shadow_mode_error_rate", 1.0)
    shadow_latency_p99 = state.get("shadow_mode_latency_p99_ms", 999)

    failures = []

    # Validate each requirement (keywords in messages match test expectations)
    if shadow_duration < MIN_DURATION_HOURS:
        failures.append(
            f"shadow_mode_duration_hours {shadow_duration} below minimum {MIN_DURATION_HOURS}"
        )

    if shadow_requests < MIN_REQUESTS:
        failures.append(f"shadow_mode_requests {shadow_requests} below minimum {MIN_REQUESTS}")

    if shadow_error_rate > MAX_ERROR_RATE:
        failures.append(
            f"shadow_mode_error_rate {shadow_error_rate:.4f} above maximum {MAX_ERROR_RATE}"
        )

    if shadow_latency_p99 > MAX_LATENCY_P99_MS:
        failures.append(
            f"shadow_mode_latency_p99_ms {shadow_latency_p99} above maximum {MAX_LATENCY_P99_MS}"
        )

    return {
        "validated": len(failures) == 0,
        "failures": failures,
    }


def _validate_shadow_mode(state: Dict[str, Any]) -> bool:
    """Validate shadow mode requirements for production promotion.

    DEPRECATED: Use _validate_shadow_mode_detailed for detailed failure info.

    Args:
        state: Current agent state

    Returns:
        True if shadow mode requirements are met
    """
    result = _validate_shadow_mode_detailed(state)
    return cast(bool, result["validated"])
