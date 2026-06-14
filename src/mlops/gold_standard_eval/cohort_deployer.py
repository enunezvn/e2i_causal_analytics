"""CohortDeployer — train on GOLD-STANDARD features + register a non-colliding model.

This is the deployable counterpart to ``walk_forward`` / ``recorder``: it trains
ONE calibrated estimator on the gold-standard, FeatureBuilder-encoded named-column
DataFrame and registers it so the Time-Series page can read its recorded metrics
back from ``ml_model_registry``.

How it DIFFERS from ``src/mlops/prediction_synthesizer_deploy.py``
------------------------------------------------------------------
1. **Features.** The serving deploy module trains on the 60 synthetic-generator
   features (``generate_scenario``). This trains on the 3-covariate-derived
   encoded set produced by :class:`FeatureBuilder` — so ``feature_names_in_``
   equals ``FeatureBuilder.feature_columns``. (The empirically-locked KEEP_COLUMNS
   measured the best held-out AUC with these covariates alone; see
   ``feature_builder.py``.)

2. **No serving manifest.** The eval model is NOT served via the
   prediction_synthesizer chat path, so it is deliberately absent from
   ``deployment_manifest.json``. We still serialize a pickle to a real
   ``artifact_path`` (loadability / honesty), but skip the manifest entirely.

3. **Staging, not production — MANDATORY collision guard.** The initiation
   target ``csu_treatment_initiation`` ALREADY has a ``production`` 60-feature
   serving champion. ``MLModelRegistryRepository.get_models_for_target`` filters
   ``_SERVING_STAGES=('production',)``, so registering this (incompatible-feature)
   model at ``stage='production'`` would make that resolver return BOTH and break
   the serving ensemble. We register at ``stage='staging'`` — a valid
   ``model_stage_enum`` value (``database/ml/mlops_tables.sql``) — which:
     * is EXCLUDED by ``get_models_for_target`` (serving is unaffected), and
     * is STILL resolvable by ``_resolve_model_id`` (matches by model_name then
       model_version, regardless of stage) so the trend read-path finds it.

   A distinct experiment (``initiation_goldstd_eval_v1``) and distinct
   model_name (``csu_initiation_goldstd_lr_v1``) keep it cleanly separated from
   the serving deploy's experiment/model namespace.

The REAL prod registration runs in T11's CLI; this module exposes the functions
(unit-tested against a fake client) it will call.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# Reuse the shared, behavior-preserving primitives from the serving deploy module.
from src.mlops.prediction_synthesizer_deploy import (
    _get_or_create_experiment,
    register_model_row,
    serialize_model,
)

logger = logging.getLogger(__name__)

# Distinct namespace from the serving deploy (csu_treatment_initiation_live_v1 /
# csu_treatment_initiation_lr_*). Same prediction_target, different experiment so
# the two never collide in the registry.
GOLDSTD_EXPERIMENT_NAME = "initiation_goldstd_eval_v1"
GOLDSTD_MODEL_NAME = "csu_initiation_goldstd_lr_v1"
GOLDSTD_MODEL_VERSION = "1.0"
GOLDSTD_ALGORITHM = "logistic_regression_calibrated"

# Eval model is NOT a serving champion and NOT production — keeps serving's
# production-only get_models_for_target filter from ever surfacing it.
GOLDSTD_STAGE = "staging"

# Re-export so callers/tests have one import site for the serialization primitive.
__all__ = [
    "GOLDSTD_ALGORITHM",
    "GOLDSTD_EXPERIMENT_NAME",
    "GOLDSTD_MODEL_NAME",
    "GOLDSTD_MODEL_VERSION",
    "GOLDSTD_STAGE",
    "register_cohort_model",
    "serialize_model",
    "train_cohort_model",
]


def train_cohort_model(spec: Any, X: pd.DataFrame, y: "pd.Series[int]") -> Any:
    """Fit a calibrated LogisticRegression on the gold-standard feature frame.

    The model is fit on the NAMED-column DataFrame *X* (the FeatureBuilder
    output), so ``estimator.feature_names_in_`` is exactly ``X.columns`` — the
    3-covariate-derived encoded set, NOT the 60 synthetic-generator features.
    This is the deliberate opposite of ``prediction_synthesizer_deploy``.

    Mirrors the production-default estimator the WalkForwardRunner uses
    (``LogisticRegression(class_weight='balanced', max_iter=1000)``) and wraps it
    in ``CalibratedClassifierCV`` so the recorded probabilities (and the metrics
    the Time-Series page reads) are calibrated. ``cv`` adapts to the smaller of 3
    or the minority-class count so tiny frames (unit tests) still fit.

    Parameters
    ----------
    spec:
        A ``CohortSpec`` (carried for symmetry / future per-cohort policy; the
        estimator config is currently cohort-agnostic).
    X:
        FeatureBuilder-encoded feature matrix (named columns, float, no NaNs).
    y:
        Integer label series aligned to *X*.

    Returns
    -------
    A fitted estimator with ``predict_proba`` and ``feature_names_in_ == X.columns``.
    """
    base = LogisticRegression(class_weight="balanced", max_iter=1000)

    # CalibratedClassifierCV needs cv folds <= the minority-class count. Real
    # runs have thousands of rows (cv=3); guard small frames so the unit test and
    # any degenerate split still fit instead of raising.
    minority = int(pd.Series(y).value_counts().min()) if len(y) else 0
    cv = max(2, min(3, minority))
    if minority < 2:
        # Not enough per-class samples to calibrate by CV — fit the bare
        # estimator (still carries feature_names_in_ from the named frame).
        logger.warning(
            "train_cohort_model: minority class has %d sample(s); skipping CV "
            "calibration and fitting the bare LogisticRegression.",
            minority,
        )
        base.fit(X, y)
        return base

    model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    model.fit(X, y)
    # CalibratedClassifierCV exposes feature_names_in_ when fit on a DataFrame.
    return model


async def register_cohort_model(
    client: Any,
    spec: Any,
    *,
    model_name: str = GOLDSTD_MODEL_NAME,
    model_version: str = GOLDSTD_MODEL_VERSION,
    artifact_path: str,
    auc: float,
    feature_count: int,
    training_samples: int | None = None,
    stage: str = GOLDSTD_STAGE,
    experiment_name: str = GOLDSTD_EXPERIMENT_NAME,
) -> str:
    """Register the gold-standard model row at ``stage='staging'`` (collision-safe).

    Resolves (or creates) a DISTINCT experiment for ``spec.target`` and writes a
    single ``ml_model_registry`` row via the shared :func:`register_model_row`
    primitive (idempotent replace at ``(model_name, model_version)``,
    artifact-exists check, read-back verification that it landed at ``stage``).

    The default ``stage='staging'`` is the mandatory guard: it keeps the model
    out of ``get_models_for_target`` (production-only serving) while leaving it
    resolvable by ``_resolve_model_id`` for the Time-Series trend endpoint. It is
    registered ``is_synthetic=False`` (a real fitted model) and ``is_champion=
    False`` (it is not the serving champion).

    Returns the registered ``model_name``.
    """
    if stage == "production":
        # Hard refusal: a production gold-standard row would collide with the
        # 60-feature serving champion in get_models_for_target. T9's whole point
        # is to avoid that — never let a caller opt into the collision here.
        raise ValueError(
            "register_cohort_model refuses stage='production': the gold-standard "
            "eval model must not enter the serving ensemble for "
            f"{getattr(spec, 'target', '?')!r} (it has incompatible features). "
            "Use stage='staging'."
        )

    experiment_id = await _resolve_goldstd_experiment(client, spec, experiment_name)

    return await register_model_row(
        client,
        experiment_id=experiment_id,
        model_name=model_name,
        model_version=model_version,
        algorithm=GOLDSTD_ALGORITHM,
        artifact_path=artifact_path,
        auc=auc,
        feature_count=feature_count,
        training_samples=training_samples,
        stage=stage,
        is_champion=False,
        is_synthetic=False,
    )


async def _resolve_goldstd_experiment(client: Any, spec: Any, experiment_name: str) -> str:
    """Resolve/create the gold-standard experiment for ``spec.target``.

    Delegates to the deploy module's ``_get_or_create_experiment``, which matches
    on ``(experiment_name, prediction_target)``. That helper reads its target
    from the deploy module's module-level ``PREDICTION_TARGET`` constant — which
    is the SAME ``csu_treatment_initiation`` target as the initiation cohort, so
    reuse is correct here. The assertion documents (and enforces) that coupling
    so a future non-initiation cohort cannot silently register under the wrong
    target.
    """
    from src.mlops import prediction_synthesizer_deploy as _deploy

    target = getattr(spec, "target", None)
    if target != _deploy.PREDICTION_TARGET:
        raise ValueError(
            f"register_cohort_model currently supports only the "
            f"{_deploy.PREDICTION_TARGET!r} target (got {target!r}); "
            "_get_or_create_experiment resolves that target. Extend the helper "
            "before deploying another cohort."
        )
    return await _get_or_create_experiment(client, experiment_name)
