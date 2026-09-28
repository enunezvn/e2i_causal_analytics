"""Which skops-untrusted types an sklearn model may be logged with (#2280).

mlflow (3.15.x, the worker's version) serializes sklearn models with skops, and skops
0.14.0 refuses any type outside its default trust list unless it is passed as
``skops_trusted_types``. A ``CalibratedClassifierCV`` holds private sklearn calibration
classes that are not on that list, so every calibrated pipeline model failed to log.

Measured 2026-09-23 on real calibrated models (mlflow 3.15.1, skops 0.14.0, sklearn
1.6.1): a sigmoid calibrator reports ``_CalibratedClassifier`` and ``_SigmoidCalibration``;
an isotonic one reports only ``_CalibratedClassifier`` (``IsotonicRegression`` and
``FrozenEstimator`` are already trusted). The allowlist starts with exactly those sklearn
classes.

A calibrated XGBoost / LightGBM is logged with the sklearn flavor too (the native flavors
cannot save a ``CalibratedClassifierCV``), and skops also reports its booster classes.
Trusting them was an owner decision after #2280. Measured 2026-09-28 (xgboost 3.1.2,
lightgbm 4.6.0 — the worker's versions), identical for sigmoid/isotonic, numpy or
DataFrame input, with or without early stopping:

* XGBoost: ``xgboost.core.Booster``, ``xgboost.sklearn.XGBClassifier``
* LightGBM: ``lightgbm.basic.Booster``, ``lightgbm.sklearn.LGBMClassifier`` and
  ``collections.OrderedDict``. skops serializes OrderedDict *instances* natively; it
  reports the name only for a reference to the class, here the ``default_factory`` of
  ``Booster.best_score = defaultdict(OrderedDict)``. That reference is trusted only when
  the model also reports a LightGBM booster, which is the one place it was measured.

Loading a trusted booster runs the library's own ``__setstate__`` (xgboost's raw model
buffer, LightGBM's model string), not pickle. A model that reports any other untrusted
type is trusted with NOTHING, so mlflow refuses it loudly rather than this module
widening trust.

mlflow stores the trusted list in the model's flavor config, and both
``mlflow.sklearn.load_model`` and ``mlflow.pyfunc.load_model`` read it back, so no load
site needs its own list. The allowlist governs what WE write; loading trusts the
artifact's own ``MLmodel`` exactly as before (an artifact that declares cloudpickle is
unpickled by every loader) — hardening the artifact-store trust boundary is out of
this module's scope.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

TRUSTED_SKLEARN_CALIBRATION_TYPES: frozenset[str] = frozenset(
    {
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
    }
)

TRUSTED_XGBOOST_TYPES: frozenset[str] = frozenset(
    {"xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"}
)

TRUSTED_LIGHTGBM_TYPES: frozenset[str] = frozenset(
    {"lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier"}
)

# Trusted only when the model also reports a LightGBM booster (see the module docstring).
LIGHTGBM_BOOSTER_STATE_TYPES: frozenset[str] = frozenset({"collections.OrderedDict"})

TRUSTED_CALIBRATED_MODEL_TYPES: frozenset[str] = (
    TRUSTED_SKLEARN_CALIBRATION_TYPES | TRUSTED_XGBOOST_TYPES | TRUSTED_LIGHTGBM_TYPES
)


def _allowlist_for(reported: set[str]) -> frozenset[str]:
    """The allowlist for a model reporting ``reported``: OrderedDict only beside a LightGBM booster."""
    if "lightgbm.basic.Booster" in reported:
        return TRUSTED_CALIBRATED_MODEL_TYPES | LIGHTGBM_BOOSTER_STATE_TYPES
    return TRUSTED_CALIBRATED_MODEL_TYPES


def skops_trusted_types_for(model: Any) -> List[str]:
    """The skops-untrusted types of ``model`` to trust: all of them, only if all are allowlisted.

    Returns ``[]`` for a model that is not a fitted calibrator, when skops is unavailable,
    or when any reported type is outside the allowlist (the sklearn calibration classes,
    the XGBoost / LightGBM booster classes, and OrderedDict beside a LightGBM booster):
    mlflow then refuses the model loudly.
    """
    if not hasattr(model, "calibrated_classifiers_"):
        return []  # only a fitted calibrator carries these types; no second serialization
    try:
        import skops.io as sio
    except ImportError:
        return []
    try:
        reported = sio.get_untrusted_types(data=sio.dumps(model))
    except Exception as e:  # noqa: BLE001 — unserializable: mlflow's own save reports it
        logger.warning("skops could not inspect %s (%s); trusting nothing", type(model).__name__, e)
        return []
    outside = sorted(set(reported) - _allowlist_for(set(reported)))
    if outside:
        logger.warning("skops: not trusting %s (outside the calibrated-model allowlist)", outside)
        return []
    return sorted(reported)


def _default_serialization_format(mlflow: Any) -> Any:
    """``mlflow.sklearn.log_model``'s default format (skops from mlflow 3.15)."""
    import inspect

    try:
        param = inspect.signature(mlflow.sklearn.log_model).parameters["serialization_format"]
    except (KeyError, TypeError, ValueError):
        return None
    return param.default


def sklearn_log_model_kwargs(mlflow: Any, model: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """``mlflow.sklearn.log_model`` kwargs plus the model's allowlisted skops trust.

    Only when the EFFECTIVE format is skops (the caller's, else mlflow's default) and the
    caller gave no trust list of its own. The format itself is never changed: callers that
    ask for cloudpickle (the NGBoost / MAPIE wrappers, via the trainer's
    ``_sklearn_serialization_kwargs``) keep it.
    """
    out = dict(kwargs)
    skops_format = mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS
    effective = out.get("serialization_format", _default_serialization_format(mlflow))
    if effective == skops_format and "skops_trusted_types" not in out:
        trusted = skops_trusted_types_for(model)
        if trusted:
            out["skops_trusted_types"] = trusted
    return out
