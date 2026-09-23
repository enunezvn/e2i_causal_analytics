"""Which skops-untrusted types an sklearn model may be logged with (#2280).

mlflow (3.15.x, the worker's version) serializes sklearn models with skops, and skops
0.14.0 refuses any type outside its default trust list unless it is passed as
``skops_trusted_types``. A ``CalibratedClassifierCV`` holds private sklearn calibration
classes that are not on that list, so every calibrated pipeline model failed to log.

Measured 2026-09-23 on real calibrated models (mlflow 3.15.1, skops 0.14.0, sklearn
1.6.1): a sigmoid calibrator reports ``_CalibratedClassifier`` and ``_SigmoidCalibration``;
an isotonic one reports only ``_CalibratedClassifier`` (``IsotonicRegression`` and
``FrozenEstimator`` are already trusted). The allowlist is exactly those sklearn classes.
A model that reports any other untrusted type is trusted with NOTHING, so mlflow refuses
it loudly rather than this module widening trust.

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


def skops_trusted_types_for(model: Any) -> List[str]:
    """The skops-untrusted types of ``model`` to trust: all of them, only if all are allowlisted.

    Returns ``[]`` for a model that is not a fitted calibrator, when skops is unavailable,
    or when any reported type is outside :data:`TRUSTED_SKLEARN_CALIBRATION_TYPES` (e.g. a
    calibrated XGBoost / LightGBM also reports its booster classes: mlflow then refuses it
    loudly — trusting booster classes is a separate decision, see #2280).
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
    outside = sorted(set(reported) - TRUSTED_SKLEARN_CALIBRATION_TYPES)
    if outside:
        logger.warning(
            "skops: not trusting %s (outside the sklearn calibration allowlist)", outside
        )
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
    rely on cloudpickle (e.g. the NGBoost / MAPIE wrappers, ``_get_mlflow_flavor``) keep
    whatever they ask for.
    """
    out = dict(kwargs)
    skops_format = mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS
    effective = out.get("serialization_format", _default_serialization_format(mlflow))
    if effective == skops_format and "skops_trusted_types" not in out:
        trusted = skops_trusted_types_for(model)
        if trusted:
            out["skops_trusted_types"] = trusted
    return out
