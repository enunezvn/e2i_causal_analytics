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
  ``Booster.best_score = defaultdict(OrderedDict)``. It is trusted only for a calibrated
  LightGBM, the one place it was measured. (skops's type list cannot say WHERE in the
  graph a class reference sits; a reference to this stdlib dict subclass constructs
  nothing on load.)

The booster family is chosen from the calibrated base estimator's exact class, never from
the reported list: a calibrated LogisticRegression that happens to hold an XGBoost object,
or a calibrated XGBoost holding LightGBM state, is trusted with nothing.

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

# Trusted only when the calibrated base IS an LGBMClassifier (see the module docstring).
LIGHTGBM_BOOSTER_STATE_TYPES: frozenset[str] = frozenset({"collections.OrderedDict"})

# The calibrated base estimator's exact class -> the extra types that base brings.
_BOOSTER_FAMILY_TYPES: Dict[str, frozenset[str]] = {
    "xgboost.sklearn.XGBClassifier": TRUSTED_XGBOOST_TYPES,
    "lightgbm.sklearn.LGBMClassifier": TRUSTED_LIGHTGBM_TYPES | LIGHTGBM_BOOSTER_STATE_TYPES,
}


def _qualname(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _calibrated_base_classes(model: Any) -> set[str]:
    """Exact classes of the estimators the calibrator wraps (``FrozenEstimator`` unwrapped)."""
    wrapped = [getattr(cc, "estimator", None) for cc in model.calibrated_classifiers_]
    wrapped.append(getattr(model, "estimator", None))
    bases = set()
    for est in wrapped:
        if type(est).__name__ == "FrozenEstimator":
            est = getattr(est, "estimator", None)
        if est is not None:
            bases.add(_qualname(est))
    return bases


def _allowlist_for(model: Any) -> frozenset[str]:
    """The sklearn calibration classes plus ONE booster family, chosen from the calibrated
    base estimator itself: a calibrated XGBoost gets the XGBoost types, a calibrated
    LightGBM the LightGBM types (+ OrderedDict). A base of any other class, or bases of
    mixed classes, get no booster types at all, so a booster type reported anywhere else
    in the object graph falls outside the allowlist."""
    bases = _calibrated_base_classes(model)
    if len(bases) == 1:
        (base,) = bases
        return TRUSTED_SKLEARN_CALIBRATION_TYPES | _BOOSTER_FAMILY_TYPES.get(base, frozenset())
    return TRUSTED_SKLEARN_CALIBRATION_TYPES


def skops_trusted_types_for(model: Any) -> List[str]:
    """The skops-untrusted types of ``model`` to trust: all of them, only if all are allowlisted.

    Returns ``[]`` for a model that is not a fitted calibrator, when skops is unavailable,
    or when any reported type is outside the allowlist (the sklearn calibration classes,
    plus the booster family of the calibrated base estimator, see ``_allowlist_for``):
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
    outside = sorted(set(reported) - _allowlist_for(model))
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
