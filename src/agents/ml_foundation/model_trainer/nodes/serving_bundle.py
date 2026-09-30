"""Serving bundle for a trained model (#2318): the exact fitted preprocessing + estimator.

A retrain candidate can only be served as-trained if its preprocessing is persisted with it
(before #2318 only the estimator was logged, so serving it needed a re-fit). The bundle holds
sklearn/numpy/builtins only, so the BentoML sidecar, which cannot import ``src.*``, can load it.

Contract (``bundle_format = "sklearn_ct_v1"``), read by the sidecar's ColumnTransformer adapter:

- ``model``: the fitted (deployed) estimator, e.g. ``CalibratedClassifierCV``.
- ``preprocessor``: the fitted sklearn ``ColumnTransformer`` (``ModelTrainerPreprocessor._pipeline``).
- ``keep_columns``: raw input column names, in fit order.
- ``numeric_columns``: the subset of ``keep_columns`` treated as numeric.
- ``feature_columns``: encoded output names, ``num__``/``cat__`` prefixes stripped.
- ``sklearn_version``: the version that pickled it (the sidecar must unpickle on the same one).

It is logged as run artifact ``serving_bundle/bundle.pkl`` with run tag
``e2i.serving_bundle_sha256`` = sha256 of those exact bytes.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

import sklearn

BUNDLE_FORMAT = "sklearn_ct_v1"
BUNDLE_ARTIFACT_DIR = "serving_bundle"
BUNDLE_FILENAME = "bundle.pkl"
BUNDLE_SHA_TAG = "e2i.serving_bundle_sha256"


def _strip(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def build_serving_bundle(*, model: Any, preprocessor: Any) -> dict[str, Any]:
    """Pair the fitted estimator with the exact ColumnTransformer it was trained behind.

    Raises ``ValueError`` when the pair cannot be served as-is: an unfitted or foreign
    preprocessor, one fit without raw column names (nothing to validate a request against),
    or an estimator whose input width or feature names differ from the preprocessor's output.
    """
    ct = getattr(preprocessor, "_pipeline", None)
    if ct is None or not getattr(preprocessor, "_is_fitted", False):
        raise ValueError("serving bundle needs a fitted ModelTrainerPreprocessor")
    raw_names = getattr(ct, "feature_names_in_", None)
    if raw_names is None:
        raise ValueError(
            "serving bundle needs a preprocessor fit on a DataFrame with raw column names"
        )

    feature_columns = [_strip(str(n)) for n in ct.get_feature_names_out()]
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(feature_columns):
        raise ValueError(
            f"model expects {int(n_in)} features but the preprocessor emits "
            f"{len(feature_columns)}: they were not trained together"
        )
    model_names = getattr(model, "feature_names_in_", None)
    if model_names is not None and [str(n) for n in model_names] != feature_columns:
        raise ValueError("model feature names differ from the preprocessor's output names")

    return {
        "bundle_format": BUNDLE_FORMAT,
        "model": model,
        "preprocessor": ct,
        "keep_columns": [str(c) for c in raw_names],
        "numeric_columns": [str(c) for c in preprocessor.numeric_features],
        "feature_columns": feature_columns,
        "sklearn_version": sklearn.__version__,
    }


def serialize_bundle(bundle: dict[str, Any]) -> tuple[bytes, str]:
    """Pickle the bundle and return ``(bytes, sha256 hex of those bytes)``."""
    blob = pickle.dumps(bundle, protocol=5)
    return blob, hashlib.sha256(blob).hexdigest()
