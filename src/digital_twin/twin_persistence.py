"""
Twin Persistence (#705 H4)
==========================

Real MLflow persistence + round-trip load for digital-twin *generative* models.

This module replaces the fabricated ``TwinModelRepository._save_to_mlflow`` stub
(which returned ``models:/twin_<type>_<brand>/latest`` without ever logging an
artifact) with honest persistence:

* the fitted sklearn estimator is logged via ``mlflow.sklearn.log_model`` →
  a real, loadable ``model_uri``;
* a single pickled *bundle* artifact captures every other piece of generator
  state needed to reproduce generation exactly — the fitted ``StandardScaler``,
  the ``LabelEncoder`` dict, the canonical ``feature_columns`` ORDER, the
  ``target_column``, the per-feature stats and categorical distributions, plus
  the ``model_id`` / ``metrics`` / ``twin_type`` / ``brand`` identity.

Logging the estimator alone (the audit's round-trip hazard) silently drops the
preprocessors, so a reloaded generator would scale/encode with ``None`` and feed
the model garbage. Bundling them guarantees a bit-for-bit round trip.

The estimator is restored with ``mlflow.sklearn.load_model``; the bundle is
downloaded from the same run and unpickled. MLflow usage is concentrated here so
the repository stays a pure DB layer and tests can point at a ``file://`` store.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .twin_generator import TwinGenerator

logger = logging.getLogger(__name__)

_DEFAULT_EXPERIMENT = "digital_twins"
_BUNDLE_ARTIFACT_DIR = "twin_bundle"
_BUNDLE_FILENAME = "twin_bundle.pkl"
_BUNDLE_ARTIFACT_PATH = f"{_BUNDLE_ARTIFACT_DIR}/{_BUNDLE_FILENAME}"

# Every non-estimator piece of TwinGenerator state required to reproduce
# generation. ORDER of feature_columns is load-bearing (it indexes the model's
# prediction array), so it must round-trip exactly — hence a pickled bundle
# rather than scattered DB columns.
_BUNDLE_FIELDS = (
    "scaler",
    "label_encoders",
    "feature_columns",
    "target_column",
    "_feature_stats",
    "_categorical_distributions",
    "model_id",
    "metrics",
    "twin_type",
    "brand",
)


@dataclass(frozen=True)
class TwinArtifactRef:
    """A real reference to a persisted twin model (NOT a fabricated URI)."""

    run_id: str
    model_uri: str


def _mlflow(tracking_uri: Optional[str] = None):
    """Import mlflow and point it at the configured tracking store."""
    import mlflow

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    return mlflow


def _extract_bundle(generator: "TwinGenerator") -> Dict[str, Any]:
    return {field: getattr(generator, field, None) for field in _BUNDLE_FIELDS}


def save_twin_artifacts(
    generator: "TwinGenerator",
    *,
    experiment: str = _DEFAULT_EXPERIMENT,
    tracking_uri: Optional[str] = None,
) -> TwinArtifactRef:
    """Persist a trained generator to MLflow and return a real run_id + model_uri.

    Raises:
        ValueError: if the generator is untrained (``model`` is ``None``) — we
            fail closed rather than persist a phantom model reference.
    """
    if getattr(generator, "model", None) is None:
        raise ValueError("Cannot persist an untrained generator (model is None)")

    mlflow = _mlflow(tracking_uri)
    mlflow.set_experiment(experiment)

    bundle = _extract_bundle(generator)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model_info = mlflow.sklearn.log_model(generator.model, name="model")
        model_uri = getattr(model_info, "model_uri", None) or f"runs:/{run_id}/model"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, _BUNDLE_FILENAME)
            with open(path, "wb") as fh:
                pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
            mlflow.log_artifact(path, artifact_path=_BUNDLE_ARTIFACT_DIR)

    logger.info("Persisted twin model: run_id=%s model_uri=%s", run_id, model_uri)
    return TwinArtifactRef(run_id=run_id, model_uri=model_uri)


def load_twin_bundle(
    model_uri: str,
    run_id: str,
    *,
    tracking_uri: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the estimator + preprocessor bundle for a persisted twin model."""
    mlflow = _mlflow(tracking_uri)
    model = mlflow.sklearn.load_model(model_uri)
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=_BUNDLE_ARTIFACT_PATH)
    with open(local, "rb") as fh:
        bundle: Dict[str, Any] = pickle.load(fh)
    return model, bundle


def hydrate_generator(
    generator: "TwinGenerator",
    model_uri: Optional[str],
    run_id: Optional[str],
    *,
    tracking_uri: Optional[str] = None,
) -> bool:
    """Restore a trained generator's full state in place.

    Returns ``True`` on success. Returns ``False`` (never raises) when the model
    reference is missing or the artifact can't be loaded, so callers can fail
    closed — e.g. surface a 503 rather than a misleading 500 or fabricated result.
    """
    if not model_uri or not run_id:
        return False
    try:
        model, bundle = load_twin_bundle(model_uri, run_id, tracking_uri=tracking_uri)
    except Exception as exc:  # noqa: BLE001 — degrade to a fail-closed signal
        logger.warning("Failed to load twin model uri=%s run_id=%s: %s", model_uri, run_id, exc)
        return False

    generator.model = model
    generator.scaler = bundle.get("scaler")
    generator.label_encoders = bundle.get("label_encoders") or {}
    generator.feature_columns = bundle.get("feature_columns") or []
    generator.target_column = bundle.get("target_column")
    generator._feature_stats = bundle.get("_feature_stats") or {}
    generator._categorical_distributions = bundle.get("_categorical_distributions") or {}
    if bundle.get("model_id") is not None:
        generator.model_id = bundle["model_id"]
    if bundle.get("metrics") is not None:
        generator.metrics = bundle["metrics"]
    return True
