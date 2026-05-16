"""In-process prediction client + deployment-manifest loader (Phase 3 / G5).

Promoted from ``scripts/run_tier1_5_test.py::Tier0ModelClient`` so the
prediction_synthesizer agent has a production-side path to a real model
without depending on the test harness.

The deployment-manifest loader is the bridge between
``model_deployer`` (which emits a ``deployment_manifest``) and the
agent ``model_clients={...}`` injection slot. It is intentionally
tolerant of multiple manifest shapes:

* ``{"model_uri": "<path-or-uri>", "metadata": {"name": "<id>"}}`` — flat
  single-model manifest.
* ``{"spec": {"models": {"<id>": {"model_uri": "<uri>"}}}}`` — multi-model.

URIs are resolved as:
* a local filesystem path → ``pickle.load``
* ``file://`` prefixed → strip and pickle.load
* anything else (``runs:/``, ``models:/``, ``s3://`` …) → ``mlflow.pyfunc.load_model``
  (skipped silently if MLflow is unavailable).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


__all__ = [
    "InProcessModelClient",
    "load_clients_from_deployment_manifest",
    "load_clients_from_deployment_manifest_file",
]


class InProcessModelClient:
    """Wraps a fitted sklearn-style model as an async prediction client.

    Conforms to the ``ModelClient`` protocol expected by
    ``prediction_synthesizer`` (see ``clients/factory.py``).
    """

    def __init__(
        self,
        model: Any,
        model_id: str = "inproc_model",
        feature_names: Optional[List[str]] = None,
        model_type: str = "logistic_regression",
    ) -> None:
        self.model = model
        self.model_id = model_id
        self.model_type = model_type
        if hasattr(model, "feature_names_in_"):
            self.feature_names: List[str] = list(model.feature_names_in_)
        else:
            self.feature_names = list(feature_names or [])

    async def predict(
        self,
        entity_id: str,
        features: Dict[str, Any],
        time_horizon: str,
    ) -> Dict[str, Any]:
        import numpy as np

        start = time.time()
        try:
            if self.feature_names:
                feature_values = [features.get(name, 0.0) for name in self.feature_names]
            else:
                feature_values = list(features.values())

            X = np.array([feature_values], dtype=float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            proba_list: Optional[List[float]]
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                proba_list = [float(p) for p in proba]
                prediction = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                prediction = float(self.model.predict(X)[0])
                proba_list = None

            confidence = abs(prediction - 0.5) * 2
            return {
                "model_id": self.model_id,
                "model_type": self.model_type,
                "prediction": prediction,
                "proba": proba_list,
                "confidence": max(0.5, min(1.0, confidence + 0.5)),
                "latency_ms": int((time.time() - start) * 1000),
                "features_used": self.feature_names or list(features.keys()),
            }
        except Exception as exc:  # noqa: BLE001 — surfaced via "error" field
            logger.warning("InProcessModelClient(%s) predict failed: %s", self.model_id, exc)
            return {
                "model_id": self.model_id,
                "model_type": self.model_type,
                "prediction": 0.5,
                "proba": None,
                "confidence": 0.3,
                "latency_ms": int((time.time() - start) * 1000),
                "features_used": self.feature_names or [],
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------


def _load_model_from_uri(uri: str) -> Any:
    """Load a model from a URI. Raises on any failure (callers wrap)."""
    if uri.startswith("file://"):
        uri = uri[len("file://") :]

    # MLflow URIs delegate to mlflow.pyfunc.
    if uri.startswith(("runs:/", "models:/", "s3://", "gs://")):
        try:
            import mlflow.pyfunc  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"MLflow URI {uri!r} cannot be loaded — install mlflow") from exc
        return mlflow.pyfunc.load_model(uri)

    # Local pickle path.
    path = Path(uri)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {uri}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def _iter_models_in_manifest(manifest: Dict[str, Any]) -> List[tuple[str, str]]:
    """Return [(model_id, model_uri), …] from a manifest, supporting both shapes."""
    out: List[tuple[str, str]] = []

    spec = manifest.get("spec") or {}
    models = spec.get("models") if isinstance(spec, dict) else None
    if isinstance(models, dict):
        for model_id, entry in models.items():
            if isinstance(entry, dict) and entry.get("model_uri"):
                out.append((str(model_id), str(entry["model_uri"])))

    if not out and manifest.get("model_uri"):
        # Flat / single-model shape — derive id from metadata.name (or "default").
        meta = manifest.get("metadata") or {}
        model_id = str(meta.get("name") or "default")
        out.append((model_id, str(manifest["model_uri"])))

    return out


def load_clients_from_deployment_manifest(
    manifest: Dict[str, Any],
) -> Dict[str, InProcessModelClient]:
    """Best-effort: load model clients from a deployment manifest dict.

    Tolerates partial failures — a single bad URI does not poison the dict.
    Returns ``{}`` when no models are configured.
    """
    if not isinstance(manifest, dict) or not manifest:
        return {}

    clients: Dict[str, InProcessModelClient] = {}
    for model_id, uri in _iter_models_in_manifest(manifest):
        try:
            model = _load_model_from_uri(uri)
        except Exception as exc:  # noqa: BLE001 — log and skip
            logger.warning("Skipping model %r from manifest (uri=%s): %s", model_id, uri, exc)
            continue
        clients[model_id] = InProcessModelClient(model=model, model_id=model_id)

    return clients


def load_clients_from_deployment_manifest_file(
    manifest_path: os.PathLike[str] | str,
) -> Dict[str, InProcessModelClient]:
    """Convenience: read a JSON manifest from disk, then load clients."""
    path = Path(manifest_path)
    if not path.exists():
        logger.info("Deployment manifest not found at %s", path)
        return {}
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse manifest %s: %s", path, exc)
        return {}
    return load_clients_from_deployment_manifest(manifest)
