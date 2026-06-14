"""E2I BentoML Model Serving Service.

Self-contained BentoML service for persistent model serving on the production
droplet. Runs as a standalone process via systemd — NO src.* imports.

Model discovery order:
  1. E2I_BENTOML_MODEL_TAG env var (exact tag, e.g. "tier0_abc123:v5")
  2. E2I_BENTOML_MODEL_NAME env var + ":latest"
  3. Auto-discover latest model from BentoML store
  4. Graceful "no model" mode (health returns degraded, predict returns error)

Framework auto-detection from model metadata:
  - sklearn, xgboost, lightgbm, or pickle fallback

Online feature retrieval:
  When ``PredictionInput.entity_ids`` and ``PredictionInput.feature_view`` are
  both supplied, the service fetches feature rows from the Feast online store
  over HTTP at ``FEAST_HTTP_ENDPOINT`` (falls back to ``FEAST_URL``, then
  ``http://feast:6566``) before running model inference. The legacy
  ``features: List[List[float]]`` path remains the default and is unchanged.

Version: 1.0.0
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import bentoml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# Feast HTTP endpoint resolution
# =============================================================================
#
# Precedence:
#   1. FEAST_HTTP_ENDPOINT (preferred — explicit serving endpoint)
#   2. FEAST_URL           (fallback — shared with src/* compose config)
#   3. "http://feast:6566" (default — Feast container in docker-compose)


def _resolve_feast_endpoint() -> str:
    """Return the Feast online-features HTTP base URL, with fallbacks."""
    return (
        os.environ.get("FEAST_HTTP_ENDPOINT") or os.environ.get("FEAST_URL") or "http://feast:6566"
    )


# =============================================================================
# Request/Response Models (matching mock_service.py contract)
# =============================================================================


class PredictionInput(BaseModel):
    """Input schema for prediction requests.

    Two mutually-supportive paths:
      1. Direct features:   ``features`` is a feature matrix (samples x features).
      2. Feast online:      ``entity_ids`` + ``feature_view`` triggers a Feast HTTP
         lookup before inference. ``features`` becomes optional in that case.

    When both are present the Feast lookup wins and ``features`` is ignored,
    matching the API-route contract that ``entity_id`` indicates a feature-store
    fetch is desired.
    """

    features: List[List[float]] = Field(
        default_factory=list,
        description=(
            "Feature matrix (samples x features). Optional when entity_ids + "
            "feature_view are provided — features will be fetched from Feast."
        ),
    )
    model_type: str = Field(
        default="classification",
        description="Type of prediction: classification or regression",
    )
    entity_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional entity IDs to look up in the Feast online store. When set "
            "alongside ``feature_view``, the service fetches features over HTTP "
            "before running inference. Ignored when ``feature_view`` is None."
        ),
    )
    feature_view: Optional[str] = Field(
        default=None,
        description=(
            "Optional Feast feature view name (e.g. 'patient_engagement_features'). "
            "Combined with ``entity_ids`` to fetch features over Feast HTTP. "
            "All features in the view are requested via the ':*' wildcard."
        ),
    )
    entity_key: str = Field(
        default="patient_id",
        description=(
            "Entity-key column name expected by the Feast feature view. "
            "Defaults to 'patient_id' to match the project's primary entity."
        ),
    )


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""

    predictions: List[float] = Field(
        ...,
        description="Model predictions",
    )
    probabilities: List[float] = Field(
        default_factory=list,
        description="Prediction probabilities (classification only)",
    )
    model_id: str = Field(
        default="unknown",
        description="Model identifier",
    )
    prediction_time_ms: float = Field(
        ...,
        description="Prediction time in milliseconds",
    )
    is_mock: bool = Field(
        default=False,
        description="Indicates this is a mock response",
    )
    feature_source: Optional[str] = Field(
        default=None,
        description=(
            "Telemetry tag describing where the features came from: "
            "'feast_online' when fetched via Feast HTTP, 'user_provided' when "
            "passed directly in the request, or None for batch/legacy paths."
        ),
    )


class BatchPredictionInput(BaseModel):
    """Input for batch predictions."""

    batch_id: str = Field(..., description="Unique batch identifier")
    features: List[List[float]] = Field(..., description="Feature matrix")


class BatchPredictionOutput(BaseModel):
    """Output for batch predictions."""

    batch_id: str
    total_samples: int
    predictions: List[float]
    processing_time_ms: float
    is_mock: bool = False


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = "e2i_model_service"
    version: str = "1.0.0"
    timestamp: str = ""
    uptime_seconds: float = 0.0
    model_loaded: bool = False
    model_tag: Optional[str] = None


# =============================================================================
# Model Discovery
# =============================================================================


def _discover_model() -> tuple[Optional[Any], Optional[str], Optional[str]]:
    """Discover and load a model from the BentoML store.

    Returns:
        (model_object, model_tag_string, framework_name) or (None, None, None)
    """

    # Strategy 1: Exact tag from env
    model_tag = os.environ.get("E2I_BENTOML_MODEL_TAG")
    if model_tag:
        logger.info("Using model tag from E2I_BENTOML_MODEL_TAG: %s", model_tag)
        return _load_model_by_tag(model_tag)

    # Strategy 2: Model name + :latest from env
    model_name = os.environ.get("E2I_BENTOML_MODEL_NAME")
    if model_name:
        tag = f"{model_name}:latest"
        logger.info("Using model name from E2I_BENTOML_MODEL_NAME: %s", tag)
        return _load_model_by_tag(tag)

    # Strategy 3: Auto-discover latest model from store
    try:
        models = bentoml.models.list()
        if models:
            # Sort by creation time (newest first) — models.list() returns
            # BentoModel objects with a creation_time attribute
            sorted_models = sorted(
                models,
                key=lambda m: getattr(m, "creation_time", ""),
                reverse=True,
            )
            latest = sorted_models[0]
            tag_str = str(latest.tag)
            logger.info("Auto-discovered latest model: %s", tag_str)
            return _load_model_by_tag(tag_str)
        else:
            logger.warning("No models found in BentoML store")
    except Exception as e:
        logger.warning("Failed to list models from BentoML store: %s", e)

    # Strategy 4: No model — graceful degraded mode
    logger.warning("No model available — running in degraded mode")
    return None, None, None


def _load_model_by_tag(tag: str) -> tuple[Optional[Any], Optional[str], Optional[str]]:
    """Load a model by its BentoML tag string.

    Tries framework-specific loaders, falls back to pickle.

    Returns:
        (model_object, tag_string, framework_name)
    """
    framework_loaders = [
        ("sklearn", "bentoml.sklearn"),
        ("xgboost", "bentoml.xgboost"),
        ("lightgbm", "bentoml.lightgbm"),
    ]

    # Try to read model metadata to detect framework
    try:
        bento_model = bentoml.models.get(tag)
        meta = bento_model.info.metadata or {}
        stored_framework = meta.get("framework", "").lower()
        if stored_framework:
            # Reorder loaders to try the stored framework first
            framework_loaders = sorted(
                framework_loaders,
                key=lambda fl: (0 if fl[0] == stored_framework else 1),
            )
    except Exception:
        pass

    # Try each framework loader
    for framework_name, module_path in framework_loaders:
        try:
            mod = __import__(module_path, fromlist=[module_path.split(".")[-1]])
            model = mod.load_model(tag)
            logger.info("Loaded model %s with %s loader", tag, framework_name)
            return model, tag, framework_name
        except (bentoml.exceptions.NotFound, Exception):
            continue

    # Pickle fallback
    try:
        import pickle

        bento_model = bentoml.models.get(tag)
        model_path = bento_model.path
        # Look for common pickle filenames
        for pkl_name in ["saved_model.pkl", "model.pkl", "model.joblib"]:
            pkl_path = os.path.join(model_path, pkl_name)
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    model = pickle.load(f)  # noqa: S301
                logger.info("Loaded model %s via pickle (%s)", tag, pkl_name)
                return model, tag, "pickle"
    except Exception as e:
        logger.error("Pickle fallback failed for %s: %s", tag, e)

    logger.error("Failed to load model: %s", tag)
    return None, None, None


# =============================================================================
# BentoML Service
# =============================================================================

SERVICE_START_TIME = time.time()


@bentoml.service(
    name="e2i_model_service",
    resources={"cpu": "1", "memory": "2Gi"},
    traffic={"timeout": 60},
)
class E2IModelService:
    """Production BentoML service for E2I trained models.

    Provides persistent model serving with automatic model discovery
    and framework detection. Matches the mock_service.py API contract.
    """

    def __init__(self):
        """Initialize service and load model."""
        self._start_time = SERVICE_START_TIME
        self._prediction_count = 0
        self._model = None
        self._model_tag: Optional[str] = None
        self._framework: Optional[str] = None
        self._preprocessor = None
        self._feature_columns: Optional[List[str]] = None

        self._model, self._model_tag, self._framework = _discover_model()

        # Unwrap bundled dict if model is a dict (contains preprocessor)
        if isinstance(self._model, dict):
            self._preprocessor = self._model.get("preprocessor")
            self._feature_columns = self._model.get("feature_columns")
            self._model = self._model.get("model")
            if self._preprocessor is not None:
                logger.info("Unwrapped bundled model with preprocessor")

        if self._model is not None:
            logger.info(
                "E2I Model Service initialized: tag=%s framework=%s",
                self._model_tag,
                self._framework,
            )
        else:
            logger.warning("E2I Model Service initialized in degraded mode (no model)")

    def _resolve_feature_columns(self) -> Optional[List[str]]:
        """Resolve the model's authoritative ordered feature names.

        Order of preference:
          1. The bundled ``feature_columns`` (the preprocessor input order) —
             this is what ``_run_prediction`` uses to build the DataFrame fed
             to the ColumnTransformer.
          2. The estimator's ``feature_names_in_`` (set by scikit-learn when
             the model was fit on a named DataFrame).

        Returns ``None`` when neither is available so callers fail closed
        instead of guessing a positional order.
        """
        if self._feature_columns:
            return list(self._feature_columns)
        names = getattr(self._model, "feature_names_in_", None)
        if names is not None:
            try:
                return [str(n) for n in names]
            except TypeError:
                return None
        return None

    def _apply_preprocessor(self, arr: Any) -> Any:
        """Apply the bundled preprocessor to a feature matrix, if present.

        Builds a named DataFrame (using ``feature_columns``) when the column
        count matches so a ColumnTransformer/preprocessor receives the named
        features it was fit on; otherwise transforms the raw array. Used by BOTH
        single and batch prediction so batch inference is not run on raw,
        un-preprocessed rows (which would error or silently mis-predict for
        bundled models with a preprocessor).
        """
        if self._preprocessor is None:
            return arr
        try:
            import pandas as pd

            if self._feature_columns and len(self._feature_columns) == arr.shape[1]:
                df = pd.DataFrame(arr, columns=self._feature_columns)
                return self._preprocessor.transform(df)
            return self._preprocessor.transform(arr)
        except Exception as e:
            logger.warning("Preprocessor transform failed, using raw input: %s", e)
            return arr

    def _run_prediction(
        self,
        features: List[List[float]],
        feature_source: Optional[str] = None,
    ) -> PredictionOutput:
        """Run prediction using the loaded model.

        Args:
            features: Feature matrix
            feature_source: Optional telemetry tag describing feature origin
                ('feast_online' | 'user_provided' | None).

        Returns:
            Prediction output
        """
        import numpy as np

        if self._model is None:
            return PredictionOutput(
                predictions=[],
                probabilities=[],
                model_id="no_model",
                prediction_time_ms=0.0,
                is_mock=False,
                feature_source=feature_source,
            )

        start = time.time()
        arr = self._apply_preprocessor(np.array(features))

        predictions = self._model.predict(arr).tolist()

        probabilities = []
        if hasattr(self._model, "predict_proba"):
            try:
                proba = self._model.predict_proba(arr)
                # Return probability of positive class for binary classification
                if proba.ndim == 2 and proba.shape[1] == 2:
                    probabilities = proba[:, 1].tolist()
                else:
                    probabilities = proba.tolist()
            except Exception:
                pass

        elapsed_ms = (time.time() - start) * 1000
        self._prediction_count += len(features)

        return PredictionOutput(
            predictions=predictions,
            probabilities=probabilities,
            model_id=self._model_tag or "unknown",
            prediction_time_ms=elapsed_ms,
            is_mock=False,
            feature_source=feature_source,
        )

    async def _fetch_features_from_feast(
        self,
        entity_ids: List[str],
        feature_view: str,
        entity_key: str,
    ) -> List[List[float]]:
        """Fetch a feature matrix from the Feast online store over HTTP.

        Calls Feast's standard ``POST /get-online-features`` endpoint and
        reshapes the per-feature column response into a row-major matrix
        ordered to match ``entity_ids``.

        Args:
            entity_ids: List of entity IDs to look up.
            feature_view: Feast feature view name. All features in the view
                are requested via ``"<feature_view>:*"``.
            entity_key: Entity-key column name expected by the feature view.

        Returns:
            Feature matrix (samples x features). Numeric coercion is applied;
            non-numeric / missing values become 0.0 so downstream model.predict
            does not crash on dtype.

        Raises:
            RuntimeError: If httpx is unavailable, the Feast call fails, or
                the response payload is malformed. The caller surfaces this
                back to the client rather than silently filling zeros.
        """
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - container always installs httpx
            raise RuntimeError(
                "httpx is required for Feast HTTP feature fetch but is not installed"
            ) from e

        endpoint = _resolve_feast_endpoint().rstrip("/")
        url = f"{endpoint}/get-online-features"
        payload = {
            "features": [f"{feature_view}:*"],
            "entities": {entity_key: list(entity_ids)},
            "full_feature_names": False,
        }

        logger.info(
            "Fetching %d entities from Feast view '%s' via %s",
            len(entity_ids),
            feature_view,
            url,
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception as e:
            raise RuntimeError(f"Feast online-features call failed ({url}): {e}") from e

        # Feast 0.43 response shape (column-oriented):
        #   {"metadata": {"feature_names": [...]},
        #    "results":  [{"values": [...], "statuses": [...], ...}, ...]}
        metadata = body.get("metadata", {})
        feature_names: List[str] = list(metadata.get("feature_names", []))
        results = body.get("results", [])

        if not feature_names or not results:
            raise RuntimeError(
                f"Feast response missing feature_names/results: keys={list(body.keys())}"
            )

        # Drop the entity-key column from feature columns; we want feature values only.
        feature_columns = {
            name: results[idx].get("values", [])
            for idx, name in enumerate(feature_names)
            if name != entity_key
        }

        n_rows = len(entity_ids)
        matrix: List[List[float]] = []
        # Track which features had any null/non-numeric values coerced to 0.0
        # so we can emit ONE aggregate WARNING per request rather than spamming
        # one log line per coerced value (3A-I-1).
        coerced_features: Dict[str, int] = {}
        for row_idx in range(n_rows):
            row: List[float] = []
            for col_name, col_values in feature_columns.items():
                raw = col_values[row_idx] if row_idx < len(col_values) else None
                if raw is None:
                    row.append(0.0)
                    coerced_features[col_name] = coerced_features.get(col_name, 0) + 1
                    continue
                try:
                    row.append(float(raw))
                except (TypeError, ValueError):
                    row.append(0.0)
                    coerced_features[col_name] = coerced_features.get(col_name, 0) + 1
            matrix.append(row)

        if coerced_features:
            total_coerced = sum(coerced_features.values())
            logger.warning(
                "Feast online-features call returned %d null/non-numeric "
                "values across %d feature(s); coerced to 0.0. Feature names: "
                "%s. Investigate Feast feature view configuration if this "
                "persists across requests.",
                total_coerced,
                len(coerced_features),
                sorted(coerced_features.keys()),
            )

        return matrix

    @bentoml.api
    async def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Run prediction on input features.

        Two paths:
          - ``entity_ids`` + ``feature_view`` set → fetch features from the
            Feast online store over HTTP, tag ``feature_source='feast_online'``.
          - Otherwise → use the supplied ``features`` matrix directly,
            tag ``feature_source='user_provided'`` (None when matrix is empty).

        Args:
            input_data: Features and configuration

        Returns:
            Model predictions with a ``feature_source`` telemetry tag.
        """
        if input_data.entity_ids and input_data.feature_view:
            features = await self._fetch_features_from_feast(
                entity_ids=input_data.entity_ids,
                feature_view=input_data.feature_view,
                entity_key=input_data.entity_key,
            )
            return self._run_prediction(features, feature_source="feast_online")

        feature_source = "user_provided" if input_data.features else None
        return self._run_prediction(input_data.features, feature_source=feature_source)

    @bentoml.api
    async def predict_batch(self, input_data: BatchPredictionInput) -> BatchPredictionOutput:
        """Run batch predictions.

        Args:
            input_data: Batch of features

        Returns:
            Batch predictions
        """
        start = time.time()

        if self._model is None:
            return BatchPredictionOutput(
                batch_id=input_data.batch_id,
                total_samples=len(input_data.features),
                predictions=[],
                processing_time_ms=0.0,
            )

        import numpy as np

        # Apply the bundled preprocessor (same path as single predict) so batch
        # inference is not run on raw, un-preprocessed rows.
        arr = self._apply_preprocessor(np.array(input_data.features))
        predictions = self._model.predict(arr).tolist()
        elapsed_ms = (time.time() - start) * 1000
        self._prediction_count += len(input_data.features)

        return BatchPredictionOutput(
            batch_id=input_data.batch_id,
            total_samples=len(input_data.features),
            predictions=predictions,
            processing_time_ms=elapsed_ms,
        )

    @bentoml.api
    async def health(self) -> HealthResponse:
        """Health check endpoint.

        Returns:
            Service health status
        """
        uptime = time.time() - self._start_time
        status = "healthy" if self._model is not None else "degraded"

        return HealthResponse(
            status=status,
            service="e2i_model_service",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=uptime,
            model_loaded=self._model is not None,
            model_tag=self._model_tag,
        )

    @bentoml.api
    async def metrics(self) -> Dict[str, Any]:
        """Return service metrics.

        Returns:
            Service metrics
        """
        uptime = time.time() - self._start_time

        return {
            "prediction_count": self._prediction_count,
            "uptime_seconds": uptime,
            "model_tag": self._model_tag,
            "framework": self._framework,
            "model_loaded": self._model is not None,
            "service": "e2i_model_service",
            "is_mock": False,
        }

    @bentoml.api
    async def model_info(self) -> Dict[str, Any]:
        """Return model information.

        Returns:
            Model metadata
        """
        info: Dict[str, Any] = {
            "model_id": self._model_tag or "no_model",
            "model_type": self._framework or "none",
            "framework": self._framework or "none",
            "version": "1.0.0",
            "is_mock": False,
            "model_loaded": self._model is not None,
            "supported_endpoints": [
                "/predict",
                "/predict_batch",
                "/health",
                "/metrics",
                "/model_info",
            ],
            # Expose the model's authoritative feature ORDER so callers can
            # vectorize a feature dict into the positional ``features`` matrix
            # the model expects. Prefer the bundled ``feature_columns`` (the
            # ColumnTransformer/preprocessor input order); fall back to the
            # estimator's own ``feature_names_in_`` (set when fit on a
            # DataFrame). Omitted (None) when the model carries no named
            # feature contract — callers MUST then fail closed rather than
            # guess an order.
            "feature_columns": self._resolve_feature_columns(),
        }

        # Add model metadata if available
        if self._model_tag:
            try:
                bento_model = bentoml.models.get(self._model_tag)
                meta = bento_model.info.metadata or {}
                info["metadata"] = meta
            except Exception:
                pass

        return info
