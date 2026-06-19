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
    raw_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "RAW covariate rows (samples) as {name: value} dicts — e.g. "
            "{'disease_severity': 5.61, 'academic_hcp': 0, "
            "'geographic_region': 'northeast'}. Used for gold-standard cohort "
            "models whose bundled preprocessor is a FeatureBuilder: the service "
            "applies preprocessor.transform(raw_df) (raw -> encoded) before "
            "inference. Categorical strings (e.g. geographic_region) are "
            "one-hot-encoded by the preprocessor, NOT coerced to float. Takes "
            "precedence over ``features`` when both are present; ignored on the "
            "Feast path."
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
    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional serving model name to ROUTE this request to (#39 "
            "multi-model serving). When set and known, the service uses the "
            "bundle registered under this name (e.g. "
            "'initiation_remibrutinib_goldstd_lr_v1', "
            "'hcp_adoption_kisqali_goldstd_lr_v1'). When None, the legacy "
            "single default model is used (tier0 / numeric / Feast contracts — "
            "backward compatible). An unknown name FAILS CLOSED (error in the "
            "response) rather than silently scoring the wrong/default model."
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
    encoded_features: List[List[float]] = Field(
        default_factory=list,
        description=(
            "The ENCODED numeric feature matrix the model actually scored "
            "(samples x len(encoded_feature_columns)). Populated on the RAW "
            "covariate path (#39) after the bundled FeatureBuilder transforms "
            "raw -> encoded, so the SHAP caller can run SHAP over the audit-grade "
            "encoded vector. Empty on the legacy/Feast paths (the caller already "
            "holds the encoded matrix)."
        ),
    )
    encoded_feature_columns: List[str] = Field(
        default_factory=list,
        description=(
            "Names for ``encoded_features`` columns (the model's encoded "
            "feature_columns order). Empty when ``encoded_features`` is empty."
        ),
    )
    error: Optional[str] = Field(
        default=None,
        description=(
            "Fail-closed error message (#39 multi-model): set when a routed "
            "``model_name`` is unknown so the caller sees an honest failure "
            "instead of a fabricated/wrong-model prediction. None on success."
        ),
    )


class BatchPredictionInput(BaseModel):
    """Input for batch predictions.

    Two paths (mirroring ``PredictionInput``):
      1. Legacy numeric: ``features`` is a PRE-ENCODED matrix (samples x features).
      2. RAW covariates (#cohort-scoring): ``raw_features`` is a list of
         ``{name: value}`` dicts; the service encodes each via the bundled
         FeatureBuilder and returns PER-ROW probabilities. ``model_name`` routes
         to a specific gold-standard bundle (#39). This lets the API score a whole
         holdout cohort in ONE chunked call instead of N single-predict round-trips.
    """

    batch_id: str = Field(..., description="Unique batch identifier")
    features: List[List[float]] = Field(
        default_factory=list,
        description=(
            "Pre-encoded feature matrix (samples x features). Optional when "
            "``raw_features`` is supplied."
        ),
    )
    raw_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "RAW covariate rows as {name: value} dicts (e.g. {'disease_severity': "
            "5.6, 'academic_hcp': 0, 'geographic_region': 'northeast'}). Encoded "
            "server-side via the bundled FeatureBuilder. Takes precedence over "
            "``features`` when present."
        ),
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional serving model name to ROUTE this batch to (#39 multi-model).",
    )


class BatchPredictionOutput(BaseModel):
    """Output for batch predictions."""

    batch_id: str
    total_samples: int
    predictions: List[float]
    probabilities: List[float] = Field(
        default_factory=list,
        description=(
            "Per-row positive-class probabilities (classification; populated on the "
            "raw-covariate path). Empty on the legacy numeric path."
        ),
    )
    processing_time_ms: float
    is_mock: bool = False
    error: Optional[str] = Field(
        default=None,
        description=(
            "Set when the batch failed closed (e.g. unknown model_name); "
            "predictions/probabilities are empty in that case."
        ),
    )


class ModelInfoInput(BaseModel):
    """Input for ``/model_info`` — optionally routed by serving model name (#39).

    When ``model_name`` is set and known, ``/model_info`` returns that bundle's
    contract (its ``keep_columns`` + encoded ``feature_columns``). When None, the
    legacy default model's info is returned (backward compatible — the legacy
    caller posts ``{}``).
    """

    model_name: Optional[str] = Field(
        default=None,
        description="Optional serving model name to describe (multi-model #39).",
    )


class ShapInput(BaseModel):
    """Input for ``/shap`` — gold-standard SHAP over the ENCODED vector (#39).

    Mirrors the ``/predict`` raw-covariate path: a routed ``model_name`` plus
    ``raw_features`` (RAW covariate rows). The service encodes the raw row via the
    bundled FeatureBuilder and runs the SHAP explainer over the ENCODED numeric
    vector the model actually scored — the API process cannot do this itself
    because the bundle lives only in this container's mount.
    """

    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Serving model name to ROUTE this SHAP request to (#39 multi-model). "
            "When set and known, the matching gold-standard bundle (model + "
            "FeatureBuilder) is used. When None, the legacy default model is used. "
            "An unknown name FAILS CLOSED (error in the response)."
        ),
    )
    raw_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "RAW covariate rows (samples) as {name: value} dicts (e.g. "
            "{'disease_severity': 5.61, 'academic_hcp': 0, "
            "'geographic_region': 'northeast'}). The bundled FeatureBuilder "
            "encodes them before SHAP. SHAP runs on the first row."
        ),
    )
    top_k: Optional[int] = Field(
        default=None,
        description=(
            "Optional cap on the number of returned per-feature SHAP values "
            "(by descending |shap|). None returns all encoded features."
        ),
    )


class ShapOutput(BaseModel):
    """Output for ``/shap`` — real per-encoded-feature SHAP values (#39).

    Fail-closed: ``error`` is set (and the value maps are empty) when the
    ``model_name`` is unknown, no FeatureBuilder is bundled, or SHAP cannot be
    computed — NEVER fabricated SHAP values for an audit-grade explanation.
    """

    shap_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Signed per-ENCODED-feature SHAP contributions {encoded_name: value}.",
    )
    base_value: float = Field(
        default=0.0,
        description="The explainer's expected value (base margin) for the encoded model.",
    )
    encoded_feature_columns: List[str] = Field(
        default_factory=list,
        description="The encoded feature order SHAP ran over (the bundle's feature_columns).",
    )
    encoded_features: List[float] = Field(
        default_factory=list,
        description="The single ENCODED row SHAP was computed for (audit trail).",
    )
    explainer_type: str = Field(
        default="LinearExplainer",
        description="The SHAP explainer used (LinearExplainer over the inner linear estimator).",
    )
    model_id: str = Field(
        default="unknown",
        description="The routed model tag/name the SHAP was computed for.",
    )
    computation_time_ms: float = Field(
        default=0.0,
        description="SHAP computation time in milliseconds.",
    )
    error: Optional[str] = Field(
        default=None,
        description=(
            "Fail-closed error message: set when SHAP could not be computed "
            "(unknown model_name, no FeatureBuilder, explainer failure). None on "
            "success. The caller MUST treat a non-None error as a hard failure "
            "and NOT present fabricated SHAP."
        ),
    )


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
# Multi-model discovery (#39) — gold-standard cohort bundles
# =============================================================================
#
# The service serves MANY gold-standard cohort models from ONE process (12+
# separate containers is infeasible on a memory-constrained box). Each bundle is
# a small dict {"model", "preprocessor": FeatureBuilder, "feature_columns"}.
# Discovery order per model:
#   1. BentoML store — any model whose tag name matches *_goldstd_lr_v1
#      (the LIVE ACTIVATION step imports each bundle as a picklable model with
#      metadata.bundled=True). Preferred: same store as the legacy path.
#   2. Filesystem fallback — data/ml_artifacts/shap_serving/<cohort>/<name>.bundle.pkl
#      (what scripts/rematerialize_goldstd_bundles.py writes).
# Eager-loading ~12 calibrated-LR bundles is trivially cheap; we still guard with
# a cap + a warning if the discovered set is unexpectedly large.

# Sentinel so resolver helpers can distinguish "use the legacy default" (arg
# omitted) from an explicitly-passed None (a routed model with no preprocessor).
_UNSET: Any = object()

_GOLDSTD_NAME_SUFFIX = "_goldstd_lr_v1"
_SHAP_SERVING_DIRNAME = os.path.join("data", "ml_artifacts", "shap_serving")
_MAX_EAGER_MODELS = 64  # sanity cap; warn (don't crash) if exceeded


def _is_goldstd_bundle_dict(obj: Any) -> bool:
    """True for the {"model","preprocessor","feature_columns"} bundle shape."""
    return (
        isinstance(obj, dict)
        and "model" in obj
        and "preprocessor" in obj
        and "feature_columns" in obj
    )


def _unwrap_bundle(obj: Any) -> Optional[Dict[str, Any]]:
    """Return a normalized {model, preprocessor, feature_columns} entry or None."""
    if not _is_goldstd_bundle_dict(obj):
        return None
    return {
        "model": obj.get("model"),
        "preprocessor": obj.get("preprocessor"),
        "feature_columns": obj.get("feature_columns"),
    }


def _discover_goldstd_bundles_from_store() -> Dict[str, Dict[str, Any]]:
    """Load every *_goldstd_lr_v1 bundle from the BentoML store, keyed by name.

    The serving name is the registry model_name (the tag's name component, e.g.
    ``initiation_remibrutinib_goldstd_lr_v1``). Best-effort: store/loader errors
    for one model are logged and skipped (one bad bundle must not sink the
    others). Only dict-shaped bundles are accepted.
    """
    found: Dict[str, Dict[str, Any]] = {}
    try:
        models = bentoml.models.list()
    except Exception as e:  # pragma: no cover - store unavailable in unit env
        logger.warning("Multi-model discovery: bentoml.models.list() failed: %s", e)
        return found

    for m in models:
        try:
            tag_str = str(m.tag)
            name = tag_str.split(":", 1)[0]
            if not name.endswith(_GOLDSTD_NAME_SUFFIX):
                continue
            loaded, _tag, _fw = _load_model_by_tag(tag_str)
            entry = _unwrap_bundle(loaded)
            if entry is None:
                logger.warning("Multi-model discovery: %s is not a bundle dict; skipping.", tag_str)
                continue
            # Newest tag wins (models.list() order is not guaranteed); keep first
            # seen per name after sorting by creation_time desc.
            found.setdefault(name, entry)
        except Exception as e:  # pragma: no cover
            logger.warning("Multi-model discovery: failed to load %s: %s", m, e)
    return found


def _discover_goldstd_bundles_from_fs(root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load *_goldstd_lr_v1 bundles from the shap_serving artifact dir.

    Fallback for when the bundles are on disk but not (yet) imported to the
    BentoML store. Walks ``data/ml_artifacts/shap_serving/<cohort>/<name>.bundle.pkl``.
    """
    found: Dict[str, Dict[str, Any]] = {}
    base = root or _SHAP_SERVING_DIRNAME
    if not os.path.isdir(base):
        return found
    import pickle as _pickle

    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            if not fn.endswith(".bundle.pkl"):
                continue
            name = fn[: -len(".bundle.pkl")]
            if not name.endswith(_GOLDSTD_NAME_SUFFIX):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "rb") as fh:
                    obj = _pickle.load(fh)  # noqa: S301 - trusted local artifact
                entry = _unwrap_bundle(obj)
                if entry is None:
                    logger.warning("FS discovery: %s is not a bundle dict; skipping.", path)
                    continue
                found.setdefault(name, entry)
            except Exception as e:
                logger.warning("FS discovery: failed to load %s: %s", path, e)
    return found


def _discover_goldstd_bundles() -> Dict[str, Dict[str, Any]]:
    """Discover all gold-standard serving bundles (store first, FS fallback)."""
    models = _discover_goldstd_bundles_from_store()
    # FS fallback fills in any names the store does not have (does not override
    # a store-loaded entry).
    for name, entry in _discover_goldstd_bundles_from_fs().items():
        models.setdefault(name, entry)
    if len(models) > _MAX_EAGER_MODELS:
        logger.warning(
            "Multi-model discovery loaded %d models (> cap %d); memory pressure "
            "possible — consider sharding the serving set.",
            len(models),
            _MAX_EAGER_MODELS,
        )
    if models:
        logger.info("Multi-model serving: loaded %d gold-standard bundles", len(models))
    return models


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
        # Multi-model registry (#39): {serving_name: {model, preprocessor,
        # feature_columns}}. Empty when no gold-standard bundles are present —
        # the legacy single default model below is then the only servable model.
        self._models: Dict[str, Dict[str, Any]] = _discover_goldstd_bundles()

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

    def _resolve_active(
        self, model_name: Optional[str]
    ) -> tuple[Any, Any, Optional[List[str]], str, Optional[str]]:
        """Resolve the (model, preprocessor, feature_columns, tag, error) to use.

        Routing (#39 multi-model):
          * ``model_name`` set AND in the registry → that bundle's components.
          * ``model_name`` None/empty → the LEGACY single default (self._model*),
            tag self._model_tag (backward compatible).
          * ``model_name`` set but UNKNOWN → ``error`` is non-None so callers FAIL
            CLOSED instead of silently scoring the default/wrong model.
        """
        if model_name:
            entry = self._models.get(model_name)
            if entry is None:
                return (None, None, None, model_name, f"Unknown model_name: {model_name}")
            return (
                entry.get("model"),
                entry.get("preprocessor"),
                entry.get("feature_columns"),
                model_name,
                None,
            )
        return (self._model, self._preprocessor, self._feature_columns, self._model_tag, None)

    def _resolve_feature_columns(
        self, model: Any = _UNSET, feature_columns: Any = _UNSET
    ) -> Optional[List[str]]:
        """Resolve the active model's authoritative ordered feature names.

        Order of preference:
          1. The bundled ``feature_columns`` (the preprocessor input order).
          2. The estimator's ``feature_names_in_`` (set when fit on a DataFrame).

        Defaults to the legacy default model when no explicit model/feature
        columns are passed (backward compatible). Returns ``None`` when neither
        is available so callers fail closed instead of guessing a positional
        order.
        """
        model = self._model if model is _UNSET else model
        feature_columns = self._feature_columns if feature_columns is _UNSET else feature_columns
        if feature_columns:
            return list(feature_columns)
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            try:
                return [str(n) for n in names]
            except TypeError:
                return None
        return None

    def _resolve_keep_columns(self, preprocessor: Any = _UNSET) -> Optional[List[str]]:
        """Resolve the RAW covariate names a caller must supply, if any.

        Returns the bundled preprocessor's ``keep_columns`` when the preprocessor
        is a FeatureBuilder (the gold-standard #39 path) — these are the RAW
        covariate names (e.g. ``disease_severity``, ``academic_hcp``,
        ``geographic_region``) the caller supplies, which the FeatureBuilder
        encodes into the numeric ``feature_columns`` SHAP runs over.

        Returns ``None`` for a bare estimator / ColumnTransformer (no raw
        covariate contract — the caller supplies the encoded matrix directly).
        Best-effort: any unexpected shape yields ``None`` so model_info never
        raises.
        """
        pre = self._preprocessor if preprocessor is _UNSET else preprocessor
        if pre is None:
            return None
        keep = getattr(pre, "keep_columns", None)
        if not keep:
            return None
        try:
            return [str(c) for c in keep]
        except TypeError:
            return None

    @staticmethod
    def _is_feature_builder(preprocessor: Any) -> bool:
        """True when ``preprocessor`` is a FeatureBuilder (raw-covariate contract):
        it exposes both ``keep_columns`` and a ``transform`` method.

        Detected structurally (duck-typed) rather than by import — the BentoML
        container deliberately cannot import ``src.*``.
        """
        return (
            preprocessor is not None
            and getattr(preprocessor, "keep_columns", None) is not None
            and callable(getattr(preprocessor, "transform", None))
        )

    def _preprocessor_is_feature_builder(self) -> bool:
        """Legacy-default convenience: is the default model's preprocessor a
        FeatureBuilder? (Preserved for any existing caller / test.)"""
        return self._is_feature_builder(self._preprocessor)

    def _run_raw_prediction(
        self,
        raw_rows: List[Dict[str, Any]],
        *,
        model: Any = _UNSET,
        preprocessor: Any = _UNSET,
        feature_columns: Any = _UNSET,
        model_tag: Any = _UNSET,
    ) -> PredictionOutput:
        """Encode RAW covariate rows via the FeatureBuilder, then predict.

        Builds a raw DataFrame (preserving each covariate's native dtype so a
        categorical string is one-hot-encoded, not float-coerced), applies
        ``preprocessor.transform`` (raw -> encoded), and runs the model. FAILS
        CLOSED if no FeatureBuilder preprocessor is bundled — a raw request has
        no meaning without one, and silently treating dicts as a numeric matrix
        would fabricate a prediction.

        Operates on the ROUTED model components (#39 multi-model) when passed;
        defaults to the legacy default model. Grain-agnostic: a covariate is a
        category iff the request value is a string (so the HCP cohort's
        ``specialty`` + ``geographic_region`` and the patient cohort's
        ``geographic_region`` all one-hot correctly) — there is NO hardcoded
        categorical name.
        """
        import numpy as np
        import pandas as pd

        model = self._model if model is _UNSET else model
        preprocessor = self._preprocessor if preprocessor is _UNSET else preprocessor
        feature_columns = self._feature_columns if feature_columns is _UNSET else feature_columns
        model_tag = self._model_tag if model_tag is _UNSET else model_tag

        if model is None:
            return PredictionOutput(
                predictions=[],
                probabilities=[],
                model_id="no_model",
                prediction_time_ms=0.0,
                is_mock=False,
                feature_source="raw_covariates",
            )

        if not self._is_feature_builder(preprocessor):
            raise RuntimeError(
                "raw_features supplied but the served model has no FeatureBuilder "
                "preprocessor; refusing to fabricate a prediction from raw "
                "covariates without an encoder."
            )

        required_columns = self._resolve_keep_columns(preprocessor) or []
        # The fitted FeatureBuilder is the AUTHORITY on which covariates are
        # numeric vs categorical — it stores a learned median per numeric column
        # in ``_numeric_medians`` and one-hot-encodes everything else. We use
        # that to validate per-column type GRAIN-AGNOSTICALLY (patient:
        # geographic_region categorical; HCP: specialty + geographic_region
        # categorical) WITHOUT a hardcoded name, while keeping codex's
        # fail-closed intent: a string in a NUMERIC column is a malformed request
        # (it would be reindexed to a fabricated 0.0), not a category.
        numeric_cols = set(getattr(preprocessor, "_numeric_medians", {}) or {})
        start = time.time()
        normalized_rows: List[Dict[str, Any]] = []
        for row in raw_rows:
            missing = [name for name in required_columns if name not in row or row[name] is None]
            if missing:
                raise RuntimeError(
                    f"raw_features omitted required gold-standard covariate(s): {missing}"
                )
            normalized_row: Dict[str, Any] = {}
            for key, value in row.items():
                is_numeric_col = key in numeric_cols
                if isinstance(value, bool):
                    # bool is a valid 0/1 numeric covariate; never a category.
                    if not is_numeric_col and numeric_cols:
                        raise RuntimeError(
                            f"Categorical covariate '{key}' must be a string, not a bool"
                        )
                    normalized_row[key] = value
                elif isinstance(value, (int, float)):
                    if not is_numeric_col and numeric_cols:
                        raise RuntimeError(
                            f"Categorical covariate '{key}' must be a non-empty string "
                            f"for the gold-standard FeatureBuilder path (got {type(value).__name__})"
                        )
                    normalized_row[key] = value
                elif isinstance(value, str):
                    if is_numeric_col:
                        raise RuntimeError(
                            f"Numeric covariate '{key}' must be a number, not a string "
                            "(a string would be reindexed to a fabricated 0.0)"
                        )
                    if not value.strip():
                        raise RuntimeError(
                            f"Raw covariate '{key}' is an empty string; refusing to "
                            "fabricate a category for the gold-standard FeatureBuilder path"
                        )
                    normalized_row[key] = value
                else:
                    raise RuntimeError(
                        f"Raw covariate '{key}' must be numeric or a categorical string "
                        f"on the gold-standard FeatureBuilder path (got {type(value).__name__})"
                    )
            normalized_rows.append(normalized_row)

        raw_df = pd.DataFrame(list(normalized_rows))
        try:
            encoded = preprocessor.transform(raw_df)
        except Exception as e:
            # FAIL CLOSED — a raw->encoded transform failure must not fall back
            # to predicting on raw values (plausible-but-wrong audit-grade output).
            logger.error("Raw-covariate preprocessor transform failed: %s", e)
            raise RuntimeError(f"Preprocessor transform failed: {e}") from e

        arr = np.asarray(encoded)
        predictions = model.predict(arr).tolist()

        probabilities: List[float] = []
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(arr)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    probabilities = proba[:, 1].tolist()
                else:
                    probabilities = proba.tolist()
            except Exception:
                pass

        elapsed_ms = (time.time() - start) * 1000
        self._prediction_count += len(raw_rows)

        # Surface the ENCODED vector the model scored so the SHAP caller runs
        # SHAP over the audit-grade encoded features (not the raw covariates).
        encoded_cols = self._resolve_feature_columns(model, feature_columns) or []
        encoded_matrix = np.asarray(encoded).tolist()

        return PredictionOutput(
            predictions=predictions,
            probabilities=probabilities,
            model_id=model_tag or "unknown",
            prediction_time_ms=elapsed_ms,
            is_mock=False,
            feature_source="raw_covariates",
            encoded_features=encoded_matrix,
            encoded_feature_columns=list(encoded_cols),
        )

    @staticmethod
    def _resolve_linear_estimator(model: Any) -> Any:
        """Return the underlying LINEAR estimator SHAP's LinearExplainer needs.

        Gold-standard cohort models are a ``CalibratedClassifierCV`` wrapping a
        ``LogisticRegression``. ``shap.LinearExplainer`` needs a bare linear
        estimator exposing ``coef_`` — the calibration wrapper does not. So we
        unwrap to ``model.calibrated_classifiers_[0].estimator`` (sklearn >=1.4)
        / ``.base_estimator`` (older). This matches the verified in-container
        disproof: ``LinearExplainer`` over the inner LR gives per-encoded-feature
        SHAP that satisfy additivity exactly (base + sum(shap) == inner-LR
        margin). Returns the model itself when it already exposes ``coef_`` (a
        bare LR bundle), or ``None`` when no linear estimator can be found
        (caller FAILS CLOSED — no fabricated SHAP).
        """
        if hasattr(model, "coef_"):
            return model
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            inner = calibrated[0]
            est = getattr(inner, "estimator", None)
            if est is None:
                est = getattr(inner, "base_estimator", None)
            if est is not None and hasattr(est, "coef_"):
                return est
        return None

    def _run_shap_explanation(
        self,
        raw_rows: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        model: Any = _UNSET,
        preprocessor: Any = _UNSET,
        feature_columns: Any = _UNSET,
        model_tag: Any = _UNSET,
    ) -> ShapOutput:
        """Compute real per-ENCODED-feature SHAP for a gold-standard cohort (#39).

        Encodes the FIRST raw covariate row via the bundled FeatureBuilder (the
        SAME raw->encoded path ``_run_raw_prediction`` uses), then runs
        ``shap.LinearExplainer`` over the inner linear estimator of the calibrated
        model. SHAP runs over the ENCODED numeric vector the model actually scored
        (NOT the raw covariates), so the explanation is audit-grade.

        Background data: a 2-row encoded background (a zeros row + the encoded
        instance itself) — sufficient for the linear masker and keeps the
        base_value the explainer's expected value over that background, with
        exact additivity preserved (verified live).

        FAILS CLOSED (returns ``ShapOutput`` with a non-None ``error`` and empty
        value maps) on: no model, no FeatureBuilder, no linear estimator, missing
        covariates, or any explainer/transform failure. It NEVER fabricates SHAP
        values for an audit-grade explanation.
        """
        import numpy as np
        import pandas as pd

        model = self._model if model is _UNSET else model
        preprocessor = self._preprocessor if preprocessor is _UNSET else preprocessor
        feature_columns = self._feature_columns if feature_columns is _UNSET else feature_columns
        model_tag = self._model_tag if model_tag is _UNSET else model_tag

        def _fail(msg: str) -> ShapOutput:
            logger.error("SHAP fail-closed: %s", msg)
            return ShapOutput(model_id=str(model_tag or "unknown"), error=msg)

        if model is None:
            return _fail("no model loaded")
        if not self._is_feature_builder(preprocessor):
            return _fail(
                "served model has no FeatureBuilder preprocessor; cannot encode "
                "raw covariates for a gold-standard SHAP explanation"
            )
        if not raw_rows:
            return _fail("no raw_features supplied for SHAP")

        estimator = self._resolve_linear_estimator(model)
        if estimator is None:
            return _fail(
                "served model exposes no linear estimator (no coef_); refusing to "
                "fabricate SHAP for a non-linear model on this endpoint"
            )

        # Validate + dtype-preserve the raw row exactly like the predict path so a
        # categorical string is one-hot-encoded (not float-coerced) and a
        # malformed request fails closed rather than fabricating an encoded vector.
        required_columns = self._resolve_keep_columns(preprocessor) or []
        numeric_cols = set(getattr(preprocessor, "_numeric_medians", {}) or {})
        row = raw_rows[0]
        missing = [name for name in required_columns if name not in row or row[name] is None]
        if missing:
            return _fail(f"raw_features omitted required gold-standard covariate(s): {missing}")
        normalized_row: Dict[str, Any] = {}
        for key, value in row.items():
            is_numeric_col = key in numeric_cols
            if isinstance(value, bool):
                if not is_numeric_col and numeric_cols:
                    return _fail(f"Categorical covariate '{key}' must be a string, not a bool")
                normalized_row[key] = value
            elif isinstance(value, (int, float)):
                if not is_numeric_col and numeric_cols:
                    return _fail(
                        f"Categorical covariate '{key}' must be a non-empty string "
                        f"(got {type(value).__name__})"
                    )
                normalized_row[key] = value
            elif isinstance(value, str):
                if is_numeric_col:
                    return _fail(f"Numeric covariate '{key}' must be a number, not a string")
                if not value.strip():
                    return _fail(f"Raw covariate '{key}' is an empty string; refusing to fabricate")
                normalized_row[key] = value
            else:
                return _fail(
                    f"Raw covariate '{key}' must be numeric or a categorical string "
                    f"(got {type(value).__name__})"
                )

        start = time.time()
        try:
            import shap

            raw_df = pd.DataFrame([normalized_row])
            encoded = np.asarray(preprocessor.transform(raw_df), dtype=float)
            encoded_cols = self._resolve_feature_columns(model, feature_columns) or [
                f"f{i}" for i in range(encoded.shape[1])
            ]
            # 2-row encoded background (zeros + the instance) for the linear masker.
            background = np.vstack([np.zeros((1, encoded.shape[1])), encoded])
            explainer = shap.LinearExplainer(estimator, background)
            sv = np.asarray(explainer.shap_values(encoded))
            row_sv = np.ravel(sv[0]) if sv.ndim > 1 else np.ravel(sv)
            base = float(np.ravel(np.asarray(explainer.expected_value))[0])
        except Exception as e:  # FAIL CLOSED — no fabricated SHAP on any failure.
            return _fail(f"SHAP computation failed: {e}")

        if len(row_sv) != len(encoded_cols):
            return _fail(
                f"SHAP/feature length mismatch (shap={len(row_sv)}, cols={len(encoded_cols)})"
            )

        shap_map = {str(name): float(val) for name, val in zip(encoded_cols, row_sv, strict=True)}
        if top_k is not None and top_k > 0:
            shap_map = dict(
                sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
            )

        elapsed_ms = (time.time() - start) * 1000
        return ShapOutput(
            shap_values=shap_map,
            base_value=base,
            encoded_feature_columns=list(encoded_cols),
            encoded_features=[float(v) for v in np.ravel(encoded[0]).tolist()],
            explainer_type="LinearExplainer",
            model_id=str(model_tag or "unknown"),
            computation_time_ms=elapsed_ms,
        )

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
            # FAIL CLOSED: a bundled preprocessor exists but its transform
            # failed. Running model.predict on the RAW (un-preprocessed) matrix
            # would emit a plausible-but-wrong prediction the explain path then
            # treats as audit-grade truth. Raise instead of returning raw input.
            logger.error("Preprocessor transform failed: %s", e)
            raise RuntimeError(f"Preprocessor transform failed: {e}") from e

    def _run_prediction(
        self,
        features: List[List[float]],
        feature_source: Optional[str] = None,
        *,
        model: Any = _UNSET,
        preprocessor: Any = _UNSET,
        feature_columns: Any = _UNSET,
        model_tag: Any = _UNSET,
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

        model = self._model if model is _UNSET else model
        preprocessor = self._preprocessor if preprocessor is _UNSET else preprocessor
        feature_columns = self._feature_columns if feature_columns is _UNSET else feature_columns
        model_tag = self._model_tag if model_tag is _UNSET else model_tag

        if model is None:
            return PredictionOutput(
                predictions=[],
                probabilities=[],
                model_id="no_model",
                prediction_time_ms=0.0,
                is_mock=False,
                feature_source=feature_source,
            )

        start = time.time()
        if preprocessor is None:
            arr = np.array(features)
        else:
            try:
                if features and feature_columns and len(feature_columns) == len(features[0]):
                    import pandas as pd

                    arr = preprocessor.transform(pd.DataFrame(features, columns=feature_columns))
                else:
                    arr = preprocessor.transform(np.array(features))
            except Exception as e:
                logger.error("Preprocessor transform failed: %s", e)
                raise RuntimeError(f"Preprocessor transform failed: {e}") from e

        predictions = model.predict(arr).tolist()

        probabilities = []
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(arr)
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
            model_id=model_tag or "unknown",
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
        feast_values: Dict[str, List[Any]] = {
            name: results[idx].get("values", [])
            for idx, name in enumerate(feature_names)
            if name != entity_key
        }

        # Build each row in the MODEL's authoritative feature_columns order, not
        # Feast's response order — Feast may return columns in a different order
        # than the bundled model expects, which would silently produce a
        # plausible-but-WRONG vector labeled feature_source='feast_online'.
        # Extra Feast columns are ignored. When the model exposes no feature
        # order, fall back to Feast order (no model contract to enforce against).
        expected_order = self._resolve_feature_columns() or list(feast_values.keys())

        n_rows = len(entity_ids)
        matrix: List[List[float]] = []
        # FAIL CLOSED on missing/null/non-numeric values. A Feast 200 can carry
        # PRESENT-but-null values; zero-filling them and labeling the response
        # feature_source='feast_online' feeds the model a fabricated vector
        # presented as real, audit-grade Feast data (the #576/#532 harm —
        # mirrors the FastAPI route's anti-null-trap guard). A real 0.0 passes;
        # missing/null/non-numeric raises.
        invalid: Dict[str, str] = {}
        for row_idx in range(n_rows):
            row: List[float] = []
            for col_name in expected_order:
                col_values = feast_values.get(col_name)
                if col_values is None:
                    invalid[col_name] = "missing"
                    row.append(0.0)  # placeholder; request fails below
                    continue
                raw = col_values[row_idx] if row_idx < len(col_values) else None
                if raw is None:
                    invalid[col_name] = "null"
                    row.append(0.0)  # placeholder; request fails below
                    continue
                try:
                    row.append(float(raw))
                except (TypeError, ValueError):
                    invalid[col_name] = "non-numeric"
                    row.append(0.0)  # placeholder; request fails below
            matrix.append(row)

        if invalid:
            # Do NOT run inference over a fabricated/mis-ordered vector mislabeled
            # feast_online. Fail loud so the caller sees an honest error.
            raise RuntimeError(
                "Feast features unusable for an audit-grade 'feast_online' "
                f"prediction: {dict(sorted(invalid.items()))}; refusing to "
                "fabricate over zero-filled features"
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

        Multi-model routing (#39): when ``model_name`` is set it selects which
        loaded gold-standard bundle to use; unknown names FAIL CLOSED. When None,
        the legacy single default model is used (Feast / numeric / tier0).

        Args:
            input_data: Features and configuration

        Returns:
            Model predictions with a ``feature_source`` telemetry tag.
        """
        # Resolve the routed model first so an unknown model_name fails closed
        # BEFORE any Feast/feature work (no fabricated/wrong-model prediction).
        model, preprocessor, feature_columns, model_tag, err = self._resolve_active(
            input_data.model_name
        )
        if err is not None:
            return PredictionOutput(
                predictions=[],
                probabilities=[],
                model_id="unknown_model",
                prediction_time_ms=0.0,
                is_mock=False,
                error=err,
            )

        if input_data.entity_ids and input_data.feature_view:
            features = await self._fetch_features_from_feast(
                entity_ids=input_data.entity_ids,
                feature_view=input_data.feature_view,
                entity_key=input_data.entity_key,
            )
            return self._run_prediction(
                features,
                feature_source="feast_online",
                model=model,
                preprocessor=preprocessor,
                feature_columns=feature_columns,
                model_tag=model_tag,
            )

        # RAW covariate path (#39): gold-standard cohort models bundle a
        # FeatureBuilder preprocessor and expect the RAW covariates. Takes
        # precedence over the legacy numeric ``features`` matrix when present.
        # Routes to the resolved (possibly multi-model) components.
        if input_data.raw_features:
            return self._run_raw_prediction(
                input_data.raw_features,
                model=model,
                preprocessor=preprocessor,
                feature_columns=feature_columns,
                model_tag=model_tag,
            )

        feature_source = "user_provided" if input_data.features else None
        return self._run_prediction(
            input_data.features,
            feature_source=feature_source,
            model=model,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
            model_tag=model_tag,
        )

    @bentoml.api
    async def predict_batch(self, input_data: BatchPredictionInput) -> BatchPredictionOutput:
        """Run batch predictions.

        Args:
            input_data: Batch of features

        Returns:
            Batch predictions
        """
        start = time.time()

        # RAW covariate batch path (#cohort-scoring): route by model_name, encode each
        # row via the bundled FeatureBuilder, and return PER-ROW probabilities. Mirrors
        # the single-predict raw dispatch (see ``predict``) and reuses the already-
        # vectorized ``_run_raw_prediction`` (ONE transform + ONE predict over the batch),
        # so a holdout cohort is scored in one chunked call. Fails closed on an unknown
        # model_name (no fabricated/wrong-model scoring).
        if input_data.raw_features:
            model, preprocessor, feature_columns, model_tag, err = self._resolve_active(
                input_data.model_name
            )
            if err is not None:
                return BatchPredictionOutput(
                    batch_id=input_data.batch_id,
                    total_samples=len(input_data.raw_features),
                    predictions=[],
                    probabilities=[],
                    processing_time_ms=0.0,
                    error=err,
                )
            out = self._run_raw_prediction(
                input_data.raw_features,
                model=model,
                preprocessor=preprocessor,
                feature_columns=feature_columns,
                model_tag=model_tag,
            )
            elapsed_ms = (time.time() - start) * 1000
            return BatchPredictionOutput(
                batch_id=input_data.batch_id,
                total_samples=len(input_data.raw_features),
                predictions=out.predictions,
                probabilities=out.probabilities,
                processing_time_ms=elapsed_ms,
            )

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
    async def model_info(self, input_data: Optional[ModelInfoInput] = None) -> Dict[str, Any]:
        """Return model information, optionally for a routed model (#39).

        With ``input_data.model_name`` set + known → that bundle's contract
        (its ``keep_columns`` + encoded ``feature_columns``). With None / legacy
        empty body → the default model's info. An unknown name returns an
        ``error`` field (fail-closed, not the default model's contract).

        ``available_models`` enumerates every loaded gold-standard serving name
        so the route/selector can list cohort×brand options.

        Returns:
            Model metadata
        """
        requested = input_data.model_name if input_data is not None else None
        model, preprocessor, feature_columns, model_tag, err = self._resolve_active(requested)

        available_models = sorted(self._models.keys())

        if err is not None:
            return {
                "model_id": requested or "no_model",
                "model_loaded": False,
                "available_models": available_models,
                "error": err,
            }

        info: Dict[str, Any] = {
            "model_id": model_tag or "no_model",
            "model_type": self._framework or "none",
            "framework": self._framework or "none",
            "version": "1.0.0",
            "is_mock": False,
            "model_loaded": model is not None,
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
            "feature_columns": self._resolve_feature_columns(model, feature_columns),
            # The RAW covariate names a caller must supply for a gold-standard
            # cohort model (#39): the bundled FeatureBuilder's ``keep_columns``
            # (patient: ['disease_severity','academic_hcp','geographic_region'];
            # HCP: the 5 hcp covariates). The service encodes these into the
            # numeric ``feature_columns`` (the SHAP vector) at serve time. None
            # for a bare estimator / ColumnTransformer (no raw-covariate
            # contract — supply the encoded matrix directly).
            "keep_columns": self._resolve_keep_columns(preprocessor),
            # Loaded gold-standard serving names (multi-model #39).
            "available_models": available_models,
        }

        # Add model metadata if available (legacy default path only — routed
        # bundles are loaded into memory, not necessarily named in the store).
        if model_tag and requested is None:
            try:
                bento_model = bentoml.models.get(model_tag)
                meta = bento_model.info.metadata or {}
                info["metadata"] = meta
            except Exception:
                pass

        return info

    @bentoml.api
    async def shap(self, input_data: ShapInput) -> ShapOutput:
        """Compute real per-ENCODED-feature SHAP for a gold-standard cohort (#39).

        NOTE: BentoML routes each ``@bentoml.api`` method at ``/<method_name>``
        (verified live: ``predict`` -> ``/predict``, ``model_info`` ->
        ``/model_info``). This method is named ``shap`` so the live path is
        ``/shap`` — the path the API client's ``get_shap`` posts to.

        Option-B placement: the gold-standard bundles (model + FeatureBuilder +
        encoded ``feature_columns``) live ONLY in this container's mount, so the
        SHAP explanation is computed HERE, where the bundle is loaded — the API
        process cannot read the bundle files. Input mirrors ``/predict``'s
        raw-covariate path (routed ``model_name`` + ``raw_features``); the service
        encodes the raw row via the bundled FeatureBuilder and runs
        ``shap.LinearExplainer`` over the ENCODED vector the model actually scored.

        Routing (#39): an unknown ``model_name`` FAILS CLOSED (``error`` set) — it
        does NOT silently SHAP the default/wrong model. ``/predict``,
        ``/predict_batch``, ``/model_info`` are unchanged.

        Returns:
            ``ShapOutput`` with signed per-encoded-feature ``shap_values``,
            ``base_value``, ``encoded_feature_columns``, and ``explainer_type``.
            A non-None ``error`` means SHAP could not be computed (fail-closed).
        """
        model, preprocessor, feature_columns, model_tag, err = self._resolve_active(
            input_data.model_name
        )
        if err is not None:
            return ShapOutput(model_id=input_data.model_name or "unknown_model", error=err)

        return self._run_shap_explanation(
            input_data.raw_features,
            top_k=input_data.top_k,
            model=model,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
            model_tag=model_tag,
        )
