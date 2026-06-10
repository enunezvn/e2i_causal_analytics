"""SHAP Computation Node - NO LLM.

Computes SHAP values for model interpretability using the SHAP library.
This is a deterministic computation node with no LLM calls.
"""

import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import shap
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


def _load_model_flavor_agnostic(model_uri: str) -> Any:
    """Load an MLflow model without assuming a specific flavor.

    Tries native flavors first (xgboost, lightgbm, sklearn) and falls back
    to pyfunc. If ``get_model_info`` itself fails (tracking server
    unreachable, stale URI), drops directly to ``pyfunc.load_model`` —
    pyfunc is the one loader guaranteed to exist for any MLflow-logged
    model.

    Args:
        model_uri: MLflow model URI (already validated by ``validate_model_uri``).

    Returns:
        Loaded model object. The exact type depends on the resolved flavor.
    """
    try:
        info = mlflow.models.get_model_info(model_uri)
        flavors = set(info.flavors.keys())
    except MlflowException:
        return mlflow.pyfunc.load_model(model_uri)

    flavor_loaders: Tuple[Tuple[str, Callable[[str], Any]], ...] = (
        ("xgboost", mlflow.xgboost.load_model),
        ("lightgbm", mlflow.lightgbm.load_model),
        ("sklearn", mlflow.sklearn.load_model),
    )
    for flavor_name, loader in flavor_loaders:
        if flavor_name in flavors:
            return loader(model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def validate_model_uri(model_uri: str) -> Tuple[bool, Optional[str]]:
    """Validate MLflow model URI format.

    Supported formats:
    - runs:/<run_id>/<artifact_path> (run artifacts)
    - models:/<model_name>/<version> (model registry by version)
    - models:/<model_name>/<stage> (model registry by stage)
    - file://<absolute_path> (local file path)

    Args:
        model_uri: The model URI to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not model_uri or not isinstance(model_uri, str):
        return False, "model_uri must be a non-empty string"

    model_uri = model_uri.strip()

    # Pattern for runs:/<run_id>/<artifact_path>
    # run_id is typically 32 hex chars but MLflow allows variations
    runs_pattern = r"^runs:/[a-zA-Z0-9_-]+/.+$"

    # Pattern for models:/<model_name>/<version_or_stage>
    # version is numeric, stage is one of: None, Staging, Production, Archived
    models_version_pattern = r"^models:/[a-zA-Z0-9_.-]+/\d+$"
    models_stage_pattern = r"^models:/[a-zA-Z0-9_.-]+/(None|Staging|Production|Archived)$"

    # Pattern for MLflow 3.x "Logged Models" format: models:/m-{uuid}
    # The model_uuid is a 32-char hex string with dashes removed
    models_logged_pattern = r"^models:/m-[a-f0-9]{32}$"

    # Pattern for file:// URIs (absolute paths)
    file_pattern = r"^file:///.+$"

    # Pattern for absolute paths without file:// prefix (Unix/Windows)
    absolute_path_pattern = r"^(/|[A-Za-z]:).+$"

    if re.match(runs_pattern, model_uri):
        return True, None
    elif re.match(models_version_pattern, model_uri):
        return True, None
    elif re.match(models_stage_pattern, model_uri):
        return True, None
    elif re.match(models_logged_pattern, model_uri):
        return True, None
    elif re.match(file_pattern, model_uri):
        return True, None
    elif re.match(absolute_path_pattern, model_uri):
        return True, None

    return False, (
        f"Invalid model_uri format: '{model_uri}'. "
        "Expected one of: 'runs:/<run_id>/<path>', 'models:/<name>/<version>', "
        "'models:/<name>/<stage>', 'models:/m-<uuid>' (MLflow 3.x), 'file:///<path>', or absolute file path"
    )


def _generate_domain_aware_background(
    feature_names: List[str],
    n_samples: int,
) -> np.ndarray:
    """Generate synthetic background data using domain-aware distributions.

    Instead of uniform random values, this function infers appropriate
    distributions from feature name patterns commonly used in pharma
    commercial analytics.

    Args:
        feature_names: List of feature names to generate data for
        n_samples: Number of samples to generate

    Returns:
        Background data array with shape (n_samples, n_features)
    """
    n_features = len(feature_names)
    background = np.zeros((n_samples, n_features))

    for i, name in enumerate(feature_names):
        name_lower = name.lower()

        # Binary/boolean features (is_*, has_*, flag_*)
        if name_lower.startswith(("is_", "has_", "flag_")):
            # Bernoulli with 30% positive rate (typical for boolean flags)
            background[:, i] = np.random.binomial(1, 0.3, n_samples)

        # Count features (poisson distribution)
        elif any(x in name_lower for x in ["_count", "num_", "total_", "n_"]):
            # Poisson with lambda=5 (typical for count data)
            background[:, i] = np.random.poisson(5, n_samples)

        # Rate/ratio features (beta distribution, 0-1 range)
        elif any(x in name_lower for x in ["_rate", "_ratio", "_pct", "_percent", "adherence"]):
            # Beta distribution for rates (mean ~0.5, some variance)
            background[:, i] = np.random.beta(2, 2, n_samples)

        # Score features (normal distribution, bounded)
        elif any(x in name_lower for x in ["_score", "index", "nps"]):
            # Normal distribution with mean 50, std 15 (like many scoring systems)
            background[:, i] = np.clip(np.random.normal(50, 15, n_samples), 0, 100)

        # Age features
        elif "age" in name_lower:
            # Normal distribution centered around 50 for patient/HCP ages
            background[:, i] = np.clip(np.random.normal(50, 15, n_samples), 18, 90)

        # Days/duration features (exponential-ish)
        elif any(x in name_lower for x in ["days_", "_days", "duration", "tenure"]):
            # Exponential with mean 30 days
            background[:, i] = np.random.exponential(30, n_samples)

        # Monetary/revenue features (log-normal)
        elif any(x in name_lower for x in ["revenue", "cost", "price", "value", "_amount"]):
            # Log-normal for monetary values (right-skewed)
            background[:, i] = np.random.lognormal(mean=5, sigma=1, size=n_samples)

        # Tier/category encoded as ordinal (1-5 typically)
        elif any(x in name_lower for x in ["tier", "category", "segment", "bucket"]):
            # Discrete uniform 1-5
            background[:, i] = np.random.randint(1, 6, n_samples)

        # Default: standard normal (mean 0, std 1)
        else:
            background[:, i] = np.random.randn(n_samples)

    return background


def _anonymize_feature_names(
    feature_names: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """Anonymize feature names to prevent schema information leakage.

    Args:
        feature_names: Original feature names

    Returns:
        Tuple of (anonymized_names, mapping from anonymous to original)
    """
    anonymized = [f"feature_{i}" for i in range(len(feature_names))]
    mapping = dict(zip(anonymized, feature_names, strict=False))
    return anonymized, mapping


def _normalize_shap_binary(vals: Any, base: Any) -> Tuple[np.ndarray, float]:
    """Normalize SHAP output to 2-D values and scalar base value for binary classification.

    SHAP >=0.42 TreeExplainer returns a 3-D ndarray (n, f, n_classes) for binary;
    legacy SHAP returned a list[ndarray, ndarray]. LinearExplainer/KernelExplainer
    can return 2-D arrays with a scalar or length-n_classes base_value.
    """
    if isinstance(vals, list):
        vals = vals[1] if len(vals) > 1 else vals[0]
    elif isinstance(vals, np.ndarray) and vals.ndim == 3:
        vals = vals[:, :, 1] if vals.shape[2] > 1 else vals[:, :, 0]

    if isinstance(base, (list, tuple)):
        base = base[1] if len(base) > 1 else base[0]
    if isinstance(base, np.ndarray):
        flat = base.flatten()
        base = float(flat[1]) if flat.size > 1 else float(flat[0])
    elif isinstance(base, np.number):
        base = float(base)
    elif not isinstance(base, float):
        base = float(base)

    return vals, base


async def compute_shap(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute SHAP values for the trained model.

    This node:
    1. Loads the model from MLflow
    2. Samples training data for SHAP computation
    3. Computes SHAP values using appropriate explainer
    4. Calculates global feature importance
    5. Determines feature directions (positive/negative/mixed)

    Args:
        state: Current agent state with model_uri and configuration

    Returns:
        State updates with SHAP values and importance metrics
    """
    start_time = time.time()

    try:
        # Extract inputs
        model_uri = state.get("model_uri")
        experiment_id = state.get("experiment_id")
        max_samples = state.get("max_samples", 1000)

        if not model_uri:
            logger.info("model_uri not provided - skipping SHAP computation")
            return {
                "shap_skipped": True,
                "skip_reason": "model_uri not provided - SHAP analysis requires a model",
                "status": "skipped",
                "data_provenance": "unavailable",
                # Provide empty defaults for downstream processing
                "global_importance_ranked": [],
                "top_features": [],
                "samples_analyzed": 0,
            }

        # Validate model_uri format before passing to MLflow
        is_valid, validation_error = validate_model_uri(model_uri)
        if not is_valid:
            return {
                "error": validation_error,
                "error_type": "invalid_model_uri",
                "status": "failed",
            }

        # Prefer in-memory model (passed via agent.py initial_state from
        # input_data["trained_model"] — survives node retries and avoids the
        # MLflow round-trip). Falls back to a flavor-agnostic loader so that
        # xgboost / lightgbm / pyfunc-only registrations don't break SHAP.
        loaded_model = state.get("loaded_model")
        if loaded_model is None:
            loaded_model = _load_model_flavor_agnostic(model_uri)

        # Get training run metadata
        run_id = model_uri.split("/")[1] if "runs:/" in model_uri else None
        if run_id:
            run = mlflow.get_run(run_id)
            training_run_id = run.info.run_id
            model_version = run.data.params.get("model_version", "unknown")
        else:
            training_run_id = state.get("training_run_id", "unknown")
            model_version = state.get("model_version", "unknown")

        # Get feature names - prioritize state's feature_columns over model attributes
        # This ensures feature names from data_preparer flow through to SHAP output
        feature_columns = state.get("feature_columns")

        if feature_columns and len(feature_columns) > 0:
            # Use preserved feature names from data_preparer
            feature_names = list(feature_columns)
            logger.info(f"Using {len(feature_names)} feature names from state.feature_columns")
        elif hasattr(loaded_model, "feature_names_in_"):
            # Use feature names from model (set during training)
            feature_names = list(loaded_model.feature_names_in_)
            logger.info(f"Using {len(feature_names)} feature names from model.feature_names_in_")
        elif hasattr(loaded_model, "feature_name_"):
            # LightGBM stores feature names differently
            feature_names = list(loaded_model.feature_name_)
            logger.info(f"Using {len(feature_names)} feature names from model.feature_name_")
        else:
            # Fallback: use generic names (last resort)
            n_features = (
                loaded_model.n_features_in_ if hasattr(loaded_model, "n_features_in_") else 10
            )
            feature_names = [f"feature_{i}" for i in range(n_features)]
            logger.warning(
                "Using generic feature names (feature_0, feature_1, ...) - "
                "feature_columns not preserved from data_preparer"
            )

        # L5 Fix: Support feature name anonymization to prevent schema leakage
        anonymize_features = state.get("anonymize_features", False)
        feature_name_mapping = None
        if anonymize_features:
            original_feature_names = feature_names
            feature_names, feature_name_mapping = _anonymize_feature_names(feature_names)
        else:
            original_feature_names = feature_names

        # Load sample data for SHAP computation. F8 fail-closed: prefer REAL data
        # (caller-provided X_sample, then the selected / raw training features bridged
        # from upstream nodes). A synthetic np.random background is gated behind an
        # explicit opt-in and stamped data_provenance="synthetic", so importances over
        # fabricated data are never silently presented as measured.
        X_sample = state.get("X_sample")
        y_sample = state.get("y_sample")
        shap_data_provenance = "real"

        if X_sample is None:
            # Bridge real selected / raw training features from upstream nodes
            # (feature_selector populates X_train_selected; X_train is the raw set).
            bridged = state.get("X_train_selected")
            if bridged is None:
                bridged = state.get("X_train")
            X_sample = bridged

        if X_sample is None:
            if not bool(state.get("allow_synthetic_background", False)):
                # Fail closed: do NOT fabricate SHAP importances from an np.random
                # background. Skip with an honest reason rather than returning
                # synthetic, unlabeled importances that look measured.
                logger.warning(
                    "SHAP: no real sample data (X_sample / X_train_selected / X_train "
                    "absent) and allow_synthetic_background is False; skipping SHAP "
                    "rather than computing importances over a synthetic background."
                )
                return {
                    "shap_skipped": True,
                    "skip_reason": (
                        "no real sample data available (X_sample/X_train_selected/"
                        "X_train absent); SHAP skipped to avoid synthetic-background "
                        "importances"
                    ),
                    "status": "skipped",
                    "data_provenance": "unavailable",
                    "global_importance_ranked": [],
                    "top_features": [],
                    "samples_analyzed": 0,
                }
            # Synthetic background explicitly opted into — generate and STAMP it so
            # downstream consumers know the importances are over fabricated data.
            shap_data_provenance = "synthetic"
            logger.warning(
                "SHAP: allow_synthetic_background=True; computing over a synthetic "
                "np.random background. Results stamped data_provenance='synthetic'."
            )
            X_sample = _generate_domain_aware_background(
                original_feature_names,  # Use original names for distribution inference
                min(max_samples, 1000),
            )

        # Limit sample size (DataFrame-safe: bridged real features may be a DataFrame)
        if len(X_sample) > max_samples:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample.iloc[indices] if hasattr(X_sample, "iloc") else X_sample[indices]
            if y_sample is not None:
                y_sample = (
                    y_sample.iloc[indices] if hasattr(y_sample, "iloc") else y_sample[indices]
                )

        samples_analyzed = len(X_sample)

        # Choose appropriate SHAP explainer based on model type. The native
        # explainer occasionally chokes on library version mismatches (e.g.
        # SHAP 0.48 + XGBoost 3.x raises ValueError on base_score parse —
        # XGBoost stores it as the array-encoded string "[5E-1]" which the
        # SHAP TreeExplainer loader cannot deserialize). On any failure we
        # fall through to KernelExplainer so the SHAP node still produces
        # importance rankings rather than dropping the whole analysis.
        explainer_type = _select_explainer_type(loaded_model)
        shap_values: Optional[np.ndarray] = None
        base_value: Optional[float] = None

        if explainer_type in ("TreeExplainer", "LinearExplainer"):
            try:
                if explainer_type == "TreeExplainer":
                    explainer = shap.TreeExplainer(loaded_model)
                else:
                    explainer = shap.LinearExplainer(loaded_model, X_sample)
                shap_values_raw = explainer.shap_values(X_sample)
                base_value_raw = explainer.expected_value
                shap_values, base_value = _normalize_shap_binary(shap_values_raw, base_value_raw)
            except Exception as native_exc:
                logger.warning(
                    f"{explainer_type} failed for "
                    f"{loaded_model.__class__.__name__}: {native_exc!r} — "
                    "falling back to KernelExplainer."
                )
                shap_values = None
                explainer_type = "KernelExplainer"

        if shap_values is None:  # KernelExplainer (selected or fallback)
            # KernelExplainer is O(n²) in background size — uncapped pyfunc
            # fallback can hang for tens of minutes. Caps mirror what
            # TreeExplainer effectively gets via `max_samples`.
            background_size = max(1, min(100, len(X_sample) // 10))
            background = shap.kmeans(X_sample, background_size)
            # Prefer predict_proba for classifiers (SHAP values align with
            # AUC); fall back to predict for regressors / pyfunc wrappers.
            # Wrap in a closure rather than passing the bound method directly:
            # shap.utils._legacy mutates `bound_method.__self__.feature_names_in_`
            # to None to suppress an sklearn warning, and that property is
            # read-only on XGBoost 3.x — the closure has no `__self__`, so
            # the mutation is skipped.
            has_proba = hasattr(loaded_model, "predict_proba")

            def _predict_fn(arr: np.ndarray) -> Any:
                return loaded_model.predict_proba(arr) if has_proba else loaded_model.predict(arr)

            explainer = shap.KernelExplainer(_predict_fn, background)
            shap_values_raw = explainer.shap_values(X_sample[:100])
            base_value_raw = explainer.expected_value
            shap_values, base_value = _normalize_shap_binary(shap_values_raw, base_value_raw)
            samples_analyzed = min(100, samples_analyzed)

        # Compute global importance (mean absolute SHAP values)
        global_importance_values = np.abs(shap_values).mean(axis=0)
        global_importance = {
            feature_names[i]: float(global_importance_values[i]) for i in range(len(feature_names))
        }

        # Rank features by importance
        global_importance_ranked = sorted(
            global_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Get top 5 features
        top_features = [feat for feat, _ in global_importance_ranked[:5]]

        # Determine feature directions (positive/negative/mixed)
        feature_directions = {}
        for i, feature in enumerate(feature_names):
            mean_shap = shap_values[:, i].mean()
            std_shap = shap_values[:, i].std()

            # If std is large relative to mean, it's mixed
            if abs(std_shap) > 2 * abs(mean_shap) and abs(mean_shap) > 0.001:
                direction = "mixed"
            elif mean_shap > 0.001:
                direction = "positive"
            elif mean_shap < -0.001:
                direction = "negative"
            else:
                direction = "neutral"

            feature_directions[feature] = direction

        computation_time = time.time() - start_time

        # Generate unique analysis ID
        shap_analysis_id = f"shap_{experiment_id}_{uuid.uuid4().hex[:8]}"

        return {
            "shap_analysis_id": shap_analysis_id,
            "data_provenance": shap_data_provenance,
            "training_run_id": training_run_id,
            "model_version": model_version,
            "loaded_model": loaded_model,
            "feature_names": feature_names,
            "original_feature_names": original_feature_names,  # L5: Always include original names
            "feature_name_mapping": feature_name_mapping,  # L5: Mapping for de-anonymization
            "features_anonymized": anonymize_features,  # L5: Flag indicating if anonymized
            "X_sample": X_sample,
            "y_sample": y_sample,
            "samples_analyzed": samples_analyzed,
            "shap_values": shap_values,
            "base_value": base_value,
            "global_importance": global_importance,
            "global_importance_ranked": global_importance_ranked,
            "feature_directions": feature_directions,
            "top_features": top_features,
            "shap_computation_time_seconds": computation_time,
            "explainer_type": explainer_type,
        }

    except Exception as e:
        return {
            "error": f"SHAP computation failed: {str(e)}",
            "error_type": "shap_computation_error",
            "error_details": {"exception": str(e)},
            "status": "failed",
        }


def _select_explainer_type(model: Any) -> str:
    """Select appropriate SHAP explainer based on model type.

    Args:
        model: Trained model object

    Returns:
        Explainer type: "TreeExplainer" | "LinearExplainer" | "KernelExplainer"
    """
    model_class = model.__class__.__name__

    # Tree-based models
    tree_models = [
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "CatBoostClassifier",
        "CatBoostRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
    ]

    if model_class in tree_models:
        return "TreeExplainer"

    # Linear models
    linear_models = [
        "LinearRegression",
        "LogisticRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDClassifier",
        "SGDRegressor",
    ]

    if model_class in linear_models:
        return "LinearExplainer"

    # Fallback to KernelExplainer (model-agnostic)
    return "KernelExplainer"
