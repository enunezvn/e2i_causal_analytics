"""Request-side encoding for the model prediction routes (split from ``predictions``).

The served BentoML sidecar is multi-model: ``/model_info``, ``/predict`` and
``/predict_batch`` route by ``input_data.model_name`` and fall back to a DEFAULT
model when it is absent. Everything here resolves the named model's input
contract from ``/model_info`` (never guessed) and fails closed on a missing
input instead of zero-filling.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from src.api.dependencies.bentoml_client import BentoMLClient

logger = logging.getLogger(__name__)


async def resolve_feature_order(client: "BentoMLClient", model_name: str) -> List[str]:
    """Resolve the served model's authoritative ordered feature names.

    The live BentoML service expects ``features`` as a POSITIONAL numeric matrix
    ordered by the model's own ``feature_columns`` (the preprocessor input
    order, or the estimator's ``feature_names_in_``). The service exposes this
    via ``POST /model_info`` -> ``feature_columns``. We fetch it from the model
    itself rather than guessing/hardcoding an order (the repo has several
    divergent feature lists; only the bundled model knows its real order).

    Fails CLOSED (503) when the model exposes no feature order — never invents a
    positional order, which would silently feed the model a mis-ordered vector
    presented as a real prediction.
    """
    try:
        info = await client.get_model_info(model_name)
    except Exception as e:
        logger.error("Could not fetch model_info for feature order (model=%s): %s", model_name, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model metadata unavailable for '{model_name}'",
        )

    columns = info.get("feature_columns")
    if not columns or not isinstance(columns, list):
        logger.error(
            "Model '%s' exposes no feature_columns order via /model_info; refusing to "
            "vectorize a feature dict against an unknown positional order.",
            model_name,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{model_name}' does not expose a feature order; cannot vectorize "
                "feature dictionary"
            ),
        )
    return [str(c) for c in columns]


def vectorize_feature_dict(
    features: Dict[str, Any], feature_order: List[str], *, context: str
) -> List[float]:
    """Build a single ordered numeric row from a feature dict + canonical order.

    Each value is read by name in ``feature_order``. A missing or null required
    feature FAILS CLOSED with a 422 — no silent zero-fill (which would fabricate
    a plausible-but-wrong prediction). Extra keys not in the order are ignored.
    Non-numeric values raise a 422 with the offending field named.
    """
    missing = [name for name in feature_order if name not in features or features[name] is None]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Missing required feature(s) for {context}: {missing}. "
                f"Expected features (in order): {feature_order}"
            ),
        )
    row: List[float] = []
    for name in feature_order:
        value = features[name]
        try:
            row.append(float(value))
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature '{name}' is not numeric (got {value!r}) for {context}",
            )
    return row


async def build_batch_input(
    client: "BentoMLClient", model_name: str, rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build the ``BatchPredictionInput`` for ``POST /predict/{model_name}/batch``.

    #2343: EVERY path carries ``model_name``. Without it the multi-model sidecar
    scored its DEFAULT model while the route echoed the requested name — plausible
    probabilities from the wrong model, silently. The encoding mirrors the single
    ``/predict`` route:

    - ``keep_columns`` exposed (goldstd FeatureBuilder bundles) -> RAW covariate
      rows as ``raw_features``, encoded server-side by the bundle. A missing
      covariate on any row is a 422 naming it.
    - otherwise (legacy positional models) -> each dict vectorized into
      ``feature_columns`` order as ``features``. A missing required feature is a
      422; no exposed order is a 503.
    """
    try:
        model_info = await client.get_model_info(model_name)
    except Exception as e:
        logger.error("Could not fetch model_info for predict_batch (model=%s): %s", model_name, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model metadata unavailable for '{model_name}'",
        )

    batch_data: Dict[str, Any] = {"batch_id": str(uuid.uuid4()), "model_name": model_name}
    keep_columns = model_info.get("keep_columns")
    if isinstance(keep_columns, list) and keep_columns:
        for i, row in enumerate(rows):
            missing = [c for c in keep_columns if row.get(c) is None]
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Missing required covariate(s) for '{model_name}' (instance={i}): "
                        f"{missing}. Expected raw covariates: {list(keep_columns)}"
                    ),
                )
        batch_data["raw_features"] = [dict(row) for row in rows]
        return batch_data

    columns = model_info.get("feature_columns")
    if not columns or not isinstance(columns, list):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{model_name}' does not expose a feature order; "
                "cannot vectorize feature dictionary"
            ),
        )
    feature_order = [str(c) for c in columns]
    batch_data["features"] = [
        vectorize_feature_dict(
            row, feature_order, context=f"predict_batch(model={model_name}, instance={i})"
        )
        for i, row in enumerate(rows)
    ]
    return batch_data
