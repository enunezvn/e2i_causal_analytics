"""
Offline Twin Training Job (#705 H4)
===================================

Trains a real ``TwinGenerator`` for a ``(brand, twin_type)`` from synthetic or
supplied data, persists the fitted artifact to MLflow via
:mod:`src.digital_twin.twin_persistence`, and records a loadable
``digital_twin_models`` row. This is the piece that lets ``/simulate`` load a
real model instead of failing closed (503) forever.

``train_and_persist_twin`` is a plain awaitable so it can run **inline** (a
script, an admin endpoint, or a future Celery ``worker_heavy`` task) — the heavy
worker ships dark on the 16 GB box, so the pipeline must not depend on it.

``data_provenance`` is recorded on the model so a synthetic-trained model is
never mistaken for an RWD-trained one (anti-mock discipline).
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from . import twin_persistence
from .models.twin_models import Brand, TwinModelConfig, TwinType
from .training_data import DEFAULT_TARGET_COLUMN, synthetic_training_frame
from .twin_generator import TwinGenerator

logger = logging.getLogger(__name__)


def _resolve_training_frame(
    *,
    twin_type: TwinType,
    data: Optional[pd.DataFrame],
    data_source: Optional[str],
    target_column: str,
    synthetic: bool,
    n_rows: int,
    seed: int,
) -> Tuple[pd.DataFrame, str]:
    """Resolve the training frame + its provenance, or fail loud.

    Priority: an explicit ``data`` frame, then a ``data_source`` file (RWD), then
    a synthetic frame. With none of these we raise rather than train on nothing.
    """
    if data is not None:
        return data, "provided"
    if data_source:
        if data_source.endswith(".parquet"):
            return pd.read_parquet(data_source), "rwd_file"
        return pd.read_csv(data_source), "rwd_file"
    if synthetic:
        frame = synthetic_training_frame(
            twin_type, n_rows=n_rows, target_col=target_column, seed=seed
        )
        return frame, "synthetic"
    raise ValueError(
        "train_and_persist_twin requires one of: data, data_source, or "
        "synthetic=True (fail closed rather than train on nothing)"
    )


async def train_and_persist_twin(
    *,
    twin_type: TwinType,
    brand: Brand,
    repo: Any,
    data: Optional[pd.DataFrame] = None,
    data_source: Optional[str] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    algorithm: str = "random_forest",
    synthetic: bool = False,
    n_rows: int = 2000,
    seed: int = 0,
    geographic_scope: str = "national",
) -> Dict[str, Any]:
    """Train + persist a digital-twin generative model for ``(brand, twin_type)``.

    Args:
        twin_type / brand: identity of the twin model.
        repo: a ``TwinRepository`` (or ``TwinModelRepository``) with
            ``save_model(config, metrics, *, mlflow_run_id, mlflow_model_uri)``.
        data / data_source / synthetic: training-frame source (exactly one).
        target_column: outcome column to model.
        algorithm: ``"random_forest"`` or ``"gradient_boosting"``.

    Returns:
        A dict with the real ``model_id``, ``run_id``, ``model_uri``,
        ``r2_score``, ``training_samples`` and ``data_provenance``.

    Raises:
        ValueError: if no training-frame source is supplied, or training data is
            insufficient (propagated from ``TwinGenerator.train``).
    """
    frame, provenance = _resolve_training_frame(
        twin_type=twin_type,
        data=data,
        data_source=data_source,
        target_column=target_column,
        synthetic=synthetic,
        n_rows=n_rows,
        seed=seed,
    )

    generator = TwinGenerator(twin_type=twin_type, brand=brand)
    # Training (sklearn fit + 5-fold CV) is blocking CPU work — run it off the
    # event loop so an inline caller (e.g. the admin /train endpoint) doesn't stall.
    metrics = await asyncio.to_thread(
        partial(generator.train, frame, target_col=target_column, algorithm=algorithm)
    )

    # Persist estimator + preprocessor bundle to MLflow (sync client → off-loop).
    ref = await asyncio.to_thread(twin_persistence.save_twin_artifacts, generator)

    config = TwinModelConfig(
        model_name=f"{twin_type.value}_{brand.value}_twin",
        model_description=(
            f"Digital-twin generative model for {brand.value}/{twin_type.value} "
            f"(data_provenance={provenance}, algorithm={algorithm})"
        ),
        twin_type=twin_type,
        brand=brand,
        algorithm=algorithm,
        feature_columns=list(generator.feature_columns),
        target_column=target_column,
        geographic_scope=geographic_scope,
    )
    model_id = await repo.save_model(
        config=config,
        metrics=metrics,
        model_artifact=generator.model,
        mlflow_run_id=ref.run_id,
        mlflow_model_uri=ref.model_uri,
    )

    logger.info(
        "Trained twin model %s for %s/%s (provenance=%s, R²=%.4f, n=%d)",
        model_id,
        brand.value,
        twin_type.value,
        provenance,
        metrics.r2_score,
        getattr(metrics, "training_samples", len(frame)),
    )
    return {
        "model_id": str(model_id),
        "run_id": ref.run_id,
        "model_uri": ref.model_uri,
        "r2_score": float(metrics.r2_score),
        "training_samples": int(getattr(metrics, "training_samples", len(frame))),
        "data_provenance": provenance,
        "twin_type": twin_type.value,
        "brand": brand.value,
    }


__all__ = ["train_and_persist_twin"]
