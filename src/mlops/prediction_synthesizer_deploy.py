"""Deploy real prediction-synthesizer models for go-live (#840).

The ``prediction_synthesizer`` agent is wired and reachable via chat but,
until now, produced no real prediction: the factory injected ``model_clients={}``
(no deployment manifest) and never a ``model_registry``, and the 72 rows in
``ml_model_registry`` were metadata-only (NULL ``artifact_path``) — nothing to
load. This module makes the agent functional by DEPLOYING real fitted models:

  1. ``train_target_models`` — train >=2 distinct, real sklearn models on the
     platform's synthetic CSU/Remibrutinib cohort (``synthetic_v2`` scenario C),
     the same DGP family the existing synthetic champions came from. Models are
     fit on a NAMED-column DataFrame so ``feature_names_in_`` is serialized with
     them and ``InProcessModelClient`` maps request features by name.
  2. ``serialize_and_write_manifest`` — pickle each model and emit a multi-model
     deployment manifest (the shape ``load_clients_from_deployment_manifest_file``
     consumes), keyed by ``model_name``.
  3. ``register_deployed_models`` — upsert ``ml_model_registry`` champion rows
     (``is_champion``, ``stage='production'``, ``artifact_path`` set,
     ``is_synthetic=False``) under a dedicated real experiment, so
     ``MLModelRegistryRepository.get_models_for_target`` resolves them.

Run as a CLI on the target box to make chat predictions go live:

    python -m src.mlops.prediction_synthesizer_deploy

The model artifacts and manifest are environment-specific (absolute local
paths) and are gitignored; this script regenerates them deterministically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.ml.synthetic_v2.api import generate_scenario
from src.ml.synthetic_v2.scenarios import ScenarioName

logger = logging.getLogger(__name__)

# The chat-reachable target these models serve. Must match
# ``ml_experiments.prediction_target`` so ``get_models_for_target`` resolves it.
PREDICTION_TARGET = "csu_treatment_initiation"
BRAND = "Remibrutinib"
MODEL_VERSION = "1.0"
DEPLOY_EXPERIMENT_NAME = "csu_treatment_initiation_live_v1"

# Default on-box locations the factory's ``_try_load_prod_model_clients`` finds
# (``data/deployment_manifest.json`` relative to the app CWD).
DEFAULT_ARTIFACT_DIR = Path("data/model_artifacts/csu_treatment_initiation")
DEFAULT_MANIFEST_PATH = Path("data/deployment_manifest.json")


@dataclass
class TrainedModel:
    """A real fitted model ready to deploy as a prediction client."""

    model_name: str
    model: Any
    feature_names: List[str]
    algorithm: str
    auc: float
    n_features: int
    training_samples: int


# Two distinct, genuinely-different real configurations. Different regularization
# (+ class weighting) yields different fitted coefficients -> a non-degenerate
# ensemble, not the same model twice.
_MODEL_CONFIGS = [
    ("csu_treatment_initiation_lr_full_v1", {"C": 1.0, "max_iter": 1000}),
    (
        "csu_treatment_initiation_lr_balanced_v1",
        {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"},
    ),
]


def train_target_models(
    *, n_total: int = 6000, seed: int = 42, name_suffix: str = ""
) -> List[TrainedModel]:
    """Train >=2 distinct real models on the synthetic CSU scenario-C cohort.

    Fits on a named-column DataFrame so each model carries ``feature_names_in_``
    (used by ``InProcessModelClient`` to map request features by name).

    ``name_suffix`` is appended to each model_name to allow an isolated
    namespace (e.g. integration tests deploying without clobbering go-live).
    """
    ds = generate_scenario(ScenarioName.C_TREATMENT_CSU_RESPONSE, seed=seed, n_total=n_total)
    feature_names = list(ds.metadata.feature_names)
    X_train = pd.DataFrame(ds.X_train, columns=feature_names)
    X_test = pd.DataFrame(ds.X_test, columns=feature_names)

    models: List[TrainedModel] = []
    for base_name, params in _MODEL_CONFIGS:
        model_name = f"{base_name}{name_suffix}"
        clf = LogisticRegression(**params)
        clf.fit(X_train, ds.y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(ds.y_test, proba))
        models.append(
            TrainedModel(
                model_name=model_name,
                model=clf,
                feature_names=feature_names,
                algorithm="logistic_regression",
                auc=auc,
                n_features=len(feature_names),
                training_samples=int(len(ds.y_train)),
            )
        )
        logger.info("Trained %s: test AUC=%.4f", model_name, auc)
    return models


def serialize_and_write_manifest(
    models: List[TrainedModel],
    artifact_dir: Path,
    manifest_path: Path,
) -> Dict[str, str]:
    """Pickle each model and write a multi-model deployment manifest.

    Returns a map ``{model_name: absolute_pickle_path}``. The manifest uses the
    ``{"spec": {"models": {name: {"model_uri": path}}}}`` shape consumed by
    ``load_clients_from_deployment_manifest_file``; URIs are absolute so they
    resolve regardless of the app's CWD.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    uri_map: Dict[str, str] = {}
    for m in models:
        pkl_path = (artifact_dir / f"{m.model_name}.pkl").resolve()
        with open(pkl_path, "wb") as fh:
            pickle.dump(m.model, fh)
        uri_map[m.model_name] = str(pkl_path)

    manifest = {"spec": {"models": {name: {"model_uri": uri} for name, uri in uri_map.items()}}}
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote deployment manifest %s (%d models)", manifest_path, len(uri_map))
    return uri_map


async def _get_or_create_experiment(
    client: Any, experiment_name: str = DEPLOY_EXPERIMENT_NAME
) -> str:
    """Resolve (or create) the dedicated real deploy experiment id."""
    existing = await (
        client.table("ml_experiments")
        .select("id")
        .eq("experiment_name", experiment_name)
        .limit(1)
        .execute()
    )
    if existing.data:
        return str(existing.data[0]["id"])

    row = {
        "experiment_name": experiment_name,
        "prediction_target": PREDICTION_TARGET,
        "brand": BRAND,
        "is_synthetic": False,
        "created_by": "prediction_synthesizer_deploy",
        "description": "Real deployable models backing live chat predictions (#840).",
    }
    created = await client.table("ml_experiments").insert(row).execute()
    return str(created.data[0]["id"])


async def register_deployed_models(
    client: Any,
    models: List[TrainedModel],
    uri_map: Dict[str, str],
    experiment_name: str = DEPLOY_EXPERIMENT_NAME,
) -> List[str]:
    """Upsert champion ``ml_model_registry`` rows for the deployed models.

    Idempotent: deletes any prior row with the same ``model_name`` first.
    Returns the registered model_names.
    """
    experiment_id = await _get_or_create_experiment(client, experiment_name)
    now = datetime.now(timezone.utc).isoformat()

    registered: List[str] = []
    for m in models:
        await client.table("ml_model_registry").delete().eq("model_name", m.model_name).execute()
        row = {
            "experiment_id": experiment_id,
            "model_name": m.model_name,
            "model_version": MODEL_VERSION,
            "algorithm": m.algorithm,
            "stage": "production",
            "is_champion": True,
            "is_synthetic": False,
            "artifact_path": uri_map[m.model_name],
            "auc": m.auc,
            "feature_count": m.n_features,
            "training_samples": m.training_samples,
            "trained_at": now,
            "registered_at": now,
            "promoted_at": now,
        }
        await client.table("ml_model_registry").insert(row).execute()
        registered.append(m.model_name)
        logger.info("Registered champion %s (auc=%.4f)", m.model_name, m.auc)
    return registered


async def deploy(
    *,
    n_total: int = 6000,
    seed: int = 42,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    register: bool = True,
    name_suffix: str = "",
    experiment_name: str = DEPLOY_EXPERIMENT_NAME,
) -> Dict[str, Any]:
    """Full deploy: train -> serialize -> manifest -> (optionally) register."""
    models = train_target_models(n_total=n_total, seed=seed, name_suffix=name_suffix)
    uri_map = serialize_and_write_manifest(models, artifact_dir, manifest_path)

    registered: List[str] = []
    if register:
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        if client is None:
            logger.warning("No Supabase client — artifacts+manifest written, registry NOT updated.")
        else:
            registered = await register_deployed_models(client, models, uri_map, experiment_name)

    return {
        "models": [m.model_name for m in models],
        "aucs": {m.model_name: m.auc for m in models},
        "manifest_path": str(Path(manifest_path).resolve()),
        "registered": registered,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Deploy real prediction_synthesizer models (#840)")
    parser.add_argument("--n-total", type=int, default=6000, help="synthetic cohort size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="write artifacts+manifest but skip DB registration",
    )
    args = parser.parse_args()

    result = asyncio.run(
        deploy(
            n_total=args.n_total,
            seed=args.seed,
            artifact_dir=args.artifact_dir,
            manifest_path=args.manifest_path,
            register=not args.no_register,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
