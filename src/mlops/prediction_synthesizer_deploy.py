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

# Default on-box locations, under the WRITABLE ``data/ml_artifacts/`` named
# volume (#857). In the prod api container ``/app/data`` is a read-only image
# directory; only named volumes mounted beneath it (here the ``e2i_ml_artifacts``
# volume at ``data/ml_artifacts``) are writable, so the documented runbook
# ``python -m src.mlops.prediction_synthesizer_deploy`` must write there or it
# fails with ``OSError: Read-only file system``. The manifest shares the same
# volume so it persists across redeploys and the factory's
# ``_try_load_prod_model_clients`` reads it back from the matching default path.
DEFAULT_ARTIFACT_DIR = Path("data/ml_artifacts/csu_treatment_initiation")
DEFAULT_MANIFEST_PATH = Path("data/ml_artifacts/deployment_manifest.json")


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


def serialize_model(model: Any, artifact_dir: Path, model_name: str) -> str:
    """Pickle one fitted model under *artifact_dir* and return its ABSOLUTE path.

    The single-model serialization primitive shared by this module's
    manifest-writing path and the gold-standard eval deployer (which serializes
    for loadability/honesty but does NOT write a serving manifest). The path is
    resolved to absolute so it loads regardless of the app's CWD.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = (artifact_dir / f"{model_name}.pkl").resolve()
    with open(pkl_path, "wb") as fh:
        pickle.dump(model, fh)
    return str(pkl_path)


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
    uri_map: Dict[str, str] = {}
    for m in models:
        uri_map[m.model_name] = serialize_model(m.model, artifact_dir, m.model_name)

    manifest = {"spec": {"models": {name: {"model_uri": uri} for name, uri in uri_map.items()}}}
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote deployment manifest %s (%d models)", manifest_path, len(uri_map))
    return uri_map


async def _get_or_create_experiment(
    client: Any,
    experiment_name: str = DEPLOY_EXPERIMENT_NAME,
    *,
    created_by: str = "prediction_synthesizer_deploy",
    description: str = "Real deployable models backing live chat predictions (#840).",
) -> str:
    """Resolve (or create) the dedicated real deploy experiment id.

    Matches on ``(experiment_name, prediction_target)`` — ``experiment_name``
    alone is not unique in the schema, and resolving the wrong target's row
    would register models that ``get_models_for_target`` could never surface.

    ``created_by`` and ``description`` are only written at INSERT time (the row
    is looked up by ``(experiment_name, prediction_target)``); callers that
    reuse this helper for a different experiment (e.g. the gold-standard eval
    deployer) should pass their own values so the DB row carries the correct
    provenance on first creation.
    """
    existing = await (
        client.table("ml_experiments")
        .select("id")
        .eq("experiment_name", experiment_name)
        .eq("prediction_target", PREDICTION_TARGET)
        .execute()
    )
    if existing.data:
        if len(existing.data) > 1:
            raise RuntimeError(
                f"ambiguous experiment '{experiment_name}' for target "
                f"'{PREDICTION_TARGET}' ({len(existing.data)} rows) — refusing to guess"
            )
        return str(existing.data[0]["id"])

    row = {
        "experiment_name": experiment_name,
        "prediction_target": PREDICTION_TARGET,
        "brand": BRAND,
        "is_synthetic": False,
        "created_by": created_by,
        "description": description,
    }
    created = await client.table("ml_experiments").insert(row).execute()
    if not created.data:
        raise RuntimeError(f"failed to create experiment '{experiment_name}' (no row returned)")
    return str(created.data[0]["id"])


async def register_model_row(
    client: Any,
    *,
    experiment_id: str,
    model_name: str,
    model_version: str,
    algorithm: str,
    artifact_path: str,
    auc: float,
    feature_count: int,
    training_samples: int | None = None,
    stage: str = "production",
    is_champion: bool = True,
    is_synthetic: bool = False,
    promoted: bool | None = None,
) -> str:
    """Write ONE ``ml_model_registry`` row idempotently, then verify it landed.

    The shared registration primitive. Idempotent at the
    ``(model_name, model_version)`` grain (the schema unique key) — replaces only
    the matching version, preserving other versions and avoiding FK fallout from
    a name-wide delete. Verifies the artifact exists on disk before writing, and
    reads the row back to confirm it actually landed AT THE INTENDED ``stage``
    (a trigger/RLS no-op, or a trigger that demotes the stage, must not be
    reported as success). Returns the registered ``model_name``.

    ``promoted`` defaults to "is this a production row" — production rows record
    a ``promoted_at`` timestamp, staging/shadow rows do not (they are not
    promoted). ``stage`` must be a valid ``model_stage_enum`` value; the caller
    is responsible for choosing a non-colliding stage (e.g. the gold-standard
    eval deployer uses ``stage='staging'`` so it is not surfaced by
    ``get_models_for_target``'s production-only serving filter).
    """
    if not Path(artifact_path).is_file():
        raise RuntimeError(
            f"artifact for {model_name} missing at {artifact_path} — refusing to "
            "register an unloadable model"
        )
    if promoted is None:
        promoted = stage == "production"
    now = datetime.now(timezone.utc).isoformat()

    # Replace only this exact (model_name, model_version) — not all versions.
    await (
        client.table("ml_model_registry")
        .delete()
        .eq("model_name", model_name)
        .eq("model_version", model_version)
        .execute()
    )
    row = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "model_version": model_version,
        "algorithm": algorithm,
        "stage": stage,
        "is_champion": is_champion,
        "is_synthetic": is_synthetic,
        "artifact_path": artifact_path,
        "auc": auc,
        "feature_count": feature_count,
        "trained_at": now,
        "registered_at": now,
    }
    if training_samples is not None:
        row["training_samples"] = training_samples
    if promoted:
        row["promoted_at"] = now
    await client.table("ml_model_registry").insert(row).execute()

    # Confirm the row actually landed at the INTENDED stage with the artifact
    # (no silent no-op, and not demoted to another stage by a trigger).
    check = await (
        client.table("ml_model_registry")
        .select("model_name, stage, artifact_path")
        .eq("model_name", model_name)
        .eq("model_version", model_version)
        .execute()
    )
    landed = [
        r
        for r in (check.data or [])
        if r.get("artifact_path") == artifact_path and r.get("stage") == stage
    ]
    if not landed:
        raise RuntimeError(
            f"registration of {model_name} did not persist as a {stage} row "
            f"(read-back found no matching {stage} row) — refusing to report it "
            "as deployed"
        )
    logger.info("Registered %s model %s (auc=%.4f)", stage, model_name, auc)
    return model_name


async def register_deployed_models(
    client: Any,
    models: List[TrainedModel],
    uri_map: Dict[str, str],
    experiment_name: str = DEPLOY_EXPERIMENT_NAME,
) -> List[str]:
    """Register production ``ml_model_registry`` rows for the deployed models.

    Idempotent at the ``(model_name, model_version)`` grain (the schema unique
    key) — replaces only the matching version, preserving other versions and
    avoiding FK fallout from a name-wide delete. Verifies the artifact exists
    before writing, and reads the row back to confirm it actually landed
    (a trigger/RLS no-op must not be reported as success).
    """
    experiment_id = await _get_or_create_experiment(client, experiment_name)

    registered: List[str] = []
    for m in models:
        await register_model_row(
            client,
            experiment_id=experiment_id,
            model_name=m.model_name,
            model_version=MODEL_VERSION,
            algorithm=m.algorithm,
            artifact_path=uri_map[m.model_name],
            auc=m.auc,
            feature_count=m.n_features,
            training_samples=m.training_samples,
            stage="production",
            is_champion=True,
            is_synthetic=False,
        )
        registered.append(m.model_name)
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


async def verify(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    """Post-deploy consistency check before enabling live chat traffic.

    Confirms the three pieces agree end-to-end: (1) the manifest loads into
    clients, (2) the live registry resolves the deployed model names for the
    target, and (3) each loaded client returns a real prediction. Returns a
    report and raises if the wiring is inconsistent. Run before treating the
    agent as traffic-eligible.
    """
    from src.agents.prediction_synthesizer.clients.inproc_model_client import (
        load_clients_from_deployment_manifest_file,
    )
    from src.agents.prediction_synthesizer.registry_adapter import LiveChampionModelRegistry

    clients = load_clients_from_deployment_manifest_file(str(manifest_path))
    if not clients:
        raise RuntimeError(f"manifest {manifest_path} loaded no clients")

    registry = LiveChampionModelRegistry()
    resolved = set(await registry.get_models_for_target(PREDICTION_TARGET, "hcp"))
    # Traffic eligibility requires EXACT agreement: the agent resolves model
    # names from the registry at runtime and then looks them up in the loaded
    # clients, so any mismatch in either direction is a serving defect.
    missing_in_registry = set(clients) - resolved
    if missing_in_registry:
        raise RuntimeError(
            f"registry does not resolve manifest models for {PREDICTION_TARGET}: "
            f"{sorted(missing_in_registry)}"
        )
    missing_in_manifest = resolved - set(clients)
    if missing_in_manifest:
        raise RuntimeError(
            f"registry has production models with no manifest client (the agent would "
            f"request them and fail): {sorted(missing_in_manifest)}"
        )

    checked: Dict[str, float] = {}
    for name, client in clients.items():
        feats = dict.fromkeys(client.feature_names, 1.0)
        out = await client.predict("VERIFY_ENTITY", feats, "30d")
        # A client that swallows an inference failure into a synthetic neutral
        # result must NOT pass verification as a real prediction.
        if out.get("error"):
            raise RuntimeError(f"model {name} failed inference during verify: {out['error']}")
        checked[name] = float(out["prediction"])

    return {
        "manifest_models": sorted(clients),
        "resolved_by_registry": sorted(resolved),
        "predictions": checked,
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing deploy (manifest <-> registry <-> live prediction) and exit",
    )
    args = parser.parse_args()

    if args.verify:
        report = asyncio.run(verify(manifest_path=args.manifest_path))
        print(json.dumps(report, indent=2))
        return

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
