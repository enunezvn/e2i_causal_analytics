"""Synthetic MLOps substrate (Shard 09): ml_model_registry / ml_training_runs /
ml_deployments.

The faithful DB carries 0 rows in all three (audit F4/F9). Generate a registry +
training runs + deployments consistent with the experiments frame (>=2 models per
experiment, first = champion -> production -> active deployment) so
model_selector.historical_analyzer and model_deployer have real synthetic rows to
read once their own code defects (out of scope) are fixed. is_synthetic=true.

Enum-exact (22P02 landmine): model_stage_enum (stage) and deployment_status_enum
(status). auc lands in the [0.62, 0.90] non-degenerate band.

IDEMPOTENT (reseed-safe): all three ids are DETERMINISTIC uuid5 derived from the
(model_name, model_version) natural key (see ``_mlops_id``), so a reseed UPDATES the
prior rows in place instead of minting fresh-uuid rows that collide on
``unique_model_version`` (23505) and orphan the children on a never-inserted
``model_registry_id`` (23503). The registry/run/deployment ids are stable across runs
given the same ``experiments_df``.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd

from .base import BaseGenerator, GeneratorConfig

_ALGOS = ["xgboost", "lightgbm", "logistic_regression", "random_forest"]

# Fixed namespace for DETERMINISTIC ids. uuid.uuid4() ignores the seed, so every run
# minted fresh registry/run/deployment PKs; the loader upserts on the PK, so a reseed
# could never UPDATE the prior row in place — it INSERTed and collided with the
# secondary UNIQUE(model_name, model_version) (unique_model_version) -> 23505 -> 0
# loaded -> ml_training_runs/ml_deployments orphaned on the never-inserted
# model_registry_id -> 23503 cascade. Deriving each id by uuid5 from its NATURAL KEY
# makes the whole MLOps sub-graph idempotent (upsert-on-PK UPDATES in place, like
# patient_journeys) AND keeps the child FK (model_registry_id) pointing at the same
# registry id across runs. Mirrors the #852 lesson ("fresh random UUIDs every run" was
# the bug) at the source rather than patching the loader.
_MLOPS_ID_NS = uuid.UUID("9f2c1e7a-3b4d-5e6f-8a90-1c2d3e4f5a6b")


def _mlops_id(*parts: str) -> str:
    """Deterministic uuid5 from a natural key (stable across runs given the same
    experiments_df), e.g. _mlops_id(model_name, model_version)."""
    return str(uuid.uuid5(_MLOPS_ID_NS, "|".join(parts)))


class MLOpsGenerator(BaseGenerator[pd.DataFrame]):
    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        experiments_df: Optional[pd.DataFrame] = None,
        models_per_experiment: int = 2,
    ):
        super().__init__(config)
        if experiments_df is None or experiments_df.empty:
            raise ValueError("MLOpsGenerator requires a non-empty experiments_df")
        self.experiments_df = experiments_df
        self.models_per_experiment = models_per_experiment

    @property
    def entity_type(self) -> str:
        return "ml_model_registry"

    def generate(self) -> Dict[str, pd.DataFrame]:  # type: ignore[override]
        now = datetime.now(timezone.utc)
        reg, runs, dep = [], [], []
        for _, exp in self.experiments_df.iterrows():
            for m in range(self.models_per_experiment):
                algo = _ALGOS[m % len(_ALGOS)]
                model_name = f"{exp['experiment_name']}_model_{m}"
                model_version = f"1.{m}"
                # Deterministic PK from the (model_name, model_version) natural key so a
                # reseed UPDATES in place instead of colliding on unique_model_version.
                rid = _mlops_id(model_name, model_version)
                auc = round(float(self._rng.uniform(0.62, 0.90)), 4)
                is_champ = m == 0  # first model per experiment is champion
                reg.append(
                    {
                        "id": rid,
                        "experiment_id": exp["id"],
                        "model_name": model_name,
                        "model_version": model_version,
                        "algorithm": algo,
                        "feature_count": int(self._rng.integers(8, 40)),
                        "training_samples": int(self._rng.integers(2000, 8000)),
                        "auc": auc,
                        "pr_auc": round(auc - float(self._rng.uniform(0.05, 0.15)), 4),
                        "brier_score": round(float(self._rng.uniform(0.08, 0.20)), 4),
                        "calibration_slope": round(float(self._rng.uniform(0.85, 1.10)), 4),
                        "stage": "production" if is_champ else "staging",
                        "is_champion": is_champ,
                        "trained_at": (
                            now - timedelta(days=int(self._rng.integers(1, 20)))
                        ).isoformat(),
                        "registered_at": now.isoformat(),
                        "is_synthetic": True,
                    }
                )
                runs.append(
                    {
                        "id": _mlops_id("run", model_name, model_version),
                        "experiment_id": exp["id"],
                        "model_registry_id": rid,
                        "run_name": f"run_{rid[:8]}",
                        "algorithm": algo,
                        "hyperparameters": {"n_estimators": 200, "max_depth": 6},
                        "training_samples": int(self._rng.integers(2000, 8000)),
                        "validation_samples": int(self._rng.integers(500, 2000)),
                        "test_samples": int(self._rng.integers(500, 2000)),
                        "feature_names": [
                            "disease_severity",
                            "academic_hcp",
                            "engagement_score",
                        ],
                        "train_metrics": {"auc": auc + 0.03},
                        "validation_metrics": {"auc": auc},
                        "test_metrics": {"auc": auc - 0.01},
                        "status": "completed",
                        "started_at": (now - timedelta(hours=2)).isoformat(),
                        "completed_at": now.isoformat(),
                        "duration_seconds": int(self._rng.integers(60, 1800)),
                        "is_best_trial": is_champ,
                        "is_synthetic": True,
                    }
                )
                if is_champ:
                    dep.append(
                        {
                            "id": _mlops_id("deploy", model_name, model_version),
                            "model_registry_id": rid,
                            "deployment_name": f"deploy_{exp['experiment_name']}",
                            "environment": "production",
                            "endpoint_name": f"ep_{rid[:8]}",
                            "status": "active",
                            "deployed_by": "synthetic_loader",
                            "deployment_config": {"replicas": 2},
                            "production_metrics": {
                                "requests_24h": int(self._rng.integers(100, 5000))
                            },
                            "created_at": now.isoformat(),
                            "deployed_at": now.isoformat(),
                            "latency_p50_ms": int(self._rng.integers(20, 80)),
                            "latency_p95_ms": int(self._rng.integers(80, 250)),
                            "error_rate": round(float(self._rng.uniform(0.0, 0.02)), 4),
                            "is_synthetic": True,
                        }
                    )
        return {
            "ml_model_registry": pd.DataFrame(reg),
            "ml_training_runs": pd.DataFrame(runs),
            "ml_deployments": pd.DataFrame(dep),
        }
