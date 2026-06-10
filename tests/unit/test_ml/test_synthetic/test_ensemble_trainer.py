"""Shard 08 T3 — train >=2 diverse models per (cohort,brand) over ALL 12 cells of
the 4x3 matrix + emit a deployment_manifest in the shape
inproc_model_client._iter_models_in_manifest expects (spec.models[id].model_uri).

>=2 models per cell give prediction_synthesizer the diversity it needs to ESCAPE
the single-model 0.30/CANNOT_ASSESS cap (ensemble_combiner.py:74). Hermetic: tiny
in-memory frames, no DB.
"""

import numpy as np
import pandas as pd

from src.agents.prediction_synthesizer.clients.inproc_model_client import (
    load_clients_from_deployment_manifest,
)
from src.ml.synthetic.artifacts.ensemble_trainer import (
    ALL_CELLS,
    BRANDS,
    COHORTS,
    build_deployment_manifest,
    train_ensemble_for_cohort_brand,
)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "disease_severity": rng.integers(0, 3, n).astype(float),
            "age_at_diagnosis": rng.normal(55, 10, n),
            "treatment_arm": rng.integers(0, 2, n),  # canonical treatment-arm column
        }
    )
    y = (
        0.3 + 0.4 * X["treatment_arm"] + 0.1 * X["disease_severity"] > rng.uniform(0, 1, n)
    ).astype(int)
    return X, y


def test_trains_two_models_and_manifest_yields_two_clients(tmp_path):
    X, y = _frame()
    paths = train_ensemble_for_cohort_brand(
        X, y, cohort="initiation", brand="Kisqali", out_dir=tmp_path
    )
    assert len(paths) >= 2, "must persist >=2 model artifacts per (cohort,brand)"
    manifest = build_deployment_manifest({("initiation", "Kisqali"): paths})
    models = manifest["spec"]["models"]
    assert len([m for m in models if "Kisqali" in m]) >= 2
    clients = load_clients_from_deployment_manifest(manifest)
    assert len(clients) >= 2, "deployment_manifest must load >=2 InProcessModelClients"


def test_manifest_enumerates_all_12_cells_two_models_each(tmp_path):
    # ALL_CELLS is the 4x3 matrix: {initiation,discontinuation,persistence,hcp_adoption}
    # x {Remibrutinib,Kisqali,Fabhalta}. Manifest must enumerate every cell (>=2 models).
    assert len(ALL_CELLS) == 12 and len(COHORTS) == 4 and len(BRANDS) == 3
    by_cell = {}
    for i, (cohort, brand) in enumerate(ALL_CELLS):
        X, y = _frame(seed=i)
        by_cell[(cohort, brand)] = train_ensemble_for_cohort_brand(
            X, y, cohort=cohort, brand=brand, out_dir=tmp_path
        )
    manifest = build_deployment_manifest(by_cell)
    models = manifest["spec"]["models"]
    assert len(models) >= 24, "12 cells x >=2 models = >=24 manifest entries"
    for cohort, brand in ALL_CELLS:
        cell_models = [m for m in models if m.startswith(f"{cohort}__{brand}__")]
        assert len(cell_models) >= 2, f"cell ({cohort},{brand}) needs >=2 models"
