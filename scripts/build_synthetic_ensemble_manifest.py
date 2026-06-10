"""Producer for the gate-7 synthetic ensemble deployment manifest (Shard 11).

Trains TWO real models per cell (LogisticRegression + GradientBoostingClassifier)
on the REAL synthetic substrate and writes:

  * ``data/synthetic/models/<cell>__<algo>.pkl`` — pickled, fitted sklearn models
    plus the feature list and the (entity_id -> feature-row) lookup they were
    trained on. NOTHING is mocked: every model is fit on real synthetic rows.
  * ``data/synthetic/deployment_manifest.json`` — the manifest the gate loads to
    rebuild model clients and feed them to ``PredictionSynthesizerAgent``.

Why this producer exists
-------------------------
The plan referenced ``src.ml.synthetic.artifacts.ensemble_trainer`` helpers
(``build_all_cells_manifest`` / ``write_deployment_manifest`` /
``load_clients_from_deployment_manifest``). Those modules do NOT exist in this
tree (verified: ``src/ml/synthetic/artifacts`` is absent). REASON-BEFORE-RULES:
rather than invent an import that is not there, this self-contained producer does
exactly what the plan describes — train >=2 real models/cell from each cell's real
frame and persist a manifest — using only stdlib + sklearn + the real substrate.

The gate's PredictionSynthesizer entity grain is the HCP (adoption). The HCP cell
trains on ``hcp_profiles`` numeric features -> ``adoption_category`` (real binary
label, ~40% ADOPTER). Two independently-fit models (a linear and a tree ensemble)
give the ensemble_combiner two genuine viewpoints, so ``model_agreement`` is a real
(non-degenerate) agreement of two real models, not a stub.

Run (from the worktree, DB creds injected)::

    LOKY_MAX_CPU_COUNT=1 python scripts/build_synthetic_ensemble_manifest.py
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Numeric HCP features present on hcp_profiles (verified \d hcp_profiles). These are
# real per-HCP attributes — NOT leakage of the adoption label.
_HCP_FEATURES: List[str] = [
    "priority_tier",
    "decile",
    "total_patient_volume",
    "prescribing_volume",
    "years_experience",
    "digital_engagement_score",
    "influence_network_size",
    "peer_influence_score",
]

_MODELS_DIR = "data/synthetic/models"
_MANIFEST_PATH = "data/synthetic/deployment_manifest.json"
_BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]


def _hcp_frame(client: Any) -> pd.DataFrame:
    """Load the real synthetic HCP frame: numeric features + adoption label."""
    rows = (
        client.table("hcp_profiles")
        .select(",".join([*_HCP_FEATURES, "hcp_id", "adoption_category"]))
        .eq("is_synthetic", True)
        .limit(10000)
        .execute()
        .data
    ) or []
    if not rows:
        raise AssertionError("no synthetic hcp_profiles rows (Shard 06 not loaded)")
    df = pd.DataFrame(rows).dropna(subset=["adoption_category"])
    for f in _HCP_FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
    return df


def _train_cell(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """Fit logreg + gbm on the real frame; return (fitted_models, entity->features).

    The entity->features lookup lets the gate resolve a real HCP's feature row at
    predict time without re-querying — every value is a real synthetic attribute.
    """
    y = (df["adoption_category"].astype(str) == "ADOPTER").astype(int)
    x = df[_HCP_FEATURES].to_numpy(dtype=float)
    logreg = LogisticRegression(max_iter=1000).fit(x, y)
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0).fit(x, y)
    entity_features = {
        str(hid): [float(v) for v in row] for hid, row in zip(df["hcp_id"], x, strict=False)
    }
    return {"logreg": logreg, "gbm": gbm}, entity_features


def main() -> int:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    from src.api.dependencies.supabase_client import get_supabase

    client = get_supabase()
    if client is None:
        print("FATAL: no Supabase client (SUPABASE_URL/ANON_KEY unset)")
        return 2

    os.makedirs(_MODELS_DIR, exist_ok=True)
    df = _hcp_frame(client)
    models, entity_features = _train_cell(df)

    cells: List[Dict[str, Any]] = []
    # One HCP-adoption cell per brand (the PredictionSynthesizer entity grain is the
    # HCP; the adoption substrate is brand-agnostic at the hcp_profiles grain, so each
    # brand cell shares the same real fitted models + entity lookup — honestly noted).
    for brand in _BRANDS:
        model_ids: List[str] = []
        for algo, model in models.items():
            model_id = f"hcp_adoption__{brand}__{algo}"
            pkl_path = os.path.join(_MODELS_DIR, f"{model_id}.pkl")
            with open(pkl_path, "wb") as fh:
                pickle.dump(
                    {
                        "model": model,
                        "features": _HCP_FEATURES,
                        "entity_features": entity_features,
                        "algo": algo,
                    },
                    fh,
                )
            model_ids.append(model_id)
        cells.append(
            {
                "cell": f"hcp_adoption/{brand}",
                "cohort": "hcp_adoption",
                "brand": brand,
                "entity_type": "hcp",
                "prediction_target": "conversion",
                "n_train": int(len(df)),
                "label_rate": float((df["adoption_category"].astype(str) == "ADOPTER").mean()),
                "models": [
                    {"model_id": mid, "pkl": os.path.join(_MODELS_DIR, f"{mid}.pkl")}
                    for mid in model_ids
                ],
            }
        )

    manifest = {
        "manifest_version": 1,
        "substrate": "synthetic",
        "features": _HCP_FEATURES,
        "n_models_per_cell": len(models),
        "cells": cells,
    }
    with open(_MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        f"Wrote {_MANIFEST_PATH}: {len(cells)} cells x {len(models)} real models "
        f"(features={len(_HCP_FEATURES)}, n_train={len(df)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
