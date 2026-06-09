"""Train >=2 diverse models per (cohort,brand) on the Shard-03 synthetic DGP frame
and emit a deployment_manifest (Shard 08 T3).

The manifest is consumed by ``prediction_synthesizer`` via
``inproc_model_client._iter_models_in_manifest`` (reads
``manifest["spec"]["models"][model_id]["model_uri"]``) and
``load_clients_from_deployment_manifest`` -> one ``InProcessModelClient`` per entry.

Two models per cell give prediction_synthesizer the diversity it needs to ESCAPE
the single-model 0.30 / CANNOT_ASSESS cap (ensemble_combiner.py:74: confidence is
capped when models_succeeded < 2). Both fit on the same synthetic frame whose true
effect (Shard 03) is known, so recovery is verifiable.

Coverage = ALL 12 cells (INDEX §CANONICAL SSOT: prediction_synthesizer = all 12
cells). ``build_all_cells_manifest`` is a parameterized loop over the full 4x3
matrix, training >=2 models per cell (>=24 manifest entries).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# The full 4x3 coverage matrix (INDEX §CANONICAL SSOT: prediction_synthesizer = all
# 12 cells). Cohorts resolved by outcome column; brands are the exact brand_type enum
# labels. Every cell gets >=2 trained models + >=2 deployment_manifest entries.
COHORTS: List[str] = ["initiation", "discontinuation", "persistence", "hcp_adoption"]
BRANDS: List[str] = ["Remibrutinib", "Kisqali", "Fabhalta"]
ALL_CELLS: List[Tuple[str, str]] = [(c, b) for c in COHORTS for b in BRANDS]


def train_ensemble_for_cohort_brand(
    X: pd.DataFrame, y: Any, *, cohort: str, brand: str, out_dir: Path
) -> List[Path]:
    """Fit LogisticRegression + GradientBoosting (diverse families) and pickle both.

    Two DIFFERENT model families guarantee a genuine disagreement signal for the
    ensemble's model_agreement (>0.5 when they broadly concur, <1.0 since distinct).
    feature_names_in_ is preserved by sklearn so InProcessModelClient aligns features.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models: Dict[str, Any] = {
        "logreg": LogisticRegression(max_iter=1000),
        "gbm": GradientBoostingClassifier(random_state=0),
    }
    paths: List[Path] = []
    for fam, est in models.items():
        est.fit(X, y)
        p = out_dir / f"{cohort}__{brand}__{fam}.pkl"
        with p.open("wb") as fh:
            pickle.dump(est, fh)
        paths.append(p)
    return paths


def build_deployment_manifest(
    by_cohort_brand: Dict[Tuple[str, str], List[Path]],
) -> Dict[str, Any]:
    """Assemble the multi-model manifest (spec.models[model_id].model_uri shape).

    model_id encodes cohort/brand/family so prediction_synthesizer's manifest carries
    >=2 models per (cohort,brand). model_uri is a resolved local file path
    (_load_model_from_uri treats a bare existing path as a pickle).
    """
    models: Dict[str, Dict[str, str]] = {}
    for (_cohort, _brand), paths in by_cohort_brand.items():
        for p in paths:
            model_id = Path(p).stem  # e.g. "initiation__Kisqali__logreg"
            models[model_id] = {"model_uri": str(Path(p).resolve())}
    return {
        "apiVersion": "e2i/v1",
        "kind": "DeploymentManifest",
        "metadata": {"name": "synthetic-causal-validation-ensemble"},
        "spec": {"models": models},
    }


def write_deployment_manifest(manifest: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def build_all_cells_manifest(
    frame_loader: Callable[[str, str], Optional[Tuple[pd.DataFrame, Any]]],
    *,
    out_dir: Path,
) -> Dict[str, Any]:
    """Train >=2 models for EVERY (cohort,brand) in ALL_CELLS and assemble one
    deployment_manifest that enumerates all 12 cells (>=2 entries per cell, >=24 total).

    ``frame_loader(cohort, brand) -> (X, y)`` on that cell's Shard-03/06 DGP frame,
    or ``None`` to SKIP a cell that has no substrate yet (fail-closed; never fabricate
    a cell). Cohorts are resolved by their canonical outcome column
    (treatment_initiated / discontinued_180d / persistent_180d /
    hcp_profiles.adoption_category) — see INDEX §CANONICAL SSOT. A skipped cell is
    LOGGED, never silently truncated; if ANY cell lacks substrate the assembly raises
    so the manifest can never claim partial coverage.
    """
    by_cell: Dict[Tuple[str, str], List[Path]] = {}
    skipped: List[Tuple[str, str]] = []
    for cohort, brand in ALL_CELLS:
        loaded = frame_loader(cohort, brand)
        if loaded is None:
            skipped.append((cohort, brand))
            logger.warning(
                "ensemble_trainer: no substrate for cell (%s, %s) -> skipping", cohort, brand
            )
            continue
        X, y = loaded
        by_cell[(cohort, brand)] = train_ensemble_for_cohort_brand(
            X, y, cohort=cohort, brand=brand, out_dir=Path(out_dir)
        )
    if skipped:
        raise ValueError(
            "deployment_manifest must cover all 12 cells; missing substrate for "
            f"{skipped} (each cell needs a resolvable Shard-03/06 DGP frame)"
        )
    return build_deployment_manifest(by_cell)


__all__ = [
    "ALL_CELLS",
    "BRANDS",
    "COHORTS",
    "build_all_cells_manifest",
    "build_deployment_manifest",
    "train_ensemble_for_cohort_brand",
    "write_deployment_manifest",
]
