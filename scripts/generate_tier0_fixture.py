#!/usr/bin/env python
"""Generate the committed Tier-0 cache fixture for the Tier 1-5 harness (#600).

WHY THIS EXISTS
---------------
The Tier 1-5 agent harness CI workflow (``.github/workflows/tier1-5-test.yml``)
only *executes the 13 agents* when a Tier-0 cache is present at
``scripts/tier0_output_cache/latest.pkl``. That path was gitignored, so the
cache was never committed and the harness graceful-skipped the agents on every
PR — a no-op for its primary purpose (catching schema drift between Tier-0
state, the ``Tier0OutputMapper``, and per-agent contracts). See issue #600.

The maintainer-decided fix is to commit a small, **sanitized** Tier-0 fixture
and un-ignore that single file. This script is the deterministic source of
truth + **refresh mechanism** for that fixture: re-run it to regenerate
``latest.pkl`` after a contract change.

DESIGN (sanitized, version-robust, faithful enough to drive all 13 agents)
--------------------------------------------------------------------------
* ``eligible_df`` is produced by the REAL Tier-0 cohort generator
  (``SampleDataGenerator.ml_patients`` — the same generator the data_preparer
  uses), so its schema is faithful to real Tier-0 output. We keep it small
  (a few hundred rows) — enough for the mappers (numeric effect-modifiers for
  ``heterogeneous_optimizer``, a ``discontinuation_flag == 1`` row for
  ``prediction_synthesizer``) without bloating the committed cache.
* The only model object is a tiny ``LogisticRegression`` fit on the numeric
  feature columns. ``Tier0ModelClient`` feeds it a purely-numeric vector via
  ``predict_proba``, so no fitted preprocessor / encoder is needed. We
  deliberately DO NOT embed the version-fragile fitted ``ColumnTransformer`` /
  ``OrdinalEncoder`` objects a full Tier-0 run produces.
* All other keys are plain ``dict`` / ``list`` / ``str`` / ``float`` / ``bool``
  metric summaries — version-robust and within the ``Tier0StateContract``
  whitelist.

This is a TEST FIXTURE, not a production artifact: the scalar metric values are
realistic-but-synthetic. Its job is to exercise the harness plumbing (the 13
mappers + agent runs + contract validation), not to represent a real model's
measured performance.

USAGE
-----
    python scripts/generate_tier0_fixture.py            # writes latest.pkl
    python scripts/generate_tier0_fixture.py --rows 600 # custom cohort size
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.repositories.sample_data import SampleDataGenerator

# Default committed fixture location — both the workflow restore-cache step and
# the Makefile ``tier1-5-test`` target resolve exactly this path.
FIXTURE_PATH = Path(__file__).resolve().parent / "tier0_output_cache" / "latest.pkl"

DEFAULT_ROWS = 600
SEED = 42

# Numeric feature columns the tiny model is trained on. Tier0ModelClient reads
# ``model.feature_names_in_`` (set because we fit on a DataFrame) and assembles a
# numeric vector from the patient row, so these must be numeric.
_NUMERIC_FEATURES = ["days_on_therapy", "hcp_visits", "prior_treatments", "data_quality_score"]


def build_fixture_state(rows: int = DEFAULT_ROWS, seed: int = SEED) -> dict[str, Any]:
    """Build the sanitized Tier-0 state dict deterministically.

    Returns a dict whose keys are all within ``Tier0StateContract`` and which
    drives every one of the 13 ``map_to_*`` methods without raising.
    """
    gen = SampleDataGenerator(seed=seed)
    eligible_df = gen.ml_patients(n_patients=rows)

    # Guarantee the prediction_synthesizer precondition (a discontinuation_flag
    # == 1 row) even at small cohort sizes / unlucky seeds.
    if not (eligible_df["discontinuation_flag"] == 1).any():
        eligible_df.loc[eligible_df.index[0], "discontinuation_flag"] = 1

    # Tiny, version-robust model: LogisticRegression on the numeric features.
    X = eligible_df[_NUMERIC_FEATURES].astype(float)
    y = eligible_df["discontinuation_flag"].astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Block-4 contract: persist entity -> split label so downstream consumers
    # reuse splits instead of re-deriving them.
    ids = eligible_df["patient_journey_id"].tolist()
    rng = np.random.default_rng(seed)
    labels = rng.choice(["train", "val", "test"], size=len(ids), p=[0.6, 0.2, 0.2])
    split_assignments = dict(zip(ids, labels.tolist(), strict=True))

    feature_importance = [
        {"feature": "prior_treatments", "importance": 0.31},
        {"feature": "hcp_visits", "importance": 0.27},
        {"feature": "days_on_therapy", "importance": 0.22},
        {"feature": "data_quality_score", "importance": 0.12},
        {"feature": "age_group", "importance": 0.05},
        {"feature": "geographic_region", "importance": 0.03},
    ]
    validation_metrics = {
        "accuracy": 0.81,
        "roc_auc": 0.84,
        "precision": 0.74,
        "recall": 0.69,
        "f1_score": 0.71,
    }

    state: dict[str, Any] = {
        # Contract-Required.
        "experiment_id": "tier0_fixture_600",
        "eligible_df": eligible_df,
        # Read by the mappers.
        "scope_spec": {
            "brand": "Kisqali",
            "problem_type": "binary_classification",
            "target_outcome": "discontinuation_flag",
        },
        "feature_importance": feature_importance,
        "feature_names": list(_NUMERIC_FEATURES),
        "validation_metrics": validation_metrics,
        "train_metrics": {"accuracy": 0.85, "roc_auc": 0.88, "precision": 0.79, "recall": 0.74},
        "test_metrics": {"accuracy": 0.80, "roc_auc": 0.83, "precision": 0.73, "recall": 0.67},
        "test_metrics_at_optimal": {"precision": 0.71, "recall": 0.72, "f1_score": 0.715},
        "test_metrics_at_05": {"precision": 0.76, "recall": 0.61, "f1_score": 0.676},
        # Tiny model object (sanitized — no fitted preprocessor/encoder).
        "trained_model": model,
        "model_uri": "models:/tier0_fixture_600/1",
        "optimal_threshold": 0.42,
        # Block-4 splits.
        "split_assignments": split_assignments,
        "split_strategy": "stratified_random",
        # Plain-data status fields (faithful, version-robust).
        "gate_passed": True,
        "success_criteria_met": True,
        "class_imbalance_info": {
            "imbalance_detected": True,
            "minority_ratio": round(float(y.mean()), 4),
            "severity": "moderate",
        },
        "deployment_manifest": {
            "model_id": "tier0_fixture_600",
            "approved": True,
            "gate": "N1",
        },
    }
    return state


def write_fixture(path: Path = FIXTURE_PATH, rows: int = DEFAULT_ROWS, seed: int = SEED) -> Path:
    """Build and pickle the fixture to ``path`` (as a real file, not a symlink)."""
    state = build_fixture_state(rows=rows, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    # If a prior symlink (from a local --run-tier0-first regen) exists, drop it
    # so we write a real committed file.
    if path.is_symlink() or path.exists():
        path.unlink()
    with open(path, "wb") as fh:
        pickle.dump(state, fh)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="cohort size (default 600)")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed (default 42)")
    parser.add_argument("--path", type=Path, default=FIXTURE_PATH, help="output .pkl path")
    args = parser.parse_args()

    out = write_fixture(path=args.path, rows=args.rows, seed=args.seed)
    size_kb = out.stat().st_size / 1024
    print(f"Wrote tier0 fixture: {out}  ({size_kb:.1f} KB, rows={args.rows}, seed={args.seed})")


if __name__ == "__main__":
    main()
