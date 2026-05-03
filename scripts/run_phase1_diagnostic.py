#!/usr/bin/env python3
"""Phase 1 W4 Day 1 — diagnostic-regime artifact runner.

Per shard 17 W4 Day 1: "Run Phase 1 pipeline across Scenario A
(diagnostic); collect NB-area + Brier + AUROC + decision curves for all
candidates. Output written to `docs/results/phase1_diagnostic_YYYYMMDD.json`."

Generator-gating mitigation (shard 17 line 106): when
`synthetic_data_generator_v2/` Scenario A is not yet delivered, this
script falls back to tier0-style synthetic data with
diagnostic-regime characteristics (prev≈0.20-0.30, N=1500). Re-run with
``--scenario A`` to consume the canonical Scenario A fixture once the
generator ships.

Usage::

    python scripts/run_phase1_diagnostic.py
    python scripts/run_phase1_diagnostic.py --output docs/results/phase1_diagnostic_20260503.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure repo root on path so ``src.*`` imports resolve when this script
# is invoked from anywhere under the repo.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (  # noqa: E402
    evaluate_model,
)
from src.agents.ml_foundation.model_trainer.nodes.model_trainer_node import (  # noqa: E402
    train_model,
)

logger = logging.getLogger(__name__)


# Diagnostic-regime parameters per shard 17 Scenario A:
#   prevalence ≈ 0.20, N = 6000 (we use 1500 for the placeholder run to
#   keep wall-clock under a minute on the droplet).
SEED = 42
N_TRAIN = 1000
N_VAL = 250
N_TEST = 250
N_FEATURES = 8

# Algorithms exercised in the placeholder run. NGBoost / NGBoost_Conformal /
# LightGBM_Monotone shipped in W2 (commits 9b852e2..526ccc0); LightGBM /
# XGBoost custom Brier-objective variants shipped in W4 day-1/2 (commits
# 400aff9 / d697f0b).
#
# Cycle-24 I-1 disclosure: the ``LightGBM`` entry below uses the DEFAULT
# logloss objective for fast smoke coverage. The W4 primary deliverable
# is the Brier-custom-objective path (`objective=callable` per shard 17
# W3 row Day 1 / Day 2 footnote [1]); a Scenario A canonical run MUST
# add a parallel entry with the custom objective so the per-algorithm
# Brier-decomposition fields can be compared apples-to-apples.
PLACEHOLDER_ALGORITHMS: List[Dict[str, Any]] = [
    {
        "name": "LogisticRegression",
        "hp": {"C": 1.0, "solver": "lbfgs", "max_iter": 500},
        "candidate": {},
    },
    {
        "name": "LightGBM",
        "hp": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "verbosity": -1,
        },
        "candidate": {},
    },
]


def _make_diagnostic_synthetic(
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    n_features: int = N_FEATURES,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Diagnostic-regime synthetic with prev ≈ 0.27, separable signal.

    Used as a placeholder until `synthetic_data_generator_v2/` Scenario A
    ships. The label is driven by a moderate logit with intercept −0.5,
    slope 1.5 on the first feature; the resulting prevalence is
    E[sigmoid(1.5·Z − 0.5)] ≈ 0.27 for Z ~ N(0,1).
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    X = rng.standard_normal((n, n_features))
    logits = 1.5 * X[:, 0] - 0.5
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val": X[n_train : n_train + n_val],
        "y_val": y[n_train : n_train + n_val],
        "X_test": X[n_train + n_val :],
        "y_test": y[n_train + n_val :],
    }


def _build_state(
    algorithm_name: str,
    hp: Dict[str, Any],
    candidate: Dict[str, Any],
    data: Dict[str, np.ndarray],
    success_criteria: Dict[str, Any],
) -> Dict[str, Any]:
    """Minimal trainer state shared across all candidate algorithms."""
    return {
        "algorithm_name": algorithm_name,
        "best_hyperparameters": hp,
        "problem_type": "binary_classification",
        "X_train_preprocessed": data["X_train"],
        "X_validation_preprocessed": data["X_val"],
        "X_test_preprocessed": data["X_test"],
        "train_data": {"y": data["y_train"]},
        "validation_data": {"y": data["y_val"]},
        "test_data": {"y": data["y_test"]},
        "success_criteria": success_criteria,
        "early_stopping": False,
        "early_stopping_patience": 10,
        "model_candidate": candidate,
    }


def _extract_artifact_fields(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    """Slice the artifact-relevant fields out of an evaluator return dict.

    Keeps the JSON small + readable; the full eval_result has many
    internal/audit fields that are noise for the diagnostic report.
    """
    test_metrics = eval_result.get("test_metrics") or {}
    cal = eval_result.get("calibration_analysis") or {}
    return {
        "auroc": test_metrics.get("roc_auc"),
        "brier_score": test_metrics.get("brier_score"),
        "brier_reliability": cal.get("brier_reliability"),
        "brier_resolution": cal.get("brier_resolution"),
        "brier_uncertainty": cal.get("brier_uncertainty"),
        "brier_recombined": cal.get("brier_recombined"),
        "brier_decomposition_residual": cal.get("brier_decomposition_residual"),
        "calibration_ece": test_metrics.get("calibration_ece"),
        "calibration_slope": test_metrics.get("calibration_slope"),
        "calibration_intercept": test_metrics.get("calibration_intercept"),
        "net_benefit_area": test_metrics.get("net_benefit_area"),
        "net_benefit_area_treat_all": test_metrics.get("net_benefit_area_treat_all"),
        "net_benefit_area_relative_to_treat_all": test_metrics.get(
            "net_benefit_area_relative_to_treat_all"
        ),
        "net_benefit_area_form": test_metrics.get("net_benefit_area_form"),
        "tau_low": test_metrics.get("tau_low"),
        "tau_high": test_metrics.get("tau_high"),
        "n_grid_points": test_metrics.get("n_grid_points"),
        "primary_tau": test_metrics.get("primary_tau"),
        "net_benefit_at_primary_tau_relative_to_treat_all": test_metrics.get(
            "net_benefit_at_primary_tau_relative_to_treat_all"
        ),
        "nb_anchor_secondary_gate_active": test_metrics.get(
            "nb_anchor_secondary_gate_active"
        ),
        "nb_anchor_passes": test_metrics.get("nb_anchor_passes"),
        "decision_curve_data": test_metrics.get("decision_curve_data"),
    }


async def _run_one_algorithm(
    algo: Dict[str, Any],
    data: Dict[str, np.ndarray],
    success_criteria: Dict[str, Any],
) -> Dict[str, Any]:
    """Train + evaluate one algorithm; return artifact slice + status."""
    state = _build_state(
        algo["name"], algo["hp"], algo["candidate"], data, success_criteria
    )
    train_result = await train_model(state)
    if "error" in train_result:
        return {
            "algorithm": algo["name"],
            "status": "train_failed",
            "error": train_result["error"],
        }
    eval_state = {**state, **train_result}
    eval_result = await evaluate_model(eval_state)
    if "error" in eval_result:
        return {
            "algorithm": algo["name"],
            "status": "eval_failed",
            "error": eval_result["error"],
        }
    artifact = _extract_artifact_fields(eval_result)
    artifact["algorithm"] = algo["name"]
    artifact["status"] = "ok"
    return artifact


async def _run_diagnostic_async(
    output_path: Path,
    algorithms: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Async entry point: run all algorithms, write JSON, return summary."""
    data = _make_diagnostic_synthetic()
    success_criteria: Dict[str, Any] = {
        "dataset_disease": "breast_cancer_recurrence",
        "clinical_threshold_range": {"use_case": "diagnostic"},
    }
    results: List[Dict[str, Any]] = []
    for algo in algorithms:
        logger.info("running algorithm=%s", algo["name"])
        result = await _run_one_algorithm(algo, data, success_criteria)
        results.append(result)
        logger.info(
            "algorithm=%s status=%s auroc=%s brier=%s nb_area_rel=%s",
            algo["name"],
            result.get("status"),
            result.get("auroc"),
            result.get("brier_score"),
            result.get("net_benefit_area_relative_to_treat_all"),
        )

    artifact: Dict[str, Any] = {
        "schema_version": "phase1_diagnostic.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scenario": "placeholder_tier0_diagnostic",
        "scenario_note": (
            "Generated from tier0-style synthetic per shard 17 line 106 "
            "mitigation; replace with synthetic_data_generator_v2 Scenario A "
            "when delivered."
        ),
        "fixture": {
            "seed": SEED,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "n_features": N_FEATURES,
            "label_dgp": "y ~ Bernoulli(sigmoid(1.5 * X[:,0] - 0.5))",
            "expected_prevalence": 0.27,
        },
        "success_criteria": success_criteria,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, default=str)
    logger.info("wrote artifact to %s", output_path)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "docs/results/phase1_diagnostic_YYYYMMDD.json"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=["placeholder"],
        default="placeholder",
        help=(
            "Scenario selector. Phase 1 ships only `placeholder`; "
            "A/B/C will become valid choices once "
            "synthetic_data_generator_v2 delivers them (shard 17 W4 "
            "Prerequisites). Cycle-24 C-1: A/B/C removed from `choices` "
            "to prevent argparse from advertising them in --help when "
            "the runtime path rejects them anyway."
        ),
    )
    args = parser.parse_args()
    # Defensive: argparse already restricts ``choices``, but a future
    # caller may pass directly to ``main`` programmatically.
    if args.scenario != "placeholder":
        parser.error(
            "scenarios A/B/C not yet supported — synthetic_data_generator_v2 "
            "must deliver them first (shard 17 W4 Prerequisites)."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    output_path = args.output or (
        ROOT / "docs" / "results" / f"phase1_diagnostic_{datetime.now(UTC).strftime('%Y%m%d')}.json"
    )
    artifact = asyncio.run(_run_diagnostic_async(output_path, PLACEHOLDER_ALGORITHMS))
    n_ok = sum(1 for r in artifact["results"] if r.get("status") == "ok")
    n_total = len(artifact["results"])
    print(f"wrote {output_path}")
    print(f"  algorithms ok: {n_ok}/{n_total}")
    # Cycle-24 I-2: partial failure must exit non-zero. CI usage of this
    # script (W4 day-2..3 multi-scenario harness) would silently swallow
    # half-broken runs under the prior `> 0` guard.
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
