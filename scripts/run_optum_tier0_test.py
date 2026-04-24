#!/usr/bin/env python3
"""Tier-0 MLOps pipeline runner for Optum RWD cohorts.

Thin wrapper around ``scripts/run_tier0_test.py`` that:
  1. Selects one of the three Optum cohorts (initiation / discontinuation /
     persistence) produced by ``scripts/convert_optum_rwd.py``.
  2. Sets the appropriate target column, brand, and AUC threshold per cohort.
  3. Invokes the shared ``run_pipeline`` step functions with
     ``--data-dir data/rwd/optum/<cohort>``.

Usage:
    # Full pipeline on the initiation cohort
    python scripts/run_optum_tier0_test.py --cohort initiation

    # Specific step only
    python scripts/run_optum_tier0_test.py --cohort initiation --step 2

    # Dry-run
    python scripts/run_optum_tier0_test.py --cohort initiation --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and reuse the canonical tier-0 runner. We override its CONFIG before
# calling into run_pipeline.
import scripts.run_tier0_test as tier0  # noqa: E402

COHORT_TARGETS: dict[str, str] = {
    "initiation": "initiated_biologic_180d",
    "discontinuation": "discontinued_180d",
    "persistence": "persistent_at_180d",
}

COHORT_DIR: dict[str, str] = {
    "initiation": "data/rwd/optum/initiation",
    "discontinuation": "data/rwd/optum/discontinuation",
    "persistence": "data/rwd/optum/persistence",
}


@dataclass
class OptumTestConfig:
    """Overrides applied to tier0.CONFIG before running the pipeline.

    These mirror the structure of ``tier0.TestConfig`` but with values tuned
    for the leakage-safe Optum V2-style data: AUC threshold raised to 0.65
    (V2 data should be cleaner than CSU V1), minority-class recall/precision
    kept low because CSU biologic initiation is a rare event in claims RWD.
    """

    brand: str = "competitor"
    indication: str = "Chronic Spontaneous Urticaria (CSU)"
    problem_type: str = "binary_classification"
    hpo_trials: int = 10
    min_eligible_patients: int = 30
    min_auc_threshold: float = 0.65
    min_minority_recall: float = 0.10
    min_minority_precision: float = 0.05
    enable_mlflow: bool = True
    enable_opik: bool = False
    min_samples_per_split: int = 10


def apply_overrides(cohort: str, overrides: OptumTestConfig) -> None:
    """Mutate ``tier0.CONFIG`` with cohort-specific values."""
    tier0.CONFIG.brand = overrides.brand
    tier0.CONFIG.indication = overrides.indication
    tier0.CONFIG.problem_type = overrides.problem_type
    tier0.CONFIG.hpo_trials = overrides.hpo_trials
    tier0.CONFIG.min_eligible_patients = overrides.min_eligible_patients
    tier0.CONFIG.min_auc_threshold = overrides.min_auc_threshold
    tier0.CONFIG.min_minority_recall = overrides.min_minority_recall
    tier0.CONFIG.min_minority_precision = overrides.min_minority_precision
    tier0.CONFIG.enable_mlflow = overrides.enable_mlflow
    tier0.CONFIG.enable_opik = overrides.enable_opik
    tier0.CONFIG.min_samples_per_split = overrides.min_samples_per_split
    tier0.CONFIG.target_outcome = COHORT_TARGETS[cohort]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier-0 pipeline runner for Optum RWD cohorts."
    )
    parser.add_argument(
        "--cohort",
        required=True,
        choices=tuple(COHORT_TARGETS.keys()),
        help="Which Optum cohort subdir to load",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, 9),
        help="Run only a specific step (1-8). Default: full pipeline.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-bentoml", action="store_true")
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking (enabled by default)",
    )
    parser.add_argument("--enable-opik", action="store_true")
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=10,
        help="Number of HPO trials (default: 10)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (used to resolve data/rwd/optum/<cohort>)",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.65,
        help="Minimum validation AUC for success (default 0.65)",
    )
    parser.add_argument(
        "--min-samples-per-split",
        type=int,
        default=10,
        help=(
            "Minimum viable samples per split for split_enforcer gate "
            "(default: 10; set to 5 for discontinuation/persistence cohorts at n=47)"
        ),
    )

    args = parser.parse_args()

    cfg = OptumTestConfig(
        min_auc_threshold=args.min_auc,
        enable_mlflow=not args.disable_mlflow,
        enable_opik=args.enable_opik,
        hpo_trials=args.hpo_trials,
        min_samples_per_split=args.min_samples_per_split,
    )
    if args.enable_opik:
        os.environ["OPIK_ENABLED"] = "true"

    apply_overrides(args.cohort, cfg)

    data_dir = (args.data_root / COHORT_DIR[args.cohort]).resolve()
    if not data_dir.exists() and not args.dry_run:
        print(
            f"ERROR: Optum cohort directory not found: {data_dir}\n"
            f"Run: python scripts/convert_optum_rwd.py --cohort {args.cohort}",
            file=sys.stderr,
        )
        return 2

    print("\n=== Optum Tier-0 Pipeline Runner ===")
    print(f"  Cohort: {args.cohort}")
    print(f"  Target: {tier0.CONFIG.target_outcome}")
    print(f"  Data dir: {data_dir}")
    print(f"  AUC threshold: {tier0.CONFIG.min_auc_threshold}")
    print(f"  MLflow: {tier0.CONFIG.enable_mlflow}, Opik: {tier0.CONFIG.enable_opik}")

    asyncio.run(
        tier0.run_pipeline(
            step=args.step,
            dry_run=args.dry_run,
            imbalance_ratio=None,
            include_bentoml=not args.no_bentoml,
            data_dir=str(data_dir),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
