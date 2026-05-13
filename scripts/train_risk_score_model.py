#!/usr/bin/env python3
"""CLI driver for the risk_score model (issue #171 PR C Sub-PR-B).

Trains an XGBoost / LightGBM classifier on the CSU initiation cohort to
predict ``initiated_biologic_180d``, calibrates on validation, logs to MLflow,
and writes predictions back to ``patient_journeys.risk_score`` +
``ml_predictions``.

Usage::

    # Synthetic smoke (no real Optum data required) — validates plumbing only
    python scripts/train_risk_score_model.py --synthetic-smoke

    # Real Optum cohort (requires data/rwd/optum/initiation/ to exist)
    python scripts/train_risk_score_model.py \\
        --data-dir data/rwd/optum/initiation \\
        --target initiated_biologic_180d

Inputs:
    --data-dir         Optum cohort directory (initiation cohort by default).
    --target           Target column name (default: initiated_biologic_180d).
    --hpo-trials       Optuna trials (default 50; lower for quick runs).
    --min-auc-pr       AUC-PR floor (default 0.65). Bar NEVER lowered silently.
    --disable-mlflow   Skip MLflow logging (still trains).
    --synthetic-smoke  Use a synthetic separable dataset (CI smoke; ignores
                       --data-dir; useful when real Optum data is unavailable).
    --json-out         Optional path to write training-result JSON.

The script intentionally does NOT write to the DB by default — DB writes are
gated by ``--write-predictions`` and require ``SUPABASE_URL`` /
``SUPABASE_SERVICE_KEY`` env vars. The Celery task ``train_risk_score_task``
(see ``src/tasks/risk_score_tasks.py`` if/when wired) is the production
write path.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.prediction_synthesizer.risk_score import (  # noqa: E402
    DEFAULT_MIN_AUC_PR,
    RiskScoreTrainer,
    RiskScoreTrainingResult,
)

logger = logging.getLogger("train_risk_score_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load_synthetic_smoke_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Return ``(X_train, y_train, X_val, y_val)`` on a separable synthetic dataset.

    This is the deferred-real-data smoke path documented in issue #171:
    "If real Optum data isn't reachable, ship the model-training plumbing
    with a synthetic-data smoke test only."
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=800,
        n_features=12,
        n_informative=6,
        n_redundant=3,
        weights=[0.7, 0.3],
        flip_y=0.02,
        class_sep=1.4,
        random_state=42,
    )
    feat_names = [f"feature_{i}" for i in range(12)]
    X_df = pd.DataFrame(X, columns=feat_names)
    X_tr, X_va, y_tr, y_va = train_test_split(X_df, y, test_size=0.25, stratify=y, random_state=42)
    return X_tr.reset_index(drop=True), y_tr, X_va.reset_index(drop=True), y_va


def _load_real_optum_data(
    data_dir: Path, target: str
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load the Optum initiation cohort from ``data_dir`` and split train / val.

    Expected layout::

        <data_dir>/
            patient_journeys.parquet (or .json / .csv)
            <feature_files>

    We look for a single tabular file with the target column + numeric feature
    columns. The exact converter output schema is documented in
    ``scripts/convert_optum_rwd.py`` §3 (``OptumDataConverter.convert_all``).
    """
    candidates = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No .parquet or .csv files in {data_dir}. "
            "Run scripts/convert_optum_rwd.py --cohort initiation first."
        )
    df_list = []
    for path in candidates:
        if path.suffix == ".parquet":
            df_list.append(pd.read_parquet(path))
        else:
            df_list.append(pd.read_csv(path))
    df = pd.concat(df_list, axis=0, ignore_index=True) if len(df_list) > 1 else df_list[0]
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not present in cohort frame.")

    # Use the existing data_split column if present, else split here.
    if "data_split" in df.columns:
        train = df[df["data_split"] == "train"]
        val = df[df["data_split"] == "validation"]
        if val.empty:
            val = df[df["data_split"] == "val"]
    else:
        from sklearn.model_selection import train_test_split

        train, val = train_test_split(
            df,
            test_size=0.20,
            stratify=df[target] if df[target].nunique() == 2 else None,
            random_state=42,
        )

    # Drop non-feature columns (metadata, IDs, target) before training.
    drop_cols = {
        target,
        "data_split",
        "patid",
        "patient_id",
        "split_config_id",
        "created_at",
        "updated_at",
        "index_date",
        "journey_id",
    }
    feat_cols = [
        c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    X_train = train[feat_cols].fillna(0).reset_index(drop=True)
    y_train = train[target].astype(int).to_numpy()
    X_val = val[feat_cols].fillna(0).reset_index(drop=True)
    y_val = val[target].astype(int).to_numpy()
    return X_train, y_train, X_val, y_val


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "rwd" / "optum" / "initiation",
        help="Optum cohort directory (default: data/rwd/optum/initiation/)",
    )
    parser.add_argument(
        "--target",
        default="initiated_biologic_180d",
        help="Target column name (default: initiated_biologic_180d)",
    )
    parser.add_argument(
        "--hpo-trials", type=int, default=50, help="Number of Optuna trials (default: 50)"
    )
    parser.add_argument(
        "--min-auc-pr",
        type=float,
        default=DEFAULT_MIN_AUC_PR,
        help=f"AUC-PR floor (default: {DEFAULT_MIN_AUC_PR})",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow logging (still trains).",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Train on a synthetic separable dataset (validates plumbing only).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the training-result JSON dump.",
    )
    parser.add_argument(
        "--model-candidates",
        nargs="+",
        choices=["xgboost", "lightgbm"],
        default=["xgboost", "lightgbm"],
        help="Which model classes to evaluate (default: both).",
    )
    args = parser.parse_args(argv)

    if args.synthetic_smoke:
        logger.info("Loading synthetic smoke dataset (plumbing-only validation).")
        X_train, y_train, X_val, y_val = _load_synthetic_smoke_data()
    else:
        if not args.data_dir.exists():
            logger.error(
                "Cohort directory %s does not exist. Either:\n"
                "  1. Run scripts/convert_optum_rwd.py --cohort initiation, OR\n"
                "  2. Use --synthetic-smoke for plumbing validation.",
                args.data_dir,
            )
            return 2
        logger.info("Loading real Optum cohort from %s.", args.data_dir)
        X_train, y_train, X_val, y_val = _load_real_optum_data(args.data_dir, args.target)

    logger.info(
        "Train: n=%d (pos=%d), Val: n=%d (pos=%d), features=%d",
        len(y_train),
        int(y_train.sum()),
        len(y_val),
        int(y_val.sum()),
        X_train.shape[1],
    )

    trainer = RiskScoreTrainer(
        min_auc_pr=args.min_auc_pr,
        hpo_trials=args.hpo_trials,
        enable_mlflow=not args.disable_mlflow,
        model_candidates=tuple(args.model_candidates),
    )
    result: RiskScoreTrainingResult = trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        mlflow_experiment="risk_score_csu_initiation",
        mlflow_run_name=("synthetic_smoke" if args.synthetic_smoke else "real_optum"),
    )

    payload: dict[str, Any] = result.to_dict()
    payload["synthetic_smoke"] = bool(args.synthetic_smoke)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        logger.info("Wrote training-result JSON to %s.", args.json_out)

    if result.honest_failures:
        logger.warning("Honest failures surfaced:")
        for failure in result.honest_failures:
            logger.warning("  - %s", failure)
        # Don't fail the run — the bar is reported, not enforced (per
        # supervisor decision: "If real Optum data fails the bar: log it
        # and SURFACE, do NOT lower the bar silently").
    return 0


if __name__ == "__main__":
    sys.exit(main())
