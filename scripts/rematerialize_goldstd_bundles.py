"""Re-materialize gold-standard cohort SERVING BUNDLES for real SHAP (#39).

The ``*_goldstd_lr_v1`` rows in ``ml_model_registry`` (initiation / persistence /
discontinuation × {remibrutinib, fabhalta, kisqali}, plus the base aggregate
specs ``csu_initiation`` / ``pnh_persistence`` / ``pnh_discontinuation``) were
registered at ``stage='staging'``, ``feature_count=9``. Their ``.pkl`` artifacts
were written inside now-reaped worktrees and are GONE — and even when present,
those were the BARE estimator (no FeatureBuilder), so they cannot serve a RAW
3-covariate request.

This script re-fits each model from the live ``patient_journeys`` synthetic rows
and writes a SERVING BUNDLE — ``{"model", "preprocessor": fitted FeatureBuilder,
"feature_columns": <9 encoded names>}`` — to a NEW durable path under
``data/ml_artifacts/shap_serving/<cohort>/<model_name>.bundle.pkl``.

IMPORTANT SCOPE GUARD: this script DOES NOT mutate ``ml_model_registry``. Row
registration is the gold-standard session's domain; here we only READ
``patient_journeys`` and WRITE bundle files. Pointing the registry rows /
BentoML store at the new bundles is a LIVE ACTIVATION step run by an operator
after merge (see the PR's LIVE ACTIVATION TODO).

Usage (live, after merge)::

    python -m scripts.rematerialize_goldstd_bundles                 # all 12
    python -m scripts.rematerialize_goldstd_bundles --model \
        csu_initiation_goldstd_lr_v1                                # one

The artifacts are environment-specific and gitignored; this script regenerates
them deterministically from the synthetic cohort in the live DB.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score

from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    DISCONTINUATION,
    INITIATION,
    PATIENT_COHORTS,
    PERSISTENCE,
    CohortSpec,
    goldstd_model_name,
    make_patient_spec,
)
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
from src.mlops.prediction_synthesizer_deploy import serialize_model_bundle

logger = logging.getLogger(__name__)

# New durable artifact root — distinct from the reaped per-worktree paths the
# registry rows currently reference. Resolved relative to the repo root so it is
# stable regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
SHAP_SERVING_ROOT = _REPO_ROOT / "data" / "ml_artifacts" / "shap_serving"


def _build_spec_registry() -> dict[str, tuple[CohortSpec, str]]:
    """Map every live ``*_goldstd_lr_v1`` model_name → (CohortSpec, cohort_dir).

    The cohort_dir is the sub-directory under ``shap_serving/`` (matches the
    cohort family so the 9 per-brand bundles and the base aggregates group
    sensibly on disk).
    """
    registry: dict[str, tuple[CohortSpec, str]] = {}

    # Base aggregate specs (the original 3 cohorts).
    registry["csu_initiation_goldstd_lr_v1"] = (INITIATION, "initiation")
    registry["pnh_persistence_goldstd_lr_v1"] = (PERSISTENCE, "persistence")
    registry["pnh_discontinuation_goldstd_lr_v1"] = (DISCONTINUATION, "discontinuation")

    # 9 per-brand cohort specs.
    for cohort in PATIENT_COHORTS:
        for brand in BRANDS:
            name = goldstd_model_name(cohort, brand)
            spec = make_patient_spec(cohort, brand)
            registry[name] = (spec, cohort)

    return registry


SPEC_REGISTRY: dict[str, tuple[CohortSpec, str]] = _build_spec_registry()


async def rematerialize_bundle(
    db: Any,
    *,
    model_name: str,
    spec: CohortSpec,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Re-fit ``spec`` from live ``patient_journeys`` and write a serving bundle.

    Loads the cohort's rows via :meth:`FeatureBuilder.load_frame` (READ-only),
    FIT-encodes (learning the 9 encoded ``feature_columns`` + medians), trains a
    calibrated LR via :func:`train_cohort_model`, computes a real in-sample AUC,
    and serializes the ``{"model", "preprocessor", "feature_columns"}`` bundle.

    Returns a summary dict ``{"bundle_path", "auc", "feature_count",
    "training_samples", "feature_columns"}``. Does NOT mutate the registry.
    """
    out_root = SHAP_SERVING_ROOT if out_root is None else Path(out_root)
    _, cohort_dir = SPEC_REGISTRY.get(model_name, (spec, spec.name))

    fb = FeatureBuilder(spec)
    frame = await fb.load_frame(db)
    if frame.empty:
        raise ValueError(
            f"rematerialize_bundle: no patient_journeys rows for model={model_name!r} "
            f"brand={spec.brand!r} label={spec.label_column!r} (is_synthetic=True)."
        )

    X, y = fb.build_from_frame(frame)
    model = train_cohort_model(spec, X, y)

    # Real in-sample AUC over the calibrated probabilities (honest metric; the
    # held-out AUC was measured at registration time — see feature_builder.py).
    proba = model.predict_proba(X)[:, 1]
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        # Single-class frame (degenerate split) — AUC undefined; report NaN-safe 0.5.
        auc = 0.5

    artifact_dir = out_root / cohort_dir
    bundle_path = serialize_model_bundle(
        model=model,
        preprocessor=fb,
        feature_columns=fb.feature_columns,
        artifact_dir=artifact_dir,
        model_name=model_name,
    )

    return {
        "bundle_path": bundle_path,
        "auc": auc,
        "feature_count": len(fb.feature_columns),
        "training_samples": int(len(y)),
        "feature_columns": list(fb.feature_columns),
    }


async def _amain(model_names: list[str]) -> int:
    from src.memory.services.factories import get_async_supabase_client

    db = await get_async_supabase_client()
    if db is None:
        logger.error("No async Supabase client available; cannot load patient_journeys.")
        return 1

    rc = 0
    for name in model_names:
        spec, _ = SPEC_REGISTRY[name]
        try:
            summary = await rematerialize_bundle(db, model_name=name, spec=spec)
            logger.info(
                "Re-materialized %s -> %s (AUC=%.4f, n=%d, %d features)",
                name,
                summary["bundle_path"],
                summary["auc"],
                summary["training_samples"],
                summary["feature_count"],
            )
            print(
                f"OK {name}: {summary['bundle_path']} "
                f"AUC={summary['auc']:.4f} n={summary['training_samples']}"
            )
        except Exception as e:  # noqa: BLE001 - CLI surfaces per-model failures
            rc = 1
            logger.error("FAILED %s: %s", name, e)
            print(f"FAIL {name}: {e}")
    return rc


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Re-materialize gold-standard serving bundles (#39)")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=sorted(SPEC_REGISTRY.keys()),
        help="Model name(s) to re-materialize (repeatable). Default: all 12.",
    )
    args = parser.parse_args()
    model_names = args.models or sorted(SPEC_REGISTRY.keys())
    return asyncio.run(_amain(model_names))


if __name__ == "__main__":
    raise SystemExit(main())
