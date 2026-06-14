"""run_patient_cohorts — runs all 9 per-brand patient slots in one pass.

Reuses the per-slot pipeline from ``run_persistence_eval`` (``_resolve_client``
and ``_run_one_cohort``) to execute every combination of
``(initiation, persistence, discontinuation) × (Remibrutinib, Fabhalta, Kisqali)``.

After all 9 slots complete, a per-brand complement validation is emitted: for
each brand the ``persistence`` and ``discontinuation`` holdout AUCs should be
close (``persistent_180d == 1 − discontinued_180d`` in the synthetic DGP), so a
warning is logged if they diverge by more than 0.05.

Run as a CLI on the target box::

    E2I_DB_INTEGRATION=1 python -m src.mlops.gold_standard_eval.run_patient_cohorts
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    PATIENT_COHORTS,
    goldstd_experiment_name,
    goldstd_model_name,
    make_patient_spec,
)
from src.mlops.gold_standard_eval.run_persistence_eval import (
    _resolve_client,
    _run_one_cohort,
)

logger = logging.getLogger(__name__)


async def run(db: Any = None) -> dict[str, Any]:
    """Run all 9 per-brand patient slots (initiation/persistence/discontinuation × 3 brands).

    Parameters
    ----------
    db:
        Optional async Supabase client.  When None the faithful docker client is
        resolved (fail-closed).  Tests pass the same client they assert against.

    Returns
    -------
    dict keyed by ``"{cohort}_{brand}"`` (e.g. ``"initiation_Remibrutinib"``),
    each value a dict with keys: ``model``, ``holdout_auc``, ``backtest_points``,
    ``n_train``, ``n_holdout``.
    """
    client = await _resolve_client(db)
    results: dict[str, Any] = {}

    for cohort in PATIENT_COHORTS:
        for brand in BRANDS:
            spec = make_patient_spec(cohort, brand)
            logger.info(
                "=== Starting slot: cohort=%s brand=%s target=%s ===",
                cohort,
                brand,
                spec.target,
            )
            res = await _run_one_cohort(
                client,
                spec,
                model_name=goldstd_model_name(cohort, brand),
                experiment_name=goldstd_experiment_name(cohort, brand),
            )
            results[f"{cohort}_{brand}"] = res
            logger.info(
                "=== Done slot: cohort=%s brand=%s holdout_auc=%.4f backtest_points=%d ===",
                cohort,
                brand,
                res["holdout_auc"],
                res["backtest_points"],
            )

    # Per-brand complement validation: persistence AUC ≈ discontinuation AUC
    # because persistent_180d == 1 − discontinued_180d in the synthetic DGP.
    for brand in BRANDS:
        pers_key = f"persistence_{brand}"
        disc_key = f"discontinuation_{brand}"
        if pers_key in results and disc_key in results:
            pers_auc = results[pers_key]["holdout_auc"]
            disc_auc = results[disc_key]["holdout_auc"]
            logger.info(
                "[%s] Complement validation: persistence holdout_auc=%.4f  "
                "discontinuation holdout_auc=%.4f  delta=%.4f",
                brand,
                pers_auc,
                disc_auc,
                abs(pers_auc - disc_auc),
            )
            if abs(pers_auc - disc_auc) > 0.05:
                logger.warning(
                    "[%s] Complement AUC divergence > 0.05 (persistence=%.4f, "
                    "discontinuation=%.4f): mirror models should match; investigate "
                    "data imbalance or feature encoding drift.",
                    brand,
                    pers_auc,
                    disc_auc,
                )

    return results


def _print_report(report: dict[str, Any]) -> None:
    logger.info("=== gold-standard per-brand patient cohort eval report ===")
    logger.info("  %-40s %-45s %-12s %-15s", "slot", "model", "holdout_auc", "backtest_points")
    for cohort in PATIENT_COHORTS:
        for brand in BRANDS:
            key = f"{cohort}_{brand}"
            sub = report.get(key, {})
            logger.info(
                "  %-40s %-45s %-12s %-15s",
                key,
                sub.get("model", "—"),
                f"{sub['holdout_auc']:.4f}" if "holdout_auc" in sub else "—",
                sub.get("backtest_points", "—"),
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-DB gold-standard eval for all 9 per-brand patient cohort slots "
            "(initiation/persistence/discontinuation × Remibrutinib/Fabhalta/Kisqali)."
        )
    )
    parser.parse_args()
    report = asyncio.run(run())
    _print_report(report)


if __name__ == "__main__":
    main()
