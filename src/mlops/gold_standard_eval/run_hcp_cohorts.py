"""run_hcp_cohorts — runs all 3 per-brand HCP adoption slots in one pass.

Reuses the per-slot pipeline from ``run_persistence_eval`` (``_resolve_client``
and ``_run_one_cohort``) to execute every combination of
``hcp_adoption × (Remibrutinib, Fabhalta, Kisqali)``.

Run as a CLI on the target box::

    E2I_DB_INTEGRATION=1 python -m src.mlops.gold_standard_eval.run_hcp_cohorts
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    HCP_ADOPTION_COHORT,
    goldstd_experiment_name,
    goldstd_model_name,
    make_hcp_spec,
)
from src.mlops.gold_standard_eval.run_persistence_eval import (
    _resolve_client,
    _run_one_cohort,
)

logger = logging.getLogger(__name__)


async def run(db: Any = None) -> dict[str, Any]:
    """Run all 3 per-brand HCP adoption slots (hcp_adoption × 3 brands).

    Parameters
    ----------
    db:
        Optional async Supabase client.  When None the faithful docker client is
        resolved (fail-closed).  Tests pass the same client they assert against.

    Returns
    -------
    dict keyed by brand lowercase (e.g. ``"remibrutinib"``), each value a dict
    with keys: ``model``, ``holdout_auc``, ``backtest_points``, ``n_train``,
    ``n_holdout``.
    """
    client = await _resolve_client(db)
    results: dict[str, Any] = {}

    for brand in BRANDS:
        spec = make_hcp_spec(brand)
        model_name = goldstd_model_name(HCP_ADOPTION_COHORT, brand)
        experiment_name = goldstd_experiment_name(HCP_ADOPTION_COHORT, brand)
        logger.info(
            "=== Starting slot: cohort=%s brand=%s target=%s ===",
            HCP_ADOPTION_COHORT,
            brand,
            spec.target,
        )
        res = await _run_one_cohort(
            client,
            spec,
            model_name=model_name,
            experiment_name=experiment_name,
        )
        brand_lower = brand.lower()
        results[brand_lower] = res
        logger.info(
            "=== Done slot: cohort=%s brand=%s holdout_auc=%.4f backtest_points=%d ===",
            HCP_ADOPTION_COHORT,
            brand,
            res["holdout_auc"],
            res["backtest_points"],
        )

    return results


def _print_report(report: dict[str, Any]) -> None:
    logger.info("=== gold-standard per-brand HCP adoption eval report ===")
    logger.info("  %-30s %-45s %-12s %-15s", "brand", "model", "holdout_auc", "backtest_points")
    for brand in BRANDS:
        brand_lower = brand.lower()
        sub = report.get(brand_lower, {})
        logger.info(
            "  %-30s %-45s %-12s %-15s",
            brand_lower,
            sub.get("model", "—"),
            f"{sub['holdout_auc']:.4f}" if "holdout_auc" in sub else "—",
            sub.get("backtest_points", "—"),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-DB gold-standard eval for all 3 per-brand HCP adoption slots "
            "(hcp_adoption × Remibrutinib/Fabhalta/Kisqali)."
        )
    )
    parser.parse_args()
    report = asyncio.run(run())
    _print_report(report)


if __name__ == "__main__":
    main()
