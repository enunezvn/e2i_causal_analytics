"""Refuse to refute a model the reported estimate did not fit (#2155).

The refutation node reconstructs on ``state["confounders"]`` (the estimate's
``covariates_adjusted`` only when that is empty), while the estimation node fits
on the graph builder's adjustment set and records exactly that list as
``covariates_adjusted``. The graph builder can drop a declared confounder (one
the DAG shows as a descendant of the treatment, or one an adjustment policy
excludes), and then the refuters would critique a different model from the one
whose ATE is reported.

Measured 2026-09-16 before this guard existed: the two sets never differed on a
live run (237/237 ``discovered_dags`` rows; 55/55 manual-DAG replays on real
cohort frames). So this is a fail-closed invariant for a latent gap, not a fix
for an observed one.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

from src.causal_engine.errors import RefutationError


def require_estimate_columns(
    common_causes: List[str],
    estimation_result: Mapping[str, Any],
    effective_columns: Callable[[List[str], Dict[str, Any]], List[str]],
) -> None:
    """Raise ``RefutationError`` unless ``common_causes`` is the estimate's column set.

    ``common_causes`` must already be what the reconstruction will condition on
    (after the #1188 efficiency rule). The reported estimator's columns are
    ``covariates_adjusted`` passed through the SAME rule (``effective_columns``,
    the node's ``_effective_reconstruction_common_causes``), so an efficiency run
    compares its baselines on both sides. The comparison is on SETS; the order is
    pinned separately (#2084).

    ``covariates_adjusted`` ABSENT is refused too, never read as "empty": an empty
    list is a validated empty backdoor (a randomized question), while an absent
    one means the columns the estimate fitted on are unknown. The estimation node
    has written the key on every result since the energy-score integration
    (2025-12-26), and no other producer of ``estimation_result`` exists.
    """
    recorded = estimation_result.get("covariates_adjusted")
    if recorded is None:
        raise RefutationError(
            "Refutation refused: the estimate did not record the columns it adjusted "
            "for (covariates_adjusted), so the refuters cannot be shown to critique "
            "the reported model.",
            details={"reason": "estimate_columns_unrecorded", "common_causes": list(common_causes)},
        )
    expected = effective_columns(list(recorded), dict(estimation_result))
    missing_from_estimate = sorted(set(common_causes) - set(expected))
    missing_from_reconstruction = sorted(set(expected) - set(common_causes))
    if missing_from_estimate or missing_from_reconstruction:
        raise RefutationError(
            "Refutation refused: the refuters would critique a different model from the "
            f"reported estimate. The estimate adjusted for {sorted(expected)}; the "
            f"reconstruction would adjust for {sorted(common_causes)} (not in the "
            f"estimate: {missing_from_estimate}; missing from the reconstruction: "
            f"{missing_from_reconstruction}).",
            details={
                "reason": "reconstruction_columns_differ_from_estimate",
                "estimate_columns": sorted(expected),
                "reconstruction_columns": sorted(common_causes),
                "missing_from_estimate": missing_from_estimate,
                "missing_from_reconstruction": missing_from_reconstruction,
            },
        )
