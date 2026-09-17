"""Pin the order of the variable lists DoWhy derives from Python sets (#2084).

DoWhy 0.14 lists the backdoor adjustment set (``auto_identifier``: set
differences over the graph's nodes) and, when ``estimate_effect`` is not handed
effect modifiers, the estimator's effect modifiers (``CausalGraph.get_effect_modifiers``:
``list(set)``) in HASH order. String hashes are salted per interpreter unless
``PYTHONHASHSEED`` is set, so the order changed on every container restart.
EconML's nuisance forests subsample features by column INDEX, so a new order is
a new fit: the reconstructed ATE and every seeded refit moved at the
1e-4..1e-3 level while the refit seeds themselves were identical.

The fix lives in code rather than in a deploy-time ``PYTHONHASHSEED``: it holds
in every process that reconstructs (api, workers, CI, a local run) without an
environment contract to keep in sync.
"""

from __future__ import annotations

from typing import Any, Dict, List


def pin_adjustment_order(identified_estimand: Any, order: List[str]) -> None:
    """Reorder every adjustment list on ``identified_estimand`` to ``order``, in place.

    Pass the adjustment set in the order the caller fit the reported estimate on,
    so the reconstruction's design matrix has the same column order as that fit.
    Membership is never changed: a name the caller did not list (not expected
    for a backdoor set built from ``common_causes``) sorts after the known ones,
    by name, so even then the order is independent of the hash seed.
    Pass the same ``order`` as ``effect_modifiers`` to ``estimate_effect`` -- that
    is the other set-derived list.
    """
    rank = {name: i for i, name in enumerate(order)}
    for sets in (
        identified_estimand.backdoor_variables,
        identified_estimand.general_adjustment_variables,
    ):
        for key, names in (sets or {}).items():
            if names:
                sets[key] = sorted(names, key=lambda name: (rank.get(name, len(rank)), name))


def fit_column_order(common_causes: List[str], estimation_result: Dict[str, Any]) -> List[str]:
    """``common_causes`` in the column order the estimation node fit the reported ATE on.

    The reconstruction's ``common_causes`` usually come from ``state["confounders"]``
    (the caller's order), while the estimator was fit on the graph's adjustment set,
    which the graph builder SORTS, recorded as ``covariates_adjusted`` (plus
    ``baseline_covariates_adjusted`` on an efficiency run). Pinning the caller's
    order would make the reconstruction repeatable yet still a different
    feature-index fit from the one on screen. So names the estimation recorded
    keep ITS order; any other name follows, by name. Membership is unchanged --
    this chooses an order, never a set.
    """
    reference = list(estimation_result.get("covariates_adjusted") or []) + list(
        estimation_result.get("baseline_covariates_adjusted") or []
    )
    rank: Dict[str, int] = {}
    for name in reference:
        rank.setdefault(name, len(rank))
    return sorted(common_causes, key=lambda name: (rank.get(name, len(rank)), name))
