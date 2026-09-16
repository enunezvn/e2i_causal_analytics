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

from typing import Any, List


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
