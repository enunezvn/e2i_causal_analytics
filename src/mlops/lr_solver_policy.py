"""LogisticRegression solver/penalty reconciliation policy.

Issue #232 pinned ``solver="saga"`` for LogisticRegression so HPO trials
sampling ``penalty="l1"`` would not crash with::

    Solver lbfgs supports only 'l2' or None penalties, got l1 penalty.

``saga`` is the only built-in sklearn solver that supports *both* l1 and
l2, so pinning it made every trial safe. The cost (observed on the Optum
mart full-population run, docs/results/tier0_optum_mart_initiation_events_disproof_20260606.md):
``saga`` does **not** converge in 1,000 epochs even on standardized data,
burning the full epoch budget on every fit (HPO ×N + the bootstrap),
whereas ``lbfgs`` converges in ~20 iters at an **identical** AUC — but
``lbfgs``/``newton-cg``/``sag`` only support l2/None.

This module centralises the rule "use the FASTEST valid solver for the
chosen penalty": ``lbfgs`` for l2/None, ``saga`` for l1/elasticnet. It is
applied at the three places a penalty becomes known:

1. the Optuna objective, post-merge, per trial (``optuna_optimizer``);
2. the final-train constructor guard (``model_trainer_node._filter_hyperparameters``);
3. the tier-0 alt-train candidate builder (``scripts/run_tier0_test.py``).

``_LR_FIXED_PARAMS`` (in ``hyperparameter_tuner``) keeps ``solver="saga"``
as the l1-SAFE floor, so even if reconciliation is somehow skipped the
#232 crash can never reappear — reconciliation only ever *downgrades* a
safe-but-slow ``saga`` to a fast ``lbfgs`` when the penalty permits.

This is a pure, dependency-free policy module so both ``src.mlops`` and
``src.agents`` can import it without an import cycle.
"""

from __future__ import annotations

from typing import Any, Dict

# Fastest built-in sklearn LR solver per penalty family.
_LR_L1_SOLVER = "saga"  # supports l1 + elasticnet (and l2), but slow to converge
_LR_L2_SOLVER = "lbfgs"  # fast; supports l2 / None only

# Penalties that REQUIRE an l1-capable solver. None / "l2" / "none" use lbfgs.
_L1_CAPABLE_PENALTIES = frozenset({"l1", "elasticnet"})

# Solvers this policy is allowed to switch between. A params dict whose
# solver is outside this set (e.g. "liblinear" in coefficient_sensitivity,
# or a non-LR estimator with no solver) is left UNTOUCHED.
_MANAGED_SOLVERS = frozenset({_LR_L1_SOLVER, _LR_L2_SOLVER})


def lr_solver_for_penalty(penalty: Any) -> str:
    """Return the fastest valid sklearn LR solver for ``penalty``.

    ``l1``/``elasticnet`` -> ``saga`` (the only l1-capable built-in);
    everything else (``l2``, ``None``, ``"none"``) -> ``lbfgs``.
    """
    if isinstance(penalty, str) and penalty.lower() in _L1_CAPABLE_PENALTIES:
        return _LR_L1_SOLVER
    return _LR_L2_SOLVER


def reconcile_lr_solver(params: Dict[str, Any]) -> Dict[str, Any]:
    """Align ``params['solver']`` with ``params['penalty']`` for LR-family params.

    Acts ONLY when ``params`` already carries a managed LR solver
    (``saga``/``lbfgs``) — the signature of an LR-family estimator routed
    through ``_LR_FIXED_PARAMS``. In that case the solver is set to the
    fastest valid choice for the (possibly absent) penalty:

    * ``penalty in {l1, elasticnet}`` -> ``saga``  (correctness: l1 needs saga)
    * otherwise                       -> ``lbfgs`` (speed: l2/None converge fast)

    Estimators with a non-managed solver (``liblinear``, ``newton-cg``,
    ``sag``) or no solver at all are returned unchanged. Mutates and
    returns ``params`` for call-site convenience.
    """
    if params.get("solver") in _MANAGED_SOLVERS:
        params["solver"] = lr_solver_for_penalty(params.get("penalty"))
    return params
