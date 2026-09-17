"""#2084: the DoWhy reconstruction must not depend on ``PYTHONHASHSEED``.

DoWhy 0.14 lists the backdoor adjustment set (``auto_identifier`` set
differences) and the estimator's effect modifiers (``CausalGraph.get_effect_modifiers``,
``list(set)``) in hash order. EconML's nuisance forests subsample features by
column INDEX, so a different order is a different fit: across container restarts
the reconstructed ATE and every seeded refit moved at the 1e-4..1e-3 level while
the seeds themselves were identical (lane-2029b cert).

The reconstruction therefore pins both lists to the column order the estimation
node fit the reported estimate on (``covariates_adjusted``, which follows the
graph builder's SORTED adjustment set) -- not to the order the node received
``common_causes`` in, which is usually ``state["confounders"]`` in the caller's
order (codex r1: a repeatable fit in the wrong order is still a different fit).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dowhy")
pytest.importorskip("econml")

from src.agents.causal_impact.nodes.refutation import _build_dowhy_estimate  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
COVARIATES = ["age_band", "severity", "academic_hcp", "prior_rx", "region_idx", "tenure"]


def _frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    X = pd.DataFrame({c: rng.normal(size=n) for c in COVARIATES})
    t = (X.sum(axis=1) * 0.3 + rng.normal(size=n) > 0).astype(int)
    y = 0.12 * t + X.to_numpy() @ np.linspace(0.05, 0.4, len(COVARIATES)) + rng.normal(size=n)
    return X.assign(treatment_arm=t, persistent_180d=y)


def _build(common_causes, covariates_adjusted=None):
    estimation_result = {"selected_estimator": "LinearDML", "random_state": 42}
    if covariates_adjusted is not None:
        estimation_result["covariates_adjusted"] = covariates_adjusted
    return _build_dowhy_estimate(
        data=_frame(),
        treatment="treatment_arm",
        outcome="persistent_180d",
        common_causes=common_causes,
        estimation_result=estimation_result,
    )


@pytest.mark.unit
@pytest.mark.parametrize("fit_order", [COVARIATES, COVARIATES[::-1]], ids=["forward", "reversed"])
def test_reconstruction_uses_the_estimations_column_order(fit_order):
    """Both lists DoWhy derives from sets come back in the order the estimation
    RECORDED, although ``common_causes`` arrive in a different (caller) order.
    Two opposite recorded orders cannot both equal one hash order, and neither
    equals the order passed in, so without the pin at least one case fails."""
    received = sorted(COVARIATES, key=lambda c: c[::-1])  # neither fit order
    assert received not in (COVARIATES, COVARIATES[::-1])

    _, identified_estimand, estimate, _ = _build(received, covariates_adjusted=list(fit_order))

    assert identified_estimand.get_backdoor_variables() == list(fit_order)
    assert list(estimate.estimator._effect_modifier_names) == list(fit_order)


@pytest.mark.unit
def test_reconstruction_without_a_recorded_order_uses_name_order():
    """No ``covariates_adjusted`` recorded: the order is still independent of the
    hash seed (by name), never DoWhy's set order."""
    _, identified_estimand, estimate, _ = _build(list(COVARIATES[::-1]))

    assert identified_estimand.get_backdoor_variables() == sorted(COVARIATES)
    assert list(estimate.estimator._effect_modifier_names) == sorted(COVARIATES)


_CHILD = """
import json, logging
logging.disable(logging.CRITICAL)
import src
from tests.unit.test_agents.test_causal_impact.test_refutation_adjustment_order_2084 import (
    COVARIATES, _build,
)
model, identified_estimand, estimate, _ = _build(list(COVARIATES), list(COVARIATES))
placebo = model.refute_estimate(
    identified_estimand, estimate, method_name="placebo_treatment_refuter",
    placebo_type="permute", num_simulations=3, random_state=11,
)
print(json.dumps({
    "src": src.__file__,
    "ate": repr(float(estimate.value)),
    "placebo": repr(float(placebo.new_effect)),
}))
"""


def _child_run(hashseed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=170,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["src"].startswith(str(REPO)), out["src"]
    return out


@pytest.mark.unit
@pytest.mark.slow
def test_reconstruction_and_refit_are_identical_across_hash_seeds():
    """The symptom itself: two fresh interpreters with different hash seeds
    reconstruct the same ATE and the same seeded placebo refit, to full
    precision. (Seeds 1 and 2 put these six covariates in different set orders.)"""
    first, second = _child_run("1"), _child_run("2")

    assert first["ate"] == second["ate"]
    assert first["placebo"] == second["placebo"]
