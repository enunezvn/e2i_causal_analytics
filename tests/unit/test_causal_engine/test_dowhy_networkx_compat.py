"""Dependency-contract test for #869: installed dowhy must work against installed networkx.

dowhy < 0.13 calls ``nx.algorithms.d_separated`` (dowhy/causal_graph.py,
dowhy/graph.py), which networkx renamed to ``is_d_separator`` in 3.3 and
REMOVED in 3.5. Under any networkx >= 3.5, every
``CausalModel.identify_effect`` / refuter call raises
``AttributeError: module 'networkx.algorithms' has no attribute 'd_separated'``
and the causal_impact refutation node fail-closes before a single refuter runs
(refutation_tests_total=0, no refutation backing for causal verdicts).

dowhy >= 0.13 ships the compat import ("version compatibility for breaking
change in networkx 3.5"): ``is_d_separator as d_separated`` with a fallback,
so it works against both old and new networkx.

This test exercises the REAL dowhy path end-to-end (no mocks): a tiny exactly
identified backdoor model -> ``identify_effect`` (the call that dies first on
an incompatible pairing) -> linear-regression estimate -> placebo refuter (the
prod refutation suite's cheapest member). It fails loudly in any environment
whose resolver paired an nx-incompatible dowhy with a modern networkx — the
exact dowhy==0.12 + networkx==3.6.1 combination uv resolved for python < 3.13
from the pre-#869 ``dowhy>=0.11.0`` floor in pyproject.toml.

Companion spec-level guard: tests/test_requirements_lock.py::
test_pyproject_dowhy_floor_is_networkx35_compatible (asserts the pyproject
floor itself excludes dowhy < 0.13, so resolvers can never recreate the
broken pairing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.xdist_group(name="dowhy_compat")


@pytest.mark.timeout(180)
def test_dowhy_d_separation_contract_against_installed_networkx() -> None:
    """identify -> estimate -> refute must run under the installed dowhy+networkx pair."""
    from dowhy import CausalModel

    rng = np.random.default_rng(869)
    n = 400
    confounder = rng.normal(size=n)
    treatment = (confounder + rng.normal(size=n) > 0).astype(int)
    outcome = 0.5 * treatment + 0.8 * confounder + rng.normal(size=n)
    frame = pd.DataFrame({"t": treatment, "y": outcome, "w": confounder})

    model = CausalModel(data=frame, treatment="t", outcome="y", common_causes=["w"])

    # Dies here with AttributeError('d_separated') on dowhy<0.13 + networkx>=3.5.
    estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    assert estimate.value == pytest.approx(0.5, abs=0.2), (
        f"backdoor ATE {estimate.value} drifted from the designed 0.5 effect"
    )

    # The placebo refuter walks the same d-separation machinery as the prod
    # refutation suite; with a placebo treatment the effect must vanish.
    refutation = model.refute_estimate(
        estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        num_simulations=2,
    )
    assert refutation.new_effect == pytest.approx(0.0, abs=0.2), (
        f"placebo refuter effect {refutation.new_effect} should be ~0"
    )
