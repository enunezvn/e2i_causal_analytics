"""Lane A (real-data causal estimation): the DoWhy rebuild must not enumerate
adjustment-set candidates combinatorially.

Measured 2026-09-22 on the Optum biologic-persistence frame (n=15,209, 77
resolved covariates; ``docs/demos/results/2026-09-22_optum_biologic_persistence_cert/
preflight.md``): ``CausalModel.identify_effect`` took 467 s of the refutation
node's 478 s reconstruction, and every full-graph pre-flight hit the agent's
900 s hard cap at that call. Mechanism (dowhy 0.14 ``auto_identifier.
identify_backdoor``): the default search accepts the full common-cause set on
its FIRST candidate, then repeats the search as ``BACKDOOR_MIN`` from the
smallest subset upward; when the minimal valid set IS the full set (a
data-built graph: every common cause points at both treatment and outcome)
that pass burns its ``MAX_BACKDOOR_ITERATIONS`` (100,000) d-separation checks
before giving up -- k=12: 4,094 checks / 0.65 s; k>=17: the cap (16.8 s on a
19-node graph); k=77: 467 s on the 79-node graph. ``optimize_backdoor=True``
takes dowhy's path-based ``Backdoor`` search instead: the same adjustment set
(the full common-cause set), a byte-identical LinearDML estimate on k=8 and
k=12 synthetic frames, 0.01 s at every k measured.

Fast: n=300 rows, a linear-regression rebuild (no forest), one call.
"""

from __future__ import annotations

import time

import dowhy.causal_identifier.auto_identifier as auto_identifier
import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.refutation import _build_dowhy_estimate

K_COMMON_CAUSES = 20  # 2**20 subsets > the 100,000-iteration cap of the enumerating search
WALL_BUDGET_S = 10.0


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    n = 300
    X = rng.normal(size=(n, K_COMMON_CAUSES))
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))).astype(int)
    y = 0.4 * t + X[:, 0] + 0.2 * X[:, 1] + rng.normal(size=n)
    cov = [f"c{i}" for i in range(K_COMMON_CAUSES)]
    df = pd.DataFrame(X, columns=cov)
    df["t"] = t
    df["y"] = y
    return df, cov


def test_reconstruction_identifies_without_enumerating_candidate_sets(frame, monkeypatch):
    df, cov = frame
    calls = {"enumerator": 0}
    real = auto_identifier.find_valid_adjustment_sets

    def counting(*args, **kwargs):
        calls["enumerator"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(auto_identifier, "find_valid_adjustment_sets", counting)

    t0 = time.perf_counter()
    _model, identified_estimand, estimate, method = _build_dowhy_estimate(
        data=df,
        treatment="t",
        outcome="y",
        common_causes=cov,
        estimation_result={"selected_estimator": "ols", "ate": 0.4},
    )
    wall = time.perf_counter() - t0

    assert method == "backdoor.linear_regression"
    # Capability: the rebuild adjusts on the FULL common-cause set (the only
    # valid backdoor set of a data-built graph) ...
    assert set(identified_estimand.get_backdoor_variables()) == set(cov)
    assert np.isfinite(float(estimate.value))
    # ... without the combinatorial candidate enumeration that scales as 2**k
    # (k=20 here would be ~20 s of d-separation checks; k=77 live was 467 s).
    assert calls["enumerator"] == 0, calls
    assert wall < WALL_BUDGET_S, f"reconstruction took {wall:.1f}s for k={K_COMMON_CAUSES}"
