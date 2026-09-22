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

The optimisation is only correct when there IS a set to find: with an EMPTY
adjustment set (a validated RCT / the negative-control rebuild) the path-based
search returns no set, DoWhy's ``estimands["backdoor"]`` is None and the estimate
carries ``value=None`` (codex r2 HIGH) -- the default search returns the explicit
empty set instantly, so that shape keeps the default. The equivalence tests below
pin both identifiers to the SAME adjustment set and the SAME estimate value on
every shape the node sees: empty, numeric, categorical-expanded, continuous
(median-binarised) treatment.

Fast: n=300 rows, linear-regression rebuilds (no forest).
"""

from __future__ import annotations

import time

import dowhy.causal_identifier.auto_identifier as auto_identifier
import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.refutation import _build_dowhy_estimate

K_COMMON_CAUSES = 20  # 2**20 subsets > the 100,000-iteration cap of the enumerating search
WALL_BUDGET_S = 20.0  # the fixed path takes ~1 s here; the enumerating path ~20 s+ at k=20


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


# --- Equivalence with the default identifier on every shape the node sees ---------


def _default_identifier_estimate(df, common_causes, monkeypatch):
    """The SAME rebuild forced onto dowhy's default (enumerating) identifier."""
    import dowhy

    real_identify = dowhy.CausalModel.identify_effect

    def default_only(self, *args, **kwargs):
        kwargs["optimize_backdoor"] = False
        return real_identify(self, *args, **kwargs)

    monkeypatch.setattr(dowhy.CausalModel, "identify_effect", default_only)
    try:
        return _build_dowhy_estimate(
            data=df,
            treatment="t",
            outcome="y",
            common_causes=common_causes,
            estimation_result={"selected_estimator": "ols", "ate": 0.4},
        )
    finally:
        monkeypatch.setattr(dowhy.CausalModel, "identify_effect", real_identify)


def _shape_frames():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=(n, 3))
    t_bin = (rng.random(n) < 1.0 / (1.0 + np.exp(-x[:, 0]))).astype(int)
    y = 0.4 * t_bin + x[:, 0] + rng.normal(size=n)
    base = pd.DataFrame({"t": t_bin, "y": y, "c0": x[:, 0], "c1": x[:, 1], "c2": x[:, 2]})
    categorical = base.copy()
    categorical["region"] = rng.choice(["north", "south", "west"], size=n)
    continuous = base.copy()
    continuous["t"] = t_bin + rng.uniform(-0.4, 0.4, size=n)  # non-integer -> median-binarised
    return {
        "empty": (base, []),
        "numeric": (base, ["c0", "c1", "c2"]),
        "categorical": (categorical, ["c0", "region"]),
        "continuous_treatment": (continuous, ["c0", "c1"]),
    }


@pytest.mark.parametrize("shape", sorted(_shape_frames()))
def test_identifier_choice_matches_the_default_identifier_on_every_shape(shape, monkeypatch):
    df, cov = _shape_frames()[shape]
    _m, est_new, e_new, _ = _build_dowhy_estimate(
        data=df,
        treatment="t",
        outcome="y",
        common_causes=cov,
        estimation_result={"selected_estimator": "ols", "ate": 0.4},
    )
    _m, est_old, e_old, _ = _default_identifier_estimate(df, cov, monkeypatch)

    # The rebuilt estimate must exist (the empty-set regression left value=None) ...
    assert e_new.value is not None and np.isfinite(float(e_new.value)), shape
    # ... adjust on the same set as the default identifier ...
    assert list(est_new.get_backdoor_variables()) == list(est_old.get_backdoor_variables()), shape
    # ... and be the same number.
    assert float(e_new.value) == float(e_old.value), shape


# --- The rebuild's preprocessing the equivalence cases do not pin (codex r3 MED) --------


def test_continuous_treatment_is_median_binarised_in_the_rebuilt_model():
    df, cov = _shape_frames()["continuous_treatment"]
    model, _est, e, _ = _build_dowhy_estimate(
        data=df,
        treatment="t",
        outcome="y",
        common_causes=cov,
        estimation_result={"selected_estimator": "ols", "ate": 0.4},
    )
    fitted = model._data["t"].to_numpy()
    expected = (df["t"].to_numpy() > np.median(df["t"].to_numpy())).astype(int)
    assert set(np.unique(fitted)) == {0, 1}
    assert np.array_equal(fitted, expected)
    assert np.isfinite(float(e.value))


def test_effect_modifiers_are_the_encoded_common_causes_on_a_forest_family_rebuild(monkeypatch):
    """OLS needs no effect modifiers, so the equivalence cases cannot see them
    dropped; a LinearDML rebuild (RF nuisances, n=300) does -- and its estimate
    must still match the default identifier."""
    df, cov = _shape_frames()["categorical"]
    result = {"selected_estimator": "LinearDML", "ate": 0.4}
    _m, est_new, e_new, method = _build_dowhy_estimate(
        data=df, treatment="t", outcome="y", common_causes=cov, estimation_result=result
    )
    assert method == "backdoor.econml.dml.LinearDML"
    encoded = list(est_new.get_backdoor_variables())
    assert "c0" in encoded and any(c.startswith("region_") for c in encoded)
    assert "region" not in encoded
    assert list(e_new.estimator._effect_modifier_names) == encoded

    import dowhy

    real_identify = dowhy.CausalModel.identify_effect

    def default_only(self, *args, **kwargs):
        kwargs["optimize_backdoor"] = False
        return real_identify(self, *args, **kwargs)

    monkeypatch.setattr(dowhy.CausalModel, "identify_effect", default_only)
    _m, est_old, e_old, _ = _build_dowhy_estimate(
        data=df, treatment="t", outcome="y", common_causes=cov, estimation_result=result
    )
    assert list(est_old.get_backdoor_variables()) == encoded
    assert float(e_new.value) == float(e_old.value)
