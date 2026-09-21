"""#2054: each estimator DECLARES which subgroup axes its scores actually resolve.

The engine used to average ``per_twin_uplift`` over the generated twins for all four
``EffectHeterogeneity`` dimensions. That rule is valid for ``TwinEffectEstimator``
(a real per-twin score varying inside every group) and invalid for
``CohortCausalEstimator``, whose per-twin score is a STEP FUNCTION OF REGION: every twin
in a region carries the identical value, so a ``by_specialty`` average is just the twin
region mixture mean and its apparent spread is sampling noise in the twin draw (measured:
specialty spread 0.049 at 100 twins decaying to 0.002 at 100k, region spread invariant).

So the estimator — the only object that knows what its scores resolve — declares it, and
the engine reports only the declared axes with the estimator's OWN evidence count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.cohort_causal_estimator import (
    CohortCausalEstimator,
    estimate_cohort_effect,
)
from src.digital_twin.effect.estimate import SUBGROUP_AXES, EffectEstimate
from src.digital_twin.effect.provider import CohortEffectDataProvider
from src.digital_twin.models.simulation_models import EffectHeterogeneity

TRUE_CATE = {"northeast": 0.45, "west": 0.30, "south": 0.18, "midwest": 0.08}


def _make_cohort(n_per_region: int = 200, seed: int = 42) -> pd.DataFrame:
    """Region-heterogeneous causal effect of engagement on conversion, confounded by
    market_share — the same DGP as ``test_cohort_causal_estimator.py``, with UNEQUAL
    region sizes so a per-region row count cannot coincide with the twin count."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, (region, tau) in enumerate(TRUE_CATE.items()):
        n = n_per_region + 40 * i  # unequal on purpose
        market = rng.uniform(0.0, 1.0, n)
        logvol = rng.normal(0.0, 1.0, n)
        eng_logit = 1.6 * (market - 0.5) + rng.normal(0.0, 0.5, n)
        engagement = 10.0 / (1.0 + np.exp(-eng_logit))
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "engagement_score": engagement,
                    "market_share": market,
                    "triggers_total_count": np.expm1(np.abs(logvol) * 2.0),
                    "_tau": tau,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    t_bin = (df["engagement_score"] > df["engagement_score"].median()).astype(float)
    df["cohort_conversion_outcome"] = (
        0.5 + 0.8 * df["market_share"] + df["_tau"] * t_bin + rng.normal(0.0, 0.25, len(df))
    ).clip(lower=0.0)
    return df.drop(columns="_tau")


def _frame(cohort: pd.DataFrame):
    return CohortEffectDataProvider(cohort).get_training_frame(
        "digital_engagement", brand="Kisqali", twin_type="hcp"
    )


def test_subgroup_axes_names_exactly_the_heterogeneity_dimensions():
    """The declaration vocabulary and the reported dimensions must not drift apart:
    an axis name that no ``by_*`` field consumes would be declared and never reported."""
    dimensions = {
        name.removeprefix("by_")
        for name in EffectHeterogeneity.model_fields
        if name.startswith("by_")
    }
    assert set(SUBGROUP_AXES) == dimensions, (
        "SUBGROUP_AXES and EffectHeterogeneity's by_* dimensions have drifted. A NEW "
        "dimension needs its name in SUBGROUP_AXES and an estimator that declares it in "
        "cate_by_axis, or nothing will ever resolve it and the engine will report {}. A "
        "REMOVED dimension must come out of SUBGROUP_AXES. A field on EffectHeterogeneity "
        "that is not a by_* subgroup dimension is not an axis and does not belong here."
    )


def test_declaration_defaults_to_empty_so_an_undeclared_estimate_resolves_nothing():
    """Fail-closed: the fields default to ``{}``, so every existing constructor call is
    unchanged and an estimator that declares nothing gets NO subgroup numbers rather
    than four averages that are only noise."""
    estimate = EffectEstimate(
        ate=0.1,
        ate_ci_lower=0.0,
        ate_ci_upper=0.2,
        att=None,
        atc=None,
        per_twin_uplift=np.array([0.1, 0.1]),
        auuc=None,
        qini=None,
        feature_importances=None,
        n_train=10,
        estimator_type="x",
        data_provenance="y",
    )
    assert estimate.cate_by_axis == {}
    assert estimate.n_by_axis == {}


def test_twin_estimator_declares_every_axis_resolved_by_its_per_twin_scores():
    """``TwinEffectEstimator``'s score is ``model.predict(x_twin)`` over ALL twin
    features, so it genuinely varies inside every subgroup and the engine's per-twin
    grouping is correct there. It declares each axis with an EMPTY mapping: resolved,
    with no precomputed group effect to report in its place."""
    from src.digital_twin.effect.estimator import TwinEffectEstimator
    from src.digital_twin.effect.provider import SyntheticEffectDataProvider

    provider = SyntheticEffectDataProvider(true_ate=0.1, seed=7)
    frame = provider.get_training_frame("email_campaign", brand="Kisqali", twin_type="hcp")
    twins = frame.df[frame.confounders].head(200).reset_index(drop=True)

    estimate = TwinEffectEstimator().estimate(frame, twins)

    assert set(estimate.cate_by_axis) == set(SUBGROUP_AXES)
    assert all(v == {} for v in estimate.cate_by_axis.values())
    assert estimate.n_by_axis == {}


def test_cohort_estimator_declares_only_region_with_cohort_row_counts():
    """The cohort estimator resolves REGION and nothing else, and its evidence is cohort
    rows — not twins. ``specialty``/``decile``/``adoption_stage`` are undeclared because
    the estimate carries no information about them (and ``decile`` is 100% null in
    ``hcp_profiles`` while ``adoption_stage`` exists in no table at all)."""
    cohort = _make_cohort()
    frame = _frame(cohort)
    twins = pd.DataFrame({"region": list(TRUE_CATE) * 50})  # 200 twins, 50 per region

    estimate = CohortCausalEstimator().estimate(frame, twins)
    reference = estimate_cohort_effect(cohort, "engagement_score")

    assert set(estimate.cate_by_axis) == {"region"}
    assert set(estimate.n_by_axis) == {"region"}
    assert estimate.cate_by_axis["region"] == pytest.approx(reference.cate_by_region)

    expected_n = cohort["region"].value_counts().to_dict()
    assert estimate.n_by_axis["region"] == {r: int(expected_n[r]) for r in reference.cate_by_region}
    # The declared evidence is cohort rows, never the twin count (50 twins per region).
    assert all(n != 50 for n in estimate.n_by_axis["region"].values())


def test_cohort_declaration_is_scoped_to_the_targeted_regions():
    """A region-targeted estimate (#2023) resolves only the regions it was estimated on.
    Declaring the untargeted regions too would put a subgroup number beside a headline
    computed on a different population — the very defect #2023 fixed for the headline."""
    cohort = _make_cohort()
    frame = _frame(cohort)
    twins = pd.DataFrame({"region": ["northeast"] * 50})

    estimate = CohortCausalEstimator(target_regions=["northeast"]).estimate(frame, twins)

    assert set(estimate.cate_by_axis["region"]) == {"northeast"}
    assert estimate.n_by_axis["region"]["northeast"] == int((cohort["region"] == "northeast").sum())
    # A single targeted region's declared CATE IS the headline: both are the forest's
    # mean effect over that region's cohort rows.
    assert estimate.cate_by_axis["region"]["northeast"] == pytest.approx(estimate.ate, abs=1e-9)


def test_cohort_declaration_is_invariant_to_the_twin_count():
    """Twin count is a generation parameter. The declared region effects and their
    evidence counts come from the cohort, so they must not move with it."""
    cohort = _make_cohort()
    frame = _frame(cohort)

    declarations = []
    for n_twins in (100, 1000, 10000):
        twins = pd.DataFrame({"region": list(TRUE_CATE) * (n_twins // len(TRUE_CATE))})
        estimate = CohortCausalEstimator().estimate(frame, twins)
        declarations.append((estimate.cate_by_axis, estimate.n_by_axis))

    first_cate, first_n = declarations[0]
    for cate, n in declarations[1:]:
        # abs=1e-12 is the repeated-fit float jitter (~4e-18 measured), twelve orders of
        # magnitude below the twin-count-driven spread this pins out (0.049 -> 0.002).
        assert cate["region"] == pytest.approx(first_cate["region"], abs=1e-12, rel=0)
        assert n == first_n  # cohort row counts are exact


def test_a_non_string_region_column_still_yields_a_declared_region_axis():
    """The region declaration must never come back as an EMPTY mapping, which would read as
    "resolved by the per-twin scores" and send the region step function through the twin
    grouping. ``_usable_rows`` coerces the region column to str, so the CATE keys and the
    row-count keys derive from that one coerced Series and cannot disagree — pinned here
    with an int region column, the case where a mismatch would show up."""
    rng = np.random.default_rng(11)
    frames = []
    for code, tau in {1: 0.40, 2: 0.10, 3: 0.25}.items():  # int labels, not str
        n = 300
        market = rng.uniform(0.0, 1.0, n)
        eng_logit = 1.6 * (market - 0.5) + rng.normal(0.0, 0.5, n)
        frames.append(
            pd.DataFrame(
                {
                    "region": code,
                    "engagement_score": 10.0 / (1.0 + np.exp(-eng_logit)),
                    "market_share": market,
                    "triggers_total_count": rng.poisson(60, n).astype(float),
                    "_tau": tau,
                }
            )
        )
    cohort = pd.concat(frames, ignore_index=True)
    assert cohort["region"].dtype == "int64"
    t_bin = (cohort["engagement_score"] > cohort["engagement_score"].median()).astype(float)
    cohort["cohort_conversion_outcome"] = (
        0.2
        + 0.8 * cohort["market_share"]
        + cohort["_tau"] * t_bin
        + rng.normal(0, 0.05, len(cohort))
    )
    cohort = cohort.drop(columns="_tau")

    estimate = CohortCausalEstimator().estimate(
        CohortEffectDataProvider(cohort).get_training_frame(
            "digital_engagement", brand="Kisqali", twin_type="hcp"
        ),
        pd.DataFrame({"region": ["1", "2", "3"] * 50}),
    )

    assert estimate.cate_by_axis["region"], "an empty declaration reads as per-twin-resolved"
    assert set(estimate.cate_by_axis["region"]) == {"1", "2", "3"}
    assert estimate.n_by_axis["region"] == {"1": 300, "2": 300, "3": 300}
