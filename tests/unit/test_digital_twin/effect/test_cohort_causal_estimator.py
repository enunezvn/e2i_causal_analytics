"""Recover-known-effect + de-confounding tests for the cohort causal estimator.

The synthetic-gold cohort DGP (scripts/backfill_segment_engagement.py) plants a
region-heterogeneous TRUE causal effect of engagement on conversion, CONFOUNDED by
market_share. A valid estimator must (a) recover the planted per-region CATE and the
population ATE, and (b) be LESS biased than a naive estimator that omits the confounder.
These are the acceptance gates for Direction 2 (design doc 2026-06-19).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.cohort_causal_estimator import (
    CohortCausalEffect,
    estimate_cohort_effect,
)
from src.digital_twin.effect.errors import EffectCause, EffectDataUnavailable

# Planted truth (mirrors TRUE_CATE_BY_REGION in the backfill DGP).
TRUE_CATE = {"northeast": 0.45, "west": 0.30, "south": 0.18, "midwest": 0.08}


def _make_confounded_cohort(n_per_region: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Region-heterogeneous causal effect of engagement on conversion, CONFOUNDED
    by market_share (drives BOTH the treatment propensity AND the outcome, beta=0.8),
    mirroring the gold-standard DGP. Treatment is binarized at the engagement median,
    and the planted effect is on that same binarization (so a correct estimator that
    binarizes at the median recovers tau)."""
    rng = np.random.default_rng(seed)
    frames = []
    for region, tau in TRUE_CATE.items():
        n = n_per_region
        market = rng.uniform(0.0, 1.0, n)  # confounder
        logvol = rng.normal(0.0, 1.0, n)
        # engagement (treatment) is confounded by market_share
        eng_logit = 1.6 * (market - 0.5) + rng.normal(0.0, 0.5, n)
        engagement = 10.0 / (1.0 + np.exp(-eng_logit))  # domain 0..10
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
        0.5
        + 0.8 * df["market_share"]  # the strong confounder on the outcome
        + df["_tau"] * t_bin  # the planted causal effect
        + rng.normal(0.0, 0.25, len(df))
    ).clip(lower=0.0)
    return df.drop(columns="_tau")


@pytest.mark.slow
def test_recovers_true_cate_by_region_and_population_ate():
    cohort = _make_confounded_cohort()
    eff = estimate_cohort_effect(cohort, "engagement_score")

    assert isinstance(eff, CohortCausalEffect)
    # Population ATE ~ n-weighted mean of region taus (~0.2525).
    true_ate = float(np.mean(list(TRUE_CATE.values())))
    assert abs(eff.ate - true_ate) < 0.10, f"ate {eff.ate} vs true {true_ate}"
    # Per-region CATE recovered within tolerance + correct ordering.
    for region, tau in TRUE_CATE.items():
        assert abs(eff.cate_by_region[region] - tau) < 0.12, (
            f"{region}: {eff.cate_by_region[region]} vs {tau}"
        )
    assert (
        eff.cate_by_region["northeast"]
        > eff.cate_by_region["south"]
        > eff.cate_by_region["midwest"]
    )
    # Honest CI: contains the ATE and is NOT the fake-tight 0.003-0.009 width.
    assert eff.ate_ci_lower < eff.ate < eff.ate_ci_upper
    assert (eff.ate_ci_upper - eff.ate_ci_lower) > 0.01


@pytest.mark.slow
def test_deconfounding_reduces_bias():
    """Omitting the market_share confounder INFLATES the estimate; adjusting for it
    moves the estimate closer to the planted truth."""
    cohort = _make_confounded_cohort()
    true_ate = float(np.mean(list(TRUE_CATE.values())))

    deconfounded = estimate_cohort_effect(
        cohort, "engagement_score", confounders=("market_share", "triggers_total_count")
    )
    naive = estimate_cohort_effect(cohort, "engagement_score", confounders=())

    assert abs(deconfounded.ate - true_ate) < abs(naive.ate - true_ate)
    assert naive.ate > deconfounded.ate  # confounding inflates upward


def test_fail_honest_on_degenerate_treatment():
    """All-constant treatment cannot identify an effect -> raise, never fabricate."""
    cohort = _make_confounded_cohort(n_per_region=200)
    cohort["engagement_score"] = 5.0  # no variation -> no contrast
    with pytest.raises(EffectDataUnavailable):
        estimate_cohort_effect(cohort, "engagement_score")


def test_fail_honest_on_insufficient_rows():
    cohort = _make_confounded_cohort(n_per_region=5)  # 20 rows total
    with pytest.raises(EffectDataUnavailable):
        estimate_cohort_effect(cohort, "engagement_score")


def test_target_regions_get_the_forests_effect_and_an_interval_on_those_rows():
    """#2015: a region-targeted question gets inference on the targeted rows. The point
    estimate is the region's CATE from the same fit; the cohort-wide fields are unchanged."""
    cohort = _make_confounded_cohort(n_per_region=400)
    whole = estimate_cohort_effect(cohort, "engagement_score")
    targeted = estimate_cohort_effect(cohort, "engagement_score", target_regions=["northeast"])
    # Two seeded fits agree to float rounding (measured: differences of ~1e-16).
    assert (targeted.ate, targeted.ate_ci_lower, targeted.ate_ci_upper) == pytest.approx(
        (whole.ate, whole.ate_ci_lower, whole.ate_ci_upper), abs=1e-9
    )
    assert targeted.target_regions == ["northeast"]
    assert targeted.target_n == 400
    assert targeted.target_ate == pytest.approx(whole.cate_by_region["northeast"], abs=1e-9)
    assert targeted.target_ci_lower < targeted.target_ate < targeted.target_ci_upper
    assert whole.target_ate is None and whole.target_n == 0

    pair = estimate_cohort_effect(
        cohort, "engagement_score", target_regions=["northeast", "midwest", "northeast"]
    )
    assert pair.target_regions == ["northeast", "midwest"]
    assert pair.target_n == 800
    assert pair.target_ate == pytest.approx(
        (whole.cate_by_region["northeast"] + whole.cate_by_region["midwest"]) / 2, abs=1e-9
    )


def test_a_target_region_without_a_contrast_is_refused():
    cohort = _make_confounded_cohort(n_per_region=300)
    with pytest.raises(EffectDataUnavailable, match="atlantis"):
        estimate_cohort_effect(cohort, "engagement_score", target_regions=["atlantis"])
    one_sided = cohort.copy()
    median = one_sided["engagement_score"].median()
    in_west = one_sided["region"] == "west"
    one_sided.loc[in_west, "engagement_score"] = median + 1.0  # every west row "treated"
    with pytest.raises(EffectDataUnavailable, match="west"):
        estimate_cohort_effect(one_sided, "engagement_score", target_regions=["west"])


def test_control_outcome_sd_is_the_outcome_spread_of_the_low_intensity_rows():
    """#2015: sizing an experiment on this contrast needs the outcome's spread in the
    comparison arm — the rows at or below the median treatment intensity, the estimator's
    own split — overall or within target regions."""
    from src.digital_twin.effect.cohort_causal_estimator import control_outcome_sd

    cohort = _make_confounded_cohort(n_per_region=300)
    control = cohort[cohort["engagement_score"] <= cohort["engagement_score"].median()]
    sd, n = control_outcome_sd(cohort, "engagement_score")
    assert n == len(control)
    assert sd == pytest.approx(control["cohort_conversion_outcome"].std(ddof=1))

    in_target = control[control["region"].isin(["northeast", "west"])]
    sd_t, n_t = control_outcome_sd(cohort, "engagement_score", regions=["northeast", "west"])
    assert n_t == len(in_target)
    assert sd_t == pytest.approx(in_target["cohort_conversion_outcome"].std(ddof=1))

    with pytest.raises(EffectDataUnavailable, match="atlantis") as caught:
        control_outcome_sd(cohort, "engagement_score", regions=["atlantis"])
    assert caught.value.cause is EffectCause.TOO_FEW_USABLE_ROWS


def test_estimator_scoped_to_target_regions_reports_that_regions_effect():
    """#2023: a region-filtered simulation must be ESTIMATED on those regions.

    The engine seam scoped to target regions reports the targeted effect and its interval
    as the headline estimate — the same numbers the chat ``counterfactual_simulator``
    reports for the same regions — and carries the cohort-wide estimate it narrowed from
    alongside, so nothing is lost.
    """
    from src.digital_twin.effect.cohort_causal_estimator import CohortCausalEstimator
    from src.digital_twin.effect.provider import CohortEffectDataProvider

    cohort = _make_confounded_cohort(n_per_region=200)
    frame = CohortEffectDataProvider(cohort).get_training_frame(
        "digital_engagement", brand="Kisqali", twin_type="hcp"
    )
    twins = pd.DataFrame({"region": ["northeast"] * 50})

    whole = CohortCausalEstimator().estimate(frame, twins)
    scoped = CohortCausalEstimator(target_regions=["northeast"]).estimate(frame, twins)
    reference = estimate_cohort_effect(cohort, "engagement_score", target_regions=["northeast"])

    assert scoped.target_regions == ["northeast"]
    assert scoped.ate == pytest.approx(reference.target_ate, abs=1e-9)
    assert scoped.ate_ci_lower == pytest.approx(reference.target_ci_lower, abs=1e-9)
    assert scoped.ate_ci_upper == pytest.approx(reference.target_ci_upper, abs=1e-9)
    assert scoped.n_train == reference.target_n

    # The cohort-wide estimate rides along rather than being replaced.
    assert scoped.cohort_ate == pytest.approx(whole.ate, abs=1e-9)
    assert scoped.cohort_ci_lower == pytest.approx(whole.ate_ci_lower, abs=1e-9)
    assert scoped.cohort_ci_upper == pytest.approx(whole.ate_ci_upper, abs=1e-9)

    # An unscoped estimate is unchanged: cohort-wide headline, nothing narrowed away.
    assert whole.target_regions == []
    assert whole.cohort_ate is None

    # northeast's planted effect (0.45) is far above the cohort mean (~0.25). Reporting
    # the cohort number for a northeast-filtered run is the #2023 defect.
    assert abs(scoped.ate - whole.ate) > 0.05


def test_estimator_refuses_a_target_region_the_cohort_cannot_estimate():
    """#2023: a region the cohort cannot contrast is refused, as the chat tool refuses it —
    never silently answered with the cohort-wide effect."""
    from src.digital_twin.effect.cohort_causal_estimator import CohortCausalEstimator
    from src.digital_twin.effect.provider import CohortEffectDataProvider

    cohort = _make_confounded_cohort(n_per_region=200)
    frame = CohortEffectDataProvider(cohort).get_training_frame(
        "digital_engagement", brand="Kisqali", twin_type="hcp"
    )
    twins = pd.DataFrame({"region": ["atlantis"] * 50})

    with pytest.raises(EffectDataUnavailable, match="atlantis"):
        CohortCausalEstimator(target_regions=["atlantis"]).estimate(frame, twins)


# ---------------------------------------------------------------------------
# #2021 9b: each refusal names its cause; the message is unchanged
# (the fit-failure and target-inference causes are asserted in test_refusal_text_is_authored_2020.py,
# next to its _FitRaises / _TargetIntervalRaises fakes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("drop", "details", "message"),
    [
        pytest.param(
            "engagement_score",
            {"n_rows": 240, "has_treatment_column": False},
            "cohort missing treatment column 'engagement_score'.",
            id="treatment",
        ),
        pytest.param(
            "cohort_conversion_outcome",
            {
                "n_rows": 240,
                "has_treatment_column": True,
                "has_outcome_column": False,
                "has_region_column": True,
            },
            "cohort missing required column(s): need 'cohort_conversion_outcome' and 'region'.",
            id="outcome",
        ),
        pytest.param(
            "region",
            {
                "n_rows": 240,
                "has_treatment_column": True,
                "has_outcome_column": True,
                "has_region_column": False,
            },
            "cohort missing required column(s): need 'cohort_conversion_outcome' and 'region'.",
            id="region",
        ),
        pytest.param(
            "market_share",
            {"n_rows": 240, "n_missing_confounder_columns": 1},
            "cohort missing required confounder column(s) ['market_share']; refusing to "
            "produce an under-adjusted estimate.",
            id="confounder",
        ),
    ],
)
def test_a_missing_column_is_named_as_the_cause(drop, details, message):
    cohort = _make_confounded_cohort(n_per_region=60).drop(columns=drop)
    with pytest.raises(EffectDataUnavailable) as caught:
        estimate_cohort_effect(cohort, "engagement_score")
    assert str(caught.value) == message
    assert caught.value.cause is EffectCause.REQUIRED_COLUMN_MISSING
    assert caught.value.details == details


def test_too_few_rows_is_named_as_the_cause():
    with pytest.raises(EffectDataUnavailable) as caught:
        estimate_cohort_effect(_make_confounded_cohort(n_per_region=5), "engagement_score")
    assert str(caught.value) == "cohort has 20 usable rows (< 200) for 'engagement_score'."
    assert caught.value.cause is EffectCause.TOO_FEW_USABLE_ROWS
    assert caught.value.details == {"n_usable_rows": 20, "n_min_usable_rows": 200}


def test_a_constant_treatment_is_named_as_no_contrast():
    cohort = _make_confounded_cohort(n_per_region=200)
    cohort["engagement_score"] = 5.0
    with pytest.raises(EffectDataUnavailable) as caught:
        estimate_cohort_effect(cohort, "engagement_score")
    assert str(caught.value) == (
        "treatment 'engagement_score' has no median contrast (all rows on one side); "
        "cannot identify an effect."
    )
    assert caught.value.cause is EffectCause.NO_TREATMENT_CONTRAST
    assert caught.value.details == {"n_usable_rows": 800, "n_distinct_treatment_values": 1}


@pytest.fixture(scope="module")
def west_all_treated() -> pd.DataFrame:
    """Every west row far above any median intensity, so west has treated rows only."""
    cohort = _make_confounded_cohort(n_per_region=100)
    cohort.loc[cohort["region"] == "west", "engagement_score"] = 100.0
    return cohort


_COHORT_REGIONS = "['midwest', 'northeast', 'south', 'west']"


@pytest.mark.parametrize(
    ("targets", "first", "absent", "one_arm"),
    [
        pytest.param(["atlantis"], "atlantis", 1, 0, id="absent"),
        pytest.param(["west"], "west", 0, 1, id="one-arm"),
        # Counted over every target; the message still names the first failing one.
        pytest.param(["northeast", "atlantis", "west"], "atlantis", 1, 1, id="both"),
    ],
)
def test_a_target_region_without_a_contrast_is_named_as_not_covered(
    west_all_treated, targets, first, absent, one_arm
):
    with pytest.raises(EffectDataUnavailable) as caught:
        estimate_cohort_effect(west_all_treated, "engagement_score", target_regions=targets)
    assert str(caught.value) == (
        f"target region {first!r} has no treated-vs-control contrast in the cohort for "
        f"'engagement_score' (cohort regions: {_COHORT_REGIONS}); its effect cannot be estimated."
    )
    assert caught.value.cause is EffectCause.TARGET_REGION_NOT_COVERED
    assert caught.value.details == {
        "n_target_regions": len(targets),
        "n_target_regions_absent": absent,
        "n_target_regions_one_arm": one_arm,
        "n_cohort_regions": 4,
    }
