"""#2162: cohort-path specialty heterogeneity is estimated from cohort rows.

These tests deliberately exercise the real causal forest.  Specialty effects must not
come from grouping generated twins, and unsupported specialty labels must never acquire a
published effect through fallback scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.cohort_causal_estimator import (
    MIN_SPECIALTY_ARM_ROWS,
    MIN_SPECIALTY_ROWS,
    CohortCausalEstimator,
    _effect_modifier_matrix,
    _specialty_aggregates,
)
from src.digital_twin.effect.provider import CohortEffectDataProvider


def _cohort(seed: int = 2162) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    # Production synthetic-gold plants region effects only. Mirror that specialty-null
    # truth: this axis is an observed-composition summary, not planted interaction truth.
    groups = (("oncology", 360), ("hematology", 300), ("rare", 40))
    for specialty, n in groups:
        region = rng.choice(["north", "south"], size=n)
        market = rng.uniform(0.0, 1.0, n)
        volume = rng.lognormal(3.0, 0.7, n)
        raw = 1.4 * (market - 0.5) + rng.normal(0.0, 0.8, n)
        treatment = 10.0 / (1.0 + np.exp(-raw))
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "specialty": specialty,
                    "engagement_score": treatment,
                    "market_share": market,
                    "triggers_total_count": volume,
                    "_tau": np.where(region == "north", 0.24, 0.14),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    high = (df["engagement_score"] > df["engagement_score"].median()).astype(float)
    df["cohort_conversion_outcome"] = (
        0.25 + 0.7 * df["market_share"] + df["_tau"] * high + rng.normal(0, 0.16, len(df))
    )
    return df.drop(columns="_tau")


def _estimate(cohort: pd.DataFrame, twins: pd.DataFrame):
    frame = CohortEffectDataProvider(cohort).get_training_frame(
        "digital_engagement", brand="Kisqali", twin_type="hcp"
    )
    return CohortCausalEstimator().estimate(frame, twins)


@pytest.mark.slow
def test_specialty_is_a_second_declared_forest_axis_with_cohort_evidence():
    cohort = _cohort()
    twins = pd.DataFrame(
        {
            "region": ["north", "south"] * 50,
            "specialty": ["oncology"] * 50 + ["hematology"] * 50,
        }
    )

    estimate = _estimate(cohort, twins)

    assert set(estimate.cate_by_axis) == {"region", "specialty"}
    assert set(estimate.cate_by_axis["specialty"]) == {"oncology", "hematology"}
    assert (
        abs(
            estimate.cate_by_axis["specialty"]["oncology"]
            - estimate.cate_by_axis["specialty"]["hematology"]
        )
        < 0.12
    )
    assert estimate.n_by_axis["specialty"] == {"hematology": 300, "oncology": 360}
    provenance = estimate.axis_provenance["specialty"]
    assert provenance.source == "hcp_profiles.specialty"
    assert provenance.basis == "cohort_rows"
    assert provenance.min_group_rows == MIN_SPECIALTY_ROWS
    assert provenance.min_treated_rows == MIN_SPECIALTY_ARM_ROWS
    assert provenance.min_control_rows == MIN_SPECIALTY_ARM_ROWS
    assert provenance.fallback == "region_then_cohort"
    assert provenance.support_unit == "cohort_rows"
    assert provenance.estimand == "observed_region_mix_mean_cate"
    assert provenance.suppressed_groups == {"rare": "group_rows_below_minimum"}


@pytest.mark.slow
def test_unsupported_specialty_falls_back_for_scoring_but_is_not_published():
    cohort = _cohort()
    twins = pd.DataFrame(
        {
            "region": ["north", "south", "north"],
            "specialty": ["oncology", "rare", "not_in_cohort"],
        }
    )

    estimate = _estimate(cohort, twins)

    assert "rare" not in estimate.cate_by_axis["specialty"]
    assert "not_in_cohort" not in estimate.cate_by_axis["specialty"]
    assert estimate.per_twin_uplift[1] == pytest.approx(estimate.cate_by_axis["region"]["south"])
    assert estimate.per_twin_uplift[2] == pytest.approx(estimate.cate_by_axis["region"]["north"])


def test_one_arm_and_under_threshold_specialties_are_suppressed():
    specialty = np.array(["one_arm"] * 120 + ["too_small"] * 80 + ["supported"] * 120)
    treatment = np.array([1] * 120 + [0, 1] * 40 + [0, 1] * 60)
    effects = np.linspace(0.0, 0.3, len(specialty))

    cate, counts, suppressed = _specialty_aggregates(
        specialty, treatment, effects, np.ones(len(specialty), dtype=bool)
    )

    assert set(cate) == {"supported"}
    assert counts == {"supported": 120}
    assert suppressed == {
        "one_arm": "control_rows_below_minimum",
        "too_small": "group_rows_below_minimum",
    }


def test_specialty_aggregation_uses_only_target_region_rows():
    specialty = np.array(["oncology"] * 240)
    treatment = np.array([0, 1] * 120)
    effects = np.array([0.4] * 120 + [0.1] * 120)
    target_region_mask = np.array([True] * 120 + [False] * 120)

    cate, counts, suppressed = _specialty_aggregates(
        specialty, treatment, effects, target_region_mask
    )

    assert cate == {"oncology": pytest.approx(0.4)}
    assert counts == {"oncology": 120}
    assert suppressed == {}


def test_one_hot_modifier_geometry_is_invariant_to_category_label_order():
    work = pd.DataFrame(
        {
            "region": ["north", "north", "south", "south"],
            "specialty": ["oncology", "hematology", "oncology", "hematology"],
        }
    )
    renamed = work.replace({"north": "z", "south": "a", "oncology": "zz", "hematology": "aa"})

    x = _effect_modifier_matrix(work)
    x_renamed = _effect_modifier_matrix(renamed)

    assert x @ x.T == pytest.approx(x_renamed @ x_renamed.T)


def test_absent_specialty_preserves_the_region_only_modifier_shape():
    work = pd.DataFrame(
        {
            "region": ["north", "south", "north"],
            "specialty": ["__missing_specialty__"] * 3,
        }
    )

    matrix = _effect_modifier_matrix(work)

    assert matrix.shape == (3, 1)
    assert matrix.ravel().tolist() == [0.0, 1.0, 0.0]
