"""#2014: ``evalue.point_e_value`` — the point E-value of an effect that has no interval."""

from __future__ import annotations

import pytest

from src.causal_engine import evalue


@pytest.mark.parametrize(
    ("effect", "ci", "baseline_risk", "naive"),
    [
        (0.3, (0.1, 0.5), None, None),  # standardized-difference path
        (-0.2, (-0.35, -0.05), None, 0.1),
        (0.1, (0.05, 0.15), 0.3, None),  # risk-ratio path
        (0.1, (0.05, 0.15), 0.3, 0.2),
    ],
)
def test_point_e_value_matches_classify_when_the_bound_is_in_domain(
    effect, ci, baseline_risk, naive
):
    reading = evalue.classify(
        effect,
        ci,
        randomized=False,
        baseline_risk=baseline_risk,
        outcome_std=None,
        naive_effect=naive,
        covariate_factors={},
        n_rows=None,
    )

    e_point, rr_point, conversion = evalue.point_e_value(
        effect, baseline_risk=baseline_risk, outcome_std=None, naive_effect=naive
    )

    assert e_point == pytest.approx(reading.e_value_point, rel=1e-12)
    assert rr_point == pytest.approx(reading.rr_point, rel=1e-12)
    assert conversion == reading.conversion


def test_point_e_value_falls_back_to_the_smd_path_outside_the_risk_domain():
    _, _, conversion = evalue.point_e_value(0.9, baseline_risk=0.3, outcome_std=None)

    assert conversion == "standardized_difference"


def test_point_e_value_refuses_non_finite_input():
    with pytest.raises(ValueError):
        evalue.point_e_value(float("nan"), baseline_risk=None, outcome_std=None)
