import numpy as np

from src.digital_twin.effect.estimate import PROVENANCE_SYNTHETIC, EffectEstimate
from src.digital_twin.effect.recommendation import (
    PolicyThresholds,
    Recommendation,
    RecommendationPolicy,
)


def _est(ate, lo, hi):
    return EffectEstimate(
        ate=ate,
        ate_ci_lower=lo,
        ate_ci_upper=hi,
        att=None,
        atc=None,
        per_twin_uplift=np.array([ate]),
        auuc=None,
        qini=None,
        feature_importances=None,
        n_train=2000,
        estimator_type="uplift_random_forest",
        data_provenance=PROVENANCE_SYNTHETIC,
    )


def test_deploy_when_ci_lower_above_threshold():
    policy = RecommendationPolicy(PolicyThresholds(min_effect=0.05))
    rec, rationale, n = policy.decide(_est(0.12, 0.07, 0.17), baseline_rate=0.3)
    assert rec is Recommendation.DEPLOY
    assert n > 0
    assert "lower bound" in rationale.lower()


def test_skip_when_ci_upper_below_threshold():
    policy = RecommendationPolicy(PolicyThresholds(min_effect=0.05))
    rec, _, n = policy.decide(_est(0.01, -0.02, 0.04), baseline_rate=0.3)
    assert rec is Recommendation.SKIP


def test_refine_when_ci_straddles_threshold():
    policy = RecommendationPolicy(PolicyThresholds(min_effect=0.05))
    rec, _, _ = policy.decide(_est(0.06, 0.01, 0.11), baseline_rate=0.3)
    assert rec is Recommendation.REFINE
