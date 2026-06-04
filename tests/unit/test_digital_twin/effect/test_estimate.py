import numpy as np

from src.digital_twin.effect.estimate import (
    PROVENANCE_RWD,
    PROVENANCE_SYNTHETIC,
    EffectEstimate,
)


def test_effect_estimate_holds_fields_and_summarizes_uplift():
    est = EffectEstimate(
        ate=0.12,
        ate_ci_lower=0.08,
        ate_ci_upper=0.16,
        att=0.13,
        atc=0.11,
        per_twin_uplift=np.array([0.10, 0.12, 0.14]),
        auuc=None,
        qini=None,
        feature_importances={"decile": 0.5},
        n_train=2000,
        estimator_type="uplift_random_forest",
        data_provenance=PROVENANCE_SYNTHETIC,
    )
    assert est.ate == 0.12
    assert est.ci_width() == 0.16 - 0.08
    summary = est.uplift_summary()
    assert summary["n"] == 3
    assert summary["mean"] > 0
    assert PROVENANCE_SYNTHETIC == "synthetic_uplift_v1"
    assert PROVENANCE_RWD == "rwd_uplift"


def test_uplift_summary_empty_returns_only_n():
    est = EffectEstimate(
        ate=0.0,
        ate_ci_lower=0.0,
        ate_ci_upper=0.0,
        att=None,
        atc=None,
        per_twin_uplift=np.array([]),
        auuc=None,
        qini=None,
        feature_importances=None,
        n_train=0,
        estimator_type="test",
        data_provenance=PROVENANCE_SYNTHETIC,
    )
    summary = est.uplift_summary()
    assert summary == {"n": 0}
    assert "mean" not in summary
