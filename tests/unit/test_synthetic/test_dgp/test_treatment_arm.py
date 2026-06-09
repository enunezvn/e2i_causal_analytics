"""Task 03.1 — confounded binary treatment arm with estimable propensity + overlap."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.ml.synthetic.dgp.treatment_arm import assign_treatment_arm


def test_arm_is_confounded_estimable_with_overlap():
    rng = np.random.default_rng(42)
    n = 1000
    disease_severity = np.clip(rng.normal(5.0, 2.0, n), 0, 10)
    academic_hcp = (rng.random(n) < 0.30).astype(int)
    X = {"disease_severity": disease_severity, "academic_hcp": academic_hcp}

    arm, propensity = assign_treatment_arm(
        X, rng, beta_severity=0.30, beta_academic=0.80, intercept=-2.0
    )

    # binary arm
    assert set(np.unique(arm)).issubset({0, 1})
    # overlap: no near-separation
    assert propensity.min() >= 0.01 and propensity.max() <= 0.99
    # both arms populated; control >= 100
    assert arm.sum() >= 30 and (n - arm.sum()) >= 100
    # confounding present + estimable: logistic on X recovers e(X), AUC > 0.5
    Xmat = np.column_stack([disease_severity, academic_hcp])
    pred = LogisticRegression(max_iter=1000).fit(Xmat, arm).predict_proba(Xmat)[:, 1]
    assert roc_auc_score(arm, pred) > 0.5
    # the designed propensity itself separates the arms (true confounding, not noise)
    assert propensity[arm == 1].mean() > propensity[arm == 0].mean()
