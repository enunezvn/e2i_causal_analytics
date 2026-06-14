"""Unit tests for FeatureBuilder — leakage-safe frame contract.

Tests are pure (no I/O): build_from_frame takes an injected DataFrame.
The live DB loader (build_for_split) is a documented stub; not tested here.
"""

import pandas as pd
import pytest
from src.mlops.gold_standard_eval.cohort_spec import INITIATION
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder, LEAKAGE_DENYLIST


def test_feature_builder_is_leakage_safe_and_complete():
    fb = FeatureBuilder(INITIATION)
    raw = pd.DataFrame({
        "patient_id": ["scvpt_1", "scvpt_2"],
        "treatment_initiated": [1, 0],
        "days_to_treatment": [10, None],      # post-anchor → must be dropped
        "disease_severity": ["high", "low"],
        "age_group": ["45-54", "65-74"],
        "risk_score": [0.7, 0.3],
    })
    X, y = fb.build_from_frame(raw)
    assert list(y) == [1, 0]
    assert "treatment_initiated" not in X.columns
    for col in LEAKAGE_DENYLIST:
        assert col not in X.columns
    assert not X.isnull().any().any()        # imputed, no NaNs reach the model
    assert len(fb.feature_columns) == X.shape[1]
