"""Unit tests for FeatureBuilder — leakage-safe frame contract + fit/transform.

Tests are pure (no I/O): build_from_frame / transform take injected DataFrames.
The live DB loader (build_for_split) is a documented stub; not tested here.
"""

import pandas as pd
import pytest

from src.mlops.gold_standard_eval.cohort_spec import INITIATION
from src.mlops.gold_standard_eval.feature_builder import (
    KEEP_COLUMNS,
    LEAKAGE_DENYLIST,
    FeatureBuilder,
)


def test_feature_builder_is_leakage_safe_and_restricts_to_keep_columns():
    fb = FeatureBuilder(INITIATION)
    raw = pd.DataFrame(
        {
            "patient_id": ["scvpt_1", "scvpt_2"],
            "treatment_initiated": [1, 0],
            "days_to_treatment": [10, None],  # post-anchor leakage → dropped
            "disease_severity": [0.8, 0.2],  # KEEP_COLUMNS numeric
            "academic_hcp": [1, 0],  # KEEP_COLUMNS numeric
            "geographic_region": ["west", "south"],  # KEEP_COLUMNS categorical
            "age_group": ["45-54", "65-74"],  # NOT in KEEP_COLUMNS → dropped
            "risk_score": [0.7, 0.3],  # NOT in KEEP_COLUMNS → dropped
        }
    )
    X, y = fb.build_from_frame(raw)
    assert list(y) == [1, 0]
    assert "treatment_initiated" not in X.columns
    for col in LEAKAGE_DENYLIST:
        assert col not in X.columns
    # KEEP_COLUMNS restriction: only allowlisted raw columns produce features.
    assert "age_group" not in X.columns
    assert any(c.startswith("age_group") for c in X.columns) is False
    assert "risk_score" not in X.columns
    assert "disease_severity" in X.columns
    assert "academic_hcp" in X.columns
    assert any(c.startswith("geographic_region_") for c in X.columns)
    assert not X.isnull().any().any()  # imputed, no NaNs reach the model
    assert len(fb.feature_columns) == X.shape[1]


def test_transform_reindexes_eval_to_fitted_columns():
    """The critical behavior: train/eval one-hot column sets WILL differ.

    transform() must align an eval frame to the train-fitted feature_columns —
    categories present at fit but absent in eval are filled with 0.0, and
    categories that appear ONLY in eval are dropped. Without this, a cross-split
    score is computed over a mismatched column space and is meaningless.
    """
    fb = FeatureBuilder(INITIATION)
    train = pd.DataFrame(
        {
            "treatment_initiated": [1, 0, 1, 0],
            "disease_severity": [0.9, 0.1, 0.8, 0.2],
            "academic_hcp": [1, 0, 1, 0],
            "geographic_region": ["west", "south", "northeast", "midwest"],
        }
    )
    X_train, _ = fb.build_from_frame(train)
    fitted_cols = list(fb.feature_columns)
    # All four regions seen at fit.
    for region in ("west", "south", "northeast", "midwest"):
        assert f"geographic_region_{region}" in fitted_cols

    eval_df = pd.DataFrame(
        {
            "treatment_initiated": [0, 1],
            "disease_severity": [0.5, 0.3],
            "academic_hcp": [0, 1],
            # Eval has only 'west' (subset) + an UNSEEN 'narnia' region.
            "geographic_region": ["west", "narnia"],
        }
    )
    X_eval = fb.transform(eval_df)

    # Identical column space + order to the fitted matrix.
    assert list(X_eval.columns) == fitted_cols
    # Absent-at-eval region columns are present and zero-filled.
    assert (X_eval["geographic_region_south"] == 0.0).all()
    assert (X_eval["geographic_region_midwest"] == 0.0).all()
    # 'west' row is correctly hot; the unseen 'narnia' row is all-zero across regions.
    assert X_eval.loc[0, "geographic_region_west"] == 1.0
    region_cols = [c for c in fitted_cols if c.startswith("geographic_region_")]
    assert X_eval.loc[1, region_cols].sum() == 0.0  # 'narnia' dropped, no leak
    # Same #columns as train → directly scorable by a model fit on X_train.
    assert X_eval.shape[1] == X_train.shape[1]


def test_transform_one_hots_pandas_string_dtype_categorical():
    """A categorical column in the pandas ``string`` dtype must one-hot, not crash.

    pandas 3.0 (future.infer_string=True) infers fresh string columns as the
    ``string`` extension dtype, NOT ``object``. The #39 SHAP serving path builds
    eval frames from raw {name: value} dicts, so ``geographic_region`` arrives as
    ``string`` in that runtime. The old ``s.dtype == object`` categorical check
    missed ``string`` → the column fell through to the numeric branch →
    ``astype(float)`` on "northeast" raised
    ``ValueError: could not convert string to float``. We force the ``string``
    dtype here so the regression reproduces on any pandas version (the droplet's
    bentoml container runs pandas 3.0; CI/venv may be 2.x).
    """
    fb = FeatureBuilder(INITIATION)
    train = pd.DataFrame(
        {
            "treatment_initiated": [1, 0, 1, 0],
            "disease_severity": [0.9, 0.1, 0.8, 0.2],
            "academic_hcp": [1, 0, 1, 0],
            "geographic_region": ["west", "south", "northeast", "midwest"],
        }
    )
    fb.build_from_frame(train)
    fitted_cols = list(fb.feature_columns)

    eval_df = pd.DataFrame(
        {
            "disease_severity": [0.5],
            "academic_hcp": [0],
            "geographic_region": pd.array(["northeast"], dtype="string"),
        }
    )
    # Precondition: the categorical column really is the extension string dtype,
    # not object — otherwise the test would not exercise the regression.
    assert eval_df["geographic_region"].dtype != object
    assert pd.api.types.is_string_dtype(eval_df["geographic_region"])

    # Must NOT raise (the old code raised "could not convert string to float").
    X_eval = fb.transform(eval_df)

    # geographic_region is one-hot-encoded (categorical), not imputed as numeric.
    assert list(X_eval.columns) == fitted_cols
    assert X_eval.loc[0, "geographic_region_northeast"] == 1.0
    region_cols = [c for c in fitted_cols if c.startswith("geographic_region_")]
    assert X_eval.loc[0, region_cols].sum() == 1.0  # exactly one region hot
    # numeric covariates still flow through the numeric (impute) path.
    assert "geographic_region" not in X_eval.columns  # raw cat col not kept as-is
    assert X_eval.loc[0, "disease_severity"] == 0.5


def test_transform_imputes_with_fitted_train_median_not_eval_median():
    fb = FeatureBuilder(INITIATION)
    train = pd.DataFrame(
        {
            "treatment_initiated": [1, 0, 1],
            "disease_severity": [0.2, 0.4, 0.6],  # train median = 0.4
            "academic_hcp": [1, 0, 1],
            "geographic_region": ["west", "south", "west"],
        }
    )
    fb.build_from_frame(train)
    eval_df = pd.DataFrame(
        {
            "disease_severity": [None, 0.9],  # eval median (0.9) must NOT be used
            "academic_hcp": [0, 1],
            "geographic_region": ["west", "south"],
        }
    )
    X_eval = fb.transform(eval_df)
    assert X_eval.loc[0, "disease_severity"] == pytest.approx(0.4)  # train median
    assert X_eval.loc[0, "disease_severity__isna"] == 1.0


def test_transform_before_fit_raises():
    fb = FeatureBuilder(INITIATION)
    with pytest.raises(RuntimeError, match="before build_from_frame"):
        fb.transform(pd.DataFrame({"disease_severity": [0.5]}))


def test_keep_columns_equals_base_covariates():
    # The locked set is the codebase-intent base covariate seed (Task 3 EXP lock).
    assert KEEP_COLUMNS == INITIATION.base_covariates


def test_empty_keep_columns_disables_allowlist():
    fb = FeatureBuilder(INITIATION, keep_columns=())
    raw = pd.DataFrame(
        {
            "treatment_initiated": [1, 0],
            "disease_severity": [0.8, 0.2],
            "age_group": ["45-54", "65-74"],  # kept when allowlist disabled
            "outcome_probability": [0.9, 0.1],  # still denylisted → dropped
        }
    )
    X, _ = fb.build_from_frame(raw)
    assert any(c.startswith("age_group") for c in X.columns)
    assert "outcome_probability" not in X.columns


def test_load_frame_omits_brand_filter_when_brand_none():
    import asyncio

    from src.mlops.gold_standard_eval.cohort_spec import PERSISTENCE
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

    calls = {"eq": []}

    class _Q:
        def select(self, *a, **k):
            return self

        def eq(self, col, val):
            calls["eq"].append(col)
            return self

        def in_(self, *a, **k):
            return self

        def lt(self, *a, **k):
            return self

        def order(self, *a, **k):
            return self

        def range(self, *a, **k):
            return self

        async def execute(self):
            class R:
                data = []

            return R()

    class _DB:
        def table(self, *a, **k):
            return _Q()

    fb = FeatureBuilder(PERSISTENCE)
    asyncio.run(fb.load_frame(_DB()))
    assert "brand" not in calls["eq"]  # all-brands: no brand filter
    assert "is_synthetic" in calls["eq"]  # synthetic provenance still enforced
