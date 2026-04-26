"""Unit tests for the ``data_transformer`` node.

Block 1B (#18): the misleading comment claiming ``LabelEncoder`` was fit on
"all unique values across splits" was deleted. These tests lock in the
correct behaviour: encoders fit on TRAIN only, with ``_safe_label_encode``
absorbing unseen categories at val/test time.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.data_transformer import (
    _safe_label_encode,
    transform_data,
)


@pytest.fixture
def disjoint_categorical_splits() -> dict:
    """Build train/val/test with deliberately disjoint categorical sets.

    The categorical column ``brand`` carries values that exist ONLY in train,
    ONLY in val, and ONLY in test. If the encoder fits on the union, val/test
    rows would be encoded with an in-vocabulary id; if the encoder is
    train-only and ``_safe_label_encode`` is in place, val/test rows fall
    onto the sentinel id ``len(classes_)``.
    """
    train_df = pd.DataFrame(
        {
            "brand": ["A", "A", "B", "B", "C", "C"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    validation_df = pd.DataFrame(
        {
            "brand": ["A", "VAL_ONLY"],
            "value": [7.0, 8.0],
            "target": [1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "brand": ["B", "TEST_ONLY"],
            "value": [9.0, 10.0],
            "target": [0, 1],
        }
    )
    return {"train_df": train_df, "validation_df": validation_df, "test_df": test_df}


@pytest.fixture
def base_state(disjoint_categorical_splits) -> dict:
    """Wrap the splits in the state dict ``transform_data`` expects."""
    return {
        "experiment_id": "exp_block1b_data_transformer_test",
        "scope_spec": {
            "target_column": "target",
            "encoding_method": "label",
            # Use minmax for determinism — assertions below don't depend on it
            # but standard scaler can produce noisy near-zero values when the
            # variance is small, which complicates other test extensions.
            "scaling_method": "minmax",
            "imputation_strategy": "mean",
            "extract_datetime_features": False,
        },
        **disjoint_categorical_splits,
    }


class TestSafeLabelEncode:
    """Direct tests on the ``_safe_label_encode`` helper."""

    def test_unseen_values_get_sentinel_id(self):
        """Unseen labels must encode to ``len(classes_)``."""
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        encoder.fit(["A", "B", "C"])

        encoded = _safe_label_encode(encoder, pd.Series(["A", "B", "Z", "C"]))
        # A=0, B=1, Z=3 (sentinel), C=2 — the order depends on alphabetical fit.
        # We assert that the unseen value is mapped to len(classes_) = 3.
        assert encoded[2] == 3
        # Known values stay within [0, len(classes_)).
        assert encoded[0] < 3 and encoded[1] < 3 and encoded[3] < 3


@pytest.mark.asyncio
class TestEncoderFitTrainOnly:
    """Block 1B coverage: encoders MUST fit on train only.

    A regression where the encoder fits on the union of splits would be
    masked by the previous (now-deleted) comment that claimed exactly that
    behaviour. These tests exercise the actual branch.
    """

    async def test_label_encoder_classes_match_train_only(self, base_state):
        """``encoder.classes_`` must equal the unique train values verbatim."""
        result = await transform_data(base_state)

        encoders = result["encoders"]
        assert "brand" in encoders, "brand encoder must be fitted"
        encoder = encoders["brand"]

        train_uniques = sorted(base_state["train_df"]["brand"].astype(str).unique())
        assert sorted(encoder.classes_.tolist()) == train_uniques, (
            "Encoder classes_ must match train uniques exactly. If this "
            "fails, the encoder leaked val/test categories into its "
            "vocabulary — i.e., the deleted comment described real (and "
            "wrong) behaviour."
        )
        # Explicit check that val/test-only categories are NOT present.
        assert "VAL_ONLY" not in encoder.classes_
        assert "TEST_ONLY" not in encoder.classes_

    async def test_unseen_val_test_values_get_sentinel(self, base_state):
        """Val/test rows with train-unseen categories use the sentinel id."""
        result = await transform_data(base_state)

        encoders = result["encoders"]
        encoder = encoders["brand"]
        sentinel = len(encoder.classes_)
        # Train brand encodes to ids [0, 1, 2] (3 classes); sentinel = 3.
        # The MinMax scaler fits on train ⇒ max(train) = 2, min = 0, so the
        # scaled sentinel = (3 - 0) / (2 - 0) = 1.5. Anything < 1.5 is a
        # known category; exactly 1.5 is the sentinel.
        n_known = len(encoder.classes_)
        train_min, train_max = 0, n_known - 1
        expected_scaled_sentinel = (sentinel - train_min) / (train_max - train_min)

        validation_df = result["X_val"]
        test_df = result["X_test"]
        assert validation_df is not None and test_df is not None

        # Find the rows that were "VAL_ONLY" / "TEST_ONLY" in the input.
        original_val = base_state["validation_df"].reset_index(drop=True)
        val_only_idx = original_val.index[original_val["brand"] == "VAL_ONLY"][0]
        original_test = base_state["test_df"].reset_index(drop=True)
        test_only_idx = original_test.index[original_test["brand"] == "TEST_ONLY"][0]

        scaled_val_brand = validation_df["brand"].iloc[val_only_idx]
        scaled_test_brand = test_df["brand"].iloc[test_only_idx]
        assert scaled_val_brand == pytest.approx(expected_scaled_sentinel), (
            "Validation row with unseen category must be encoded at the "
            "sentinel value (= len(classes_) before scaling). If the encoder "
            "fitted on the union of splits, this row would have a known id "
            "instead and the scaled value would land within [0, 1]."
        )
        assert scaled_test_brand == pytest.approx(expected_scaled_sentinel), (
            "Test row with unseen category must use the sentinel encoding."
        )

    async def test_only_train_seen_values_stay_in_known_range(self, base_state):
        """Val/test rows whose category exists in train are encoded normally."""
        result = await transform_data(base_state)
        encoder = result["encoders"]["brand"]
        n_classes = len(encoder.classes_)

        # Validation row 0 (brand=A) must NOT be the sentinel post-scaling.
        # The sentinel post-MinMax = sentinel_id / (n_known - 1), which is
        # strictly greater than 1.0 when sentinel_id == n_known and
        # n_known > 1. Known categories produce values within [0, 1].
        validation_df = result["X_val"]
        assert validation_df is not None
        # Row 0 of validation has brand="A" which is known to the encoder.
        scaled_known = validation_df["brand"].iloc[0]
        assert 0.0 <= scaled_known <= 1.0, (
            f"Known category should scale within [0, 1], got {scaled_known}"
        )

        # Sanity: the encoder vocabulary is strictly smaller than the union.
        all_uniques = (
            set(base_state["train_df"]["brand"])
            | set(base_state["validation_df"]["brand"])
            | set(base_state["test_df"]["brand"])
        )
        assert n_classes < len(all_uniques)


@pytest.mark.asyncio
async def test_imputer_fits_on_train_only(disjoint_categorical_splits):
    """The imputer is also fit-on-train; verify with disjoint NaN patterns."""
    train_df = disjoint_categorical_splits["train_df"].copy()
    train_df.loc[0, "value"] = np.nan  # one NaN in train
    val_df = disjoint_categorical_splits["validation_df"].copy()
    val_df.loc[0, "value"] = np.nan  # NaN in val
    test_df = disjoint_categorical_splits["test_df"].copy()
    test_df.loc[0, "value"] = np.nan  # NaN in test

    state = {
        "experiment_id": "exp_imputer_train_only",
        "scope_spec": {
            "target_column": "target",
            "encoding_method": "label",
            "scaling_method": "minmax",
            "imputation_strategy": "mean",
            "extract_datetime_features": False,
        },
        "train_df": train_df,
        "validation_df": val_df,
        "test_df": test_df,
    }

    result = await transform_data(state)
    imputers = result["imputers"]
    assert "numeric" in imputers
    # SimpleImputer's ``statistics_`` is computed from the training data only.
    expected_train_mean = train_df["value"].dropna().mean()
    assert imputers["numeric"].statistics_[0] == pytest.approx(expected_train_mean)
