"""Tier-0 harness categorical encoding contract.

The harness must ONE-HOT encode nominal categoricals before the model sees
them. Ordinal/integer codes (the prior ``OrdinalEncoder`` path) impose a false
magnitude order on nominal categories, which miscalibrates the LINEAR champion:
on the disc cohort, post-Platt calibration slope deviation is ~0.18 with
ordinal codes (FAILS the 0.15 gate) vs ~0.07 with one-hot (PASSES) — same data,
same splits. One-hot lets the deployable linear model actually ship.

These tests pin the encode/re-apply contract on a tiny synthetic frame so the
faithful tier0 run (the deploy arbiter) is not the only safety net.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_tier0_test import (
    _apply_categorical_onehot,
    _fit_categorical_onehot,
    _raw_feature_cols,
)


class TestFitCategoricalOnehot:
    def test_expands_nominal_categorical_to_binary_columns(self) -> None:
        X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "payer": ["HMO", "PPO", "HMO", "EPO"]})
        X_enc, info = _fit_categorical_onehot(X, ["payer"])
        # the integer-coded ordinal column must NOT survive (that was the bug)
        assert "payer" not in X_enc.columns
        # one binary indicator per observed category
        for col in ("payer_HMO", "payer_PPO", "payer_EPO"):
            assert col in X_enc.columns, f"missing one-hot column {col}"
        block = X_enc[["payer_HMO", "payer_PPO", "payer_EPO"]].values
        assert set(np.unique(block)) <= {0.0, 1.0}
        # numeric columns pass through untouched
        assert "num" in X_enc.columns
        assert list(X_enc["num"]) == [1.0, 2.0, 3.0, 4.0]
        assert info["method"] == "onehot"
        assert info["columns"] == ["payer"]

    def test_no_categoricals_is_passthrough(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X_enc, info = _fit_categorical_onehot(X, [])
        assert list(X_enc.columns) == ["a", "b"]
        assert info["columns"] == []

    def test_multiple_categoricals_all_expanded(self) -> None:
        X = pd.DataFrame({"sex": ["F", "M", "F"], "region": ["west", "south", "west"]})
        X_enc, info = _fit_categorical_onehot(X, ["sex", "region"])
        assert "sex" not in X_enc.columns and "region" not in X_enc.columns
        assert {"sex_F", "sex_M", "region_west", "region_south"} <= set(X_enc.columns)


class TestApplyCategoricalOnehot:
    def test_reproduces_training_feature_space(self) -> None:
        Xtr = pd.DataFrame({"num": [1.0, 2.0, 3.0], "payer": ["HMO", "PPO", "EPO"]})
        Xtr_enc, info = _fit_categorical_onehot(Xtr, ["payer"])
        Xte = pd.DataFrame({"num": [4.0, 5.0], "payer": ["HMO", "EPO"]})
        Xte_enc = _apply_categorical_onehot(Xte, info)
        # identical column set + order to the training frame
        assert list(Xte_enc.columns) == list(Xtr_enc.columns)
        assert Xte_enc.iloc[0]["payer_HMO"] == 1.0
        assert Xte_enc.iloc[0]["payer_EPO"] == 0.0
        assert Xte_enc.iloc[1]["payer_EPO"] == 1.0

    def test_unknown_category_at_transform_is_all_zero(self) -> None:
        Xtr = pd.DataFrame({"payer": ["HMO", "PPO"]})
        _, info = _fit_categorical_onehot(Xtr, ["payer"])
        Xte = pd.DataFrame({"payer": ["BRAND_NEW"]})
        Xte_enc = _apply_categorical_onehot(Xte, info)
        assert Xte_enc.iloc[0]["payer_HMO"] == 0.0
        assert Xte_enc.iloc[0]["payer_PPO"] == 0.0

    def test_no_encoding_info_is_passthrough(self) -> None:
        X = pd.DataFrame({"a": [1.0]})
        assert list(_apply_categorical_onehot(X, None).columns) == ["a"]


class TestRawFeatureCols:
    """Map the model's one-hot-EXPANDED feature_cols back to the ORIGINAL
    pre-encode column names, so a raw frame (eligible_df, which carries the
    string categoricals, NOT the expanded indicators) can be re-selected and
    re-encoded for SHAP. Regression guard for the Step-6 SHAP wiring."""

    def test_maps_onehot_expanded_back_to_raw(self) -> None:
        feature_cols = ["num", "payer_HMO", "payer_PPO", "payer_EPO"]
        info = {
            "encoder": None,
            "columns": ["payer"],
            "onehot_columns": ["payer_HMO", "payer_PPO", "payer_EPO"],
            "method": "onehot",
        }
        # numeric kept; the 3 indicators collapse back to the single raw "payer"
        assert _raw_feature_cols(feature_cols, info) == ["num", "payer"]

    def test_no_cat_enc_returns_feature_cols_unchanged(self) -> None:
        assert _raw_feature_cols(["a", "b"], None) == ["a", "b"]
        assert _raw_feature_cols(["a", "b"], {"columns": []}) == ["a", "b"]
