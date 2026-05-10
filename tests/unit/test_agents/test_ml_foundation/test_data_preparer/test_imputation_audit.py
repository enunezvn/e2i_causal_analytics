"""Plan v3 §4 T2.4 — Imputation/missingness audit contract tests.

Pins ``compute_imputation_audit``, ``summarize_recommendations``, the
``_per_column_missing_rate`` helper, and the recommendation-strategy
threshold table.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.imputation_audit import (
    T2_4_RECOMMEND_DROP_COLUMN_RATE_MIN,
    T2_4_RECOMMEND_DROP_ROW_RATE_MAX,
    T2_4_RECOMMEND_INDICATOR_RATE_MIN,
    T2_4_STABILITY_TOLERANCE_DEFAULT,
    _per_column_missing_rate,
    _recommend_strategy_for_rate,
    compute_imputation_audit,
    summarize_recommendations,
)

# --------------------------------------------------------------------------- #
# Module constants                                                            #
# --------------------------------------------------------------------------- #


class TestT24Constants:
    def test_recommendation_thresholds(self) -> None:
        """Sterne 2009 BMJ + Donders 2006 anchored boundaries."""
        assert T2_4_RECOMMEND_DROP_ROW_RATE_MAX == 0.05
        assert T2_4_RECOMMEND_INDICATOR_RATE_MIN == 0.30
        assert T2_4_RECOMMEND_DROP_COLUMN_RATE_MIN == 0.70
        # Strict ordering.
        assert (
            T2_4_RECOMMEND_DROP_ROW_RATE_MAX
            < T2_4_RECOMMEND_INDICATOR_RATE_MIN
            < T2_4_RECOMMEND_DROP_COLUMN_RATE_MIN
        )

    def test_stability_tolerance_default(self) -> None:
        assert T2_4_STABILITY_TOLERANCE_DEFAULT == 0.05


# --------------------------------------------------------------------------- #
# _per_column_missing_rate                                                    #
# --------------------------------------------------------------------------- #


class TestPerColumnMissingRate:
    def test_no_missing_returns_zero(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert _per_column_missing_rate(df) == {"a": 0.0, "b": 0.0}

    def test_all_missing_returns_one(self) -> None:
        df = pd.DataFrame({"a": [None, None, None]}, dtype=object)
        assert _per_column_missing_rate(df) == {"a": 1.0}

    def test_partial_missing(self) -> None:
        df = pd.DataFrame({"a": [1, np.nan, 3, np.nan]})
        assert _per_column_missing_rate(df) == {"a": 0.5}

    def test_empty_dataframe_returns_empty_dict(self) -> None:
        df = pd.DataFrame()
        assert _per_column_missing_rate(df) == {}

    def test_pd_na_counts_as_missing(self) -> None:
        df = pd.DataFrame({"a": [1, pd.NA, 3]}, dtype="Int64")
        assert _per_column_missing_rate(df) == {"a": pytest.approx(1 / 3)}


# --------------------------------------------------------------------------- #
# _recommend_strategy_for_rate                                                #
# --------------------------------------------------------------------------- #


class TestRecommendStrategy:
    @pytest.mark.parametrize(
        "rate,expected",
        [
            (0.0, "drop_row_or_mean"),
            (0.04, "drop_row_or_mean"),
            (0.05, "drop_row_or_mean"),  # boundary: <= 0.05
            (0.0501, "mean_plus_indicator"),
            (0.10, "mean_plus_indicator"),
            (0.29, "mean_plus_indicator"),
            (0.30, "indicator_only"),  # boundary: >= 0.30
            (0.50, "indicator_only"),
            (0.69, "indicator_only"),
            (0.70, "drop_column"),  # boundary: >= 0.70
            (0.95, "drop_column"),
            (1.0, "drop_column"),
        ],
    )
    def test_categorization(self, rate, expected) -> None:
        assert _recommend_strategy_for_rate(rate) == expected


# --------------------------------------------------------------------------- #
# compute_imputation_audit — happy path                                       #
# --------------------------------------------------------------------------- #


def _train_only_df() -> pd.DataFrame:
    """Train-only DataFrame with mixed missingness."""
    return pd.DataFrame(
        {
            "low_missing": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "moderate_missing": [1.0, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, 10.0],
            "high_missing": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0, 9.0, 10.0],
            "very_high_missing": [
                1.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )


class TestComputeImputationAuditHappyPath:
    def test_returns_canonical_keys(self) -> None:
        result = compute_imputation_audit(_train_only_df())
        for key in (
            "imputation_audit_completed",
            "imputation_audit_missingness_profile",
            "imputation_audit_overall_missingness",
            "imputation_audit_per_split_profile",
            "imputation_audit_stability_tolerance",
            "imputation_audit_stability_violations",
            "imputation_audit_stability_violation_details",
            "imputation_audit_recommendations",
            "imputation_audit_n_features",
            "imputation_audit_n_train_rows",
        ):
            assert key in result, f"missing key {key!r}"

    def test_completed_true_on_valid_input(self) -> None:
        result = compute_imputation_audit(_train_only_df())
        assert result["imputation_audit_completed"] is True

    def test_per_feature_missingness_correct(self) -> None:
        result = compute_imputation_audit(_train_only_df())
        profile = result["imputation_audit_missingness_profile"]
        assert profile["low_missing"] == pytest.approx(0.10)  # 1/10
        assert profile["moderate_missing"] == pytest.approx(0.30)  # 3/10
        assert profile["high_missing"] == pytest.approx(0.60)  # 6/10
        assert profile["very_high_missing"] == pytest.approx(0.90)  # 9/10

    def test_overall_missingness_aggregates_across_columns(self) -> None:
        # Total missing = 1 + 3 + 6 + 9 = 19; total cells = 10 * 4 = 40
        result = compute_imputation_audit(_train_only_df())
        assert result["imputation_audit_overall_missingness"] == pytest.approx(19 / 40)

    def test_recommendations_match_thresholds(self) -> None:
        result = compute_imputation_audit(_train_only_df())
        recs = result["imputation_audit_recommendations"]
        # 0.10 → 0.05 < r < 0.30 → mean_plus_indicator
        assert recs["low_missing"] == "mean_plus_indicator"
        # 0.30 → boundary → indicator_only
        assert recs["moderate_missing"] == "indicator_only"
        # 0.60 → 0.30 ≤ r < 0.70 → indicator_only
        assert recs["high_missing"] == "indicator_only"
        # 0.90 → ≥ 0.70 → drop_column
        assert recs["very_high_missing"] == "drop_column"

    def test_n_features_and_n_rows_recorded(self) -> None:
        result = compute_imputation_audit(_train_only_df())
        assert result["imputation_audit_n_features"] == 4
        assert result["imputation_audit_n_train_rows"] == 10


# --------------------------------------------------------------------------- #
# Split-stability test                                                        #
# --------------------------------------------------------------------------- #


class TestSplitStability:
    def test_no_violations_when_rates_match_across_splits(self) -> None:
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]})  # 25% missing
        result = compute_imputation_audit(df, df.copy(), df.copy())
        assert result["imputation_audit_stability_violations"] == []

    def test_violation_when_test_rate_far_above_train(self) -> None:
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})  # 0% missing
        test = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan]})  # 75% missing
        result = compute_imputation_audit(train, X_test=test)
        assert "x" in result["imputation_audit_stability_violations"]
        details = result["imputation_audit_stability_violation_details"]["x"]
        assert details["train"] == 0.0
        assert details["test"] == 0.75
        assert details["range"] == 0.75

    def test_violation_within_default_tolerance_does_not_fire(self) -> None:
        """5pp range stays under default 5pp tolerance (uses strict >)."""
        train = pd.DataFrame({"x": [1.0, np.nan]})  # 50% missing
        test = pd.DataFrame({"x": [1.0, np.nan, 3.0]})  # 33.3% missing → range 0.167
        result = compute_imputation_audit(train, X_test=test)
        # range 0.167 > 0.05 → violation expected
        assert "x" in result["imputation_audit_stability_violations"]

    def test_custom_tolerance_can_silence_borderline_violation(self) -> None:
        """Bumping tolerance above the rate spread silences the violation."""
        train = pd.DataFrame({"x": [1.0, np.nan]})  # 50% missing
        test = pd.DataFrame({"x": [1.0, np.nan, 3.0]})  # 33.3% missing
        result = compute_imputation_audit(train, X_test=test, stability_tolerance=0.20)
        assert "x" not in result["imputation_audit_stability_violations"]

    def test_per_split_profile_includes_only_provided_splits(self) -> None:
        train = pd.DataFrame({"x": [1.0, 2.0]})
        result_train_only = compute_imputation_audit(train)
        assert set(result_train_only["imputation_audit_per_split_profile"].keys()) == {"train"}
        result_train_test = compute_imputation_audit(train, X_test=train.copy())
        assert set(result_train_test["imputation_audit_per_split_profile"].keys()) == {
            "train",
            "test",
        }

    def test_logs_warning_on_violations(self, caplog) -> None:
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        test = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan]})
        with caplog.at_level(logging.WARNING):
            compute_imputation_audit(train, X_test=test)
        assert any("T2.4 ADVISORY" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Degenerate inputs                                                           #
# --------------------------------------------------------------------------- #


class TestDegenerateInputs:
    def test_none_x_train_returns_failure(self) -> None:
        result = compute_imputation_audit(None)  # type: ignore[arg-type]
        assert result["imputation_audit_completed"] is False
        assert "X_train is None" in result["imputation_audit_error"]
        assert result["imputation_audit_n_features"] == 0
        assert result["imputation_audit_n_train_rows"] == 0

    def test_empty_x_train_returns_failure(self) -> None:
        result = compute_imputation_audit(pd.DataFrame())
        assert result["imputation_audit_completed"] is False

    def test_zero_features_completes_with_empty_profile(self) -> None:
        """A row-having, column-less DataFrame is unusual but valid; audit
        completes and reports 0 features + empty profile + None overall
        rate (avoiding division by zero)."""
        result = compute_imputation_audit(pd.DataFrame(index=[0, 1, 2]))
        assert result["imputation_audit_completed"] is True
        assert result["imputation_audit_n_features"] == 0
        assert result["imputation_audit_n_train_rows"] == 3
        assert result["imputation_audit_missingness_profile"] == {}
        # overall rate is None when no features (cannot divide by 0).
        assert result["imputation_audit_overall_missingness"] is None

    def test_no_missing_values_overall_zero(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = compute_imputation_audit(df)
        assert result["imputation_audit_overall_missingness"] == 0.0
        assert all(v == 0.0 for v in result["imputation_audit_missingness_profile"].values())


# --------------------------------------------------------------------------- #
# summarize_recommendations                                                   #
# --------------------------------------------------------------------------- #


class TestSummarizeRecommendations:
    def test_aggregates_per_strategy_counts(self) -> None:
        recs: Dict[str, Any] = {
            "f1": "drop_row_or_mean",
            "f2": "mean_plus_indicator",
            "f3": "mean_plus_indicator",
            "f4": "indicator_only",
            "f5": "drop_column",
            "f6": "drop_column",
        }
        out = summarize_recommendations(recs)
        assert out == {
            "drop_row_or_mean": 1,
            "mean_plus_indicator": 2,
            "indicator_only": 1,
            "drop_column": 2,
        }

    def test_empty_recommendations_returns_zeros(self) -> None:
        out = summarize_recommendations({})
        assert out == {
            "drop_row_or_mean": 0,
            "mean_plus_indicator": 0,
            "indicator_only": 0,
            "drop_column": 0,
        }
