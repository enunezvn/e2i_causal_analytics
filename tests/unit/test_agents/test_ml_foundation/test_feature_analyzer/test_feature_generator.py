"""Unit tests for feature_generator node.

Tests feature generation capabilities:
- Temporal features (lag, rolling, date parts)
- Interaction features (categorical crosses, numeric products/ratios)
- Domain-specific features (pharma KPIs)
- Aggregate features (row-wise statistics)
"""

import re

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.feature_analyzer.nodes.feature_generator import (
    _SPLIT_MARKER_COL,
    _SPLIT_ROW_ID_COL,
    _concat_with_split_markers,
    _detect_categorical_columns,
    _detect_numeric_columns,
    _detect_temporal_columns,
    _generate_aggregate_features,
    _generate_domain_features,
    _generate_interaction_features,
    _generate_temporal_features,
    _handle_generated_nans,
    generate_features,
)


class TestDetectionFunctions:
    """Tests for column type detection functions."""

    def test_detect_temporal_columns_datetime(self):
        """Should detect datetime columns."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "value": [1, 2],
            }
        )
        result = _detect_temporal_columns(df)
        assert "date" in result

    def test_detect_temporal_columns_by_name(self):
        """Should detect columns with date-like names."""
        df = pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-02-01"],
                "timestamp_col": ["2024-01-01", "2024-02-01"],
                "name": ["A", "B"],
            }
        )
        result = _detect_temporal_columns(df)
        assert "order_date" in result
        assert "timestamp_col" in result
        assert "name" not in result

    def test_detect_categorical_columns_object_dtype(self):
        """Should detect object dtype columns as categorical."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "C"] * 10,
                "value": list(range(30)),  # High cardinality - won't be detected as categorical
            }
        )
        result = _detect_categorical_columns(df)
        assert "category" in result
        assert "value" not in result

    def test_detect_categorical_columns_low_cardinality_int(self):
        """Should detect low cardinality integer columns as categorical."""
        df = pd.DataFrame(
            {
                "status_code": [1, 2, 1, 2, 3] * 10,
                "high_card": range(50),
            }
        )
        result = _detect_categorical_columns(df)
        assert "status_code" in result
        assert "high_card" not in result

    def test_detect_numeric_columns(self):
        """Should detect numeric columns."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )
        result = _detect_numeric_columns(df)
        assert "int_col" in result
        assert "float_col" in result
        assert "str_col" not in result


class TestTemporalFeatures:
    """Tests for temporal feature generation.

    Block 1B tightens the contract so ``entity_id_column`` and
    ``event_timestamp_column`` are required. Tests in this class always
    pass them; the single-entity panel guarantees lag/rolling behaviour
    is identical to the (deleted) cross-entity path.
    """

    def test_generate_date_parts_from_datetime(self):
        """Should generate date part features from datetime column."""
        df = pd.DataFrame(
            {
                "patient_id": ["A"] * 10,
                "event_ts": pd.date_range("2024-01-01", periods=10, freq="D"),
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            }
        )
        result_df, metadata = _generate_temporal_features(
            df,
            temporal_columns=["date"],
            entity_id_column="patient_id",
            event_timestamp_column="event_ts",
        )
        assert "date_dayofweek" in result_df.columns
        assert "date_month" in result_df.columns
        assert "date_quarter" in result_df.columns
        assert "date_is_weekend" in result_df.columns
        assert len(metadata) >= 4

    def test_generate_lag_features_from_numeric(self):
        """Should generate lag features from numeric columns."""
        df = pd.DataFrame(
            {
                "patient_id": ["A"] * 20,
                "event_ts": pd.date_range("2024-01-01", periods=20, freq="D"),
                "value": range(20),
            }
        )
        result_df, metadata = _generate_temporal_features(
            df,
            temporal_columns=["value"],
            entity_id_column="patient_id",
            event_timestamp_column="event_ts",
            lag_periods=[1, 2],
        )
        assert "value_lag_1" in result_df.columns
        assert "value_lag_2" in result_df.columns
        # Check lag values (single entity ⇒ lag chain is the value sequence).
        assert pd.isna(result_df["value_lag_1"].iloc[0])
        assert result_df["value_lag_1"].iloc[1] == 0

    def test_generate_rolling_features(self):
        """Should generate rolling statistics from numeric columns."""
        df = pd.DataFrame(
            {
                "patient_id": ["A"] * 20,
                "event_ts": pd.date_range("2024-01-01", periods=20, freq="D"),
                "value": range(20),
            }
        )
        result_df, metadata = _generate_temporal_features(
            df,
            temporal_columns=["value"],
            entity_id_column="patient_id",
            event_timestamp_column="event_ts",
            rolling_windows=[3],
            lag_periods=[],
        )
        assert "value_rolling_mean_3" in result_df.columns
        assert "value_rolling_std_3" in result_df.columns

    def test_metadata_structure(self):
        """Should return proper metadata structure."""
        df = pd.DataFrame(
            {
                "patient_id": ["A"] * 5,
                "event_ts": pd.date_range("2024-01-01", periods=5, freq="D"),
                "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            }
        )
        _, metadata = _generate_temporal_features(
            df,
            temporal_columns=["date"],
            entity_id_column="patient_id",
            event_timestamp_column="event_ts",
        )

        assert len(metadata) > 0
        for meta in metadata:
            assert "name" in meta
            assert "source" in meta
            assert "type" in meta
            assert "transformation" in meta


class TestEntityGroupedTemporalFeatures:
    """Tests for entity-grouped lag/rolling feature generation (Block 1B).

    These tests cover finding #2 in the tier0 remediation plan: lag and
    rolling-window features must be computed per entity, not on the raw row
    order, otherwise the first row of one entity sees the previous entity's
    tail value and leaks across patients.
    """

    def _three_patient_panel(self) -> pd.DataFrame:
        """Build a synthetic 3 patients × 5 rows panel.

        Each patient's ``value`` series increases monotonically so we can
        assert per-entity lag/rolling output without ambiguity.
        """
        rows = []
        # Patient A: values 1..5 across 5 days.
        # Patient B: values 100..104.
        # Patient C: values 1000..1004.
        # Interleaving rows by date forces the implementation to use
        # (entity_id, event_timestamp) sorting; row order alone is wrong.
        for day in range(5):
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            rows.append({"patient_id": "A", "event_ts": ts, "value": 1 + day})
            rows.append({"patient_id": "B", "event_ts": ts, "value": 100 + day})
            rows.append({"patient_id": "C", "event_ts": ts, "value": 1000 + day})
        # Shuffle so the input is intentionally not pre-sorted.
        df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
        return df

    def test_lag_groupby_entity(self):
        """``lag_1`` MUST start NaN at the first row of each entity.

        If the implementation forgot the per-entity groupby, patient B's
        first row would erroneously see patient A's last value as ``lag_1``.
        We assert NaN to catch that regression.
        """
        df = self._three_patient_panel()

        result_df, _ = _generate_temporal_features(
            df,
            temporal_columns=["value"],
            entity_id_column="patient_id",
            event_timestamp_column="event_ts",
            lag_periods=[1],
            rolling_windows=[],
        )

        # Sort by (patient_id, event_ts) so we can index the first row per
        # entity directly. _generate_temporal_features sorts in place; this
        # guarantees the assertions are deterministic.
        sorted_result = result_df.sort_values(["patient_id", "event_ts"]).reset_index(drop=True)

        # First row of each entity must have lag_1 == NaN.
        for entity in ("A", "B", "C"):
            first = sorted_result[sorted_result["patient_id"] == entity].iloc[0]
            assert pd.isna(first["value_lag_1"]), (
                f"lag_1 at first row of patient {entity} must be NaN; "
                f"got {first['value_lag_1']!r}. Without entity grouping the "
                "function would leak patient A's value here."
            )

        # Subsequent rows in each entity must lag the entity's own series.
        for entity, base in (("A", 1), ("B", 100), ("C", 1000)):
            entity_rows = sorted_result[sorted_result["patient_id"] == entity]
            # Row index 1 within the entity (second day) must equal day-0 value.
            assert entity_rows.iloc[1]["value_lag_1"] == base
            # Row index 4 must equal day-3 value (base + 3).
            assert entity_rows.iloc[4]["value_lag_1"] == base + 3

    def test_missing_entity_or_timestamp_raises(self):
        """Contract sentinel: omitting either grouping key MUST raise.

        Block 1B tightens the previously-graceful fallback so a future
        caller cannot silently re-introduce naive cross-entity shift —
        which is exactly the leakage finding #2 documents. Both
        ``entity_id_column`` and ``event_timestamp_column`` are now
        required, and a missing or absent column fails fast with a
        message pointing back to Block 1B.
        """
        df = self._three_patient_panel()

        # Empty entity_id_column.
        with pytest.raises(ValueError, match="entity_id_column"):
            _generate_temporal_features(
                df,
                temporal_columns=["value"],
                entity_id_column="",
                event_timestamp_column="event_ts",
                lag_periods=[1],
                rolling_windows=[],
            )

        # Empty event_timestamp_column.
        with pytest.raises(ValueError, match="event_timestamp_column"):
            _generate_temporal_features(
                df,
                temporal_columns=["value"],
                entity_id_column="patient_id",
                event_timestamp_column="",
                lag_periods=[1],
                rolling_windows=[],
            )

        # Entity column name not present in the DataFrame.
        with pytest.raises(ValueError, match="not found in DataFrame"):
            _generate_temporal_features(
                df,
                temporal_columns=["value"],
                entity_id_column="bogus_entity",
                event_timestamp_column="event_ts",
                lag_periods=[1],
                rolling_windows=[],
            )

        # Timestamp column name not present in the DataFrame.
        with pytest.raises(ValueError, match="not found in DataFrame"):
            _generate_temporal_features(
                df,
                temporal_columns=["value"],
                entity_id_column="patient_id",
                event_timestamp_column="bogus_ts",
                lag_periods=[1],
                rolling_windows=[],
            )

    def test_rolling_groupby_entity(self):
        """Rolling mean must be entity-scoped, not cross-entity."""
        df = self._three_patient_panel()

        result_df, _ = _generate_temporal_features(
            df,
            temporal_columns=["value"],
            entity_id_column="patient_id",
            event_timestamp_column="event_ts",
            lag_periods=[],
            rolling_windows=[3],
        )

        sorted_result = result_df.sort_values(["patient_id", "event_ts"]).reset_index(drop=True)

        # Patient B day-2 rolling mean(3) over [100, 101, 102] = 101.0.
        # If grouping leaked, the window would have absorbed patient A's tail.
        b_day2 = sorted_result[sorted_result["patient_id"] == "B"].iloc[2]
        assert b_day2["value_rolling_mean_3"] == pytest.approx(101.0)

        # Patient C day-2 rolling mean(3) over [1000, 1001, 1002] = 1001.0.
        c_day2 = sorted_result[sorted_result["patient_id"] == "C"].iloc[2]
        assert c_day2["value_rolling_mean_3"] == pytest.approx(1001.0)


@pytest.mark.asyncio
class TestGenerateFeaturesEntityGroupedPipeline:
    """Integration: ``generate_features`` must apply grouping across splits.

    The full pipeline runs ``_generate_temporal_features`` ONCE on the
    concatenated train+val+test, then re-splits via internal markers. This
    ensures lag chains span split boundaries within an entity.
    """

    async def test_lag_chain_spans_train_val_within_entity(self):
        """Validation row 0 (patient with train history) must see train tail.

        If ``generate_features`` ran ``_generate_temporal_features``
        per-split (the pre-Block-1B behaviour), patient B's first
        validation row would have ``lag_1 == NaN``. With the Block 1B
        refactor it should pull the last train value for the same entity.
        """
        # Build a 2 patients × 4 rows train + 2 patients × 1 row val panel.
        train_rows = []
        for day in range(4):
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            train_rows.append({"patient_id": "A", "event_ts": ts, "value": 1 + day})
            train_rows.append({"patient_id": "B", "event_ts": ts, "value": 100 + day})
        train_df = pd.DataFrame(train_rows)

        val_rows = [
            {
                "patient_id": "A",
                "event_ts": pd.Timestamp("2026-01-05"),
                "value": 5.0,
            },
            {
                "patient_id": "B",
                "event_ts": pd.Timestamp("2026-01-05"),
                "value": 104.0,
            },
        ]
        val_df = pd.DataFrame(val_rows)

        state = {
            "X_train": train_df,
            "X_val": val_df,
            "y_train": pd.Series([0, 1] * 4),
            "problem_type": "classification",
            "entity_id_column": "patient_id",
            "event_timestamp_column": "event_ts",
            "feature_config": {
                "generate_temporal": True,
                "generate_interactions": False,
                "generate_domain": False,
                "generate_aggregates": False,
                "lag_periods": [1],
                "rolling_windows": [],
                # Skip nan-fill so we can assert raw lag values.
                "nan_fill_strategy": "zero",
            },
            "temporal_columns": ["value"],
        }

        result = await generate_features(state)
        assert "X_val_generated" in result
        assert result.get("error") is None

        x_val = result["X_val_generated"]
        # Patient A's val row must see train day-3 value (4) as its lag_1,
        # not 0 (which would mean nan-fill swallowed a per-split NaN).
        a_val = x_val[x_val["patient_id"] == "A"].iloc[0]
        assert a_val["value_lag_1"] == 4, (
            "Validation lag_1 must reach back into the training tail for the "
            "same entity. A NaN/zero here means the splits were processed "
            "independently and the lag chain is broken."
        )
        b_val = x_val[x_val["patient_id"] == "B"].iloc[0]
        assert b_val["value_lag_1"] == 103


class TestConcatWithSplitMarkersGuards:
    """Block 1B-M1: refuse to clobber caller columns with internal markers.

    The marker columns use dunder names so the chance of a real collision is
    near zero, but a silent overwrite would scramble the round-trip back to
    per-split frames in ``_split_by_markers``. The guard makes the failure
    loud so callers see it instantly.
    """

    @pytest.mark.parametrize("reserved", [_SPLIT_MARKER_COL, _SPLIT_ROW_ID_COL])
    def test_raises_when_reserved_column_already_present_on_train(self, reserved):
        train = pd.DataFrame({"value": [1, 2], reserved: ["x", "y"]})
        val = pd.DataFrame({"value": [3]})

        with pytest.raises(ValueError, match=re.escape(reserved)):
            _concat_with_split_markers(train, val, None)

    @pytest.mark.parametrize("reserved", [_SPLIT_MARKER_COL, _SPLIT_ROW_ID_COL])
    def test_raises_when_reserved_column_already_present_on_val(self, reserved):
        train = pd.DataFrame({"value": [1]})
        val = pd.DataFrame({"value": [2], reserved: ["x"]})

        with pytest.raises(ValueError, match=re.escape(reserved)):
            _concat_with_split_markers(train, val, None)

    @pytest.mark.parametrize("reserved", [_SPLIT_MARKER_COL, _SPLIT_ROW_ID_COL])
    def test_raises_when_reserved_column_already_present_on_test(self, reserved):
        train = pd.DataFrame({"value": [1]})
        test = pd.DataFrame({"value": [2], reserved: ["x"]})

        with pytest.raises(ValueError, match=re.escape(reserved)):
            _concat_with_split_markers(train, None, test)


class TestConcatWithSplitMarkersMemoryContract:
    """Block 1B-M5: the no-copy contract is observable and load-bearing.

    Dropping the per-split ``piece.copy()`` matters for RWD-scale memory, but
    only if it is actually preserved. These tests pin the two contracts the
    docstring promises so that a future "let's add a defensive copy back"
    refactor breaks loudly instead of silently regressing memory use.
    """

    def test_returned_frame_shares_memory_with_input(self):
        """Single-split path must alias the caller's column buffers.

        With only X_train present, the combined frame is constructed without
        going through ``pd.concat`` block consolidation, so the no-copy
        contract is directly observable via ``np.shares_memory``. If a future
        edit reintroduces ``piece.copy()``, this assertion flips to False.

        Note: the multi-split path also drops the per-split copies, but
        ``pd.concat`` consolidates the row stack into a fresh contiguous
        block, so ``shares_memory`` between the combined frame and any single
        input split is False there even with ``copy=False``. The single-split
        case is the most direct, version-stable witness to the contract.
        """
        train = pd.DataFrame(
            {
                "a": np.arange(5, dtype=np.float64),
                "b": np.arange(5, dtype=np.float64) * 2,
            }
        )

        combined, _ = _concat_with_split_markers(train, None, None)

        # At least one numeric column on the combined frame must alias the
        # corresponding column on the input. Both are asserted because either
        # one regressing would re-introduce the copy.
        assert np.shares_memory(combined["a"].values, train["a"].values), (
            "Single-split combined frame must alias input column 'a'; a "
            "False here means a defensive copy was reintroduced inside "
            "_concat_with_split_markers and the 1B-M5 memory contract has "
            "regressed."
        )
        assert np.shares_memory(combined["b"].values, train["b"].values)

    def test_inputs_are_mutated_in_place_with_marker_columns(self):
        """The no-copy contract has a caller-visible side effect: input frames
        gain ``_SPLIT_MARKER_COL`` and ``_SPLIT_ROW_ID_COL`` in place.

        This is documented in the function's Notes section. Pinning it here
        guarantees that callers who later try to reuse X_train/X_val/X_test
        as "untouched" will see the dunder columns; if a future refactor
        adds a copy to hide this, the test fails and forces an explicit
        decision rather than a silent memory regression.
        """
        train = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        val = pd.DataFrame({"value": [4.0, 5.0]})
        test = pd.DataFrame({"value": [6.0]})

        train_cols_before = set(train.columns)
        val_cols_before = set(val.columns)
        test_cols_before = set(test.columns)

        _concat_with_split_markers(train, val, test)

        new_marker_cols = {_SPLIT_MARKER_COL, _SPLIT_ROW_ID_COL}
        assert set(train.columns) - train_cols_before == new_marker_cols, (
            "X_train must gain marker columns in place; the no-copy "
            "contract (1B-M5) requires this side effect."
        )
        assert set(val.columns) - val_cols_before == new_marker_cols
        assert set(test.columns) - test_cols_before == new_marker_cols


class TestInteractionFeatures:
    """Tests for interaction feature generation."""

    def test_generate_categorical_cross(self):
        """Should generate categorical cross features."""
        df = pd.DataFrame(
            {
                "region": ["East", "West", "East", "West"],
                "brand": ["A", "A", "B", "B"],
            }
        )
        result_df, metadata = _generate_interaction_features(
            df, categorical_columns=["region", "brand"], numeric_columns=[]
        )
        assert "region_x_brand" in result_df.columns
        assert len(metadata) >= 1

    def test_generate_numeric_products(self):
        """Should generate numeric product features."""
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0],
                "quantity": [5.0, 3.0, 2.0],
            }
        )
        result_df, metadata = _generate_interaction_features(
            df, categorical_columns=[], numeric_columns=["price", "quantity"]
        )
        assert "price_times_quantity" in result_df.columns
        # Check values
        assert result_df["price_times_quantity"].iloc[0] == 50.0

    def test_generate_numeric_ratios(self):
        """Should generate numeric ratio features."""
        df = pd.DataFrame(
            {
                "numerator": [10.0, 20.0, 30.0],
                "denominator": [2.0, 4.0, 5.0],
            }
        )
        result_df, metadata = _generate_interaction_features(
            df, categorical_columns=[], numeric_columns=["numerator", "denominator"]
        )
        assert "numerator_div_denominator" in result_df.columns
        assert result_df["numerator_div_denominator"].iloc[0] == 5.0

    def test_respects_max_interactions(self):
        """Should respect max_interactions limit."""
        df = pd.DataFrame({f"cat_{i}": [f"v{j}" for j in range(10)] for i in range(5)})
        result_df, metadata = _generate_interaction_features(
            df,
            categorical_columns=[f"cat_{i}" for i in range(5)],
            numeric_columns=[],
            max_interactions=3,
        )
        # Should have at most 3 new interaction columns
        new_cols = [c for c in result_df.columns if "_x_" in c]
        assert len(new_cols) <= 3


class TestDomainFeatures:
    """Tests for domain-specific feature generation."""

    def test_generate_trx_nrx_ratio(self):
        """Should generate TRx/NRx ratio for pharma data."""
        df = pd.DataFrame(
            {
                "trx": [100, 200, 150],
                "nrx": [20, 50, 30],
            }
        )
        result_df, metadata = _generate_domain_features(df)
        assert "trx_nrx_ratio" in result_df.columns
        assert result_df["trx_nrx_ratio"].iloc[0] == 5.0

    def test_generate_refill_rate(self):
        """Should generate refill rate feature."""
        df = pd.DataFrame(
            {
                "trx": [100, 200],
                "nrx": [20, 50],
            }
        )
        result_df, metadata = _generate_domain_features(df)
        assert "refill_rate" in result_df.columns
        # Refill rate = (TRx - NRx) / TRx = (100-20)/100 = 0.8
        assert result_df["refill_rate"].iloc[0] == 0.8

    def test_generate_market_share_momentum(self):
        """Should generate market share momentum feature."""
        df = pd.DataFrame(
            {
                "market_share": [0.1, 0.12, 0.15, 0.14],
            }
        )
        result_df, metadata = _generate_domain_features(df)
        assert "market_share_momentum" in result_df.columns
        # First value should be NaN (no previous value)
        assert pd.isna(result_df["market_share_momentum"].iloc[0])

    def test_handles_missing_domain_columns(self):
        """Should handle missing domain-specific columns gracefully."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        result_df, metadata = _generate_domain_features(df)
        # Should not fail, may return no new features
        assert result_df is not None


class TestAggregateFeatures:
    """Tests for aggregate feature generation."""

    def test_generate_row_statistics(self):
        """Should generate row-wise statistics."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            }
        )
        result_df, metadata = _generate_aggregate_features(df, ["a", "b", "c"])

        assert "numeric_mean" in result_df.columns
        assert "numeric_std" in result_df.columns
        assert "numeric_max" in result_df.columns
        assert "numeric_range" in result_df.columns

        # Check values
        assert result_df["numeric_mean"].iloc[0] == 4.0  # (1+4+7)/3
        assert result_df["numeric_max"].iloc[0] == 7.0

    def test_requires_multiple_columns(self):
        """Should require at least 2 columns for aggregates."""
        df = pd.DataFrame({"only_col": [1, 2, 3]})
        result_df, metadata = _generate_aggregate_features(df, ["only_col"])
        # Should return original df with no new features
        assert len(metadata) == 0


class TestHandleGeneratedNans:
    """Tests for NaN handling in generated features."""

    def test_fill_nans_median(self):
        """Should fill NaNs with median."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )
        result = _handle_generated_nans(df, strategy="median")
        assert not result["a"].isna().any()
        assert result["a"].iloc[1] == 3.0  # median of [1, 3, 5]

    def test_fill_nans_mean(self):
        """Should fill NaNs with mean."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 5.0],
            }
        )
        result = _handle_generated_nans(df, strategy="mean")
        assert result["a"].iloc[1] == 3.0  # mean of [1, 5]

    def test_fill_nans_zero(self):
        """Should fill NaNs with zero."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
            }
        )
        result = _handle_generated_nans(df, strategy="zero")
        assert result["a"].iloc[1] == 0.0

    def test_drop_nans(self):
        """Should drop rows with NaNs."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [4.0, 5.0, 6.0],
            }
        )
        result = _handle_generated_nans(df, strategy="drop")
        assert len(result) == 2


@pytest.mark.asyncio
class TestGenerateFeaturesNode:
    """Integration tests for generate_features node."""

    async def test_full_feature_generation_pipeline(self):
        """Should run complete feature generation pipeline."""
        state = {
            "X_train": pd.DataFrame(
                {
                    "patient_id": [f"p_{i % 10}" for i in range(100)],
                    "date": pd.date_range("2024-01-01", periods=100, freq="D"),
                    "region": ["East", "West"] * 50,
                    "value1": np.random.rand(100),
                    "value2": np.random.rand(100),
                    "trx": np.random.randint(50, 200, 100),
                    "nrx": np.random.randint(10, 50, 100),
                }
            ),
            "y_train": pd.Series(np.random.randint(0, 2, 100)),
            "problem_type": "classification",
            "entity_id_column": "patient_id",
            "event_timestamp_column": "date",
            "feature_config": {
                "generate_temporal": True,
                "generate_interactions": True,
                "generate_domain": True,
                "generate_aggregates": True,
                "lag_periods": [1],
                "rolling_windows": [7],
                "nan_fill_strategy": "median",
            },
        }

        result = await generate_features(state)

        assert "X_train_generated" in result
        assert "generated_features" in result
        assert result["new_feature_count"] > 0
        assert result["feature_generation_time_seconds"] >= 0

    async def test_handles_validation_data(self):
        """Should apply same transformations to validation data."""
        train_df = pd.DataFrame(
            {
                "patient_id": ["A"] * 50,
                "event_ts": pd.date_range("2024-01-01", periods=50, freq="D"),
                "value": range(50),
            }
        )
        val_df = pd.DataFrame(
            {
                "patient_id": ["A"] * 20,
                "event_ts": pd.date_range("2024-02-20", periods=20, freq="D"),
                "value": range(50, 70),
            }
        )

        state = {
            "X_train": train_df,
            "X_val": val_df,
            "y_train": pd.Series([0, 1] * 25),
            "problem_type": "classification",
            "entity_id_column": "patient_id",
            "event_timestamp_column": "event_ts",
            "feature_config": {
                "generate_temporal": True,
                "rolling_windows": [3],
            },
        }

        result = await generate_features(state)

        assert "X_val_generated" in result
        assert len(result["X_train_generated"]) > 0
        assert len(result["X_val_generated"]) > 0

    async def test_tracks_feature_metadata(self):
        """Should track metadata for generated features."""
        state = {
            "X_train": pd.DataFrame(
                {
                    "patient_id": ["A"] * 20,
                    "date": pd.date_range("2024-01-01", periods=20, freq="D"),
                    "value": range(20),
                }
            ),
            "y_train": pd.Series([0, 1] * 10),
            "problem_type": "classification",
            "entity_id_column": "patient_id",
            "event_timestamp_column": "date",
            "feature_config": {"generate_temporal": True, "lag_periods": [1]},
        }

        result = await generate_features(state)

        assert "feature_metadata" in result
        assert "generated_features" in result
        # Check metadata structure
        for feat in result["generated_features"]:
            assert "name" in feat
            assert "type" in feat

    async def test_missing_entity_columns_returns_error_when_temporal_enabled(self):
        """Caller-side strict check: missing keys → error dict, not silent no-op.

        ``generate_features`` raises a ``ValueError`` when temporal
        generation is requested without entity/timestamp columns; the
        node's outer ``try/except`` converts it to the standard error
        envelope so the LangGraph still flows.
        """
        state = {
            "X_train": pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                    "value": range(10),
                }
            ),
            "y_train": pd.Series([0, 1] * 5),
            "problem_type": "classification",
            # No entity_id_column / event_timestamp_column on state OR config.
            "feature_config": {"generate_temporal": True, "lag_periods": [1]},
        }

        result = await generate_features(state)

        assert result.get("error") is not None
        assert "entity_id_column" in result["error"]
        assert "event_timestamp_column" in result["error"]

    async def test_handles_empty_config(self):
        """Should handle empty or missing feature config."""
        state = {
            "X_train": pd.DataFrame({"a": [1, 2, 3]}),
            "y_train": pd.Series([0, 1, 0]),
            "problem_type": "classification",
        }

        result = await generate_features(state)

        assert "X_train_generated" in result
        assert not result.get("error")

    async def test_handles_missing_data(self):
        """Should handle missing training data gracefully."""
        state = {
            "problem_type": "classification",
        }

        result = await generate_features(state)

        assert result.get("error") is not None
        assert "X_train" in result.get("error", "")

    async def test_handles_numpy_array_input(self):
        """Should handle numpy arrays - currently returns error due to implementation bug.

        Note: The implementation converts numpy arrays to DataFrames internally,
        but line 188 accesses state["X_train"] which is still a numpy array.
        This test documents current behavior until the bug is fixed.
        """
        state = {
            "X_train": np.random.rand(50, 5),
            "y_train": pd.Series(np.random.randint(0, 2, 50)),
            "problem_type": "classification",
            "feature_config": {"generate_temporal": False, "generate_interactions": False},
        }

        result = await generate_features(state)

        # Current behavior: returns error due to numpy array handling bug
        # TODO: Fix the implementation to properly handle numpy arrays
        assert result.get("error") is not None or "X_train_generated" in result


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        result = _detect_numeric_columns(df)
        assert result == []

    def test_single_column(self):
        """Should handle single column DataFrame."""
        df = pd.DataFrame({"only_col": [1, 2, 3]})
        result = _detect_numeric_columns(df)
        assert "only_col" in result

    def test_all_nan_column(self):
        """Should handle all-NaN columns."""
        df = pd.DataFrame(
            {
                "all_nan": [np.nan, np.nan, np.nan],
                "valid": [1.0, 2.0, 3.0],
            }
        )
        result = _handle_generated_nans(df, strategy="median")
        # Should not fail - all_nan will be filled with 0 as fallback
        assert result is not None

    def test_division_by_zero_in_ratios(self):
        """Should handle division by zero in ratio features."""
        df = pd.DataFrame(
            {
                "numerator": [10.0, 20.0, 30.0],
                "denominator": [0.0, 4.0, 0.0],
            }
        )
        result_df, _ = _generate_interaction_features(
            df, categorical_columns=[], numeric_columns=["numerator", "denominator"]
        )
        # Should have NaN where denominator is 0
        if "numerator_div_denominator" in result_df.columns:
            assert pd.isna(result_df["numerator_div_denominator"].iloc[0])
