"""
Unit tests for Data Splitter - Phase 1: Data Loading Foundation.

Tests:
- Random splitting
- Temporal splitting
- Stratified splitting
- Entity-level splitting
- Combined splitting
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.repositories.data_splitter import (
    DataSplitter,
    SplitConfig,
    SplitResult,
    get_data_splitter,
)


class TestSplitConfig:
    """Tests for SplitConfig."""

    def test_default_ratios_sum_to_one(self):
        """Test that default ratios sum to 1.0."""
        config = SplitConfig()
        total = config.train_ratio + config.val_ratio + config.test_ratio
        assert np.isclose(total, 1.0)

    def test_raises_error_if_ratios_dont_sum_to_one(self):
        """Test that invalid ratios raise ValueError."""
        with pytest.raises(ValueError):
            SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_accepts_holdout_ratio(self):
        """Test that holdout ratio can be specified."""
        config = SplitConfig(
            train_ratio=0.5,
            val_ratio=0.2,
            test_ratio=0.2,
            holdout_ratio=0.1,
        )
        assert config.holdout_ratio == 0.1


class TestSplitResult:
    """Tests for SplitResult."""

    @pytest.fixture
    def sample_result(self):
        """Create sample split result."""
        return SplitResult(
            train=pd.DataFrame({"a": [1, 2, 3]}),
            val=pd.DataFrame({"a": [4]}),
            test=pd.DataFrame({"a": [5, 6]}),
        )

    def test_to_dict(self, sample_result):
        """Test to_dict method."""
        result = sample_result.to_dict()
        assert "train" in result
        assert "val" in result
        assert "test" in result

    def test_summary(self, sample_result):
        """Test summary method."""
        summary = sample_result.summary()
        assert summary["train_size"] == 3
        assert summary["val_size"] == 1
        assert summary["test_size"] == 2


class TestDataSplitter:
    """Tests for DataSplitter."""

    @pytest.fixture
    def splitter(self):
        """Create splitter instance."""
        return DataSplitter(random_seed=42)

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame(
            {
                "id": range(n),
                "value": np.random.randn(n),
                "category": np.random.choice(["A", "B", "C"], n),
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "entity_id": np.random.choice(["e1", "e2", "e3", "e4", "e5"], n),
            }
        )


class TestRandomSplit(TestDataSplitter):
    """Tests for random_split method."""

    def test_returns_split_result(self, splitter, sample_df):
        """Test that random_split returns SplitResult."""
        result = splitter.random_split(sample_df)
        assert isinstance(result, SplitResult)

    def test_preserves_total_size(self, splitter, sample_df):
        """Test that total size is preserved after split."""
        result = splitter.random_split(sample_df)
        total = len(result.train) + len(result.val) + len(result.test)
        assert total == len(sample_df)

    def test_respects_ratios(self, splitter, sample_df):
        """Test that splits respect configured ratios."""
        config = SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        result = splitter.random_split(sample_df, config)

        # Allow some tolerance due to rounding
        assert abs(len(result.train) / len(sample_df) - 0.6) < 0.02
        assert abs(len(result.val) / len(sample_df) - 0.2) < 0.02

    def test_reproducible_with_seed(self, sample_df):
        """Test that splits are reproducible with same seed."""
        splitter1 = DataSplitter(random_seed=42)
        splitter2 = DataSplitter(random_seed=42)

        result1 = splitter1.random_split(sample_df)
        result2 = splitter2.random_split(sample_df)

        pd.testing.assert_frame_equal(result1.train, result2.train)

    def test_creates_holdout_when_specified(self, splitter, sample_df):
        """Test that holdout set is created when ratio > 0."""
        config = SplitConfig(
            train_ratio=0.5,
            val_ratio=0.2,
            test_ratio=0.2,
            holdout_ratio=0.1,
        )
        result = splitter.random_split(sample_df, config)
        assert result.holdout is not None
        assert len(result.holdout) > 0


class TestTemporalSplit(TestDataSplitter):
    """Tests for temporal_split method."""

    def test_returns_split_result(self, splitter, sample_df):
        """Test that temporal_split returns SplitResult."""
        result = splitter.temporal_split(sample_df, date_column="date")
        assert isinstance(result, SplitResult)

    def test_train_before_val(self, splitter, sample_df):
        """Test that training data is before validation data."""
        result = splitter.temporal_split(
            sample_df,
            date_column="date",
            val_days=30,
            test_days=30,
        )

        if len(result.train) > 0 and len(result.val) > 0:
            train_max = result.train["date"].max()
            val_min = result.val["date"].min()
            assert train_max < val_min

    def test_val_before_test(self, splitter, sample_df):
        """Test that validation data is before test data."""
        result = splitter.temporal_split(
            sample_df,
            date_column="date",
            val_days=30,
            test_days=30,
        )

        if len(result.val) > 0 and len(result.test) > 0:
            val_max = result.val["date"].max()
            test_min = result.test["date"].min()
            assert val_max < test_min

    def test_uses_specified_split_date(self, splitter, sample_df):
        """Test that specified split_date is used."""
        split_date = "2024-06-01"
        result = splitter.temporal_split(
            sample_df,
            date_column="date",
            split_date=split_date,
            val_days=30,
            test_days=30,
        )

        assert result.metadata["split_date"].startswith("2024-06-01")


class TestStratifiedSplit(TestDataSplitter):
    """Tests for stratified_split method."""

    def test_returns_split_result(self, splitter, sample_df):
        """Test that stratified_split returns SplitResult."""
        result = splitter.stratified_split(sample_df, stratify_column="category")
        assert isinstance(result, SplitResult)

    def test_maintains_class_distribution(self, splitter, sample_df):
        """Test that class distribution is maintained in splits."""
        result = splitter.stratified_split(sample_df, stratify_column="category")

        original_dist = sample_df["category"].value_counts(normalize=True)
        train_dist = result.train["category"].value_counts(normalize=True)

        # Check that distributions are similar (within 10%)
        for category in original_dist.index:
            if category in train_dist.index:
                assert abs(original_dist[category] - train_dist[category]) < 0.1

    def test_handles_small_strata(self, splitter):
        """Test handling of strata with few samples."""
        df = pd.DataFrame(
            {
                "id": range(10),
                "category": ["A"] * 5 + ["B"] * 3 + ["C"] * 2,
            }
        )
        result = splitter.stratified_split(df, stratify_column="category")
        assert isinstance(result, SplitResult)


class TestEntitySplit(TestDataSplitter):
    """Tests for entity_split method."""

    def test_returns_split_result(self, splitter, sample_df):
        """Test that entity_split returns SplitResult."""
        result = splitter.entity_split(sample_df, entity_column="entity_id")
        assert isinstance(result, SplitResult)

    def test_entities_not_split_across_sets(self, splitter, sample_df):
        """Test that entities don't appear in multiple splits."""
        result = splitter.entity_split(sample_df, entity_column="entity_id")

        train_entities = set(result.train["entity_id"].unique())
        val_entities = set(result.val["entity_id"].unique())
        test_entities = set(result.test["entity_id"].unique())

        # Check no overlap
        assert len(train_entities & val_entities) == 0
        assert len(train_entities & test_entities) == 0
        assert len(val_entities & test_entities) == 0

    def test_deterministic_assignment(self, sample_df):
        """Test that entity assignment is deterministic."""
        splitter1 = DataSplitter(random_seed=42)
        splitter2 = DataSplitter(random_seed=99)  # Different seed

        result1 = splitter1.entity_split(sample_df, entity_column="entity_id")
        result2 = splitter2.entity_split(sample_df, entity_column="entity_id")

        # Entity splits should be same (hash-based, not seed-based)
        train_entities1 = set(result1.train["entity_id"].unique())
        train_entities2 = set(result2.train["entity_id"].unique())
        assert train_entities1 == train_entities2


class TestCombinedSplit(TestDataSplitter):
    """Tests for combined_split method."""

    def test_returns_split_result(self, splitter, sample_df):
        """Test that combined_split returns SplitResult."""
        result = splitter.combined_split(
            sample_df,
            date_column="date",
            entity_column="entity_id",
        )
        assert isinstance(result, SplitResult)

    def test_respects_both_temporal_and_entity(self, splitter, sample_df):
        """Test that both temporal and entity constraints are respected."""
        result = splitter.combined_split(
            sample_df,
            date_column="date",
            entity_column="entity_id",
            val_days=100,
            test_days=100,
        )

        # Check entities don't cross splits
        train_entities = set(result.train["entity_id"].unique())
        test_entities = set(result.test["entity_id"].unique())
        assert len(train_entities & test_entities) == 0

        # Check metadata
        assert result.metadata["split_type"] == "combined_temporal_entity"


class TestGetDataSplitter:
    """Tests for get_data_splitter function."""

    def test_returns_splitter_instance(self):
        """Test that function returns DataSplitter."""
        splitter = get_data_splitter()
        assert isinstance(splitter, DataSplitter)

    def test_uses_specified_seed(self):
        """Test that specified seed is used."""
        splitter = get_data_splitter(random_seed=123)
        assert splitter.random_seed == 123
