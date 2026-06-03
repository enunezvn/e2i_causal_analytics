"""
Unit tests for Data Splitter - Phase 1: Data Loading Foundation.

Tests:
- Random splitting
- Temporal splitting
- Stratified splitting
- Entity-level splitting
- Combined splitting
"""

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

    @pytest.mark.parametrize("n", [7, 11, 13, 17, 23, 101, 103])
    def test_no_tail_rows_dropped_when_holdout_zero(self, splitter, n):
        """Regression: with the default 0-holdout config, int-truncated
        cumulative cutoffs left rows in indices[test_end:] assigned to NO
        split (holdout stayed None), silently dropping tail rows. Every row
        must land in exactly one of train/val/test."""
        df = pd.DataFrame({"id": range(n)})
        result = splitter.random_split(df)  # default SplitConfig, holdout_ratio=0
        assert result.holdout is None
        total = len(result.train) + len(result.val) + len(result.test)
        assert total == n, (
            f"random_split dropped {n - total} tail row(s): "
            f"train={len(result.train)} val={len(result.val)} test={len(result.test)}"
        )
        # No row appears twice and the union equals the original id set.
        recovered = set(result.train["id"]) | set(result.val["id"]) | set(result.test["id"])
        assert recovered == set(range(n))

    @pytest.mark.parametrize("n", [11, 13, 23, 101])
    def test_holdout_behavior_unchanged_when_ratio_positive(self, splitter, n):
        """The holdout_ratio>0 path must keep partitioning train/val/test/holdout
        without absorbing the tail into test (every row still lands exactly once)."""
        df = pd.DataFrame({"id": range(n)})
        config = SplitConfig(
            train_ratio=0.5,
            val_ratio=0.2,
            test_ratio=0.2,
            holdout_ratio=0.1,
        )
        result = splitter.random_split(df, config)
        assert result.holdout is not None
        total = len(result.train) + len(result.val) + len(result.test) + len(result.holdout)
        assert total == n
        recovered = (
            set(result.train["id"])
            | set(result.val["id"])
            | set(result.test["id"])
            | set(result.holdout["id"])
        )
        assert recovered == set(range(n))


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

    @pytest.mark.parametrize("per_stratum", [7, 11, 13, 17, 23])
    def test_no_tail_rows_dropped_when_holdout_zero(self, splitter, per_stratum):
        """Regression: stratified_split partitioned each stratum by int-truncated
        cumulative cutoffs; with the default 0-holdout config, rows in
        indices[test_end:] of every stratum were dropped. With strata that are
        not clean multiples of the ratios, no row may vanish."""
        n_per = per_stratum
        df = pd.DataFrame(
            {
                "id": range(2 * n_per),
                "category": ["A"] * n_per + ["B"] * n_per,
            }
        )
        result = splitter.stratified_split(df, stratify_column="category")
        assert result.holdout is None
        total = len(result.train) + len(result.val) + len(result.test)
        assert total == len(df), (
            f"stratified_split dropped {len(df) - total} tail row(s): "
            f"train={len(result.train)} val={len(result.val)} test={len(result.test)}"
        )
        recovered = set(result.train["id"]) | set(result.val["id"]) | set(result.test["id"])
        assert recovered == set(range(2 * n_per))

    def test_warns_when_small_stratum_dumped_to_train(self, splitter, caplog):
        """A stratum with <3 samples is assigned entirely to train, so it is
        ABSENT from val/test/holdout — warn loudly (the split_enforcer
        class-presence guard then hard-blocks the resulting rare-event split)."""
        import logging

        df = pd.DataFrame(
            {
                "id": range(10),
                "category": ["A"] * 5 + ["B"] * 3 + ["C"] * 2,  # C has n=2 (<3)
            }
        )
        with caplog.at_level(logging.WARNING):
            splitter.stratified_split(df, stratify_column="category")

        assert any(
            "C" in r.getMessage() and "train" in r.getMessage().lower() for r in caplog.records
        ), [r.getMessage() for r in caplog.records]


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


class TestDefaultCombinedWhenEntityAndDatePresent:
    """Tests for Block 4 (Finding #7) — when both entity_column and
    date_column are available, the splitter's combined_split must be the
    default choice.

    These tests cover both the underlying ``combined_split`` contract and
    the auto-resolution logic in ``run_tier0_test.step_5_model_trainer``,
    which is the consumer responsible for picking the strategy.
    """

    def _multi_period_frame(self, n_entities: int = 60) -> pd.DataFrame:
        """Build a DataFrame with N unique entities spread across 90 days.

        Each entity gets a single row (matching the synthetic
        ``ml_patients`` schema). Dates are spread linearly so combined_split
        produces non-empty train/val/test buckets.
        """
        rng = np.random.default_rng(42)
        start = pd.Timestamp("2026-01-01")
        rows = []
        for i in range(n_entities):
            rows.append(
                {
                    "patient_journey_id": f"patient-{i:03d}",
                    "journey_start_date": start + pd.Timedelta(days=i),
                    "feature_a": rng.normal(0, 1),
                    "feature_b": rng.normal(0, 1),
                    "y": int(rng.random() < 0.3),
                }
            )
        return pd.DataFrame(rows)

    def test_combined_split_is_returned_when_entity_and_date_present(self):
        """combined_split must populate train/val/test when given entity +
        date columns and a span wide enough to allow real partitioning."""
        df = self._multi_period_frame(n_entities=60)
        splitter = DataSplitter(random_seed=42)
        result = splitter.combined_split(
            df,
            date_column="journey_start_date",
            entity_column="patient_journey_id",
            val_days=18,
            test_days=14,
        )
        assert isinstance(result, SplitResult)
        # All three primary splits populated
        assert len(result.train) > 0
        assert len(result.val) > 0
        assert len(result.test) > 0
        # Entity isolation across splits
        train_entities = set(result.train["patient_journey_id"])
        val_entities = set(result.val["patient_journey_id"])
        test_entities = set(result.test["patient_journey_id"])
        assert train_entities & val_entities == set()
        assert train_entities & test_entities == set()
        assert val_entities & test_entities == set()
        # Metadata contract carries the combined-split signature
        assert result.metadata["split_type"] == "combined_temporal_entity"
        assert result.metadata["entity_count"] == 60

    def test_step_5_defaults_to_combined_when_entity_and_date_columns_present(self):
        """``step_5_model_trainer`` auto-resolves to the combined split
        path when caller threads entity_ids + dates and split_mode is
        left at the default ``"auto"`` (per the Block 4 contract)."""
        # Importing the script module triggers heavy ML imports; gate
        # behind importlib like the rest of the tier0 script tests do
        # so this still runs in a thin environment.
        import importlib.util
        from pathlib import Path

        script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_tier0_test.py"
        spec = importlib.util.spec_from_file_location("run_tier0_test", script_path)
        if spec is None or spec.loader is None:
            pytest.skip("Could not build import spec for run_tier0_test")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - environment-specific
            pytest.skip(f"Could not import run_tier0_test: {exc}")

        # Validate the function signature so the contract drift we just
        # added is checked even if ML deps are unhappy in CI.
        import inspect

        sig = inspect.signature(module.step_5_model_trainer)
        params = sig.parameters
        for required in ("entity_ids", "dates", "split_mode", "pre_assigned_splits"):
            assert required in params, f"step_5_model_trainer must accept {required!r} for Block 4"
        assert params["split_mode"].default == "auto", (
            "split_mode default must remain 'auto' so combined wins when entity + date are present"
        )
        assert params["pre_assigned_splits"].default is None

    def test_combined_split_falls_back_safely_when_only_entity_present(self):
        """Sanity: when caller has no date column, the splitter still
        works on the entity-only path (used by the random fallback)."""
        df = self._multi_period_frame(n_entities=30)
        splitter = DataSplitter(random_seed=42)
        result = splitter.entity_split(df, entity_column="patient_journey_id")
        assert isinstance(result, SplitResult)
        train_entities = set(result.train["patient_journey_id"])
        test_entities = set(result.test["patient_journey_id"])
        assert train_entities & test_entities == set()
