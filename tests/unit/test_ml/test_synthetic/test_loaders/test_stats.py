"""
Unit tests for src/ml/synthetic/loaders/stats.py

Tests statistics utility functions for synthetic datasets.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.loaders.stats import (
    ColumnStats,
    DatasetStats,
    SplitStats,
    get_all_datasets_stats,
    get_column_stats,
    get_dataset_stats,
    get_split_stats,
    print_dataset_summary,
    validate_supabase_data,
)


@pytest.mark.unit
class TestColumnStats:
    """Test ColumnStats dataclass."""

    def test_column_stats_creation(self):
        """Test ColumnStats creation."""
        stats = ColumnStats(
            column_name="test_col",
            dtype="int64",
            null_count=5,
            null_percentage=10.0,
            unique_count=20,
        )

        assert stats.column_name == "test_col"
        assert stats.dtype == "int64"
        assert stats.null_count == 5
        assert stats.null_percentage == 10.0
        assert stats.unique_count == 20

    def test_column_stats_with_numeric_values(self):
        """Test ColumnStats with numeric statistics."""
        stats = ColumnStats(
            column_name="score",
            dtype="float64",
            null_count=0,
            null_percentage=0.0,
            unique_count=100,
            mean=0.75,
            std=0.15,
            min_value=0.0,
            max_value=1.0,
            percentiles={"25%": 0.6, "50%": 0.75, "75%": 0.9},
        )

        assert stats.mean == 0.75
        assert stats.std == 0.15
        assert stats.min_value == 0.0
        assert stats.max_value == 1.0
        assert stats.percentiles["50%"] == 0.75


@pytest.mark.unit
class TestSplitStats:
    """Test SplitStats dataclass."""

    def test_split_stats_creation(self):
        """Test SplitStats creation."""
        stats = SplitStats(
            split_name="train",
            record_count=1000,
            percentage=60.0,
        )

        assert stats.split_name == "train"
        assert stats.record_count == 1000
        assert stats.percentage == 60.0
        assert stats.date_range is None

    def test_split_stats_with_date_range(self):
        """Test SplitStats with date range."""
        stats = SplitStats(
            split_name="train",
            record_count=1000,
            percentage=60.0,
            date_range=("2022-01-01", "2023-06-30"),
        )

        assert stats.date_range == ("2022-01-01", "2023-06-30")


@pytest.mark.unit
class TestDatasetStats:
    """Test DatasetStats dataclass."""

    def test_dataset_stats_creation(self):
        """Test DatasetStats creation."""
        stats = DatasetStats(
            table_name="patient_journeys",
            total_records=1000,
            column_count=15,
        )

        assert stats.table_name == "patient_journeys"
        assert stats.total_records == 1000
        assert stats.column_count == 15
        assert stats.column_stats == {}
        assert stats.split_stats is None
        assert stats.brand_distribution is None

    def test_dataset_stats_to_dict(self):
        """Test DatasetStats to_dict conversion."""
        col_stats = ColumnStats(
            column_name="test_col",
            dtype="int64",
            null_count=5,
            null_percentage=10.0,
            unique_count=20,
            mean=50.0,
            std=10.0,
        )

        stats = DatasetStats(
            table_name="test_table",
            total_records=100,
            column_count=5,
            column_stats={"test_col": col_stats},
            memory_usage_mb=2.5,
        )

        result = stats.to_dict()

        assert result["table_name"] == "test_table"
        assert result["total_records"] == 100
        assert result["column_count"] == 5
        assert result["memory_usage_mb"] == 2.5
        assert "test_col" in result["column_stats"]

    def test_dataset_stats_summary(self):
        """Test DatasetStats summary generation."""
        split_stats = [
            SplitStats("train", 600, 60.0),
            SplitStats("validation", 200, 20.0),
            SplitStats("test", 150, 15.0),
            SplitStats("holdout", 50, 5.0),
        ]

        stats = DatasetStats(
            table_name="patient_journeys",
            total_records=1000,
            column_count=15,
            split_stats=split_stats,
            brand_distribution={"Remibrutinib": 350, "Fabhalta": 325, "Kisqali": 325},
            memory_usage_mb=5.25,
        )

        summary = stats.summary()

        assert "patient_journeys" in summary
        assert "1,000" in summary
        assert "15" in summary
        assert "train" in summary
        assert "Remibrutinib" in summary


@pytest.mark.unit
class TestGetColumnStats:
    """Test get_column_stats function."""

    def test_get_column_stats_numeric(self):
        """Test column stats for numeric column."""
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        stats = get_column_stats(df, "values")

        assert stats.column_name == "values"
        assert stats.dtype == "int64"
        assert stats.null_count == 0
        assert stats.null_percentage == 0.0
        assert stats.unique_count == 10
        assert stats.mean == 5.5
        assert stats.min_value == 1
        assert stats.max_value == 10

    def test_get_column_stats_with_nulls(self):
        """Test column stats with null values."""
        df = pd.DataFrame({
            "values": [1, 2, None, 4, None, 6, 7, 8, 9, 10]
        })

        stats = get_column_stats(df, "values")

        assert stats.null_count == 2
        assert stats.null_percentage == 20.0
        assert stats.unique_count == 8

    def test_get_column_stats_categorical(self):
        """Test column stats for categorical column."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "B", "A", "C", "C"]
        })

        stats = get_column_stats(df, "category")

        assert stats.column_name == "category"
        assert stats.dtype == "object"
        assert stats.unique_count == 3
        assert stats.mean is None  # No numeric stats for strings

    def test_get_column_stats_percentiles(self):
        """Test column stats percentile calculation."""
        df = pd.DataFrame({
            "values": list(range(1, 101))  # 1 to 100
        })

        stats = get_column_stats(df, "values")

        assert stats.percentiles is not None
        assert abs(stats.percentiles["25%"] - 25.75) < 1
        assert abs(stats.percentiles["50%"] - 50.5) < 1
        assert abs(stats.percentiles["75%"] - 75.25) < 1

    def test_get_column_stats_sample_values(self):
        """Test column stats sample values."""
        df = pd.DataFrame({
            "values": ["val1", "val2", "val3", "val4", "val5", "val6"]
        })

        stats = get_column_stats(df, "values")

        assert len(stats.sample_values) <= 5
        assert "val1" in stats.sample_values

    def test_get_column_stats_empty_dataframe(self):
        """Test column stats on empty DataFrame."""
        df = pd.DataFrame({"values": []})

        stats = get_column_stats(df, "values")

        assert stats.null_percentage == 0.0
        assert stats.unique_count == 0


@pytest.mark.unit
class TestGetSplitStats:
    """Test get_split_stats function."""

    def test_get_split_stats_basic(self):
        """Test split stats calculation."""
        df = pd.DataFrame({
            "data_split": ["train"] * 60 + ["validation"] * 20 + ["test"] * 15 + ["holdout"] * 5
        })

        stats = get_split_stats(df, "data_split")

        assert len(stats) == 4
        assert stats[0].split_name == "train"
        assert stats[0].record_count == 60
        assert stats[0].percentage == 60.0

    def test_get_split_stats_with_dates(self):
        """Test split stats with date ranges."""
        df = pd.DataFrame({
            "data_split": ["train"] * 3 + ["validation"] * 3,
            "date": pd.date_range("2022-01-01", periods=6),
        })

        stats = get_split_stats(df, "data_split", "date")

        assert stats[0].date_range is not None
        assert stats[0].date_range[0] == "2022-01-01"

    def test_get_split_stats_missing_column(self):
        """Test split stats when column doesn't exist."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        stats = get_split_stats(df, "data_split")

        assert stats == []

    def test_get_split_stats_partial_splits(self):
        """Test split stats when not all splits are present."""
        df = pd.DataFrame({
            "data_split": ["train"] * 70 + ["test"] * 30
        })

        stats = get_split_stats(df, "data_split")

        # Should only return splits that exist (train and test)
        split_names = [s.split_name for s in stats]
        assert "train" in split_names
        assert "test" in split_names


@pytest.mark.unit
class TestGetDatasetStats:
    """Test get_dataset_stats function."""

    def test_get_dataset_stats_basic(self):
        """Test basic dataset stats calculation."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
            "category": ["A", "B", "A", "C", "B"],
        })

        stats = get_dataset_stats(df, "test_table")

        assert stats.table_name == "test_table"
        assert stats.total_records == 5
        assert stats.column_count == 3
        assert "id" in stats.column_stats
        assert "value" in stats.column_stats
        assert "category" in stats.column_stats

    def test_get_dataset_stats_with_splits(self):
        """Test dataset stats with split information."""
        df = pd.DataFrame({
            "id": list(range(100)),
            "data_split": ["train"] * 60 + ["validation"] * 20 + ["test"] * 15 + ["holdout"] * 5,
        })

        stats = get_dataset_stats(df, "test_table", split_column="data_split")

        assert stats.split_stats is not None
        assert len(stats.split_stats) == 4

    def test_get_dataset_stats_with_brands(self):
        """Test dataset stats with brand distribution."""
        df = pd.DataFrame({
            "id": list(range(100)),
            "brand": (
                ["Remibrutinib"] * 35 + ["Fabhalta"] * 33 + ["Kisqali"] * 32
            ),
        })

        stats = get_dataset_stats(df, "test_table", brand_column="brand")

        assert stats.brand_distribution is not None
        assert "Remibrutinib" in stats.brand_distribution
        assert stats.brand_distribution["Remibrutinib"] == 35

    def test_get_dataset_stats_memory_usage(self):
        """Test dataset stats calculates memory usage."""
        df = pd.DataFrame({
            "values": list(range(1000))
        })

        stats = get_dataset_stats(df, "test_table")

        assert stats.memory_usage_mb > 0

    def test_get_dataset_stats_empty_dataframe(self):
        """Test dataset stats on empty DataFrame."""
        df = pd.DataFrame({"col1": [], "col2": []})

        stats = get_dataset_stats(df, "empty_table")

        assert stats.total_records == 0
        assert stats.column_count == 2


@pytest.mark.unit
class TestGetAllDatasetsStats:
    """Test get_all_datasets_stats function."""

    def test_get_all_datasets_stats(self):
        """Test stats calculation for multiple datasets."""
        datasets = {
            "table1": pd.DataFrame({"id": [1, 2, 3]}),
            "table2": pd.DataFrame({"value": [10, 20, 30, 40]}),
        }

        all_stats = get_all_datasets_stats(datasets)

        assert len(all_stats) == 2
        assert "table1" in all_stats
        assert "table2" in all_stats
        assert all_stats["table1"].total_records == 3
        assert all_stats["table2"].total_records == 4

    def test_get_all_datasets_stats_with_splits(self):
        """Test all datasets stats with split information."""
        datasets = {
            "table1": pd.DataFrame({
                "id": [1, 2, 3, 4],
                "data_split": ["train", "train", "test", "test"],
            }),
        }

        all_stats = get_all_datasets_stats(datasets, split_column="data_split")

        assert all_stats["table1"].split_stats is not None

    def test_get_all_datasets_stats_empty(self):
        """Test all datasets stats with empty dict."""
        all_stats = get_all_datasets_stats({})

        assert all_stats == {}


@pytest.mark.unit
class TestValidateSupabaseData:
    """Test validate_supabase_data function."""

    def test_validate_supabase_data_valid(self):
        """Test validation with valid data."""
        datasets = {
            "patient_journeys": pd.DataFrame({
                "patient_id": [1, 2, 3, 4],
                "brand": ["Remibrutinib", "Fabhalta", "Kisqali", "Remibrutinib"],
                "data_split": ["train", "validation", "test", "holdout"],  # Need all 4 splits
            }),
        }

        result = validate_supabase_data(datasets)

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_supabase_data_invalid_brands(self):
        """Test validation with invalid brand values."""
        datasets = {
            "patient_journeys": pd.DataFrame({
                "patient_id": [1, 2],
                "brand": ["InvalidBrand", "Remibrutinib"],
                "data_split": ["train", "test"],
            }),
        }

        result = validate_supabase_data(datasets)

        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert any("Invalid brand" in err for err in result["errors"])

    def test_validate_supabase_data_lowercase_brands(self):
        """Test validation catches lowercase brand names."""
        datasets = {
            "patient_journeys": pd.DataFrame({
                "patient_id": [1, 2],
                "brand": ["remibrutinib", "fabhalta"],
                "data_split": ["train", "test"],
            }),
        }

        result = validate_supabase_data(datasets)

        assert result["is_valid"] is False
        assert any("Lowercase brand" in err for err in result["errors"])

    def test_validate_supabase_data_missing_splits(self):
        """Test validation with missing splits."""
        datasets = {
            "patient_journeys": pd.DataFrame({
                "patient_id": [1, 2],
                "brand": ["Remibrutinib", "Fabhalta"],
                "data_split": ["train", "test"],  # Missing validation and holdout
            }),
        }

        result = validate_supabase_data(datasets)

        assert len(result["warnings"]) > 0
        assert any("Missing splits" in warn for warn in result["warnings"])

    def test_validate_supabase_data_missing_holdout(self):
        """Test validation flags missing holdout as error."""
        datasets = {
            "patient_journeys": pd.DataFrame({
                "patient_id": [1, 2, 3],
                "brand": ["Remibrutinib"] * 3,
                "data_split": ["train", "validation", "test"],  # Missing holdout
            }),
        }

        result = validate_supabase_data(datasets)

        assert result["is_valid"] is False
        assert any("Missing holdout" in err for err in result["errors"])

    def test_validate_supabase_data_empty_dataset(self):
        """Test validation with empty dataset."""
        datasets = {
            "patient_journeys": pd.DataFrame(),
        }

        result = validate_supabase_data(datasets)

        assert result["is_valid"] is False
        assert any("Empty dataset" in err for err in result["errors"])

    def test_validate_supabase_data_custom_brands(self):
        """Test validation with custom expected brands."""
        datasets = {
            "patient_journeys": pd.DataFrame({
                "patient_id": [1, 2],
                "brand": ["BrandA", "BrandB"],
            }),
        }

        result = validate_supabase_data(datasets, expected_brands=["BrandA", "BrandB"])

        assert result["is_valid"] is True

    def test_validate_supabase_data_stats_included(self):
        """Test validation includes stats for all tables."""
        datasets = {
            "table1": pd.DataFrame({"id": [1, 2, 3]}),
            "table2": pd.DataFrame({"id": [4, 5]}),
        }

        result = validate_supabase_data(datasets)

        assert "stats" in result
        assert "table1" in result["stats"]
        assert "table2" in result["stats"]


@pytest.mark.unit
class TestPrintDatasetSummary:
    """Test print_dataset_summary function."""

    def test_print_dataset_summary(self, capsys):
        """Test dataset summary printing."""
        datasets = {
            "patient_journeys": pd.DataFrame({
                "patient_id": list(range(100)),
                "brand": ["Remibrutinib"] * 100,
            }),
            "hcp_profiles": pd.DataFrame({
                "hcp_id": list(range(50)),
            }),
        }

        print_dataset_summary(datasets)

        captured = capsys.readouterr()
        assert "SYNTHETIC DATA SUMMARY" in captured.out
        assert "patient_journeys" in captured.out
        assert "hcp_profiles" in captured.out
        assert "100" in captured.out
        assert "50" in captured.out

    def test_print_dataset_summary_empty(self, capsys):
        """Test summary printing with empty datasets."""
        datasets = {}

        print_dataset_summary(datasets)

        captured = capsys.readouterr()
        assert "SYNTHETIC DATA SUMMARY" in captured.out


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_column_stats_all_null(self):
        """Test column stats when all values are null."""
        df = pd.DataFrame({"values": [None, None, None]})

        stats = get_column_stats(df, "values")

        assert stats.null_count == 3
        assert stats.null_percentage == 100.0
        assert stats.unique_count == 0

    def test_column_stats_single_value(self):
        """Test column stats with single value."""
        df = pd.DataFrame({"values": [42]})

        stats = get_column_stats(df, "values")

        assert stats.unique_count == 1
        assert stats.mean == 42
        # Pandas returns NaN for std of single value (expected behavior)
        assert stats.std is not None

    def test_split_stats_unordered_splits(self):
        """Test split stats with non-standard split names."""
        df = pd.DataFrame({
            "data_split": ["custom1", "custom2", "train"]
        })

        stats = get_split_stats(df, "data_split")

        # Should only return standard splits in order
        split_names = [s.split_name for s in stats]
        assert "train" in split_names

    def test_dataset_stats_large_dataframe(self):
        """Test dataset stats with large DataFrame."""
        df = pd.DataFrame({
            "id": list(range(100000)),
            "value": np.random.randn(100000),
        })

        stats = get_dataset_stats(df, "large_table")

        assert stats.total_records == 100000
        assert stats.memory_usage_mb > 0

    def test_validate_supabase_data_no_brand_column(self):
        """Test validation when brand column is missing."""
        datasets = {
            "table": pd.DataFrame({"id": [1, 2, 3]}),
        }

        result = validate_supabase_data(datasets)

        # Should not fail, just skip brand validation
        assert isinstance(result, dict)
        assert "is_valid" in result

    def test_validate_supabase_data_no_split_column(self):
        """Test validation when split column is missing."""
        datasets = {
            "table": pd.DataFrame({
                "id": [1, 2, 3],
                "brand": ["Remibrutinib"] * 3,
            }),
        }

        result = validate_supabase_data(datasets)

        # Should not fail, just skip split validation
        assert isinstance(result, dict)
        assert "is_valid" in result
