"""
Synthetic Data Statistics Utility.

Provides comprehensive statistics for synthetic datasets, useful for:
- Validation and quality checks
- Reporting generation
- DGP comparison

Ported from external system (E2i synthetic data/) and enhanced for
the main synthetic data system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class SplitStats:
    """Statistics for a single data split."""

    split_name: str
    record_count: int
    percentage: float
    date_range: Optional[Tuple[str, str]] = None


@dataclass
class ColumnStats:
    """Statistics for a single column."""

    column_name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any] = field(default_factory=list)

    # Numeric stats (if applicable)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    percentiles: Optional[Dict[str, float]] = None


@dataclass
class DatasetStats:
    """Comprehensive statistics for a synthetic dataset."""

    table_name: str
    total_records: int
    column_count: int
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)
    split_stats: Optional[List[SplitStats]] = None
    brand_distribution: Optional[Dict[str, int]] = None
    memory_usage_mb: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "table_name": self.table_name,
            "total_records": self.total_records,
            "column_count": self.column_count,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "column_stats": {
                name: {
                    "dtype": stats.dtype,
                    "null_count": stats.null_count,
                    "null_percentage": round(stats.null_percentage, 2),
                    "unique_count": stats.unique_count,
                    "mean": round(stats.mean, 4) if stats.mean else None,
                    "std": round(stats.std, 4) if stats.std else None,
                }
                for name, stats in self.column_stats.items()
            },
            "split_stats": (
                [
                    {
                        "split": s.split_name,
                        "count": s.record_count,
                        "percentage": round(s.percentage, 2),
                    }
                    for s in self.split_stats
                ]
                if self.split_stats
                else None
            ),
            "brand_distribution": self.brand_distribution,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Dataset: {self.table_name}",
            f"  Records: {self.total_records:,}",
            f"  Columns: {self.column_count}",
            f"  Memory: {self.memory_usage_mb:.2f} MB",
        ]

        if self.split_stats:
            lines.append("  Splits:")
            for split in self.split_stats:
                lines.append(
                    f"    {split.split_name}: {split.record_count:,} ({split.percentage:.1f}%)"
                )

        if self.brand_distribution:
            lines.append("  Brands:")
            for brand, count in self.brand_distribution.items():
                lines.append(f"    {brand}: {count:,}")

        return "\n".join(lines)


def get_column_stats(df: pd.DataFrame, column: str) -> ColumnStats:
    """
    Get statistics for a single column.

    Args:
        df: DataFrame containing the column
        column: Column name

    Returns:
        ColumnStats with comprehensive statistics
    """
    col = df[column]
    null_count = col.isna().sum()

    stats = ColumnStats(
        column_name=column,
        dtype=str(col.dtype),
        null_count=int(null_count),
        null_percentage=float(null_count / len(df) * 100) if len(df) > 0 else 0.0,
        unique_count=int(col.nunique()),
        sample_values=col.dropna().head(5).tolist(),
    )

    # Add numeric stats if applicable
    if pd.api.types.is_numeric_dtype(col):
        non_null = col.dropna()
        if len(non_null) > 0:
            stats.mean = float(non_null.mean())
            stats.std = float(non_null.std())
            stats.min_value = float(non_null.min())
            stats.max_value = float(non_null.max())
            stats.percentiles = {
                "25%": float(non_null.quantile(0.25)),
                "50%": float(non_null.quantile(0.50)),
                "75%": float(non_null.quantile(0.75)),
            }

    return stats


def get_split_stats(
    df: pd.DataFrame,
    split_column: str = "data_split",
    date_column: Optional[str] = None,
) -> List[SplitStats]:
    """
    Get statistics for each data split.

    Args:
        df: DataFrame with split column
        split_column: Name of the split column
        date_column: Optional date column for range calculation

    Returns:
        List of SplitStats for each split
    """
    if split_column not in df.columns:
        return []

    split_counts = df[split_column].value_counts()
    total = len(df)

    stats = []
    expected_order = ["train", "validation", "test", "holdout"]

    for split in expected_order:
        if split in split_counts.index:
            count = int(split_counts[split])

            # Get date range if date column provided
            date_range = None
            if date_column and date_column in df.columns:
                split_df = df[df[split_column] == split]
                dates = pd.to_datetime(split_df[date_column])
                if len(dates) > 0:
                    date_range = (
                        dates.min().strftime("%Y-%m-%d"),
                        dates.max().strftime("%Y-%m-%d"),
                    )

            stats.append(
                SplitStats(
                    split_name=split,
                    record_count=count,
                    percentage=float(count / total * 100) if total > 0 else 0.0,
                    date_range=date_range,
                )
            )

    return stats


def get_dataset_stats(
    df: pd.DataFrame,
    table_name: str,
    split_column: Optional[str] = "data_split",
    brand_column: Optional[str] = "brand",
    date_column: Optional[str] = None,
) -> DatasetStats:
    """
    Get comprehensive statistics for a synthetic dataset.

    This is the main function for analyzing synthetic data quality.

    Args:
        df: DataFrame to analyze
        table_name: Name of the table/dataset
        split_column: Column containing data splits (optional)
        brand_column: Column containing brand values (optional)
        date_column: Column containing dates (optional)

    Returns:
        DatasetStats with comprehensive statistics

    Example:
        >>> stats = get_dataset_stats(patient_df, "patient_journeys")
        >>> print(stats.summary())
        Dataset: patient_journeys
          Records: 10,000
          Columns: 12
          Memory: 2.45 MB
          Splits:
            train: 6,000 (60.0%)
            validation: 2,000 (20.0%)
            test: 1,500 (15.0%)
            holdout: 500 (5.0%)
          Brands:
            Remibrutinib: 3,500
            Fabhalta: 3,200
            Kisqali: 3,300
    """
    # Basic stats
    stats = DatasetStats(
        table_name=table_name,
        total_records=len(df),
        column_count=len(df.columns),
        memory_usage_mb=float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
    )

    # Column stats
    for col in df.columns:
        stats.column_stats[col] = get_column_stats(df, col)

    # Split stats
    if split_column and split_column in df.columns:
        stats.split_stats = get_split_stats(df, split_column, date_column)

    # Brand distribution
    if brand_column and brand_column in df.columns:
        brand_counts = df[brand_column].value_counts()
        stats.brand_distribution = {str(k): int(v) for k, v in brand_counts.items()}

    return stats


def get_all_datasets_stats(
    datasets: Dict[str, pd.DataFrame],
    split_column: Optional[str] = "data_split",
    brand_column: Optional[str] = "brand",
) -> Dict[str, DatasetStats]:
    """
    Get statistics for multiple datasets.

    Args:
        datasets: Dictionary of table_name -> DataFrame
        split_column: Column containing data splits
        brand_column: Column containing brand values

    Returns:
        Dictionary of table_name -> DatasetStats
    """
    return {
        table_name: get_dataset_stats(
            df=df,
            table_name=table_name,
            split_column=split_column if split_column in df.columns else None,
            brand_column=brand_column if brand_column in df.columns else None,
        )
        for table_name, df in datasets.items()
    }


def validate_supabase_data(
    datasets: Dict[str, pd.DataFrame],
    expected_brands: Optional[List[str]] = None,
    expected_splits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate synthetic data for Supabase compatibility.

    This replaces the validate_data_in_supabase() function from
    the external system with enhanced local validation.

    Args:
        datasets: Dictionary of table_name -> DataFrame
        expected_brands: Expected brand values (default: Remibrutinib, Fabhalta, Kisqali)
        expected_splits: Expected split values (default: train, validation, test, holdout)

    Returns:
        Dictionary with validation results
    """
    if expected_brands is None:
        expected_brands = ["Remibrutinib", "Fabhalta", "Kisqali"]

    if expected_splits is None:
        expected_splits = ["train", "validation", "test", "holdout"]

    results: Dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    for table_name, df in datasets.items():
        # Get stats
        stats = get_dataset_stats(df, table_name)
        results["stats"][table_name] = stats.to_dict()

        # Check for empty data
        if len(df) == 0:
            results["errors"].append(f"{table_name}: Empty dataset")
            results["is_valid"] = False
            continue

        # Check brand values (if brand column exists)
        if "brand" in df.columns:
            actual_brands = set(df["brand"].dropna().unique())
            invalid_brands = actual_brands - set(expected_brands)
            if invalid_brands:
                results["errors"].append(
                    f"{table_name}: Invalid brand values: {invalid_brands}. "
                    f"Expected: {expected_brands}"
                )
                results["is_valid"] = False

            # Check for lowercase brands (external system bug)
            lowercase_brands = [b for b in actual_brands if b.islower()]
            if lowercase_brands:
                results["errors"].append(
                    f"{table_name}: Lowercase brand values found: {lowercase_brands}. "
                    "Brands should be capitalized for Supabase compatibility."
                )
                results["is_valid"] = False

        # Check split values (if split column exists)
        if "data_split" in df.columns:
            actual_splits = set(df["data_split"].dropna().unique())
            missing_splits = set(expected_splits) - actual_splits
            if missing_splits:
                results["warnings"].append(f"{table_name}: Missing splits: {missing_splits}")
                if "holdout" in missing_splits:
                    results["errors"].append(
                        f"{table_name}: Missing holdout split. "
                        "This was a bug in the external system."
                    )
                    results["is_valid"] = False

    return results


def print_dataset_summary(datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Print a formatted summary of all datasets.

    Args:
        datasets: Dictionary of table_name -> DataFrame
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA SUMMARY")
    print("=" * 60)

    all_stats = get_all_datasets_stats(datasets)

    total_records = 0
    total_memory = 0.0

    for _table_name, stats in all_stats.items():
        print(f"\n{stats.summary()}")
        total_records += stats.total_records
        total_memory += stats.memory_usage_mb

    print("\n" + "-" * 60)
    print(f"TOTAL: {total_records:,} records, {total_memory:.2f} MB")
    print("=" * 60)
