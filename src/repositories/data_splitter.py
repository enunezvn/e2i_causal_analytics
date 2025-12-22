"""
Data Splitter - Phase 1: Data Loading Foundation

Provides train/validation/test splitting utilities with:
- Temporal awareness (prevents future data leakage)
- Stratified splitting for classification tasks
- Patient/entity-level splitting (prevents entity leakage)
- Reproducible splits via random seeds

Version: 1.0.0
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42
    stratify_column: Optional[str] = None
    entity_column: Optional[str] = None  # For entity-level splits
    date_column: Optional[str] = None  # For temporal splits
    holdout_ratio: float = 0.0  # Optional holdout set
    _skip_validation: bool = False  # Internal: skip ratio validation for computed configs

    def __post_init__(self):
        if self._skip_validation:
            return
        total = self.train_ratio + self.val_ratio + self.test_ratio + self.holdout_ratio
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class SplitResult:
    """Result of data splitting operation."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    holdout: Optional[pd.DataFrame] = None
    config: SplitConfig = field(default_factory=SplitConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        """Return splits as a dictionary."""
        result = {"train": self.train, "val": self.val, "test": self.test}
        if self.holdout is not None:
            result["holdout"] = self.holdout
        return result

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "train_size": len(self.train),
            "val_size": len(self.val),
            "test_size": len(self.test),
            "holdout_size": len(self.holdout) if self.holdout is not None else 0,
            "total_size": len(self.train)
            + len(self.val)
            + len(self.test)
            + (len(self.holdout) if self.holdout is not None else 0),
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "holdout_ratio": self.config.holdout_ratio,
                "stratify_column": self.config.stratify_column,
                "entity_column": self.config.entity_column,
                "date_column": self.config.date_column,
            },
            **self.metadata,
        }


class DataSplitter:
    """
    Data splitting utility with multiple strategies.

    Supports:
    1. Random splits: Simple random partitioning
    2. Temporal splits: Based on date column (prevents future leakage)
    3. Stratified splits: Maintains class distribution
    4. Entity-level splits: All records for an entity in same split

    Example:
        splitter = DataSplitter()

        # Simple random split
        result = splitter.random_split(df, config=SplitConfig())

        # Temporal split (for time series)
        result = splitter.temporal_split(
            df,
            date_column="created_at",
            split_date="2024-06-01"
        )

        # Entity-level split (prevents patient leakage)
        result = splitter.entity_split(
            df,
            entity_column="patient_id"
        )
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize splitter.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def random_split(
        self,
        df: pd.DataFrame,
        config: Optional[SplitConfig] = None,
    ) -> SplitResult:
        """
        Perform simple random split.

        Args:
            df: DataFrame to split
            config: Split configuration

        Returns:
            SplitResult with train/val/test DataFrames
        """
        config = config or SplitConfig(random_seed=self.random_seed)
        np.random.seed(config.random_seed)

        n = len(df)
        indices = np.random.permutation(n)

        train_end = int(n * config.train_ratio)
        val_end = train_end + int(n * config.val_ratio)
        test_end = val_end + int(n * config.test_ratio)

        train_df = df.iloc[indices[:train_end]].reset_index(drop=True)
        val_df = df.iloc[indices[train_end:val_end]].reset_index(drop=True)
        test_df = df.iloc[indices[val_end:test_end]].reset_index(drop=True)

        holdout_df = None
        if config.holdout_ratio > 0:
            holdout_df = df.iloc[indices[test_end:]].reset_index(drop=True)

        return SplitResult(
            train=train_df,
            val=val_df,
            test=test_df,
            holdout=holdout_df,
            config=config,
            metadata={"split_type": "random", "original_size": n},
        )

    def temporal_split(
        self,
        df: pd.DataFrame,
        date_column: str,
        split_date: Optional[str] = None,
        val_days: int = 30,
        test_days: int = 30,
    ) -> SplitResult:
        """
        Perform temporal split to prevent future data leakage.

        Split Strategy:
        - Training: data before (split_date - val_days - test_days)
        - Validation: (split_date - val_days - test_days) to (split_date - test_days)
        - Test: (split_date - test_days) to split_date

        Args:
            df: DataFrame to split
            date_column: Column containing dates
            split_date: Reference date (defaults to max date in data)
            val_days: Days for validation set
            test_days: Days for test set

        Returns:
            SplitResult with temporally ordered splits
        """
        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Determine split date
        if split_date:
            ref_date = pd.to_datetime(split_date)
        else:
            ref_date = df[date_column].max()

        # Calculate boundaries
        test_start = ref_date - pd.Timedelta(days=test_days)
        val_start = test_start - pd.Timedelta(days=val_days)

        # Split data
        train_df = df[df[date_column] < val_start].reset_index(drop=True)
        val_df = df[(df[date_column] >= val_start) & (df[date_column] < test_start)].reset_index(
            drop=True
        )
        test_df = df[(df[date_column] >= test_start) & (df[date_column] <= ref_date)].reset_index(
            drop=True
        )

        config = SplitConfig(
            train_ratio=len(train_df) / len(df) if len(df) > 0 else 0,
            val_ratio=len(val_df) / len(df) if len(df) > 0 else 0,
            test_ratio=len(test_df) / len(df) if len(df) > 0 else 0,
            date_column=date_column,
            random_seed=self.random_seed,
            _skip_validation=True,  # Ratios are computed from actual data
        )

        return SplitResult(
            train=train_df,
            val=val_df,
            test=test_df,
            config=config,
            metadata={
                "split_type": "temporal",
                "split_date": ref_date.isoformat(),
                "val_start": val_start.isoformat(),
                "test_start": test_start.isoformat(),
                "original_size": len(df),
            },
        )

    def stratified_split(
        self,
        df: pd.DataFrame,
        stratify_column: str,
        config: Optional[SplitConfig] = None,
    ) -> SplitResult:
        """
        Perform stratified split maintaining class distribution.

        Args:
            df: DataFrame to split
            stratify_column: Column to stratify by
            config: Split configuration

        Returns:
            SplitResult with stratified splits
        """
        config = config or SplitConfig(
            stratify_column=stratify_column, random_seed=self.random_seed
        )
        np.random.seed(config.random_seed)

        train_dfs = []
        val_dfs = []
        test_dfs = []
        holdout_dfs = []

        # Split each stratum independently
        for stratum_value in df[stratify_column].unique():
            stratum_df = df[df[stratify_column] == stratum_value]
            n = len(stratum_df)

            if n < 3:
                # Too few samples, put all in training
                train_dfs.append(stratum_df)
                continue

            indices = np.random.permutation(n)

            train_end = int(n * config.train_ratio)
            val_end = train_end + int(n * config.val_ratio)
            test_end = val_end + int(n * config.test_ratio)

            train_dfs.append(stratum_df.iloc[indices[:train_end]])
            val_dfs.append(stratum_df.iloc[indices[train_end:val_end]])
            test_dfs.append(stratum_df.iloc[indices[val_end:test_end]])

            if config.holdout_ratio > 0:
                holdout_dfs.append(stratum_df.iloc[indices[test_end:]])

        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        holdout_df = pd.concat(holdout_dfs, ignore_index=True) if holdout_dfs else None

        return SplitResult(
            train=train_df,
            val=val_df,
            test=test_df,
            holdout=holdout_df,
            config=config,
            metadata={
                "split_type": "stratified",
                "stratify_column": stratify_column,
                "strata_count": df[stratify_column].nunique(),
                "original_size": len(df),
            },
        )

    def entity_split(
        self,
        df: pd.DataFrame,
        entity_column: str,
        config: Optional[SplitConfig] = None,
    ) -> SplitResult:
        """
        Perform entity-level split to prevent entity leakage.

        All records for a given entity (e.g., patient_id) will be in the same split.
        This prevents data leakage when entities have multiple records.

        Args:
            df: DataFrame to split
            entity_column: Column containing entity IDs
            config: Split configuration

        Returns:
            SplitResult with entity-level splits
        """
        config = config or SplitConfig(entity_column=entity_column, random_seed=self.random_seed)

        # Get unique entities
        entities = df[entity_column].unique()
        n_entities = len(entities)

        # Assign entities to splits using hash for determinism
        entity_splits = {}
        for entity in entities:
            # Use hash for deterministic assignment
            hash_val = int(hashlib.md5(str(entity).encode()).hexdigest(), 16)
            normalized = hash_val / (2**128)

            train_end = config.train_ratio
            val_end = train_end + config.val_ratio
            test_end = val_end + config.test_ratio

            if normalized < train_end:
                entity_splits[entity] = "train"
            elif normalized < val_end:
                entity_splits[entity] = "val"
            elif normalized < test_end:
                entity_splits[entity] = "test"
            else:
                entity_splits[entity] = "holdout"

        # Split data based on entity assignments
        df = df.copy()
        df["_split"] = df[entity_column].map(entity_splits)

        train_df = df[df["_split"] == "train"].drop("_split", axis=1).reset_index(drop=True)
        val_df = df[df["_split"] == "val"].drop("_split", axis=1).reset_index(drop=True)
        test_df = df[df["_split"] == "test"].drop("_split", axis=1).reset_index(drop=True)
        holdout_df = None
        if config.holdout_ratio > 0:
            holdout_df = df[df["_split"] == "holdout"].drop("_split", axis=1).reset_index(drop=True)

        return SplitResult(
            train=train_df,
            val=val_df,
            test=test_df,
            holdout=holdout_df,
            config=config,
            metadata={
                "split_type": "entity",
                "entity_column": entity_column,
                "entity_count": n_entities,
                "original_size": len(df),
            },
        )

    def combined_split(
        self,
        df: pd.DataFrame,
        date_column: str,
        entity_column: str,
        split_date: Optional[str] = None,
        val_days: int = 30,
        test_days: int = 30,
    ) -> SplitResult:
        """
        Combined temporal + entity split for maximum leakage prevention.

        First splits by date, then ensures entities don't cross splits.
        Records for entities that span multiple time periods are assigned
        to the earliest period where they appear.

        Args:
            df: DataFrame to split
            date_column: Column containing dates
            entity_column: Column containing entity IDs
            split_date: Reference date
            val_days: Days for validation set
            test_days: Days for test set

        Returns:
            SplitResult with combined splits
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Determine split date
        if split_date:
            ref_date = pd.to_datetime(split_date)
        else:
            ref_date = df[date_column].max()

        # Calculate boundaries
        test_start = ref_date - pd.Timedelta(days=test_days)
        val_start = test_start - pd.Timedelta(days=val_days)

        # Find earliest period for each entity
        entity_first_date = df.groupby(entity_column)[date_column].min()

        def assign_entity_split(first_date):
            if first_date < val_start:
                return "train"
            elif first_date < test_start:
                return "val"
            else:
                return "test"

        entity_splits = entity_first_date.apply(assign_entity_split)

        # Apply splits
        df["_split"] = df[entity_column].map(entity_splits)

        train_df = df[df["_split"] == "train"].drop("_split", axis=1).reset_index(drop=True)
        val_df = df[df["_split"] == "val"].drop("_split", axis=1).reset_index(drop=True)
        test_df = df[df["_split"] == "test"].drop("_split", axis=1).reset_index(drop=True)

        config = SplitConfig(
            train_ratio=len(train_df) / len(df) if len(df) > 0 else 0,
            val_ratio=len(val_df) / len(df) if len(df) > 0 else 0,
            test_ratio=len(test_df) / len(df) if len(df) > 0 else 0,
            date_column=date_column,
            entity_column=entity_column,
            random_seed=self.random_seed,
            _skip_validation=True,  # Ratios are computed from actual data
        )

        return SplitResult(
            train=train_df,
            val=val_df,
            test=test_df,
            config=config,
            metadata={
                "split_type": "combined_temporal_entity",
                "split_date": ref_date.isoformat(),
                "val_start": val_start.isoformat(),
                "test_start": test_start.isoformat(),
                "entity_count": df[entity_column].nunique(),
                "original_size": len(df),
            },
        )


# Convenience function
def get_data_splitter(random_seed: int = 42) -> DataSplitter:
    """Get a DataSplitter instance."""
    return DataSplitter(random_seed=random_seed)
