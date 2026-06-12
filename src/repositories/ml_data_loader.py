"""
ML Data Loader - Phase 1: Data Loading Foundation

Provides Supabase data extraction for ML pipelines with:
- Split-aware querying (prevents data leakage)
- Temporal awareness (train/val/test splits by date)
- Support for all ML-relevant tables
- Pandas DataFrame output for sklearn/XGBoost compatibility

Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.repositories.base import SplitAwareRepository
from src.repositories.provenance import (
    PROVENANCE_TAGGED_TABLES as _PROVENANCE_TAGGED_TABLES,
)

logger = logging.getLogger(__name__)


# Supported tables for ML data loading
ML_TABLES = [
    "business_metrics",
    "ml_predictions",  # #894: was "predictions" — a table that does not exist
    # in the live schema, so the allowlist both blocked the REAL tagged
    # prediction table and let "predictions" loads 404 into empty DataFrames.
    "triggers",
    "causal_paths",
    "patient_journeys",
    "agent_activities",
]

# Tables that carry the is_synthetic provenance column, re-derived from
# migrations 063/067/069 via the provenance SSOT (#894). The previous local
# subset hard-excluded causal_paths and agent_activities on a stale pre-063
# "avoid a 42703" rationale — both ARE tagged (063:17-18), so real-mode loads
# of those tables leaked synthetic rows. (The old set also listed a
# "predictions" table that does not exist in the live schema — the tagged
# table is ml_predictions.)
PROVENANCE_TAGGED_TABLES = _PROVENANCE_TAGGED_TABLES


@dataclass
class MLDataset:
    """Container for ML dataset with train/val/test splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict[str, Any]

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def val_size(self) -> int:
        return len(self.val)

    @property
    def test_size(self) -> int:
        return len(self.test)

    @property
    def total_size(self) -> int:
        return self.train_size + self.val_size + self.test_size

    def summary(self) -> Dict[str, Any]:
        """Return dataset summary statistics."""
        return {
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "total_size": self.total_size,
            "train_columns": list(self.train.columns),
            "split_ratios": {
                "train": self.train_size / max(self.total_size, 1),
                "val": self.val_size / max(self.total_size, 1),
                "test": self.test_size / max(self.total_size, 1),
            },
            **self.metadata,
        }


class MLDataLoader(SplitAwareRepository):
    """
    ML-focused data loader for Supabase tables.

    Provides:
    - Temporal split management (prevents future data leakage)
    - Multiple table support (business_metrics, predictions, triggers, etc.)
    - DataFrame output compatible with sklearn/XGBoost
    - Configurable filtering and date ranges

    Example:
        loader = MLDataLoader(supabase_client)
        dataset = await loader.load_for_training(
            table="business_metrics",
            filters={"brand": "Kisqali"},
            split_date="2024-06-01",
            target_column="value"
        )
        X_train, y_train = dataset.train.drop("value", axis=1), dataset.train["value"]
    """

    table_name = "business_metrics"  # Default, can be changed per query
    model_class = None

    def __init__(self, supabase_client=None):
        """
        Initialize ML Data Loader.

        Args:
            supabase_client: Supabase client instance (optional, uses factory if not provided)
        """
        if supabase_client is None:
            try:
                from src.repositories import get_supabase_client

                supabase_client = get_supabase_client()
            except Exception as e:
                logger.warning(f"Could not get Supabase client: {e}")
        super().__init__(supabase_client)

    async def load_for_training(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        split_date: Optional[str] = None,
        date_column: str = "created_at",
        val_days: int = 30,
        test_days: int = 30,
        limit: int = 100000,
        columns: Optional[List[str]] = None,
        include_synthetic: bool = False,
    ) -> MLDataset:
        """
        Load data for ML training with temporal splits.

        Temporal Split Strategy:
        - Training: data before (split_date - val_days - test_days)
        - Validation: (split_date - val_days - test_days) to (split_date - test_days)
        - Test: (split_date - test_days) to split_date

        This prevents future data leakage by ensuring test data is always
        the most recent and training data is always the oldest.

        Args:
            table: Table name to load from (must be in ML_TABLES)
            filters: Column-value filters to apply
            split_date: Reference date for splits (defaults to today)
            date_column: Column to use for temporal splitting
            val_days: Days of data for validation set
            test_days: Days of data for test set
            limit: Maximum records to load
            columns: Specific columns to select (None = all)

        Returns:
            MLDataset with train, val, test DataFrames

        Raises:
            ValueError: If table not in ML_TABLES
        """
        if table not in ML_TABLES:
            raise ValueError(f"Table '{table}' not supported. Use one of: {ML_TABLES}")

        # Parse split date
        if split_date:
            ref_date = datetime.fromisoformat(split_date)
        else:
            ref_date = datetime.now()

        # Calculate split boundaries
        test_start = ref_date - timedelta(days=test_days)
        val_start = test_start - timedelta(days=val_days)

        logger.info(
            f"Loading {table} with splits: train<{val_start.date()}, "
            f"val=[{val_start.date()}, {test_start.date()}), "
            f"test>={test_start.date()}"
        )

        # Load data for each split
        train_data = await self._load_date_range(
            table=table,
            filters=filters,
            date_column=date_column,
            end_date=val_start.isoformat(),
            limit=limit,
            columns=columns,
            include_synthetic=include_synthetic,
        )

        val_data = await self._load_date_range(
            table=table,
            filters=filters,
            date_column=date_column,
            start_date=val_start.isoformat(),
            end_date=test_start.isoformat(),
            limit=limit,
            columns=columns,
            include_synthetic=include_synthetic,
        )

        test_data = await self._load_date_range(
            table=table,
            filters=filters,
            date_column=date_column,
            start_date=test_start.isoformat(),
            end_date=ref_date.isoformat(),
            limit=limit,
            columns=columns,
            include_synthetic=include_synthetic,
        )

        # Convert to DataFrames
        train_df = pd.DataFrame(train_data) if train_data else pd.DataFrame()
        val_df = pd.DataFrame(val_data) if val_data else pd.DataFrame()
        test_df = pd.DataFrame(test_data) if test_data else pd.DataFrame()

        metadata = {
            "table": table,
            "filters": filters or {},
            "split_date": ref_date.isoformat(),
            "date_column": date_column,
            "val_days": val_days,
            "test_days": test_days,
            "loaded_at": datetime.now().isoformat(),
        }

        return MLDataset(
            train=train_df,
            val=val_df,
            test=test_df,
            metadata=metadata,
        )

    async def _load_date_range(
        self,
        table: str,
        filters: Optional[Dict[str, Any]],
        date_column: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100000,
        columns: Optional[List[str]] = None,
        include_synthetic: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Load data from a table within a date range.

        Args:
            table: Table name
            filters: Column-value filters
            date_column: Column for date filtering
            start_date: Start of date range (inclusive)
            end_date: End of date range (exclusive)
            limit: Maximum records
            columns: Columns to select
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of row dictionaries
        """
        if not self.client:
            logger.warning("No Supabase client available, returning empty data")
            return []

        # Build column selection
        select_cols = ",".join(columns) if columns else "*"

        query = self.client.table(table).select(select_cols)

        # Apply filters
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)

        # Apply date range
        if start_date:
            query = query.gte(date_column, start_date)
        if end_date:
            query = query.lt(date_column, end_date)

        # Provenance default-exclude (taggable tables only -> avoid 42703)
        if not include_synthetic and table in PROVENANCE_TAGGED_TABLES:
            from src.repositories.provenance import apply_provenance_filter

            query = apply_provenance_filter(query, include_synthetic=False)

        # Order and limit
        query = query.order(date_column, desc=False).limit(limit)

        try:
            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to load data from {table}: {e}")
            return []

    async def load_table_sample(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
        columns: Optional[List[str]] = None,
        include_synthetic: bool = False,
    ) -> pd.DataFrame:
        """
        Load a sample from a table without splits.

        Useful for exploration and schema discovery.

        Args:
            table: Table name
            filters: Optional filters
            limit: Maximum records
            columns: Columns to select
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            DataFrame with sampled data
        """
        if table not in ML_TABLES:
            raise ValueError(f"Table '{table}' not supported. Use one of: {ML_TABLES}")

        if not self.client:
            logger.warning("No Supabase client available")
            return pd.DataFrame()

        select_cols = ",".join(columns) if columns else "*"
        query = self.client.table(table).select(select_cols)

        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)

        # Provenance default-exclude (taggable tables only -> avoid 42703).
        # NB: the predicate is a .eq filter, NOT a selected column, so an explicit
        # `columns` list (e.g. from the heterogeneous connector) never re-leaks the tag.
        if not include_synthetic and table in PROVENANCE_TAGGED_TABLES:
            from src.repositories.provenance import apply_provenance_filter

            query = apply_provenance_filter(query, include_synthetic=False)

        query = query.limit(limit)

        try:
            result = query.execute()
            return pd.DataFrame(result.data) if result.data else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load sample from {table}: {e}")
            return pd.DataFrame()

    async def get_table_schema(self, table: str) -> Dict[str, str]:
        """
        Get column names and types for a table.

        Args:
            table: Table name

        Returns:
            Dict mapping column names to types
        """
        # Load a single row to infer schema
        df = await self.load_table_sample(table, limit=1)
        if df.empty:
            return {}
        return {col: str(df[col].dtype) for col in df.columns}

    async def get_date_range(
        self,
        table: str,
        date_column: str = "created_at",
        filters: Optional[Dict[str, Any]] = None,
        include_synthetic: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the min and max dates for a table.

        Args:
            table: Table name
            date_column: Date column to check
            filters: Optional filters
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Tuple of (min_date, max_date) as ISO strings
        """
        if not self.client:
            return None, None

        query = self.client.table(table).select(date_column)

        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)

        if not include_synthetic and table in PROVENANCE_TAGGED_TABLES:
            from src.repositories.provenance import apply_provenance_filter

            query = apply_provenance_filter(query, include_synthetic=False)

        # Get min
        min_result = query.order(date_column, desc=False).limit(1).execute()
        min_date = min_result.data[0][date_column] if min_result.data else None

        # Get max
        max_result = query.order(date_column, desc=True).limit(1).execute()
        max_date = max_result.data[0][date_column] if max_result.data else None

        return min_date, max_date

    async def count_records(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        include_synthetic: bool = False,
    ) -> int:
        """
        Count records in a table.

        Args:
            table: Table name
            filters: Optional filters
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Record count
        """
        if not self.client:
            return 0

        query = self.client.table(table).select("id", count="exact")

        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)

        if not include_synthetic and table in PROVENANCE_TAGGED_TABLES:
            from src.repositories.provenance import apply_provenance_filter

            query = apply_provenance_filter(query, include_synthetic=False)

        try:
            result = query.execute()
            return result.count if hasattr(result, "count") else len(result.data)
        except Exception as e:
            logger.error(f"Failed to count records in {table}: {e}")
            return 0


# Convenience function for getting a loader instance
def get_ml_data_loader(supabase_client=None) -> MLDataLoader:
    """
    Get an ML Data Loader instance.

    Args:
        supabase_client: Optional Supabase client

    Returns:
        MLDataLoader instance
    """
    return MLDataLoader(supabase_client)
