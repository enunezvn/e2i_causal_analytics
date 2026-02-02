"""
Batch Loader for Supabase.

Loads synthetic data to Supabase in batches with validation and error handling.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of a batch load operation."""

    table_name: str
    records_loaded: int
    records_failed: int
    total_batches: int
    failed_batches: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.records_loaded + self.records_failed
        return self.records_loaded / total if total > 0 else 0.0

    @property
    def is_success(self) -> bool:
        """Check if load was successful (>95% success rate)."""
        return self.success_rate >= 0.95


@dataclass
class LoaderConfig:
    """Configuration for batch loader."""

    batch_size: int = 1000
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    validate_before_load: bool = True
    dry_run: bool = False
    verbose: bool = False

    # Connection settings
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    def __post_init__(self):
        """Load from environment if not provided."""
        if not self.supabase_url:
            self.supabase_url = os.getenv("SUPABASE_URL")
        if not self.supabase_key:
            self.supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")


# Table loading order (respects foreign key dependencies)
# Note: engagement_events and business_outcomes don't exist in current schema
LOADING_ORDER = [
    "hcp_profiles",
    "patient_journeys",
    "treatment_events",
    "ml_predictions",
    "triggers",
    "business_metrics",
    # Feature store tables (after feature_groups and features are seeded)
    "feature_groups",
    "features",
    "feature_values",
]

# Column mappings for each table (aligned with actual Supabase schema)
TABLE_COLUMNS = {
    "hcp_profiles": [
        "hcp_id",
        "npi",
        "specialty",
        "practice_type",
        "geographic_region",
        "years_experience",
        "total_patient_volume",
    ],
    "patient_journeys": [
        "patient_journey_id",
        "patient_id",
        "hcp_id",
        "brand",
        "journey_start_date",
        "insurance_type",
        "geographic_region",
        "disease_severity",
        "academic_hcp",
        "engagement_score",
        "treatment_initiated",
        "days_to_treatment",
        "age_at_diagnosis",
        "data_split",
    ],
    "treatment_events": [
        "treatment_event_id",
        "patient_journey_id",
        "patient_id",
        "brand",
        "event_date",
        "event_type",
        "duration_days",
        "data_split",
    ],
    "ml_predictions": [
        "prediction_id",
        "patient_id",
        "hcp_id",
        "prediction_type",
        "prediction_value",
        "confidence_score",
        "model_version",
        "prediction_timestamp",
        "data_split",
    ],
    "triggers": [
        "trigger_id",
        "patient_id",
        "hcp_id",
        "trigger_timestamp",
        "trigger_type",
        "priority",
        "confidence_score",
        "lead_time_days",
        "expiration_date",
        "delivery_channel",
        "delivery_status",
        "acceptance_status",
        "outcome_tracked",
        "outcome_value",
        "trigger_reason",
        "causal_chain",
        "supporting_evidence",
        "recommended_action",
        "data_split",
    ],
    "business_metrics": [
        "metric_id",
        "metric_date",
        "metric_type",
        "metric_name",
        "brand",
        "region",
        "value",
        "target",
        "achievement_rate",
        "year_over_year_change",
        "month_over_month_change",
        "roi",
        "statistical_significance",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "sample_size",
        "data_split",
    ],
    "feature_groups": [
        "id",
        "name",
        "description",
        "owner",
        "tags",
        "source_table",
        "expected_update_frequency_hours",
        "max_age_hours",
    ],
    "features": [
        "id",
        "feature_group_id",
        "name",
        "description",
        "value_type",
        "entity_keys",
        "owner",
        "tags",
        "drift_threshold",
    ],
    "feature_values": [
        "id",
        "feature_id",
        "entity_values",
        "value",
        "event_timestamp",
        "freshness_status",
    ],
}


class BatchLoader:
    """
    Batch loader for synthetic data to Supabase.

    Features:
    - Respects foreign key loading order
    - Batch processing with configurable size
    - Retry logic for transient failures
    - Validation before loading
    - Dry run mode for testing
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize the batch loader.

        Args:
            config: Loader configuration.
        """
        self.config = config or LoaderConfig()
        self._client = None

    @property
    def client(self):
        """Get or create Supabase client."""
        if self._client is None and not self.config.dry_run:
            try:
                from supabase import create_client

                self._client = create_client(
                    self.config.supabase_url,
                    self.config.supabase_key,
                )
            except ImportError:
                logger.warning("Supabase client not available")
            except Exception as e:
                logger.error(f"Failed to create Supabase client: {e}")
        return self._client

    def load_all(
        self,
        datasets: Dict[str, pd.DataFrame],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, LoadResult]:
        """
        Load all datasets in dependency order.

        Args:
            datasets: Dictionary of table_name -> DataFrame.
            progress_callback: Optional callback(table, current, total).

        Returns:
            Dictionary of table_name -> LoadResult.
        """
        results = {}

        # Determine tables to load
        tables_to_load = [t for t in LOADING_ORDER if t in datasets]
        total_tables = len(tables_to_load)

        for i, table_name in enumerate(tables_to_load):
            df = datasets[table_name]

            if progress_callback:
                progress_callback(table_name, i + 1, total_tables)

            if self.config.verbose:
                logger.info(f"Loading {table_name} ({len(df)} records)")

            result = self.load_table(table_name, df)
            results[table_name] = result

            if not result.is_success:
                logger.warning(
                    f"Table {table_name} load had failures: "
                    f"{result.records_failed}/{result.records_loaded + result.records_failed}"
                )

        return results

    def load_table(
        self,
        table_name: str,
        df: pd.DataFrame,
    ) -> LoadResult:
        """
        Load a single table in batches.

        Args:
            table_name: Target table name.
            df: DataFrame to load.

        Returns:
            LoadResult with statistics.
        """
        start_time = datetime.now()
        records_loaded = 0
        records_failed = 0
        failed_batches = []
        errors = []

        # Select and order columns
        available_columns = TABLE_COLUMNS.get(table_name, list(df.columns))
        columns_to_load = [c for c in available_columns if c in df.columns]
        df_to_load = df[columns_to_load].copy()

        # Handle JSON columns
        json_columns = ["causal_chain", "supporting_evidence"]
        for col in json_columns:
            if col in df_to_load.columns:
                df_to_load[col] = df_to_load[col].apply(
                    lambda x: x if isinstance(x, (dict, list)) else {}
                )

        # Handle None/NaN values - replace with None for JSON compatibility
        import numpy as np

        df_to_load = df_to_load.replace({np.nan: None, np.inf: None, -np.inf: None})
        df_to_load = df_to_load.where(pd.notnull(df_to_load), None)

        # Calculate batches
        total_records = len(df_to_load)
        batch_size = self.config.batch_size
        total_batches = (total_records + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_records)
            batch_df = df_to_load.iloc[start_idx:end_idx]

            success, error = self._load_batch(table_name, batch_df, batch_idx)

            if success:
                records_loaded += len(batch_df)
            else:
                records_failed += len(batch_df)
                failed_batches.append(batch_idx)
                if error:
                    errors.append(f"Batch {batch_idx}: {error}")

        duration = (datetime.now() - start_time).total_seconds()

        return LoadResult(
            table_name=table_name,
            records_loaded=records_loaded,
            records_failed=records_failed,
            total_batches=total_batches,
            failed_batches=failed_batches,
            errors=errors,
            duration_seconds=duration,
        )

    def _load_batch(
        self,
        table_name: str,
        batch_df: pd.DataFrame,
        batch_idx: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Load a single batch with retry logic.

        Args:
            table_name: Target table name.
            batch_df: Batch DataFrame.
            batch_idx: Batch index for logging.

        Returns:
            Tuple of (success, error_message).
        """
        if self.config.dry_run:
            if self.config.verbose:
                logger.info(f"[DRY RUN] Would load {len(batch_df)} records to {table_name}")
            return True, None

        if not self.client:
            return False, "Supabase client not available"

        records = batch_df.to_dict(orient="records")

        for attempt in range(self.config.max_retries):
            try:
                self.client.table(table_name).upsert(records).execute()
                return True, None
            except Exception as e:
                error_msg = str(e)
                if attempt < self.config.max_retries - 1:
                    import time

                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    logger.error(
                        f"Batch {batch_idx} failed after {self.config.max_retries} attempts: {error_msg}"
                    )
                    return False, error_msg

        return False, "Unknown error"

    def validate_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
    ) -> Tuple[bool, List[str]]:
        """
        Validate datasets before loading.

        Args:
            datasets: Dictionary of table_name -> DataFrame.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []

        for table_name, df in datasets.items():
            # Check if table is known
            if table_name not in TABLE_COLUMNS:
                errors.append(f"Unknown table: {table_name}")
                continue

            # Check required columns
            required_columns = TABLE_COLUMNS[table_name]
            missing_columns = [c for c in required_columns if c not in df.columns]

            # Some columns are optional (like data_split for static entities)
            critical_missing = [c for c in missing_columns if not c.endswith("_split")]
            if critical_missing:
                errors.append(f"{table_name}: Missing columns {critical_missing}")

            # Check for empty DataFrame
            if len(df) == 0:
                errors.append(f"{table_name}: Empty DataFrame")

        return len(errors) == 0, errors

    def get_loading_summary(
        self,
        results: Dict[str, LoadResult],
    ) -> str:
        """
        Generate a summary of loading results.

        Args:
            results: Dictionary of table_name -> LoadResult.

        Returns:
            Formatted summary string.
        """
        lines = ["=" * 60, "SYNTHETIC DATA LOADING SUMMARY", "=" * 60]

        total_loaded = 0
        total_failed = 0
        total_duration = 0.0

        for table_name in LOADING_ORDER:
            if table_name not in results:
                continue

            result = results[table_name]
            total_loaded += result.records_loaded
            total_failed += result.records_failed
            total_duration += result.duration_seconds

            status = "✓" if result.is_success else "✗"
            lines.append(
                f"{status} {table_name}: "
                f"{result.records_loaded:,} loaded, "
                f"{result.records_failed:,} failed "
                f"({result.success_rate:.1%}) "
                f"[{result.duration_seconds:.1f}s]"
            )

        lines.append("-" * 60)
        overall_rate = (
            total_loaded / (total_loaded + total_failed) if (total_loaded + total_failed) > 0 else 0
        )
        lines.append(
            f"TOTAL: {total_loaded:,} loaded, {total_failed:,} failed "
            f"({overall_rate:.1%}) [{total_duration:.1f}s]"
        )
        lines.append("=" * 60)

        return "\n".join(lines)


class AsyncBatchLoader(BatchLoader):
    """
    Async version of batch loader for concurrent loading.

    Use when loading large datasets where parallelism can help.
    """

    async def load_all_async(
        self,
        datasets: Dict[str, pd.DataFrame],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, LoadResult]:
        """
        Load all datasets asynchronously (but respecting order).

        Args:
            datasets: Dictionary of table_name -> DataFrame.
            progress_callback: Optional callback(table, current, total).

        Returns:
            Dictionary of table_name -> LoadResult.
        """
        # For now, just run synchronously since Supabase sync client
        # In future, could use async client for true async loading
        return self.load_all(datasets, progress_callback)

    async def load_table_async(
        self,
        table_name: str,
        df: pd.DataFrame,
    ) -> LoadResult:
        """
        Load a single table asynchronously.

        Args:
            table_name: Target table name.
            df: DataFrame to load.

        Returns:
            LoadResult with statistics.
        """
        # Run in executor for non-blocking behavior
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.load_table(table_name, df))
