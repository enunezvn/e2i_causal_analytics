"""Data loader node for data_preparer agent.

This node loads data from Supabase using MLDataLoader from Phase 1.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

# Use direct module imports to avoid circular import with src.repositories
from src.repositories.data_splitter import DataSplitter, get_data_splitter
from src.repositories.ml_data_loader import MLDataLoader, get_ml_data_loader
from src.repositories.sample_data import SampleDataGenerator

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


async def load_data(state: DataPreparerState) -> Dict[str, Any]:
    """Load and split data for ML training.

    This node:
    1. Extracts data source configuration from scope_spec
    2. Loads data from Supabase using MLDataLoader
    3. Applies appropriate splitting strategy (temporal, entity, or combined)
    4. Populates train_df, validation_df, test_df, holdout_df in state

    Args:
        state: Current agent state

    Returns:
        Updated state with loaded data splits
    """
    start_time = datetime.now()
    experiment_id = state.get("experiment_id", "unknown")
    logger.info(f"Loading data for experiment {experiment_id}")

    try:
        scope_spec = state.get("scope_spec", {})
        data_source = state.get("data_source") or scope_spec.get("data_source", "business_metrics")

        # Extract configuration from scope_spec
        filters = scope_spec.get("filters", {})
        date_column = scope_spec.get("date_column", "created_at")
        entity_column = scope_spec.get("entity_column")
        split_date = scope_spec.get("split_date")
        val_days = scope_spec.get("val_days", 30)
        test_days = scope_spec.get("test_days", 30)
        use_sample_data = scope_spec.get("use_sample_data", False)

        # Check if we should use sample data (for testing/development)
        if use_sample_data:
            logger.info("Using sample data generator")
            dataset = await _load_sample_data(
                data_source=data_source,
                n_samples=scope_spec.get("sample_size", 1000),
                entity_column=entity_column,
                date_column=date_column,
            )
        else:
            # Load from Supabase
            dataset = await _load_from_supabase(
                data_source=data_source,
                filters=filters,
                date_column=date_column,
                entity_column=entity_column,
                split_date=split_date,
                val_days=val_days,
                test_days=test_days,
            )

        # Calculate loading duration
        load_duration = (datetime.now() - start_time).total_seconds()

        # Prepare update
        updates = {
            "train_df": dataset["train"],
            "validation_df": dataset["val"],
            "test_df": dataset["test"],
            "holdout_df": dataset.get("holdout"),
        }

        logger.info(
            f"Data loaded successfully: "
            f"train={len(dataset['train'])}, "
            f"val={len(dataset['val'])}, "
            f"test={len(dataset['test'])}, "
            f"duration={load_duration:.2f}s"
        )

        return updates

    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "data_loading_error",
            "blocking_issues": [f"Data loading failed: {str(e)}"],
        }


async def _load_from_supabase(
    data_source: str,
    filters: Dict[str, Any],
    date_column: str,
    entity_column: Optional[str],
    split_date: Optional[str],
    val_days: int,
    test_days: int,
) -> Dict[str, Any]:
    """Load data from Supabase and split.

    Args:
        data_source: Table name
        filters: Query filters
        date_column: Date column for temporal splits
        entity_column: Entity column for entity-level splits
        split_date: Reference date for temporal split
        val_days: Days for validation set
        test_days: Days for test set

    Returns:
        Dict with train, val, test, holdout DataFrames
    """
    loader = get_ml_data_loader()

    # Load with temporal split
    dataset = await loader.load_for_training(
        table=data_source,
        filters=filters,
        date_column=date_column,
        split_date=split_date,
        val_days=val_days,
        test_days=test_days,
    )

    result = {
        "train": dataset.train,
        "val": dataset.val,
        "test": dataset.test,
        "holdout": None,
    }

    # If entity column specified, apply entity-level split to ensure no leakage
    if entity_column and entity_column in dataset.train.columns:
        logger.info(f"Applying entity-level split on column: {entity_column}")
        splitter = get_data_splitter()

        # Combined temporal + entity split
        # This is already temporal, now ensure entity integrity
        combined_result = splitter.combined_split(
            dataset.train.append(dataset.val).append(dataset.test),
            date_column=date_column,
            entity_column=entity_column,
            split_date=split_date,
            val_days=val_days,
            test_days=test_days,
        )

        result = {
            "train": combined_result.train,
            "val": combined_result.val,
            "test": combined_result.test,
            "holdout": combined_result.holdout,
        }

    return result


async def _load_sample_data(
    data_source: str,
    n_samples: int,
    entity_column: Optional[str],
    date_column: str,
) -> Dict[str, Any]:
    """Load sample data for testing/development.

    Args:
        data_source: Table name to emulate
        n_samples: Number of samples to generate
        entity_column: Entity column for entity-level splits
        date_column: Date column for temporal splits

    Returns:
        Dict with train, val, test DataFrames
    """
    generator = SampleDataGenerator(seed=42)
    splitter = get_data_splitter(random_seed=42)

    # Generate sample data based on table type
    if data_source == "business_metrics":
        df = generator.business_metrics(n_samples=n_samples)
    elif data_source == "predictions":
        df = generator.predictions(n_samples=n_samples)
    elif data_source == "triggers":
        df = generator.triggers(n_samples=n_samples)
    elif data_source == "patient_journeys":
        # Use ml_patients() for ML-ready patient data with discontinuation_flag
        df = generator.ml_patients(n_patients=n_samples)
    elif data_source == "agent_activities":
        df = generator.agent_activities(n_samples=n_samples)
    elif data_source == "causal_paths":
        df = generator.causal_paths(n_samples=n_samples)
    else:
        # Default to business metrics
        df = generator.business_metrics(n_samples=n_samples)

    # Apply temporal split if date column exists
    if date_column in df.columns:
        result = splitter.temporal_split(
            df,
            date_column=date_column,
            val_days=30,
            test_days=30,
        )
    elif entity_column and entity_column in df.columns:
        result = splitter.entity_split(df, entity_column=entity_column)
    else:
        result = splitter.random_split(df)

    return {
        "train": result.train,
        "val": result.val,
        "test": result.test,
        "holdout": result.holdout,
    }
