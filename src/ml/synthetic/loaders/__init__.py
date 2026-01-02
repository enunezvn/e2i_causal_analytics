"""
E2I Synthetic Data Loaders

Batch loaders for Supabase:
- BatchLoader: Load data in batches with validation
- AsyncBatchLoader: Async version for concurrent loading

Statistics utilities:
- get_dataset_stats: Get comprehensive dataset statistics
- validate_supabase_data: Validate data for Supabase compatibility
"""

from .batch_loader import (
    BatchLoader,
    AsyncBatchLoader,
    LoadResult,
    LoaderConfig,
    LOADING_ORDER,
    TABLE_COLUMNS,
)

from .stats import (
    get_dataset_stats,
    get_all_datasets_stats,
    get_column_stats,
    get_split_stats,
    validate_supabase_data,
    print_dataset_summary,
    DatasetStats,
    ColumnStats,
    SplitStats,
)

__all__ = [
    # Batch loader
    "BatchLoader",
    "AsyncBatchLoader",
    "LoadResult",
    "LoaderConfig",
    "LOADING_ORDER",
    "TABLE_COLUMNS",
    # Stats
    "get_dataset_stats",
    "get_all_datasets_stats",
    "get_column_stats",
    "get_split_stats",
    "validate_supabase_data",
    "print_dataset_summary",
    "DatasetStats",
    "ColumnStats",
    "SplitStats",
]
