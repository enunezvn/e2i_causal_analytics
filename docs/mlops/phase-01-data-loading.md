# Phase 1: Data Loading Foundation

**Goal**: Enable data loading from Supabase for ML pipelines

**Status**: Complete

**Dependencies**: None (first phase)

---

## Tasks

- [x] **Task 1.1**: Create `src/repositories/ml_data_loader.py`
  - Supabase data extraction for ML tables
  - Support for business_metrics, predictions, triggers, causal_paths, patient_journeys, agent_activities
  - Async interface with MLDataset container

- [x] **Task 1.2**: Add data split utilities
  - Train/validation/test splits
  - Temporal awareness (no future data leakage)
  - Stratified splitting for classification tasks
  - Entity-level splitting to prevent entity leakage
  - Combined temporal + entity splitting

- [x] **Task 1.3**: Implement data caching layer
  - Redis-based caching for repeated experiments
  - Cache invalidation on data updates
  - Configurable TTL
  - Decorator for automatic caching

- [x] **Task 1.4**: Create sample data generators
  - Generate realistic test data
  - Match schema of production tables
  - Configurable sample sizes
  - Support for all 6 table types

- [x] **Task 1.5**: Add unit tests for data loading
  - Test data extraction
  - Test splitting logic
  - Test caching behavior
  - Comprehensive test coverage

- [x] **Task 1.6**: Document data loading patterns
  - Usage examples
  - Configuration options
  - Best practices
  - Integration patterns

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/repositories/ml_data_loader.py` | Created | Main data loading module with MLDataLoader, MLDataset |
| `src/repositories/data_splitter.py` | Created | Train/val/test splitting with 5 strategies |
| `src/repositories/data_cache.py` | Created | Redis caching layer with decorator |
| `src/repositories/sample_data.py` | Created | Sample data generators for 6 table types |
| `src/repositories/__init__.py` | Modified | Export new ML data loading modules |
| `tests/unit/test_repositories/test_ml_data_loader.py` | Created | Unit tests for MLDataLoader |
| `tests/unit/test_repositories/test_data_splitter.py` | Created | Unit tests for DataSplitter |
| `docs/data-loading.md` | Created | Comprehensive documentation |

---

## Technical Notes

### Data Loading Patterns

```python
# Example usage pattern
from src.repositories import MLDataLoader, get_ml_data_loader

loader = get_ml_data_loader()
dataset = await loader.load_for_training(
    table="business_metrics",
    filters={"brand": "Kisqali"},
    split_date="2024-06-01"
)

train, val, test = dataset.train, dataset.val, dataset.test
```

### Split Strategies Implemented

1. **Random Split** - Simple random partitioning
2. **Temporal Split** - Date-based to prevent future leakage
3. **Stratified Split** - Maintains class distribution
4. **Entity Split** - All entity records in same split (hash-based)
5. **Combined Split** - Temporal + Entity for maximum protection

### Temporal Split Strategy

- Training: data before split_date - val_days - test_days
- Validation: (split_date - val_days) to (split_date - test_days)
- Test: (split_date - test_days) to split_date

### Key Design Decisions

1. **MLDataset Container**: Wraps train/val/test DataFrames with metadata
2. **Hash-based Entity Assignment**: Deterministic entity splits across runs
3. **Async-first**: All data loading is async for performance
4. **Factory Functions**: `get_ml_data_loader()`, `get_data_splitter()`, etc.
5. **Configurable Splits**: SplitConfig dataclass for all split parameters

---

## Progress Log

| Date | Update |
|------|--------|
| 2024-12-22 | Phase created |
| 2024-12-22 | Task 1.1 completed - ml_data_loader.py |
| 2024-12-22 | Task 1.2 completed - data_splitter.py |
| 2024-12-22 | Task 1.3 completed - data_cache.py |
| 2024-12-22 | Task 1.4 completed - sample_data.py |
| 2024-12-22 | Task 1.5 completed - unit tests |
| 2024-12-22 | Task 1.6 completed - documentation |
| 2024-12-22 | **Phase 1 Complete** |

---

## Blockers

None - Phase completed successfully.

---

## Next Phase

**Phase 2: Data Preparer Agent Completion** - See `phase-02-data-preparer.md`
