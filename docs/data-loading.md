# Data Loading for ML Pipelines

This document describes the data loading infrastructure for E2I Causal Analytics ML pipelines.

## Overview

The data loading layer provides:

1. **MLDataLoader** - Load data from Supabase with temporal splitting
2. **DataSplitter** - Multiple splitting strategies (random, temporal, stratified, entity)
3. **DataCache** - Redis-based caching for repeated experiments
4. **SampleDataGenerator** - Generate realistic test data

All components are designed to prevent data leakage in ML pipelines.

---

## Quick Start

```python
from src.repositories import (
    MLDataLoader,
    DataSplitter,
    DataCache,
    SampleDataGenerator,
    get_ml_data_loader,
)

# Load data for training
loader = get_ml_data_loader()
dataset = await loader.load_for_training(
    table="business_metrics",
    filters={"brand": "Kisqali"},
    split_date="2024-06-01",
)

# Access splits
train_df = dataset.train
val_df = dataset.val
test_df = dataset.test

print(dataset.summary())
```

---

## MLDataLoader

The primary interface for loading ML training data from Supabase.

### Supported Tables

| Table | Description | Primary Keys |
|-------|-------------|--------------|
| `business_metrics` | KPI snapshots | metric_name, brand, date |
| `predictions` | ML predictions | model_id, target_id |
| `triggers` | HCP triggers | hcp_id, trigger_type |
| `causal_paths` | Causal relationships | source_node, target_node |
| `patient_journeys` | Patient journey data | patient_id, stage |
| `agent_activities` | Agent analysis outputs | agent_name, activity_type |

### Usage Examples

#### Basic Loading

```python
from src.repositories import get_ml_data_loader

loader = get_ml_data_loader()

# Load with default temporal split
dataset = await loader.load_for_training(
    table="business_metrics",
)

print(f"Train: {dataset.train_size} rows")
print(f"Val: {dataset.val_size} rows")
print(f"Test: {dataset.test_size} rows")
```

#### With Filters

```python
# Filter by brand and date range
dataset = await loader.load_for_training(
    table="business_metrics",
    filters={"brand": "Kisqali", "region": "US"},
    split_date="2024-06-01",
    val_days=30,
    test_days=30,
)
```

#### Select Specific Columns

```python
# Load only needed columns
dataset = await loader.load_for_training(
    table="predictions",
    columns=["target_id", "score", "rank", "created_at"],
)
```

#### Load Sample Data

```python
# Quick sample for exploration
sample_df = await loader.load_table_sample(
    table="business_metrics",
    limit=100,
    filters={"brand": "Fabhalta"},
)
```

### MLDataset Container

The `MLDataset` class wraps train/val/test DataFrames with metadata:

```python
@dataclass
class MLDataset:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict[str, Any]

    @property
    def train_size(self) -> int: ...
    @property
    def val_size(self) -> int: ...
    @property
    def test_size(self) -> int: ...
    @property
    def total_size(self) -> int: ...

    def summary(self) -> Dict[str, Any]: ...
```

---

## DataSplitter

Provides multiple splitting strategies for different ML scenarios.

### Split Strategies

| Strategy | Use Case | Prevents |
|----------|----------|----------|
| `random_split` | General ML | - |
| `temporal_split` | Time series | Future data leakage |
| `stratified_split` | Classification | Class imbalance |
| `entity_split` | Entity-level | Entity leakage |
| `combined_split` | Time series + entities | Both leakage types |

### Usage Examples

#### Random Split

```python
from src.repositories import DataSplitter, SplitConfig

splitter = DataSplitter(random_seed=42)

# Default 60/20/20 split
result = splitter.random_split(df)

# Custom ratios
config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
result = splitter.random_split(df, config)
```

#### Temporal Split (Prevents Future Leakage)

```python
# Split by date to prevent future data leakage
result = splitter.temporal_split(
    df,
    date_column="created_at",
    split_date="2024-06-01",
    val_days=30,
    test_days=30,
)

# Training data is strictly before validation
# Validation data is strictly before test
assert result.train["created_at"].max() < result.val["created_at"].min()
```

#### Stratified Split (Maintains Class Distribution)

```python
# Maintain class distribution in each split
result = splitter.stratified_split(
    df,
    stratify_column="category",
)

# Class ratios are preserved
train_dist = result.train["category"].value_counts(normalize=True)
original_dist = df["category"].value_counts(normalize=True)
```

#### Entity Split (Prevents Entity Leakage)

```python
# All records for an entity stay in the same split
result = splitter.entity_split(
    df,
    entity_column="patient_id",
)

# No patient appears in multiple splits
train_patients = set(result.train["patient_id"])
test_patients = set(result.test["patient_id"])
assert len(train_patients & test_patients) == 0
```

#### Combined Split (Temporal + Entity)

```python
# Maximum leakage prevention
result = splitter.combined_split(
    df,
    date_column="created_at",
    entity_column="patient_id",
    val_days=30,
    test_days=30,
)
```

### SplitConfig Options

```python
@dataclass
class SplitConfig:
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42
    stratify_column: Optional[str] = None
    entity_column: Optional[str] = None
    date_column: Optional[str] = None
    holdout_ratio: float = 0.0  # Optional holdout set
```

### SplitResult

```python
@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    holdout: Optional[pd.DataFrame] = None
    config: SplitConfig
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, pd.DataFrame]: ...
    def summary(self) -> Dict[str, Any]: ...
```

---

## DataCache

Redis-based caching for ML data to speed up repeated experiments.

### Usage Examples

#### Basic Caching

```python
from src.repositories import DataCache, get_data_cache

cache = get_data_cache()

# Cache a DataFrame
await cache.set("my_experiment_data", df, ttl_seconds=3600)

# Retrieve from cache
cached_df = await cache.get("my_experiment_data")
if cached_df is not None:
    print("Cache hit!")
```

#### With Decorator

```python
@cache.cached(ttl_seconds=3600)
async def load_training_data(table: str, filters: dict):
    loader = get_ml_data_loader()
    return await loader.load_for_training(table, filters=filters)

# First call loads from DB
data = await load_training_data("business_metrics", {"brand": "Kisqali"})

# Second call uses cache
data = await load_training_data("business_metrics", {"brand": "Kisqali"})
```

#### Cache Invalidation

```python
# Invalidate all cached data for a table
deleted = await cache.invalidate_table("business_metrics")
print(f"Deleted {deleted} cache entries")

# Clear all ML data cache
deleted = await cache.clear_all()
```

#### Cache Statistics

```python
stats = await cache.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Total hits: {stats['total_hits']}")
print(f"By table: {stats['tables']}")
```

### CacheConfig Options

```python
@dataclass
class CacheConfig:
    ttl_seconds: int = 3600  # 1 hour default
    prefix: str = "ml_data"
    serialize_format: str = "pickle"  # pickle or json
    compression: bool = True
```

---

## SampleDataGenerator

Generate realistic test data matching production schemas.

### Usage Examples

#### Generate Business Metrics

```python
from src.repositories import SampleDataGenerator, get_sample_generator

generator = get_sample_generator()

# Generate 1000 business metric records
df = generator.business_metrics(
    n_samples=1000,
    brands=["Kisqali", "Fabhalta"],
    regions=["US", "EU"],
)
```

#### Generate All Table Types

```python
# Predictions
predictions_df = generator.predictions(n_samples=500)

# Triggers
triggers_df = generator.triggers(n_samples=500)

# Patient journeys
journeys_df = generator.patient_journeys(n_patients=200)

# Agent activities
activities_df = generator.agent_activities(n_samples=300)

# Causal paths
paths_df = generator.causal_paths(n_samples=200)
```

#### Generate Pre-Split Dataset

```python
# Generate data already split for ML
dataset = generator.generate_ml_dataset(
    table="business_metrics",
    n_samples=1000,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
)

train_df = dataset["train"]
val_df = dataset["val"]
test_df = dataset["test"]
```

### Available Constants

```python
from src.repositories.sample_data import BRANDS, REGIONS, KPIS, AGENT_NAMES

BRANDS = ["Kisqali", "Fabhalta", "Remibrutinib"]
REGIONS = ["US", "EU", "APAC", "LATAM"]
KPIS = [
    "TRx_volume", "NRx_volume", "conversion_rate", "market_share",
    "hcp_reach", "patient_starts", "refill_rate", "abandonment_rate"
]
AGENT_NAMES = [
    "orchestrator", "causal_impact", "gap_analyzer", ...
]
```

---

## Best Practices

### 1. Always Use Temporal Splits for Time Series Data

```python
# GOOD: Prevents future data leakage
result = splitter.temporal_split(df, date_column="created_at")

# BAD: Random split can leak future information
result = splitter.random_split(df)  # Avoid for time series!
```

### 2. Use Entity Splits When Entities Have Multiple Records

```python
# GOOD: All patient records stay together
result = splitter.entity_split(df, entity_column="patient_id")

# BAD: Patient data could appear in train AND test
result = splitter.random_split(df)  # Avoid for patient data!
```

### 3. Use Caching for Repeated Experiments

```python
# GOOD: Cache expensive data loads
@cache.cached(ttl_seconds=3600)
async def load_experiment_data():
    return await loader.load_for_training(...)

# BAD: Reload data every time
async def run_experiment():
    data = await loader.load_for_training(...)  # Slow!
```

### 4. Use Sample Data for Unit Tests

```python
# GOOD: Use sample generator for deterministic tests
def test_model_training():
    generator = SampleDataGenerator(random_seed=42)
    df = generator.business_metrics(n_samples=100)
    # Tests are reproducible

# BAD: Use production data in unit tests
def test_model_training():
    data = await loader.load_for_training(...)  # Non-deterministic!
```

### 5. Check Split Metadata

```python
result = splitter.temporal_split(df, date_column="created_at")

# Always verify split boundaries
print(f"Split date: {result.metadata['split_date']}")
print(f"Val start: {result.metadata['val_start']}")
print(f"Test start: {result.metadata['test_start']}")
print(result.summary())
```

---

## Integration with Agents

### Data Preparer Agent

```python
from src.agents.ml_foundation.data_preparer import DataPreparerState
from src.repositories import get_ml_data_loader, get_data_splitter

async def prepare_data(state: DataPreparerState):
    loader = get_ml_data_loader()
    splitter = get_data_splitter()

    # Load data
    dataset = await loader.load_for_training(
        table=state.table,
        filters=state.filters,
    )

    # Apply additional splitting if needed
    if state.entity_column:
        result = splitter.entity_split(
            dataset.train,
            entity_column=state.entity_column,
        )
        state.processed_data = result

    return state
```

### Model Trainer Agent

```python
from src.agents.ml_foundation.model_trainer import ModelTrainerState
from src.repositories import MLDataset

async def train_model(state: ModelTrainerState):
    dataset: MLDataset = state.prepared_data

    # Train on train set
    model.fit(dataset.train[features], dataset.train[target])

    # Validate on val set
    val_score = model.score(dataset.val[features], dataset.val[target])

    # Final evaluation on test set
    test_score = model.score(dataset.test[features], dataset.test[target])

    return state
```

---

## Troubleshooting

### Common Issues

**1. Empty dataset returned**
```python
dataset = await loader.load_for_training(table="business_metrics")
if dataset.total_size == 0:
    # Check if table has data
    sample = await loader.load_table_sample(table="business_metrics", limit=10)
    print(f"Table has {len(sample)} rows")
```

**2. Split ratios don't sum to 1.0**
```python
# This will raise ValueError
config = SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

# Correct: ratios must sum to 1.0
config = SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
```

**3. Cache not working**
```python
# Check Redis connection
cache = get_data_cache()
stats = await cache.get_stats()
if stats.get("status") == "unavailable":
    print("Redis not connected")
```

---

## API Reference

See source files for complete API documentation:

- `src/repositories/ml_data_loader.py` - MLDataLoader, MLDataset
- `src/repositories/data_splitter.py` - DataSplitter, SplitConfig, SplitResult
- `src/repositories/data_cache.py` - DataCache, CacheConfig
- `src/repositories/sample_data.py` - SampleDataGenerator
