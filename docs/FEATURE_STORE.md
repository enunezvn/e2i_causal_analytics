# E2I Feature Store

**Status**: Production-Ready ✅
**Created**: 2025-12-18
**Updated**: 2025-12-22 (Feast Integration)
**Architecture**: Supabase (PostgreSQL) + Redis + MLflow + Feast

---

## Overview

E2I uses a **hybrid feature store architecture** combining a custom lightweight implementation with Feast 0.58.0 for advanced capabilities.

### Hybrid Architecture

| Layer | Custom Store | Feast |
|-------|--------------|-------|
| **Primary Use** | Simple feature serving | Point-in-time joins, versioning |
| **Offline Store** | Supabase PostgreSQL | Supabase PostgreSQL |
| **Online Store** | Redis (1hr TTL) | Redis |
| **Feature Registry** | Basic metadata | Full registry with statistics |
| **Agent Integration** | Direct LangGraph | Via adapter |

### When to Use Which

| Use Case | Recommended |
|----------|-------------|
| Simple online feature lookup | Custom Store |
| Training data with point-in-time joins | Feast |
| Feature versioning/time-travel | Feast |
| Real-time agent features | Custom Store |
| Feature statistics/monitoring | Feast |

> **Note**: The system uses Feast as primary with Custom Store as fallback.
> See [Phase 13 Documentation](mlops/phase-13-feast-feature-store.md) for migration details.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  E2I Feature Store                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   Supabase   │    │    Redis     │    │  MLflow  │ │
│  │ (PostgreSQL) │◄───┤   (Cache)    │    │(Tracking)│ │
│  │              │    │              │    │          │ │
│  │  - Features  │    │  - Online    │    │  - Feat  │ │
│  │  - Values    │    │    Serving   │    │    Defs  │ │
│  │  - Metadata  │    │  - 1hr TTL   │    │  - Stats │ │
│  └──────────────┘    └──────────────┘    └──────────┘ │
│         ▲                    ▲                  ▲       │
│         │                    │                  │       │
│         └────────────────────┴──────────────────┘       │
│                   FeatureStoreClient                    │
└─────────────────────────────────────────────────────────┘
```

### Components

1. **Supabase PostgreSQL**: Offline storage for:
   - Feature metadata (groups, definitions)
   - Time-series feature values
   - Freshness tracking

2. **Redis**: Online serving with:
   - 1-hour default TTL
   - Automatic cache invalidation on writes
   - Sub-millisecond latency

3. **MLflow**: Feature tracking for:
   - Feature definitions
   - Feature statistics
   - Version tracking

---

## Database Schema

### Tables

#### `feature_groups`
Logical grouping of related features.

```sql
- id: UUID
- name: VARCHAR(255) UNIQUE
- description: TEXT
- owner: VARCHAR(100)
- tags: JSONB
- source_table: VARCHAR(255)
- expected_update_frequency_hours: INTEGER
- max_age_hours: INTEGER
- mlflow_experiment_id: VARCHAR(255)
```

#### `features`
Individual feature definitions.

```sql
- id: UUID
- feature_group_id: UUID (FK)
- name: VARCHAR(255)
- value_type: feature_value_type ENUM
- entity_keys: JSONB (e.g., ["hcp_id"])
- computation_query: TEXT
- statistics: JSONB
- mlflow_run_id: VARCHAR(255)
```

#### `feature_values`
Time-series feature value storage.

```sql
- id: UUID
- feature_id: UUID (FK)
- entity_values: JSONB (e.g., {"hcp_id": "HCP123"})
- value: JSONB (any JSON-serializable type)
- event_timestamp: TIMESTAMPTZ
- freshness_status: ENUM('fresh', 'stale', 'expired')
```

### Views

- `feature_values_latest`: Most recent value per entity (online serving)
- `feature_freshness_monitor`: Monitoring freshness across all features

---

## Installation

### 1. Run Database Migration

```bash
# Apply schema migration to Supabase
python scripts/run_migration.py database/migrations/004_create_feature_store_schema.sql
```

### 2. Verify Services

Ensure these services are running:
- ✅ Supabase (PostgreSQL)
- ✅ Redis (port 6379)
- ✅ MLflow (port 5000)

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps
```

---

## Quick Start

### Initialize Client

```python
from src.feature_store import FeatureStoreClient
import os

# Initialize feature store
fs = FeatureStoreClient(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_ANON_KEY"),
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    mlflow_tracking_uri="http://localhost:5000",
    cache_ttl_seconds=3600,  # 1 hour
)

# Check health
health = fs.health_check()
print(health)  # {'supabase': True, 'redis': True, 'mlflow': True}
```

### Create Feature Group

```python
# Create HCP demographics feature group
hcp_demographics = fs.create_feature_group(
    name="hcp_demographics",
    description="Healthcare provider demographic features",
    owner="data-team",
    source_table="hcps",
    expected_update_frequency_hours=168,  # Weekly
    max_age_hours=720,  # 30 days
    tags=["demographics", "hcp", "core"],
    mlflow_experiment_name="hcp_features"
)
```

### Define Features

```python
# Create specialty feature
specialty_feature = fs.create_feature(
    feature_group_name="hcp_demographics",
    name="specialty",
    value_type="string",
    entity_keys=["hcp_id"],
    description="Primary medical specialty of HCP",
    owner="data-team",
    tags=["categorical"]
)

# Create years in practice feature
years_feature = fs.create_feature(
    feature_group_name="hcp_demographics",
    name="years_in_practice",
    value_type="int64",
    entity_keys=["hcp_id"],
    description="Number of years HCP has been practicing",
    owner="data-team",
    tags=["numerical"]
)
```

### Write Feature Values

```python
from datetime import datetime

# Write single feature value
fs.write_feature_value(
    feature_name="specialty",
    entity_values={"hcp_id": "HCP123"},
    value="Oncology",
    event_timestamp=datetime.utcnow(),
    feature_group="hcp_demographics"
)

# Write batch feature values
feature_values = [
    {
        "feature_name": "specialty",
        "entity_values": {"hcp_id": "HCP456"},
        "value": "Cardiology",
        "event_timestamp": datetime.utcnow(),
    },
    {
        "feature_name": "years_in_practice",
        "entity_values": {"hcp_id": "HCP456"},
        "value": 15,
        "event_timestamp": datetime.utcnow(),
    }
]

count = fs.write_batch_features(feature_values)
print(f"Wrote {count} features")
```

### Retrieve Features (Online Serving)

```python
# Get all features for an HCP
entity_features = fs.get_entity_features(
    entity_values={"hcp_id": "HCP123"},
    feature_group="hcp_demographics",
    use_cache=True  # Uses Redis cache
)

print(entity_features.features)
# {'specialty': 'Oncology', 'years_in_practice': 12}

print(entity_features.metadata)
# {'specialty': {'feature_group': 'hcp_demographics', 'event_timestamp': '...', 'freshness_status': 'fresh'}}

# Convert to flat dict for ML models
feature_dict = entity_features.to_dict()
# {'hcp_id': 'HCP123', 'specialty': 'Oncology', 'years_in_practice': 12}
```

### Get Historical Features

```python
from datetime import datetime, timedelta

# Get feature history for the last 30 days
historical = fs.get_historical_features(
    entity_values={"hcp_id": "HCP123"},
    feature_names=["specialty", "years_in_practice"],
    start_time=datetime.utcnow() - timedelta(days=30),
    end_time=datetime.utcnow()
)

for record in historical:
    print(f"{record['feature_name']}: {record['value']} at {record['event_timestamp']}")
```

---

## E2I Use Cases

### 1. HCP Targeting Features

```python
# Create feature group
fs.create_feature_group(
    name="hcp_targeting",
    description="Features for HCP targeting models",
    owner="ml-team",
    expected_update_frequency_hours=24,  # Daily
    tags=["targeting", "ml"]
)

# Define features
fs.create_feature("hcp_targeting", "total_trx_30d", "int64", ["hcp_id"])
fs.create_feature("hcp_targeting", "brand_affinity_score", "float64", ["hcp_id", "brand_id"])
fs.create_feature("hcp_targeting", "last_rep_visit_days_ago", "int64", ["hcp_id"])
```

### 2. Brand Performance Features

```python
# Create feature group
fs.create_feature_group(
    name="brand_performance",
    description="Brand-level performance metrics",
    owner="analytics-team",
    expected_update_frequency_hours=24,
    tags=["brand", "performance"]
)

# Define features
fs.create_feature("brand_performance", "total_nrx_30d", "int64", ["brand_id", "geography_id"])
fs.create_feature("brand_performance", "market_share", "float64", ["brand_id", "geography_id"])
fs.create_feature("brand_performance", "growth_rate_qoq", "float64", ["brand_id"])
```

### 3. Causal Inference Features

```python
# Create feature group for causal features
fs.create_feature_group(
    name="causal_features",
    description="Computed causal relationships",
    owner="causal-team",
    expected_update_frequency_hours=168,  # Weekly
    tags=["causal", "treatment_effects"]
)

# Features computed by Causal Impact Agent
fs.create_feature("causal_features", "rep_visit_ate", "float64", ["brand_id", "geography_id"])
fs.create_feature("causal_features", "sample_impact_cate", "float64", ["hcp_id", "brand_id"])
```

---

## Integration with E2I Agents

### Gap Analyzer Agent

```python
# In src/agents/gap_analyzer/agent.py
from src.feature_store import FeatureStoreClient

class GapAnalyzerAgent:
    def __init__(self):
        self.feature_store = FeatureStoreClient(...)

    def analyze_gaps(self, brand_id: str):
        # Get brand performance features
        brand_features = self.feature_store.get_entity_features(
            entity_values={"brand_id": brand_id},
            feature_group="brand_performance"
        )

        # Use features for gap detection
        market_share = brand_features.features.get("market_share", 0)
        growth_rate = brand_features.features.get("growth_rate_qoq", 0)

        # ... gap analysis logic ...
```

### Prediction Synthesizer Agent

```python
# In src/agents/prediction_synthesizer/agent.py

class PredictionSynthesizerAgent:
    def predict_churn(self, hcp_id: str):
        # Retrieve HCP features
        hcp_features = self.feature_store.get_entity_features(
            entity_values={"hcp_id": hcp_id},
            feature_group="hcp_targeting"
        )

        # Convert to model input
        X = hcp_features.to_dict()

        # Make prediction
        prediction = self.model.predict([X])

        return prediction
```

---

## Caching Strategy

### Cache Keys

Format: `fs:{entity_hash}:fg:{feature_group}:fn:{feature_names_hash}`

Example:
```
fs:a1b2c3d4:fg:hcp_demographics:fn:e5f6g7h8
```

### Cache Invalidation

Automatic invalidation on:
- ✅ New feature value write
- ✅ Batch feature write
- ✅ Manual invalidation via `retriever.invalidate_cache()`

### Cache Performance

- **Hit Rate Target**: >80%
- **Latency**: <1ms (cache hit), <50ms (cache miss)
- **TTL**: 1 hour (configurable)

---

## Monitoring

### Freshness Monitoring

```python
# Check feature freshness
freshness = fs.supabase.table("feature_freshness_monitor").select("*").execute()

for record in freshness.data:
    print(f"{record['feature_group']}.{record['feature_name']}")
    print(f"  Fresh: {record['fresh_count']}/{record['total_entities']}")
    print(f"  Last event: {record['time_since_last_event']}")
```

### Update Freshness Status

```python
# Run freshness update (should be scheduled daily)
fs.update_freshness_status()
```

---

## MLflow Integration

All feature definitions are automatically logged to MLflow:

- **Parameters**: name, type, entity keys, version
- **Artifacts**: description, computation query
- **Experiments**: One per feature group

**View in MLflow UI**: http://localhost:5000

---

## Performance

### Benchmarks

- **Online Serving** (Redis cache hit): <1ms
- **Online Serving** (cache miss): <50ms
- **Batch Write** (1000 features): <2s
- **Historical Query** (30 days): <500ms

### Optimization Tips

1. **Use batch writes** for bulk inserts
2. **Enable caching** for frequently accessed features
3. **Set appropriate TTLs** based on update frequency
4. **Use feature groups** to organize related features
5. **Monitor freshness** to ensure data quality

---

## Best Practices

### 1. Feature Naming

```python
# ✅ Good: Descriptive, includes aggregation window
"total_trx_30d", "avg_nrx_7d", "last_visit_days_ago"

# ❌ Bad: Ambiguous, no context
"total", "average", "last"
```

### 2. Entity Keys

```python
# ✅ Good: Clear entity identification
entity_keys=["hcp_id"]
entity_keys=["brand_id", "geography_id"]

# ❌ Bad: Too generic
entity_keys=["id"]
```

### 3. Feature Groups

```python
# ✅ Good: Logical grouping by domain
"hcp_demographics", "brand_performance", "causal_features"

# ❌ Bad: Too broad or too narrow
"all_features", "feature_1"
```

### 4. Update Frequency

Match feature update frequency to business requirements:
- **Real-time**: Rep visit events → hourly
- **Daily**: Brand metrics → 24 hours
- **Weekly**: Causal relationships → 168 hours

---

## API Reference

See:
- `src/feature_store/client.py` - Main client API
- `src/feature_store/models.py` - Data models
- `src/feature_store/retrieval.py` - Retrieval logic
- `src/feature_store/writer.py` - Write logic

---

## Troubleshooting

### Issue: Features not found

```python
# Check feature exists
feature = fs.get_feature("hcp_demographics", "specialty")
if not feature:
    print("Feature not defined!")
```

### Issue: Stale features

```python
# Update freshness status
fs.update_freshness_status()

# Check freshness monitor
freshness = fs.supabase.table("feature_freshness_monitor").select("*").execute()
```

### Issue: Cache not working

```python
# Check Redis connection
health = fs.health_check()
print(health["redis"])  # Should be True

# Manually invalidate cache
fs.retriever.invalidate_cache({"hcp_id": "HCP123"})
```

---

## Future Enhancements

**v1.1** (Short-term):
- [ ] Feature schema validation
- [ ] Automatic drift detection
- [ ] Feature lineage tracking

**v2.0** (Medium-term):
- [ ] Point-in-time correctness for training data
- [ ] Feature transformation pipelines
- [ ] Advanced statistics computation

**v3.0** (Long-term):
- [ ] Integration with feature discovery UI
- [ ] Automated feature engineering
- [ ] Multi-region support

---

## Resources

- **Migration**: `database/migrations/004_create_feature_store_schema.sql`
- **Client Code**: `src/feature_store/`
- **Example Usage**: `scripts/feature_store_example.py`
- **MLflow UI**: http://localhost:5000
- **Supabase Dashboard**: https://your-project.supabase.co

---

**Last Updated**: 2025-12-18
**Version**: 1.0.0
**Status**: ✅ Production-Ready
