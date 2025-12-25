# Feast Feature Store Troubleshooting Guide

**Version**: 1.0
**Last Updated**: 2025-12-25
**Applies To**: E2I Causal Analytics Platform v4.3+

---

## Overview

This guide covers common issues with Feast Feature Store integration across the three ML agents:
- **data_preparer**: Feature registration
- **model_trainer**: Historical feature retrieval (point-in-time joins)
- **prediction_synthesizer**: Online feature serving

---

## Quick Diagnostics

### Check Feast Availability

```python
# Test Feast client connection
from src.feature_store.feast_client import FeastClient

client = FeastClient()
print(f"Feast available: {client.is_available}")
print(f"Feature views: {client.list_feature_views()}")
```

### Check Agent Integration

```python
# Test FeastFeatureStore adapter (prediction_synthesizer)
from src.agents.prediction_synthesizer.nodes import get_feast_feature_store

store = get_feast_feature_store()
print(f"Adapter available: {store.is_available}")
```

---

## Common Issues

### 1. Feast Connection Failures

**Symptom**: `ConnectionError` or `Could not initialize FeatureAnalyzerAdapter`

**Cause**: Feast offline/online store not accessible

**Solutions**:

```bash
# Check Feast registry path
echo $FEAST_REPO_PATH  # Should be /home/claude/e2i/feature_repo

# Verify feature repository exists
ls -la $FEAST_REPO_PATH/feature_store.yaml

# Apply feature definitions
cd $FEAST_REPO_PATH
feast apply
```

**Graceful Degradation**: All agents are designed to continue without Feast:

```python
# Agents return empty results when Feast unavailable
features = await feast_store.get_online_features("hcp_123")
# Returns {} instead of raising exception
```

---

### 2. Point-in-Time Join Errors (model_trainer)

**Symptom**: `ValueError: entity_df must contain event_timestamp column`

**Cause**: Missing timestamp column for temporal joins

**Solution**:

```python
# CORRECT: Include event_timestamp in entity_df
entity_df = pd.DataFrame({
    "hcp_id": ["001", "002", "003"],
    "event_timestamp": pd.to_datetime([
        "2024-01-15",
        "2024-01-20",
        "2024-01-25"
    ]),
})

# Get historical features with point-in-time correctness
features = feast_client.get_historical_features(
    entity_df=entity_df,
    features=["hcp_features:call_frequency", "hcp_features:prescription_count"],
)
```

---

### 3. Stale Features Warning (prediction_synthesizer)

**Symptom**: `Stale features detected: call_frequency, prescription_count`

**Cause**: Features not materialized to online store recently

**Solutions**:

```bash
# Run materialization
cd $FEAST_REPO_PATH
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)

# Or trigger via Celery task
python -c "from src.tasks.feast_tasks import materialize_features; materialize_features.delay()"
```

**Check Freshness Threshold**:

```python
# Default is 24 hours - adjust if needed
store = FeastFeatureStore(
    adapter=adapter,
    default_feature_view="hcp_features",
    entity_key="hcp_id",
)

# Check freshness with custom threshold
freshness = await store.check_feature_freshness(
    entity_id="hcp_123",
    max_staleness_hours=48.0,  # Allow up to 48 hours
)
```

---

### 4. Feature View Not Found

**Symptom**: `KeyError: 'hcp_features'` or `Feature view not found`

**Cause**: Feature view not registered or wrong name

**Solutions**:

```bash
# List registered feature views
cd $FEAST_REPO_PATH
feast feature-views list

# Check feature view definition
cat feature_repo/features/hcp_features.py
```

```python
# Use correct feature view name
store = FeastFeatureStore(
    default_feature_view="hcp_engagement_features",  # Match actual name
    entity_key="hcp_id",
)
```

---

### 5. Entity Key Mismatch

**Symptom**: `KeyError: 'hcp_id'` in returned features

**Cause**: Entity key in query doesn't match Feast entity definition

**Solutions**:

```python
# Check entity definition
from feast import Entity
# Entity defined as: Entity(name="hcp", join_keys=["hcp_id"])

# Use matching key
store = FeastFeatureStore(
    entity_key="hcp_id",  # Must match join_key in entity definition
)
```

---

### 6. Online Store Empty

**Symptom**: `get_online_features` returns empty dict for valid entities

**Cause**: Features not materialized to online store

**Solutions**:

```bash
# Check online store status
cd $FEAST_REPO_PATH
feast materialize 2024-01-01T00:00:00 $(date +%Y-%m-%dT%H:%M:%S)

# Verify entity exists in online store
python -c "
from feast import FeatureStore
store = FeatureStore(repo_path='$FEAST_REPO_PATH')
features = store.get_online_features(
    features=['hcp_features:call_frequency'],
    entity_rows=[{'hcp_id': 'hcp_123'}]
)
print(features.to_dict())
"
```

---

### 7. Feature Registration Failures (data_preparer)

**Symptom**: `FeastRegistrarNode` fails after QC passes

**Cause**: Invalid feature data or schema mismatch

**Solutions**:

```python
# Validate feature data before registration
from src.agents.ml_foundation.data_preparer.nodes.feast_registrar import (
    FeastRegistrarNode,
)

# Check required columns
required = ["hcp_id", "event_timestamp"]
missing = [col for col in required if col not in features_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Check data types
assert features_df["hcp_id"].dtype == "object"  # String entity ID
assert pd.api.types.is_datetime64_any_dtype(features_df["event_timestamp"])
```

---

### 8. Import/Circular Dependency Errors

**Symptom**: `ImportError` when importing Feast-related modules

**Cause**: Circular imports between feature_store and agent modules

**Solution**: Use lazy loading pattern:

```python
# CORRECT: Lazy import in feast_feature_store.py
def _get_feature_analyzer_adapter():
    """Get FeatureAnalyzerAdapter (lazy import to avoid circular deps)."""
    try:
        from src.feature_store.client import FeatureStoreClient
        from src.feature_store.feature_analyzer_adapter import (
            get_feature_analyzer_adapter,
        )
        fs_client = FeatureStoreClient()
        return get_feature_analyzer_adapter(
            feature_store_client=fs_client,
            enable_feast=True,
        )
    except Exception as e:
        logger.warning(f"Could not initialize FeatureAnalyzerAdapter: {e}")
        return None
```

---

## Testing Feast Integration

### Unit Tests

```bash
# Test all Feast-related components
pytest tests/unit/test_feature_store/ -v
pytest tests/unit/test_agents/test_ml_foundation/test_data_preparer/ -v -k feast
pytest tests/unit/test_agents/test_ml_foundation/test_model_trainer/ -v -k feast
pytest tests/unit/test_agents/test_prediction_synthesizer/test_feast_feature_store.py -v
```

### Integration Tests

```bash
# Full pipeline test with Feast
pytest tests/integration/test_tier0_pipeline.py -v -k feast
```

### Memory-Safe Testing

```bash
# ALWAYS use 4 workers max (per CLAUDE.md)
make test  # Uses default 4 workers

# NEVER use:
# pytest -n auto  # Can spawn 14 workers, exhausts RAM
```

---

## Monitoring & Observability

### Feast Metrics

```python
# Track Feast operations via logging
import logging
logging.getLogger("feast").setLevel(logging.DEBUG)
```

### Agent Metrics

All Feast operations emit spans with:
- `feast.operation`: registration, historical_retrieval, online_retrieval
- `feast.entity_count`: Number of entities processed
- `feast.feature_count`: Number of features retrieved
- `feast.latency_ms`: Operation latency

---

## Configuration Reference

### Environment Variables

```bash
# Feast repository path
FEAST_REPO_PATH=/home/claude/e2i/feature_repo

# Feature materialization config
FEAST_MATERIALIZE_INTERVAL_HOURS=6
FEAST_MAX_STALENESS_HOURS=24
```

### Agent Configuration

```yaml
# config/agents/prediction_synthesizer.yaml
feast:
  enabled: true
  default_feature_view: hcp_features
  entity_key: hcp_id
  max_staleness_hours: 24.0
  graceful_degradation: true
```

---

## Support

For additional help:
1. Check Feast documentation: https://docs.feast.dev/
2. Review agent CLAUDE.md files for integration specifics
3. Check `.claude/specialists/MLOps_Integration/mlops_integration.md` for architecture
4. See `.claude/contracts/tier0-contracts.md` for data flow contracts

---

**End of Troubleshooting Guide**
