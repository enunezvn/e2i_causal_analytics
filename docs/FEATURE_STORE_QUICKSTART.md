# Feature Store Quick Start Guide

**Goal**: Get the E2I Feature Store running in under 10 minutes.

> **Note**: This guide covers the custom lightweight feature store. For Feast integration,
> see [Phase 13 Documentation](mlops/phase-13-feast-feature-store.md) and the
> [feature_repo/](../feature_repo/) directory.

---

## Prerequisites

Before starting, ensure you have:

- ✅ Supabase project created
- ✅ Redis running (Docker or local)
- ✅ MLflow running (Docker or local)
- ✅ Python 3.11+ with dependencies installed

---

## Step 1: Configure Environment Variables

### 1.1 Get Supabase Credentials

**Using Self-Hosted Supabase:**

E2I uses self-hosted Supabase running via Docker Compose. Access Supabase Studio at:
- Local: http://localhost:3001
- Droplet: http://138.197.4.36:3001

Configure your environment:

   ```bash
   # Self-hosted Supabase URL
   SUPABASE_URL=http://localhost:54321  # or http://138.197.4.36:54321 on droplet

   # API Keys (from self-hosted Supabase config)
   SUPABASE_ANON_KEY=your-anon-key-from-self-hosted
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-from-self-hosted  # Optional but recommended
   ```

See `config/supabase_self_hosted.example.env` for complete configuration reference.

### 1.2 Get Database Connection String

**For self-hosted Supabase:**

The PostgreSQL database is accessible directly via port 5433:

   ```bash
   # Self-hosted Supabase (local)
   DATABASE_URL=postgresql://postgres:your-password@localhost:5433/postgres

   # Self-hosted Supabase (droplet)
   DATABASE_URL=postgresql://postgres:your-password@138.197.4.36:5433/postgres
   ```

   **Note**: Port **5433** is the external mapping for the internal PostgreSQL port 5432.

### 1.3 Configure Redis and MLflow

```bash
# Redis (default if running locally)
REDIS_URL=redis://localhost:6379

# MLflow (default if running locally)
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 1.4 Update .env File

Add all variables to your `.env` file:

```bash
# Self-Hosted Supabase
SUPABASE_URL=http://localhost:54321  # or http://138.197.4.36:54321 on droplet
SUPABASE_ANON_KEY=your-anon-key-from-self-hosted
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-from-self-hosted
DATABASE_URL=postgresql://postgres:your-password@localhost:5433/postgres

# Redis
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

See `config/supabase_self_hosted.example.env` for complete configuration reference.

---

## Step 2: Start Required Services

### 2.1 Start Redis and MLflow (Docker)

```bash
# Start backend services
docker compose up redis mlflow -d

# Verify services are running
docker compose ps
```

Expected output:
```
NAME                  SERVICE    STATUS
e2i-redis             redis      running (healthy)
e2i-mlflow            mlflow     running (healthy)
```

### 2.2 Verify Services

- **Redis**: Should be accessible at `localhost:6379`
- **MLflow UI**: http://localhost:5000

---

## Step 3: Run Database Migration

The migration creates the feature store schema in Supabase.

```bash
# Make sure you're in the project root
cd /path/to/e2i_causal_analytics

# Install psycopg2 if not already installed
pip install psycopg2-binary

# Run migration
python scripts/run_migration.py database/migrations/004_create_feature_store_schema.sql
```

Expected output:
```
📄 Reading migration: database/migrations/004_create_feature_store_schema.sql
🔌 Connecting to Supabase: https://your-project.supabase.co
🔌 Connecting via psycopg2...
✅ Migration completed successfully!
```

### Verify in Supabase

1. Go to Supabase Studio → **Table Editor**
   - Local: http://localhost:3001
   - Droplet: http://138.197.4.36:3001
2. You should see new tables:
   - `feature_groups`
   - `features`
   - `feature_values`
3. Check the seed data:
   - Feature group: `hcp_demographics`
   - Feature group: `brand_performance`

---

## Step 4: Run Example Script

Test the feature store with the example script:

```bash
python scripts/feature_store_example.py
```

The script will:
1. ✅ Check environment variables
2. ✅ Connect to Supabase, Redis, and MLflow
3. ✅ Run health checks
4. ✅ Create HCP demographic features
5. ✅ Create brand performance features
6. ✅ Write sample feature values
7. ✅ Retrieve features (with caching)
8. ✅ Query historical features

Expected output:
```
================================================================================
  E2I Lightweight Feature Store - Demo & Test
================================================================================

... (detailed output showing all operations) ...

================================================================================
  Summary
================================================================================
✅ Feature store demo completed successfully!

📚 Next steps:
   1. Check MLflow UI (http://localhost:5000) for feature tracking
   2. Monitor feature freshness in Supabase
   3. Integrate with E2I agents
   4. Set up scheduled freshness updates
```

---

## Step 5: Verify Feature Tracking in MLflow

1. Open MLflow UI: http://localhost:5000
2. You should see experiments:
   - `hcp_features` (from hcp_demographics group)
3. Click on an experiment to see logged features
4. Each feature has:
   - Parameters: name, type, entity keys, version
   - Artifacts: description, computation query

---

## Quick Validation Checklist

After running the example script, verify:

- [ ] **Supabase Tables**: Check that `feature_groups`, `features`, and `feature_values` tables exist
- [ ] **Feature Data**: Verify sample HCP and brand features are in `feature_values` table
- [ ] **Redis Cache**: Run example twice - second run should be faster (cache hit)
- [ ] **MLflow Experiments**: Check that `hcp_features` experiment exists with feature runs
- [ ] **Health Check**: All services (Supabase, Redis, MLflow) show as healthy

---

## Common Issues and Solutions

### Issue 1: "DATABASE_URL not set"

**Solution**: Make sure DATABASE_URL is in your `.env` file with the correct format:
```bash
# Self-hosted Supabase
DATABASE_URL=postgresql://postgres:your-password@localhost:5433/postgres
```

### Issue 2: "Redis connection failed"

**Solution**:
```bash
# Check if Redis is running
docker compose ps redis

# If not running, start it
docker compose up redis -d

# Test connection
redis-cli ping  # Should return "PONG"
```

### Issue 3: "No module named 'psycopg2'"

**Solution**:
```bash
pip install psycopg2-binary
```

### Issue 4: "relation 'feature_groups' does not exist"

**Solution**: Migration hasn't been run yet.
```bash
python scripts/run_migration.py database/migrations/004_create_feature_store_schema.sql
```

### Issue 5: "Feature group already exists"

**Solution**: This is expected if you run the example script multiple times. The script handles this gracefully with warnings.

### Issue 6: MLflow not accessible

**Solution**:
```bash
# Check if MLflow is running
docker compose ps mlflow

# If not running, start it
docker compose up mlflow -d

# Access UI
open http://localhost:5000
```

---

## Next Steps

### 1. Integrate with E2I Agents

Example: Gap Analyzer Agent

```python
from src.feature_store import FeatureStoreClient
from src.agents.gap_analyzer.agent import GapAnalyzerAgent

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

### 2. Set Up Scheduled Freshness Updates

Add a cron job or scheduled task to update feature freshness:

```python
# In your scheduler (e.g., Celery, APScheduler)
from src.feature_store import FeatureStoreClient

def update_freshness():
    fs = FeatureStoreClient(...)
    fs.update_freshness_status()
    fs.close()

# Run daily
schedule.every().day.at("02:00").do(update_freshness)
```

### 3. Create Production Feature Pipelines

Example: Daily brand metrics update

```python
from datetime import datetime
from src.feature_store import FeatureStoreClient

def update_brand_metrics():
    fs = FeatureStoreClient(...)

    # Fetch latest metrics from your data sources
    brand_metrics = fetch_brand_metrics()  # Your data pipeline

    # Prepare feature values
    feature_values = []
    for metric in brand_metrics:
        feature_values.append({
            "feature_name": "total_nrx_30d",
            "entity_values": {
                "brand_id": metric["brand_id"],
                "geography_id": metric["geography_id"]
            },
            "value": metric["nrx_count"],
            "event_timestamp": datetime.utcnow(),
            "feature_group": "brand_performance"
        })

    # Batch write
    count = fs.write_batch_features(feature_values)
    print(f"Updated {count} brand metrics")

    fs.close()
```

### 4. Monitor Feature Quality

Use the `feature_freshness_monitor` view:

```sql
-- In Supabase SQL Editor
SELECT
    feature_group,
    feature_name,
    fresh_count,
    total_entities,
    ROUND(100.0 * fresh_count / NULLIF(total_entities, 0), 2) as fresh_percentage,
    time_since_last_event
FROM feature_freshness_monitor
ORDER BY fresh_percentage ASC;
```

---

## Resources

- **Full Documentation**: `docs/FEATURE_STORE.md`
- **Database Schema**: `database/migrations/004_create_feature_store_schema.sql`
- **Python Client**: `src/feature_store/client.py`
- **Example Script**: `scripts/feature_store_example.py`
- **MLflow UI**: http://localhost:5000
- **Supabase Studio**: http://localhost:3001 (local) or http://138.197.4.36:3001 (droplet)

---

## Support

If you encounter issues:

1. Check the **Common Issues** section above
2. Review logs: `docker compose logs mlflow redis`
3. Verify environment variables: `cat .env | grep -E "SUPABASE|REDIS|MLFLOW|DATABASE"`
4. Run health check:
   ```python
   from src.feature_store import FeatureStoreClient
   fs = FeatureStoreClient(...)
   print(fs.health_check())
   ```

---

**Last Updated**: 2025-12-18
**Version**: 1.0.0
**Status**: ✅ Production-Ready
