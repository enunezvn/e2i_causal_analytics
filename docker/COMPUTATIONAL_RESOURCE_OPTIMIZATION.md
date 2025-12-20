# Computational Resource Optimization Analysis

**Date:** 2025-12-17
**Issue:** Current worker resource limits (2 CPUs, 4GB RAM) insufficient for intensive workloads

---

## Executive Summary

After analyzing the computationally intensive operations in the E2I Causal Analytics platform, the current worker resource allocation is **significantly under-resourced**. Three critical workloads require substantially more resources:

1. **SHAP Calculations** (shap_explainer_realtime.py)
2. **Causal Refutation** (Bootstrap & permutation tests)
3. **Twin Generation** (digital_twin/twin_generator.py)

**Recommendation:** Implement a **multi-worker architecture** with specialized resource pools.

---

## Detailed Workload Analysis

### 1. SHAP Explainer (src/mlops/shap_explainer_realtime.py)

#### Current Implementation
```python
# Line 36: ThreadPoolExecutor with max_workers=4
_executor = ThreadPoolExecutor(max_workers=4)

# Line 52: Background sample size for KernelExplainer
background_sample_size: int = 100  # For KernelExplainer

# Line 84: Background data cache
background_data_cache_size: int = 1000
```

#### Resource Requirements

**CPU:**
- TreeExplainer (tree-based models): **4-8 cores** for parallel tree traversal
- KernelExplainer (model-agnostic): **4-8 cores** for parallel perturbation samples
- DeepExplainer (neural networks): **8+ cores** + GPU acceleration (optional)

**Memory:**
- Small models (<100 features): **2-4 GB**
- Medium models (100-1000 features): **4-8 GB**
- Large models (1000+ features): **8-16 GB**
- Background data cache (1000 samples × features): **+1-4 GB**

**Calculation Example:**
```
Model: 500 features, 10000 rows
Background sample: 100 rows
SHAP values: 100 samples × 500 features × 8 bytes (float64) = 400 KB per sample
With overhead: ~2-4 GB for computation
Total estimated: 6-10 GB RAM, 4-8 CPUs
```

#### Bottlenecks
- ⚠️ **Memory:** Large feature sets cause OOM
- ⚠️ **CPU:** KernelExplainer is O(n²) in sample size
- ⚠️ **Latency:** Real-time API needs fast response (<5s)

#### Mitigation in Code
- ✅ Explainer caching (line 92-95)
- ✅ Background data sampling (limits computation)
- ✅ ThreadPoolExecutor (4 workers)
- ❌ **Limited by container**: 2 CPUs, 4GB RAM

---

### 2. Causal Refutation

#### Operations
Based on tool_composer documentation, causal refutation includes:
- **Bootstrap refutation**: 100-1000 resamples
- **Placebo treatment refutation**: Multiple iterations
- **Data subset refutation**: Cross-validation style
- **Random common cause**: Adding confounders

#### Resource Requirements

**CPU:**
- **Minimum:** 4 cores (sequential refutation)
- **Optimal:** 8-16 cores (parallel refutation tests)
- Each bootstrap iteration is independent → perfect for parallelization

**Memory:**
- Dataset in memory: **N rows × M features × 8 bytes**
- Bootstrap samples: **N × M × 8 × num_iterations**
- Example: 100K rows × 200 features × 1000 iterations = **160 GB** (if not streaming)

**Calculation Example:**
```
Dataset: 100,000 rows × 200 features = 160 MB base
Bootstrap iterations: 1000 resamples
Parallel execution (8 cores): 8 simultaneous resamples
Memory per core: 160 MB × 1.5 overhead = 240 MB
Total: 8 cores × 240 MB = ~2 GB (optimized)
But: Full dataset + results = 4-8 GB safe estimate
```

#### Bottlenecks
- ⚠️ **CPU:** Sequential execution with 2 cores = slow
- ⚠️ **Memory:** Multiple resamples in memory
- ⚠️ **Time:** 1000 iterations × 30s each = 8.3 hours (sequential)
- ⚠️ **Time:** With 8 cores: 1.25 hours (acceptable)

---

### 3. Twin Generator (src/digital_twin/twin_generator.py)

#### Current Implementation
```python
# Line 61: Minimum training samples
MIN_TRAINING_SAMPLES = 1000

# Line 23-26: ML models used
RandomForestRegressor
GradientBoostingRegressor
cross_val_score
train_test_split
```

#### Resource Requirements

**CPU:**
- **RandomForest training:** O(n_trees × n_samples × log(n_samples) × n_features)
- **GradientBoosting training:** O(n_estimators × n_samples × n_features)
- Parallel tree building: **4-16 cores** optimal
- Cross-validation: **Multiply by K folds**

**Memory:**
- Training data: **n_samples × n_features × 8 bytes**
- Model storage: **n_trees × tree_size** (can be 100MB-1GB)
- Feature matrices: **Multiple copies during cross-validation**

**Calculation Example:**
```
Training set: 100,000 HCPs × 15 features = 12 MB
Random Forest: 500 trees × 50 KB avg = 25 MB
Cross-validation (5 folds): 5 × (12 MB + 25 MB) = 185 MB
With overhead and intermediate results: 2-4 GB

Large scenario:
Training set: 1,000,000 HCPs × 100 features = 800 MB
Random Forest: 1000 trees × 200 KB = 200 MB
Cross-validation: 5 × 1 GB = 5 GB
Total: 8-12 GB
```

#### Bottlenecks
- ⚠️ **CPU:** Tree building is CPU-intensive
- ⚠️ **Memory:** Large datasets + model + CV folds
- ⚠️ **Time:** Training can take 10-60 minutes
- ⚠️ **Scalability:** Generating 10,000+ twins needs vectorization

---

## Current vs Required Resources

| Workload | Current Limit | Required (Min) | Required (Optimal) | Deficit |
|----------|---------------|----------------|--------------------|---------|
| **SHAP Explainer** |
| - CPU | 2 cores | 4 cores | 8 cores | 4-6 cores |
| - Memory | 4 GB | 6 GB | 10 GB | 2-6 GB |
| **Causal Refutation** |
| - CPU | 2 cores | 4 cores | 16 cores | 2-14 cores |
| - Memory | 4 GB | 4 GB | 8 GB | 0-4 GB |
| **Twin Generator** |
| - CPU | 2 cores | 4 cores | 8 cores | 2-6 cores |
| - Memory | 4 GB | 4 GB | 12 GB | 0-8 GB |

### Risk Assessment

**With Current Limits (2 CPUs, 4GB RAM):**
- 🔴 **SHAP**: Will OOM on large models, slow on complex models
- 🟡 **Refutation**: Will work but take 4-8x longer than optimal
- 🟡 **Twin**: Will work for small datasets, fail/slow on large datasets

---

## Recommended Solutions

### Option 1: Increase Single Worker Resources ❌ Not Recommended

```yaml
worker:
  deploy:
    resources:
      limits:
        cpus: '16'      # Increased from 2
        memory: 16G     # Increased from 4GB
      reservations:
        cpus: '8'
        memory: 8G
```

**Pros:**
- Simple configuration
- All tasks can run

**Cons:**
- **Expensive:** 16 cores × $0.08/hr = $1.28/hr (vs $0.16/hr currently)
- **Underutilized:** Most Celery tasks are lightweight
- **Single point of failure:** One heavy task blocks all tasks
- **Poor isolation:** Heavy task can starve light tasks

**Verdict:** ❌ Not cost-effective

---

### Option 2: Multi-Worker Architecture ✅ RECOMMENDED

Create specialized worker pools with different resource profiles.

```yaml
# Light worker for quick tasks (API calls, cache updates, etc.)
worker_light:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '1'
        memory: 1G
  command: >
    celery -A src.workers.celery_app worker
    --loglevel=INFO
    --concurrency=4
    --queues=default,quick,api
    --hostname=worker_light@%h

# Medium worker for standard analytics (reports, aggregations)
worker_medium:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
  command: >
    celery -A src.workers.celery_app worker
    --loglevel=INFO
    --concurrency=2
    --queues=analytics,reports
    --hostname=worker_medium@%h

# Heavy worker for compute-intensive tasks
worker_heavy:
  deploy:
    resources:
      limits:
        cpus: '16'
        memory: 32G
      reservations:
        cpus: '8'
        memory: 16G
  command: >
    celery -A src.workers.celery_app worker
    --loglevel=INFO
    --concurrency=2
    --queues=shap,causal,ml,twins
    --max-tasks-per-child=10
    --hostname=worker_heavy@%h
```

**Queue Assignment:**
- `default, quick, api` → worker_light (2 CPUs, 2GB)
- `analytics, reports` → worker_medium (4 CPUs, 8GB)
- `shap, causal, ml, twins` → worker_heavy (16 CPUs, 32GB)

**Task Routing (in Celery config):**
```python
# src/workers/celery_config.py
task_routes = {
    'tasks.shap_explain': {'queue': 'shap'},
    'tasks.causal_refutation': {'queue': 'causal'},
    'tasks.train_twin_model': {'queue': 'ml'},
    'tasks.generate_twins': {'queue': 'twins'},
    'tasks.api_*': {'queue': 'quick'},
    'tasks.report_*': {'queue': 'analytics'},
}
```

**Benefits:**
- ✅ **Cost-effective:** Only run heavy worker when needed
- ✅ **Scalable:** Can scale each pool independently
- ✅ **Isolated:** Heavy tasks don't block light tasks
- ✅ **Flexible:** Add/remove workers based on queue depth
- ✅ **Monitorable:** Separate metrics per worker type

**Cost Comparison:**
```
Current (1 worker × 2 CPU × 4GB):           $0.16/hr
Option 1 (1 worker × 16 CPU × 16GB):        $1.28/hr
Option 2 (3 workers, on-demand heavy):      $0.32-0.80/hr (avg)
  - Light (always on): $0.08/hr
  - Medium (8hr/day): $0.16/hr
  - Heavy (4hr/day): $0.32/hr
  - Total: ~$0.56/hr (56% cheaper than Option 1)
```

---

### Option 3: Hybrid Cloud + On-Premise ⚡ Advanced

For very large workloads, consider:
- **Light/Medium workers:** Always-on cloud instances
- **Heavy workers:** Burst to cloud on-demand (AWS EC2 Spot, GCP Preemptible)
- **GPU workers:** Separate pool for DeepExplainer (if using neural networks)

**Tools:**
- Kubernetes with HPA (Horizontal Pod Autoscaler)
- Docker Swarm with autoscaling
- Celery with AWS SQS + Lambda

**Complexity:** High
**Cost Savings:** 40-70% vs always-on
**Recommended for:** Production at scale (>100K tasks/day)

---

## Recommended Implementation Plan

### Phase 1: Immediate (This Session)
1. ✅ Document current resource constraints
2. 🔄 Design multi-worker architecture
3. ⏳ Update docker-compose.yml with 3 worker pools
4. ⏳ Configure Celery task routing

### Phase 2: Testing (Next Session)
1. Test light worker with API tasks
2. Test heavy worker with SHAP/refutation/twin tasks
3. Benchmark performance improvements
4. Monitor resource usage

### Phase 3: Optimization (Future)
1. Fine-tune concurrency settings
2. Add autoscaling based on queue depth
3. Implement graceful degradation (fallback to smaller models)
4. Add resource usage monitoring (Prometheus + Grafana)

---

## Updated Resource Allocation Table

| Worker Type | CPU Limit | Memory Limit | Queues | Use Cases |
|-------------|-----------|--------------|--------|-----------|
| **worker_light** | 2 | 2GB | default, quick, api | API calls, cache, quick queries |
| **worker_medium** | 4 | 8GB | analytics, reports | Standard analytics, aggregations |
| **worker_heavy** | 16 | 32GB | shap, causal, ml, twins | SHAP, refutation, ML training |

**Total Resources:**
- **Minimum (all reservations):** 11 CPUs, 22 GB RAM
- **Maximum (all limits):** 22 CPUs, 42 GB RAM

**Recommended Production Host:**
- **Cloud:** c6i.8xlarge (32 vCPU, 64 GB) or n2-highmem-16 (16 vCPU, 128 GB)
- **On-Premise:** Dual Xeon Silver 4214R (24 cores, 48 threads) + 128 GB RAM

---

## Task Routing Configuration

### Celery Configuration File

Create `src/workers/celery_config.py`:

```python
from kombu import Queue

# Queue definitions
task_queues = (
    Queue('default', routing_key='default'),
    Queue('quick', routing_key='quick'),
    Queue('api', routing_key='api'),
    Queue('analytics', routing_key='analytics'),
    Queue('reports', routing_key='reports'),
    Queue('shap', routing_key='shap'),
    Queue('causal', routing_key='causal'),
    Queue('ml', routing_key='ml'),
    Queue('twins', routing_key='twins'),
)

# Task routing
task_routes = {
    # Light worker tasks
    'src.tasks.api.*': {'queue': 'api'},
    'src.tasks.cache.*': {'queue': 'quick'},
    'src.tasks.notify.*': {'queue': 'quick'},

    # Medium worker tasks
    'src.tasks.reports.*': {'queue': 'reports'},
    'src.tasks.analytics.*': {'queue': 'analytics'},
    'src.tasks.aggregations.*': {'queue': 'analytics'},

    # Heavy worker tasks
    'src.tasks.shap_explain': {'queue': 'shap'},
    'src.tasks.causal_refutation': {'queue': 'causal'},
    'src.tasks.causal_sensitivity': {'queue': 'causal'},
    'src.tasks.train_twin_model': {'queue': 'ml'},
    'src.tasks.generate_twins': {'queue': 'twins'},
    'src.tasks.cross_validate_model': {'queue': 'ml'},
}

# Concurrency and prefetch settings
worker_prefetch_multiplier = 1  # Only prefetch 1 task to avoid blocking
task_acks_late = True  # Acknowledge after completion
task_reject_on_worker_lost = True  # Requeue if worker crashes
```

---

## Monitoring & Alerting

### Metrics to Track

```python
# Celery metrics
- celery_tasks_queued{queue="shap"}
- celery_tasks_running{queue="shap"}
- celery_task_duration_seconds{task="shap_explain"}
- celery_worker_cpu_percent{worker="worker_heavy"}
- celery_worker_memory_mb{worker="worker_heavy"}

# Custom metrics
- shap_computation_time_ms
- shap_explainer_cache_hit_rate
- twin_generation_count
- causal_refutation_iterations
```

### Alerts

```yaml
# Alert: Heavy worker queue depth
- alert: HeavyWorkerQueueDepth
  expr: celery_tasks_queued{queue=~"shap|causal|ml|twins"} > 50
  for: 10m
  annotations:
    summary: "Heavy worker queue building up"
    action: "Consider scaling worker_heavy"

# Alert: Worker OOM
- alert: WorkerMemoryExhaustion
  expr: celery_worker_memory_mb / worker_memory_limit_mb > 0.9
  for: 5m
  annotations:
    summary: "Worker approaching memory limit"
    action: "Increase memory or reduce concurrency"
```

---

## Next Steps

1. **Approve multi-worker architecture** (confirm resource budget)
2. **Update docker-compose.yml** with 3 worker types
3. **Create Celery task routing configuration**
4. **Test with sample SHAP/refutation/twin tasks**
5. **Monitor and tune** concurrency settings

---

## Questions for Decision

1. **Budget:** What's the acceptable cost increase for heavy worker pool?
   - Current: ~$120/month (2 CPU, 4GB, 24/7)
   - Proposed: ~$400/month (with 4hr/day heavy worker)

2. **Availability:** Should heavy worker be:
   - Always-on (higher cost, instant availability)
   - On-demand (lower cost, 2-5 min startup delay)
   - Auto-scaled (optimal cost, requires orchestration)

3. **Scale:** What's the expected workload?
   - SHAP requests/day: ?
   - Refutation tests/day: ?
   - Twin generations/week: ?

---

## Conclusion

The current worker resource allocation (2 CPUs, 4GB RAM) is **insufficient** for SHAP explanations, causal refutation, and twin generation tasks. A multi-worker architecture with specialized resource pools is strongly recommended to:

✅ Enable heavy computations to complete successfully
✅ Maintain fast response times for API tasks
✅ Optimize cost (only pay for heavy resources when needed)
✅ Improve reliability (task isolation)

**Recommended Action:** Implement Option 2 (Multi-Worker Architecture) with 3 worker tiers.
