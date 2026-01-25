# Tier 5 Self-Improvement Agents: Operations Guide

**Version**: 4.3
**Last Updated**: 2026-01-24
**Agents**: Explainer, Feedback Learner

---

## Table of Contents

1. [Deployment Checklist](#deployment-checklist)
2. [Configuration Reference](#configuration-reference)
3. [Monitoring & Observability](#monitoring--observability)
4. [Scaling Considerations](#scaling-considerations)
5. [Troubleshooting](#troubleshooting)
6. [Maintenance](#maintenance)

---

## Deployment Checklist

### Prerequisites

- [ ] Python 3.11+ installed
- [ ] Virtual environment activated (`/opt/e2i_causal_analytics/.venv/`)
- [ ] Redis running (port 6382) for working memory
- [ ] Supabase accessible for episodic memory
- [ ] FalkorDB running (port 6381) for semantic memory
- [ ] MLflow tracking server running (port 5000)
- [ ] Opik observability (optional, ports 5173/8080)

### Explainer Agent

```bash
# Verify module imports
python -c "from src.agents.explainer import ExplainerAgent; print('OK')"

# Test with sample data
python -c "
import asyncio
from src.agents.explainer import ExplainerAgent

async def test():
    agent = ExplainerAgent(use_llm=False)
    result = await agent.explain(
        analysis_results=[{'findings': ['Test']}],
        query='Test query'
    )
    print(f'Status: {result.status}')

asyncio.run(test())
"
```

### Feedback Learner Agent

```bash
# Verify module imports
python -c "from src.agents.feedback_learner import FeedbackLearnerAgent; print('OK')"

# Test scheduler
python -c "
from src.agents.feedback_learner import create_scheduler, FeedbackLearnerAgent
agent = FeedbackLearnerAgent()
scheduler = create_scheduler(agent, interval_hours=6)
print(f'Scheduler state: {scheduler.state}')
"
```

### Memory Backends

```bash
# Check Redis
redis-cli -p 6382 ping

# Check FalkorDB
redis-cli -p 6381 ping

# Check Supabase (via API)
curl -s "$SUPABASE_URL/rest/v1/" -H "apikey: $SUPABASE_KEY" | head -c 100
```

---

## Configuration Reference

### Explainer Agent Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_llm` | `None` | LLM mode: `None`=auto, `True`=always, `False`=never |
| `llm_threshold` | `0.5` | Complexity score threshold for auto LLM |
| `result_count_weight` | `0.25` | Weight for result count in complexity |
| `query_complexity_weight` | `0.30` | Weight for query complexity |
| `causal_discovery_weight` | `0.25` | Weight for causal discovery presence |
| `expertise_weight` | `0.20` | Weight for expertise level |

#### Environment Variables

```bash
# Optional: Override defaults
export E2I_EXPLAINER_LLM_THRESHOLD=0.6
export E2I_EXPLAINER_DEFAULT_FORMAT=narrative
```

### Feedback Learner Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_llm` | `False` | Use LLM for pattern analysis |
| `interval_hours` | `6.0` | Scheduler interval |
| `min_feedback_threshold` | `10` | Minimum feedback for cycle |
| `cycle_timeout_seconds` | `300` | Cycle timeout (5 min) |
| `max_batch_size` | `1000` | Maximum feedback items per cycle |

#### GEPA Trigger Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_signals` | `100` | Minimum signals for optimization |
| `min_reward_delta` | `0.05` | Minimum reward change |
| `cooldown_hours` | `24` | Hours between optimizations |
| `max_hours_without_optimization` | `168` | Force optimization after 7 days |

---

## Monitoring & Observability

### Key Metrics to Track

#### Explainer Agent

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| `explainer_latency_ms` | < 5000ms | > 10000ms |
| `explainer_success_rate` | > 95% | < 90% |
| `explainer_llm_usage_rate` | 30-70% | < 10% or > 90% |
| `explainer_insight_count` | > 0 | = 0 for 10 consecutive |

#### Feedback Learner

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| `feedback_learner_cycle_success_rate` | > 90% | < 80% |
| `feedback_learner_patterns_per_cycle` | 1-10 | 0 for 5 cycles |
| `feedback_learner_recommendation_rate` | > 0.5/cycle | 0 for 10 cycles |
| `feedback_learner_queue_size` | < 10000 | > 50000 |

### MLflow Tracking

Both agents log to MLflow:

```python
# View experiments
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiments = mlflow.search_experiments()

# Query explainer runs
runs = mlflow.search_runs(
    experiment_names=["explainer"],
    filter_string="metrics.total_latency_ms < 5000"
)
```

### Prometheus Metrics (if enabled)

```yaml
# prometheus.yml scrape config
scrape_configs:
  - job_name: 'e2i-agents'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Log Monitoring

```bash
# Tail explainer logs
journalctl -u e2i-api -f | grep explainer

# Tail feedback learner logs
journalctl -u e2i-api -f | grep feedback_learner

# Check for errors
journalctl -u e2i-api --since "1 hour ago" | grep -i error
```

---

## Scaling Considerations

### Horizontal Scaling

#### Explainer Agent

- **Stateless**: Can run multiple instances behind load balancer
- **Memory isolation**: Each request is independent
- **Recommended**: 1 instance per 100 concurrent users

```python
# Multiple workers with gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app
```

#### Feedback Learner

- **Single scheduler**: Only ONE scheduler instance should run
- **Multiple agents**: Can process batches in parallel
- **Recommended**: Use distributed lock (Redis) for scheduler

```python
# Distributed lock for scheduler
import redis
import uuid

lock_id = str(uuid.uuid4())
r = redis.Redis(host='localhost', port=6382)

if r.set('feedback_scheduler_lock', lock_id, nx=True, ex=300):
    # This instance runs the scheduler
    await scheduler.start()
```

### Rate Limiting

```python
# LLM rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=10, period=60)  # 10 calls per minute
async def call_llm(prompt):
    ...
```

### Memory Backend Scaling

| Backend | Scaling Strategy |
|---------|-----------------|
| Redis | Redis Cluster or Redis Sentinel |
| Supabase | Connection pooling (pgbouncer) |
| FalkorDB | Read replicas for queries |

### Resource Estimates

| Load Level | Explainer | Feedback Learner |
|------------|-----------|------------------|
| Low (< 100 req/day) | 1 CPU, 512MB | 0.5 CPU, 256MB |
| Medium (100-1000 req/day) | 2 CPU, 1GB | 1 CPU, 512MB |
| High (> 1000 req/day) | 4 CPU, 2GB | 2 CPU, 1GB |

---

## Troubleshooting

### Common Issues

#### 1. Explainer Returns Empty Response

**Symptoms**: `executive_summary` and `detailed_explanation` are empty

**Causes**:
- Empty `analysis_results` input
- LLM timeout with no fallback
- Memory backend failure

**Resolution**:
```python
# Check input
assert len(analysis_results) > 0, "No analysis results provided"

# Check with deterministic mode
agent = ExplainerAgent(use_llm=False)
result = await agent.explain(analysis_results, query)
```

#### 2. Feedback Learner Scheduler Not Running

**Symptoms**: No learning cycles executed

**Causes**:
- Scheduler not started
- Below minimum feedback threshold
- Distributed lock held by another instance

**Resolution**:
```python
# Check scheduler state
print(scheduler.state)  # Should be "running"

# Check pending feedback count
count = await scheduler.check_pending_feedback()
print(f"Pending feedback: {count}")

# Force a cycle
result = await scheduler.run_cycle_now(force=True)
```

#### 3. GEPA Optimization Not Triggering

**Symptoms**: No prompt optimization runs

**Causes**:
- Below signal threshold (< 100 signals)
- Cooldown period active
- Reward delta too small

**Resolution**:
```python
from src.agents.feedback_learner import GEPAOptimizationTrigger

trigger = GEPAOptimizationTrigger()
should_run, reason = trigger.should_trigger(
    signal_count=current_signals,
    current_reward=current_reward,
    baseline_reward=baseline,
    last_optimization=last_opt_time,
)
print(f"Should trigger: {should_run}, Reason: {reason}")
```

#### 4. Memory Backend Connection Failures

**Symptoms**: Warnings about unavailable memory, degraded responses

**Causes**:
- Redis/Supabase/FalkorDB down
- Network connectivity issues
- Connection pool exhausted

**Resolution**:
```bash
# Check Redis
redis-cli -p 6382 info clients

# Check connection pools
curl http://localhost:8000/health | jq .memory_backends
```

#### 5. High Latency in Explainer

**Symptoms**: Responses take > 10 seconds

**Causes**:
- LLM rate limiting
- Large analysis results
- Memory backend latency

**Resolution**:
```python
# Use deterministic mode for faster responses
agent = ExplainerAgent(use_llm=False)

# Or increase LLM threshold to reduce LLM usage
config = ExplainerConfig(llm_threshold=0.8)
agent = ExplainerAgent(config=config)
```

### Error Codes

| Error | Meaning | Action |
|-------|---------|--------|
| `E5001` | Memory backend unavailable | Check Redis/Supabase/FalkorDB |
| `E5002` | LLM timeout | Increase timeout or use deterministic |
| `E5003` | Invalid input format | Validate analysis_results structure |
| `E5004` | Scheduler conflict | Check distributed lock |
| `E5005` | GEPA optimization failed | Check signal quality |

---

## Maintenance

### Regular Tasks

#### Daily
- [ ] Check scheduler cycle history
- [ ] Review error logs
- [ ] Verify memory backend connectivity

#### Weekly
- [ ] Review GEPA optimization results
- [ ] Analyze feedback patterns
- [ ] Check MLflow experiment metrics

#### Monthly
- [ ] Review and tune thresholds
- [ ] Clean up old training signals
- [ ] Backup knowledge stores

### Backup Procedures

```bash
# Backup Redis
redis-cli -p 6382 BGSAVE

# Export MLflow experiments
mlflow experiments export --experiment-id 1 --output-dir /backup/mlflow/

# Backup Supabase (via pg_dump)
pg_dump -h $SUPABASE_HOST -U postgres -d postgres > backup.sql
```

### Update Procedures

```bash
# Update agents
cd /opt/e2i_causal_analytics
git pull origin main
source .venv/bin/activate

# Run tests
pytest tests/unit/test_agents/test_explainer -v
pytest tests/unit/test_agents/test_feedback_learner -v

# Restart services
sudo systemctl restart e2i-api
```

---

## Support

For issues not covered in this guide:

1. Check logs: `journalctl -u e2i-api -n 100`
2. Review MLflow experiments for anomalies
3. Create issue at: https://github.com/enunezvn/e2i_causal_analytics/issues

Include:
- Agent version (`cat src/agents/*/agent.py | grep Version`)
- Error messages
- Scheduler status (if applicable)
- Memory backend health
