# BentoML Operations Runbook

**Version**: 1.0.0
**Last Updated**: 2025-12-26
**Maintainer**: E2I Causal Analytics Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Service Management](#service-management)
5. [Model Deployment](#model-deployment)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Troubleshooting](#troubleshooting)
8. [Disaster Recovery](#disaster-recovery)

---

## Overview

### Purpose

This runbook provides operational guidance for managing BentoML model serving infrastructure in the E2I Causal Analytics platform.

### Services

| Service | Port | Purpose |
|---------|------|---------|
| classification-service | 3001 | Binary/multiclass classification models |
| regression-service | 3002 | Continuous value prediction models |
| causal-service | 3003 | CATE estimation and treatment effects |
| prometheus | 9090 | Metrics collection |
| grafana | 3100 | Metrics visualization |

### Key Files

| File | Purpose |
|------|---------|
| `docker/bentoml/docker-compose.yaml` | Service orchestration |
| `docker/bentoml/Dockerfile` | Container image definition |
| `docker/bentoml/mock_service.py` | Mock service for testing |
| `config/model_endpoints.yaml` | Endpoint configuration |
| `src/mlops/bentoml_service.py` | Core BentoML integration |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  /api/predict   │  │  /api/explain   │  │  /health/bentoml│  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BentoML Client Layer                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  src/api/dependencies/bentoml_client.py                     ││
│  │  - Connection pooling (httpx)                               ││
│  │  - Circuit breaker pattern                                  ││
│  │  - Retry with exponential backoff                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ Classification    │ │ Regression        │ │ Causal            │
│ Service (:3001)   │ │ Service (:3002)   │ │ Service (:3003)   │
│ - churn_model     │ │ - ltv_model       │ │ - cate_model      │
│ - conversion_model│ │ - adoption_model  │ │ - treatment_effect│
└───────────────────┘ └───────────────────┘ └───────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
                    ┌───────────────────────┐
                    │   Prometheus (:9090)  │
                    │   - Request metrics   │
                    │   - Latency histograms│
                    │   - Error rates       │
                    └───────────────────────┘
```

---

## Quick Start

### Start All Services

```bash
# Start full stack
docker compose -f docker/bentoml/docker-compose.yaml up -d

# Start specific service
docker compose -f docker/bentoml/docker-compose.yaml up -d classification-service prometheus
```

### Stop All Services

```bash
docker compose -f docker/bentoml/docker-compose.yaml down
```

### Check Status

```bash
# Container status
docker compose -f docker/bentoml/docker-compose.yaml ps

# Health check
curl http://localhost:3001/healthz

# Detailed health
curl -X POST http://localhost:3001/health -H "Content-Type: application/json" -d '{}'
```

### Run Tests Against Live Stack

```bash
# Set environment variables
export BENTOML_SERVICE_URL=http://localhost:3001
export PROMETHEUS_URL=http://localhost:9090

# Run E2E tests
./venv/bin/python -m pytest tests/e2e/test_model_serving/ -v --tb=short
```

---

## Service Management

### Container Operations

```bash
# View logs
docker logs e2i-classification-service --tail 100 -f

# Restart service
docker restart e2i-classification-service

# Rebuild and restart
docker compose -f docker/bentoml/docker-compose.yaml up -d --build classification-service

# Scale service (if using swarm/kubernetes)
docker compose -f docker/bentoml/docker-compose.yaml up -d --scale classification-service=3
```

### Resource Limits

Configured in `docker-compose.yaml`:

| Service | CPU Limit | Memory Limit | CPU Reserved | Memory Reserved |
|---------|-----------|--------------|--------------|-----------------|
| classification | 2 | 4G | 0.5 | 1G |
| regression | 2 | 4G | 0.5 | 1G |
| causal | 4 | 8G | 1 | 2G |

### Health Checks

Each service has built-in health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3000/healthz"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 30s  # 60s for causal-service
```

---

## Model Deployment

### Register a New Model

```python
import bentoml
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save to BentoML model store
saved_model = bentoml.sklearn.save_model(
    "churn_model",
    model,
    signatures={"predict": {"batchable": True}},
    labels={"team": "ml", "environment": "production"},
    metadata={"accuracy": 0.95},
)
print(f"Model saved: {saved_model.tag}")
```

### Deploy via model_deployer Agent

```python
from src.agents.ml_foundation.model_deployer.graph import run_deployment

result = await run_deployment({
    "experiment_id": "exp-123",
    "model_uri": "mlflow://models/churn_model/1",
    "target_environment": "staging",
    "resources": {"cpu": "2", "memory": "4Gi"},
})
```

### Manual Deployment

```bash
# Build bento
bentoml build

# Containerize
bentoml containerize churn_model:latest

# Deploy
docker run -p 3001:3000 churn_model:latest
```

---

## Monitoring & Alerting

### Prometheus Metrics

Access Prometheus: http://localhost:9090

**Key Metrics**:

| Metric | Description |
|--------|-------------|
| `bentoml_request_total` | Total requests by endpoint |
| `bentoml_request_duration_seconds` | Request latency histogram |
| `bentoml_request_in_progress` | Current in-flight requests |
| `bentoml_model_predict_total` | Predictions per model |

### Example Queries

```promql
# Request rate (last 5 min)
rate(bentoml_request_total[5m])

# P95 latency
histogram_quantile(0.95, rate(bentoml_request_duration_seconds_bucket[5m]))

# Error rate
rate(bentoml_request_total{status_code=~"5.."}[5m]) / rate(bentoml_request_total[5m])
```

### Grafana Dashboards

Access Grafana: http://localhost:3100 (admin/admin)

Pre-configured dashboards:
- Model Serving Overview
- Latency Analysis
- Error Tracking

### Alerting Rules

Configure in `docker/bentoml/prometheus.yml`:

```yaml
groups:
  - name: bentoml
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(bentoml_request_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency detected"

      - alert: HighErrorRate
        expr: rate(bentoml_request_total{status_code=~"5.."}[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
```

---

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Symptoms**: Container exits immediately or keeps restarting

**Check**:
```bash
docker logs e2i-classification-service --tail 50
```

**Common Causes**:
- Missing model in BentoML store
- Port already in use
- Insufficient memory

**Fix**:
```bash
# Check if port is in use
ss -tlnp | grep 3001

# Kill existing process
sudo kill -9 $(lsof -t -i:3001)

# Restart with more memory
docker compose -f docker/bentoml/docker-compose.yaml up -d
```

#### 2. 400 Validation Error on Predict

**Symptoms**: `ValidationError: Field required [input_data]`

**Cause**: BentoML expects request body wrapped with parameter name

**Fix**: Ensure request body wraps data in `input_data`:
```bash
# Correct format
curl -X POST http://localhost:3001/churn_model/predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"features": [[0.5, 0.3]], "model_type": "classification"}}'

# Wrong format (missing wrapper)
curl -X POST ... -d '{"features": [[0.5, 0.3]]}'  # ❌ Will fail
```

The `BentoMLClient` handles this automatically via `bentoml_client.py` line 277-279.

#### 3. 404 Not Found on Model Endpoint

**Symptoms**: `404 Not Found for url 'http://localhost:3001/churn_model/predict'`

**Cause**: Mock service not rebuilt with model-specific routes

**Fix**:
```bash
# Rebuild and restart services
docker compose -f docker/bentoml/docker-compose.yaml build --no-cache
docker compose -f docker/bentoml/docker-compose.yaml up -d
```

The mock service supports these endpoints:
- `/predict` - Generic prediction
- `/churn_model/predict` - Churn model
- `/conversion_model/predict` - Conversion model
- `/ltv_model/predict` - LTV model
- `/cate_model/predict` - CATE model

#### 4. Circuit Breaker Open

**Symptoms**: `RuntimeError: Circuit breaker open for model 'churn_model'`

**Cause**: Too many consecutive failures

**Fix**:
```python
# Check service health
curl http://localhost:3001/healthz

# If healthy, wait for circuit breaker to reset (30s default)
# Or restart the client
```

#### 4. High Latency

**Symptoms**: Predictions taking >100ms

**Check**:
```bash
# Check container resources
docker stats e2i-classification-service

# Check Prometheus metrics
curl 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(bentoml_request_duration_seconds_bucket[5m]))'
```

**Fix**:
- Increase container resources
- Enable request batching
- Scale horizontally

#### 5. Memory Exhaustion

**Symptoms**: Container killed with OOMKilled

**Fix**:
```yaml
# Increase memory limit in docker-compose.yaml
deploy:
  resources:
    limits:
      memory: 8G  # Increase from 4G
```

### Debug Commands

```bash
# Container inspection
docker inspect e2i-classification-service

# Network debugging
docker exec e2i-classification-service curl http://localhost:3000/healthz

# Check model store
docker exec e2i-classification-service bentoml models list

# View service configuration
docker exec e2i-classification-service cat /home/bentoml/bentoml/configuration.yaml
```

---

## Disaster Recovery

### Backup Procedures

#### Model Store Backup

```bash
# Backup volume
docker run --rm \
  -v e2i-bentoml-models:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/bentoml-models-$(date +%Y%m%d).tar.gz /data
```

#### Prometheus Data Backup

```bash
docker run --rm \
  -v e2i-prometheus-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/prometheus-$(date +%Y%m%d).tar.gz /data
```

### Restore Procedures

```bash
# Restore model store
docker run --rm \
  -v e2i-bentoml-models:/data \
  -v $(pwd)/backups:/backup \
  alpine sh -c "cd /data && tar xzf /backup/bentoml-models-20251226.tar.gz --strip 1"
```

### Failover

If primary service fails:

1. **Check health**: `curl http://localhost:3001/healthz`
2. **Restart service**: `docker restart e2i-classification-service`
3. **If persistent**: Rebuild container
4. **If data corrupted**: Restore from backup

### Rollback

```bash
# List available images
docker images | grep classification-service

# Rollback to previous version
docker compose -f docker/bentoml/docker-compose.yaml stop classification-service
docker tag bentoml-classification-service:latest bentoml-classification-service:rollback
docker pull bentoml-classification-service:v1.0.0  # previous version
docker compose -f docker/bentoml/docker-compose.yaml up -d classification-service
```

---

## Appendix

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENTOML_HOME` | `/home/bentoml/bentoml` | Model store location |
| `SERVICE_TYPE` | `classification` | Service type identifier |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `BENTOML_SERVICE_URL` | - | Base URL for client |
| `PROMETHEUS_URL` | - | Prometheus endpoint |

### API Reference

#### Prediction Endpoint

```bash
POST /predict
Content-Type: application/json

{
  "input_data": {
    "features": [[1.0, 2.0, 3.0]],
    "model_type": "classification"
  }
}
```

#### Health Endpoint

```bash
POST /health
Content-Type: application/json

{}
```

#### Model Info Endpoint

```bash
POST /model_info
Content-Type: application/json

{}
```

### Contact

- **On-Call**: #ml-ops-oncall (Slack)
- **Escalation**: ml-platform@company.com
- **Documentation**: [Confluence - BentoML](https://confluence.company.com/bentoml)
