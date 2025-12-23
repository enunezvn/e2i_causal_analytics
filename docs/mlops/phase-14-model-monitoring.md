# Phase 14: Model Monitoring & Drift Detection

**Goal**: Production-grade model monitoring with automated drift detection and alerting

**Status**: In Progress

**Dependencies**: Phase 12 (End-to-End Integration), Phase 13 (Feast Feature Store)

---

## Current State Assessment

### Existing Components

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Drift Monitor Agent | `src/agents/drift_monitor/` | Partial | Tier 3 agent with data/model drift |
| Data Drift Node | `nodes/data_drift.py` | Mock | PSI + KS test (uses MockDataConnector) |
| Model Drift Node | `nodes/model_drift.py` | Mock | KS + Chi-square test (uses MockDataConnector) |
| Concept Drift Node | `nodes/concept_drift.py` | Placeholder | Not implemented |
| Alert Aggregator | `nodes/alert_aggregator.py` | Complete | Severity scoring & recommendations |
| BentoML Monitoring | `src/mlops/bentoml_monitoring.py` | Complete | Service health + Prometheus metrics |
| MLOps Tables | `database/ml/mlops_tables.sql` | Complete | 8 tables, no drift history |

### Gap Analysis

| Capability | Current State | Target State |
|------------|---------------|--------------|
| Data connectors | MockDataConnector | SupabaseDataConnector |
| Concept drift | Placeholder | Feature importance comparison |
| Drift history storage | None | Database tables |
| Scheduled monitoring | None | Celery beat tasks |
| API endpoints | None | FastAPI endpoints |
| Alert routing | Logger only | Slack + email |
| Performance tracking | None | Time-series metrics |
| Retraining triggers | None | Automated pipeline |

---

## Implementation Plan

### Task 14.1: Database Schema for Monitoring
**Goal**: Create tables for drift history and performance metrics
**Estimate**: Medium complexity

**Files to Create**:
```
database/ml/015_model_monitoring_tables.sql
```

**Tables**:
1. `ml_drift_history` - Stores drift detection results
   - drift_type (data/model/concept)
   - feature_name
   - test_statistic, p_value
   - severity, drift_detected
   - baseline_period, current_period
   - model_id, experiment_id

2. `ml_performance_metrics` - Time-series model performance
   - model_id
   - metric_name (accuracy, precision, recall, auc, etc.)
   - metric_value
   - data_split
   - measured_at

3. `ml_monitoring_alerts` - Alert history
   - alert_id
   - severity (critical/warning/info)
   - drift_type
   - affected_features
   - message
   - recommended_action
   - acknowledged_at, acknowledged_by

**Subtasks**:
- [ ] 14.1.1: Create drift_history table with indexes
- [ ] 14.1.2: Create performance_metrics table with time-series index
- [ ] 14.1.3: Create monitoring_alerts table
- [ ] 14.1.4: Create views for drift trends
- [ ] 14.1.5: Add migration script
- [ ] 14.1.6: Add table documentation

---

### Task 14.2: Supabase Data Connectors
**Goal**: Replace MockDataConnector with real database access
**Estimate**: Medium complexity

**Files to Create/Modify**:
```
src/agents/drift_monitor/connectors/
├── __init__.py
├── base.py              # Abstract base connector
├── supabase_connector.py # Real Supabase implementation
└── mock_connector.py     # Keep for testing
```

**Interface**:
```python
class BaseDataConnector(ABC):
    @abstractmethod
    async def query_features(
        self,
        feature_names: list[str],
        time_window: str,
        filters: dict[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        """Query feature values for drift detection."""
        pass

    @abstractmethod
    async def query_predictions(
        self,
        model_id: str,
        time_window: str,
        filters: dict[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        """Query prediction data for model drift detection."""
        pass
```

**Subtasks**:
- [ ] 14.2.1: Create base connector abstract class
- [ ] 14.2.2: Implement SupabaseDataConnector
- [ ] 14.2.3: Move MockDataConnector to separate file
- [ ] 14.2.4: Update DataDriftNode to use connector interface
- [ ] 14.2.5: Update ModelDriftNode to use connector interface
- [ ] 14.2.6: Add connector factory with config
- [ ] 14.2.7: Add integration tests with real Supabase

---

### Task 14.3: Concept Drift Detection
**Goal**: Implement feature importance-based concept drift detection
**Estimate**: High complexity

**Approach**: Compare feature importance between periods using lightweight models

**Algorithm**:
1. Train lightweight model (e.g., RandomForest) on baseline data
2. Train same model on current data
3. Compare feature importances using correlation
4. Detect significant changes in feature-target relationships

**Files to Modify**:
```
src/agents/drift_monitor/nodes/concept_drift.py
```

**Implementation**:
```python
class ConceptDriftNode:
    """Detect drift in feature-target relationships."""

    async def _detect_importance_drift(
        self,
        baseline_features: pd.DataFrame,
        baseline_labels: pd.Series,
        current_features: pd.DataFrame,
        current_labels: pd.Series,
    ) -> list[DriftResult]:
        """Compare feature importance between periods."""
        # Train lightweight models
        baseline_model = RandomForestClassifier(n_estimators=50)
        current_model = RandomForestClassifier(n_estimators=50)

        baseline_model.fit(baseline_features, baseline_labels)
        current_model.fit(current_features, current_labels)

        # Compare feature importances
        importance_correlation = np.corrcoef(
            baseline_model.feature_importances_,
            current_model.feature_importances_
        )[0, 1]

        # Detect significant changes
        ...
```

**Subtasks**:
- [ ] 14.3.1: Implement baseline/current data fetching
- [ ] 14.3.2: Implement lightweight model training
- [ ] 14.3.3: Implement feature importance comparison
- [ ] 14.3.4: Implement correlation-based drift detection
- [ ] 14.3.5: Add performance degradation analysis
- [ ] 14.3.6: Add unit tests
- [ ] 14.3.7: Add integration tests

---

### Task 14.4: Drift Result Persistence
**Goal**: Store all drift results to database
**Estimate**: Medium complexity

**Files to Create/Modify**:
```
src/agents/drift_monitor/repositories/
├── __init__.py
├── drift_repository.py    # CRUD for drift history
└── alert_repository.py    # CRUD for alerts
```

**Integration Points**:
- Alert aggregator saves results after each run
- API can query historical drift data
- Dashboard can show drift trends

**Subtasks**:
- [ ] 14.4.1: Create DriftRepository class
- [ ] 14.4.2: Create AlertRepository class
- [ ] 14.4.3: Update AlertAggregatorNode to persist results
- [ ] 14.4.4: Add batch insert for efficiency
- [ ] 14.4.5: Add repository tests

---

### Task 14.5: Scheduled Monitoring Jobs
**Goal**: Automated drift checks via Celery beat
**Estimate**: Medium complexity

**Files to Create**:
```
src/tasks/monitoring_tasks.py
```

**Celery Tasks**:
| Task | Schedule | Description |
|------|----------|-------------|
| `check_data_drift` | Every 6 hours | Run data drift detection |
| `check_model_drift` | Every 6 hours | Run model drift detection |
| `check_concept_drift` | Daily | Run concept drift detection |
| `generate_drift_report` | Daily | Generate drift summary report |
| `cleanup_old_drift_data` | Weekly | Archive old drift history |

**Beat Schedule**:
```python
"monitoring-data-drift": {
    "task": "src.tasks.monitoring_tasks.check_data_drift",
    "schedule": 21600.0,  # 6 hours
    "args": (["remibrutinib", "fabhalta", "kisqali"],),
}
```

**Subtasks**:
- [ ] 14.5.1: Create monitoring_tasks.py
- [ ] 14.5.2: Implement check_data_drift task
- [ ] 14.5.3: Implement check_model_drift task
- [ ] 14.5.4: Implement check_concept_drift task
- [ ] 14.5.5: Implement generate_drift_report task
- [ ] 14.5.6: Update celery_app.py beat schedule
- [ ] 14.5.7: Add task tests

---

### Task 14.6: API Endpoints for Monitoring
**Goal**: REST API for drift status and history
**Estimate**: Medium complexity

**Files to Create**:
```
src/api/routes/monitoring.py
```

**Endpoints**:
| Method | Path | Description |
|--------|------|-------------|
| GET | `/monitoring/drift/status` | Current drift status for all models |
| GET | `/monitoring/drift/history` | Historical drift data |
| GET | `/monitoring/drift/{model_id}` | Drift status for specific model |
| GET | `/monitoring/performance/{model_id}` | Performance metrics over time |
| GET | `/monitoring/alerts` | Active alerts |
| POST | `/monitoring/alerts/{alert_id}/acknowledge` | Acknowledge alert |
| POST | `/monitoring/drift/check` | Trigger manual drift check |

**Response Schema**:
```python
class DriftStatusResponse(BaseModel):
    model_id: str
    overall_drift_score: float
    data_drift: list[DriftResult]
    model_drift: list[DriftResult]
    concept_drift: list[DriftResult]
    alerts: list[DriftAlert]
    last_checked: datetime
```

**Subtasks**:
- [ ] 14.6.1: Create monitoring router
- [ ] 14.6.2: Implement drift status endpoint
- [ ] 14.6.3: Implement drift history endpoint
- [ ] 14.6.4: Implement performance metrics endpoint
- [ ] 14.6.5: Implement alerts endpoints
- [ ] 14.6.6: Implement manual drift check trigger
- [ ] 14.6.7: Add API tests
- [ ] 14.6.8: Add OpenAPI documentation

---

### Task 14.7: Alert Routing
**Goal**: Send alerts to Slack and email
**Estimate**: Medium complexity

**Files to Create**:
```
src/notifications/
├── __init__.py
├── base.py              # Abstract notifier
├── slack_notifier.py    # Slack webhook integration
├── email_notifier.py    # SMTP email integration
└── composite.py         # Send to multiple channels
```

**Integration**:
- Configure in `.env`:
  ```
  SLACK_WEBHOOK_URL=https://hooks.slack.com/...
  SMTP_HOST=smtp.example.com
  ALERT_EMAIL_RECIPIENTS=ml-team@example.com
  ```

- Alert thresholds by severity:
  - Critical: Immediate Slack + email
  - High: Slack + daily digest email
  - Medium/Low: Log only

**Subtasks**:
- [ ] 14.7.1: Create base notifier interface
- [ ] 14.7.2: Implement SlackNotifier
- [ ] 14.7.3: Implement EmailNotifier
- [ ] 14.7.4: Implement CompositeNotifier
- [ ] 14.7.5: Configure alert routing rules
- [ ] 14.7.6: Update AlertAggregatorNode to use notifiers
- [ ] 14.7.7: Add notification tests

---

### Task 14.8: Performance Tracking
**Goal**: Track model performance metrics over time
**Estimate**: Medium complexity

**Metrics to Track**:
- Classification: AUC, PR-AUC, precision, recall, F1
- Regression: RMSE, MAE, R2
- Calibration: Brier score, calibration slope
- Business: Prediction volume, error rate, latency

**Files to Create**:
```
src/mlops/performance_tracker.py
```

**Integration with MLflow**:
- Read metrics from MLflow
- Store to `ml_performance_metrics` table
- Enable time-series queries

**Subtasks**:
- [ ] 14.8.1: Create PerformanceTracker class
- [ ] 14.8.2: Implement MLflow metrics extraction
- [ ] 14.8.3: Implement metrics persistence
- [ ] 14.8.4: Add time-series query methods
- [ ] 14.8.5: Create Celery task for periodic collection
- [ ] 14.8.6: Add performance tests

---

### Task 14.9: Automated Retraining Triggers
**Goal**: Trigger model retraining when drift exceeds thresholds
**Estimate**: High complexity

**Trigger Rules**:
1. Critical data drift → Immediate retraining
2. Critical model drift → Immediate retraining
3. High drift persisting > 24 hours → Schedule retraining
4. Performance degradation > threshold → Schedule retraining

**Integration with Phase 12 Pipeline**:
```python
async def trigger_retraining(
    model_id: str,
    reason: str,
    priority: str = "normal"
) -> str:
    """Trigger model retraining pipeline."""
    # Create training job
    job_id = await ml_foundation_pipeline.start_training(
        model_id=model_id,
        reason=reason,
        use_latest_data=True,
    )
    return job_id
```

**Subtasks**:
- [ ] 14.9.1: Define retraining trigger rules
- [ ] 14.9.2: Implement RetrainingTrigger class
- [ ] 14.9.3: Integrate with ML Foundation Pipeline
- [ ] 14.9.4: Add cooldown periods to prevent spam
- [ ] 14.9.5: Add retraining history tracking
- [ ] 14.9.6: Add integration tests

---

### Task 14.10: Documentation & Testing
**Goal**: Comprehensive documentation and test coverage
**Estimate**: Medium complexity

**Files to Create**:
```
docs/MODEL_MONITORING.md                    # User guide
tests/unit/test_monitoring/
├── test_data_drift_node.py
├── test_model_drift_node.py
├── test_concept_drift_node.py
├── test_alert_aggregator.py
├── test_drift_repository.py
├── test_monitoring_api.py
└── test_monitoring_tasks.py
tests/integration/test_monitoring_e2e.py
```

**Subtasks**:
- [ ] 14.10.1: Create MODEL_MONITORING.md user guide
- [ ] 14.10.2: Add unit tests for all nodes
- [ ] 14.10.3: Add repository tests
- [ ] 14.10.4: Add API tests
- [ ] 14.10.5: Add Celery task tests
- [ ] 14.10.6: Add end-to-end integration test
- [ ] 14.10.7: Update docs/mlops/README.md

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Drift detection latency | < 5 seconds |
| Alert delivery time | < 1 minute |
| False positive rate | < 5% |
| Test coverage | > 90% |
| API response time | < 500ms |

---

## Files Summary

### New Files
```
database/ml/015_model_monitoring_tables.sql
src/agents/drift_monitor/connectors/
├── __init__.py
├── base.py
├── supabase_connector.py
└── mock_connector.py
src/agents/drift_monitor/repositories/
├── __init__.py
├── drift_repository.py
└── alert_repository.py
src/tasks/monitoring_tasks.py
src/api/routes/monitoring.py
src/notifications/
├── __init__.py
├── base.py
├── slack_notifier.py
├── email_notifier.py
└── composite.py
src/mlops/performance_tracker.py
docs/MODEL_MONITORING.md
tests/unit/test_monitoring/
tests/integration/test_monitoring_e2e.py
```

### Modified Files
```
src/agents/drift_monitor/nodes/data_drift.py      # Use connector interface
src/agents/drift_monitor/nodes/model_drift.py     # Use connector interface
src/agents/drift_monitor/nodes/concept_drift.py   # Full implementation
src/agents/drift_monitor/nodes/alert_aggregator.py # Add persistence
src/workers/celery_app.py                          # Add beat schedule
src/api/main.py                                    # Add monitoring router
docs/mlops/README.md                               # Update progress
```

---

## Progress Tracking

| Task | Status | Progress |
|------|--------|----------|
| 14.1: Database Schema | Not Started | 0/6 |
| 14.2: Supabase Connectors | Not Started | 0/7 |
| 14.3: Concept Drift | Not Started | 0/7 |
| 14.4: Drift Persistence | Not Started | 0/5 |
| 14.5: Scheduled Jobs | Not Started | 0/7 |
| 14.6: API Endpoints | Not Started | 0/8 |
| 14.7: Alert Routing | Not Started | 0/7 |
| 14.8: Performance Tracking | Not Started | 0/6 |
| 14.9: Retraining Triggers | Not Started | 0/6 |
| 14.10: Documentation | Not Started | 0/7 |

**Overall Progress**: 0/66 tasks (0%)

---

## Related Documentation

- [Phase 12: End-to-End Integration](./phase-12-integration.md)
- [Phase 13: Feast Feature Store](./phase-13-feast-feature-store.md)
- [Drift Monitor Agent Specialist](./../.claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md)
- [BentoML Monitoring](../src/mlops/bentoml_monitoring.py)

---

## Last Updated

2025-12-22
