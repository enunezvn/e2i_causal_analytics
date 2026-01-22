# Experiment Monitor Agent Specialist

## Overview

| Property | Value |
|----------|-------|
| **Agent Name** | experiment_monitor |
| **Tier** | 3 (Design & Monitoring) |
| **Type** | Standard (Fast Path) |
| **Model** | None (computational) |
| **Latency Target** | <5s per experiment check |
| **Critical Path** | No |

## Purpose

The Experiment Monitor agent continuously monitors active A/B experiments for health issues, statistical anomalies, and analysis triggers. It provides early warning detection for experiment problems.

## Capabilities

### Health Monitoring
- **Enrollment Rate Tracking**: Monitors daily enrollment rates against targets
- **Data Staleness Detection**: Alerts when experiment data hasn't been updated
- **Data Quality Checks**: Validates data completeness and consistency

### Statistical Anomaly Detection
- **Sample Ratio Mismatch (SRM)**: Detects assignment bias using chi-squared tests
- **Early Stopping Triggers**: Identifies when interim analysis is warranted
- **Variance Inflation**: Detects unusual variance in treatment effects

### Digital Twin Integration
- **Fidelity Monitoring**: Compares Digital Twin predictions vs actual outcomes
- **Calibration Drift**: Detects when twin predictions diverge from reality

## State Definition

```python
class ExperimentMonitorState(TypedDict, total=False):
    # Input
    query: str
    experiment_ids: List[str]
    check_all_active: bool

    # Thresholds
    srm_threshold: float  # Default: 0.001
    enrollment_threshold: float  # Default: 5.0 per day
    fidelity_threshold: float  # Default: 0.2
    stale_data_threshold_hours: float  # Default: 24.0

    # Detection results
    experiments: List[ExperimentSummary]
    srm_issues: List[SRMIssue]
    enrollment_issues: List[EnrollmentIssue]
    stale_data_issues: List[StaleDataIssue]
    fidelity_issues: List[FidelityIssue]
    interim_triggers: List[InterimTrigger]

    # Output
    alerts: List[MonitorAlert]
    monitor_summary: str
    recommended_actions: List[str]
    check_latency_ms: int
    experiments_checked: int

    # Execution
    status: Literal["pending", "processing", "completed", "failed"]
    errors: List[Dict[str, Any]]
```

## Node Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   FETCH      │───►│   ANALYZE    │───►│   GENERATE   │
│  EXPERIMENTS │    │   HEALTH     │    │   ALERTS     │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │   SUMMARIZE  │
                                        │   RESULTS    │
                                        └──────────────┘
```

### Node 1: Fetch Experiments
- Queries experiment repository for active experiments
- Applies filters (specific IDs or all active)
- Retrieves experiment metadata and current statistics

### Node 2: Analyze Health
- Performs SRM detection using chi-squared test
- Checks enrollment rates against thresholds
- Validates data freshness
- Checks Digital Twin fidelity if available
- Evaluates interim analysis triggers

### Node 3: Generate Alerts
- Creates MonitorAlert objects for detected issues
- Assigns severity levels (INFO, WARNING, CRITICAL)
- Generates recommended actions per issue

### Node 4: Summarize Results
- Aggregates all findings into monitor_summary
- Counts experiments by health status
- Prioritizes recommended actions

## Usage

```python
from src.agents.experiment_monitor import ExperimentMonitorAgent, ExperimentMonitorInput

agent = ExperimentMonitorAgent()
result = await agent.run_async(ExperimentMonitorInput(
    query="Check all active experiments for issues",
    check_all_active=True,
    srm_threshold=0.001,
    enrollment_threshold=5.0
))

print(f"Experiments checked: {result.experiments_checked}")
print(f"Healthy: {result.healthy_count}")
print(f"Warnings: {result.warning_count}")
print(f"Critical: {result.critical_count}")

for alert in result.alerts:
    print(f"[{alert.severity}] {alert.message}")
```

## Alert Types

| Alert Type | Severity | Description |
|------------|----------|-------------|
| SRM_DETECTED | CRITICAL | Sample ratio mismatch detected (p < threshold) |
| LOW_ENROLLMENT | WARNING | Daily enrollment below target |
| STALE_DATA | WARNING | Data not updated within threshold |
| TWIN_DRIFT | WARNING | Digital Twin predictions diverging |
| INTERIM_TRIGGER | INFO | Interim analysis recommended |

## Integration Points

### Upstream Dependencies
- **Experiment Designer**: Receives experiment configurations
- **Database**: Queries experiment_assignments, experiment_results

### Downstream Consumers
- **Health Score**: Contributes to overall system health
- **Drift Monitor**: Shares data quality signals
- **Orchestrator**: Routes monitoring queries

### Memory Integration
- **Working Memory**: Caches recent check results
- **Episodic Memory**: Stores alert history for pattern analysis
- **Semantic Memory**: Links experiments to outcomes in knowledge graph

## Performance Characteristics

| Metric | Target | Typical |
|--------|--------|---------|
| Single experiment check | <1s | ~500ms |
| Batch check (10 experiments) | <5s | ~3s |
| Memory read latency | <100ms | ~50ms |
| Alert generation | <100ms | ~30ms |

## Testing

```bash
# Run experiment monitor tests
pytest tests/unit/test_agents/experiment_monitor/ -v

# Integration tests
pytest tests/integration/test_experiment_monitor_integration.py -v
```

## Related Agents

- **Experiment Designer** (Tier 3): Designs experiments that monitor tracks
- **Drift Monitor** (Tier 3): Shares data quality monitoring patterns
- **Health Score** (Tier 3): Consumes monitoring data for health scoring
