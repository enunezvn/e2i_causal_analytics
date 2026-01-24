# CLAUDE.md - Tier 0: ML Foundation

## Overview

Tier 0 (ML Foundation) is the **base layer** of the E2I 21-agent architecture. All other tiers depend on models, features, and infrastructure established by these 8 agents. Tier 0 handles the complete ML lifecycle from problem definition to production deployment.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 0: ML FOUNDATION (This Tier)                                      │
│                                                                         │
│  scope_definer ──▶ cohort_constructor ──▶ data_preparer ──▶             │
│  model_selector ──▶ model_trainer                                       │
│                                                              │          │
│                                                              ▼          │
│  observability_connector ◀── model_deployer ◀── feature_analyzer       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 1-5: Consume Tier 0 Outputs                                       │
│  - prediction_synthesizer ← Deployed models                             │
│  - drift_monitor ← Baseline metrics                                     │
│  - explainer ← SHAP analyses                                            │
│  - causal_impact ← Feature relationships                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Agents in This Tier

| Agent | Type | SLA | Primary Output |
|-------|------|-----|----------------|
| `scope_definer` | Standard | <5s | ScopeSpec, SuccessCriteria |
| `cohort_constructor` | Standard | <120s | CohortDefinition, EligibilityLog |
| `data_preparer` | Standard | <60s | QCReport, BaselineMetrics |
| `model_selector` | Standard | <120s | ModelCandidate, SelectionRationale |
| `model_trainer` | Standard | varies | TrainedModel, ValidationMetrics |
| `feature_analyzer` | **Hybrid** | <120s | SHAPAnalysis, FeatureImpacts |
| `model_deployer` | Standard | <30s | DeploymentManifest, VersionRecord |
| `observability_connector` | Standard | <100ms | Spans, QualityMetrics |

## Directory Structure

```
src/agents/ml_foundation/
├── __init__.py
├── CLAUDE.md                    # This file
│
├── scope_definer/
│   ├── __init__.py
│   ├── agent.py                 # Main agent implementation
│   ├── scope_builder.py         # Scope specification logic
│   ├── criteria_validator.py    # Success criteria validation
│   ├── prompts.py               # LLM prompts
│   └── CLAUDE.md                # Agent-specific instructions
│
├── data_preparer/
│   ├── __init__.py
│   ├── agent.py
│   ├── quality_checker.py       # Great Expectations integration
│   ├── baseline_computer.py     # Baseline metrics computation
│   ├── leakage_detector.py      # Data leakage detection
│   ├── prompts.py
│   └── CLAUDE.md
│
├── model_selector/
│   ├── __init__.py
│   ├── agent.py
│   ├── algorithm_registry.py    # Candidate algorithm catalog
│   ├── baseline_comparator.py   # Compare to baselines
│   ├── prompts.py
│   └── CLAUDE.md
│
├── model_trainer/
│   ├── __init__.py
│   ├── agent.py
│   ├── training_orchestrator.py # Coordinate training
│   ├── split_enforcer.py        # ML split enforcement
│   ├── hyperparameter_tuner.py  # Optuna integration
│   ├── prompts.py
│   └── CLAUDE.md
│
├── feature_analyzer/
│   ├── __init__.py
│   ├── agent.py                 # HYBRID: Computation + Interpretation
│   ├── shap_computer.py         # SHAP value computation
│   ├── interaction_detector.py  # Feature interaction analysis
│   ├── importance_narrator.py   # NL explanation generation
│   ├── prompts.py
│   └── CLAUDE.md
│
├── model_deployer/
│   ├── __init__.py
│   ├── agent.py
│   ├── registry_manager.py      # MLflow registry operations
│   ├── deployment_orchestrator.py # Deploy/promote/rollback
│   ├── endpoint_manager.py      # BentoML endpoint management
│   ├── prompts.py
│   └── CLAUDE.md
│
└── observability_connector/
    ├── __init__.py
    ├── agent.py
    ├── opik_emitter.py          # Emit spans to Opik
    ├── metrics_collector.py     # Quality metrics collection
    ├── context_propagator.py    # Trace/span context propagation
    ├── prompts.py
    └── CLAUDE.md
```

## Database Tables (Migration 007)

Tier 0 agents primarily write to these 8 ML tables:

| Table | Primary Writers | Primary Readers |
|-------|-----------------|-----------------|
| `ml_experiments` | scope_definer | data_preparer, model_selector, model_trainer |
| `ml_model_registry` | model_selector, model_deployer | prediction_synthesizer, orchestrator |
| `ml_training_runs` | model_trainer | feature_analyzer, model_deployer, feedback_learner |
| `ml_feature_store` | data_preparer | model_trainer, prediction_synthesizer |
| `ml_data_quality_reports` | data_preparer | model_trainer (gate), drift_monitor |
| `ml_shap_analyses` | feature_analyzer | explainer, causal_impact |
| `ml_deployments` | model_deployer | prediction_synthesizer, health_score |
| `ml_observability_spans` | observability_connector | health_score, orchestrator |

## Memory Architecture

Tier 0 agents use all 4 memory types:

### Working Memory (Redis)
- **All agents**: Active context, scratchpad, immediate state
- TTL: 86400 seconds (24 hours)
- Implementation: `src/memory/working_memory.py`

### Episodic Memory (Supabase/pgvector)
- **scope_definer**: Past scope definitions for similar use cases
- **data_preparer**: Historical QC reports and patterns
- **model_trainer**: Training run logs and outcomes
- **model_deployer**: Deployment history and rollback events
- **observability_connector**: Span archives
- Implementation: `src/memory/episodic_memory.py`

### Procedural Memory (Supabase/pgvector)
- **scope_definer**: Successful scope patterns by use case type
- **data_preparer**: Effective validation patterns
- **model_selector**: Algorithm selection heuristics
- **model_trainer**: Winning hyperparameter configurations
- **model_deployer**: Successful deployment procedures
- Implementation: `src/memory/procedural_memory.py`

### Semantic Memory (FalkorDB/Graphity)
- **feature_analyzer**: Feature relationships, interaction graphs
- Feeds into: causal_impact, explainer
- Implementation: `src/memory/semantic_memory.py`

## MLOps Tool Integration

| Tool | Location | Primary Agents |
|------|----------|----------------|
| MLflow | `src/mlops/mlflow_client.py` | model_trainer, model_selector, model_deployer |
| Opik | `src/mlops/opik_connector.py` | observability_connector |
| Great Expectations | `src/mlops/great_expectations_validator.py` | data_preparer |
| Feast | `src/mlops/feast_client.py` | data_preparer, model_trainer |
| Optuna | `src/mlops/optuna_tuner.py` | model_trainer |
| SHAP | `src/mlops/shap_explainer.py` | feature_analyzer |
| BentoML | `src/mlops/bentoml_service.py` | model_deployer |

## Data Flow

### Internal Tier 0 Flow

```
1. scope_definer
   └─▶ ScopeSpec ──▶ ml_experiments
   └─▶ SuccessCriteria ──▶ ml_experiments

2. data_preparer (reads ScopeSpec)
   └─▶ QCReport ──▶ ml_data_quality_reports
   └─▶ BaselineMetrics ──▶ ml_data_quality_reports
   └─▶ FeatureDefinitions ──▶ ml_feature_store

3. model_selector (reads ScopeSpec, QCReport)
   └─▶ ModelCandidate ──▶ ml_model_registry (stage: development)

4. model_trainer (reads ModelCandidate, QCReport must pass)
   └─▶ TrainedModel ──▶ ml_training_runs
   └─▶ ValidationMetrics ──▶ ml_training_runs

5. feature_analyzer (reads TrainedModel)
   └─▶ SHAPAnalysis ──▶ ml_shap_analyses
   └─▶ FeatureInteractions ──▶ semantic_memory

6. model_deployer (reads TrainedModel, ValidationMetrics)
   └─▶ DeploymentManifest ──▶ ml_deployments
   └─▶ VersionRecord ──▶ ml_model_registry (stage: production)

7. observability_connector (always running)
   └─▶ Spans ──▶ ml_observability_spans
   └─▶ QualityMetrics ──▶ ml_observability_spans
```

### Handoffs to Tier 1-5

| Output | From | To | Purpose |
|--------|------|-----|---------|
| Deployed model endpoints | model_deployer | prediction_synthesizer | Run inference |
| SHAP analyses | feature_analyzer | explainer | Generate explanations |
| Feature relationships | feature_analyzer | causal_impact | Inform causal graph |
| Baseline metrics | data_preparer | drift_monitor | Detect distribution drift |
| Observability data | observability_connector | health_score | System health |
| Model metadata | model_deployer | orchestrator | Route by capability |

## Implementation Patterns

### Base Agent Pattern

All Tier 0 agents extend `BaseAgent`:

```python
from src.agents.base_agent import BaseAgent
from src.database.repositories.ml_experiment import MLExperimentRepository

class ScopeDefinerAgent(BaseAgent):
    """Scope Definer: Problem definition and success criteria."""
    
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 5
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.experiment_repo = MLExperimentRepository()
        
    async def execute(self, state: AgentState) -> AgentState:
        """Main execution method."""
        # Implementation
        pass
```

### Hybrid Agent Pattern (feature_analyzer)

```python
class FeatureAnalyzerAgent(BaseAgent):
    """Feature Analyzer: SHAP computation + LLM interpretation."""
    
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "hybrid"  # Computation + LLM
    sla_seconds = 120
    
    async def execute(self, state: AgentState) -> AgentState:
        # Node 1: Computation (no LLM)
        shap_values = await self._compute_shap(state)
        
        # Node 2: Interpretation (LLM)
        interpretation = await self._interpret_shap(shap_values)
        
        return state.with_updates(
            shap_analysis=shap_values,
            interpretation=interpretation
        )
```

### Error Handling

All agents use the standard error handling chain:

```python
from src.agents.errors import AgentError, RetryConfig, FallbackChain

retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0
)

fallback_chain = FallbackChain([
    "claude-sonnet-4-20250514",  # Primary
    "claude-haiku",              # Fallback 1
    "template_response"          # Fallback 2
])
```

### Computation Fallback (for model_trainer, feature_analyzer)

```python
computation_fallback = ComputationFallbackChain([
    "CausalForest",    # Primary
    "LinearDML",       # Fallback 1
    "OLS"              # Fallback 2
])
```

## Testing

### Unit Tests

```
tests/unit/test_agents/test_ml_foundation/
├── test_scope_definer.py
├── test_data_preparer.py
├── test_model_selector.py
├── test_model_trainer.py
├── test_feature_analyzer.py
├── test_model_deployer.py
└── test_observability_connector.py
```

### Integration Tests

```
tests/integration/
├── test_ml_pipeline_flow.py        # Full Tier 0 pipeline
├── test_tier0_to_tier4_handoff.py  # Model handoff
└── test_memory_integration.py      # Memory across agents
```

### Key Test Scenarios

1. **Scope → Data → Model flow**: End-to-end pipeline
2. **QC gate enforcement**: Training blocked on QC failure
3. **SHAP computation accuracy**: Verify SHAP values
4. **Deployment rollback**: Test rollback procedures
5. **Observability span emission**: Verify Opik integration

## Configuration

### agent_config.yaml

```yaml
ml_foundation:
  scope_definer:
    tier: 0
    type: standard
    sla_seconds: 5
    memory_types: [working, episodic, procedural]
    tools: []
    
  data_preparer:
    tier: 0
    type: standard
    sla_seconds: 60
    memory_types: [working, episodic, procedural]
    tools: [great_expectations, feast]
    
  model_selector:
    tier: 0
    type: standard
    sla_seconds: 120
    memory_types: [working, episodic, procedural]
    tools: [mlflow]
    
  model_trainer:
    tier: 0
    type: standard
    sla_seconds: null  # Variable
    memory_types: [working, episodic, procedural]
    tools: [mlflow, optuna, feast]
    
  feature_analyzer:
    tier: 0
    type: hybrid
    sla_seconds: 120
    memory_types: [working, semantic]
    tools: [shap]
    
  model_deployer:
    tier: 0
    type: standard
    sla_seconds: 30
    memory_types: [working, episodic, procedural]
    tools: [mlflow, bentoml]
    
  observability_connector:
    tier: 0
    type: standard
    sla_seconds: 0.1  # Async
    memory_types: [working, episodic]
    tools: [opik]
```

## Key Principles

1. **Sequential Dependencies**: Agents must respect the pipeline order
2. **QC Gate**: model_trainer MUST verify QC passed before training
3. **Split Enforcement**: Always use ML splits (60/20/15/5)
4. **Observability**: All operations emit spans via observability_connector
5. **Memory Updates**: Update appropriate memory type after each operation
6. **Idempotency**: Operations should be safely retryable
7. **Artifact Versioning**: All models and experiments are versioned

## Common Pitfalls

1. **Skipping QC check**: Never train without verifying data quality
2. **Data leakage**: Always use split-aware data access
3. **Missing spans**: Ensure observability_connector captures all operations
4. **Stale baselines**: Update drift_monitor when baselines change
5. **Unversioned models**: Always register models in MLflow

## Related Documentation

- `src/mlops/CLAUDE.md` - MLOps integration details
- `src/memory/CLAUDE.md` - Memory architecture
- `src/database/migrations/007_mlops_tables.sql` - Schema reference
- `config/domain_vocabulary.yaml` - Agent vocabularies (V3.0.0)
