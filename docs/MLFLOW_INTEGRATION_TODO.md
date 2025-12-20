# MLflow Integration Todo List

**Decision**: Use MLflow 2.16.0+ for unified ML and LLM observability
**Rationale**: Native LangChain/LangGraph support, no dependency conflicts, self-hosted, unified platform

---

## 1. Backend Code Updates

### 1.1 Core MLflow Configuration
- [ ] Create `src/mlops/mlflow_config.py` module
  - [ ] Define MLflow tracking URI configuration
  - [ ] Set up experiment naming conventions (by brand, agent, use case)
  - [ ] Configure artifact storage paths
  - [ ] Add authentication if needed
  - [ ] **File**: `src/mlops/mlflow_config.py`

### 1.2 LangGraph Agent Integration
- [ ] Add MLflow callbacks to all 11 agents
  - [ ] **Tier 1**: Orchestrator Agent (`src/agents/orchestrator/agent.py`)
  - [ ] **Tier 2**: Causal Impact Agent (`src/agents/causal_impact/agent.py`)
  - [ ] **Tier 2**: Gap Analyzer Agent (`src/agents/gap_analyzer/agent.py`)
  - [ ] **Tier 2**: Heterogeneous Optimizer Agent (`src/agents/heterogeneous_optimizer/agent.py`)
  - [ ] **Tier 3**: Drift Monitor Agent (`src/agents/drift_monitor/agent.py`)
  - [ ] **Tier 3**: Experiment Designer Agent (`src/agents/experiment_designer/agent.py`)
  - [ ] **Tier 3**: Health Score Agent (`src/agents/health_score/agent.py`)
  - [ ] **Tier 4**: Prediction Synthesizer Agent (`src/agents/prediction_synthesizer/agent.py`)
  - [ ] **Tier 4**: Resource Optimizer Agent (`src/agents/resource_optimizer/agent.py`)
  - [ ] **Tier 5**: Explainer Agent (`src/agents/explainer/agent.py`)
  - [ ] **Tier 5**: Feedback Learner Agent (`src/agents/feedback_learner/agent.py`)

### 1.3 MLflow Callback Implementation
- [ ] Implement `MlflowCallbackHandler` wrapper for LangGraph
  - [ ] Log prompts (input to Claude API)
  - [ ] Log completions (output from Claude API)
  - [ ] Log token usage (prompt_tokens, completion_tokens, total_tokens)
  - [ ] Log latency (time per API call)
  - [ ] Log agent metadata (tier, brand, KPI context)
  - [ ] **Reference**: [MLflow LangChain Integration](https://mlflow.org/docs/latest/llms/langchain/index.html)

### 1.4 RAG System Integration
- [ ] Add MLflow tracing to RAG pipeline (`src/rag/`)
  - [ ] Log retrieval queries
  - [ ] Log retrieved documents (top-k, scores)
  - [ ] Log reranking results
  - [ ] Log final context sent to LLM
  - [ ] **File**: `src/rag/retriever.py`, `src/rag/reranker.py`

### 1.5 Experiment Tracking
- [ ] Create experiment tracking utilities
  - [ ] Auto-create experiments per brand (Remibrutinib, Fabhalta, Kisqali)
  - [ ] Auto-create experiments per agent tier
  - [ ] Log hyperparameters (temperature, max_tokens, model version)
  - [ ] Log custom metrics (causal_confidence, gap_score, drift_magnitude)
  - [ ] **File**: `src/mlops/experiment_tracker.py`

---

## 2. Configuration Updates

### 2.1 Environment Variables
- [ ] Add to `.env` and `.env.dev`:
  ```bash
  # MLflow Configuration
  MLFLOW_TRACKING_URI=http://mlflow:5000
  MLFLOW_EXPERIMENT_NAME=e2i-causal-analytics
  MLFLOW_ARTIFACT_LOCATION=./mlflow-artifacts
  MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
  ```
  - [ ] **Files**: `.env.example`, `.env.dev`

### 2.2 Agent Configuration Files
- [ ] Update agent YAML configs to include MLflow settings
  - [ ] Add `mlflow_experiment` field to each agent config
  - [ ] Add `log_prompts: true` flag
  - [ ] Add `log_artifacts: true` flag
  - [ ] **Files**: `config/agents/*.yaml` (11 agent configs)

### 2.3 MLflow Server Configuration
- [ ] Verify MLflow backend store (PostgreSQL)
- [ ] Verify artifact storage (local volume in dev, S3/DO Spaces in prod)
- [ ] Configure MLflow autologging
  - [ ] **File**: `docker/mlflow/Dockerfile`, `docker-compose.yml`

---

## 3. Documentation Updates

### 3.1 Architecture Documentation
- [ ] Update `docs/ARCHITECTURE.md`
  - [ ] Add MLflow as unified observability layer
  - [ ] Document MLflow integration with LangGraph agents
  - [ ] Add architecture diagram showing MLflow position
  - [ ] Remove references to opik/langfuse

### 3.2 MLflow Setup Guide
- [ ] Create `docs/MLFLOW_SETUP.md`
  - [ ] Installation instructions
  - [ ] Configuration walkthrough
  - [ ] Experiment naming conventions
  - [ ] Accessing MLflow UI (http://localhost:5000)
  - [ ] Querying experiments programmatically

### 3.3 Developer Guide
- [ ] Create `docs/MLFLOW_DEVELOPER_GUIDE.md`
  - [ ] How to add MLflow logging to new agents
  - [ ] How to log custom metrics
  - [ ] How to version prompts
  - [ ] How to compare experiments
  - [ ] Best practices for LLM observability

### 3.4 Operational Runbook
- [ ] Create `docs/MLFLOW_OPERATIONS.md`
  - [ ] Monitoring MLflow server health
  - [ ] Backup and restore procedures
  - [ ] Scaling artifact storage
  - [ ] Troubleshooting common issues

### 3.5 README Updates
- [ ] Update `README.md`
  - [ ] Add MLflow to technology stack section
  - [ ] Update observability section
  - [ ] Add link to MLflow UI
  - [ ] Add "LLM Observability with MLflow" section

---

## 4. Code Examples and Templates

### 4.1 Agent Template with MLflow
- [ ] Create `examples/agent_with_mlflow.py`
  - [ ] Complete example showing MLflow integration
  - [ ] Demonstrates prompt logging
  - [ ] Demonstrates metric logging
  - [ ] Demonstrates artifact logging (generated graphs, etc.)

### 4.2 RAG Pipeline with MLflow
- [ ] Create `examples/rag_with_mlflow.py`
  - [ ] Complete RAG pipeline example
  - [ ] Logs retrieval metrics
  - [ ] Logs reranking scores
  - [ ] Compares different retrieval strategies

### 4.3 Experiment Comparison Script
- [ ] Create `scripts/compare_experiments.py`
  - [ ] Load experiments from MLflow
  - [ ] Compare metrics across runs
  - [ ] Generate comparison reports
  - [ ] Identify best-performing configurations

### 4.4 Prompt Versioning Example
- [ ] Create `examples/prompt_versioning.py`
  - [ ] Store prompts as MLflow artifacts
  - [ ] Version prompts by experiment run
  - [ ] Load specific prompt versions
  - [ ] A/B test different prompts

---

## 5. Testing and Validation

### 5.1 Integration Tests
- [ ] Create `tests/integration/test_mlflow_integration.py`
  - [ ] Test MLflow connection
  - [ ] Test experiment creation
  - [ ] Test run logging
  - [ ] Test artifact storage
  - [ ] Test metric retrieval

### 5.2 Agent Logging Tests
- [ ] Create `tests/agents/test_agent_mlflow_logging.py`
  - [ ] Test that all agents log to MLflow
  - [ ] Test that prompts are captured
  - [ ] Test that tokens are counted
  - [ ] Test that custom metrics are logged

### 5.3 Performance Tests
- [ ] Create `tests/performance/test_mlflow_overhead.py`
  - [ ] Measure latency added by MLflow logging
  - [ ] Ensure < 50ms overhead per agent call
  - [ ] Test concurrent logging

---

## 6. Migration Tasks

### 6.1 Remove Old Observability Code
- [ ] Search codebase for opik/langfuse references
  - [ ] **Command**: `grep -r "opik\|langfuse" src/`
  - [ ] Remove any remaining imports
  - [ ] Remove configuration files

### 6.2 Update Dependencies
- [x] Remove langfuse from `requirements.txt` ✅
- [ ] Verify MLflow version is 2.16.0+
- [ ] Document MLflow as core dependency

---

## 7. Production Readiness

### 7.1 MLflow Server Deployment
- [ ] Plan MLflow production deployment strategy
  - [ ] Decide on artifact storage (S3, DO Spaces, GCS)
  - [ ] Configure authentication (basic auth, OAuth)
  - [ ] Set up HTTPS/TLS
  - [ ] Plan for high availability

### 7.2 Data Retention Policy
- [ ] Define MLflow data retention policy
  - [ ] How long to keep experiments
  - [ ] How to archive old runs
  - [ ] Artifact cleanup strategy

### 7.3 Monitoring and Alerting
- [ ] Set up monitoring for MLflow server
  - [ ] Disk space alerts (artifacts can grow large)
  - [ ] Database connection monitoring
  - [ ] API availability checks

---

## 8. Advanced Features (Future)

### 8.1 Model Registry
- [ ] Explore MLflow Model Registry for agent versions
  - [ ] Register production agent configurations
  - [ ] Track agent deployment history
  - [ ] Enable agent rollback

### 8.2 AutoML Integration
- [ ] Integrate MLflow with hyperparameter optimization
  - [ ] Log Optuna trials to MLflow
  - [ ] Track agent performance across configurations
  - [ ] Auto-select best agent configurations

### 8.3 Custom Metrics and Visualizations
- [ ] Build custom MLflow plugins
  - [ ] Causal chain visualization
  - [ ] Agent collaboration diagrams
  - [ ] KPI impact dashboards

---

## Priority Order

**Phase 1 (Immediate - Week 1)**:
1. Core MLflow configuration (1.1)
2. LangGraph agent integration (1.2, 1.3)
3. Environment variables (2.1)
4. Basic documentation (3.1, 3.5)

**Phase 2 (Short-term - Week 2)**:
1. RAG system integration (1.4)
2. Experiment tracking utilities (1.5)
3. Developer guide (3.3)
4. Code examples (4.1, 4.2)

**Phase 3 (Medium-term - Week 3-4)**:
1. Testing and validation (5.1, 5.2, 5.3)
2. Migration cleanup (6.1, 6.2)
3. Operational runbook (3.4)
4. Experiment comparison tools (4.3, 4.4)

**Phase 4 (Production - Month 2)**:
1. Production deployment (7.1)
2. Data retention (7.2)
3. Monitoring and alerting (7.3)

---

## Key MLflow Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow LangChain Integration](https://mlflow.org/docs/latest/llms/langchain/index.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

**Last Updated**: 2025-12-18
**Status**: Planning Phase
**Owner**: Development Team
