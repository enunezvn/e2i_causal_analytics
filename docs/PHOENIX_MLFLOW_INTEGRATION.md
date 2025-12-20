# MLflow + Phoenix Integration Guide

**Decision**: Hybrid observability stack combining MLflow and Phoenix
**Created**: 2025-12-18

---

## Architecture

```
E2I Causal Analytics Observability Stack
├── MLflow (port 5000)
│   └── ML Model Tracking & Experiment Management
├── Phoenix (port 6006)
│   └── LLM Tracing & Conversational AI Observability
└── Both feed into unified analytics
```

---

## Services Overview

### MLflow (http://localhost:5000)
**Purpose**: ML model tracking and experiment management

**Capabilities**:
- Traditional ML model versioning and tracking
- Hyperparameter logging
- Metric comparison across experiments
- Artifact storage (models, plots, data)
- Causal inference experiment tracking

**Use for**:
- Causal model experiments (DoWhy, EconML)
- Feature engineering experiments
- ML model comparison (churn prediction, propensity models)
- A/B test analysis tracking

### Phoenix (http://localhost:6006)
**Purpose**: LLM tracing and conversational AI observability

**Capabilities**:
- LangChain/LangGraph auto-instrumentation
- Prompt playground for testing
- User feedback collection
- Conversational trace visualization
- Token usage and cost tracking
- Latency monitoring
- Advanced prompt analytics

**Use for**:
- 11-agent system tracing
- Prompt engineering and optimization
- Agent conversation debugging
- Multi-turn dialogue analysis
- RAG pipeline observability

---

## Installation

Already configured in:
- ✅ `requirements.txt`: Phoenix packages added
- ✅ `docker-compose.yml`: Phoenix service configured
- ✅ `docker-compose.dev.yml`: Development overrides

---

## Quick Start

### 1. Start Services

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### 2. Access UIs

- **MLflow UI**: http://localhost:5000
- **Phoenix UI**: http://localhost:6006

### 3. Instrument Your Code

#### For LangChain/LangGraph Agents (Phoenix):

```python
from phoenix.trace.langchain import LangChainInstrumentor

# Initialize Phoenix instrumentation (once at app startup)
LangChainInstrumentor().instrument()

# Your LangGraph agents will now auto-trace to Phoenix!
# No code changes needed in agent implementations
```

#### For ML Experiments (MLflow):

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("causal-impact-analysis")

# Start a run
with mlflow.start_run(run_name="remibrutinib_q4_2025"):
    # Log parameters
    mlflow.log_param("brand", "remibrutinib")
    mlflow.log_param("treatment", "rep_visit")

    # Log metrics
    mlflow.log_metric("ate", 0.15)
    mlflow.log_metric("confidence", 0.95)

    # Log artifacts
    mlflow.log_artifact("causal_graph.png")
```

---

## Integration Patterns

### Pattern 1: Agent Execution with Dual Tracking

```python
import mlflow
from phoenix.trace.langchain import LangChainInstrumentor

# Initialize Phoenix (once at startup)
LangChainInstrumentor().instrument()

# Set MLflow experiment
mlflow.set_experiment("agent-orchestration")

# Execute agent with dual tracking
with mlflow.start_run(run_name="gap_analyzer_run"):
    # Phoenix automatically traces LangChain execution
    # MLflow tracks high-level metrics

    result = gap_analyzer_agent.run(query="What gaps exist in Fabhalta?")

    # Log agent-level metrics to MLflow
    mlflow.log_metric("gaps_identified", len(result.gaps))
    mlflow.log_metric("total_tokens", result.token_count)
    mlflow.log_metric("execution_time_sec", result.duration)

    # Phoenix already captured detailed traces
    # MLflow stores summary metrics
```

### Pattern 2: RAG Pipeline Observability

```python
from phoenix.trace import trace

# Phoenix traces the entire RAG pipeline
@trace
def rag_retrieval(query: str):
    # Retrieval step (Phoenix traces this)
    docs = vector_store.similarity_search(query, k=10)

    # Reranking step (Phoenix traces this)
    reranked = reranker.rerank(query, docs)

    # LLM generation step (Phoenix traces this)
    response = llm.generate(query=query, context=reranked)

    return response

# MLflow tracks experiment-level metrics
with mlflow.start_run(run_name="rag_experiment_v3"):
    response = rag_retrieval("What drives NRx for Kisqali?")

    # Log RAG metrics to MLflow
    mlflow.log_metric("retrieval_k", 10)
    mlflow.log_metric("response_relevance", score_relevance(response))
```

---

## Environment Variables

Add to `.env` and `.env.dev`:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=e2i-causal-analytics

# Phoenix Configuration
PHOENIX_HOST=phoenix
PHOENIX_PORT=6006
PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006
```

---

## Agent-Specific Integration

### Orchestrator Agent (Tier 1)

```python
# src/agents/orchestrator/agent.py
import mlflow
from phoenix.trace.langchain import LangChainInstrumentor

class OrchestratorAgent:
    def __init__(self):
        # Phoenix instruments automatically
        LangChainInstrumentor().instrument()

        # Set MLflow experiment
        self.mlflow_experiment = "orchestrator-coordination"

    def coordinate_agents(self, query: str):
        with mlflow.start_run(run_name=f"orchestration_{timestamp}"):
            # Phoenix traces all LangChain calls
            # MLflow logs coordination metrics

            result = self.graph.invoke({"query": query})

            # Log orchestration metrics
            mlflow.log_metric("agents_invoked", len(result.agents_used))
            mlflow.log_metric("total_execution_time", result.duration)

            return result
```

### Causal Impact Agent (Tier 2)

```python
# src/agents/causal_impact/agent.py
import mlflow
from phoenix.trace import trace

class CausalImpactAgent:
    @trace  # Phoenix traces this method
    def analyze_treatment_effect(self, treatment: str, outcome: str):
        with mlflow.start_run(run_name=f"causal_analysis_{treatment}"):
            # Run causal inference
            ate = self.estimate_ate(treatment, outcome)
            confidence = self.calculate_confidence()

            # Log to MLflow
            mlflow.log_param("treatment", treatment)
            mlflow.log_param("outcome", outcome)
            mlflow.log_metric("ate", ate)
            mlflow.log_metric("confidence", confidence)

            # Phoenix already captured LLM traces
            return {"ate": ate, "confidence": confidence}
```

---

## Monitoring Dashboard Setup

### MLflow Dashboard
- Navigate to http://localhost:5000
- Create experiments per:
  - **Brand**: Remibrutinib, Fabhalta, Kisqali
  - **Agent Tier**: Tier 1-5
  - **Use Case**: Causal analysis, gap detection, predictions

### Phoenix Dashboard
- Navigate to http://localhost:6006
- View traces by:
  - **Agent**: Filter by agent type
  - **Time Range**: Last hour, day, week
  - **Performance**: Sort by latency, tokens
  - **Errors**: View failed traces

---

## Best Practices

### 1. Experiment Naming Convention

**MLflow**:
```
{brand}_{agent_tier}_{use_case}_{version}
# Example: remibrutinib_tier2_causal_v3
```

**Phoenix**:
- Automatically tagged by agent name
- Add custom metadata via `@trace(metadata={...})`

### 2. Metric Logging

**Log to MLflow**:
- High-level metrics (success rate, execution time)
- Aggregate statistics (total tokens, cost)
- Business KPIs (gaps found, predictions made)

**Log to Phoenix**:
- Happens automatically for LangChain
- Individual prompt/response pairs
- Token-level details
- Latency per step

### 3. Artifact Storage

**MLflow Artifacts**:
- Causal graphs (PNG/SVG)
- Model checkpoints
- Experiment reports (PDF)

**Phoenix Artifacts**:
- Stored automatically (prompts, responses)
- Exportable via UI

---

## Troubleshooting

### Phoenix Not Receiving Traces

**Check instrumentation**:
```python
from phoenix.trace.langchain import LangChainInstrumentor

# Verify instrumentation
instrumentor = LangChainInstrumentor()
instrumentor.instrument()

# Check if instrumented
print(instrumentor.is_instrumented())  # Should be True
```

**Check endpoint**:
```bash
# Verify Phoenix is running
curl http://localhost:6006

# Check network connectivity from container
docker exec e2i-fastapi curl http://phoenix:6006
```

### MLflow Not Logging

**Check tracking URI**:
```python
import mlflow
print(mlflow.get_tracking_uri())  # Should be http://mlflow:5000
```

**Verify experiment exists**:
```python
mlflow.set_experiment("test-experiment")
mlflow.create_experiment("test-experiment")  # If doesn't exist
```

---

## Performance Impact

### Phoenix
- **Overhead**: <10ms per trace
- **Async logging**: Non-blocking
- **Storage**: ~1KB per trace

### MLflow
- **Overhead**: <50ms per run
- **Async logging**: Available (enable via config)
- **Storage**: Varies by artifacts

---

## Production Considerations

### 1. Data Retention
- **Phoenix**: Configure trace TTL (default: 30 days)
- **MLflow**: Archive old experiments quarterly

### 2. Access Control
- **Phoenix**: Add authentication layer (nginx basic auth)
- **MLflow**: Use built-in authentication (MLflow 2.0+)

### 3. Scaling
- **Phoenix**: Horizontal scaling supported
- **MLflow**: Use remote backend (PostgreSQL + S3)

### 4. Monitoring
- Set up alerts for:
  - High error rates (Phoenix)
  - Slow traces (>5s)
  - Disk usage (both)

---

## Resources

### Phoenix Documentation
- [Arize Phoenix Docs](https://arize.com/docs/phoenix/)
- [LangChain Integration](https://arize.com/docs/phoenix/integrations/python/langchain)
- [GitHub Repository](https://github.com/Arize-ai/phoenix)

### MLflow Documentation
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [LLM Tracking](https://mlflow.org/docs/latest/llms/index.html)
- [Tracking API](https://mlflow.org/docs/latest/tracking.html)

---

## Next Steps

1. **Immediate** (This week):
   - [ ] Verify services are running
   - [ ] Test Phoenix with single agent
   - [ ] Test MLflow with sample experiment
   - [ ] Document initial findings

2. **Short-term** (Next 2 weeks):
   - [ ] Instrument all 11 agents
   - [ ] Create MLflow experiments per brand
   - [ ] Set up Phoenix dashboards per agent tier
   - [ ] Train team on both UIs

3. **Medium-term** (Month 1-2):
   - [ ] Build custom analysis scripts
   - [ ] Set up automated reporting
   - [ ] Implement alerting
   - [ ] Optimize trace sampling

---

**Last Updated**: 2025-12-18
**Status**: Initial Setup
**Owner**: Development Team
