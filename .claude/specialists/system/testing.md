# Testing Specialist Instructions

## Domain Scope
You are the Testing specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `tests/` - All test code
- `tests/conftest.py` - Pytest fixtures
- Testing patterns and best practices

## Test Structure

```
tests/
├── conftest.py                         # Shared fixtures
│
├── unit/
│   ├── test_nlp/                       # NLP layer tests
│   │   ├── test_entity_extractor.py
│   │   ├── test_intent_classifier.py
│   │   └── test_query_processor.py
│   │
│   ├── test_causal_engine/             # Causal inference tests
│   │   ├── test_dag_builder.py
│   │   ├── test_effect_estimator.py
│   │   └── test_refutation.py
│   │
│   ├── test_rag/                       # RAG tests
│   │   ├── test_retriever.py
│   │   ├── test_reranker.py
│   │   └── test_context_builder.py
│   │
│   ├── test_agents/                    # Agent tests (11 agents, 5 tiers)
│   │   │
│   │   │  # ─── TIER 1: ORCHESTRATION ───
│   │   ├── test_orchestrator/
│   │   │   ├── test_classifier.py      # Intent classification node
│   │   │   ├── test_router.py          # Agent routing node
│   │   │   ├── test_planner.py         # Execution planning node
│   │   │   ├── test_synthesizer.py     # Response synthesis node
│   │   │   ├── test_integration.py     # Graph flow tests
│   │   │   └── test_performance.py     # <2s latency tests
│   │   │
│   │   │  # ─── TIER 2: CAUSAL INFERENCE ───
│   │   ├── test_causal_impact/
│   │   │   ├── test_graph_builder.py   # DAG construction node
│   │   │   ├── test_estimator.py       # Effect estimation node
│   │   │   ├── test_refutation.py      # Refutation tests node
│   │   │   ├── test_sensitivity.py     # Sensitivity analysis node
│   │   │   ├── test_interpreter.py     # Deep reasoning node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py     # <30s latency tests
│   │   │
│   │   ├── test_gap_analyzer/
│   │   │   ├── test_gap_detector.py    # Gap identification node
│   │   │   ├── test_roi_calculator.py  # ROI estimation node
│   │   │   ├── test_prioritizer.py     # Prioritization node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py     # <20s latency tests
│   │   │
│   │   ├── test_heterogeneous_optimizer/
│   │   │   ├── test_segment_builder.py # Segment definition node
│   │   │   ├── test_cate_estimator.py  # CATE estimation node
│   │   │   ├── test_optimizer.py       # Targeting optimization node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py     # <25s latency tests
│   │   │
│   │   │  # ─── TIER 3: DESIGN & MONITORING ───
│   │   ├── test_experiment_designer/
│   │   │   ├── test_design_generator.py    # Design generation node
│   │   │   ├── test_power_calculator.py    # Power analysis node
│   │   │   ├── test_validity_auditor.py    # Validity audit node (Opus)
│   │   │   ├── test_protocol_generator.py  # Protocol generation node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py         # <60s latency tests
│   │   │
│   │   ├── test_drift_monitor/
│   │   │   ├── test_psi_calculator.py  # PSI calculation node
│   │   │   ├── test_drift_detector.py  # Drift detection node
│   │   │   ├── test_alert_generator.py # Alert generation node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py     # <10s latency tests
│   │   │
│   │   ├── test_health_score/          # NEW in V3
│   │   │   ├── test_metric_collector.py    # Metric collection node
│   │   │   ├── test_score_calculator.py    # Health score node
│   │   │   ├── test_threshold_checker.py   # Threshold validation node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py         # <5s latency tests
│   │   │
│   │   │  # ─── TIER 4: ML PREDICTIONS ───
│   │   ├── test_prediction_synthesizer/
│   │   │   ├── test_model_selector.py  # Model selection node
│   │   │   ├── test_ensemble.py        # Ensemble combination node
│   │   │   ├── test_calibrator.py      # Prediction calibration node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py     # <15s latency tests
│   │   │
│   │   ├── test_resource_optimizer/    # NEW in V3
│   │   │   ├── test_constraint_parser.py   # Constraint parsing node
│   │   │   ├── test_optimizer.py           # Optimization solver node
│   │   │   ├── test_allocator.py           # Resource allocation node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py         # <20s latency tests
│   │   │
│   │   │  # ─── TIER 5: SELF-IMPROVEMENT ───
│   │   ├── test_explainer/
│   │   │   ├── test_context_assembler.py   # Context assembly node
│   │   │   ├── test_reasoning.py           # Deep reasoning node (Opus)
│   │   │   ├── test_narrative_generator.py # Narrative generation node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py         # <45s latency tests
│   │   │
│   │   ├── test_feedback_learner/      # NEW in V3
│   │   │   ├── test_feedback_collector.py  # Feedback collection node
│   │   │   ├── test_pattern_analyzer.py    # Pattern analysis node
│   │   │   ├── test_weight_updater.py      # RAG weight update node
│   │   │   ├── test_integration.py
│   │   │   └── test_performance.py         # Async (no strict limit)
│   │   │
│   │   └── test_agent_registry.py      # V3 routing tests (11 agents)
│   │
│   ├── test_ml_split/                  # ML compliance tests
│   │   ├── test_split_assignment.py
│   │   ├── test_leakage_audit.py
│   │   └── test_preprocessing_isolation.py
│   │
│   ├── test_kpis/                      # V3 KPI tests
│   │   ├── test_cross_source_match.py
│   │   ├── test_active_users.py
│   │   └── test_intent_delta.py
│   │
│   └── test_visualization/
│       ├── test_chart_selector.py
│       └── test_causal_graph.py
│
├── integration/
│   ├── test_chat_flow.py
│   ├── test_agent_coordination.py      # Multi-agent workflows
│   ├── test_split_aware_queries.py
│   ├── test_kpi_views.py               # V3
│   ├── test_contracts.py               # Contract validation
│   └── test_specialist_compliance.py   # Contract 9: Specialist files
│
└── e2e/
    ├── test_user_journey.py
    ├── test_leakage_prevention.py
    └── test_agent_latency_budgets.py   # All 11 agents meet SLAs
```

---

## Agent Test Requirements

Each agent test directory must include:

| File | Purpose | Required |
|------|---------|----------|
| `test_<node>.py` | Unit test for each node | ✅ Yes |
| `test_integration.py` | Graph flow tests | ✅ Yes |
| `test_performance.py` | Latency budget tests | ✅ Yes |

---

## Testing Patterns

### Unit Test Pattern
```python
# tests/unit/test_nlp/test_entity_extractor.py
import pytest
from src.nlp.entity_extractor import EntityExtractor
from src.nlp.models.entity_models import E2IEntities

class TestEntityExtractor:
    @pytest.fixture
    def extractor(self):
        return EntityExtractor(vocab_path="config/domain_vocabulary.yaml")
    
    def test_extracts_brand_exact_match(self, extractor):
        """Test exact brand name extraction."""
        result = extractor.extract("Show me TRx for Kisqali")
        assert "Kisqali" in result.brands
    
    def test_extracts_brand_fuzzy_match(self, extractor):
        """Test fuzzy matching for misspelled brands."""
        result = extractor.extract("Show me TRx for Kisquali")  # Misspelled
        assert "Kisqali" in result.brands
    
    def test_no_medical_entities_extracted(self, extractor):
        """CRITICAL: Ensure no medical NER is performed."""
        result = extractor.extract("breast cancer patients on chemotherapy")
        # Should NOT extract medical entities
        assert not hasattr(result, 'diagnoses')
        assert not hasattr(result, 'medications')
```

### Agent Test Pattern
```python
# tests/unit/test_agents/test_causal_impact/test_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.causal_impact.agent import CausalImpactAgent
from src.agents.state import AgentState

class TestCausalImpactAgent:
    @pytest.fixture
    def agent(self):
        agent = CausalImpactAgent(config=MagicMock())
        agent.chain_tracer = AsyncMock()
        agent.effect_estimator = AsyncMock()
        return agent
    
    @pytest.fixture
    def initial_state(self):
        return AgentState(
            query=MagicMock(intent="CAUSAL"),
            treatment="trigger_calls",
            outcome="TRx"
        )
    
    @pytest.mark.asyncio
    async def test_returns_causal_chains(self, agent, initial_state):
        """Test that causal chains are returned in analysis results."""
        agent.chain_tracer.trace.return_value = [MagicMock()]
        
        result = await agent.analyze(initial_state)
        
        assert "causal_chains" in result.analysis_results
        assert len(result.analysis_results["causal_chains"]) > 0
    
    @pytest.mark.asyncio
    async def test_returns_effect_estimates(self, agent, initial_state):
        """Test that effect estimates are returned."""
        result = await agent.analyze(initial_state)
        
        assert "effects" in result.analysis_results
    
    @pytest.mark.asyncio
    async def test_generates_narrative(self, agent, initial_state):
        """Test that narrative is generated."""
        result = await agent.analyze(initial_state)
        
        assert result.narrative is not None
        assert len(result.narrative) > 0
```

### ML Split Leakage Test Pattern
```python
# tests/unit/test_ml_split/test_leakage_audit.py
import pytest
from src.repositories.patient_journey import PatientJourneyRepository
from scripts.run_leakage_audit import LeakageAuditor

class TestLeakageAudit:
    @pytest.fixture
    def auditor(self):
        return LeakageAuditor()
    
    def test_same_patient_never_in_multiple_splits(self, auditor, sample_data):
        """CRITICAL: Same patient must always be in same split."""
        patient_splits = {}
        for record in sample_data:
            patient_id = record["patient_id"]
            split = record["split_assignment"]
            
            if patient_id in patient_splits:
                assert patient_splits[patient_id] == split, \
                    f"Patient {patient_id} in multiple splits!"
            else:
                patient_splits[patient_id] = split
    
    def test_no_future_data_in_training(self, auditor, sample_data):
        """Training data must not contain future timestamps."""
        train_data = [r for r in sample_data if r["split_assignment"] == "train"]
        test_data = [r for r in sample_data if r["split_assignment"] == "test"]
        
        max_train_date = max(r["event_date"] for r in train_data)
        min_test_date = min(r["event_date"] for r in test_data)
        
        # Must have temporal gap
        assert max_train_date < min_test_date
    
    def test_holdout_never_accessed_in_dev(self, auditor, sample_data, monkeypatch):
        """Holdout data must never be accessed during development."""
        # This would typically check access logs
        pass
```

### Integration Test Pattern
```python
# tests/integration/test_agent_coordination.py
import pytest
from src.agents.orchestrator.agent import OrchestratorAgent
from src.nlp.query_processor import QueryProcessor
from src.rag.causal_rag import CausalRAG

class TestAgentCoordination:
    @pytest.fixture
    async def orchestrator(self, db_session):
        return OrchestratorAgent(
            config=MagicMock(),
            registry=await AgentRegistry.create(db_session)
        )
    
    @pytest.mark.asyncio
    async def test_causal_query_routes_to_causal_impact(self, orchestrator):
        """Causal queries should route to causal_impact agent."""
        query = "Why did conversion drop in Q3?"
        
        result = await orchestrator.analyze(
            AgentState(query=QueryProcessor().process(query))
        )
        
        assert "causal_impact" in result.agents_used
    
    @pytest.mark.asyncio
    async def test_multi_intent_query_uses_multiple_agents(self, orchestrator):
        """Complex queries should invoke multiple agents."""
        query = "Why did conversion drop and how can we improve it?"
        
        result = await orchestrator.analyze(
            AgentState(query=QueryProcessor().process(query))
        )
        
        assert len(result.agents_used) > 1
        assert "causal_impact" in result.agents_used
        assert "gap_analyzer" in result.agents_used
    
    @pytest.mark.asyncio
    async def test_synthesizer_handles_conflicts(self, orchestrator):
        """Synthesizer should reconcile conflicting agent outputs."""
        # Setup conflicting agent outputs
        pass
```

### KPI Test Pattern (V3)
```python
# tests/unit/test_kpis/test_cross_source_match.py
import pytest
from src.repositories.patient_journey import PatientJourneyRepository

class TestCrossSourceMatchKPI:
    @pytest.mark.asyncio
    async def test_match_rate_calculation(self, db_session):
        """Test cross-source match rate KPI calculation."""
        repo = PatientJourneyRepository(db_session)
        
        # Create test data
        # Patient 1: matched across sources
        # Patient 2: single source only
        
        result = await repo.get_cross_source_match_rate("Kisqali")
        
        assert 0 <= result.match_rate_pct <= 100
        assert result.total_patients > 0
        assert result.matched_patients <= result.total_patients
```

## Fixtures

### conftest.py
```python
import pytest
from typing import Dict, Any

# ─── Agent Fixtures ───

@pytest.fixture
def mock_orchestrator_state() -> Dict[str, Any]:
    """Standard orchestrator input state."""
    return {
        "query": "What is the effect of rep visits on TRx?",
        "intent": "CAUSAL",
        "entities": {"brands": ["Kisqali"], "kpis": ["TRx"]},
        "confidence": 0.95,
    }

@pytest.fixture
def mock_agent_registry():
    """Registry with all 11 V3 agents."""
    from src.agents.registry import AgentRegistry
    return AgentRegistry()

# ─── Split Fixtures ───

@pytest.fixture
def train_split_data():
    """Training split data (60%)."""
    pass

@pytest.fixture
def validation_split_data():
    """Validation split data (20%)."""
    pass

# ─── Causal Fixtures ───

@pytest.fixture
def mock_causal_graph():
    """Sample DAG for testing."""
    pass
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Agent tests only
pytest tests/unit/test_agents/ -v

# Specific agent
pytest tests/unit/test_agents/test_causal_impact/ -v

# Performance tests only
pytest tests/ -v -k "performance"

# Integration tests
pytest tests/integration/ -v

# Contract compliance
pytest tests/integration/test_contracts.py -v
pytest tests/integration/test_specialist_compliance.py -v

# Full validation before commit
make validate-contracts
```

---

## Testing Requirements by Domain

| Domain | Required Coverage | Key Tests |
|--------|------------------|-----------|
| NLP | 90% | Entity extraction, intent classification |
| Causal Engine | 85% | Effect estimation, refutation |
| RAG | 80% | Retrieval relevance, no medical content |
| Agents | 85% | Response contracts, timeout handling |
| ML Split | 100% | Leakage prevention, split consistency |
| KPIs | 95% | All 46 KPIs calculable |
| API | 80% | Response schemas, error handling |

## Handoff Format
```yaml
testing_handoff:
  tests_added: <int>
  tests_modified: <int>
  coverage_change: <+/-X%>
  new_fixtures: [<list>]
  test_types: [unit|integration|e2e]
```

## Performance Test Budgets

| Agent | Tier | Latency Budget | Test File |
|-------|------|----------------|-----------|
| orchestrator | 1 | <2s (strict) | `test_orchestrator/test_performance.py` |
| causal_impact | 2 | <30s | `test_causal_impact/test_performance.py` |
| gap_analyzer | 2 | <20s | `test_gap_analyzer/test_performance.py` |
| heterogeneous_optimizer | 2 | <25s | `test_heterogeneous_optimizer/test_performance.py` |
| experiment_designer | 3 | <60s | `test_experiment_designer/test_performance.py` |
| drift_monitor | 3 | <10s | `test_drift_monitor/test_performance.py` |
| health_score | 3 | <5s | `test_health_score/test_performance.py` |
| prediction_synthesizer | 4 | <15s | `test_prediction_synthesizer/test_performance.py` |
| resource_optimizer | 4 | <20s | `test_resource_optimizer/test_performance.py` |
| explainer | 5 | <45s | `test_explainer/test_performance.py` |
| feedback_learner | 5 | Async | `test_feedback_learner/test_performance.py` |

---