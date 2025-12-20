# Integration Contracts

## Purpose
This file defines the contracts between system components. Before completing any cross-domain task, validate that changes comply with these contracts.

**Last Updated**: 2025-12-04

---

## Contract 1: NLP → Agent Routing

### Producer: `src/nlp/query_processor.py`
### Consumer: `src/agents/registry.py`

```python
class ParsedQueryContract:
    """
    NLP must produce ParsedQuery with:
    - intent: Valid IntentType enum
    - entities: E2IEntities with at least one populated field
    - confidence: float 0.0-1.0
    """
    
    @staticmethod
    def validate(query: ParsedQuery) -> bool:
        assert query.intent in IntentType
        assert query.confidence >= 0.0 and query.confidence <= 1.0
        assert any([
            query.entities.brands,
            query.entities.regions,
            query.entities.kpis,
            query.entities.time_periods
        ]), "At least one entity type required"
        return True
```

### Validation Test
```bash
pytest tests/integration/test_nlp_to_agent.py
```

---

## Contract 2: Agent → Orchestrator Response

### Producer: All agents in `src/agents/*/agent.py`
### Consumer: `src/agents/orchestrator/synthesizer.py`

```python
class AgentResponseContract:
    """
    All agents must return AgentState with:
    - analysis_results: Dict with agent-specific keys
    - narrative: Non-empty string
    - confidence: float 0.0-1.0
    - processing_time_ms: int > 0
    """
    
    REQUIRED_KEYS_BY_AGENT = {
        "causal_impact": ["causal_chains", "effects"],
        "gap_analyzer": ["gaps", "roi_estimates"],
        "heterogeneous_optimizer": ["cate_by_segment"],
        "drift_monitor": ["drift_detected", "drift_metrics"],
        "experiment_designer": ["experiment_design"],
        "health_score": ["health_metrics"],
        "prediction_synthesizer": ["predictions", "model_metrics"],
        "resource_optimizer": ["resource_allocation"],
        "explainer": ["explanation"],
        "feedback_learner": ["learned_patterns"],
    }
    
    @staticmethod
    def validate(agent_name: str, state: AgentState) -> bool:
        required_keys = AgentResponseContract.REQUIRED_KEYS_BY_AGENT[agent_name]
        for key in required_keys:
            assert key in state.analysis_results, f"Missing {key} in {agent_name} response"
        assert len(state.narrative) > 0, "Narrative required"
        return True
```

### Validation Test
```bash
pytest tests/integration/test_agent_responses.py
```

---

## Contract 3: RAG → Agent Context

### Producer: `src/rag/causal_rag.py`
### Consumer: All agents

```python
class RAGContextContract:
    """
    RAG must provide RetrievalContext with:
    - results: List of at least 1 result if query is valid
    - Each result must have source attribution
    - No medical/clinical content
    """
    
    FORBIDDEN_SOURCES = [
        "pubmed", "clinical_trials", "fda", "ema",
        "drug_labels", "adverse_events"
    ]
    
    @staticmethod
    def validate(context: RetrievalContext) -> bool:
        for result in context.results:
            assert result.source not in RAGContextContract.FORBIDDEN_SOURCES, \
                f"Forbidden source: {result.source}"
            assert result.source_id is not None, "Source ID required"
        return True
```

### Validation Test
```bash
pytest tests/integration/test_rag_sources.py
```

---

## Contract 4: API → Frontend Response

### Producer: `src/api/routes/*.py`
### Consumer: `src/frontend/services/api.ts`

```typescript
interface ChatResponseContract {
  // Required fields
  conversation_id: string;      // UUID format
  message_id: string;           // UUID format
  response_text: string;        // Non-empty
  agents_used: string[];        // At least ["orchestrator"]
  confidence: number;           // 0.0-1.0
  processing_time_ms: number;   // > 0
  
  // Optional fields
  visualizations?: VisualizationData[];
}

interface VisualizationData {
  type: 'line' | 'bar' | 'pie' | 'causal_graph' | 'waterfall';
  title: string;
  data: any[];
  config: Record<string, any>;
}
```

### Validation Test
```bash
npm run test:api-contracts
```

---

## Contract 5: ML Split Enforcement

### Enforced By: `src/repositories/base.py`
### Consumers: All data access

```python
class SplitEnforcementContract:
    """
    CRITICAL: Prevents ML data leakage.
    
    Rules:
    1. Training queries MUST filter to split='train'
    2. Test/holdout data NEVER exposed in production
    3. Cross-split access logged and audited
    """
    
    @staticmethod
    def validate_query(query: str, intended_split: str, env: str) -> bool:
        if env == "production":
            assert "test" not in query.lower() or intended_split == "test_explicit"
            assert "holdout" not in query.lower()
        return True
```

### Validation Test
```bash
pytest tests/unit/test_ml_split/test_leakage_audit.py
```

---

## Contract 6: Causal Engine → Effect Estimates

### Producer: `src/causal_engine/effect_estimator.py`
### Consumer: Tier 2 agents

```python
class EffectEstimateContract:
    """
    All causal effect estimates must include:
    - Point estimate with confidence interval
    - P-value or equivalent uncertainty measure
    - Refutation test results
    - Sample size
    """
    
    @staticmethod
    def validate(estimate: EffectEstimate) -> bool:
        assert estimate.ate_ci_lower < estimate.ate < estimate.ate_ci_upper
        assert 0.0 <= estimate.p_value <= 1.0
        assert estimate.sample_size > 0
        assert estimate.refutation_passed is not None
        return True
```

### Validation Test
```bash
pytest tests/unit/test_causal_engine/test_effect_estimates.py
```

---

## Contract 7: Agent Tier Priority

### Enforced By: `src/agents/registry.py`
### Description: Routing priority based on tier

```python
class TierPriorityContract:
    """
    Agent selection follows tier priority:
    Tier 1 > Tier 2 > Tier 3 > Tier 4 > Tier 5
    
    Within same tier, select by intent specificity.
    """
    
    TIER_ORDER = {
        1: ["orchestrator"],
        2: ["causal_impact", "gap_analyzer", "heterogeneous_optimizer"],
        3: ["experiment_designer", "drift_monitor", "health_score"],
        4: ["prediction_synthesizer", "resource_optimizer"],
        5: ["explainer", "feedback_learner"],
    }
    
    AGENT_TYPES = {
        "standard": [
            "orchestrator", "gap_analyzer", "heterogeneous_optimizer",
            "drift_monitor", "health_score", "prediction_synthesizer",
            "resource_optimizer"
        ],
        "hybrid": ["causal_impact", "experiment_designer"],
        "deep": ["explainer", "feedback_learner"],
    }
    
    @staticmethod
    def get_routing_priority(intent: IntentType) -> List[str]:
        # Return agents in priority order for this intent
        pass
```

---

## Contract 8: KPI Calculation Accuracy

### Producer: `v_kpi_*` views in database
### Consumer: `src/api/routes/kpis.py`

```python
class KPIAccuracyContract:
    """
    All 46 KPIs must be:
    1. Calculable from existing tables
    2. Documented in kpi_definitions.yaml
    3. Tested for edge cases (nulls, zeros)
    """
    
    REQUIRED_KPIS = 46  # V3 count
    
    @staticmethod
    def validate_coverage() -> bool:
        calculated = len(get_calculable_kpis())
        assert calculated >= KPIAccuracyContract.REQUIRED_KPIS, \
            f"Only {calculated}/46 KPIs calculable"
        return True
```

### Validation Test
```bash
python scripts/validate_kpi_coverage.py
```

---

## Contract 9: Agent Specialist File Compliance (NEW)

### Enforced By: `make validate-specialists`
### Description: Ensures all agents have compliant specialist documentation

```python
class AgentSpecialistContract:
    """
    All 11 agents must have specialist files containing:
    1. State TypedDict definition
    2. Node implementations with async execute methods
    3. Error handling with fallback chains
    4. Performance budget compliance
    5. Integration contract references
    """
    
    REQUIRED_AGENTS = [
        "orchestrator",
        "causal_impact",
        "gap_analyzer", 
        "heterogeneous_optimizer",
        "experiment_designer",
        "drift_monitor",
        "health_score",
        "prediction_synthesizer",
        "resource_optimizer",
        "explainer",
        "feedback_learner",
    ]
    
    REQUIRED_SECTIONS = [
        "## State Definition",
        "## Node Implementations",
        "## Graph Assembly",
        "## Error Handling",
        "## Performance Budget",
    ]
    
    LATENCY_BUDGETS = {
        "orchestrator": 2000,           # 2s strict
        "causal_impact": 30000,         # 30s
        "gap_analyzer": 20000,          # 20s
        "heterogeneous_optimizer": 25000,  # 25s
        "experiment_designer": 60000,   # 60s
        "drift_monitor": 10000,         # 10s
        "health_score": 5000,           # 5s
        "prediction_synthesizer": 15000, # 15s
        "resource_optimizer": 20000,    # 20s
        "explainer": 45000,             # 45s
        "feedback_learner": None,       # Async, no limit
    }
    
    @staticmethod
    def validate_specialist_file(agent_name: str, content: str) -> bool:
        for section in AgentSpecialistContract.REQUIRED_SECTIONS:
            assert section in content, f"Missing {section} in {agent_name} specialist"
        return True
    
    @staticmethod
    def validate_all_specialists() -> bool:
        for agent in AgentSpecialistContract.REQUIRED_AGENTS:
            file_path = f"specialists/{agent.replace('_', '-')}.md"
            assert os.path.exists(file_path), f"Missing specialist file: {file_path}"
        return True
```

### Validation Test
```bash
pytest tests/integration/test_specialist_compliance.py
```

---

## Pre-Commit Validation

Run before any commit affecting multiple domains:

```bash
# Full contract validation
make validate-contracts

# Or individually:
pytest tests/integration/test_contracts.py -v
npm run test:contracts
python scripts/validate_kpi_coverage.py
pytest tests/integration/test_specialist_compliance.py
```

---

## Contract Violation Handling

When a contract violation is detected:

1. **STOP** - Do not proceed with the change
2. **IDENTIFY** - Which contract is violated
3. **TRACE** - Find the root cause (producer or consumer)
4. **FIX** - Update the violating component
5. **VERIFY** - Re-run contract tests
6. **DOCUMENT** - If contract needs updating, propose change

---

## Adding New Contracts

When adding cross-domain functionality:

1. Define contract in this file
2. Add validation function
3. Add integration test
4. Update `make validate-contracts`
5. Document in relevant specialist files

---

## Change Log

| Date | Contract | Change |
|------|----------|--------|
| 2025-12-04 | Contract 7 | Confirmed Experiment Designer in Tier 3 |
| 2025-12-04 | Contract 7 | Added AGENT_TYPES classification |
| 2025-12-04 | Contract 9 | Added Agent Specialist File Compliance contract |
| 2025-12-04 | All | Added Last Updated date |
