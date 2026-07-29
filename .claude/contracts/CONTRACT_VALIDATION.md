# E2I Contract Validation Matrix

**Last Updated**: 2025-12-23
**Status**: Phase 2 Complete - All Agent Contracts Validated
**Reference**: [AGENT_IMPLEMENTATION_AUDIT.md](../AGENT_IMPLEMENTATION_AUDIT.md)

---

## Overview

This document provides the validation matrix for all 18 E2I agents across:
- **4-Memory Architecture** compliance (memory_hooks.py)
- **DSPy Integration** compliance (dspy_integration.py)
- Contract specification adherence

---

## Compliance Summary

| Category | Compliant | Total | Rate | Status |
|----------|-----------|-------|------|--------|
| Memory Hooks (Tier 1-5) | 12 | 12 | 100% | ✅ COMPLETE |
| DSPy Integration (Tier 1-5) | 12 | 12 | 100% | ✅ COMPLETE |
| RAG DSPy Integration | 1 | 1 | 100% | ✅ COMPLETE |
| Tier 0 ML Foundation | N/A | 7 | N/A | OUT OF SCOPE |

---

## Tier 1: Orchestration Layer

### orchestrator

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/orchestrator/memory_hooks.py` |
| Memory Types | Working, Episodic, Semantic | Full tri-memory integration |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/orchestrator/dspy_integration.py` |
| DSPy Type | Hub | Central optimization coordinator |
| DSPy Signatures | AgentRoutingSignature, IntentClassificationSignature | Routes queries to appropriate agents |
| **Contract** | ✅ VALIDATED | tier1-contracts.md (lines 467-605) |

**Blocking Items**: None

---

### tool_composer

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/tool_composer/memory_hooks.py` |
| Memory Types | Working, Procedural | Tool sequence optimization |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/tool_composer/dspy_integration.py` |
| DSPy Type | Hybrid | Generates signals AND receives optimized prompts |
| DSPy Signatures | VisualizationConfigSignature, ToolSelectionSignature | Multi-faceted query decomposition |
| **Contract** | ✅ VALIDATED | tier1-contracts.md (extended) |

**Blocking Items**: None

---

## Tier 2: Causal Analytics Layer

### causal_impact

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/causal_impact/memory_hooks.py` |
| Memory Types | Working, Episodic, Semantic | Causal path discovery and storage |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/causal_impact/dspy_integration.py` |
| DSPy Type | Sender | Generates CausalAnalysisTrainingSignal |
| DSPy Signatures | CausalGraphSignature, EvidenceSynthesisSignature | Causal chain tracing |
| **Contract** | ✅ VALIDATED | tier2-contracts.md (lines 416-560) |

**Blocking Items**: None

**Training Signal Schema**:
```python
@dataclass
class CausalAnalysisTrainingSignal:
    # Input Context
    signal_id: str
    session_id: str
    query: str
    treatment_var: str
    outcome_var: str
    confounders_count: int
    # Graph Building Phase
    dag_nodes_count: int
    dag_edges_count: int
    adjustment_sets_found: int
    graph_confidence: float
    # Estimation Phase
    estimation_method: str
    ate_estimate: float
    ate_ci_width: float
    statistical_significance: bool
    effect_size: str  # small, medium, large
    sample_size: int
    # Refutation Phase
    refutation_tests_passed: int
    refutation_tests_failed: int
    overall_robust: bool
    # Outcome Metrics
    total_latency_ms: float
    confidence_score: float
    user_satisfaction: Optional[float]
    def compute_reward(self) -> float: ...
```

---

### gap_analyzer

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/gap_analyzer/memory_hooks.py` |
| Memory Types | Working, Episodic | Gap pattern memory |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/gap_analyzer/dspy_integration.py` |
| DSPy Type | Sender | Generates GapAnalysisTrainingSignal |
| DSPy Signatures | GapDetectionSignature, ROIEstimationSignature, EvidenceRelevanceSignature | Gap discovery |
| **Contract** | ✅ VALIDATED | tier2-contracts.md (lines 561-705) |

**Blocking Items**: None

**Training Signal Schema**:
```python
@dataclass
class GapAnalysisTrainingSignal:
    # Input Context
    signal_id: str
    session_id: str
    query: str
    brand: str
    metrics_analyzed: List[str]
    segments_analyzed: int
    # Detection Phase
    gaps_detected_count: int
    total_gap_value: float
    gap_types: List[str]
    # ROI Phase
    roi_estimates_count: int
    total_addressable_value: float
    avg_expected_roi: float
    high_roi_count: int
    # Prioritization Phase
    quick_wins_count: int
    strategic_bets_count: int
    prioritization_confidence: float
    # Output Quality
    actionable_recommendations: int
    # Outcome Metrics
    total_latency_ms: float
    user_satisfaction: Optional[float]
    def compute_reward(self) -> float: ...
```

---

### heterogeneous_optimizer

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/heterogeneous_optimizer/memory_hooks.py` |
| Memory Types | Working, Episodic | CATE analysis history |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/heterogeneous_optimizer/dspy_integration.py` |
| DSPy Type | Sender | Generates HeterogeneousOptimizationTrainingSignal |
| DSPy Signatures | SegmentIdentificationSignature, PolicyRecommendationSignature, EvidenceSynthesisSignature | Segment CATE |
| **Contract** | ✅ VALIDATED | tier2-contracts.md (lines 706-808) |

**Blocking Items**: None

**Training Signal Schema**:
```python
@dataclass
class HeterogeneousOptimizationTrainingSignal:
    # Input Context
    signal_id: str
    session_id: str
    query: str
    treatment_var: str
    outcome_var: str
    segment_vars_count: int
    effect_modifiers_count: int
    # CATE Estimation Phase
    overall_ate: float
    heterogeneity_score: float
    cate_segments_count: int
    significant_cate_count: int
    # Segment Discovery Phase
    high_responders_count: int
    low_responders_count: int
    responder_spread: float
    # Policy Learning Phase
    policy_recommendations_count: int
    expected_total_lift: float
    actionable_policies: int
    # Outcome Metrics
    total_latency_ms: float
    confidence_score: float
    user_satisfaction: Optional[float]
    def compute_reward(self) -> float: ...
```

---

## Tier 3: Monitoring Layer

### drift_monitor

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/drift_monitor/memory_hooks.py` |
| Memory Types | Working, Episodic, Semantic | Full tri-memory integration for drift pattern detection |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/drift_monitor/dspy_integration.py` |
| DSPy Type | Sender | Generates DriftDetectionTrainingSignal |
| DSPy Signatures | DriftDetectionSignature, HopDecisionSignature, DriftInterpretationSignature | Drift detection |
| **Contract** | ✅ VALIDATED | tier3-contracts.md (lines 375-540) |

**Blocking Items**: None

**Training Signal Schema**:
```python
@dataclass
class DriftDetectionTrainingSignal:
    # Input Context
    signal_id: str
    session_id: str
    query: str
    model_id: str
    features_monitored: int
    time_window: str
    # Detection Configuration
    check_data_drift: bool
    check_model_drift: bool
    check_concept_drift: bool
    psi_threshold: float
    significance_level: float
    # Detection Results
    data_drift_count: int
    model_drift_count: int
    concept_drift_count: int
    overall_drift_score: float
    severity_distribution: Dict[str, int]
    # Alert Generation
    alerts_generated: int
    critical_alerts: int
    warnings: int
    recommended_actions_count: int
    # Outcome Metrics
    total_latency_ms: float
    drift_correctly_identified: Optional[bool]
    user_satisfaction: Optional[float]
    def compute_reward(self) -> float: ...
```

---

### experiment_designer

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/experiment_designer/memory_hooks.py` |
| Memory Types | Working, Episodic | Experiment history tracking and validity threat learning |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/experiment_designer/dspy_integration.py` |
| DSPy Type | Sender | Generates ExperimentDesignTrainingSignal |
| DSPy Signatures | DesignReasoningSignature, InvestigationPlanSignature, ValidityAssessmentSignature | A/B test design |
| **Contract** | ✅ VALIDATED | tier3-contracts.md (lines 541-706) |

**Blocking Items**: None

**Training Signal Schema**:
```python
@dataclass
class ExperimentDesignTrainingSignal:
    # Input Context
    signal_id: str
    session_id: str
    business_question: str
    preregistration_formality: str  # light, medium, heavy
    max_redesign_iterations: int
    # Design Reasoning Phase
    design_type_chosen: str  # RCT, quasi_experiment, etc.
    treatments_count: int
    outcomes_count: int
    randomization_unit: str
    # Power Analysis Phase
    required_sample_size: int
    achieved_power: float
    minimum_detectable_effect: float
    duration_estimate_days: int
    # Validity Audit Phase
    validity_threats_identified: int
    critical_threats: int
    mitigations_proposed: int
    overall_validity_score: float
    redesign_iterations: int
    # Template Generation
    template_generated: bool
    causal_graph_generated: bool
    analysis_code_generated: bool
    # Outcome Metrics
    total_llm_tokens_used: int
    total_latency_ms: float
    experiment_approved: Optional[bool]
    user_satisfaction: Optional[float]
    def compute_reward(self) -> float: ...
```

---

### health_score

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/health_score/memory_hooks.py` |
| Memory Types | Working, Episodic | Health trend tracking |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/health_score/dspy_integration.py` |
| DSPy Type | Recipient | Receives optimized prompts from feedback_learner |
| DSPy Signatures | HealthSummarySignature, HealthRecommendationSignature | System health reporting |
| **Contract** | ✅ VALIDATED | tier3-contracts.md (lines 707-796) |

**Blocking Items**: None

**Recipient Configuration**:
```python
class HealthScoreRecipient:
    dspy_type: Literal["recipient"] = "recipient"
    prompt_refresh_interval_hours: int = 24
    optimizable_signatures: List[str] = [
        "HealthSummarySignature",
        "HealthRecommendationSignature",
    ]
```

---

## Tier 4: ML Prediction Layer

### prediction_synthesizer

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/prediction_synthesizer/memory_hooks.py` |
| Memory Types | Working, Episodic | Prediction history |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/prediction_synthesizer/dspy_integration.py` |
| DSPy Type | Sender | Generates PredictionSynthesisTrainingSignal |
| DSPy Signatures | PredictionSynthesisSignature, PredictionInterpretationSignature, UncertaintyQuantificationSignature | ML ensemble |
| **Contract** | ✅ VALIDATED | tier4-contracts.md (lines 331-477) |

**Blocking Items**: None

**Training Signal Schema**:
```python
@dataclass
class PredictionSynthesisTrainingSignal:
    # Input Context
    signal_id: str
    session_id: str
    query: str
    entity_id: str
    entity_type: str  # hcp, territory, patient
    prediction_target: str
    time_horizon: str
    # Model Orchestration
    models_requested: int
    models_succeeded: int
    models_failed: int
    ensemble_method: str  # average, weighted, stacking, voting
    # Ensemble Results
    point_estimate: float
    prediction_interval_width: float
    ensemble_confidence: float
    model_agreement: float
    # Context Enrichment
    similar_cases_found: int
    feature_importance_calculated: bool
    historical_accuracy: float
    trend_direction: str
    # Outcome Metrics
    total_latency_ms: float
    prediction_accuracy: Optional[float]
    user_satisfaction: Optional[float]
    def compute_reward(self) -> float: ...
```

---

### resource_optimizer

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/resource_optimizer/memory_hooks.py` |
| Memory Types | Working, Procedural | Optimization procedure learning |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/resource_optimizer/dspy_integration.py` |
| DSPy Type | Recipient | Receives optimized prompts from feedback_learner |
| DSPy Signatures | OptimizationSummarySignature, AllocationRecommendationSignature, ScenarioNarrativeSignature | Resource allocation |
| **Contract** | ✅ VALIDATED | tier4-contracts.md (lines 478-601) |

**Blocking Items**: None

**Recipient Configuration**:
```python
class ResourceOptimizerRecipient:
    dspy_type: Literal["recipient"] = "recipient"
    prompt_refresh_interval_hours: int = 24
    optimizable_signatures: List[str] = [
        "OptimizationSummarySignature",
        "AllocationRecommendationSignature",
        "ScenarioNarrativeSignature",
    ]
```

---

## Tier 5: Self-Improvement Layer

### explainer

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/explainer/memory_hooks.py` |
| Memory Types | Working, Semantic | Explanation pattern learning |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/explainer/dspy_integration.py` |
| DSPy Type | Recipient | Receives optimized prompts from feedback_learner |
| DSPy Signatures | ExplanationSynthesisSignature, InsightExtractionSignature, NarrativeStructureSignature, QueryRewriteForExplanationSignature | Natural language explanations |
| **Contract** | ✅ VALIDATED | tier5-contracts.md (lines 511-643) |

**Blocking Items**: None

**Recipient Configuration**:
```python
class ExplainerRecipient:
    dspy_type: Literal["recipient"] = "recipient"
    prompt_refresh_interval_hours: int = 12
    optimizable_signatures: List[str] = [
        "ExplanationSynthesisSignature",
        "InsightExtractionSignature",
        "NarrativeStructureSignature",
        "QueryRewriteForExplanationSignature",
    ]
```

---

### feedback_learner

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ✅ COMPLIANT | `src/agents/feedback_learner/memory_hooks.py` |
| Memory Types | Working, Episodic, Procedural | Full learning memory stack |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/feedback_learner/dspy_integration.py` |
| DSPy Type | Hybrid | Central optimization hub - ingests signals AND distributes optimized prompts |
| DSPy Signatures | PatternDetectionSignature, RecommendationGenerationSignature, KnowledgeUpdateSignature, LearningSummarySignature, MemoryWorthinessSignature, ProcedureLearningSignature | Self-improvement |
| **Contract** | ✅ VALIDATED | tier5-contracts.md (lines 644-917) |

**Blocking Items**: None

**Hybrid Coordinator Configuration**:
```python
class FeedbackLearnerHybridCoordinator:
    dspy_type: Literal["hybrid"] = "hybrid"
    signal_batch_size: int = 100
    signal_retention_hours: int = 168  # 1 week
    min_signals_for_optimization: int = 100
    optimization_interval_hours: int = 24
    min_improvement_threshold: float = 0.05
    recipient_agents: List[str] = ["health_score", "resource_optimizer", "explainer"]
```

---

## Tier 0: ML Foundation Layer (Out of Scope)

Tier 0 agents operate within the ML pipeline and do not require 4-Memory Architecture or DSPy Integration contracts.

| Agent | Purpose | Status |
|-------|---------|--------|
| scope_definer | Define ML problem scope | N/A - Pipeline internal |
| data_preparer | Data preparation & validation | N/A - Pipeline internal |
| feature_analyzer | Feature engineering & selection | N/A - Pipeline internal |
| model_selector | Model selection & benchmarking | N/A - Pipeline internal |
| model_trainer | Model training & tuning | N/A - Pipeline internal |
| model_deployer | Model deployment & versioning | N/A - Pipeline internal |
| observability_connector | MLflow, Opik, monitoring | N/A - Pipeline internal |

---

## RAG System DSPy Integration (✅ COMPLETE)

| Aspect | Status | Details |
|--------|--------|---------|
| **Location** | `src/rag/cognitive_rag_dspy.py` | CausalRAG DSPy implementation |
| **DSPy Type** | Core | Central to 11 Cognitive RAG signatures (50+ total across system) |
| **Contract** | ✅ COMPLETE | Full DSPy integration with 4-phase cognitive workflow |
| **Tests** | ✅ COMPLETE | `tests/rag/test_cognitive_rag_dspy.py` (950 lines) |

**Implemented DSPy Signatures (11 in Cognitive RAG, 50+ total system-wide)**:

| Phase | Signature | Purpose | Status |
|-------|-----------|---------|--------|
| 1 (Summarizer) | QueryRewriteSignature | Rewrite queries for retrieval | ✅ line 97 |
| 1 (Summarizer) | EntityExtractionSignature | Extract business entities | ✅ line 116 |
| 1 (Summarizer) | IntentClassificationSignature | Classify query intent | ✅ line 134 |
| 2 (Investigator) | InvestigationPlanSignature | Plan multi-hop investigation | ✅ line 208 |
| 2 (Investigator) | HopDecisionSignature | Decide next hop in investigation | ✅ line 230 |
| 2 (Investigator) | EvidenceRelevanceSignature | Score evidence relevance | ✅ line 249 |
| 3 (Agent) | EvidenceSynthesisSignature | Synthesize evidence for answer | ✅ line 368 |
| 3 (Agent) | AgentRoutingSignature | Route to appropriate agent | ✅ line 388 |
| 3 (Agent) | VisualizationConfigSignature | Configure visualization output | ✅ line 407 |
| 4 (Reflector) | MemoryWorthinessSignature | Decide what to store in memory | ✅ line 486 |
| 4 (Reflector) | ProcedureLearningSignature | Learn optimized procedures | ✅ line 505 |

**Implemented DSPy Modules**:

| Module | Purpose | Location |
|--------|---------|----------|
| SummarizerModule | Phase 1: Query understanding | cognitive_rag_dspy.py:152-205 |
| InvestigatorModule | Phase 2: Multi-hop evidence gathering | cognitive_rag_dspy.py:269-366 |
| AgentModule | Phase 3: Response synthesis & routing | cognitive_rag_dspy.py:427-483 |
| ReflectorModule | Phase 4: Memory management & learning | cognitive_rag_dspy.py:527-590 |

**Additional Components**:

| Component | Purpose | Location |
|-----------|---------|----------|
| CognitiveRAGOptimizer | MIPROv2 optimization with phase-specific metrics | cognitive_rag_dspy.py:600-750 |
| SignalCollectorAdapter | Training signal collection for feedback_learner | memory_adapters.py:700-855 |
| LangGraph Workflow | 4-phase cognitive workflow via `create_dspy_cognitive_workflow()` | cognitive_rag_dspy.py:750-850 |

**Blocking Items**: None - All items complete

---

## Signal Flow Validation

### Signal Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SIGNAL FLOW                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tier 2 Senders                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐   │
│  │causal_impact │ │ gap_analyzer │ │heterogeneous_optimizer │   │
│  └──────┬───────┘ └──────┬───────┘ └───────────┬────────────┘   │
│         │                │                      │                │
│         └────────────────┼──────────────────────┘                │
│                          ▼                                       │
│  Tier 3 Senders                                                  │
│  ┌──────────────┐ ┌────────────────────┐                        │
│  │drift_monitor │ │experiment_designer │                        │
│  └──────┬───────┘ └─────────┬──────────┘                        │
│         │                   │                                    │
│         └───────────────────┘                                    │
│                    │                                             │
│                    ▼                                             │
│  Tier 4 Sender                                                   │
│  ┌──────────────────────┐                                       │
│  │prediction_synthesizer│                                       │
│  └───────────┬──────────┘                                       │
│              │                                                   │
│              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            feedback_learner (Tier 5 Hybrid)               │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Ingest signals from all Sender agents            │ │   │
│  │  │ 2. Run MIPROv2 optimization when threshold met      │ │   │
│  │  │ 3. Distribute optimized prompts to Recipient agents │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│  Tier 3-5 Recipients                                            │
│  ┌────────────┐ ┌──────────────────┐ ┌──────────┐              │
│  │health_score│ │resource_optimizer│ │ explainer│              │
│  └────────────┘ └──────────────────┘ └──────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Signal Flow Contract Validation

| Aspect | Status | Location |
|--------|--------|----------|
| Hub Agent | ✅ orchestrator | tier1-contracts.md |
| Sender Agents (6) | ✅ All defined | tier2-4-contracts.md |
| Recipient Agents (3) | ✅ All defined | tier3-5-contracts.md |
| Hybrid Agents (2) | ✅ All defined | tier1,5-contracts.md |
| SignalFlowContract | ✅ Exists | integration-contracts.md:1454-1543 |

---

## Validation Checklist

### Memory Architecture Validation

- [x] Working memory backend (`src/memory/working_memory.py`)
- [x] Episodic memory backend (`src/memory/episodic_memory.py`)
- [x] Semantic memory backend (`src/memory/semantic_memory.py`)
- [x] Procedural memory backend (`src/memory/procedural_memory.py`)
- [x] MemoryType enum in base-contract.md
- [x] MemoryHooksInterface ABC in base-contract.md
- [x] drift_monitor memory_hooks.py implementation
- [x] experiment_designer memory_hooks.py implementation

### DSPy Integration Validation

- [x] Orchestrator Hub role (tier1-contracts.md)
- [x] Tool Composer Hybrid role (tier1-contracts.md extended)
- [x] Tier 2 Sender roles (tier2-contracts.md)
- [x] Tier 3 Sender/Recipient roles (tier3-contracts.md)
- [x] Tier 4 Sender/Recipient roles (tier4-contracts.md)
- [x] Tier 5 Recipient/Hybrid roles (tier5-contracts.md)
- [x] SignalFlowContract (integration-contracts.md)
- [x] RAG DSPy signatures implementation (cognitive_rag_dspy.py)

### Test Coverage Validation

- [x] 211 DSPy tests across 7 agents created
- [x] Memory hooks unit tests per agent
- [x] Signal collector tests per Sender agent
- [x] Recipient configuration tests per Recipient agent
- [x] Integration tests for signal flow end-to-end (**131 tests**)

#### Signal Flow Integration Tests

Location: `tests/integration/test_signal_flow/`

| Batch | File | Tests | Status |
|-------|------|-------|--------|
| 1 | test_sender_signals.py | 34 | ✅ PASSED |
| 2 | test_signal_collection.py | 15 | ✅ PASSED |
| 3 | test_hub_coordination.py | 25 | ✅ PASSED |
| 4 | test_recipient_prompts.py | 37 | ✅ PASSED |
| 5 | test_e2e_signal_flow.py | 20 | ✅ PASSED |
| **Total** | | **131** | ✅ ALL PASSED |

**Test Coverage by Role**:
- Sender agents (6): causal_impact, gap_analyzer, heterogeneous_optimizer, drift_monitor, experiment_designer, prediction_synthesizer
- Hub agents (2): orchestrator, feedback_learner
- Recipient agents (3): health_score, resource_optimizer, explainer
- Hybrid agents (2): tool_composer, feedback_learner

**Signal Flow Contract Compliance Tests**:
- Minimum signals threshold (100) validation
- Minimum quality threshold (0.6) validation
- Optimization interval (24 hours) validation
- Signal JSON roundtrip serialization
- Multi-agent coordination
- Memory contribution integration

---

## Causal Discovery Validation Rules (V4.4+)

The causal discovery module (`src/causal_engine/discovery/`) adds automatic DAG structure learning. These validation rules ensure discovery is used correctly.

### Data Requirements

| Rule | Validation | Error |
|------|------------|-------|
| **D1** | If `auto_discover=True`, `data_cache` must contain DataFrame | `DiscoveryRequiresDataError` |
| **D2** | DataFrame must have >= 100 rows for valid discovery | `InsufficientDataError` |
| **D3** | DataFrame must have >= 3 columns (treatment + outcome + confounder) | `InsufficientFeaturesError` |
| **D4** | No column may have >50% missing values | `ExcessiveMissingDataError` |

### Algorithm Configuration

| Rule | Validation | Error |
|------|------------|-------|
| **A1** | `discovery_algorithms` must be subset of `["ges", "pc", "fci", "lingam"]` | `InvalidAlgorithmError` |
| **A2** | `ensemble_threshold` must be in range `[0.0, 1.0]` | `InvalidThresholdError` |
| **A3** | `discovery_alpha` must be in range `(0.0, 1.0)` | `InvalidAlphaError` |
| **A4** | At least one algorithm must be specified | `NoAlgorithmError` |

### Gate Decision Validation

| Rule | Validation | Error |
|------|------------|-------|
| **G1** | `gate_decision` must be one of: `accept`, `augment`, `review`, `reject` | `InvalidGateDecisionError` |
| **G2** | `gate_confidence` must be in range `[0.0, 1.0]` | `InvalidConfidenceError` |
| **G3** | If `gate_decision="accept"`, `gate_confidence >= 0.8` | `AcceptThresholdViolation` |
| **G4** | If `gate_decision="reject"`, `gate_confidence < 0.3` | `RejectThresholdViolation` |

### DAG Structure Validation

| Rule | Validation | Error |
|------|------------|-------|
| **S1** | Discovered DAG must be acyclic | `CyclicGraphError` |
| **S2** | Discovered DAG must be connected (weak connectivity) | `DisconnectedGraphWarning` |
| **S3** | Edge confidence scores must be in `[0.0, 1.0]` | `InvalidEdgeConfidenceError` |
| **S4** | Edge votes must be <= number of algorithms used | `InvalidVoteCountError` |

### Integration Validation

| Rule | Validation | Error |
|------|------------|-------|
| **I1** | If downstream agents use `discovered_confounders`, `gate_decision` must be `accept` or `augment` | `UseOfRejectedDiscoveryWarning` |
| **I2** | If `effect_modifiers_validated=True`, discovery must have occurred | `ValidationWithoutDiscoveryError` |
| **I3** | Discovery metadata must include `algorithms`, `threshold`, `confidence` | `IncompletMetadataError` |

### Validation Code

```python
from typing import Dict, Any, List, Optional
import pandas as pd


class DiscoveryValidationError(Exception):
    """Base exception for discovery validation errors."""
    pass


class DiscoveryRequiresDataError(DiscoveryValidationError):
    """Raised when auto_discover=True but no data provided."""
    pass


class InvalidGateDecisionError(DiscoveryValidationError):
    """Raised when gate decision is not in allowed set."""
    pass


VALID_ALGORITHMS = {"ges", "pc", "fci", "lingam"}
VALID_GATE_DECISIONS = {"accept", "augment", "review", "reject"}


def validate_discovery_input(state: Dict[str, Any]) -> List[str]:
    """
    Validate discovery input configuration.

    Args:
        state: Agent state with discovery configuration

    Returns:
        List of validation warnings (empty if all valid)

    Raises:
        DiscoveryValidationError: If critical validation fails
    """
    warnings = []

    # D1: Data requirement
    if state.get("auto_discover"):
        data_cache = state.get("data_cache", {})
        data = data_cache.get("data")

        if data is None:
            raise DiscoveryRequiresDataError(
                "auto_discover=True requires data_cache with DataFrame"
            )

        if not isinstance(data, pd.DataFrame):
            raise DiscoveryRequiresDataError(
                f"data_cache['data'] must be DataFrame, got {type(data)}"
            )

        # D2: Minimum rows
        if len(data) < 100:
            warnings.append(f"DataFrame has {len(data)} rows, <100 may give unreliable discovery")

        # D3: Minimum columns
        if len(data.columns) < 3:
            raise DiscoveryValidationError(
                f"DataFrame has {len(data.columns)} columns, need >=3 for discovery"
            )

        # D4: Missing values
        for col in data.columns:
            missing_pct = data[col].isna().mean()
            if missing_pct > 0.5:
                warnings.append(f"Column '{col}' has {missing_pct:.0%} missing values")

    # A1: Valid algorithms
    algorithms = state.get("discovery_algorithms", ["ges", "pc"])
    invalid_algos = set(algorithms) - VALID_ALGORITHMS
    if invalid_algos:
        raise DiscoveryValidationError(
            f"Invalid algorithms: {invalid_algos}. Valid: {VALID_ALGORITHMS}"
        )

    # A2: Threshold range
    threshold = state.get("discovery_ensemble_threshold", 0.5)
    if not 0.0 <= threshold <= 1.0:
        raise DiscoveryValidationError(
            f"ensemble_threshold must be in [0.0, 1.0], got {threshold}"
        )

    # A3: Alpha range
    alpha = state.get("discovery_alpha", 0.05)
    if not 0.0 < alpha < 1.0:
        raise DiscoveryValidationError(
            f"discovery_alpha must be in (0.0, 1.0), got {alpha}"
        )

    return warnings


def validate_discovery_output(state: Dict[str, Any]) -> List[str]:
    """
    Validate discovery output.

    Args:
        state: Agent state with discovery results

    Returns:
        List of validation warnings

    Raises:
        DiscoveryValidationError: If critical validation fails
    """
    warnings = []
    gate_eval = state.get("discovery_gate_evaluation", {})

    # G1: Valid gate decision
    decision = gate_eval.get("decision")
    if decision and decision not in VALID_GATE_DECISIONS:
        raise InvalidGateDecisionError(
            f"Invalid gate decision: {decision}. Valid: {VALID_GATE_DECISIONS}"
        )

    # G2: Confidence range
    confidence = gate_eval.get("confidence", 0.0)
    if not 0.0 <= confidence <= 1.0:
        raise DiscoveryValidationError(
            f"gate confidence must be in [0.0, 1.0], got {confidence}"
        )

    # G3: Accept threshold
    if decision == "accept" and confidence < 0.8:
        warnings.append(
            f"ACCEPT decision with confidence {confidence:.2f} < 0.8 threshold"
        )

    # G4: Reject threshold
    if decision == "reject" and confidence >= 0.3:
        warnings.append(
            f"REJECT decision with confidence {confidence:.2f} >= 0.3 (unusual)"
        )

    return warnings
```

### Test Coverage

| Test Category | Count | Status |
|---------------|-------|--------|
| Discovery base types (base.py) | 18 | ✅ |
| DiscoveryRunner (runner.py) | 25 | ✅ |
| DiscoveryGate (gate.py) | 18 | ✅ |
| DriverRanker (driver_ranker.py) | 16 | ✅ |
| GraphBuilder discovery integration | 21 | ✅ |
| **Total Discovery Tests** | **98** | ✅ |

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-30 | Added Causal Discovery Validation Rules (V4.4) - 14 validation rules |
| 2025-12-23 | Initial creation - Phase 2 of audit complete |
| 2025-12-23 | Added all 18 agents with validation status |
| 2025-12-23 | Added signal flow architecture diagram |
| 2025-12-23 | Identified 2 memory hooks gaps, 1 RAG DSPy gap |
| 2025-12-23 | ✅ RAG DSPy validated as COMPLETE - all 11 signatures exist in cognitive_rag_dspy.py |
| 2025-12-23 | DSPy Integration now at 100% (13/13) including RAG |
| 2025-12-23 | ✅ drift_monitor memory_hooks.py implemented - Memory hooks now at 11/12 (92%) |
| 2025-12-23 | ✅ experiment_designer memory_hooks.py implemented - Memory hooks now at 12/12 (100%) COMPLETE |
| 2025-12-23 | ✅ Added 131 signal flow integration tests (5 batches) |
| 2025-12-23 | ✅ Updated all training signal schemas to match actual implementations |
