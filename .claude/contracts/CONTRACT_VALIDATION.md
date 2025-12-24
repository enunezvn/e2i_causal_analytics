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
| Memory Hooks (Tier 1-5) | 10 | 12 | 83% | GOOD |
| DSPy Integration (Tier 1-5) | 12 | 12 | 100% | COMPLETE |
| RAG DSPy Integration | 0 | 1 | 0% | BLOCKING |
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
    signal_id: str
    session_id: str
    query: str
    treatment_identified: bool
    outcome_identified: bool
    confounders_found: int
    ate_computed: bool
    refutation_passed: bool
    confidence_score: float
    total_latency_ms: float
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
    signal_id: str
    session_id: str
    query: str
    gaps_identified: int
    roi_estimates_computed: int
    actionable_recommendations: int
    evidence_quality: float
    total_latency_ms: float
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
    signal_id: str
    session_id: str
    query: str
    segments_identified: int
    cate_estimates_computed: int
    policy_recommendations: int
    optimization_quality: float
    total_latency_ms: float
    def compute_reward(self) -> float: ...
```

---

## Tier 3: Monitoring Layer

### drift_monitor

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ⚠️ MISSING | Required: Working, Episodic, Semantic |
| Memory Types | N/A | Drift pattern detection needs semantic memory |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/drift_monitor/dspy_integration.py` |
| DSPy Type | Sender | Generates DriftDetectionTrainingSignal |
| DSPy Signatures | DriftDetectionSignature, HopDecisionSignature, DriftInterpretationSignature | Drift detection |
| **Contract** | ✅ VALIDATED | tier3-contracts.md (lines 375-540) |

**Blocking Items**:
- [ ] Implement `memory_hooks.py` with Working, Episodic, Semantic memory integration

**Training Signal Schema**:
```python
@dataclass
class DriftDetectionTrainingSignal:
    signal_id: str
    session_id: str
    query: str
    drift_detected: bool
    drift_type: str  # "data" | "model" | "concept"
    severity: float
    features_affected: int
    alert_generated: bool
    total_latency_ms: float
    def compute_reward(self) -> float: ...
```

---

### experiment_designer

| Aspect | Status | Details |
|--------|--------|---------|
| **Memory Hooks** | ⚠️ MISSING | Required: Working, Episodic |
| Memory Types | N/A | Experiment history tracking |
| **DSPy Integration** | ✅ COMPLIANT | `src/agents/experiment_designer/dspy_integration.py` |
| DSPy Type | Sender | Generates ExperimentDesignTrainingSignal |
| DSPy Signatures | DesignReasoningSignature, InvestigationPlanSignature, ValidityAssessmentSignature | A/B test design |
| **Contract** | ✅ VALIDATED | tier3-contracts.md (lines 541-706) |

**Blocking Items**:
- [ ] Implement `memory_hooks.py` with Working, Episodic memory integration

**Training Signal Schema**:
```python
@dataclass
class ExperimentDesignTrainingSignal:
    signal_id: str
    session_id: str
    query: str
    design_complete: bool
    sample_size_calculated: bool
    power_analysis_done: bool
    validity_threats_addressed: int
    digital_twin_simulated: bool
    total_latency_ms: float
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
    signal_id: str
    session_id: str
    query: str
    model_count: int
    ensemble_method: str
    predictions_generated: int
    model_agreement: float
    prediction_confidence: float
    total_latency_ms: float
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

## RAG System DSPy Integration (BLOCKING)

| Aspect | Status | Details |
|--------|--------|---------|
| **Location** | `src/rag/` | CausalRAG implementation |
| **DSPy Type** | Core | Central to all 11 DSPy signatures |
| **Contract** | ⚠️ BLOCKING | Missing DSPy integration |

**Required DSPy Signatures (11 total)**:

| Phase | Signature | Purpose | Status |
|-------|-----------|---------|--------|
| 1 (Summarizer) | QueryRewriteSignature | Rewrite queries for retrieval | ⚠️ MISSING |
| 1 (Summarizer) | EntityExtractionSignature | Extract business entities | ⚠️ MISSING |
| 1 (Summarizer) | IntentClassificationSignature | Classify query intent | ⚠️ MISSING |
| 2 (Investigator) | InvestigationPlanSignature | Plan multi-hop investigation | ⚠️ MISSING |
| 2 (Investigator) | HopDecisionSignature | Decide next hop in investigation | ⚠️ MISSING |
| 2 (Investigator) | EvidenceRelevanceSignature | Score evidence relevance | ⚠️ MISSING |
| 3 (Agent) | EvidenceSynthesisSignature | Synthesize evidence for answer | ⚠️ MISSING |
| 3 (Agent) | AgentRoutingSignature | Route to appropriate agent | ⚠️ MISSING |
| 3 (Agent) | VisualizationConfigSignature | Configure visualization output | ⚠️ MISSING |
| 4 (Reflector) | MemoryWorthinessSignature | Decide what to store in memory | ⚠️ MISSING |
| 4 (Reflector) | ProcedureLearningSignature | Learn optimized procedures | ⚠️ MISSING |

**Blocking Items**:
- [ ] Implement `src/rag/dspy_signatures.py` with all 11 DSPy signatures
- [ ] Integrate DSPy signatures into RAG pipeline phases
- [ ] Add signal collection for RAG operations
- [ ] Connect RAG to feedback_learner signal flow

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
- [ ] drift_monitor memory_hooks.py implementation
- [ ] experiment_designer memory_hooks.py implementation

### DSPy Integration Validation

- [x] Orchestrator Hub role (tier1-contracts.md)
- [x] Tool Composer Hybrid role (tier1-contracts.md extended)
- [x] Tier 2 Sender roles (tier2-contracts.md)
- [x] Tier 3 Sender/Recipient roles (tier3-contracts.md)
- [x] Tier 4 Sender/Recipient roles (tier4-contracts.md)
- [x] Tier 5 Recipient/Hybrid roles (tier5-contracts.md)
- [x] SignalFlowContract (integration-contracts.md)
- [ ] RAG DSPy signatures implementation

### Test Coverage Validation

- [x] 211 DSPy tests across 7 agents created
- [x] Memory hooks unit tests per agent
- [x] Signal collector tests per Sender agent
- [x] Recipient configuration tests per Recipient agent
- [ ] Integration tests for signal flow end-to-end

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-23 | Initial creation - Phase 2 of audit complete |
| 2025-12-23 | Added all 18 agents with validation status |
| 2025-12-23 | Added signal flow architecture diagram |
| 2025-12-23 | Identified 2 memory hooks gaps, 1 RAG DSPy gap |
