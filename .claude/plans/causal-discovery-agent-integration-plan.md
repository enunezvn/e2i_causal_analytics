# Causal Discovery Agent Integration Plan

**Version**: 1.0.0
**Created**: 2025-12-31
**Status**: Ready for Implementation
**Feature**: Causal-vs-Predictive Ranking with Structure Learning (GES/PC/FCI/LiNGAM)

---

## Executive Summary

This plan integrates the new causal discovery module (`src/causal_engine/discovery/`) with the E2I agent architecture. The module provides:
- **Multi-algorithm ensemble**: GES, PC, FCI (with latent confounder detection), LiNGAM
- **DriverRanker**: Compares causal importance (DAG structure) vs predictive importance (SHAP)
- **DiscoveryGate**: Quality gating with ACCEPT/REVIEW/REJECT/AUGMENT decisions
- **Observability**: Opik tracing for all discovery operations

---

## Current State Analysis

### Already Updated (V4.4)
| Component | File | Status |
|-----------|------|--------|
| Tier 2 Contracts | `.claude/contracts/tier2-contracts.md` | ✅ CausalImpactInput/Output with discovery fields |
| Orchestrator Contracts | `.claude/contracts/orchestrator-contracts.md` | ✅ AgentSelectionCriteria with enable_discovery |

### Needs Updates
| Component | File | Priority |
|-----------|------|----------|
| Feature Analyzer State | `src/agents/ml_foundation/feature_analyzer/state.py` | CRITICAL |
| Feature Analyzer Nodes | `src/agents/ml_foundation/feature_analyzer/nodes/` | CRITICAL |
| Causal Impact Graph Builder | `src/agents/causal_impact/nodes/graph_builder.py` | CRITICAL |
| Data Contracts | `.claude/contracts/data-contracts.md` | HIGH |
| Tier 0 Contracts | `.claude/contracts/tier0-contracts.md` | HIGH |
| Gap Analyzer Prioritizer | `src/agents/gap_analyzer/nodes/prioritizer.py` | HIGH |
| Het. Optimizer Nodes | `src/agents/heterogeneous_optimizer/nodes/` | HIGH |
| Experiment Designer | `src/agents/experiment_designer/nodes/` | HIGH |
| Integration Contracts | `.claude/contracts/integration-contracts.md` | MEDIUM |
| Tier 3 Contracts | `.claude/contracts/tier3-contracts.md` | MEDIUM |
| Orchestrator Routing | `src/agents/orchestrator/graph.py` | MEDIUM |
| Explainer Nodes | `src/agents/explainer/nodes/` | MEDIUM |
| Drift Monitor | `src/agents/drift_monitor/nodes/` | MEDIUM |

---

## Implementation Waves

### Wave 1: CRITICAL - Core Integration (Feature Analyzer + Causal Impact)
**Estimated Tests**: ~35 tests in 4 batches

### Wave 2: HIGH - Downstream Agents (Gap Analyzer + Het. Optimizer + Exp. Designer)
**Estimated Tests**: ~30 tests in 3 batches

### Wave 3: MEDIUM - Supporting Integration (Orchestrator + Explainer + Drift Monitor)
**Estimated Tests**: ~26 tests in 3 batches

---

## Wave 1: CRITICAL - Core Integration

### Phase 1.1: Feature Analyzer State Extension
**Priority**: CRITICAL
**Files to Modify**:
- `src/agents/ml_foundation/feature_analyzer/state.py`

**Changes Required**:
```python
# Add to FeatureAnalyzerState TypedDict

# Discovery Integration Section
discovery_enabled: NotRequired[bool]
discovery_config: NotRequired[Dict[str, Any]]
discovery_result: NotRequired[Dict[str, Any]]
discovery_gate_decision: NotRequired[str]  # ACCEPT|REVIEW|REJECT|AUGMENT

# Causal Ranking Section
causal_rankings: NotRequired[List[Dict[str, Any]]]  # FeatureRanking dicts
predictive_rankings: NotRequired[List[Dict[str, Any]]]
rank_correlation: NotRequired[float]  # Spearman correlation
divergent_features: NotRequired[List[str]]  # High rank difference
causal_only_features: NotRequired[List[str]]
predictive_only_features: NotRequired[List[str]]
concordant_features: NotRequired[List[str]]
```

**Validation**:
- [ ] State TypedDict compiles without errors
- [ ] Backward compatible (all new fields NotRequired)

---

### Phase 1.2: Feature Analyzer Causal Ranker Node
**Priority**: CRITICAL
**Files to Create**:
- `src/agents/ml_foundation/feature_analyzer/nodes/causal_ranker.py`

**Implementation**:
```python
"""Causal Ranker Node for Feature Analyzer.

Integrates DriverRanker to compare causal vs predictive feature importance.
"""

from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from src.causal_engine.discovery.runner import DiscoveryRunner
from src.causal_engine.discovery.driver_ranker import DriverRanker
from src.causal_engine.discovery.base import DiscoveryConfig
from ..state import FeatureAnalyzerState

async def causal_ranker_node(
    state: FeatureAnalyzerState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Compare causal vs predictive feature importance.

    Uses DiscoveryRunner for DAG learning and DriverRanker for comparison.
    """
    # Skip if discovery not enabled
    if not state.get("discovery_enabled", False):
        return {}

    # Get required inputs
    data = state.get("prepared_data")
    target = state.get("target_variable")
    shap_values = state.get("shap_values")
    feature_names = state.get("feature_names")

    if data is None or target is None or shap_values is None:
        return {"discovery_gate_decision": "REJECT"}

    # Run discovery
    discovery_config = DiscoveryConfig(
        **state.get("discovery_config", {})
    )
    runner = DiscoveryRunner()
    result = await runner.discover_dag(data, discovery_config)

    # Rank drivers
    ranker = DriverRanker()
    ranking_result = ranker.rank_from_discovery_result(
        result=result,
        target=target,
        shap_values=shap_values,
        feature_names=feature_names,
    )

    return {
        "discovery_result": result.to_dict(),
        "discovery_gate_decision": result.gate_decision.value if result.gate_decision else None,
        "causal_rankings": [r.__dict__ for r in ranking_result.rankings],
        "rank_correlation": ranking_result.rank_correlation,
        "divergent_features": [
            r.feature_name for r in ranking_result.rankings
            if abs(r.rank_difference) > 3
        ],
        "causal_only_features": ranking_result.causal_only_features,
        "predictive_only_features": ranking_result.predictive_only_features,
        "concordant_features": ranking_result.concordant_features,
    }
```

**Files to Modify**:
- `src/agents/ml_foundation/feature_analyzer/nodes/__init__.py` (export)
- `src/agents/ml_foundation/feature_analyzer/graph.py` (add node to graph)

**Tests to Create** (Batch 1.2, ~8 tests):
- `tests/unit/test_agents/test_ml_foundation/test_feature_analyzer/test_causal_ranker.py`

---

### Phase 1.3: Feature Analyzer Importance Narrator Update
**Priority**: CRITICAL
**Files to Modify**:
- `src/agents/ml_foundation/feature_analyzer/nodes/importance_narrator.py`

**Changes Required**:
- Add causal vs predictive comparison to narrative
- Highlight divergent features (high rank difference)
- Include gate decision context

**Key Narrative Elements**:
```python
# Add to narrative generation
if state.get("causal_rankings"):
    narrative += f"\n\n## Causal vs Predictive Analysis\n"
    narrative += f"Rank correlation: {state.get('rank_correlation', 0):.2%}\n"

    divergent = state.get("divergent_features", [])
    if divergent:
        narrative += f"\n**Divergent Features** (causal ≠ predictive):\n"
        for feat in divergent[:5]:
            narrative += f"- {feat}\n"

    causal_only = state.get("causal_only_features", [])
    if causal_only:
        narrative += f"\n**Causal-only Features** (no predictive signal):\n"
        for feat in causal_only[:3]:
            narrative += f"- {feat}\n"
```

**Tests** (Batch 1.3, ~6 tests):
- Add to existing `test_importance_narrator.py`

---

### Phase 1.4: Causal Impact Agent Integration
**Priority**: CRITICAL
**Files to Modify**:
- `src/agents/causal_impact/nodes/graph_builder.py`
- `src/agents/causal_impact/state.py`

**Changes Required**:
```python
# In graph_builder.py - use DiscoveryRunner instead of manual DAG

from src.causal_engine.discovery.runner import DiscoveryRunner
from src.causal_engine.discovery.gate import DiscoveryGate

async def build_causal_graph(state: CausalImpactState, config: RunnableConfig):
    """Build causal graph using discovery module."""

    if state.get("auto_discover", True):
        # Use discovery module
        runner = DiscoveryRunner()
        gate = DiscoveryGate()

        result = await runner.discover_dag(
            data=state["data"],
            config=state.get("discovery_config", {}),
        )

        evaluation = gate.evaluate(result)

        return {
            "discovered_dag": result.ensemble_dag,
            "discovery_result": result.to_dict(),
            "gate_decision": evaluation.decision.value,
            "gate_confidence": evaluation.confidence,
            "high_confidence_edges": [
                (e.source, e.target) for e in evaluation.high_confidence_edges
            ],
        }
    else:
        # Use provided manual DAG
        return {"discovered_dag": state.get("manual_dag")}
```

**Tests** (Batch 1.4, ~10 tests):
- `tests/unit/test_agents/test_causal_impact/test_graph_builder_discovery.py`

---

### Phase 1.5: Tier 0 Contract Updates
**Priority**: HIGH
**Files to Modify**:
- `.claude/contracts/tier0-contracts.md`

**Additions**:
```markdown
## Feature Analyzer Discovery Integration (V4.4)

### FeatureAnalyzerDiscoveryInput
- `discovery_enabled: bool` - Enable causal discovery
- `discovery_config: DiscoveryConfig` - Algorithm selection, thresholds
- `target_variable: str` - Target for causal path analysis

### FeatureAnalyzerDiscoveryOutput
- `causal_rankings: List[FeatureRanking]` - Causal importance rankings
- `predictive_rankings: List[FeatureRanking]` - SHAP-based rankings
- `rank_correlation: float` - Spearman correlation between rankings
- `divergent_features: List[str]` - Features with high rank difference
- `discovery_gate_decision: GateDecision` - ACCEPT/REVIEW/REJECT/AUGMENT
```

---

### Wave 1 Testing Batches

| Batch | Tests | Files | Command |
|-------|-------|-------|---------|
| 1.1 | ~6 | test_state.py | `pytest tests/unit/test_agents/test_ml_foundation/test_feature_analyzer/test_state.py -n 2` |
| 1.2 | ~8 | test_causal_ranker.py | `pytest tests/unit/test_agents/test_ml_foundation/test_feature_analyzer/test_causal_ranker.py -n 2` |
| 1.3 | ~6 | test_importance_narrator.py | `pytest tests/unit/test_agents/test_ml_foundation/test_feature_analyzer/test_importance_narrator.py -n 2` |
| 1.4 | ~10 | test_graph_builder_discovery.py | `pytest tests/unit/test_agents/test_causal_impact/test_graph_builder_discovery.py -n 2` |
| 1.5 | ~5 | Contract validation | Manual review |

---

## Wave 2: HIGH - Downstream Agents

### Phase 2.1: Gap Analyzer Causal Filtering
**Priority**: HIGH
**Files to Modify**:
- `src/agents/gap_analyzer/nodes/prioritizer.py`
- `src/agents/gap_analyzer/state.py`

**Changes Required**:
- Filter opportunities by causal evidence
- Prioritize gaps with direct causal paths to outcome
- Use gate decision to weight confidence

```python
# In prioritizer.py
def prioritize_with_causal_evidence(
    opportunities: List[Opportunity],
    causal_rankings: List[FeatureRanking],
    gate_decision: str,
) -> List[Opportunity]:
    """Prioritize opportunities with causal evidence."""

    # Build causal feature lookup
    causal_features = {r.feature_name: r for r in causal_rankings}

    for opp in opportunities:
        if opp.feature in causal_features:
            ranking = causal_features[opp.feature]
            # Boost score for direct causes
            if ranking.is_direct_cause:
                opp.confidence *= 1.2
            # Penalize if only predictive (no causal evidence)
            elif ranking.causal_score == 0:
                opp.confidence *= 0.7
                opp.warnings.append("No causal evidence - correlation only")

    # Apply gate decision weighting
    if gate_decision == "REJECT":
        for opp in opportunities:
            opp.warnings.append("Causal discovery rejected - use with caution")

    return sorted(opportunities, key=lambda o: o.confidence, reverse=True)
```

**Tests** (Batch 2.1, ~10 tests):
- `tests/unit/test_agents/test_gap_analyzer/test_causal_prioritization.py`

---

### Phase 2.2: Heterogeneous Optimizer DAG Validation
**Priority**: HIGH
**Files to Modify**:
- `src/agents/heterogeneous_optimizer/nodes/segment_analyzer.py`
- `src/agents/heterogeneous_optimizer/state.py`

**Changes Required**:
- Validate segment-specific effects against discovered DAG
- Detect when segment effects contradict causal structure
- Use FCI bidirected edges to identify latent confounders

```python
# In segment_analyzer.py
def validate_segment_effects(
    segment_effects: Dict[str, float],
    discovered_dag: nx.DiGraph,
    edge_types: Dict[Tuple[str, str], EdgeType],
) -> Dict[str, Any]:
    """Validate segment effects against causal structure."""

    warnings = []
    validated_effects = {}

    for segment, effect in segment_effects.items():
        # Check if treatment -> segment path exists
        if not nx.has_path(discovered_dag, treatment, segment):
            warnings.append(f"No causal path to {segment}")
            continue

        # Check for bidirected edges (latent confounders)
        for edge, etype in edge_types.items():
            if etype == EdgeType.BIDIRECTED and segment in edge:
                warnings.append(f"Latent confounder detected for {segment}")

        validated_effects[segment] = effect

    return {
        "validated_effects": validated_effects,
        "warnings": warnings,
    }
```

**Tests** (Batch 2.2, ~10 tests):
- `tests/unit/test_agents/test_heterogeneous_optimizer/test_dag_validation.py`

---

### Phase 2.3: Experiment Designer DAG-Aware Validation
**Priority**: HIGH
**Files to Modify**:
- `src/agents/experiment_designer/nodes/validator.py`
- `src/agents/experiment_designer/state.py`

**Changes Required**:
- Use discovered DAG to identify required controls
- Validate experiment design against causal structure
- Detect potential confounders from FCI bidirected edges

```python
# In validator.py
def validate_experiment_design(
    design: ExperimentDesign,
    discovered_dag: nx.DiGraph,
    edge_types: Dict[Tuple[str, str], EdgeType],
) -> ValidationResult:
    """Validate experiment design against discovered causal structure."""

    issues = []
    recommendations = []

    # Find all paths from treatment to outcome
    treatment = design.treatment_variable
    outcome = design.outcome_variable

    if not nx.has_path(discovered_dag, treatment, outcome):
        issues.append("No causal path from treatment to outcome in discovered DAG")

    # Identify confounders (common causes)
    confounders = find_confounders(discovered_dag, treatment, outcome)
    missing_controls = [c for c in confounders if c not in design.control_variables]

    if missing_controls:
        recommendations.append(f"Add controls for confounders: {missing_controls}")

    # Check for latent confounders (FCI bidirected edges)
    for (src, tgt), etype in edge_types.items():
        if etype == EdgeType.BIDIRECTED:
            if src in [treatment, outcome] or tgt in [treatment, outcome]:
                issues.append(f"Latent confounder between {src} and {tgt} - IV may be needed")

    return ValidationResult(
        valid=len(issues) == 0,
        issues=issues,
        recommendations=recommendations,
    )
```

**Tests** (Batch 2.3, ~10 tests):
- `tests/unit/test_agents/test_experiment_designer/test_dag_validation.py`

---

### Phase 2.4: Contract Updates
**Priority**: HIGH
**Files to Modify**:
- `.claude/contracts/data-contracts.md`
- `.claude/contracts/tier3-contracts.md`

**Additions to data-contracts.md**:
```markdown
## DiscoveredDAGSchema (V4.4)

### Core Fields
- `adjacency_matrix: NDArray[int]` - Binary adjacency matrix
- `edge_list: List[Tuple[str, str]]` - (source, target) pairs
- `node_names: List[str]` - Variable names
- `edge_types: Dict[str, EdgeType]` - DIRECTED|BIDIRECTED|UNDIRECTED

### Metadata
- `algorithm_results: List[AlgorithmResult]` - Per-algorithm results
- `ensemble_method: str` - Voting method used
- `gate_decision: GateDecision` - ACCEPT|REVIEW|REJECT|AUGMENT
- `gate_confidence: float` - Overall confidence score
```

---

### Wave 2 Testing Batches

| Batch | Tests | Files | Command |
|-------|-------|-------|---------|
| 2.1 | ~10 | test_causal_prioritization.py | `pytest tests/unit/test_agents/test_gap_analyzer/test_causal_prioritization.py -n 2` |
| 2.2 | ~10 | test_dag_validation.py | `pytest tests/unit/test_agents/test_heterogeneous_optimizer/test_dag_validation.py -n 2` |
| 2.3 | ~10 | test_dag_validation.py | `pytest tests/unit/test_agents/test_experiment_designer/test_dag_validation.py -n 2` |

---

## Wave 3: MEDIUM - Supporting Integration

### Phase 3.1: Orchestrator Discovery Routing
**Priority**: MEDIUM
**Files to Modify**:
- `src/agents/orchestrator/graph.py`
- `src/agents/orchestrator/nodes/router.py`

**Changes Required**:
- Route discovery-enabled queries to appropriate agents
- Pass discovery_config through agent chain
- Handle gate decisions in routing logic

---

### Phase 3.2: Explainer Causal Narrative
**Priority**: MEDIUM
**Files to Modify**:
- `src/agents/explainer/nodes/narrator.py`

**Changes Required**:
- Generate explanations that distinguish causal vs predictive
- Explain divergent features
- Translate gate decisions to user-friendly language

---

### Phase 3.3: Drift Monitor Structural Drift
**Priority**: MEDIUM
**Files to Create**:
- `src/agents/drift_monitor/nodes/structural_drift_detector.py`

**Implementation**:
```python
"""Structural Drift Detector for DAG changes."""

from src.causal_engine.discovery.runner import DiscoveryRunner

async def detect_structural_drift(
    current_dag: nx.DiGraph,
    new_data: pd.DataFrame,
    config: DiscoveryConfig,
) -> StructuralDriftResult:
    """Detect changes in causal structure over time."""

    runner = DiscoveryRunner()
    new_result = await runner.discover_dag(new_data, config)

    if not new_result.success:
        return StructuralDriftResult(detected=False, reason="Discovery failed")

    new_dag = new_result.ensemble_dag

    # Compare edges
    old_edges = set(current_dag.edges())
    new_edges = set(new_dag.edges())

    added_edges = new_edges - old_edges
    removed_edges = old_edges - new_edges

    # Calculate drift score
    total_edges = len(old_edges | new_edges)
    changed_edges = len(added_edges) + len(removed_edges)
    drift_score = changed_edges / total_edges if total_edges > 0 else 0

    return StructuralDriftResult(
        detected=drift_score > 0.1,
        drift_score=drift_score,
        added_edges=list(added_edges),
        removed_edges=list(removed_edges),
        recommendation="Re-train models" if drift_score > 0.2 else None,
    )
```

---

### Phase 3.4: Integration Contract Updates
**Priority**: MEDIUM
**Files to Modify**:
- `.claude/contracts/integration-contracts.md`

**Additions**:
```markdown
## Causal Engine Integration Contract (V4.4)

### Discovery Runner Interface
- Input: `pd.DataFrame`, `DiscoveryConfig`
- Output: `DiscoveryResult` with ensemble DAG, algorithm results, gate decision

### DriverRanker Interface
- Input: `nx.DiGraph`, target, SHAP values, feature names
- Output: `DriverRankingResult` with rankings, correlations, divergent features

### DiscoveryGate Interface
- Input: `DiscoveryResult`, optional expected_edges
- Output: `GateEvaluation` with decision, confidence, high-confidence edges
```

---

### Wave 3 Testing Batches

| Batch | Tests | Files | Command |
|-------|-------|-------|---------|
| 3.1 | ~8 | test_discovery_routing.py | `pytest tests/unit/test_agents/test_orchestrator/test_discovery_routing.py -n 2` |
| 3.2 | ~8 | test_causal_narrative.py | `pytest tests/unit/test_agents/test_explainer/test_causal_narrative.py -n 2` |
| 3.3 | ~10 | test_structural_drift.py | `pytest tests/unit/test_agents/test_drift_monitor/test_structural_drift.py -n 2` |

---

## Implementation Checklist

### Wave 1: CRITICAL
- [ ] Phase 1.1: Feature Analyzer State Extension
- [ ] Phase 1.2: Feature Analyzer Causal Ranker Node
- [ ] Phase 1.3: Feature Analyzer Importance Narrator Update
- [ ] Phase 1.4: Causal Impact Agent Integration
- [ ] Phase 1.5: Tier 0 Contract Updates
- [ ] Wave 1 Tests Pass (4 batches, ~35 tests)

### Wave 2: HIGH
- [ ] Phase 2.1: Gap Analyzer Causal Filtering
- [ ] Phase 2.2: Heterogeneous Optimizer DAG Validation
- [ ] Phase 2.3: Experiment Designer DAG-Aware Validation
- [ ] Phase 2.4: Contract Updates (data, tier3)
- [ ] Wave 2 Tests Pass (3 batches, ~30 tests)

### Wave 3: MEDIUM
- [ ] Phase 3.1: Orchestrator Discovery Routing
- [ ] Phase 3.2: Explainer Causal Narrative
- [ ] Phase 3.3: Drift Monitor Structural Drift
- [ ] Phase 3.4: Integration Contract Updates
- [ ] Wave 3 Tests Pass (3 batches, ~26 tests)

### Final Validation
- [ ] All 91 new tests pass
- [ ] Existing discovery tests pass (124 tests)
- [ ] Integration tests pass
- [ ] Contracts reviewed and approved

---

## Resource Constraints

**Memory-Safe Testing**:
- Max 2 workers per batch (`-n 2`)
- Run batches sequentially
- Use `--dist=loadscope` for module grouping

**Recommended Test Command**:
```bash
# Wave 1
pytest tests/unit/test_agents/test_ml_foundation/test_feature_analyzer/ -n 2 --dist=loadscope

# Wave 2
pytest tests/unit/test_agents/test_gap_analyzer/test_causal_prioritization.py -n 2
pytest tests/unit/test_agents/test_heterogeneous_optimizer/test_dag_validation.py -n 2
pytest tests/unit/test_agents/test_experiment_designer/test_dag_validation.py -n 2

# Wave 3
pytest tests/unit/test_agents/test_orchestrator/test_discovery_routing.py -n 2
pytest tests/unit/test_agents/test_explainer/test_causal_narrative.py -n 2
pytest tests/unit/test_agents/test_drift_monitor/test_structural_drift.py -n 2
```

---

## Dependencies

### Causal Discovery Module (Complete)
- `src/causal_engine/discovery/runner.py` - DiscoveryRunner
- `src/causal_engine/discovery/gate.py` - DiscoveryGate
- `src/causal_engine/discovery/driver_ranker.py` - DriverRanker
- `src/causal_engine/discovery/algorithms/fci_wrapper.py` - FCI with latent confounders
- `src/causal_engine/discovery/observability.py` - Opik tracing

### External Dependencies
- `causal-learn` - FCI, PC algorithms
- `lingam` - LiNGAM algorithm
- `networkx` - DAG operations
- `scipy` - Spearman correlation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Memory exhaustion during tests | Use 2 workers max, run batches sequentially |
| Discovery timeout on large datasets | Add timeout config to DiscoveryConfig |
| Gate rejection rate too high | Tune thresholds in GateConfig |
| FCI latent confounder false positives | Cross-validate with domain knowledge |

---

## Success Criteria

1. **All agents integrate with DiscoveryRunner** without breaking existing functionality
2. **DriverRanker comparisons** available in Feature Analyzer output
3. **Gate decisions** propagate through agent chain
4. **Latent confounders** detected by FCI influence downstream agents
5. **All tests pass** within memory constraints
6. **Contracts updated** to V4.4 with discovery fields
