# Tier 3: Drift Monitor Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 3 (Monitoring) |
| **Agent Type** | Standard (Fast Path) |
| **Model Tier** | Haiku/Sonnet |
| **Latency Tolerance** | Low (<10s) |
| **Critical Path** | No - monitoring agent |

## Domain Scope

You are the specialist for the Tier 3 Drift Monitor Agent:
- `src/agents/drift_monitor/` - Data and model drift detection

This is a **Standard Fast Agent** optimized for:
- Quick drift detection
- Statistical tests for distribution shift
- Alert generation
- Minimal LLM usage

## Design Principles

### Fast Path Optimization
The Drift Monitor is optimized for speed:
- Pre-computed baselines
- Efficient statistical tests
- Batch processing
- Alert thresholds, not deep analysis

### Responsibilities
1. **Data Drift Detection** - Monitor feature distribution changes
2. **Model Drift Detection** - Monitor prediction distribution changes
3. **Concept Drift Detection** - Monitor outcome relationship changes
4. **Alert Generation** - Trigger alerts when thresholds exceeded

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DRIFT MONITOR AGENT                         │
│                      (Fast Path - <10s)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │    DATA     │    │   MODEL     │    │  CONCEPT    │         │
│  │   DRIFT     │    │   DRIFT     │    │   DRIFT     │         │
│  │  DETECTOR   │    │  DETECTOR   │    │  DETECTOR   │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                   │                 │
│         └──────────────────┼───────────────────┘                │
│                            ▼                                     │
│                  ┌─────────────────┐                            │
│                  │     ALERT       │                            │
│                  │   AGGREGATOR    │                            │
│                  └─────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
drift_monitor/
├── agent.py              # Main DriftMonitorAgent class
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── nodes/
│   ├── data_drift.py     # Feature distribution drift
│   ├── model_drift.py    # Prediction drift
│   ├── concept_drift.py  # Target relationship drift
│   └── alert_aggregator.py # Alert generation
├── detectors/
│   ├── ks_test.py        # Kolmogorov-Smirnov test
│   ├── psi.py            # Population Stability Index
│   ├── chi_square.py     # Chi-square test for categorical
│   └── js_divergence.py  # Jensen-Shannon divergence
└── baselines.py          # Baseline distribution management
```

## LangGraph State Definition

```python
# src/agents/drift_monitor/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class DriftResult(TypedDict):
    """Individual drift detection result"""
    feature: str
    drift_type: Literal["data", "model", "concept"]
    test_statistic: float
    p_value: float
    drift_detected: bool
    severity: Literal["none", "low", "medium", "high", "critical"]
    baseline_period: str
    current_period: str

class DriftAlert(TypedDict):
    """Drift alert for notification"""
    alert_id: str
    severity: Literal["warning", "critical"]
    drift_type: Literal["data", "model", "concept"]
    affected_features: List[str]
    message: str
    recommended_action: str
    timestamp: str

class DriftMonitorState(TypedDict):
    """Complete state for Drift Monitor agent"""
    
    # === INPUT ===
    query: str
    model_id: Optional[str]
    features_to_monitor: List[str]
    time_window: str  # e.g., "7d", "30d"
    brand: Optional[str]
    
    # === CONFIGURATION ===
    significance_level: float  # Default: 0.05
    psi_threshold: float  # Default: 0.1 (warning), 0.25 (critical)
    check_data_drift: bool
    check_model_drift: bool
    check_concept_drift: bool
    
    # === DETECTION OUTPUTS ===
    data_drift_results: Optional[List[DriftResult]]
    model_drift_results: Optional[List[DriftResult]]
    concept_drift_results: Optional[List[DriftResult]]
    
    # === AGGREGATED OUTPUTS ===
    overall_drift_score: Optional[float]  # 0-1, composite score
    features_with_drift: Optional[List[str]]
    alerts: Optional[List[DriftAlert]]
    
    # === SUMMARY ===
    drift_summary: Optional[str]
    recommended_actions: Optional[List[str]]
    
    # === EXECUTION METADATA ===
    detection_latency_ms: int
    features_checked: int
    baseline_timestamp: str
    current_timestamp: str
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "detecting", "aggregating", "completed", "failed"]
```

## Node Implementations

### Data Drift Detector Node

```python
# src/agents/drift_monitor/nodes/data_drift.py

import asyncio
import time
from typing import List, Dict, Any
import numpy as np
from scipy import stats

from ..state import DriftMonitorState, DriftResult

class DataDriftDetectorNode:
    """
    Detect distribution drift in input features
    Uses PSI and KS tests
    """
    
    def __init__(self, data_connector, baseline_store):
        self.data_connector = data_connector
        self.baseline_store = baseline_store
    
    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        start_time = time.time()
        
        if not state.get("check_data_drift", True):
            return {**state, "data_drift_results": []}
        
        try:
            features = state["features_to_monitor"]
            
            # Fetch current and baseline data in parallel
            current_data, baseline_data = await asyncio.gather(
                self._fetch_current_data(state),
                self._fetch_baseline_data(state)
            )
            
            # Run drift detection in parallel for all features
            drift_tasks = [
                self._detect_feature_drift(
                    feature, 
                    current_data.get(feature), 
                    baseline_data.get(feature),
                    state
                )
                for feature in features
            ]
            
            results = await asyncio.gather(*drift_tasks)
            
            detection_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "data_drift_results": [r for r in results if r is not None],
                "detection_latency_ms": state.get("detection_latency_ms", 0) + detection_time
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "data_drift", "error": str(e)}],
                "status": "failed"
            }
    
    async def _fetch_current_data(self, state: DriftMonitorState) -> Dict[str, np.ndarray]:
        """Fetch current period data"""
        return await self.data_connector.query_distributions(
            features=state["features_to_monitor"],
            time_window=state["time_window"],
            brand=state.get("brand")
        )
    
    async def _fetch_baseline_data(self, state: DriftMonitorState) -> Dict[str, np.ndarray]:
        """Fetch baseline distributions"""
        return await self.baseline_store.get_baselines(
            features=state["features_to_monitor"],
            model_id=state.get("model_id")
        )
    
    async def _detect_feature_drift(
        self,
        feature: str,
        current: np.ndarray,
        baseline: np.ndarray,
        state: DriftMonitorState
    ) -> DriftResult:
        """Detect drift for a single feature"""
        
        if current is None or baseline is None:
            return None
        
        if len(current) < 30 or len(baseline) < 30:
            return None
        
        # Calculate PSI
        psi = self._calculate_psi(baseline, current)
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(baseline, current)
        
        # Determine drift severity
        significance = state.get("significance_level", 0.05)
        psi_warning = state.get("psi_threshold", 0.1)
        psi_critical = 0.25
        
        if psi >= psi_critical or p_value < significance / 10:
            severity = "critical"
            drift_detected = True
        elif psi >= psi_warning or p_value < significance:
            severity = "high" if psi >= 0.2 else "medium"
            drift_detected = True
        elif psi >= 0.05:
            severity = "low"
            drift_detected = True
        else:
            severity = "none"
            drift_detected = False
        
        return DriftResult(
            feature=feature,
            drift_type="data",
            test_statistic=psi,
            p_value=p_value,
            drift_detected=drift_detected,
            severity=severity,
            baseline_period="baseline",
            current_period=state["time_window"]
        )
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        
        # Create bins from expected distribution
        _, bin_edges = np.histogram(expected, bins=bins)
        
        # Calculate proportions
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
        
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_pct = np.clip(expected_pct, 0.0001, None)
        actual_pct = np.clip(actual_pct, 0.0001, None)
        
        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return float(psi)
```

### Model Drift Detector Node

```python
# src/agents/drift_monitor/nodes/model_drift.py

import asyncio
import time
from typing import List
import numpy as np
from scipy import stats

from ..state import DriftMonitorState, DriftResult

class ModelDriftDetectorNode:
    """
    Detect drift in model predictions
    Monitors prediction distribution changes
    """
    
    def __init__(self, prediction_store):
        self.prediction_store = prediction_store
    
    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        start_time = time.time()
        
        if not state.get("check_model_drift", True):
            return {**state, "model_drift_results": []}
        
        try:
            model_id = state.get("model_id")
            if not model_id:
                return {
                    **state,
                    "model_drift_results": [],
                    "warnings": state.get("warnings", []) + ["No model_id provided for model drift check"]
                }
            
            # Fetch prediction distributions
            current_preds = await self.prediction_store.get_predictions(
                model_id=model_id,
                time_window=state["time_window"]
            )
            
            baseline_preds = await self.prediction_store.get_baseline_predictions(
                model_id=model_id
            )
            
            results = []
            
            # Check prediction score distribution
            if current_preds.get("scores") is not None and baseline_preds.get("scores") is not None:
                score_drift = self._check_distribution_drift(
                    baseline_preds["scores"],
                    current_preds["scores"],
                    "prediction_score",
                    state
                )
                results.append(score_drift)
            
            # Check prediction class distribution (for classification)
            if current_preds.get("classes") is not None and baseline_preds.get("classes") is not None:
                class_drift = self._check_categorical_drift(
                    baseline_preds["classes"],
                    current_preds["classes"],
                    "prediction_class",
                    state
                )
                results.append(class_drift)
            
            detection_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "model_drift_results": [r for r in results if r is not None],
                "detection_latency_ms": state.get("detection_latency_ms", 0) + detection_time
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "model_drift", "error": str(e)}],
                "model_drift_results": []
            }
    
    def _check_distribution_drift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        feature_name: str,
        state: DriftMonitorState
    ) -> DriftResult:
        """Check continuous distribution drift"""
        
        ks_stat, p_value = stats.ks_2samp(baseline, current)
        significance = state.get("significance_level", 0.05)
        
        if p_value < significance / 10:
            severity = "critical"
        elif p_value < significance:
            severity = "high"
        elif p_value < significance * 2:
            severity = "medium"
        else:
            severity = "none"
        
        return DriftResult(
            feature=feature_name,
            drift_type="model",
            test_statistic=ks_stat,
            p_value=p_value,
            drift_detected=p_value < significance,
            severity=severity,
            baseline_period="baseline",
            current_period=state["time_window"]
        )
    
    def _check_categorical_drift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        feature_name: str,
        state: DriftMonitorState
    ) -> DriftResult:
        """Check categorical distribution drift using chi-square"""
        
        # Get unique classes
        all_classes = np.unique(np.concatenate([baseline, current]))
        
        # Count frequencies
        baseline_counts = np.array([np.sum(baseline == c) for c in all_classes])
        current_counts = np.array([np.sum(current == c) for c in all_classes])
        
        # Chi-square test
        # Normalize to same total
        expected = baseline_counts * len(current) / len(baseline)
        chi2, p_value = stats.chisquare(current_counts, expected)
        
        significance = state.get("significance_level", 0.05)
        
        return DriftResult(
            feature=feature_name,
            drift_type="model",
            test_statistic=chi2,
            p_value=p_value,
            drift_detected=p_value < significance,
            severity="high" if p_value < significance else "none",
            baseline_period="baseline",
            current_period=state["time_window"]
        )
```

### Alert Aggregator Node

```python
# src/agents/drift_monitor/nodes/alert_aggregator.py

import time
from typing import List, Dict
from datetime import datetime
import uuid

from ..state import DriftMonitorState, DriftResult, DriftAlert

class AlertAggregatorNode:
    """
    Aggregate drift results and generate alerts
    Pure logic - no LLM needed
    """
    
    SEVERITY_WEIGHTS = {
        "none": 0,
        "low": 0.25,
        "medium": 0.5,
        "high": 0.75,
        "critical": 1.0
    }
    
    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            # Collect all drift results
            all_results = []
            all_results.extend(state.get("data_drift_results", []))
            all_results.extend(state.get("model_drift_results", []))
            all_results.extend(state.get("concept_drift_results", []))
            
            # Calculate overall drift score
            drift_score = self._calculate_drift_score(all_results)
            
            # Identify features with drift
            features_with_drift = [
                r["feature"] for r in all_results if r["drift_detected"]
            ]
            
            # Generate alerts
            alerts = self._generate_alerts(all_results)
            
            # Generate summary
            summary = self._generate_summary(all_results, drift_score, alerts)
            
            # Generate recommended actions
            actions = self._generate_recommendations(all_results, alerts)
            
            total_time = state.get("detection_latency_ms", 0) + int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "overall_drift_score": drift_score,
                "features_with_drift": features_with_drift,
                "alerts": alerts,
                "drift_summary": summary,
                "recommended_actions": actions,
                "features_checked": len(all_results),
                "detection_latency_ms": total_time,
                "current_timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "alert_aggregator", "error": str(e)}],
                "status": "failed"
            }
    
    def _calculate_drift_score(self, results: List[DriftResult]) -> float:
        """Calculate composite drift score 0-1"""
        if not results:
            return 0.0
        
        weighted_sum = sum(
            self.SEVERITY_WEIGHTS.get(r["severity"], 0)
            for r in results
        )
        
        return weighted_sum / len(results)
    
    def _generate_alerts(self, results: List[DriftResult]) -> List[DriftAlert]:
        """Generate alerts for significant drift"""
        
        alerts = []
        
        # Group by drift type and severity
        critical_data = [r for r in results if r["drift_type"] == "data" and r["severity"] == "critical"]
        critical_model = [r for r in results if r["drift_type"] == "model" and r["severity"] == "critical"]
        critical_concept = [r for r in results if r["drift_type"] == "concept" and r["severity"] == "critical"]
        
        high_severity = [r for r in results if r["severity"] in ["high", "critical"]]
        
        # Critical data drift alert
        if critical_data:
            alerts.append(DriftAlert(
                alert_id=str(uuid.uuid4())[:8],
                severity="critical",
                drift_type="data",
                affected_features=[r["feature"] for r in critical_data],
                message=f"Critical data drift detected in {len(critical_data)} feature(s)",
                recommended_action="Investigate data pipeline and retrain model if necessary",
                timestamp=datetime.now().isoformat()
            ))
        
        # Critical model drift alert
        if critical_model:
            alerts.append(DriftAlert(
                alert_id=str(uuid.uuid4())[:8],
                severity="critical",
                drift_type="model",
                affected_features=[r["feature"] for r in critical_model],
                message=f"Critical model prediction drift detected",
                recommended_action="Review model performance and consider retraining",
                timestamp=datetime.now().isoformat()
            ))
        
        # Warning for high severity
        warning_results = [r for r in results if r["severity"] == "high" and r not in critical_data + critical_model]
        if warning_results:
            alerts.append(DriftAlert(
                alert_id=str(uuid.uuid4())[:8],
                severity="warning",
                drift_type="data",
                affected_features=[r["feature"] for r in warning_results],
                message=f"Elevated drift detected in {len(warning_results)} feature(s)",
                recommended_action="Monitor closely and prepare for potential intervention",
                timestamp=datetime.now().isoformat()
            ))
        
        return alerts
    
    def _generate_summary(self, results: List[DriftResult], score: float, alerts: List[DriftAlert]) -> str:
        """Generate human-readable summary"""
        
        if not results:
            return "No features monitored in this check."
        
        drift_detected = [r for r in results if r["drift_detected"]]
        critical = len([a for a in alerts if a["severity"] == "critical"])
        
        if score < 0.1:
            status = "healthy"
        elif score < 0.3:
            status = "showing minor drift"
        elif score < 0.5:
            status = "showing moderate drift"
        else:
            status = "showing significant drift"
        
        summary = f"Drift monitoring complete. System is {status}. "
        summary += f"Checked {len(results)} features, {len(drift_detected)} showing drift. "
        
        if critical > 0:
            summary += f"{critical} critical alert(s) generated."
        
        return summary
    
    def _generate_recommendations(self, results: List[DriftResult], alerts: List[DriftAlert]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        critical_alerts = [a for a in alerts if a["severity"] == "critical"]
        
        if critical_alerts:
            recommendations.append("URGENT: Address critical drift alerts immediately")
            
            data_alerts = [a for a in critical_alerts if a["drift_type"] == "data"]
            if data_alerts:
                recommendations.append("Investigate data pipeline for schema or distribution changes")
            
            model_alerts = [a for a in critical_alerts if a["drift_type"] == "model"]
            if model_alerts:
                recommendations.append("Evaluate model performance metrics and consider retraining")
        
        high_drift = [r for r in results if r["severity"] in ["high", "medium"]]
        if high_drift and not critical_alerts:
            recommendations.append("Schedule review of features showing elevated drift")
            recommendations.append("Consider updating baseline distributions after investigation")
        
        if not recommendations:
            recommendations.append("Continue routine monitoring")
        
        return recommendations
```

## Graph Assembly

```python
# src/agents/drift_monitor/graph.py

from langgraph.graph import StateGraph, END

from .state import DriftMonitorState
from .nodes.data_drift import DataDriftDetectorNode
from .nodes.model_drift import ModelDriftDetectorNode
from .nodes.concept_drift import ConceptDriftDetectorNode
from .nodes.alert_aggregator import AlertAggregatorNode

def build_drift_monitor_graph(
    data_connector,
    baseline_store,
    prediction_store
):
    """
    Build the Drift Monitor agent graph
    
    Architecture (parallel detection):
        ┌─► [data_drift] ──┐
        │                  │
    START─┼─► [model_drift] ─┼─► [aggregate] ─► END
        │                  │
        └─► [concept_drift]┘
    """
    
    # Initialize nodes
    data_drift = DataDriftDetectorNode(data_connector, baseline_store)
    model_drift = ModelDriftDetectorNode(prediction_store)
    concept_drift = ConceptDriftDetectorNode(data_connector)
    aggregator = AlertAggregatorNode()
    
    # Build graph
    workflow = StateGraph(DriftMonitorState)
    
    # Add nodes
    workflow.add_node("data_drift", data_drift.execute)
    workflow.add_node("model_drift", model_drift.execute)
    workflow.add_node("concept_drift", concept_drift.execute)
    workflow.add_node("aggregate", aggregator.execute)
    
    # Parallel entry to all drift detectors
    workflow.set_entry_point("data_drift")
    
    # Sequential for simplicity (could be parallel with fan-out/fan-in)
    workflow.add_edge("data_drift", "model_drift")
    workflow.add_edge("model_drift", "concept_drift")
    workflow.add_edge("concept_drift", "aggregate")
    workflow.add_edge("aggregate", END)
    
    return workflow.compile()
```

## Integration Contracts

### Input Contract
```python
class DriftMonitorInput(BaseModel):
    query: str
    model_id: Optional[str] = None
    features_to_monitor: List[str]
    time_window: str = "7d"
    brand: Optional[str] = None
    significance_level: float = 0.05
```

### Output Contract
```python
class DriftMonitorOutput(BaseModel):
    overall_drift_score: float
    features_with_drift: List[str]
    alerts: List[DriftAlert]
    drift_summary: str
    recommended_actions: List[str]
    detection_latency_ms: int
```

## Handoff Format

```yaml
drift_monitor_handoff:
  agent: drift_monitor
  analysis_type: drift_detection
  key_findings:
    - drift_score: <0-1>
    - features_with_drift: [<feature 1>, <feature 2>]
    - critical_alerts: <count>
  alerts:
    - severity: <warning|critical>
      type: <data|model|concept>
      message: <description>
  recommendations:
    - <recommendation 1>
  requires_further_analysis: <bool>
  suggested_next_agent: <health_score|experiment_designer>
```

## Testing Requirements

```
tests/unit/test_agents/test_drift_monitor/
├── test_data_drift.py     # PSI and KS tests
├── test_model_drift.py    # Prediction drift
├── test_concept_drift.py  # Target relationship drift
├── test_alert_aggregator.py # Alert generation
└── test_integration.py    # End-to-end flow
```

### Performance Requirements
- Total detection: <10s for 50 features
- Alert generation: <100ms
- No LLM calls in critical path

---

## Cognitive RAG DSPy Integration

### Integration Overview

The Drift Monitor integrates with the Cognitive RAG DSPy system to receive historical drift patterns and contribute drift alerts to the episodic memory for future reference.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DRIFT MONITOR - COGNITIVE RAG INTEGRATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 2: INVESTIGATOR                    PHASE 3: AGENT                    │
│  ┌──────────────────────┐                ┌─────────────────────────────┐    │
│  │ EvidenceSynthesis    │                │      DRIFT MONITOR          │    │
│  │ Signature            │                │                             │    │
│  │  - evidence_items    │───────────────►│  1. Parse drift history     │    │
│  │  - user_query        │                │  2. Compare to baselines    │    │
│  │                      │                │  3. Contextualize alerts    │    │
│  │ OUTPUTS:             │                │  4. Generate recommendations│    │
│  │  - synthesized_summary                │                             │    │
│  │  - drift_history     ─────────────────│──► Historical drift events  │    │
│  │  - resolution_patterns────────────────│──► How past drift resolved  │    │
│  └──────────────────────┘                └──────────────┬──────────────┘    │
│                                                         │                   │
│                                                         ▼                   │
│                                          ┌─────────────────────────────┐    │
│                                          │    TRAINING SIGNAL          │    │
│                                          │  DriftMonitorTrainingSignal │    │
│                                          │   - alert_accuracy          │    │
│                                          │   - false_positive_rate     │    │
│                                          │   - resolution_time         │    │
│                                          └──────────────┬──────────────┘    │
│                                                         │                   │
│  PHASE 4: REFLECTOR                                     ▼                   │
│  ┌──────────────────────┐                ┌─────────────────────────────┐    │
│  │ MemoryWorthiness     │◄───────────────│   MEMORY CONTRIBUTION       │    │
│  │ Signature            │                │   - drift_events (episodic) │    │
│  │   store_drift_event  │                │   - alert_patterns          │    │
│  └──────────────────────┘                └─────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DSPy Signatures Consumed

The Drift Monitor consumes evidence about historical drift patterns:

```python
# From CognitiveRAG Phase 2 - Consumed by Drift Monitor

class EvidenceSynthesisSignature(dspy.Signature):
    """Synthesize evidence from multi-hop retrieval into coherent context."""
    evidence_items: List[Evidence] = dspy.InputField(desc="Retrieved evidence items")
    user_query: str = dspy.InputField(desc="Original user query")
    investigation_path: List[Dict] = dspy.InputField(desc="Multi-hop retrieval path")

    synthesized_summary: str = dspy.OutputField(desc="Coherent synthesis of all evidence")
    drift_history: List[Dict] = dspy.OutputField(desc="Historical drift events for features")
    resolution_patterns: List[Dict] = dspy.OutputField(desc="How past drift was resolved")
    baseline_evolution: Dict[str, Any] = dspy.OutputField(desc="How baselines changed over time")
    confidence_score: float = dspy.OutputField(desc="Confidence in synthesis 0.0-1.0")
```

### Integration in Alert Aggregator Node

```python
# src/agents/drift_monitor/nodes/alert_aggregator.py

from typing import TypedDict, List, Dict, Any, Optional

class DriftCognitiveContext(TypedDict):
    """Cognitive context from CognitiveRAG for drift monitoring."""
    synthesized_summary: str
    drift_history: List[Dict[str, Any]]
    resolution_patterns: List[Dict[str, Any]]
    baseline_evolution: Dict[str, Any]
    evidence_confidence: float

class AlertAggregatorNode:
    """Alert Aggregator with Cognitive RAG integration."""

    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        try:
            # Extract cognitive context if available
            cognitive_context = self._extract_cognitive_context(state)

            # Collect all drift results
            all_results = self._collect_drift_results(state)

            # Calculate drift score
            drift_score = self._calculate_drift_score(all_results)

            # Generate alerts with historical context
            alerts = self._generate_alerts_with_context(
                all_results, cognitive_context
            )

            # Generate contextualized recommendations
            recommendations = self._generate_recommendations_with_history(
                all_results, alerts, cognitive_context
            )

            return {
                **state,
                "overall_drift_score": drift_score,
                "alerts": alerts,
                "recommended_actions": recommendations,
                "cognitive_enrichment_applied": cognitive_context is not None,
                "status": "completed"
            }

        except Exception as e:
            return {
                **state,
                "errors": [{"node": "alert_aggregator", "error": str(e)}],
                "status": "failed"
            }

    def _extract_cognitive_context(
        self,
        state: DriftMonitorState
    ) -> Optional[DriftCognitiveContext]:
        """Extract cognitive context from state."""

        cognitive_input = state.get("cognitive_enrichment")
        if not cognitive_input:
            return None

        return DriftCognitiveContext(
            synthesized_summary=cognitive_input.get("synthesized_summary", ""),
            drift_history=cognitive_input.get("drift_history", []),
            resolution_patterns=cognitive_input.get("resolution_patterns", []),
            baseline_evolution=cognitive_input.get("baseline_evolution", {}),
            evidence_confidence=cognitive_input.get("confidence_score", 0.5)
        )

    def _generate_alerts_with_context(
        self,
        results: List[DriftResult],
        cognitive_context: Optional[DriftCognitiveContext]
    ) -> List[DriftAlert]:
        """Generate alerts enriched with historical context."""

        alerts = []

        # Group by drift type and severity
        critical_results = [r for r in results if r["severity"] == "critical"]
        high_results = [r for r in results if r["severity"] == "high"]

        for result in critical_results + high_results:
            # Check for historical pattern match
            historical_match = None
            if cognitive_context:
                historical_match = self._find_historical_match(
                    result, cognitive_context["drift_history"]
                )

            alert = DriftAlert(
                alert_id=str(uuid.uuid4())[:8],
                severity="critical" if result["severity"] == "critical" else "warning",
                drift_type=result["drift_type"],
                affected_features=[result["feature"]],
                message=self._generate_alert_message(result, historical_match),
                recommended_action=self._get_recommended_action(
                    result, cognitive_context
                ),
                timestamp=datetime.now().isoformat()
            )

            # Enrich with historical context
            if historical_match:
                alert["historical_occurrence"] = historical_match
                alert["recurrence_pattern"] = True

            alerts.append(alert)

        return alerts

    def _find_historical_match(
        self,
        result: DriftResult,
        drift_history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find matching historical drift event."""

        for historical in drift_history:
            if (historical.get("feature") == result["feature"] and
                historical.get("drift_type") == result["drift_type"]):
                return {
                    "previous_occurrence": historical.get("timestamp"),
                    "previous_severity": historical.get("severity"),
                    "resolution": historical.get("resolution"),
                    "time_to_resolve": historical.get("time_to_resolve")
                }

        return None

    def _get_recommended_action(
        self,
        result: DriftResult,
        cognitive_context: Optional[DriftCognitiveContext]
    ) -> str:
        """Get recommended action, informed by resolution patterns."""

        default_action = self._get_default_action(result)

        if not cognitive_context or not cognitive_context["resolution_patterns"]:
            return default_action

        # Find matching resolution pattern
        for pattern in cognitive_context["resolution_patterns"]:
            if (pattern.get("feature") == result["feature"] or
                pattern.get("drift_type") == result["drift_type"]):
                if pattern.get("success_rate", 0) > 0.7:
                    return f"{pattern.get('resolution_action', default_action)} (historically effective)"

        return default_action

    def _get_default_action(self, result: DriftResult) -> str:
        """Get default action based on drift type."""
        if result["drift_type"] == "data":
            return "Investigate data pipeline for schema or distribution changes"
        elif result["drift_type"] == "model":
            return "Evaluate model performance metrics and consider retraining"
        else:
            return "Review feature-outcome relationships for concept drift"
```

### Training Signal for MIPROv2

```python
# src/agents/drift_monitor/training_signals.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class DriftMonitorTrainingSignal:
    """
    Training signal for CognitiveRAG's EvidenceSynthesisSignature.

    Emitted after drift detection to inform MIPROv2 about
    the quality of drift history and resolution patterns.
    """

    # Identifiers
    monitoring_session_id: str
    monitoring_timestamp: datetime

    # Detection Metrics
    features_monitored: int
    drift_detected_count: int
    alerts_generated: int
    critical_alerts: int

    # Historical Context Usage
    drift_history_used: bool
    resolution_patterns_used: bool
    historical_matches_found: int

    # Alert Quality (validated later)
    false_positive_rate: Optional[float] = None  # % of alerts that were false
    true_positive_rate: Optional[float] = None  # % of actual drift caught
    resolution_time_hours: Optional[float] = None  # Average time to resolve

    def compute_reward(self) -> float:
        """Compute reward signal for MIPROv2 optimization."""

        base_reward = 0.5

        # Reward for using historical context effectively
        if self.drift_history_used and self.historical_matches_found > 0:
            base_reward += 0.15 * min(self.historical_matches_found / 3, 1.0)

        # Reward for appropriate alert generation
        if self.features_monitored > 0:
            alert_rate = self.alerts_generated / self.features_monitored
            # Reward moderate alert rates (not too many, not too few)
            if 0.05 <= alert_rate <= 0.3:
                base_reward += 0.1

        # Penalty for false positives (if validated)
        if self.false_positive_rate is not None:
            base_reward -= 0.2 * self.false_positive_rate

        # Reward for accurate detection (if validated)
        if self.true_positive_rate is not None:
            base_reward += 0.15 * self.true_positive_rate

        return max(0.0, min(base_reward, 1.0))

    def to_training_example(self) -> Dict[str, Any]:
        """Convert to DSPy training example format."""
        return {
            "input": {
                "query_type": "drift_monitoring",
                "features_monitored": self.features_monitored,
                "drift_history_available": self.drift_history_used,
            },
            "output": {
                "drift_detected": self.drift_detected_count,
                "alerts_generated": self.alerts_generated,
                "historical_matches": self.historical_matches_found,
            },
            "reward": self.compute_reward(),
            "metadata": {
                "session_id": self.monitoring_session_id,
                "timestamp": self.monitoring_timestamp.isoformat(),
            }
        }


async def emit_drift_monitor_signal(
    result: DriftMonitorOutput,
    state: DriftMonitorState,
    signal_store: Any
) -> None:
    """Emit training signal after drift monitoring completion."""

    cognitive_enrichment = state.get("cognitive_enrichment")
    alerts = result.alerts

    signal = DriftMonitorTrainingSignal(
        monitoring_session_id=str(uuid.uuid4()),
        monitoring_timestamp=datetime.now(),
        features_monitored=result.features_checked,
        drift_detected_count=len(result.features_with_drift),
        alerts_generated=len(alerts),
        critical_alerts=len([a for a in alerts if a["severity"] == "critical"]),
        drift_history_used=cognitive_enrichment is not None and len(cognitive_enrichment.get("drift_history", [])) > 0,
        resolution_patterns_used=cognitive_enrichment is not None and len(cognitive_enrichment.get("resolution_patterns", [])) > 0,
        historical_matches_found=len([a for a in alerts if a.get("recurrence_pattern")]),
    )

    await signal_store.store_signal(
        agent="drift_monitor",
        signal=signal.to_training_example()
    )
```

### Memory Contribution

```python
# src/agents/drift_monitor/memory_contribution.py

from typing import Any
from datetime import datetime

async def contribute_to_memory(
    result: DriftMonitorOutput,
    state: DriftMonitorState,
    memory_backend: Any
) -> None:
    """
    Contribute drift events to CognitiveRAG's episodic memory.

    Stores in 'drift_events' for future pattern matching.
    """

    for alert in result.alerts:
        drift_event = {
            "alert_id": alert["alert_id"],
            "severity": alert["severity"],
            "drift_type": alert["drift_type"],
            "affected_features": alert["affected_features"],
            "message": alert["message"],
            "recommended_action": alert["recommended_action"],
            "timestamp": alert["timestamp"],
            "model_id": state.get("model_id"),
            "time_window": state.get("time_window"),
            "overall_drift_score": result.overall_drift_score,
            "status": "open",  # Track lifecycle: open -> investigating -> resolved
            "resolution": None,  # Filled when resolved
            "time_to_resolve": None,
        }

        await memory_backend.store(
            memory_type="EPISODIC",
            content=drift_event,
            metadata={
                "agent": "drift_monitor",
                "index": "drift_events",
                "embedding_fields": ["message", "affected_features", "recommended_action"],
                "temporal_weight": 0.9  # Recent events weighted higher
            }
        )
```

### Cognitive-Enriched Input TypedDict

```python
# src/agents/drift_monitor/state.py (additions)

class DriftMonitorCognitiveInput(TypedDict):
    """Extended input with cognitive enrichment from CognitiveRAG."""

    # Standard inputs
    query: str
    model_id: Optional[str]
    features_to_monitor: List[str]
    time_window: str

    # Cognitive enrichment from Phase 2
    cognitive_enrichment: Optional[Dict[str, Any]]
    # Contains:
    #   - synthesized_summary: str
    #   - drift_history: List[Dict]
    #   - resolution_patterns: List[Dict]
    #   - baseline_evolution: Dict[str, Any]
    #   - confidence_score: float

    # Working memory context
    working_memory: Optional[Dict[str, Any]]
```

### Configuration

```yaml
# config/agents/drift_monitor.yaml

drift_monitor:
  # ... existing config ...

  cognitive_rag_integration:
    enabled: true

    # Evidence consumption
    consume_drift_history: true
    consume_resolution_patterns: true
    use_baseline_evolution: true

    # Historical pattern matching
    match_historical_patterns: true
    min_history_confidence: 0.6

    # Training signal emission
    emit_training_signals: true
    signal_destination: "feedback_learner"

    # Memory contribution
    contribute_to_memory: true
    memory_type: "EPISODIC"  # Time-sensitive events
    memory_index: "drift_events"
```

### Testing Requirements

```python
# tests/unit/test_agents/test_drift_monitor/test_cognitive_integration.py

import pytest
from src.agents.drift_monitor.nodes.alert_aggregator import AlertAggregatorNode
from src.agents.drift_monitor.training_signals import DriftMonitorTrainingSignal

class TestDriftMonitorCognitiveIntegration:
    """Tests for Drift Monitor Cognitive RAG integration."""

    @pytest.mark.asyncio
    async def test_cognitive_context_extraction(self):
        """Test extraction of cognitive context from state."""
        pass

    @pytest.mark.asyncio
    async def test_historical_pattern_matching(self):
        """Test matching current drift to historical events."""
        pass

    @pytest.mark.asyncio
    async def test_resolution_pattern_usage(self):
        """Test using resolution patterns for recommendations."""
        pass

    @pytest.mark.asyncio
    async def test_training_signal_emission(self):
        """Test training signal generation and reward computation."""
        signal = DriftMonitorTrainingSignal(
            monitoring_session_id="test",
            monitoring_timestamp=datetime.now(),
            features_monitored=50,
            drift_detected_count=3,
            alerts_generated=2,
            critical_alerts=1,
            drift_history_used=True,
            resolution_patterns_used=True,
            historical_matches_found=1,
        )

        reward = signal.compute_reward()
        assert 0.0 <= reward <= 1.0

    @pytest.mark.asyncio
    async def test_episodic_memory_contribution(self):
        """Test contributing drift events to episodic memory."""
        pass

    @pytest.mark.asyncio
    async def test_without_cognitive_enrichment(self):
        """Test graceful operation without cognitive context."""
        pass
```
