# Tier 4: Prediction Synthesizer Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 4 (ML Predictions) |
| **Agent Type** | Standard (Computational) |
| **Model Tier** | Sonnet |
| **Latency Tolerance** | Low (<15s) |
| **Critical Path** | No - predictions can be async |

## Domain Scope

You are the specialist for the Tier 4 Prediction Synthesizer Agent:
- `src/agents/prediction_synthesizer/` - ML prediction aggregation and synthesis

This is a **Standard Computational Agent** for:
- Aggregating predictions from multiple models
- Ensemble methods
- Uncertainty quantification
- Prediction explanations

## Design Principles

### Ensemble-First Design
The Prediction Synthesizer combines multiple model outputs:
- Model averaging
- Stacking
- Confidence-weighted aggregation
- Disagreement detection

### Responsibilities
1. **Model Orchestration** - Coordinate predictions from multiple models
2. **Ensemble Synthesis** - Combine predictions optimally
3. **Uncertainty Estimation** - Quantify prediction confidence
4. **Prediction Context** - Provide relevant context for predictions

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  PREDICTION SYNTHESIZER AGENT                    │
│                     (Standard Pattern)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 MODEL ORCHESTRATOR                       │    │
│  │   [Model 1] [Model 2] [Model 3] ... [Model N]           │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 ENSEMBLE COMBINER                        │    │
│  │   • Weighted Average  • Stacking  • Voting               │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              UNCERTAINTY QUANTIFIER                      │    │
│  │   • Prediction Intervals  • Model Disagreement           │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               CONTEXT ENRICHER                           │    │
│  │   • Feature Importance  • Similar Cases  • Trends        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
prediction_synthesizer/
├── agent.py              # Main PredictionSynthesizerAgent class
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── nodes/
│   ├── model_orchestrator.py  # Parallel model invocation
│   ├── ensemble_combiner.py   # Prediction aggregation
│   ├── uncertainty.py         # Confidence estimation
│   └── context_enricher.py    # Add prediction context
├── ensemble_methods.py   # Ensemble algorithms
└── model_registry.py     # Available model catalog
```

## LangGraph State Definition

```python
# src/agents/prediction_synthesizer/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class ModelPrediction(TypedDict):
    """Individual model prediction"""
    model_id: str
    model_type: str
    prediction: float
    prediction_proba: Optional[List[float]]
    confidence: float
    latency_ms: int
    features_used: List[str]

class EnsemblePrediction(TypedDict):
    """Combined ensemble prediction"""
    point_estimate: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    confidence: float
    ensemble_method: str
    model_agreement: float  # 0-1, how much models agree

class PredictionContext(TypedDict):
    """Context for interpreting prediction"""
    similar_cases: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    historical_accuracy: float
    trend_direction: Literal["increasing", "stable", "decreasing"]

class PredictionSynthesizerState(TypedDict):
    """Complete state for Prediction Synthesizer agent"""
    
    # === INPUT ===
    query: str
    entity_id: str  # HCP ID, territory ID, etc.
    entity_type: str  # "hcp", "territory", "patient"
    prediction_target: str  # What to predict
    features: Dict[str, Any]
    time_horizon: str  # e.g., "30d", "90d"
    
    # === CONFIGURATION ===
    models_to_use: Optional[List[str]]  # Specific models, or None for all
    ensemble_method: Literal["average", "weighted", "stacking", "voting"]
    confidence_level: float  # Default: 0.95
    include_context: bool
    
    # === MODEL OUTPUTS ===
    individual_predictions: Optional[List[ModelPrediction]]
    models_succeeded: int
    models_failed: int
    
    # === ENSEMBLE OUTPUTS ===
    ensemble_prediction: Optional[EnsemblePrediction]
    prediction_summary: Optional[str]
    
    # === CONTEXT OUTPUTS ===
    prediction_context: Optional[PredictionContext]
    
    # === EXECUTION METADATA ===
    orchestration_latency_ms: int
    ensemble_latency_ms: int
    total_latency_ms: int
    timestamp: str
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "predicting", "combining", "enriching", "completed", "failed"]
```

## Node Implementations

### Model Orchestrator Node

```python
# src/agents/prediction_synthesizer/nodes/model_orchestrator.py

import asyncio
import time
from typing import List, Dict, Any

from ..state import PredictionSynthesizerState, ModelPrediction

class ModelOrchestratorNode:
    """
    Orchestrate predictions from multiple models in parallel
    """
    
    def __init__(self, model_registry, model_clients: Dict[str, Any]):
        self.registry = model_registry
        self.clients = model_clients
        self.timeout_per_model = 5  # seconds
    
    async def execute(self, state: PredictionSynthesizerState) -> PredictionSynthesizerState:
        start_time = time.time()
        
        try:
            # Determine which models to use
            models_to_use = state.get("models_to_use")
            if not models_to_use:
                models_to_use = await self.registry.get_models_for_target(
                    target=state["prediction_target"],
                    entity_type=state["entity_type"]
                )
            
            if not models_to_use:
                return {
                    **state,
                    "errors": [{"node": "orchestrator", "error": "No models available for this prediction target"}],
                    "status": "failed"
                }
            
            # Run predictions in parallel
            tasks = [
                self._get_prediction(model_id, state)
                for model_id in models_to_use
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            predictions = []
            succeeded = 0
            failed = 0
            
            for model_id, result in zip(models_to_use, results):
                if isinstance(result, Exception):
                    failed += 1
                    state = {
                        **state,
                        "warnings": state.get("warnings", []) + [f"Model {model_id} failed: {str(result)}"]
                    }
                elif result is not None:
                    predictions.append(result)
                    succeeded += 1
                else:
                    failed += 1
            
            orchestration_time = int((time.time() - start_time) * 1000)
            
            if not predictions:
                return {
                    **state,
                    "errors": [{"node": "orchestrator", "error": "All models failed"}],
                    "status": "failed"
                }
            
            return {
                **state,
                "individual_predictions": predictions,
                "models_succeeded": succeeded,
                "models_failed": failed,
                "orchestration_latency_ms": orchestration_time,
                "status": "combining"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "orchestrator", "error": str(e)}],
                "status": "failed"
            }
    
    async def _get_prediction(
        self, 
        model_id: str, 
        state: PredictionSynthesizerState
    ) -> ModelPrediction:
        """Get prediction from a single model"""
        
        start = time.time()
        
        client = self.clients.get(model_id)
        if not client:
            raise ValueError(f"No client for model {model_id}")
        
        try:
            result = await asyncio.wait_for(
                client.predict(
                    entity_id=state["entity_id"],
                    features=state["features"],
                    time_horizon=state["time_horizon"]
                ),
                timeout=self.timeout_per_model
            )
            
            latency = int((time.time() - start) * 1000)
            
            return ModelPrediction(
                model_id=model_id,
                model_type=result.get("model_type", "unknown"),
                prediction=result["prediction"],
                prediction_proba=result.get("proba"),
                confidence=result.get("confidence", 0.5),
                latency_ms=latency,
                features_used=result.get("features_used", [])
            )
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Model {model_id} timed out")
```

### Ensemble Combiner Node

```python
# src/agents/prediction_synthesizer/nodes/ensemble_combiner.py

import time
from typing import List
import numpy as np
from scipy import stats

from ..state import PredictionSynthesizerState, EnsemblePrediction, ModelPrediction

class EnsembleCombinerNode:
    """
    Combine individual model predictions into ensemble
    """
    
    async def execute(self, state: PredictionSynthesizerState) -> PredictionSynthesizerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            predictions = state["individual_predictions"]
            method = state.get("ensemble_method", "weighted")
            confidence_level = state.get("confidence_level", 0.95)
            
            # Extract prediction values
            pred_values = np.array([p["prediction"] for p in predictions])
            confidences = np.array([p["confidence"] for p in predictions])
            
            # Combine based on method
            if method == "average":
                point_estimate = np.mean(pred_values)
            elif method == "weighted":
                # Weight by confidence
                weights = confidences / np.sum(confidences)
                point_estimate = np.average(pred_values, weights=weights)
            elif method == "voting":
                # For classification - majority vote
                point_estimate = stats.mode(pred_values, keepdims=False).mode
            else:
                point_estimate = np.mean(pred_values)
            
            # Calculate prediction interval
            std = np.std(pred_values)
            z = stats.norm.ppf((1 + confidence_level) / 2)
            interval_lower = point_estimate - z * std
            interval_upper = point_estimate + z * std
            
            # Calculate model agreement (1 - normalized std)
            if np.mean(pred_values) != 0:
                cv = std / abs(np.mean(pred_values))
                agreement = max(0, 1 - cv)
            else:
                agreement = 1.0 if std == 0 else 0.5
            
            # Overall confidence
            ensemble_confidence = np.mean(confidences) * agreement
            
            ensemble_pred = EnsemblePrediction(
                point_estimate=float(point_estimate),
                prediction_interval_lower=float(interval_lower),
                prediction_interval_upper=float(interval_upper),
                confidence=float(ensemble_confidence),
                ensemble_method=method,
                model_agreement=float(agreement)
            )
            
            # Generate summary
            summary = self._generate_summary(ensemble_pred, predictions)
            
            ensemble_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "ensemble_prediction": ensemble_pred,
                "prediction_summary": summary,
                "ensemble_latency_ms": ensemble_time,
                "status": "enriching" if state.get("include_context") else "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "ensemble", "error": str(e)}],
                "status": "failed"
            }
    
    def _generate_summary(
        self, 
        ensemble: EnsemblePrediction, 
        individual: List[ModelPrediction]
    ) -> str:
        """Generate prediction summary"""
        
        pred = ensemble["point_estimate"]
        lower = ensemble["prediction_interval_lower"]
        upper = ensemble["prediction_interval_upper"]
        agreement = ensemble["model_agreement"]
        
        confidence_desc = "high" if ensemble["confidence"] > 0.7 else "moderate" if ensemble["confidence"] > 0.4 else "low"
        agreement_desc = "strong" if agreement > 0.8 else "moderate" if agreement > 0.5 else "weak"
        
        summary = f"Prediction: {pred:.2f} (95% CI: [{lower:.2f}, {upper:.2f}]). "
        summary += f"Confidence: {confidence_desc}. "
        summary += f"Model agreement: {agreement_desc} across {len(individual)} models."
        
        return summary
```

### Context Enricher Node

```python
# src/agents/prediction_synthesizer/nodes/context_enricher.py

import asyncio
import time
from typing import Dict, Any

from ..state import PredictionSynthesizerState, PredictionContext

class ContextEnricherNode:
    """
    Enrich prediction with context for interpretation
    """
    
    def __init__(self, context_store, feature_store):
        self.context_store = context_store
        self.feature_store = feature_store
    
    async def execute(self, state: PredictionSynthesizerState) -> PredictionSynthesizerState:
        start_time = time.time()
        
        if state.get("status") in ["failed", "completed"]:
            return state
        
        if not state.get("include_context", False):
            return {**state, "status": "completed"}
        
        try:
            # Fetch context elements in parallel
            similar_task = self._get_similar_cases(state)
            importance_task = self._get_feature_importance(state)
            accuracy_task = self._get_historical_accuracy(state)
            trend_task = self._get_trend(state)
            
            similar, importance, accuracy, trend = await asyncio.gather(
                similar_task, importance_task, accuracy_task, trend_task,
                return_exceptions=True
            )
            
            context = PredictionContext(
                similar_cases=similar if not isinstance(similar, Exception) else [],
                feature_importance=importance if not isinstance(importance, Exception) else {},
                historical_accuracy=accuracy if not isinstance(accuracy, Exception) else 0.0,
                trend_direction=trend if not isinstance(trend, Exception) else "stable"
            )
            
            total_time = (
                state.get("orchestration_latency_ms", 0) +
                state.get("ensemble_latency_ms", 0) +
                int((time.time() - start_time) * 1000)
            )
            
            return {
                **state,
                "prediction_context": context,
                "total_latency_ms": total_time,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "warnings": state.get("warnings", []) + [f"Context enrichment failed: {str(e)}"],
                "status": "completed"  # Non-fatal
            }
    
    async def _get_similar_cases(self, state: PredictionSynthesizerState) -> list:
        """Find similar historical cases"""
        return await self.context_store.find_similar(
            entity_type=state["entity_type"],
            features=state["features"],
            limit=5
        )
    
    async def _get_feature_importance(self, state: PredictionSynthesizerState) -> Dict[str, float]:
        """Get feature importance for prediction"""
        # Aggregate importance across models
        importances = {}
        for pred in state.get("individual_predictions", []):
            model_importance = await self.feature_store.get_importance(pred["model_id"])
            for feature, importance in model_importance.items():
                if feature in importances:
                    importances[feature] = (importances[feature] + importance) / 2
                else:
                    importances[feature] = importance
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])
    
    async def _get_historical_accuracy(self, state: PredictionSynthesizerState) -> float:
        """Get historical accuracy for this prediction type"""
        return await self.context_store.get_accuracy(
            prediction_target=state["prediction_target"],
            entity_type=state["entity_type"]
        )
    
    async def _get_trend(self, state: PredictionSynthesizerState) -> str:
        """Determine trend direction"""
        history = await self.context_store.get_prediction_history(
            entity_id=state["entity_id"],
            prediction_target=state["prediction_target"],
            limit=10
        )
        
        if not history or len(history) < 3:
            return "stable"
        
        values = [h["prediction"] for h in history]
        slope = (values[-1] - values[0]) / len(values)
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
```

## Graph Assembly

```python
# src/agents/prediction_synthesizer/graph.py

from langgraph.graph import StateGraph, END

from .state import PredictionSynthesizerState
from .nodes.model_orchestrator import ModelOrchestratorNode
from .nodes.ensemble_combiner import EnsembleCombinerNode
from .nodes.context_enricher import ContextEnricherNode

def build_prediction_synthesizer_graph(
    model_registry,
    model_clients,
    context_store,
    feature_store
):
    """
    Build the Prediction Synthesizer agent graph
    
    Architecture:
        [orchestrate] → [combine] → [enrich] → END
    """
    
    # Initialize nodes
    orchestrator = ModelOrchestratorNode(model_registry, model_clients)
    combiner = EnsembleCombinerNode()
    enricher = ContextEnricherNode(context_store, feature_store)
    
    # Build graph
    workflow = StateGraph(PredictionSynthesizerState)
    
    # Add nodes
    workflow.add_node("orchestrate", orchestrator.execute)
    workflow.add_node("combine", combiner.execute)
    workflow.add_node("enrich", enricher.execute)
    workflow.add_node("error_handler", error_handler_node)
    
    # Flow
    workflow.set_entry_point("orchestrate")
    
    workflow.add_conditional_edges(
        "orchestrate",
        lambda s: "error" if s.get("status") == "failed" else "combine",
        {"combine": "combine", "error": "error_handler"}
    )
    
    workflow.add_conditional_edges(
        "combine",
        lambda s: "error" if s.get("status") == "failed" else "enrich",
        {"enrich": "enrich", "error": "error_handler"}
    )
    
    workflow.add_edge("enrich", END)
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()

async def error_handler_node(state: PredictionSynthesizerState) -> PredictionSynthesizerState:
    return {
        **state,
        "prediction_summary": "Prediction could not be generated due to errors.",
        "status": "failed"
    }
```

## Integration Contracts

### Input Contract
```python
class PredictionSynthesizerInput(BaseModel):
    query: str
    entity_id: str
    entity_type: str
    prediction_target: str
    features: Dict[str, Any]
    time_horizon: str = "30d"
    ensemble_method: Literal["average", "weighted", "stacking", "voting"] = "weighted"
    include_context: bool = True
```

### Output Contract
```python
class PredictionSynthesizerOutput(BaseModel):
    ensemble_prediction: EnsemblePrediction
    individual_predictions: List[ModelPrediction]
    prediction_context: Optional[PredictionContext]
    prediction_summary: str
    models_succeeded: int
    models_failed: int
    total_latency_ms: int
```

## Handoff Format

```yaml
prediction_synthesizer_handoff:
  agent: prediction_synthesizer
  analysis_type: prediction
  key_findings:
    - prediction: <point estimate>
    - confidence_interval: [<lower>, <upper>]
    - confidence: <0-1>
    - model_agreement: <0-1>
  models:
    succeeded: <count>
    failed: <count>
  context:
    trend: <increasing|stable|decreasing>
    historical_accuracy: <0-1>
  recommendations:
    - <recommendation 1>
  requires_further_analysis: <bool>
  suggested_next_agent: <explainer>
```

---

## Cognitive RAG DSPy Integration

### Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               PREDICTION SYNTHESIZER ↔ COGNITIVE RAG DSPY                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐ │
│  │   PREDICTION    │    │            COGNITIVE RAG DSPY                   │ │
│  │   SYNTHESIZER   │◄───│                                                 │ │
│  │                 │    │  ┌─────────────────────────────────────────┐   │ │
│  │ ┌─────────────┐ │    │  │  EvidenceSynthesisSignature             │   │ │
│  │ │    MODEL    │ │    │  │  ├─ entity_history: prior predictions   │   │ │
│  │ │ ORCHESTRATOR│◄├────│  │  ├─ model_performance: accuracy trends  │   │ │
│  │ └─────────────┘ │    │  │  ├─ similar_entities: comparable cases  │   │ │
│  │       ↓         │    │  │  └─ domain_context: business insights   │   │ │
│  │ ┌─────────────┐ │    │  └─────────────────────────────────────────┘   │ │
│  │ │  ENSEMBLE   │ │    │                                                 │ │
│  │ │  COMBINER   │◄├────│  ┌─────────────────────────────────────────┐   │ │
│  │ └─────────────┘ │    │  │  EnsembleOptimizationSignature          │   │ │
│  │       ↓         │    │  │  ├─ historical_weights: past optimal    │   │ │
│  │ ┌─────────────┐ │    │  │  ├─ model_correlations: redundancy info │   │ │
│  │ │  CONTEXT    │ │    │  │  └─ regime_context: current conditions  │   │ │
│  │ │  ENRICHER   │ │    │  └─────────────────────────────────────────┘   │ │
│  │ └─────────────┘ │    │                                                 │ │
│  │       ↓         │    └─────────────────────────────────────────────────┘ │
│  │       │         │                                                        │
│  │       ▼         │                                                        │
│  │ ┌─────────────┐ │                                                        │
│  │ │  TRAINING   │─┼───────────────────────────────────────────────────────►│
│  │ │  SIGNAL     │ │    MIPROv2 Optimizer (prediction accuracy metrics)    │
│  │ └─────────────┘ │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  MEMORY CONTRIBUTION: predictions (SEMANTIC + EPISODIC)                 ││
│  │  ├─ Stores: entity_id, prediction, actual (when known), model_weights   ││
│  │  └─ Embedding: entity_features + prediction_context + model_rationale   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DSPy Signature Consumption

The Prediction Synthesizer consumes cognitive context for model orchestration and ensemble optimization:

```python
# Cognitive context fields consumed by Prediction Synthesizer

class PredictionCognitiveContext(TypedDict):
    """Cognitive context from CognitiveRAG for prediction synthesis."""
    synthesized_summary: str  # Overall context synthesis
    entity_history: List[Dict[str, Any]]  # Prior predictions for this entity
    model_performance: Dict[str, Dict[str, float]]  # Model -> metrics history
    similar_entities: List[Dict[str, Any]]  # Comparable entities for calibration
    domain_context: str  # Business context for interpretation
    evidence_confidence: float  # Confidence in retrieved context


class EnsembleCognitiveContext(TypedDict):
    """Cognitive context for ensemble optimization."""
    historical_weights: Dict[str, float]  # Past optimal model weights
    model_correlations: Dict[str, Dict[str, float]]  # Model correlation matrix
    regime_context: str  # Current market/business regime
    weight_stability: float  # How stable weights have been historically
```

### Model Orchestrator with Cognitive Integration

```python
# src/agents/prediction_synthesizer/nodes/model_orchestrator.py

from typing import Optional
from ..state import PredictionSynthesizerState, ModelPrediction


class PredictionCognitiveContext(TypedDict):
    """Cognitive context from CognitiveRAG for prediction synthesis."""
    synthesized_summary: str
    entity_history: List[Dict[str, Any]]
    model_performance: Dict[str, Dict[str, float]]
    similar_entities: List[Dict[str, Any]]
    domain_context: str
    evidence_confidence: float


class ModelOrchestratorNode:
    """Model orchestration with cognitive-informed model selection."""

    async def execute(
        self,
        state: PredictionSynthesizerState,
        cognitive_context: Optional[PredictionCognitiveContext] = None
    ) -> PredictionSynthesizerState:
        """Execute model orchestration with cognitive enrichment."""
        start_time = time.time()

        try:
            # Determine which models to use
            models_to_use = state.get("models_to_use")

            if not models_to_use:
                models_to_use = await self.registry.get_models_for_target(
                    target=state["prediction_target"],
                    entity_type=state["entity_type"]
                )

            # Cognitive-informed model selection/prioritization
            if cognitive_context and cognitive_context.get("model_performance"):
                models_to_use = self._prioritize_models_by_performance(
                    models_to_use,
                    cognitive_context["model_performance"],
                    state["prediction_target"]
                )

            # Run predictions in parallel
            tasks = [
                self._get_prediction(model_id, state, cognitive_context)
                for model_id in models_to_use
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            predictions = []
            succeeded = 0
            failed = 0

            for model_id, result in zip(models_to_use, results):
                if isinstance(result, Exception):
                    failed += 1
                    state = {
                        **state,
                        "warnings": state.get("warnings", []) + [f"Model {model_id} failed: {str(result)}"]
                    }
                elif result is not None:
                    predictions.append(result)
                    succeeded += 1
                else:
                    failed += 1

            orchestration_time = int((time.time() - start_time) * 1000)

            if not predictions:
                return {
                    **state,
                    "errors": [{"node": "orchestrator", "error": "All models failed"}],
                    "status": "failed"
                }

            return {
                **state,
                "individual_predictions": predictions,
                "models_succeeded": succeeded,
                "models_failed": failed,
                "orchestration_latency_ms": orchestration_time,
                "cognitive_model_selection_used": cognitive_context is not None,
                "status": "combining"
            }

        except Exception as e:
            return {
                **state,
                "errors": [{"node": "orchestrator", "error": str(e)}],
                "status": "failed"
            }

    def _prioritize_models_by_performance(
        self,
        models: List[str],
        performance: Dict[str, Dict[str, float]],
        target: str
    ) -> List[str]:
        """Prioritize models based on historical performance for this target."""

        def get_score(model_id: str) -> float:
            metrics = performance.get(model_id, {})
            # Weighted score of accuracy and recency
            accuracy = metrics.get(f"{target}_accuracy", 0.5)
            recency_weight = metrics.get("recency_weight", 0.8)
            return accuracy * recency_weight

        # Sort by score, but keep all models (parallel execution)
        return sorted(models, key=get_score, reverse=True)

    async def _get_prediction(
        self,
        model_id: str,
        state: PredictionSynthesizerState,
        cognitive_context: Optional[PredictionCognitiveContext] = None
    ) -> ModelPrediction:
        """Get prediction with optional cognitive enrichment of features."""

        features = state["features"].copy()

        # Enrich features with entity history if available
        if cognitive_context and cognitive_context.get("entity_history"):
            history = cognitive_context["entity_history"]
            if history:
                # Add lagged prediction features
                features["prior_prediction_1"] = history[-1].get("prediction", None)
                if len(history) >= 2:
                    features["prior_prediction_2"] = history[-2].get("prediction", None)
                features["prior_prediction_trend"] = self._compute_trend(history)

        # Get prediction from model
        start = time.time()
        client = self.clients.get(model_id)

        result = await asyncio.wait_for(
            client.predict(
                entity_id=state["entity_id"],
                features=features,
                time_horizon=state["time_horizon"]
            ),
            timeout=self.timeout_per_model
        )

        latency = int((time.time() - start) * 1000)

        return ModelPrediction(
            model_id=model_id,
            model_type=result.get("model_type", "unknown"),
            prediction=result["prediction"],
            prediction_proba=result.get("proba"),
            confidence=result.get("confidence", 0.5),
            latency_ms=latency,
            features_used=list(features.keys())
        )

    def _compute_trend(self, history: List[Dict[str, Any]]) -> str:
        """Compute trend from prediction history."""
        if len(history) < 2:
            return "stable"

        values = [h.get("prediction", 0) for h in history[-5:]]
        if len(values) < 2:
            return "stable"

        slope = (values[-1] - values[0]) / len(values)
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        return "stable"
```

### Ensemble Combiner with Cognitive Optimization

```python
# src/agents/prediction_synthesizer/nodes/ensemble_combiner.py

from typing import Optional
import numpy as np
from scipy import stats


class EnsembleCognitiveContext(TypedDict):
    """Cognitive context for ensemble optimization."""
    historical_weights: Dict[str, float]
    model_correlations: Dict[str, Dict[str, float]]
    regime_context: str
    weight_stability: float


class EnsembleCombinerNode:
    """Ensemble combiner with cognitive-optimized weighting."""

    async def execute(
        self,
        state: PredictionSynthesizerState,
        cognitive_context: Optional[EnsembleCognitiveContext] = None
    ) -> PredictionSynthesizerState:
        """Combine predictions with cognitive-informed weights."""
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        try:
            predictions = state["individual_predictions"]
            method = state.get("ensemble_method", "weighted")
            confidence_level = state.get("confidence_level", 0.95)

            # Extract prediction values
            pred_values = np.array([p["prediction"] for p in predictions])
            confidences = np.array([p["confidence"] for p in predictions])
            model_ids = [p["model_id"] for p in predictions]

            # Determine weights
            if method == "weighted" and cognitive_context:
                weights = self._compute_cognitive_weights(
                    model_ids,
                    confidences,
                    cognitive_context
                )
            elif method == "weighted":
                weights = confidences / np.sum(confidences)
            else:
                weights = np.ones(len(predictions)) / len(predictions)

            # Combine predictions
            if method == "voting":
                point_estimate = stats.mode(pred_values, keepdims=False).mode
            else:
                point_estimate = np.average(pred_values, weights=weights)

            # Calculate prediction interval (accounting for model correlations)
            if cognitive_context and cognitive_context.get("model_correlations"):
                std = self._compute_correlated_std(
                    pred_values,
                    weights,
                    model_ids,
                    cognitive_context["model_correlations"]
                )
            else:
                std = np.sqrt(np.average((pred_values - point_estimate) ** 2, weights=weights))

            z = stats.norm.ppf((1 + confidence_level) / 2)
            interval_lower = point_estimate - z * std
            interval_upper = point_estimate + z * std

            # Calculate model agreement
            if np.mean(pred_values) != 0:
                cv = std / abs(np.mean(pred_values))
                agreement = max(0, 1 - cv)
            else:
                agreement = 1.0 if std == 0 else 0.5

            # Overall confidence
            ensemble_confidence = np.average(confidences, weights=weights) * agreement

            ensemble_pred = EnsemblePrediction(
                point_estimate=float(point_estimate),
                prediction_interval_lower=float(interval_lower),
                prediction_interval_upper=float(interval_upper),
                confidence=float(ensemble_confidence),
                ensemble_method=method,
                model_agreement=float(agreement)
            )

            # Generate summary with cognitive context
            summary = self._generate_summary(ensemble_pred, predictions, cognitive_context)

            ensemble_time = int((time.time() - start_time) * 1000)

            return {
                **state,
                "ensemble_prediction": ensemble_pred,
                "prediction_summary": summary,
                "ensemble_latency_ms": ensemble_time,
                "ensemble_weights_used": dict(zip(model_ids, weights.tolist())),
                "cognitive_ensemble_optimization": cognitive_context is not None,
                "status": "enriching" if state.get("include_context") else "completed"
            }

        except Exception as e:
            return {
                **state,
                "errors": [{"node": "ensemble", "error": str(e)}],
                "status": "failed"
            }

    def _compute_cognitive_weights(
        self,
        model_ids: List[str],
        confidences: np.ndarray,
        context: EnsembleCognitiveContext
    ) -> np.ndarray:
        """Compute weights using historical performance and correlations."""

        historical = context.get("historical_weights", {})
        stability = context.get("weight_stability", 0.5)

        # Start with confidence-based weights
        weights = confidences.copy()

        # Blend with historical weights based on stability
        for i, model_id in enumerate(model_ids):
            if model_id in historical:
                hist_weight = historical[model_id]
                # Higher stability = more reliance on historical weights
                weights[i] = (stability * hist_weight + (1 - stability) * weights[i])

        # Penalize highly correlated models (reduce redundancy)
        correlations = context.get("model_correlations", {})
        if correlations:
            for i, model_i in enumerate(model_ids):
                for j, model_j in enumerate(model_ids):
                    if i != j and model_i in correlations:
                        corr = correlations.get(model_i, {}).get(model_j, 0)
                        if corr > 0.8:  # Highly correlated
                            # Reduce weight of less confident model
                            if weights[i] < weights[j]:
                                weights[i] *= (1 - corr * 0.3)

        # Normalize
        weights = weights / np.sum(weights)
        return weights

    def _compute_correlated_std(
        self,
        predictions: np.ndarray,
        weights: np.ndarray,
        model_ids: List[str],
        correlations: Dict[str, Dict[str, float]]
    ) -> float:
        """Compute std accounting for model correlations."""

        n = len(predictions)
        if n == 1:
            return 0.0

        # Build correlation matrix
        corr_matrix = np.eye(n)
        for i, model_i in enumerate(model_ids):
            for j, model_j in enumerate(model_ids):
                if i != j and model_i in correlations:
                    corr_matrix[i, j] = correlations.get(model_i, {}).get(model_j, 0)

        # Compute weighted variance with correlations
        mean_pred = np.average(predictions, weights=weights)
        deviations = predictions - mean_pred

        # σ² = Σᵢ Σⱼ wᵢ wⱼ ρᵢⱼ σᵢ σⱼ
        variance = 0.0
        for i in range(n):
            for j in range(n):
                variance += weights[i] * weights[j] * corr_matrix[i, j] * deviations[i] * deviations[j]

        return np.sqrt(max(variance, 0))

    def _generate_summary(
        self,
        ensemble: EnsemblePrediction,
        individual: List[ModelPrediction],
        context: Optional[EnsembleCognitiveContext] = None
    ) -> str:
        """Generate prediction summary with cognitive context."""

        pred = ensemble["point_estimate"]
        lower = ensemble["prediction_interval_lower"]
        upper = ensemble["prediction_interval_upper"]
        agreement = ensemble["model_agreement"]

        confidence_desc = "high" if ensemble["confidence"] > 0.7 else "moderate" if ensemble["confidence"] > 0.4 else "low"
        agreement_desc = "strong" if agreement > 0.8 else "moderate" if agreement > 0.5 else "weak"

        summary = f"Prediction: {pred:.2f} (95% CI: [{lower:.2f}, {upper:.2f}]). "
        summary += f"Confidence: {confidence_desc}. "
        summary += f"Model agreement: {agreement_desc} across {len(individual)} models."

        # Add regime context if available
        if context and context.get("regime_context"):
            summary += f" Current context: {context['regime_context']}."

        return summary
```

### Training Signal for MIPROv2

```python
# src/agents/prediction_synthesizer/training_signal.py

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PredictionSynthesizerTrainingSignal:
    """Training signal for prediction synthesis quality."""

    # Prediction quality (when actual is known)
    prediction_made: float
    actual_outcome: Optional[float]
    prediction_interval_width: float

    # Ensemble quality
    model_agreement: float
    models_succeeded: int
    models_failed: int

    # Cognitive integration
    cognitive_context_used: bool
    cognitive_model_selection: bool
    cognitive_weight_optimization: bool
    entity_history_used: bool

    # Performance
    total_latency_ms: int
    max_latency_threshold_ms: int = 15000

    def compute_reward(self) -> float:
        """Compute reward for MIPROv2 optimization."""

        base_reward = 0.3

        # Prediction accuracy (when actual is known)
        if self.actual_outcome is not None:
            error = abs(self.prediction_made - self.actual_outcome)
            relative_error = error / max(abs(self.actual_outcome), 1e-6)

            # High accuracy = high reward
            if relative_error < 0.1:
                base_reward += 0.3
            elif relative_error < 0.2:
                base_reward += 0.2
            elif relative_error < 0.5:
                base_reward += 0.1
            else:
                base_reward -= 0.1  # Penalize large errors

        # Reward model agreement (reduces uncertainty)
        base_reward += 0.1 * self.model_agreement

        # Reward narrow prediction intervals (precision)
        if self.prediction_interval_width < 0.5:
            base_reward += 0.1
        elif self.prediction_interval_width > 2.0:
            base_reward -= 0.05

        # Penalize model failures
        if self.models_failed > 0:
            failure_ratio = self.models_failed / (self.models_succeeded + self.models_failed)
            base_reward -= 0.1 * failure_ratio

        # Reward cognitive integration
        if self.cognitive_context_used:
            base_reward += 0.05
            if self.cognitive_model_selection:
                base_reward += 0.05
            if self.cognitive_weight_optimization:
                base_reward += 0.05
            if self.entity_history_used:
                base_reward += 0.05

        # Penalize slow predictions
        if self.total_latency_ms > self.max_latency_threshold_ms:
            latency_penalty = (self.total_latency_ms - self.max_latency_threshold_ms) / 10000
            base_reward -= min(latency_penalty, 0.15)

        return max(0.0, min(base_reward, 1.0))
```

### Memory Contribution

```python
# Memory contribution for predictions

async def contribute_to_memory(
    state: PredictionSynthesizerState,
    memory_backend: MemoryBackend
) -> None:
    """Store prediction in organizational memory."""

    prediction_record = {
        "prediction_id": f"pred_{state['entity_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "entity_id": state["entity_id"],
        "entity_type": state["entity_type"],
        "prediction_target": state["prediction_target"],
        "prediction": state["ensemble_prediction"]["point_estimate"],
        "prediction_interval": [
            state["ensemble_prediction"]["prediction_interval_lower"],
            state["ensemble_prediction"]["prediction_interval_upper"]
        ],
        "confidence": state["ensemble_prediction"]["confidence"],
        "model_agreement": state["ensemble_prediction"]["model_agreement"],
        "ensemble_weights": state.get("ensemble_weights_used", {}),
        "features_snapshot": state["features"],
        "time_horizon": state["time_horizon"],
        "models_used": [p["model_id"] for p in state["individual_predictions"]],
        "timestamp": state.get("timestamp", ""),
        "actual_outcome": None,  # To be updated when known
    }

    await memory_backend.store(
        memory_type="SEMANTIC",  # For similarity search
        content=prediction_record,
        metadata={
            "agent": "prediction_synthesizer",
            "index": "predictions",
            "embedding_fields": [
                "entity_type",
                "prediction_target",
                "features_snapshot"
            ],
        }
    )

    # Also store in EPISODIC for temporal tracking
    await memory_backend.store(
        memory_type="EPISODIC",
        content=prediction_record,
        metadata={
            "agent": "prediction_synthesizer",
            "index": "prediction_history",
            "entity_id": state["entity_id"],
            "temporal_weight": 0.9
        }
    )
```

### Cognitive Input TypedDict

```python
# src/agents/prediction_synthesizer/cognitive_input.py

from typing import TypedDict, List, Dict, Any, Optional, Literal


class PredictionSynthesizerCognitiveInput(TypedDict):
    """Full cognitive input for Prediction Synthesizer agent."""

    # Standard input
    query: str
    entity_id: str
    entity_type: str
    prediction_target: str
    features: Dict[str, Any]
    time_horizon: str
    ensemble_method: Literal["average", "weighted", "stacking", "voting"]
    include_context: bool

    # Cognitive enrichment (optional)
    prediction_cognitive_context: Optional[PredictionCognitiveContext]
    ensemble_cognitive_context: Optional[EnsembleCognitiveContext]
```

### Configuration

```yaml
# config/agents/prediction_synthesizer.yaml

prediction_synthesizer:
  tier: 4
  type: standard_computational

  cognitive_rag:
    enabled: true
    context_sources:
      - entity_predictions  # Historical predictions for entity
      - model_performance   # Model accuracy tracking
      - similar_entities    # Comparable entities
      - ensemble_weights    # Historical optimal weights
    min_confidence_threshold: 0.3
    entity_history_limit: 10

  dspy:
    optimizer: MIPROv2
    training_signals:
      - prediction_accuracy
      - ensemble_agreement
      - interval_calibration
    optimization_target: prediction_rmse

  memory:
    contribution_enabled: true
    indices:
      - name: predictions
        memory_type: SEMANTIC
      - name: prediction_history
        memory_type: EPISODIC
    embedding_fields:
      - entity_type
      - prediction_target
      - features_snapshot
```

### Testing Requirements

```python
# tests/unit/test_agents/test_prediction_synthesizer/test_cognitive_integration.py

@pytest.mark.asyncio
async def test_prediction_with_cognitive_context():
    """Test prediction synthesis with cognitive enrichment."""
    agent = PredictionSynthesizerAgent()

    prediction_context = PredictionCognitiveContext(
        synthesized_summary="Entity has shown stable growth...",
        entity_history=[
            {"prediction": 100, "actual": 98, "timestamp": "2024-01-01"},
            {"prediction": 105, "actual": 107, "timestamp": "2024-02-01"}
        ],
        model_performance={
            "model_a": {"accuracy": 0.92, "recency_weight": 0.9},
            "model_b": {"accuracy": 0.85, "recency_weight": 0.8}
        },
        similar_entities=[
            {"entity_id": "E123", "similarity": 0.95}
        ],
        domain_context="Q4 typically shows 10% higher volume",
        evidence_confidence=0.82
    )

    ensemble_context = EnsembleCognitiveContext(
        historical_weights={"model_a": 0.6, "model_b": 0.4},
        model_correlations={"model_a": {"model_b": 0.3}},
        regime_context="growth",
        weight_stability=0.75
    )

    result = await agent.predict(
        entity_id="HCP_001",
        prediction_target="rx_volume",
        features={"tenure_months": 24, "call_frequency": 2.5},
        prediction_cognitive_context=prediction_context,
        ensemble_cognitive_context=ensemble_context
    )

    assert result.cognitive_model_selection_used is True
    assert result.cognitive_ensemble_optimization is True


@pytest.mark.asyncio
async def test_model_prioritization_by_performance():
    """Test that models are prioritized by cognitive performance data."""
    node = ModelOrchestratorNode(model_registry, model_clients)

    cognitive_context = PredictionCognitiveContext(
        model_performance={
            "model_a": {"rx_volume_accuracy": 0.95, "recency_weight": 0.9},
            "model_b": {"rx_volume_accuracy": 0.75, "recency_weight": 0.8},
            "model_c": {"rx_volume_accuracy": 0.85, "recency_weight": 0.95}
        }
    )

    prioritized = node._prioritize_models_by_performance(
        ["model_a", "model_b", "model_c"],
        cognitive_context["model_performance"],
        "rx_volume"
    )

    # model_a should be first (highest score)
    assert prioritized[0] == "model_a"


def test_training_signal_rewards_accuracy():
    """Test training signal rewards accurate predictions."""
    accurate_signal = PredictionSynthesizerTrainingSignal(
        prediction_made=100.0,
        actual_outcome=98.0,  # 2% error
        prediction_interval_width=10.0,
        model_agreement=0.9,
        models_succeeded=3,
        models_failed=0,
        cognitive_context_used=True,
        cognitive_model_selection=True,
        cognitive_weight_optimization=True,
        entity_history_used=True,
        total_latency_ms=5000
    )

    inaccurate_signal = PredictionSynthesizerTrainingSignal(
        prediction_made=100.0,
        actual_outcome=150.0,  # 50% error
        **{k: v for k, v in accurate_signal.__dict__.items() if k not in ["prediction_made", "actual_outcome"]}
    )

    assert accurate_signal.compute_reward() > inaccurate_signal.compute_reward()
```
