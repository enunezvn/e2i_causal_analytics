"""
E2I Prediction Synthesizer Agent
Tier 4 - ML Prediction Aggregation and Ensemble

Provides:
- Multi-model prediction orchestration
- Ensemble methods (average, weighted, voting)
- Uncertainty quantification
- Prediction context enrichment
- HTTP-based model clients for BentoML endpoints
"""

from .agent import (
    PredictionSynthesizerAgent,
    PredictionSynthesizerInput,
    PredictionSynthesizerOutput,
    synthesize_predictions,
)
from .clients import (
    HTTPModelClient,
    HTTPModelClientConfig,
    ModelClientFactory,
    get_model_client,
)
from .graph import build_prediction_synthesizer_graph
from .state import (
    EnsemblePrediction,
    ModelPrediction,
    PredictionContext,
    PredictionSynthesizerState,
)

__all__ = [
    # Agent
    "PredictionSynthesizerAgent",
    "PredictionSynthesizerInput",
    "PredictionSynthesizerOutput",
    "synthesize_predictions",
    # State
    "PredictionSynthesizerState",
    "ModelPrediction",
    "EnsemblePrediction",
    "PredictionContext",
    # Graph
    "build_prediction_synthesizer_graph",
    # Model Clients
    "HTTPModelClient",
    "HTTPModelClientConfig",
    "ModelClientFactory",
    "get_model_client",
]
