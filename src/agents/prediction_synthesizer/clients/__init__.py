"""Prediction Synthesizer Model Clients.

This package provides model client implementations for the prediction_synthesizer agent.
Supports both HTTP-based (BentoML endpoints) and in-process model clients.
"""

from src.agents.prediction_synthesizer.clients.factory import (
    ModelClientFactory,
    get_model_client,
)
from src.agents.prediction_synthesizer.clients.http_model_client import (
    HTTPModelClient,
    HTTPModelClientConfig,
)

__all__ = [
    "HTTPModelClient",
    "HTTPModelClientConfig",
    "ModelClientFactory",
    "get_model_client",
]
