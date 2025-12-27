"""GEPA MLOps Integrations.

This module provides integrations with E2I's MLOps stack:
- MLflow: Experiment tracking for optimization runs
- Opik: LLM/Agent observability and tracing
- RAGAS: RAG evaluation feedback for cognitive RAG agents

These integrations enable:
1. Tracking optimization experiments in MLflow
2. Tracing GEPA evolution in Opik
3. Using RAGAS metrics as GEPA feedback signals
"""

from src.optimization.gepa.integration.mlflow_integration import (
    GEPAMLflowCallback,
    log_optimization_run,
)
from src.optimization.gepa.integration.opik_integration import (
    GEPAOpikTracer,
    trace_optimization,
)
from src.optimization.gepa.integration.ragas_feedback import (
    RAGASFeedbackProvider,
    create_ragas_metric,
)

__all__ = [
    # MLflow
    "GEPAMLflowCallback",
    "log_optimization_run",
    # Opik
    "GEPAOpikTracer",
    "trace_optimization",
    # RAGAS
    "RAGASFeedbackProvider",
    "create_ragas_metric",
]
