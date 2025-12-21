"""
E2I Causal Analytics - Orchestrator Tools
==========================================

Tools available to the orchestrator agent for handling various intents.

Available Tools:
----------------
- explain_tool: Real-Time SHAP explanation integration (v4.1)
  - ExplainAPITool: HTTP client for /explain API
  - ExplainIntentHandler: Intent routing and response formatting
  - classify_explain_intent: Intent classification
  - extract_explanation_entities: Entity extraction

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

from .explain_tool import (
    ExplainAPITool,
    ExplainIntentHandler,
    ExplainSubIntent,
    ExplanationEntities,
    classify_explain_intent,
    extract_explanation_entities,
)

__all__ = [
    "ExplainAPITool",
    "ExplainIntentHandler",
    "classify_explain_intent",
    "extract_explanation_entities",
    "ExplanationEntities",
    "ExplainSubIntent",
]

__version__ = "4.1.0"
