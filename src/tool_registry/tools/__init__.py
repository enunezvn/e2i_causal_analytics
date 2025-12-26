"""E2I Tool Registry - Composable Tools.

This package contains tools that agents can use for their workflows.
"""

from src.tool_registry.tools.model_inference import (
    ModelInferenceTool,
    ModelInferenceInput,
    ModelInferenceOutput,
    get_model_inference_tool,
    model_inference,
)

__all__ = [
    "ModelInferenceTool",
    "ModelInferenceInput",
    "ModelInferenceOutput",
    "get_model_inference_tool",
    "model_inference",
]
