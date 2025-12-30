"""E2I Tool Registry - Composable Tools.

This package contains tools that agents can use for their workflows.

Available Tools:
- model_inference: Real-time model predictions via BentoML
- discover_dag: Automatic DAG structure learning (GES, PC, etc.)
- rank_drivers: Causal vs predictive feature importance comparison
"""

from src.tool_registry.tools.causal_discovery import (
    CausalDiscoveryTool,
    DiscoverDagInput,
    DiscoverDagOutput,
    DriverRankerTool,
    FeatureRankingItem,
    RankDriversInput,
    RankDriversOutput,
    discover_dag,
    get_discovery_tool,
    get_ranker_tool,
    rank_drivers,
    register_all_discovery_tools,
)
from src.tool_registry.tools.model_inference import (
    ModelInferenceTool,
    ModelInferenceInput,
    ModelInferenceOutput,
    get_model_inference_tool,
    model_inference,
)

__all__ = [
    # Model Inference
    "ModelInferenceTool",
    "ModelInferenceInput",
    "ModelInferenceOutput",
    "get_model_inference_tool",
    "model_inference",
    # Causal Discovery
    "CausalDiscoveryTool",
    "DiscoverDagInput",
    "DiscoverDagOutput",
    "get_discovery_tool",
    "discover_dag",
    # Driver Ranking
    "DriverRankerTool",
    "RankDriversInput",
    "RankDriversOutput",
    "FeatureRankingItem",
    "get_ranker_tool",
    "rank_drivers",
    # Registration helpers
    "register_all_discovery_tools",
]
