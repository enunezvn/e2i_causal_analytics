"""E2I Tool Registry - Composable Tools.

This package contains tools that agents can use for their workflows.

Available Tools:
- model_inference: Real-time model predictions via BentoML
- discover_dag: Automatic DAG structure learning (GES, PC, etc.)
- rank_drivers: Causal vs predictive feature importance comparison
- detect_structural_drift: Detect drift in causal DAG structure over time
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
from src.tool_registry.tools.structural_drift import (
    EdgeTypeChange,
    StructuralDriftInput,
    StructuralDriftOutput,
    StructuralDriftTool,
    detect_structural_drift,
    get_structural_drift_tool,
    register_structural_drift_tool,
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
    # Structural Drift
    "StructuralDriftTool",
    "StructuralDriftInput",
    "StructuralDriftOutput",
    "EdgeTypeChange",
    "get_structural_drift_tool",
    "detect_structural_drift",
    "register_structural_drift_tool",
    # Registration helpers
    "register_all_discovery_tools",
]
