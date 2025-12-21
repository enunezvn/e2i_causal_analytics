"""
E2I Tool Composer - Data Models
"""

from .composition_models import (
    # Enums
    CompositionPhase,
    ExecutionStatus,
    DependencyType,
    # Phase 1 models
    SubQuestion,
    DecompositionResult,
    # Phase 2 models
    ToolMapping,
    ExecutionStep,
    ExecutionPlan,
    # Phase 3 models
    ToolInput,
    ToolOutput,
    StepResult,
    ExecutionTrace,
    # Phase 4 models
    SynthesisInput,
    ComposedResponse,
    # Top-level
    CompositionResult,
)

__all__ = [
    "CompositionPhase",
    "ExecutionStatus",
    "DependencyType",
    "SubQuestion",
    "DecompositionResult",
    "ToolMapping",
    "ExecutionStep",
    "ExecutionPlan",
    "ToolInput",
    "ToolOutput",
    "StepResult",
    "ExecutionTrace",
    "SynthesisInput",
    "ComposedResponse",
    "CompositionResult",
]
