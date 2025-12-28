"""
E2I Tool Composer - Data Models
"""

from .composition_models import (
    ComposedResponse,
    # Enums
    CompositionPhase,
    # Top-level
    CompositionResult,
    # Contract-compliant status enum (V4.3)
    CompositionStatus,
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    StepResult,
    # Phase 1 models
    SubQuestion,
    # Phase 4 models
    SynthesisInput,
    # Phase 3 models
    ToolInput,
    # Phase 2 models
    ToolMapping,
    ToolOutput,
)

__all__ = [
    "CompositionPhase",
    "CompositionStatus",  # Contract-compliant status enum
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
