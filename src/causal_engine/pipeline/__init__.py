"""Pipeline orchestration for multi-library causal analysis.

This module provides orchestration capabilities for routing queries to the
appropriate causal inference library and coordinating multi-library workflows.

Multi-Library Synergies Architecture:
- Library Router: Routes queries based on question type
- Sequential Pipeline: NetworkX → DoWhy → EconML → CausalML
- Parallel Analysis: All four libraries simultaneously
- Validation Loop: DoWhy ↔ CausalML cross-validation

Reference: docs/Data Architecture & Integration.html
"""

from .orchestrator import (
    CausalMLExecutor,
    DoWhyExecutor,
    EconMLExecutor,
    LibraryExecutor,
    NetworkXExecutor,
    PipelineOrchestrator,
)
from .parallel import (
    ParallelPipeline,
    ParallelPipelineBuilder,
    create_parallel_pipeline,
)
from .router import CausalLibrary, LibraryCapability, LibraryRouter, QuestionType
from .sequential import (
    LIBRARY_STAGES,
    SEQUENTIAL_ORDER,
    SequentialPipeline,
    SequentialPipelineBuilder,
    create_sequential_pipeline,
)
from .state import (
    LibraryExecutionResult,
    PipelineConfig,
    PipelineInput,
    PipelineOutput,
    PipelineStage,
    PipelineState,
)
from .validators import (
    DoWhyToEconMLValidator,
    EconMLToCausalMLValidator,
    NetworkXToDoWhyValidator,
    PipelineValidator,
    StageValidator,
    ValidationResult,
    validate_pipeline_state,
)

__all__ = [
    # Router
    "LibraryRouter",
    "QuestionType",
    "CausalLibrary",
    "LibraryCapability",
    # State
    "PipelineState",
    "PipelineStage",
    "PipelineConfig",
    "PipelineInput",
    "PipelineOutput",
    "LibraryExecutionResult",
    # Orchestrator
    "PipelineOrchestrator",
    "LibraryExecutor",
    "NetworkXExecutor",
    "DoWhyExecutor",
    "EconMLExecutor",
    "CausalMLExecutor",
    # Sequential Pipeline
    "SequentialPipeline",
    "SequentialPipelineBuilder",
    "create_sequential_pipeline",
    "SEQUENTIAL_ORDER",
    "LIBRARY_STAGES",
    # Parallel Pipeline
    "ParallelPipeline",
    "ParallelPipelineBuilder",
    "create_parallel_pipeline",
    # Validators
    "PipelineValidator",
    "StageValidator",
    "ValidationResult",
    "NetworkXToDoWhyValidator",
    "DoWhyToEconMLValidator",
    "EconMLToCausalMLValidator",
    "validate_pipeline_state",
]
