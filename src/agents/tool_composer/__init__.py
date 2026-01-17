"""
E2I Tool Composer
Version: 4.2

Dynamic tool composition for multi-faceted queries.

The Tool Composer enables answering complex queries that span multiple
agent capabilities by decomposing them into atomic sub-questions,
mapping to tools, executing in dependency order, and synthesizing results.

Pipeline:
    Phase 1: DECOMPOSE - Break query into atomic sub-questions
    Phase 2: PLAN     - Map sub-questions to tools, create execution plan
    Phase 3: EXECUTE  - Run tools in dependency order
    Phase 4: SYNTHESIZE - Combine results into coherent response

Usage:
    from src.agents.tool_composer import ToolComposer, compose_query

    # Async usage
    composer = ToolComposer(llm_client=anthropic_client)
    result = await composer.compose("Compare X and predict Y")
    print(result.response.answer)

    # Sync convenience function
    result = compose_query_sync("Compare X and predict Y", llm_client)
"""

from .agent import ToolComposerAgent, ToolComposerOutput
from .composer import (
    ToolComposer,
    ToolComposerIntegration,
    compose_query,
    compose_query_sync,
)
from .decomposer import (
    DecompositionError,
    QueryDecomposer,
    decompose_sync,
)
from .executor import (
    ExecutionError,
    PlanExecutor,
    execute_sync,
)
from .models.composition_models import (
    ComposedResponse,
    # Enums
    CompositionPhase,
    # Top-level
    CompositionResult,
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
from .planner import (
    PlanningError,
    ToolPlanner,
    plan_sync,
)
from .synthesizer import (
    ResponseSynthesizer,
    synthesize_results,
    synthesize_sync,
)

__all__ = [
    # Agent class (for factory registration)
    "ToolComposerAgent",
    "ToolComposerOutput",
    # Main classes
    "ToolComposer",
    "ToolComposerIntegration",
    # Phase handlers
    "QueryDecomposer",
    "ToolPlanner",
    "PlanExecutor",
    "ResponseSynthesizer",
    # Convenience functions
    "compose_query",
    "compose_query_sync",
    "decompose_sync",
    "plan_sync",
    "execute_sync",
    "synthesize_results",
    "synthesize_sync",
    # Exceptions
    "DecompositionError",
    "PlanningError",
    "ExecutionError",
    # Enums
    "CompositionPhase",
    "ExecutionStatus",
    "DependencyType",
    # Models
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

__version__ = "4.2.0"
