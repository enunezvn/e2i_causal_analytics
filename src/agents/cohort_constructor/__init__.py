"""CohortConstructor: Tier 0 ML Foundation Agent for Patient Cohort Construction.

This agent implements explicit rule-based patient cohort construction following
FDA/EMA label criteria. It sits between scope_definer and data_preparer in the
Tier 0 ML Foundation pipeline.

Position: scope_definer → **cohort_constructor** → data_preparer
Type: Standard (tool-heavy, SLA-bound, no LLM)
SLA: <120 seconds for 100K patients

Key Features:
- 10 comparison operators (EQUAL, GREATER, IN, BETWEEN, etc.)
- Pre-built brand configurations (Remibrutinib, Fabhalta, Kisqali)
- Temporal eligibility validation (lookback/followup periods)
- Full audit trail with eligibility logging
- MLflow experiment tracking integration
- Opik distributed tracing

Usage:
    from src.agents.cohort_constructor import (
        CohortConstructor,
        CohortConfig,
        Criterion,
        Operator,
        CriterionType,
        get_brand_config,
    )

    # Use pre-built brand configuration
    config = get_brand_config("remibrutinib")

    # Or create custom configuration
    config = CohortConfig(
        cohort_name="Custom Cohort",
        brand="custom",
        indication="example",
        inclusion_criteria=[
            Criterion(
                field="age",
                operator=Operator.GREATER_EQUAL,
                value=18,
                criterion_type=CriterionType.INCLUSION,
                description="Adult patients only",
            )
        ],
        exclusion_criteria=[],
    )

    constructor = CohortConstructor(config)
    eligible_df, metadata = constructor.construct_cohort(patient_df)
"""

# Types
from .types import (
    Operator,
    CriterionType,
    Criterion,
    TemporalRequirements,
    CohortConfig,
    EligibilityLogEntry,
    PatientAssignment,
    CohortExecutionResult,
)

# Constants
from .constants import (
    CohortErrorCode,
    ERROR_DESCRIPTIONS,
    ERROR_RECOVERY,
    SLAThreshold,
    Defaults,
    SUPPORTED_BRANDS,
    ClinicalCodeSystem,
    AGENT_METADATA,
)

# State
from .state import (
    CohortConstructorState,
    CohortConfigInput,
    EligibilityStats,
    ExecutionMetadata,
    create_initial_state,
)

# Brand Configurations
from .configs import (
    get_brand_config,
    list_available_configs,
    get_config_for_brand_indication,
)

# Core Constructor
from .constructor import CohortConstructor

# Graph Workflow
from .graph import (
    create_cohort_constructor_graph,
    create_simple_cohort_constructor_graph,
)

# Node Functions
from .nodes import (
    validate_config,
    apply_criteria,
    validate_temporal,
    generate_metadata,
)

# Agent Wrapper
from .agent import (
    CohortConstructorAgent,
    create_cohort_constructor_agent,
)

# Observability
from .observability import (
    CohortMLflowLogger,
    CohortOpikTracer,
    CohortTraceContext,
    get_cohort_mlflow_logger,
    get_cohort_opik_tracer,
    track_cohort_step,
    track_cohort_construction,
    reset_observability_singletons,
)

__all__ = [
    # Types
    "Operator",
    "CriterionType",
    "Criterion",
    "TemporalRequirements",
    "CohortConfig",
    "EligibilityLogEntry",
    "PatientAssignment",
    "CohortExecutionResult",
    # Constants
    "CohortErrorCode",
    "ERROR_DESCRIPTIONS",
    "ERROR_RECOVERY",
    "SLAThreshold",
    "Defaults",
    "SUPPORTED_BRANDS",
    "ClinicalCodeSystem",
    "AGENT_METADATA",
    # State
    "CohortConstructorState",
    "CohortConfigInput",
    "EligibilityStats",
    "ExecutionMetadata",
    "create_initial_state",
    # Brand Configurations
    "get_brand_config",
    "list_available_configs",
    "get_config_for_brand_indication",
    # Core Constructor
    "CohortConstructor",
    # Graph Workflow
    "create_cohort_constructor_graph",
    "create_simple_cohort_constructor_graph",
    # Node Functions
    "validate_config",
    "apply_criteria",
    "validate_temporal",
    "generate_metadata",
    # Agent Wrapper
    "CohortConstructorAgent",
    "create_cohort_constructor_agent",
    # Observability
    "CohortMLflowLogger",
    "CohortOpikTracer",
    "CohortTraceContext",
    "get_cohort_mlflow_logger",
    "get_cohort_opik_tracer",
    "track_cohort_step",
    "track_cohort_construction",
    "reset_observability_singletons",
]

__version__ = "1.0.0"
