"""State definition for CohortConstructor agent.

Defines the TypedDict-based state for the 4-node cohort construction workflow:
1. validate_config - Validate configuration and input data
2. apply_criteria - Apply inclusion/exclusion criteria
3. validate_temporal - Validate temporal eligibility
4. generate_metadata - Generate execution metadata and audit trail
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class CriterionSpec(TypedDict, total=False):
    """Specification for a single eligibility criterion."""

    field: str
    operator: str  # ==, !=, >, >=, <, <=, in, not_in, between, contains
    value: Any
    criterion_type: Literal["inclusion", "exclusion"]
    description: str
    clinical_rationale: str


class TemporalSpec(TypedDict, total=False):
    """Temporal eligibility requirements."""

    lookback_days: int
    followup_days: int
    index_date_field: str


class CohortConfigInput(TypedDict, total=False):
    """Input configuration for cohort construction."""

    cohort_name: str
    brand: str
    indication: str
    inclusion_criteria: List[CriterionSpec]
    exclusion_criteria: List[CriterionSpec]
    temporal_requirements: TemporalSpec
    required_fields: List[str]
    version: str
    status: Literal["active", "draft", "archived", "locked"]
    clinical_rationale: str
    regulatory_justification: str


class EligibilityLogEntry(TypedDict, total=False):
    """Single entry in the eligibility audit log."""

    criterion_name: str
    criterion_type: Literal["inclusion", "exclusion", "temporal"]
    criterion_order: int
    operator: str
    value: Any
    removed_count: int
    remaining_count: int
    description: str
    clinical_rationale: str
    applied_at: str  # ISO timestamp


class PatientAssignmentRecord(TypedDict, total=False):
    """Individual patient eligibility record."""

    patient_journey_id: str
    is_eligible: bool
    failed_criteria: List[str]
    lookback_complete: Optional[bool]
    followup_complete: Optional[bool]
    index_date: Optional[str]
    journey_start_date: Optional[str]
    journey_end_date: Optional[str]


class EligibilityStats(TypedDict, total=False):
    """Statistics from cohort construction."""

    total_input_patients: int
    eligible_patient_count: int
    excluded_patient_count: int
    exclusion_rate: float  # 0.0 - 1.0
    eligibility_log: List[EligibilityLogEntry]
    temporal_validation: Dict[str, Any]


class ExecutionMetadata(TypedDict, total=False):
    """Metadata from cohort construction execution."""

    execution_id: str
    execution_timestamp: str  # ISO timestamp
    execution_time_ms: int
    environment: Literal["development", "staging", "production"]
    executed_by: Optional[str]

    # Node latencies
    validate_config_ms: int
    apply_criteria_ms: int
    validate_temporal_ms: int
    generate_metadata_ms: int

    # Database records
    database_records: Dict[str, Any]


class CohortConstructorState(TypedDict, total=False):
    """Complete state for CohortConstructor agent.

    This is the main state object passed through the 4-node workflow:
    validate_config → apply_criteria → validate_temporal → generate_metadata
    """

    # ========================================================================
    # INPUT FIELDS (From scope_definer or direct input)
    # ========================================================================

    # Configuration
    config: Optional[CohortConfigInput]
    config_override: Optional[Dict[str, Any]]  # Override pre-built config

    # Brand/indication for pre-built config lookup
    brand: Optional[str]
    indication: Optional[str]

    # Source population (from scope_definer)
    source_population: Optional[Dict[str, Any]]  # DataFrame info or reference
    input_patient_ids: Optional[List[str]]

    # Execution context
    execution_context: Dict[str, Any]
    environment: Literal["development", "staging", "production"]
    executed_by: Optional[str]

    # ========================================================================
    # NODE 1 OUTPUT: Configuration Validation
    # ========================================================================

    # Validated configuration
    validated_config: Optional[CohortConfigInput]
    config_valid: bool
    config_errors: Annotated[List[str], operator.add]

    # Required fields validation
    required_fields_present: bool
    missing_fields: List[str]

    # Timing
    validate_config_ms: int

    # ========================================================================
    # NODE 2 OUTPUT: Criteria Application
    # ========================================================================

    # After inclusion criteria
    post_inclusion_count: int
    inclusion_log: List[EligibilityLogEntry]

    # After exclusion criteria
    post_exclusion_count: int
    exclusion_log: List[EligibilityLogEntry]

    # Timing
    apply_criteria_ms: int

    # Internal: eligible indices from criteria application (for next node)
    _eligible_indices: List[Any]

    # ========================================================================
    # NODE 3 OUTPUT: Temporal Validation
    # ========================================================================

    # Temporal eligibility
    post_temporal_count: int
    temporal_log: List[EligibilityLogEntry]

    # Temporal validation details
    lookback_exclusions: int
    followup_exclusions: int
    total_temporal_exclusions: int

    # Timing
    validate_temporal_ms: int

    # Internal: eligible indices after temporal validation (for next node)
    _temporal_eligible_indices: List[Any]

    # ========================================================================
    # NODE 4 OUTPUT: Metadata Generation
    # ========================================================================

    # Final cohort
    cohort_id: Optional[str]
    eligible_patient_ids: List[str]
    eligible_patient_count: int

    # Statistics
    eligibility_stats: Optional[EligibilityStats]

    # Patient assignments (for audit trail)
    patient_assignments: List[PatientAssignmentRecord]

    # Execution metadata
    execution_metadata: Optional[ExecutionMetadata]

    # Timing
    generate_metadata_ms: int

    # ========================================================================
    # WORKFLOW STATE
    # ========================================================================

    # Current phase
    current_phase: Literal[
        "initializing",
        "validating_config",
        "applying_criteria",
        "validating_temporal",
        "generating_metadata",
        "complete",
    ]

    # Overall status
    status: Literal["pending", "processing", "completed", "failed", "partial"]

    # ========================================================================
    # TIMING METADATA
    # ========================================================================

    # Total execution
    start_time: Optional[str]  # ISO timestamp
    end_time: Optional[str]  # ISO timestamp
    total_latency_ms: int

    # SLA compliance
    sla_target_ms: int  # 120,000 ms for 100K patients
    sla_compliant: bool

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]

    # Primary error (for status=failed)
    error: Optional[str]
    error_code: Optional[str]  # CC_001 through CC_007
    error_category: Optional[
        Literal[
            "INVALID_CONFIG",
            "MISSING_FIELDS",
            "EMPTY_COHORT",
            "DATA_VALIDATION",
            "TEMPORAL_VALIDATION",
            "DATABASE_ERROR",
            "TIMEOUT",
        ]
    ]

    # ========================================================================
    # HANDOFF FIELDS (For data_preparer)
    # ========================================================================

    # Context for next agent
    context_for_next_agent: Optional[Dict[str, Any]]
    suggested_next_agent: Optional[str]  # "data_preparer" or null if blocked
    pipeline_blocked: bool
    block_reason: Optional[str]

    # Key findings summary
    key_findings: List[str]
    summary_report: Optional[str]
    recommended_actions: List[str]

    # Assumptions and limitations
    assumptions: List[str]
    limitations: List[str]

    # Further analysis flag
    requires_further_analysis: bool
    confidence: float  # 0.0 - 1.0


# ============================================================================
# INITIAL STATE FACTORY
# ============================================================================


def create_initial_state(
    brand: Optional[str] = None,
    indication: Optional[str] = None,
    config: Optional[CohortConfigInput] = None,
    environment: str = "production",
) -> CohortConstructorState:
    """Create initial state for cohort construction.

    Args:
        brand: Brand name for pre-built config lookup
        indication: Indication for pre-built config lookup
        config: Explicit configuration (overrides brand/indication)
        environment: Execution environment

    Returns:
        Initialized CohortConstructorState
    """
    return CohortConstructorState(
        # Input
        brand=brand,
        indication=indication,
        config=config,
        environment=environment,
        # Workflow state
        current_phase="initializing",
        status="pending",
        # Initialize lists
        config_errors=[],
        missing_fields=[],
        inclusion_log=[],
        exclusion_log=[],
        temporal_log=[],
        eligible_patient_ids=[],
        patient_assignments=[],
        errors=[],
        warnings=[],
        key_findings=[],
        recommended_actions=[],
        assumptions=[],
        limitations=[],
        # Defaults
        config_valid=False,
        required_fields_present=False,
        pipeline_blocked=False,
        requires_further_analysis=False,
        sla_compliant=True,
        sla_target_ms=120_000,
        confidence=0.0,
        # Counters
        post_inclusion_count=0,
        post_exclusion_count=0,
        post_temporal_count=0,
        eligible_patient_count=0,
        lookback_exclusions=0,
        followup_exclusions=0,
        total_temporal_exclusions=0,
        total_latency_ms=0,
        validate_config_ms=0,
        apply_criteria_ms=0,
        validate_temporal_ms=0,
        generate_metadata_ms=0,
        # Internal indices (passed between nodes)
        _eligible_indices=[],
        _temporal_eligible_indices=[],
    )
