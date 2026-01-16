"""Constants and error codes for CohortConstructor agent.

Defines error codes CC_001 through CC_007, default values,
and SLA thresholds.
"""

from enum import Enum
from typing import Dict, Any


# ==============================================================================
# ERROR CODES
# ==============================================================================


class CohortErrorCode(str, Enum):
    """Error codes for cohort construction failures.

    CC_001 - CC_007 cover validation, data, and execution errors.
    """

    CC_001 = "CC_001"  # Invalid configuration
    CC_002 = "CC_002"  # Missing required fields
    CC_003 = "CC_003"  # Empty cohort (all patients excluded)
    CC_004 = "CC_004"  # Data validation error
    CC_005 = "CC_005"  # Temporal validation failed
    CC_006 = "CC_006"  # Database storage error
    CC_007 = "CC_007"  # Timeout exceeded


# Error code descriptions for logging and user display
ERROR_DESCRIPTIONS: Dict[str, str] = {
    CohortErrorCode.CC_001: "Invalid cohort configuration: criteria malformed or missing required fields",
    CohortErrorCode.CC_002: "Missing required fields in patient data: cannot apply eligibility criteria",
    CohortErrorCode.CC_003: "Empty cohort: all patients excluded by eligibility criteria",
    CohortErrorCode.CC_004: "Data validation error: patient data does not meet schema requirements",
    CohortErrorCode.CC_005: "Temporal validation failed: insufficient lookback or followup period",
    CohortErrorCode.CC_006: "Database storage error: failed to persist cohort execution results",
    CohortErrorCode.CC_007: "Timeout exceeded: cohort construction exceeded SLA threshold",
}


# Recovery suggestions for each error code
ERROR_RECOVERY: Dict[str, list] = {
    CohortErrorCode.CC_001: [
        "Validate configuration against CohortConfig schema",
        "Check that all criteria have valid operators",
        "Ensure brand and indication are specified",
    ],
    CohortErrorCode.CC_002: [
        "Check that patient data contains all required fields",
        "Validate field names match configuration",
        "Review data schema documentation",
    ],
    CohortErrorCode.CC_003: [
        "Relax eligibility criteria thresholds",
        "Check data quality for high missing value rates",
        "Review clinical criteria with medical team",
    ],
    CohortErrorCode.CC_004: [
        "Run data quality checks on input dataset",
        "Validate data types match schema expectations",
        "Check for encoding issues in categorical fields",
    ],
    CohortErrorCode.CC_005: [
        "Increase lookback or followup period tolerance",
        "Check index date field contains valid dates",
        "Verify patient journey dates are complete",
    ],
    CohortErrorCode.CC_006: [
        "Check Supabase connection credentials",
        "Verify table exists and has correct schema",
        "Retry with smaller batch size",
    ],
    CohortErrorCode.CC_007: [
        "Reduce input dataset size",
        "Increase SLA timeout threshold",
        "Optimize query with indexes",
    ],
}


# ==============================================================================
# SLA THRESHOLDS
# ==============================================================================


class SLAThreshold:
    """SLA thresholds for CohortConstructor agent.

    Primary SLA: <120 seconds for 100K patients
    """

    # Total execution time (milliseconds)
    TOTAL_EXECUTION_MS = 120_000  # 120 seconds for 100K patients

    # Per-node latency targets (milliseconds)
    VALIDATE_CONFIG_MS = 100  # <100ms
    APPLY_CRITERIA_MS = 90_000  # <90 seconds
    VALIDATE_TEMPORAL_MS = 25_000  # <25 seconds
    GENERATE_METADATA_MS = 5_000  # <5 seconds

    # Size-based SLA thresholds (milliseconds)
    SMALL_COHORT_MS = 500  # <1,000 patients
    MEDIUM_COHORT_MS = 5_000  # 1,000-10,000 patients
    LARGE_COHORT_MS = 50_000  # 10,000-100,000 patients
    VERY_LARGE_COHORT_MS = 500_000  # >100,000 patients

    # Patients per second target
    PATIENTS_PER_SECOND = 833  # 100,000 / 120 seconds

    # Memory threshold
    MAX_MEMORY_MB = 2048  # 2GB for 100K patients


# ==============================================================================
# DEFAULT VALUES
# ==============================================================================


class Defaults:
    """Default values for cohort configuration."""

    # Temporal requirements
    LOOKBACK_DAYS = 180
    FOLLOWUP_DAYS = 90
    INDEX_DATE_FIELD = "diagnosis_date"

    # Batch processing
    BATCH_SIZE = 1000  # Patients per batch for Supabase inserts
    MAX_PATIENT_ASSIGNMENTS = 100_000

    # Versioning
    DEFAULT_VERSION = "1.0.0"
    DEFAULT_STATUS = "active"

    # Execution
    DEFAULT_ENVIRONMENT = "production"


# ==============================================================================
# SUPPORTED BRANDS AND INDICATIONS
# ==============================================================================


SUPPORTED_BRANDS: Dict[str, Dict[str, Any]] = {
    "remibrutinib": {
        "indication": "csu",
        "full_name": "Remibrutinib",
        "therapeutic_area": "Immunology",
        "mechanism": "BTK inhibitor",
        "description": "Chronic Spontaneous Urticaria (CSU)",
    },
    "fabhalta": {
        "indication": "pnh",
        "full_name": "Fabhalta (iptacopan)",
        "therapeutic_area": "Hematology",
        "mechanism": "Factor B inhibitor",
        "description": "Paroxysmal Nocturnal Hemoglobinuria (PNH) / C3 Glomerulopathy",
    },
    "kisqali": {
        "indication": "hr_her2_bc",
        "full_name": "Kisqali (ribociclib)",
        "therapeutic_area": "Oncology",
        "mechanism": "CDK4/6 inhibitor",
        "description": "HR+/HER2- Advanced Breast Cancer",
    },
}


# ==============================================================================
# CLINICAL CODE SYSTEMS
# ==============================================================================


class ClinicalCodeSystem(str, Enum):
    """Supported clinical code systems for diagnosis and procedure criteria."""

    ICD10_CM = "ICD-10-CM"  # Diagnosis codes (US)
    ICD10 = "ICD-10"  # Diagnosis codes (International)
    CPT = "CPT"  # Procedure codes
    HCPCS = "HCPCS"  # Healthcare Common Procedure Coding System
    NDC = "NDC"  # National Drug Code
    LOINC = "LOINC"  # Lab/observation codes


# ==============================================================================
# AGENT METADATA
# ==============================================================================


AGENT_METADATA: Dict[str, Any] = {
    "name": "cohort_constructor",
    "tier": 0,
    "type": "standard",  # tool-heavy, SLA-bound, no LLM
    "position": "scope_definer → cohort_constructor → data_preparer",
    "sla_seconds": 120,
    "supported_brands": list(SUPPORTED_BRANDS.keys()),
    "version": "1.0.0",
}
