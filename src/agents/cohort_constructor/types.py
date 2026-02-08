"""Type definitions for CohortConstructor agent.

Defines the core data structures for explicit rule-based patient cohort construction.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Operator(str, Enum):
    """Comparison operators for eligibility criteria.

    Supports 10 operators covering equality, inequality, ordering,
    set membership, range checking, and string containment.
    """

    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    CONTAINS = "contains"


class CriterionType(str, Enum):
    """Type of eligibility criterion."""

    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


@dataclass
class Criterion:
    """Single eligibility criterion with clinical context.

    Attributes:
        field: Patient data field to evaluate
        operator: Comparison operator
        value: Value(s) to compare against
        criterion_type: Whether this is inclusion or exclusion
        description: Human-readable description
        clinical_rationale: FDA/EMA label justification
    """

    field: str
    operator: Operator
    value: Any
    criterion_type: CriterionType
    description: str = ""
    clinical_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "criterion_type": self.criterion_type.value,
            "description": self.description,
            "clinical_rationale": self.clinical_rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Criterion":
        """Create Criterion from dictionary."""
        return cls(
            field=data["field"],
            operator=Operator(data["operator"]),
            value=data["value"],
            criterion_type=CriterionType(data["criterion_type"]),
            description=data.get("description", ""),
            clinical_rationale=data.get("clinical_rationale", ""),
        )


@dataclass
class TemporalRequirements:
    """Temporal eligibility requirements.

    Ensures sufficient historical data (lookback) and outcome
    observation period (followup) relative to index date.
    """

    lookback_days: int = 180
    followup_days: int = 90
    index_date_field: str = "diagnosis_date"


@dataclass
class CohortConfig:
    """Complete cohort configuration with versioning.

    Defines all eligibility criteria, temporal requirements,
    and metadata for a cohort definition.
    """

    cohort_name: str
    brand: str
    indication: str
    inclusion_criteria: List[Criterion]
    exclusion_criteria: List[Criterion]
    temporal_requirements: TemporalRequirements = field(default_factory=TemporalRequirements)
    required_fields: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    status: str = "active"  # active, draft, archived, locked
    clinical_rationale: str = ""
    regulatory_justification: str = ""

    def __post_init__(self):
        """Compute config hash after initialization."""
        self._config_hash: Optional[str] = None

    @property
    def config_hash(self) -> str:
        """Generate SHA256 hash of configuration for version tracking."""
        if self._config_hash is None:
            config_str = json.dumps(self.to_dict(), sort_keys=True)
            self._config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return self._config_hash

    @property
    def cohort_id(self) -> str:
        """Generate unique cohort ID from brand, indication, and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"cohort_{self.brand}_{self.indication}_v{self.version.replace('.', '')}_{timestamp}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Handle None temporal_requirements gracefully
        if self.temporal_requirements is not None:
            temporal_dict = {
                "lookback_days": self.temporal_requirements.lookback_days,
                "followup_days": self.temporal_requirements.followup_days,
                "index_date_field": self.temporal_requirements.index_date_field,
            }
        else:
            temporal_dict = None

        return {
            "cohort_name": self.cohort_name,
            "brand": self.brand,
            "indication": self.indication,
            "inclusion_criteria": [c.to_dict() for c in self.inclusion_criteria],
            "exclusion_criteria": [c.to_dict() for c in self.exclusion_criteria],
            "temporal_requirements": temporal_dict,
            "required_fields": self.required_fields,
            "version": self.version,
            "status": self.status,
            "clinical_rationale": self.clinical_rationale,
            "regulatory_justification": self.regulatory_justification,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohortConfig":
        """Create CohortConfig from dictionary."""
        temporal = data.get("temporal_requirements")
        # Handle None temporal_requirements explicitly
        if temporal is None:
            temporal_req = None
        else:
            temporal_req = TemporalRequirements(
                lookback_days=temporal.get("lookback_days", 180),
                followup_days=temporal.get("followup_days", 90),
                index_date_field=temporal.get("index_date_field", "diagnosis_date"),
            )

        return cls(
            cohort_name=data["cohort_name"],
            brand=data["brand"],
            indication=data["indication"],
            inclusion_criteria=[Criterion.from_dict(c) for c in data.get("inclusion_criteria", [])],
            exclusion_criteria=[Criterion.from_dict(c) for c in data.get("exclusion_criteria", [])],
            temporal_requirements=temporal_req or TemporalRequirements(),
            required_fields=data.get("required_fields", []),
            version=data.get("version", "1.0.0"),
            status=data.get("status", "active"),
            clinical_rationale=data.get("clinical_rationale", ""),
            regulatory_justification=data.get("regulatory_justification", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CohortConfig":
        """Create CohortConfig from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class EligibilityLogEntry:
    """Single entry in the eligibility audit log.

    Records the impact of applying one criterion on the population.
    """

    criterion_name: str
    criterion_type: str  # inclusion, exclusion, temporal
    criterion_order: int
    operator: str
    value: Any
    removed_count: int
    remaining_count: int
    description: str = ""
    clinical_rationale: str = ""
    applied_at: Optional[datetime] = None

    def __post_init__(self):
        if self.applied_at is None:
            self.applied_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "criterion_name": self.criterion_name,
            "criterion_type": self.criterion_type,
            "criterion_order": self.criterion_order,
            "operator": self.operator,
            "value": self.value,
            "removed_count": self.removed_count,
            "remaining_count": self.remaining_count,
            "description": self.description,
            "clinical_rationale": self.clinical_rationale,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }


@dataclass
class PatientAssignment:
    """Individual patient eligibility assignment.

    Records whether a patient is eligible and why they were excluded.
    """

    patient_journey_id: str
    is_eligible: bool
    failed_criteria: List[str] = field(default_factory=list)
    lookback_complete: Optional[bool] = None
    followup_complete: Optional[bool] = None
    index_date: Optional[str] = None
    journey_start_date: Optional[str] = None
    journey_end_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "patient_journey_id": self.patient_journey_id,
            "is_eligible": self.is_eligible,
            "failed_criteria": self.failed_criteria,
            "lookback_complete": self.lookback_complete,
            "followup_complete": self.followup_complete,
            "index_date": self.index_date,
            "journey_start_date": self.journey_start_date,
            "journey_end_date": self.journey_end_date,
        }


@dataclass
class CohortExecutionResult:
    """Complete result from cohort construction.

    Contains eligible patients, statistics, and full audit trail.
    """

    cohort_id: str
    execution_id: str
    eligible_patient_ids: List[str]
    eligibility_stats: Dict[str, Any]
    eligibility_log: List[EligibilityLogEntry]
    patient_assignments: List[PatientAssignment]
    execution_metadata: Dict[str, Any]
    status: str = "success"  # success, failed, partial
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "cohort_id": self.cohort_id,
            "execution_id": self.execution_id,
            "eligible_patient_ids": self.eligible_patient_ids,
            "eligibility_stats": self.eligibility_stats,
            "eligibility_log": [e.to_dict() for e in self.eligibility_log],
            "patient_assignments": [p.to_dict() for p in self.patient_assignments],
            "execution_metadata": self.execution_metadata,
            "status": self.status,
            "error_message": self.error_message,
            "error_code": self.error_code,
        }
