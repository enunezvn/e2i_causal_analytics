"""
Digital Twin Pydantic Models
============================

Data models for twin entities and configurations.
"""

from enum import Enum
from typing import Optional, Dict, List, Any
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class TwinType(str, Enum):
    """Types of digital twins supported by the system."""
    HCP = "hcp"
    PATIENT = "patient"
    TERRITORY = "territory"


class Brand(str, Enum):
    """E2I pharmaceutical brands."""
    REMIBRUTINIB = "Remibrutinib"
    FABHALTA = "Fabhalta"
    KISQALI = "Kisqali"


class Region(str, Enum):
    """Geographic regions."""
    NORTHEAST = "northeast"
    SOUTH = "south"
    MIDWEST = "midwest"
    WEST = "west"


# =============================================================================
# HCP Twin Models
# =============================================================================

class HCPTwinFeatures(BaseModel):
    """Features for Healthcare Professional digital twins."""
    
    # Demographics
    specialty: str
    sub_specialty: Optional[str] = None
    years_experience: int = Field(ge=0, le=60)
    practice_type: str  # "academic", "community", "private"
    practice_size: str  # "solo", "small", "medium", "large"
    
    # Geographic
    region: Region
    state: str
    urban_rural: str  # "urban", "suburban", "rural"
    
    # Performance
    decile: int = Field(ge=1, le=10)
    priority_tier: int = Field(ge=1, le=5)
    total_patient_volume: int = Field(ge=0)
    target_patient_volume: int = Field(ge=0)
    prescribing_volume: int = Field(ge=0)
    
    # Engagement
    digital_engagement_score: float = Field(ge=0, le=1)
    preferred_channel: str
    last_interaction_days: int = Field(ge=0)
    interaction_frequency: float = Field(ge=0)
    
    # Adoption
    adoption_stage: str  # "innovator", "early_adopter", "early_majority", etc.
    peer_influence_score: float = Field(ge=0, le=1)


class PatientTwinFeatures(BaseModel):
    """Features for Patient journey digital twins."""
    
    # Demographics
    age_group: str
    gender: str
    geographic_region: Region
    socioeconomic_index: float = Field(ge=0, le=1)
    
    # Clinical
    primary_diagnosis_code: str
    comorbidity_count: int = Field(ge=0)
    risk_score: float = Field(ge=0, le=1)
    journey_complexity_score: float = Field(ge=0, le=1)
    
    # Insurance
    insurance_type: str
    insurance_coverage_flag: bool
    prior_auth_required: bool = False
    
    # Journey
    journey_stage: str
    journey_duration_days: int = Field(ge=0)
    treatment_line: int = Field(ge=1)


class TerritoryTwinFeatures(BaseModel):
    """Features for Territory digital twins."""
    
    # Geographic
    region: Region
    state_count: int = Field(ge=1)
    zip_count: int = Field(ge=1)
    
    # HCP coverage
    total_hcps: int = Field(ge=0)
    covered_hcps: int = Field(ge=0)
    coverage_rate: float = Field(ge=0, le=1)
    
    # Performance
    total_patient_volume: int = Field(ge=0)
    market_share: float = Field(ge=0, le=1)
    growth_rate: float
    
    # Competitive
    competitor_presence: float = Field(ge=0, le=1)


# =============================================================================
# Core Twin Models
# =============================================================================

class DigitalTwin(BaseModel):
    """A single digital twin entity."""
    
    twin_id: UUID = Field(default_factory=uuid4)
    twin_type: TwinType
    brand: Brand
    
    # Features (polymorphic based on twin_type)
    features: Dict[str, Any]
    
    # Baseline outcome (pre-intervention)
    baseline_outcome: float
    baseline_propensity: float = Field(ge=0, le=1)
    
    # Metadata
    source_entity_id: Optional[str] = None  # Original entity if derived
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v: Dict, info) -> Dict:
        """Ensure features are non-empty."""
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class TwinPopulation(BaseModel):
    """A population of digital twins for simulation."""
    
    population_id: UUID = Field(default_factory=uuid4)
    twin_type: TwinType
    brand: Brand
    
    twins: List[DigitalTwin]
    
    # Population statistics
    size: int
    feature_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Generation metadata
    model_id: Optional[UUID] = None
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v: int, info) -> int:
        """Ensure size matches twins list."""
        return v
    
    def __len__(self) -> int:
        return len(self.twins)
    
    def filter(self, **criteria) -> "TwinPopulation":
        """Filter twins by feature criteria."""
        filtered = [
            t for t in self.twins
            if all(t.features.get(k) == v for k, v in criteria.items())
        ]
        return TwinPopulation(
            twin_type=self.twin_type,
            brand=self.brand,
            twins=filtered,
            size=len(filtered),
            feature_summary={},
            model_id=self.model_id,
            generation_config=self.generation_config,
        )


# =============================================================================
# Model Configuration
# =============================================================================

class TwinModelConfig(BaseModel):
    """Configuration for training a twin generator model."""
    
    # Model identification
    model_name: str
    model_description: Optional[str] = None
    twin_type: TwinType
    brand: Brand
    
    # Algorithm settings
    algorithm: str = "random_forest"  # "random_forest", "gradient_boosting", "neural_net"
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: Optional[int] = Field(default=10, ge=1, le=50)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    
    # Training configuration
    training_samples: int = Field(default=50000, ge=1000)
    validation_split: float = Field(default=0.2, gt=0, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    random_state: int = 42
    
    # Feature configuration
    feature_columns: List[str]
    target_column: str
    propensity_column: Optional[str] = None
    
    # Outcome modeling
    outcome_model_type: str = "regression"  # "regression", "classification"
    
    # Geographic scope
    geographic_scope: str = "national"  # "national", "regional"


class TwinModelMetrics(BaseModel):
    """Performance metrics for a trained twin model."""
    
    model_id: UUID
    
    # Regression metrics
    r2_score: Optional[float] = Field(default=None, ge=-1, le=1)
    rmse: Optional[float] = Field(default=None, ge=0)
    mae: Optional[float] = Field(default=None, ge=0)
    
    # Cross-validation
    cv_scores: List[float] = Field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Feature importance
    feature_importances: Dict[str, float] = Field(default_factory=dict)
    top_features: List[str] = Field(default_factory=list)
    
    # Training metadata
    training_samples: int
    training_duration_seconds: float
    
    # Evaluation timestamp
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "r2_score": self.r2_score,
            "rmse": self.rmse,
            "mae": self.mae,
            "cv_scores": self.cv_scores,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "feature_importances": self.feature_importances,
            "top_features": self.top_features,
            "training_samples": self.training_samples,
            "training_duration_seconds": self.training_duration_seconds,
        }
