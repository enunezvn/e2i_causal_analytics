# CLAUDE.md - Scope Definer Agent

## Overview

The **Scope Definer** is the first agent in the Tier 0 ML Foundation pipeline. It transforms business requirements into formal ML problem specifications, establishing success criteria before any data preparation or modeling begins.

| Attribute | Value |
|-----------|-------|
| **Tier** | 0 (ML Foundation) |
| **Type** | Standard |
| **SLA** | <5 seconds |
| **Primary Output** | ScopeSpec, SuccessCriteria |
| **Database Table** | `ml_experiments` |
| **Memory Types** | Working, Episodic, Procedural |

## Responsibilities

1. **Problem Definition**: Translate business objectives into ML problem types
2. **Success Criteria**: Define measurable thresholds for model acceptance
3. **Constraint Specification**: Document regulatory, ethical, and technical constraints
4. **Baseline Expectations**: Establish minimum performance baselines
5. **Feature Requirements**: Specify required and excluded feature categories

## Position in Pipeline

```
┌──────────────────┐
│  scope_definer   │ ◀── YOU ARE HERE
│  (Problem Def)   │
└────────┬─────────┘
         │ ScopeSpec, SuccessCriteria
         ▼
┌──────────────────┐
│  data_preparer   │
│  (Quality Check) │
└────────┬─────────┘
         │
         ▼
    [continues...]
```

## Inputs

### From Orchestrator/User Request

```python
@dataclass
class ScopeRequest:
    """Input to scope_definer."""
    business_objective: str          # "Improve Remibrutinib adoption"
    target_outcome: str              # "Increase HCP prescriptions"
    brand: BrandType                 # Remibrutinib | Fabhalta | Kisqali
    region: Optional[RegionType]     # northeast | south | midwest | west
    time_horizon: str                # "Q1 2025"
    constraints: List[str]           # ["No PII in features", "AUC > 0.75"]
    prior_experiments: List[str]     # ["exp_001", "exp_002"]
```

### From Memory

- **Episodic**: Past scope definitions for similar objectives
- **Procedural**: Successful scope patterns by problem type

## Outputs

### ScopeSpec

```python
@dataclass
class ScopeSpec:
    """Formal ML problem specification."""
    experiment_id: str               # "exp_remib_northeast_2025q1"
    experiment_name: str             # "Remibrutinib Adoption - Northeast"
    
    # Problem Definition
    problem_type: ProblemType        # classification | regression | ranking
    prediction_target: str           # "will_prescribe_30d"
    prediction_horizon_days: int     # 30
    
    # Population
    target_population: str           # "HCPs with >5 CSU patients"
    inclusion_criteria: List[str]    # ["active_license", "specialty_match"]
    exclusion_criteria: List[str]    # ["already_prescribing", "restricted_territory"]
    
    # Features
    required_features: List[str]     # ["call_frequency", "sample_requests"]
    excluded_features: List[str]     # ["hcp_name", "dea_number"]
    feature_categories: List[str]    # ["engagement", "clinical", "market"]
    
    # Constraints
    regulatory_constraints: List[str]  # ["HIPAA", "state_privacy_laws"]
    ethical_constraints: List[str]     # ["no_race_features", "fairness_audit"]
    technical_constraints: List[str]   # ["inference_latency_<100ms"]
    
    # Metadata
    brand: BrandType
    region: Optional[RegionType]
    workstream: WorkstreamType       # WS1 | WS2 | WS3
    created_by: str                  # "scope_definer"
    created_at: datetime
```

### SuccessCriteria

```python
@dataclass
class SuccessCriteria:
    """Model acceptance thresholds."""
    experiment_id: str
    
    # Performance Thresholds
    minimum_auc: float               # 0.75
    minimum_precision_at_k: float    # 0.30 at k=100
    minimum_recall: float            # 0.60
    maximum_false_positive_rate: float  # 0.20
    
    # Baseline Comparison
    baseline_model: str              # "random" | "heuristic" | "prior_model"
    minimum_lift_over_baseline: float  # 1.5x
    
    # Fairness Criteria
    maximum_demographic_parity_gap: float  # 0.10
    maximum_equalized_odds_gap: float      # 0.10
    
    # Operational Criteria
    maximum_inference_latency_ms: int      # 100
    minimum_data_quality_score: float      # 0.95
    
    # Validation Requirements
    required_refutation_tests: List[str]   # ["placebo", "random_common_cause"]
    minimum_confidence_interval_coverage: float  # 0.95
```

## Database Schema

### ml_experiments Table

```sql
CREATE TABLE ml_experiments (
    experiment_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    brand brand_type NOT NULL,
    region region_type,
    workstream workstream_type DEFAULT 'WS1',
    
    -- Problem Definition
    problem_type TEXT NOT NULL,  -- classification, regression, ranking
    prediction_target TEXT NOT NULL,
    prediction_horizon_days INTEGER,
    
    -- Population
    target_population TEXT,
    inclusion_criteria JSONB DEFAULT '[]',
    exclusion_criteria JSONB DEFAULT '[]',
    
    -- Features
    required_features JSONB DEFAULT '[]',
    excluded_features JSONB DEFAULT '[]',
    
    -- Success Criteria
    success_criteria JSONB NOT NULL,
    
    -- Constraints
    regulatory_constraints JSONB DEFAULT '[]',
    ethical_constraints JSONB DEFAULT '[]',
    technical_constraints JSONB DEFAULT '[]',
    
    -- Status
    status experiment_status DEFAULT 'draft',
    
    -- Metadata
    created_by agent_name_enum DEFAULT 'scope_definer',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Implementation

### agent.py

```python
from src.agents.base_agent import BaseAgent
from src.database.repositories.ml_experiment import MLExperimentRepository
from src.memory.episodic_memory import EpisodicMemory
from src.memory.procedural_memory import ProceduralMemory
from .scope_builder import ScopeBuilder
from .criteria_validator import CriteriaValidator
from .models import ScopeSpec, SuccessCriteria, ScopeRequest

class ScopeDefinerAgent(BaseAgent):
    """
    Scope Definer: Transform business requirements into ML specifications.
    
    First agent in Tier 0 pipeline. Establishes problem definition,
    success criteria, and constraints before data preparation.
    """
    
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 5
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.experiment_repo = MLExperimentRepository()
        self.scope_builder = ScopeBuilder()
        self.criteria_validator = CriteriaValidator()
        self.episodic_memory = EpisodicMemory(agent="scope_definer")
        self.procedural_memory = ProceduralMemory(agent="scope_definer")
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Main execution: Build scope specification from request.
        
        Steps:
        1. Parse business objective
        2. Retrieve similar past scopes (episodic)
        3. Apply successful patterns (procedural)
        4. Build ScopeSpec
        5. Define SuccessCriteria
        6. Validate and persist
        """
        request = ScopeRequest.from_state(state)
        
        # Step 1: Retrieve relevant context
        similar_scopes = await self.episodic_memory.retrieve(
            query=request.business_objective,
            filters={"brand": request.brand},
            limit=3
        )
        
        scope_patterns = await self.procedural_memory.retrieve(
            query=f"scope for {request.target_outcome}",
            limit=2
        )
        
        # Step 2: Build scope specification
        scope_spec = await self.scope_builder.build(
            request=request,
            similar_scopes=similar_scopes,
            patterns=scope_patterns
        )
        
        # Step 3: Define success criteria
        success_criteria = await self._define_success_criteria(
            request=request,
            scope_spec=scope_spec
        )
        
        # Step 4: Validate
        validation = self.criteria_validator.validate(
            scope_spec=scope_spec,
            criteria=success_criteria
        )
        
        if not validation.is_valid:
            return state.with_error(
                agent="scope_definer",
                error_type="validation_failed",
                details=validation.errors
            )
        
        # Step 5: Persist to database
        await self.experiment_repo.create(
            scope_spec=scope_spec,
            success_criteria=success_criteria
        )
        
        # Step 6: Update episodic memory
        await self.episodic_memory.store(
            event_type="scope_defined",
            content={
                "experiment_id": scope_spec.experiment_id,
                "business_objective": request.business_objective,
                "problem_type": scope_spec.problem_type
            }
        )
        
        return state.with_updates(
            scope_spec=scope_spec,
            success_criteria=success_criteria,
            experiment_id=scope_spec.experiment_id
        )
    
    async def _define_success_criteria(
        self,
        request: ScopeRequest,
        scope_spec: ScopeSpec
    ) -> SuccessCriteria:
        """Define success criteria based on problem type and constraints."""
        
        # Default thresholds by problem type
        defaults = {
            "classification": {
                "minimum_auc": 0.75,
                "minimum_precision_at_k": 0.30,
                "minimum_recall": 0.60
            },
            "regression": {
                "maximum_rmse": None,  # Depends on target
                "minimum_r2": 0.50
            },
            "ranking": {
                "minimum_ndcg": 0.70,
                "minimum_map": 0.60
            }
        }
        
        base_criteria = defaults.get(scope_spec.problem_type, {})
        
        # Parse any explicit constraints from request
        for constraint in request.constraints:
            if "AUC" in constraint.upper():
                # Parse "AUC > 0.75" style constraints
                base_criteria["minimum_auc"] = self._parse_threshold(constraint)
        
        return SuccessCriteria(
            experiment_id=scope_spec.experiment_id,
            **base_criteria,
            baseline_model="heuristic",
            minimum_lift_over_baseline=1.5,
            maximum_demographic_parity_gap=0.10,
            maximum_equalized_odds_gap=0.10,
            maximum_inference_latency_ms=100,
            minimum_data_quality_score=0.95,
            required_refutation_tests=["placebo", "random_common_cause"],
            minimum_confidence_interval_coverage=0.95
        )
```

### scope_builder.py

```python
class ScopeBuilder:
    """Build ScopeSpec from request and context."""
    
    def __init__(self):
        self.problem_type_classifier = ProblemTypeClassifier()
    
    async def build(
        self,
        request: ScopeRequest,
        similar_scopes: List[ScopeSpec],
        patterns: List[ScopePattern]
    ) -> ScopeSpec:
        """
        Build a ScopeSpec by:
        1. Classifying problem type
        2. Inferring target variable
        3. Defining population criteria
        4. Specifying feature requirements
        """
        
        # Classify problem type from objective
        problem_type = self.problem_type_classifier.classify(
            request.business_objective,
            request.target_outcome
        )
        
        # Generate experiment ID
        experiment_id = self._generate_experiment_id(request)
        
        # Infer prediction target
        prediction_target = self._infer_target(
            request.target_outcome,
            problem_type
        )
        
        # Use patterns for feature requirements
        feature_requirements = self._apply_feature_patterns(
            request.brand,
            problem_type,
            patterns
        )
        
        return ScopeSpec(
            experiment_id=experiment_id,
            experiment_name=f"{request.brand.value} {request.target_outcome}",
            problem_type=problem_type,
            prediction_target=prediction_target,
            prediction_horizon_days=30,  # Default
            target_population=self._infer_population(request),
            inclusion_criteria=self._default_inclusion_criteria(request.brand),
            exclusion_criteria=self._default_exclusion_criteria(),
            required_features=feature_requirements.required,
            excluded_features=feature_requirements.excluded,
            feature_categories=feature_requirements.categories,
            regulatory_constraints=["HIPAA"],
            ethical_constraints=["no_race_features"],
            technical_constraints=["inference_latency_<100ms"],
            brand=request.brand,
            region=request.region,
            workstream=WorkstreamType.WS1,
            created_by="scope_definer",
            created_at=datetime.utcnow()
        )
    
    def _generate_experiment_id(self, request: ScopeRequest) -> str:
        """Generate unique experiment ID."""
        brand_code = request.brand.value[:4].lower()
        region_code = request.region.value[:2] if request.region else "all"
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
        return f"exp_{brand_code}_{region_code}_{timestamp}"
```

## Prompts

### prompts.py

```python
SCOPE_DEFINITION_SYSTEM = """You are a Machine Learning Problem Scoping specialist for pharmaceutical commercial analytics.

Your role is to translate business objectives into formal ML problem specifications.

CONTEXT:
- Platform: E2I Causal Analytics
- Domain: Pharmaceutical HCP targeting and patient journey optimization
- Brands: Remibrutinib (CSU), Fabhalta (PNH/C3G), Kisqali (breast cancer)

PROBLEM TYPES:
- Classification: Binary outcomes (will_prescribe, will_churn)
- Regression: Continuous outcomes (prescription_volume, time_to_event)
- Ranking: Ordered lists (hcp_priority_score)

OUTPUT FORMAT:
Return a structured specification with:
1. Problem type and target variable
2. Population definition (inclusion/exclusion)
3. Feature requirements
4. Success criteria thresholds
5. Constraints (regulatory, ethical, technical)
"""

SCOPE_DEFINITION_USER = """Business Objective: {business_objective}
Target Outcome: {target_outcome}
Brand: {brand}
Region: {region}
Time Horizon: {time_horizon}
Constraints: {constraints}

Similar Past Scopes:
{similar_scopes}

Successful Patterns:
{patterns}

Define the ML problem scope for this objective."""
```

## Memory Patterns

### Episodic Memory Usage

```python
# Store after successful scope definition
await episodic_memory.store(
    event_type="scope_defined",
    content={
        "experiment_id": "exp_remib_ne_202501",
        "business_objective": "Improve Remibrutinib adoption",
        "problem_type": "classification",
        "success": True
    },
    metadata={
        "brand": "Remibrutinib",
        "region": "northeast"
    }
)

# Retrieve similar scopes
similar = await episodic_memory.retrieve(
    query="Improve drug adoption",
    filters={"brand": "Remibrutinib"},
    limit=3
)
```

### Procedural Memory Usage

```python
# Retrieve successful patterns
patterns = await procedural_memory.retrieve(
    query="classification scope for HCP targeting",
    filters={"problem_type": "classification"},
    limit=2
)

# Store successful pattern after validation
if experiment_successful:
    await procedural_memory.store(
        pattern_type="scope_definition",
        content={
            "problem_type": "classification",
            "feature_categories": ["engagement", "clinical"],
            "success_rate": 0.85
        }
    )
```

## Error Handling

```python
class ScopeDefinerError(AgentError):
    """Base error for scope_definer."""
    pass

class InvalidObjectiveError(ScopeDefinerError):
    """Business objective cannot be parsed."""
    pass

class ConflictingConstraintsError(ScopeDefinerError):
    """Constraints are mutually exclusive."""
    pass

class MissingRequiredFieldError(ScopeDefinerError):
    """Required field not provided."""
    pass
```

## Validation Rules

1. **Business Objective**: Must be non-empty and parseable
2. **Brand**: Must be valid BrandType enum value
3. **Success Criteria**: AUC threshold must be ≥ 0.5 (better than random)
4. **Constraints**: No conflicting constraints allowed
5. **Features**: Excluded features cannot overlap with required features

## Testing

### Unit Tests

```python
# tests/unit/test_agents/test_ml_foundation/test_scope_definer.py

class TestScopeDefiner:
    
    async def test_basic_scope_creation(self):
        """Test creating a basic classification scope."""
        agent = ScopeDefinerAgent(config)
        state = AgentState(
            request=ScopeRequest(
                business_objective="Improve Remibrutinib adoption",
                target_outcome="Increase prescriptions",
                brand=BrandType.REMIBRUTINIB
            )
        )
        
        result = await agent.execute(state)
        
        assert result.scope_spec is not None
        assert result.scope_spec.problem_type == "classification"
        assert result.success_criteria.minimum_auc >= 0.5
    
    async def test_constraint_parsing(self):
        """Test parsing explicit constraints."""
        request = ScopeRequest(
            constraints=["AUC > 0.80", "No PII features"]
        )
        # ... verify constraints are parsed correctly
    
    async def test_episodic_memory_retrieval(self):
        """Test retrieval of similar past scopes."""
        # ... verify memory integration
```

## Integration Points

### Downstream: data_preparer

```python
# data_preparer reads ScopeSpec to validate data against requirements
scope_spec = await experiment_repo.get(experiment_id)

# Validate features exist
missing_features = set(scope_spec.required_features) - available_features
if missing_features:
    raise MissingFeaturesError(missing_features)

# Apply exclusion criteria to population
population = population.exclude(scope_spec.exclusion_criteria)
```

### Upstream: Orchestrator

```python
# Orchestrator routes scope requests to scope_definer
if intent.type == "SCOPE":
    return await scope_definer.execute(state)
```

## Observability

All operations emit spans via observability_connector:

```python
with observability.span("scope_definer.execute") as span:
    span.set_attribute("experiment_id", scope_spec.experiment_id)
    span.set_attribute("problem_type", scope_spec.problem_type)
    span.set_attribute("brand", request.brand.value)
```
