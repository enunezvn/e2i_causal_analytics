# CLAUDE.md - Data Preparer Agent

## Overview

The **Data Preparer** is the quality gatekeeper of the ML pipeline. It validates data against scope requirements, establishes baseline metrics, detects potential data leakage, and ensures data quality before model training can proceed.

| Attribute | Value |
|-----------|-------|
| **Tier** | 0 (ML Foundation) |
| **Type** | Standard |
| **SLA** | <60 seconds |
| **Primary Output** | QCReport, BaselineMetrics, DataReadiness |
| **Database Tables** | `ml_data_quality_reports`, `ml_feature_store` |
| **Memory Types** | Working, Episodic, Procedural |
| **MLOps Tools** | Great Expectations, Pandera, Feast |

## Responsibilities

1. **Schema Validation**: Run Pandera schema checks (types, nullability, enums)
2. **Data Quality Validation**: Run Great Expectations suites against data
3. **Baseline Metrics**: Compute feature distributions for drift detection
4. **Leakage Detection**: Identify temporal and target leakage risks
5. **Label Quality**: Assess annotation quality and inter-annotator agreement
6. **Feature Registration**: Register features in Feast feature store
7. **QC Gating**: Block downstream training if quality thresholds fail

## Position in Pipeline

```
┌──────────────────┐
│  scope_definer   │
│  (Problem Def)   │
└────────┬─────────┘
         │ ScopeSpec
         ▼
┌──────────────────┐
│  data_preparer   │ ◀── YOU ARE HERE
│  (Quality Check) │
└────────┬─────────┘
         │ QCReport (must PASS)
         ▼
┌──────────────────┐
│  model_selector  │
│  (Algorithm)     │
└──────────────────┘
```

## Inputs

### From scope_definer

```python
@dataclass
class ScopeSpec:
    experiment_id: str
    required_features: List[str]
    excluded_features: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    prediction_target: str
    minimum_data_quality_score: float  # From SuccessCriteria
```

### From Data Sources

```python
@dataclass
class DataPrepRequest:
    experiment_id: str
    data_source: str                 # "patient_journeys" | "treatment_events"
    split_id: str                    # ML split registry ID
    validation_suite: str            # Great Expectations suite name
```

## Outputs

### QCReport

```python
@dataclass
class QCReport:
    """Data quality check report."""
    report_id: str
    experiment_id: str
    
    # Overall Status
    status: DQStatus                  # passed | failed | warning | skipped
    overall_score: float              # 0.0 - 1.0
    
    # Dimension Scores
    completeness_score: float         # Missing value analysis
    validity_score: float             # Type/range validation
    consistency_score: float          # Cross-field consistency
    uniqueness_score: float           # Duplicate detection
    timeliness_score: float           # Data freshness
    
    # Detailed Results
    expectation_results: List[ExpectationResult]
    failed_expectations: List[str]
    warnings: List[str]
    
    # Recommendations
    remediation_steps: List[str]
    blocking_issues: List[str]        # Issues that block training
    
    # Metadata
    row_count: int
    column_count: int
    validated_at: datetime
    validation_duration_seconds: float
```

### BaselineMetrics

```python
@dataclass
class BaselineMetrics:
    """Feature distribution baselines for drift detection."""
    experiment_id: str
    split_type: str                   # "train" - baselines from training set only
    
    # Feature Statistics
    feature_stats: Dict[str, FeatureStats]
    # {
    #   "call_frequency": {
    #     "mean": 4.2, "std": 2.1, "min": 0, "max": 25,
    #     "percentiles": {25: 2, 50: 4, 75: 6, 99: 15},
    #     "null_rate": 0.02
    #   }
    # }
    
    # Target Statistics
    target_rate: float                # For classification: positive rate
    target_distribution: Dict         # For regression: distribution stats
    
    # Correlation Matrix (train only)
    correlation_matrix: Dict[str, Dict[str, float]]
    
    # Computed At
    computed_at: datetime
    training_samples: int
```

### DataReadiness

```python
@dataclass
class DataReadiness:
    """Summary of data readiness for training."""
    experiment_id: str
    is_ready: bool
    
    # Counts
    total_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    holdout_samples: int
    
    # Feature Availability
    available_features: List[str]
    missing_required_features: List[str]
    
    # Quality Summary
    qc_passed: bool
    qc_score: float
    
    # Blocking Issues
    blockers: List[str]               # Must be empty for is_ready=True
```

## Database Schema

### ml_data_quality_reports Table

```sql
CREATE TABLE ml_data_quality_reports (
    report_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES ml_experiments(experiment_id),
    
    -- Status
    status dq_status_enum NOT NULL,   -- passed, failed, warning, skipped
    overall_score NUMERIC(4,3),       -- 0.000 - 1.000
    
    -- Dimension Scores
    completeness_score NUMERIC(4,3),
    validity_score NUMERIC(4,3),
    consistency_score NUMERIC(4,3),
    uniqueness_score NUMERIC(4,3),
    timeliness_score NUMERIC(4,3),
    
    -- Results
    expectation_results JSONB NOT NULL,
    failed_expectations JSONB DEFAULT '[]',
    warnings JSONB DEFAULT '[]',
    
    -- Baseline Metrics (for drift_monitor)
    baseline_metrics JSONB,
    
    -- Metadata
    row_count INTEGER,
    column_count INTEGER,
    validation_suite TEXT,
    validated_by agent_name_enum DEFAULT 'data_preparer',
    validated_at TIMESTAMPTZ DEFAULT NOW(),
    validation_duration_seconds NUMERIC(8,3)
);

-- Index for fast lookup by experiment
CREATE INDEX idx_dq_experiment ON ml_data_quality_reports(experiment_id);
```

### ml_feature_store Table

```sql
CREATE TABLE ml_feature_store (
    feature_id TEXT PRIMARY KEY,
    feature_name TEXT NOT NULL,
    feature_group TEXT,               -- "engagement", "clinical", "market"
    
    -- Definition
    data_type TEXT NOT NULL,          -- "numeric", "categorical", "boolean"
    description TEXT,
    computation_logic TEXT,           -- SQL or Python snippet
    
    -- Statistics (from training set)
    statistics JSONB,
    
    -- Lineage
    source_table TEXT,
    source_columns JSONB,
    
    -- Metadata
    registered_by agent_name_enum DEFAULT 'data_preparer',
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    last_computed_at TIMESTAMPTZ
);
```

## Implementation

### agent.py

```python
from src.agents.base_agent import BaseAgent
from src.mlops.great_expectations_validator import GEValidator
from src.mlops.feast_client import FeastClient
from src.database.repositories.ml_data_quality import MLDataQualityRepository
from src.database.repositories.ml_feature_store import MLFeatureStoreRepository
from .quality_checker import QualityChecker
from .baseline_computer import BaselineComputer
from .leakage_detector import LeakageDetector

class DataPreparerAgent(BaseAgent):
    """
    Data Preparer: Validate data quality and establish baselines.
    
    CRITICAL: This agent acts as a GATE. If QC fails, training CANNOT proceed.
    """
    
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 60
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.ge_validator = GEValidator()
        self.feast_client = FeastClient()
        self.quality_checker = QualityChecker()
        self.baseline_computer = BaselineComputer()
        self.leakage_detector = LeakageDetector()
        self.dq_repo = MLDataQualityRepository()
        self.feature_repo = MLFeatureStoreRepository()
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Main execution: Validate data and compute baselines.
        
        Steps:
        1. Load scope specification
        2. Validate data against Great Expectations suite
        3. Detect potential data leakage
        4. Compute baseline metrics (train set only)
        5. Register features in Feast
        6. Generate QC report
        7. GATE: Block if quality fails
        """
        experiment_id = state.experiment_id
        scope_spec = state.scope_spec
        
        # Step 1: Run Great Expectations validation
        ge_results = await self.ge_validator.validate(
            data_source=state.data_source,
            suite_name=f"e2i_{scope_spec.problem_type}_suite",
            split_filter="train"  # Validate training data
        )
        
        # Step 2: Check for data leakage
        leakage_report = await self.leakage_detector.detect(
            experiment_id=experiment_id,
            target_column=scope_spec.prediction_target,
            feature_columns=scope_spec.required_features,
            time_column="event_date"
        )
        
        if leakage_report.has_leakage:
            return state.with_error(
                agent="data_preparer",
                error_type="data_leakage_detected",
                details=leakage_report.leakage_sources
            )
        
        # Step 3: Compute baseline metrics (train only)
        baseline_metrics = await self.baseline_computer.compute(
            experiment_id=experiment_id,
            features=scope_spec.required_features,
            target=scope_spec.prediction_target,
            split="train"
        )
        
        # Step 4: Register features in Feast
        await self._register_features(scope_spec)
        
        # Step 5: Generate QC report
        qc_report = self.quality_checker.generate_report(
            experiment_id=experiment_id,
            ge_results=ge_results,
            leakage_report=leakage_report,
            baseline_metrics=baseline_metrics
        )
        
        # Step 6: Persist report
        await self.dq_repo.create(qc_report)
        
        # Step 7: GATE CHECK
        if qc_report.status == DQStatus.FAILED:
            return state.with_updates(
                qc_report=qc_report,
                data_ready=False,
                blocking_issues=qc_report.blocking_issues
            ).with_gate_blocked(
                gate="data_quality",
                reason="QC score below threshold",
                details=qc_report.failed_expectations
            )
        
        # Step 8: Update episodic memory
        await self.episodic_memory.store(
            event_type="data_validated",
            content={
                "experiment_id": experiment_id,
                "qc_score": qc_report.overall_score,
                "status": qc_report.status.value
            }
        )
        
        return state.with_updates(
            qc_report=qc_report,
            baseline_metrics=baseline_metrics,
            data_ready=True
        )
    
    async def _register_features(self, scope_spec: ScopeSpec):
        """Register features in Feast feature store."""
        for feature_name in scope_spec.required_features:
            feature_def = await self._get_feature_definition(feature_name)
            await self.feast_client.register_feature(
                feature_name=feature_name,
                feature_group=feature_def.group,
                entity="hcp_id",
                dtype=feature_def.dtype
            )
```

### quality_checker.py

```python
class QualityChecker:
    """Generate QC reports from validation results."""
    
    def generate_report(
        self,
        experiment_id: str,
        ge_results: GEValidationResult,
        leakage_report: LeakageReport,
        baseline_metrics: BaselineMetrics
    ) -> QCReport:
        """Generate comprehensive QC report."""
        
        # Calculate dimension scores
        completeness = self._score_completeness(ge_results)
        validity = self._score_validity(ge_results)
        consistency = self._score_consistency(ge_results)
        uniqueness = self._score_uniqueness(ge_results)
        timeliness = self._score_timeliness(ge_results)
        
        # Overall score (weighted average)
        overall_score = (
            completeness * 0.25 +
            validity * 0.25 +
            consistency * 0.20 +
            uniqueness * 0.15 +
            timeliness * 0.15
        )
        
        # Determine status
        if overall_score >= 0.95:
            status = DQStatus.PASSED
        elif overall_score >= 0.80:
            status = DQStatus.WARNING
        else:
            status = DQStatus.FAILED
        
        # Identify blocking issues
        blocking_issues = []
        if completeness < 0.90:
            blocking_issues.append("Missing data exceeds 10% threshold")
        if leakage_report.has_leakage:
            blocking_issues.append(f"Data leakage detected: {leakage_report.summary}")
        
        return QCReport(
            report_id=f"qc_{experiment_id}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
            experiment_id=experiment_id,
            status=status,
            overall_score=overall_score,
            completeness_score=completeness,
            validity_score=validity,
            consistency_score=consistency,
            uniqueness_score=uniqueness,
            timeliness_score=timeliness,
            expectation_results=ge_results.results,
            failed_expectations=[r.expectation for r in ge_results.results if not r.success],
            warnings=self._extract_warnings(ge_results),
            remediation_steps=self._generate_remediation(ge_results),
            blocking_issues=blocking_issues,
            row_count=ge_results.statistics.row_count,
            column_count=ge_results.statistics.column_count,
            validated_at=datetime.utcnow(),
            validation_duration_seconds=ge_results.duration_seconds
        )
```

### leakage_detector.py

```python
class LeakageDetector:
    """Detect potential data leakage in ML datasets."""
    
    async def detect(
        self,
        experiment_id: str,
        target_column: str,
        feature_columns: List[str],
        time_column: str
    ) -> LeakageReport:
        """
        Detect data leakage:
        1. Temporal leakage: Features from after prediction time
        2. Target leakage: Features derived from target
        3. Train-test leakage: Information shared across splits
        """
        leakage_sources = []
        
        # Check 1: Temporal leakage
        temporal_issues = await self._check_temporal_leakage(
            feature_columns, time_column
        )
        leakage_sources.extend(temporal_issues)
        
        # Check 2: Target leakage (high correlation with target)
        target_issues = await self._check_target_leakage(
            feature_columns, target_column
        )
        leakage_sources.extend(target_issues)
        
        # Check 3: Train-test contamination
        split_issues = await self._check_split_leakage(experiment_id)
        leakage_sources.extend(split_issues)
        
        return LeakageReport(
            experiment_id=experiment_id,
            has_leakage=len(leakage_sources) > 0,
            leakage_sources=leakage_sources,
            summary=self._summarize(leakage_sources) if leakage_sources else None
        )
    
    async def _check_temporal_leakage(
        self,
        feature_columns: List[str],
        time_column: str
    ) -> List[LeakageSource]:
        """Check for features derived from future data."""
        issues = []
        
        # Features that should not exist before prediction time
        future_indicators = ["_future", "_next", "_after", "outcome_"]
        
        for col in feature_columns:
            for indicator in future_indicators:
                if indicator in col.lower():
                    issues.append(LeakageSource(
                        type="temporal",
                        feature=col,
                        reason=f"Feature name suggests future data: {indicator}"
                    ))
        
        return issues
```

### baseline_computer.py

```python
class BaselineComputer:
    """Compute baseline metrics from training data."""
    
    async def compute(
        self,
        experiment_id: str,
        features: List[str],
        target: str,
        split: str = "train"
    ) -> BaselineMetrics:
        """
        Compute baseline metrics from training set ONLY.
        
        These metrics are used by drift_monitor to detect distribution shifts.
        """
        # Load training data only
        train_data = await self._load_split_data(experiment_id, split="train")
        
        # Compute feature statistics
        feature_stats = {}
        for feature in features:
            if feature in train_data.columns:
                stats = self._compute_feature_stats(train_data[feature])
                feature_stats[feature] = stats
        
        # Compute target statistics
        if target in train_data.columns:
            target_series = train_data[target]
            if target_series.dtype == bool or target_series.nunique() == 2:
                # Classification: compute positive rate
                target_rate = target_series.mean()
                target_distribution = {"positive_rate": target_rate}
            else:
                # Regression: compute distribution
                target_rate = None
                target_distribution = self._compute_feature_stats(target_series)
        
        # Compute correlation matrix
        numeric_cols = train_data[features].select_dtypes(include=[np.number])
        correlation_matrix = numeric_cols.corr().to_dict()
        
        return BaselineMetrics(
            experiment_id=experiment_id,
            split_type="train",
            feature_stats=feature_stats,
            target_rate=target_rate,
            target_distribution=target_distribution,
            correlation_matrix=correlation_matrix,
            computed_at=datetime.utcnow(),
            training_samples=len(train_data)
        )
    
    def _compute_feature_stats(self, series: pd.Series) -> Dict:
        """Compute comprehensive statistics for a feature."""
        return {
            "mean": float(series.mean()) if series.dtype in [np.float64, np.int64] else None,
            "std": float(series.std()) if series.dtype in [np.float64, np.int64] else None,
            "min": float(series.min()) if series.dtype in [np.float64, np.int64] else None,
            "max": float(series.max()) if series.dtype in [np.float64, np.int64] else None,
            "percentiles": {
                25: float(series.quantile(0.25)),
                50: float(series.quantile(0.50)),
                75: float(series.quantile(0.75)),
                99: float(series.quantile(0.99))
            } if series.dtype in [np.float64, np.int64] else None,
            "null_rate": float(series.isnull().mean()),
            "unique_count": int(series.nunique()),
            "dtype": str(series.dtype)
        }
```

## Pandera Schema Validation

Pandera runs BEFORE Great Expectations as a fast-fail validation step (~10ms).

### Validation Pipeline Order

```
Input DataFrame
     │
     ▼
[1] PANDERA SCHEMA VALIDATION (Fast, ~10ms)
    - Column existence & naming
    - Data types (int, float, str, datetime)
    - Nullability constraints
    - Value ranges & E2I categories (brands, regions)
     │
     ▼
[2] QUALITY CHECKER (5 dimensions)
    - Completeness, Validity, Consistency, Uniqueness, Timeliness
     │
     ▼
[3] GREAT EXPECTATIONS (Business rules)
    - Statistical expectations
    - Cross-column consistency
```

### Schema Registry

The following Pandera schemas are defined in `src/mlops/pandera_schemas.py`:

| Data Source | Schema Class | Key Validations |
|-------------|-------------|-----------------|
| `business_metrics` | `BusinessMetricsSchema` | brand IN E2I_BRANDS, metric_date not null |
| `predictions` | `PredictionsSchema` | confidence_score 0-1, prediction_value 0-1 |
| `triggers` | `TriggersSchema` | priority IN E2I_PRIORITY_TYPES, confidence 0-1 |
| `patient_journeys` | `PatientJourneysSchema` | brand/region enums, patient_id not null |
| `causal_paths` | `CausalPathsSchema` | effect_strength -1 to 1, confidence 0-1 |
| `agent_activities` | `AgentActivitiesSchema` | agent_tier IN E2I_AGENT_TIERS |

### E2I Constants

```python
from src.mlops.pandera_schemas import E2I_BRANDS, E2I_REGIONS

E2I_BRANDS = ["Remibrutinib", "Fabhalta", "Kisqali", "All_Brands"]
E2I_REGIONS = ["northeast", "south", "midwest", "west"]
```

### Schema Validation Node

```python
# src/agents/ml_foundation/data_preparer/nodes/schema_validator.py

async def run_schema_validation(state: DataPreparerState) -> Dict[str, Any]:
    """Run Pandera schema validation on loaded data.

    Returns:
        schema_validation_status: "passed", "failed", "skipped", "error"
        schema_validation_errors: List of error dicts
        schema_splits_validated: Number of splits validated
        blocking_issues: Extended if schema fails
    """
```

### State Fields

```python
# Schema validation (Pandera)
schema_validation_status: Literal["passed", "failed", "skipped", "error"]
schema_validation_errors: List[Dict[str, Any]]
schema_splits_validated: int
schema_validation_time_ms: int
```

### Schema Failures Are Blocking

If Pandera schema validation fails, blocking_issues are added to state:

```python
if all_errors:
    blocking_issues.append(f"Schema validation failed: {error_summary}")
    # This blocks downstream training via QC gate
```

---

## Great Expectations Integration

### Expectation Suites

```python
# src/mlops/great_expectations_validator.py

class GEValidator:
    """Great Expectations integration for data validation."""
    
    def __init__(self):
        self.context = ge.get_context()
    
    async def validate(
        self,
        data_source: str,
        suite_name: str,
        split_filter: str = None
    ) -> GEValidationResult:
        """Run validation against expectation suite."""
        
        # Load data
        batch = self._get_batch(data_source, split_filter)
        
        # Load suite
        suite = self.context.get_expectation_suite(suite_name)
        
        # Validate
        results = self.context.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[batch],
            expectation_suite=suite
        )
        
        return GEValidationResult.from_ge_result(results)

# Default E2I expectation suite
E2I_CLASSIFICATION_SUITE = {
    "expectations": [
        # Completeness
        {"expectation_type": "expect_column_values_to_not_be_null",
         "kwargs": {"column": "hcp_id"}},
        {"expectation_type": "expect_column_values_to_not_be_null",
         "kwargs": {"column": "brand"}},
        
        # Validity
        {"expectation_type": "expect_column_values_to_be_in_set",
         "kwargs": {"column": "brand", 
                   "value_set": ["Remibrutinib", "Fabhalta", "Kisqali"]}},
        {"expectation_type": "expect_column_values_to_be_between",
         "kwargs": {"column": "call_frequency", "min_value": 0, "max_value": 100}},
        
        # Uniqueness
        {"expectation_type": "expect_column_values_to_be_unique",
         "kwargs": {"column": "patient_id"}},
        
        # Consistency
        {"expectation_type": "expect_column_pair_values_A_to_be_greater_than_B",
         "kwargs": {"column_A": "end_date", "column_B": "start_date"}}
    ]
}
```

## Downstream Integration

### model_trainer Gate Check

```python
# In model_trainer.execute():

# MANDATORY: Check QC status before training
qc_report = await self.dq_repo.get_latest(experiment_id)

if qc_report.status == DQStatus.FAILED:
    raise QCGateBlockedError(
        f"Cannot train: QC failed with score {qc_report.overall_score}"
    )

if qc_report.status == DQStatus.WARNING:
    logger.warning(f"Training with QC warnings: {qc_report.warnings}")
```

### drift_monitor Integration

```python
# In drift_monitor.execute():

# Load baseline metrics from data_preparer
baseline = await self.dq_repo.get_baseline_metrics(experiment_id)

# Compare current distribution to baseline
drift_scores = self.psi_calculator.compute(
    baseline=baseline.feature_stats,
    current=current_stats
)
```

## Error Handling

```python
class DataPreparerError(AgentError):
    """Base error for data_preparer."""
    pass

class QCFailedError(DataPreparerError):
    """Data quality check failed."""
    pass

class LeakageDetectedError(DataPreparerError):
    """Data leakage detected."""
    pass

class FeatureNotFoundError(DataPreparerError):
    """Required feature not found in data."""
    pass

class SplitNotFoundError(DataPreparerError):
    """Requested ML split not found."""
    pass
```

## Testing

```python
# tests/unit/test_agents/test_ml_foundation/test_data_preparer.py

class TestDataPreparer:
    
    async def test_qc_pass(self):
        """Test QC passes with clean data."""
        # ... setup clean data
        result = await agent.execute(state)
        assert result.qc_report.status == DQStatus.PASSED
    
    async def test_qc_fail_blocks_training(self):
        """Test QC failure blocks downstream training."""
        # ... setup dirty data
        result = await agent.execute(state)
        assert result.qc_report.status == DQStatus.FAILED
        assert result.gate_blocked is True
    
    async def test_leakage_detection(self):
        """Test temporal leakage is detected."""
        # ... setup data with future-looking features
        result = await agent.execute(state)
        assert "data_leakage_detected" in result.errors
    
    async def test_baseline_computation(self):
        """Test baseline metrics are computed from train only."""
        result = await agent.execute(state)
        assert result.baseline_metrics.split_type == "train"
```

## Key Principles

1. **Train-Only Baselines**: Baseline metrics computed ONLY from training data
2. **Gate Enforcement**: QC failure MUST block model_trainer
3. **Leakage Prevention**: Detect and block temporal/target leakage
4. **Feature Registration**: All features registered in Feast for lineage
5. **Comprehensive Logging**: All validation results persisted for audit
