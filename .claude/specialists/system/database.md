# Database Specialist Instructions

## Domain Scope
You are the Database specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `src/repositories/` - All data access layer code
- `scripts/setup_db.py` - Database schema initialization
- `scripts/load_v3_data.py` - Data loading scripts
- PostgreSQL/Supabase schema design

## Agent Activity Type Enum
```sql
-- V3: 11 agents
CREATE TYPE agent_type AS ENUM (
    'orchestrator',
    'causal_impact',
    'gap_analyzer',
    'heterogeneous_optimizer',
    'experiment_designer',
    'drift_monitor',
    'health_score',
    'prediction_synthesizer',
    'resource_optimizer',
    'explainer',
    'feedback_learner'
);
```

Reference: `migrations/006_update_agent_enum.sql`

## agent_activities Table

| Column | Type | Description |
|--------|------|-------------|
| agent_type | agent_type | One of 11 V3 agents |
| tier | integer | Agent tier (1-5) |
| ... | ... | ... |

## V3 Schema Overview

### Core Tables (Existing)
```
patient_journeys       - Patient journey data with source tracking
ml_predictions         - Model predictions with rank metrics
triggers               - HCP triggers with change tracking
business_metrics       - KPI snapshots
causal_paths           - Discovered causal relationships
agent_activities       - Agent analysis outputs
conversations          - Chat history for RAG
```

### V3 New Tables (KPI Gap Fillers)
```
user_sessions          - MAU/WAU/DAU tracking
data_source_tracking   - Cross-source match rates
ml_annotations         - Label quality (IAA)
etl_pipeline_metrics   - Time-to-release
hcp_intent_surveys     - Intent-to-prescribe delta
reference_universe     - Coverage targets
agent_registry         - 11 agents with tier assignments
```

## Repository Pattern

All repositories follow this pattern:
```python
class BaseRepository(Generic[T]):
    """
    Base repository with split-aware querying.
    
    CRITICAL: All queries must respect ML splits to prevent leakage.
    """
    
    def __init__(self, supabase: SupabaseClient):
        self.client = supabase
        self.table_name: str
        self.model_class: Type[T]
    
    async def get_by_id(self, id: str, split: Optional[str] = None) -> Optional[T]:
        query = self.client.table(self.table_name).select("*").eq("id", id)
        if split:
            query = query.eq("split_assignment", split)
        return self._to_model(await query.execute())
    
    async def get_many(
        self,
        filters: Dict[str, Any],
        split: Optional[str] = None,
        limit: int = 100
    ) -> List[T]:
        pass
```

## Split-Aware Queries (CRITICAL)

### ML Split Configuration
```yaml
# config/ml_split_config.yaml
train_ratio: 0.60
validation_ratio: 0.20
test_ratio: 0.15
holdout_ratio: 0.05
temporal_gap_days: 7
```

### Split Assignment Logic
```python
class SplitAssignment:
    """
    Assign records to splits based on patient_id hash.
    
    RULES:
    1. Same patient always in same split (prevent leakage)
    2. Temporal gap between splits
    3. No information from future splits in training
    """
    
    @staticmethod
    def assign(patient_id: str) -> str:
        hash_val = int(hashlib.md5(patient_id.encode()).hexdigest(), 16)
        normalized = hash_val / (2**128)
        
        if normalized < 0.60:
            return "train"
        elif normalized < 0.80:
            return "validation"
        elif normalized < 0.95:
            return "test"
        else:
            return "holdout"
```

### Leakage Prevention
```python
class SplitAwareRepository(BaseRepository):
    """
    Repository that enforces split boundaries.
    """
    
    async def get_training_data(self) -> List[T]:
        """Only returns train split data."""
        return await self.get_many(filters={}, split="train")
    
    async def get_validation_data(self) -> List[T]:
        """Only returns validation split data."""
        return await self.get_many(filters={}, split="validation")
    
    # NEVER expose test/holdout in normal operations
```

## V3 Table Schemas

### patient_journeys (Enhanced)
```sql
CREATE TABLE patient_journeys (
    id UUID PRIMARY KEY,
    patient_id VARCHAR(255) NOT NULL,
    brand VARCHAR(100) NOT NULL,
    journey_stage VARCHAR(50),
    -- V3 additions
    data_source VARCHAR(100),           -- Source system
    data_sources_matched TEXT[],        -- Cross-source matches
    source_stacking_flag BOOLEAN,       -- Multi-source flag
    source_timestamp TIMESTAMP,         -- Original timestamp
    ingestion_timestamp TIMESTAMP,      -- When ingested
    data_lag_hours INTEGER,             -- Staleness
    -- ML split
    split_assignment VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### ml_predictions (Enhanced)
```sql
CREATE TABLE ml_predictions (
    id UUID PRIMARY KEY,
    patient_id VARCHAR(255) NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    prediction_score FLOAT,
    -- V3 additions
    model_pr_auc FLOAT,                 -- Model PR-AUC at prediction time
    rank_metrics JSONB,                 -- recall_at_5, recall_at_10, etc.
    brier_score FLOAT,                  -- Calibration score
    -- ML split
    split_assignment VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### user_sessions (NEW - V3)
```sql
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_start TIMESTAMP NOT NULL,
    session_end TIMESTAMP,
    session_duration_seconds INTEGER,
    pages_viewed INTEGER,
    actions_taken JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- For MAU/WAU/DAU calculations
CREATE INDEX idx_user_sessions_date ON user_sessions(DATE(session_start));
```

### agent_registry (NEW - V3)
```sql
CREATE TABLE agent_registry (
    id UUID PRIMARY KEY,
    agent_name VARCHAR(100) UNIQUE NOT NULL,
    tier INTEGER NOT NULL,              -- 1-5
    intents TEXT[],                     -- Handled intent types
    description TEXT,
    config JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Seed data for 11 agents
INSERT INTO agent_registry (agent_name, tier, intents, description) VALUES
('orchestrator', 1, ARRAY['*'], 'Coordinates all agents'),
('causal_impact', 2, ARRAY['CAUSAL', 'WHAT_IF'], 'Traces causal chains'),
('gap_analyzer', 2, ARRAY['COMPARATIVE', 'EXPLORATORY'], 'ROI opportunity detection'),
-- ... etc
```

## KPI Helper Views (V3)

### v_kpi_active_users
```sql
CREATE VIEW v_kpi_active_users AS
SELECT 
    DATE_TRUNC('month', session_start) as period,
    COUNT(DISTINCT user_id) as mau,
    COUNT(DISTINCT CASE 
        WHEN session_start >= DATE_TRUNC('week', CURRENT_DATE) 
        THEN user_id 
    END) as wau,
    COUNT(DISTINCT CASE 
        WHEN DATE(session_start) = CURRENT_DATE 
        THEN user_id 
    END) as dau
FROM user_sessions
GROUP BY DATE_TRUNC('month', session_start);
```

### v_kpi_cross_source_match
```sql
CREATE VIEW v_kpi_cross_source_match AS
SELECT
    brand,
    COUNT(*) as total_patients,
    COUNT(*) FILTER (WHERE array_length(data_sources_matched, 1) > 1) as matched_patients,
    ROUND(
        COUNT(*) FILTER (WHERE array_length(data_sources_matched, 1) > 1)::NUMERIC / 
        NULLIF(COUNT(*), 0) * 100, 2
    ) as match_rate_pct
FROM patient_journeys
GROUP BY brand;
```

## Repository Implementations

### patient_journey.py
```python
class PatientJourneyRepository(SplitAwareRepository[PatientJourney]):
    table_name = "patient_journeys"
    model_class = PatientJourney
    
    async def get_by_brand(
        self,
        brand: str,
        split: Optional[str] = None
    ) -> List[PatientJourney]:
        pass
    
    async def get_cross_source_matches(
        self,
        brand: str
    ) -> List[PatientJourney]:
        """Get patients with multi-source data."""
        return await self.get_many(
            filters={
                "brand": brand,
                "source_stacking_flag": True
            }
        )
```

### agent_registry.py (NEW - V3)
```python
class AgentRegistryRepository(BaseRepository[AgentRegistry]):
    table_name = "agent_registry"
    model_class = AgentRegistry
    
    async def get_by_tier(self, tier: int) -> List[AgentRegistry]:
        pass
    
    async def get_by_intent(self, intent: str) -> List[AgentRegistry]:
        """Find agents that handle a specific intent."""
        pass
    
    async def route_intent_to_agent(self, intent: IntentType) -> AgentRegistry:
        """
        Route an intent to the appropriate agent.
        Priority: Lower tier number = higher priority
        """
        agents = await self.get_by_intent(intent.value)
        return sorted(agents, key=lambda a: a.tier)[0]
```

## Integration Contracts

### Repository Contract
```python
class RepositoryContract(Protocol[T]):
    async def get_by_id(self, id: str) -> Optional[T]: ...
    async def get_many(self, filters: Dict, limit: int) -> List[T]: ...
    async def create(self, entity: T) -> T: ...
    async def update(self, id: str, updates: Dict) -> T: ...
    async def delete(self, id: str) -> bool: ...
```

### Split Enforcement Contract
```python
# ALL repositories touching patient data MUST:
# 1. Accept split parameter
# 2. Never return test/holdout data in production
# 3. Log all cross-split access attempts
```

## Testing Requirements
- `tests/unit/test_repositories/`
- `tests/unit/test_ml_split/`
- All queries must be tested for leakage prevention
- KPI views must match expected calculations

## Handoff Format
```yaml
database_handoff:
  tables_affected: [<list>]
  new_columns: [<list>]
  migrations_needed: <bool>
  kpi_views_updated: [<list>]
  split_compliance: <verified|needs_review>
```
