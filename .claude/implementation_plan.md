# E2I Gap Resolution - Implementation Plan

**Date:** 2025-12-15
**Target:** Partial MVP Implementation (Configuration & Infrastructure)
**Scope:** Address critical configuration gaps without full UI implementation

---

## Implementation Philosophy

Since this is **not a complete implementation**, we will focus on:
- ✅ Configuration files (YAML, SQL schemas)
- ✅ Documentation (methodology, specifications)
- ✅ Backend infrastructure definitions
- ❌ Full frontend React components
- ❌ Backend Python agent implementations
- ❌ Complete WebSocket/API layer

**Goal:** Create a production-ready architecture blueprint that can be implemented later.

---

## Current Status (from TODO)

**Completed (6/14):**
- ✅ Gap 3: Agent → Viz Mapping (visualization_rules_v4_1.yaml)
- ✅ Gap 4: Tier 0 UI Visibility (dashboard mock updated)
- ✅ Gap 5: Validation Badges (dashboard mock updated)
- ✅ Gap 6: Chat Interface (dashboard mock updated)
- ✅ Gap 8: Brand KPIs (kpi_definitions.yaml exists)
- ✅ Gap 14: Viz Library Clarification (visualization_rules_v4_1.yaml)

**To Implement (8/14):**
- 🔴 Gap 1: Filter → Query Flow
- 🔴 Gap 2: Tab → Agent Routing
- 🟡 Gap 7: Data Sources Schema
- 🟡 Gap 9: ROI Calculation
- 🟡 Gap 10: Time Granularity
- 🟡 Gap 11: Alert System
- 🟡 Gap 12: Confidence Logic
- 🟡 Gap 13: Experiment Lifecycle

---

## Phase 1: Critical Configuration (Gaps 1, 2)

### 1.1 Gap 1: Filter → Query Flow Mapping

**Objective:** Define how dashboard filters map to query pipeline

**Deliverables:**
1. `config/filter_mapping.yaml` - Filter state to entity mapping
2. Update `domain_vocabulary_v3.1.0.yaml` - Add filter injection rules
3. `docs/filter_query_flow.md` - Technical specification

**Configuration Structure:**
```yaml
filter_mapping:
  filter_to_entity:
    brand:
      entity_type: brand
      vocabulary_key: brands
      sql_column: patient_journeys.brand
    region:
      entity_type: region
      vocabulary_key: regions
      sql_column: hcp_profiles.region
    date_range:
      entity_type: temporal_range
      sql_columns:
        - start: treatment_events.event_date >= ?
        - end: treatment_events.event_date <= ?

  query_modes:
    with_nlp: inject_into_extracted_entities
    without_nlp: generate_sql_where_clause
```

**Estimated Time:** 2 hours

---

### 1.2 Gap 2: Tab → Agent Routing Configuration

**Objective:** Map each dashboard tab to responsible agents

**Deliverables:**
1. Update `agent_config.yaml` - Add tab_routing section
2. `config/tab_cache_strategies.yaml` - Cache TTLs and refresh logic

**Configuration Structure:**
```yaml
# Add to agent_config.yaml
tab_routing:
  ai_agent_insights:
    primary_agents: [orchestrator, gap_analyzer, causal_impact]
    secondary_agents: [drift_monitor, experiment_designer, heterogeneous_optimizer, health_score]
    cache_ttl: 300  # 5 minutes
    prefetch: [causal_analysis]

  overview:
    primary_agents: [orchestrator, health_score]
    cache_ttl: 600  # 10 minutes

  ws1_data_quality:
    primary_agents: [data_preparer, observability_connector]
    cache_ttl: 1800  # 30 minutes
    refresh_on_data_update: true

  ws1_ml_model:
    primary_agents: [model_trainer, feature_analyzer, model_deployer]
    cache_ttl: 900  # 15 minutes

  ws2_triggers:
    primary_agents: [prediction_synthesizer, drift_monitor]
    cache_ttl: 300

  ws3_impact:
    primary_agents: [causal_impact, gap_analyzer, resource_optimizer]
    cache_ttl: 600

  causal_analysis:
    primary_agents: [causal_impact, heterogeneous_optimizer, explainer]
    cache_ttl: 900
    validation_required: true

  kpi_dictionary:
    static: true
    cache_ttl: 86400  # 24 hours

  status_legend:
    static: true
    cache_ttl: 86400

  methodology:
    static: true
    cache_ttl: 86400
```

**Estimated Time:** 1.5 hours

---

## Phase 2: Data & Schema (Gap 7)

### 2.1 Gap 7: Data Sources Reference Table

**Objective:** Formalize data source tracking with quality metrics

**Deliverables:**
1. `e2i_mlops/012_data_sources.sql` - New migration
2. Seed data for IQVIA, HealthVerity, Komodo, Veeva

**Schema:**
```sql
CREATE TABLE data_sources (
    source_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('claims', 'lab', 'emr', 'crm', 'specialty')),
    vendor TEXT,
    coverage_percent DECIMAL(5,2),
    completeness_score DECIMAL(5,4),
    freshness_days INTEGER,
    match_rate DECIMAL(5,4),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed data
INSERT INTO data_sources (source_id, source_name, source_type, vendor, coverage_percent, completeness_score, freshness_days, match_rate) VALUES
('iqvia_apld', 'IQVIA APLD', 'claims', 'IQVIA', 92.5, 0.9450, 7, 0.8900),
('iqvia_laad', 'IQVIA LAAD', 'lab', 'IQVIA', 85.0, 0.9200, 14, 0.8500),
('healthverity', 'HealthVerity', 'claims', 'HealthVerity', 88.0, 0.9350, 10, 0.8700),
('komodo', 'Komodo Health', 'emr', 'Komodo', 78.0, 0.8900, 21, 0.8200),
('veeva_oce', 'Veeva OCE', 'crm', 'Veeva', 95.0, 0.9800, 1, 0.9500);

-- Update data_source_tracking to reference this table
ALTER TABLE data_source_tracking
ADD COLUMN source_id TEXT REFERENCES data_sources(source_id);
```

**Estimated Time:** 1 hour

---

## Phase 3: Business Logic Documentation (Gaps 9, 12)

### 3.1 Gap 9: ROI Calculation Methodology

**Objective:** Document ROI calculation formula and confidence intervals

**Deliverables:**
1. `docs/roi_methodology.md` - Complete methodology
2. Update `agent_config.yaml` (gap_analyzer section)

**Content Structure:**
```markdown
# ROI Calculation Methodology

## Formula
ROI = (Incremental Value - Implementation Cost) / Implementation Cost

## Value Drivers
- TRx Lift Value: $850 per incremental TRx
- Patient Identification Value: $1,200 per identified patient
- Action Rate Value: $45 per percentage point improvement
- Intent-to-Prescribe Value: $320 per percentage point gain

## Cost Inputs
- Engineering Day Cost: $2,500
- Data Acquisition Cost: Variable by source ($15K-$50K annual)
- Training Cost: $15,000 per initiative
- Deployment Cost: $8,000 per quarter

## Confidence Intervals
- Method: Bootstrap sampling with 1,000 simulations
- Reported: 95% CI
- Sensitivity Analysis: E-value calculation for unmeasured confounding

## Example Calculation
[Detailed example with real numbers]
```

**Estimated Time:** 2 hours

---

### 3.2 Gap 12: Confidence Calculation Logic

**Objective:** Specify confidence score calculation

**Deliverables:**
1. `config/confidence_logic.yaml` - Formula and reduction rules
2. `docs/confidence_methodology.md` - Technical documentation

**Configuration:**
```yaml
confidence_calculation:
  base_formula: "min(1.0, sample_size_factor * temporal_stability * effect_consistency)"

  sample_size_factor:
    formula: "1 - exp(-n / 100)"  # Asymptotic approach to 1.0
    min_samples: 30
    confidence_at_30: 0.26
    confidence_at_100: 0.63
    confidence_at_500: 0.99

  temporal_stability:
    formula: "1 - coefficient_of_variation(weekly_effects)"
    threshold_high: 0.15  # CV < 15% → high stability
    threshold_medium: 0.30

  effect_consistency:
    formula: "1 - abs(subgroup_effect_variance) / overall_effect"
    heterogeneity_penalty: 0.2  # -20% if heterogeneous effects detected

  reduction_rules:
    heterogeneous_effects_detected: -0.20
    missing_covariates: -0.15
    failed_sensitivity_test: -0.30
    short_time_window: -0.10  # < 4 weeks

  display_threshold: 0.50  # Only show if confidence >= 50%
```

**Estimated Time:** 1.5 hours

---

## Phase 4: Operational Specifications (Gaps 10, 11, 13)

### 4.1 Gap 10: Time Granularity Specification

**Objective:** Define temporal aggregation rules

**Deliverables:**
1. Add to `domain_vocabulary_v3.1.0.yaml`

**Configuration:**
```yaml
temporal:
  granularities:
    - daily
    - weekly
    - monthly
    - quarterly

  fiscal_calendar:
    fiscal_year_start: "02-01"  # Feb 1
    quarters:
      Q1: ["02-01", "04-30"]
      Q2: ["05-01", "07-31"]
      Q3: ["08-01", "10-31"]
      Q4: ["11-01", "01-31"]

  business_calendar:
    timezone: "America/New_York"
    exclude_weekends: false
    exclude_holidays: false

  aggregation_rules:
    weekly:
      start_day: "monday"
      label_format: "Week {week_num}"
    monthly:
      label_format: "{month_abbr} {year}"
    quarterly:
      label_format: "Q{quarter} {fiscal_year}"

  default_lookback:
    daily: 30
    weekly: 12
    monthly: 6
    quarterly: 4
```

**Estimated Time:** 1 hour

---

### 4.2 Gap 11: Alert System Configuration

**Objective:** Define alert triggers and delivery

**Deliverables:**
1. `config/alert_config.yaml`

**Configuration:**
```yaml
alert_system:
  priority_levels:
    critical:
      color: "#fc8181"
      icon: "⚠️"
      delivery: [in_app, email, slack]
      acknowledgment_required: true
      escalation_time: 3600  # 1 hour

    warning:
      color: "#f6ad55"
      icon: "⚡"
      delivery: [in_app, email]
      acknowledgment_required: false

    info:
      color: "#63b3ed"
      icon: "ℹ️"
      delivery: [in_app]
      acknowledgment_required: false

  triggers:
    drift_detected:
      priority: critical
      condition: "psi_score > 0.25"
      message: "Feature Drift PSI exceeds threshold ({psi_score} > 0.25)"
      agent: drift_monitor

    model_performance_degradation:
      priority: warning
      condition: "auc_drop > 0.05"
      message: "Model AUC dropped by {auc_drop_pct}% (now {current_auc})"
      agent: model_deployer

    causal_validation_blocked:
      priority: critical
      condition: "gate_decision == 'block'"
      message: "Causal validation blocked: {failed_tests} tests failed"
      agent: causal_impact

    experiment_significance:
      priority: info
      condition: "p_value < 0.05 AND active_days >= min_duration"
      message: "Experiment {exp_id} reached significance (p={p_value})"
      agent: experiment_designer

  notification_config:
    email:
      from: "alerts@e2i-analytics.com"
      smtp_server: "smtp.sendgrid.net"

    slack:
      webhook_url: "${SLACK_WEBHOOK_URL}"
      channel: "#e2i-alerts"
      mention_on_critical: "@channel"
```

**Estimated Time:** 2 hours

---

### 4.3 Gap 13: Experiment Lifecycle Workflow

**Objective:** Define experiment status transitions

**Deliverables:**
1. `config/experiment_lifecycle.yaml`

**Configuration:**
```yaml
experiment_lifecycle:
  states:
    - proposed
    - approved
    - active
    - completed
    - archived
    - cancelled

  transitions:
    proposed_to_approved:
      requires:
        - power_analysis_completed
        - stakeholder_approval
        - resource_allocation
      validations:
        - min_sample_size: 100
        - min_duration_days: 14
        - valid_success_metrics: true

    approved_to_active:
      requires:
        - implementation_ready
        - data_pipeline_validated
      auto_trigger: true

    active_to_completed:
      requires:
        - min_duration_reached
        - sample_size_achieved
      early_stopping_allowed: true
      early_stopping_rules:
        - p_value < 0.001 AND days >= 7
        - futility_detected AND days >= min_duration / 2

    completed_to_archived:
      auto_trigger_after_days: 90

  interim_analysis:
    schedule: [0.25, 0.50, 0.75]  # Fraction of planned duration
    alpha_spending:
      method: "obrien_fleming"
      overall_alpha: 0.05

  early_stopping_criteria:
    efficacy:
      method: "alpha_spending"
      min_duration_fraction: 0.5

    futility:
      method: "conditional_power"
      threshold: 0.20
      min_duration_fraction: 0.5

  status_display:
    proposed: {color: "#718096", icon: "📝"}
    approved: {color: "#48bb78", icon: "✅"}
    active: {color: "#4299e1", icon: "▶️"}
    completed: {color: "#9f7aea", icon: "🏁"}
    archived: {color: "#a0aec0", icon: "📦"}
    cancelled: {color: "#f56565", icon: "❌"}
```

**Estimated Time:** 2 hours

---

## Phase 5: Integration & Documentation

### 5.1 Update Core Documentation

**Tasks:**
1. Update `docs/e2i_gap_todo.md` - Mark all gaps as complete
2. Update `.claude/codebase_index.md` - Add new configuration files
3. Create `docs/configuration_guide.md` - How to use all config files

**Estimated Time:** 1.5 hours

---

## Summary

**Total Deliverables:**
1. `config/filter_mapping.yaml` (NEW)
2. `docs/filter_query_flow.md` (NEW)
3. `agent_config.yaml` (UPDATE - add tab_routing)
4. `config/tab_cache_strategies.yaml` (NEW)
5. `e2i_mlops/012_data_sources.sql` (NEW)
6. `docs/roi_methodology.md` (NEW)
7. `agent_config.yaml` (UPDATE - gap_analyzer ROI config)
8. `config/confidence_logic.yaml` (NEW)
9. `docs/confidence_methodology.md` (NEW)
10. `domain_vocabulary_v3.1.0.yaml` (UPDATE - temporal section)
11. `config/alert_config.yaml` (NEW)
12. `config/experiment_lifecycle.yaml` (NEW)
13. `docs/configuration_guide.md` (NEW)
14. Documentation updates (gaps, index)

**Total Estimated Time:** ~15-17 hours

**Success Criteria:**
- ✅ All critical gaps (1, 2) addressed with configuration
- ✅ All moderate gaps (7, 9, 10, 11, 12, 13) documented
- ✅ Configuration files are production-ready
- ✅ Documentation is complete and actionable
- ✅ Architecture blueprint is implementation-ready

---

**Next Steps:**
1. Confirm priority with user
2. Begin Phase 1: Critical Configuration
3. Iterate through phases sequentially
4. Test configurations for YAML validity
5. Update progress tracker

