# E2I Gap Analysis: Process Mapping vs Dashboard Mock Functionality

## Executive Summary

After detailed review of the orchestration documentation, project structure, domain vocabulary, and dashboard mock, I've identified **14 critical gaps** and **8 moderate gaps** that would prevent full reproduction of the dashboard functionality. These fall into four categories:

1. **Missing Data Flow Mappings** (5 gaps)
2. **Unmapped Agent Outputs → Dashboard Components** (6 gaps)
3. **Schema/Data Source Gaps** (7 gaps)
4. **UI/UX Flow Gaps** (4 gaps)

---

## 🔴 CRITICAL GAPS (Blocks Dashboard Reproduction)

### Gap 1: Filter Controls → Query Pipeline Not Mapped

**Dashboard Mock Shows:**
```html
<select id="brandFilter">
    <option value="all">All Brands</option>
    <option value="remi">Remibrutinib</option>
    <option value="fabhalta">Fabhalta</option>
    <option value="kisqali">Kisqali</option>
</select>
<select id="regionFilter">...</select>
<input type="date" id="startDate">
```

**Missing from Process Mapping:**
- How filter selections transform into NLP ParsedQuery entities
- Whether filters bypass NLP entirely (direct SQL injection)
- Filter state management in Redux/Zustand
- Filter → Agent dispatch routing rules

**Recommended Addition:**
```
Filter Controls → FilterState{brand, region, dateRange}
    ↓
    if user_typed_query:
        FilterState → NLP Entity Injection (pre-populate ExtractedEntities)
    else:
        FilterState → Direct SQL WHERE clause generation
    ↓
Dashboard Refresh
```

---

### Gap 2: Dashboard Tab Routing Not Mapped

**Dashboard Mock Has 10 Tabs:**
1. 🤖 AI Agent Insights
2. 📊 Overview
3. 🗃️ WS1: Data Quality
4. 🤖 WS1: ML & Model
5. 🎯 WS2: Triggers
6. 📈 WS3: Impact
7. ⚡ Causal Analysis
8. 📖 KPI Dictionary
9. 🏷️ Status Legend
10. 📚 Agentic Methodology

**Missing from Process Mapping:**
- Which agents serve which tabs
- Tab switch → data refresh flow
- Tab-specific cache strategies
- Pre-fetching logic for adjacent tabs

**Recommended Addition:**
```yaml
tab_agent_mapping:
  ai_agent_insights: [orchestrator, gap_analyzer, causal_impact, drift_monitor, experiment_designer, heterogeneous_optimizer, health_score]
  overview: [orchestrator, health_score]
  ws1_data: [data_preparer, observability_connector]
  ws1_ml: [model_trainer, feature_analyzer, model_deployer]
  ws2_triggers: [prediction_synthesizer, drift_monitor]
  ws3_impact: [causal_impact, gap_analyzer, resource_optimizer]
  causal: [causal_impact, heterogeneous_optimizer, explainer]
  kpi_dictionary: null  # Static content
  legend: null  # Static content
  methodology: null  # Static content
```

---

### Gap 3: Agent Output → Visualization Component Mapping Not Specified

**Dashboard Mock Shows These Visualizations:**
| Visualization | Data Source | Agent Output |
|--------------|-------------|--------------|
| CATE Heatmap (Plotly) | Heterogeneous effects by segment × time | ? |
| Resource Sankey (Plotly) | Budget allocation flows | ? |
| Health Radar (Plotly) | 8 health dimensions | ? |
| Causal Chain (CSS) | Node → Effect → Node chains | ? |
| PSI Timeline | Drift predictions | ? |
| Geographic Coverage (Chart.js) | Regional percentages | ? |
| Model Performance (Chart.js) | AUC, Precision trends | ? |

**Missing from Process Mapping:**
- Exact schema of agent output → visualization config
- Chart.js vs Plotly selection rules
- Visualization configuration generator logic

**Recommended Addition to `visualization_rules.yaml`:**
```yaml
agent_viz_mapping:
  heterogeneous_optimizer:
    output: CATEAnalysis
    viz_type: heatmap
    library: plotly
    config:
      x_axis: time_periods
      y_axis: hcp_segments
      values: treatment_effects
      colorscale: diverging
  
  resource_optimizer:
    output: AllocationPlan
    viz_type: sankey
    library: plotly
    config:
      nodes: [current_budget, optimal_budget, categories]
      links: reallocation_flows
  
  health_score:
    output: HealthScore
    viz_type: radar
    library: plotly
    config:
      dimensions: [data_coverage, model_auc, trigger_accept, business_impact, 
                   data_freshness, fairness, system_uptime, user_adoption]
      
  causal_impact:
    output: CausalChain
    viz_type: chain
    library: css_custom
    config:
      nodes: variables
      edges: effects
      layout: horizontal_flow
```

---

### Gap 4: 8 Dashboard Agents vs 18 Architecture Agents

**Dashboard Mock Shows 8 Active Agents:**
1. Orchestrator
2. Causal Impact Agent
3. Gap Analyzer Agent
4. Drift Monitor Agent
5. Experiment Designer Agent
6. Heterogeneous Optimizer Agent
7. Health Score Agent
8. Resource Optimizer Agent

**Architecture Has 18 Agents (Missing from Dashboard):**
- Tier 0: scope_definer, data_preparer, model_selector, model_trainer, feature_analyzer, model_deployer, observability_connector (7 agents)
- Tier 5: explainer, feedback_learner (2 agents)
- prediction_synthesizer (shown indirectly)

**Impact:**
- Dashboard doesn't show ML Foundation agent activities
- No visibility into Tier 0 operations
- Self-improvement loop (feedback_learner) not surfaced

**Recommended Addition:**
Add "🔧 ML Foundation" tab showing:
- Scope definition status
- QC gate status (pass/block)
- Model training progress
- Feature importance (SHAP)
- Deployment status
- Observability metrics

---

### Gap 5: Validation Badge UI Missing from Dashboard Mock

**V4.1 Architecture Specifies:**
```
Validation Badge (V4.1)
├── Status: proceed | review | block
├── Tests passed: 4/5 ✓
└── Confidence score: 87%
```

**Dashboard Mock Does NOT Show:**
- Any validation badge component
- Refutation test results
- Gate decision indicators
- Expert review status

**Impact:**
- Core V4.1 feature not visualized
- Causal estimates shown without validation context
- Users can't see which estimates are blocked/pending review

**Recommended Addition:**
Each causal insight card should include:
```html
<div class="validation-badge">
    <span class="gate-status gate-proceed">✓ PROCEED</span>
    <span class="tests-passed">4/5 tests passed</span>
    <span class="confidence">87% confidence</span>
    <span class="e-value">E-value: 2.3</span>
</div>
```

---

### Gap 6: Chat Interface Not in Dashboard Mock

**Architecture Specifies:**
- Natural language query input
- Streaming text response
- Agent badges with tier colors
- WebSocket communication

**Dashboard Mock Shows:**
- Filter dropdowns only
- No text input field
- No chat history
- No streaming indicators

**Impact:**
- Core NLP capability not testable
- Entity extraction not exercised
- Intent classification not demonstrated

**Recommended Addition:**
Add persistent chat sidebar or modal:
```
┌────────────────────────────────────┐
│ 💬 Ask E2I                         │
├────────────────────────────────────┤
│ User: Why did Kisqali trigger     │
│       acceptance drop in Q3?       │
│                                    │
│ [🔵 Causal Impact] [🟢 Gap Analyzer]│
│ Agent: Analysis shows 3 causal    │
│ factors contributing to the drop...│
│ ███████████░░░░░░ Streaming...    │
├────────────────────────────────────┤
│ [Type your question...]      [Ask] │
└────────────────────────────────────┘
```

---

## 🟡 MODERATE GAPS (Degrades Functionality)

### Gap 7: Data Sources Not in Schema

**Dashboard Mock References These Data Sources:**
- IQVIA APLD
- IQVIA LAAD
- HealthVerity
- Komodo
- Veeva OCE

**Schema Has:**
- `data_source_tracking` table (generic)

**Missing:**
- Source-specific quality metrics
- Source-to-table mapping
- Cross-source match rate tracking

**Recommendation:**
Add `data_sources` reference table:
```sql
CREATE TABLE data_sources (
    source_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,  -- 'IQVIA APLD', 'HealthVerity', etc.
    source_type TEXT,  -- 'claims', 'lab', 'emr', 'crm'
    coverage_percent DECIMAL(5,2),
    completeness_score DECIMAL(5,4),
    freshness_days INTEGER,
    match_rate DECIMAL(5,4)
);
```

---

### Gap 8: Brand-Specific KPI Calculations Not Documented

**Dashboard Mock Shows Brand KPIs:**
| Brand | KPIs Shown |
|-------|-----------|
| Remibrutinib | AH Uncontrolled Patients Identified, Intent-to-Prescribe |
| Fabhalta | PNH Patients Identified, Specialist Engagement |
| Kisqali | CDK Naive Patient Identification, Oncology Territory Coverage |

**Missing from Process Mapping:**
- Brand-specific KPI formulas
- Brand → therapeutic area mapping
- Brand-specific patient identification logic

**Recommendation:**
Add to `kpi_definitions.yaml`:
```yaml
brand_kpis:
  Remibrutinib:
    therapeutic_area: CSU  # Chronic Spontaneous Urticaria
    kpis:
      - name: ah_uncontrolled_identified
        formula: "(patients_flagged_ah_uncontrolled / total_ah_patients) * 100"
        target: 65
      - name: intent_to_prescribe_delta
        formula: "itp_current - itp_baseline"
        target: 10  # pp
        
  Fabhalta:
    therapeutic_area: PNH  # Paroxysmal Nocturnal Hemoglobinuria
    kpis:
      - name: pnh_patients_identified
        formula: "(patients_flagged_pnh / total_pnh_eligible) * 100"
        target: 70
        
  Kisqali:
    therapeutic_area: Oncology
    kpis:
      - name: cdk_naive_identified
        formula: "(patients_cdk_naive / total_hr_positive) * 100"
        target: 55
```

---

### Gap 9: ROI Calculation Methods Not Specified

**Dashboard Mock Shows:**
- ROI: 8.2x, 6.4x, 4.1x, 1.3x for different actions
- Annual values: $10.5M, $8M, $5M, etc.

**Missing from Process Mapping:**
- ROI formula specification
- Cost model inputs
- Value attribution methodology
- Confidence intervals on ROI

**Recommendation:**
Add to gap_analyzer agent specification:
```yaml
roi_calculation:
  formula: "(incremental_value - implementation_cost) / implementation_cost"
  
  value_drivers:
    trx_lift_value: 850  # $ per incremental TRx
    patient_identification_value: 1200
    action_rate_value: 45  # $ per pp improvement
    
  cost_inputs:
    engineering_day_cost: 2500
    data_acquisition_cost: varies
    training_cost: 15000
    
  confidence_method: bootstrap_sampling
  n_simulations: 1000
```

---

### Gap 10: Time Granularity Not Specified

**Dashboard Mock Shows:**
- Weekly data points (Week 1, Week 2, etc.)
- Monthly trends
- Quarterly comparisons
- 7-day forecasts

**Missing from Process Mapping:**
- Aggregation rules by time granularity
- Time zone handling
- Business calendar vs calendar days
- Fiscal quarter mapping

---

### Gap 11: Alert/Notification System Not Mapped

**Dashboard Mock Shows:**
```html
<div class="alert-banner" id="alertBanner">
    ⚠️ Critical Alert: Feature Drift PSI exceeds threshold (0.28 > 0.25)
</div>
```

**Missing from Process Mapping:**
- Alert generation triggers
- Alert priority levels
- Notification delivery (in-app, email, Slack)
- Alert acknowledgment flow

---

### Gap 12: Confidence Indicator Logic Not Specified

**Dashboard Mock Shows:**
- "Confidence: 92% (based on 6-week trend)"
- "Confidence: 78% (heterogeneous effects detected)"

**Missing from Process Mapping:**
- Confidence calculation formula
- Confidence → display threshold
- Heterogeneous effects detection → confidence reduction rules

---

### Gap 13: Experiment Status Workflow Not Mapped

**Dashboard Mock Shows:**
- PROPOSED status
- ACTIVE - Day 4/14 status
- Interim results with p-values

**Missing from Process Mapping:**
- Experiment lifecycle states
- Status transition triggers
- Interim analysis rules
- Early stopping criteria

---

### Gap 14: D3.js vs Plotly/Chart.js Mismatch

**Architecture Specifies:**
- "D3.js causal graph viz"

**Dashboard Mock Uses:**
- Chart.js for basic charts
- Plotly for Sankey, heatmaps, radar
- CSS for causal chains

**Recommendation:**
Clarify visualization library selection:
```yaml
visualization_libraries:
  primary: Chart.js  # Simple bar, line, doughnut
  advanced: Plotly   # Sankey, heatmap, radar, box plots
  interactive: D3.js # Causal DAG (force-directed), custom interactions
  static: CSS        # Causal chain flows, badges
```

---

## 📊 Gap Summary Matrix

| Gap ID | Category | Severity | Blocks Reproduction? | Effort to Fix |
|--------|----------|----------|---------------------|---------------|
| 1 | Data Flow | 🔴 Critical | Yes - filters broken | Medium |
| 2 | Data Flow | 🔴 Critical | Yes - navigation broken | Low |
| 3 | Data Flow | 🔴 Critical | Yes - no visualizations | High |
| 4 | Agent Coverage | 🔴 Critical | Partial - missing 10 agents | Medium |
| 5 | UI Component | 🔴 Critical | Yes - V4.1 not visible | Medium |
| 6 | UI Component | 🔴 Critical | Yes - no NLP testing | High |
| 7 | Schema | 🟡 Moderate | No - degraded quality | Low |
| 8 | KPI Logic | 🟡 Moderate | No - inaccurate metrics | Medium |
| 9 | Business Logic | 🟡 Moderate | No - unverified ROI | Medium |
| 10 | Data Model | 🟡 Moderate | No - inconsistent time | Low |
| 11 | Feature | 🟡 Moderate | No - no alerts | Medium |
| 12 | Business Logic | 🟡 Moderate | No - unclear confidence | Low |
| 13 | Feature | 🟡 Moderate | No - limited experiments | Medium |
| 14 | Tech Stack | 🟡 Moderate | No - wrong library | Low |

---

## Recommended Priority Order

1. **Gap 3** - Agent → Visualization mapping (blocks all charts)
2. **Gap 6** - Chat interface (core NLP feature)
3. **Gap 5** - Validation badges (V4.1 differentiator)
4. **Gap 1** - Filter → Query flow (basic interactivity)
5. **Gap 2** - Tab routing (navigation)
6. **Gap 4** - Missing agents (complete architecture)
7. **Gap 8** - Brand KPIs (accuracy)
8. **Gap 9** - ROI calculations (credibility)

---

## Files That Need Updates

| File | Gaps Addressed |
|------|----------------|
| `visualization_rules.yaml` | 3, 14 |
| `agent_config.yaml` | 2, 4 |
| `kpi_definitions.yaml` | 8 |
| `thresholds.yaml` | 11, 12 |
| `E2I_Causal_Dashboard_V3.html` | 5, 6 |
| `010_causal_validation_tables.sql` | 5 |
| `domain_vocabulary.yaml` | 7 |
| New: `experiment_lifecycle.yaml` | 13 |
| New: `roi_methodology.md` | 9 |
| Updated: `e2i_query_flow_documentation.md` | 1, 2 |

---

*Gap Analysis v1.0 | E2I Causal Analytics V4.1 | December 2025*
