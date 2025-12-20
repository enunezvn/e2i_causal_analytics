# E2I Query Processing Flow - Granular Data Handoff Diagram

## Overview

This document provides a granular breakdown of the query processing flow in the E2I Causal Analytics V4.1 system, mapping:
- **Data Handoffs**: What data moves between components
- **Processing Steps**: What transformation happens at each node
- **End States**: How results populate the dashboard

---

## Complete Flow Sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as 👤 User<br/>(Dashboard Chat)
    participant QP as 🔤 Query Processor<br/>(Layer 1: NLP)
    participant IC as 🎯 Intent Classifier
    participant EE as 📦 Entity Extractor
    participant O as 🎛️ Orchestrator<br/>(Tier 1)
    participant R as 🔀 Router
    participant T0 as 🤖 Tier 0 Agents<br/>(ML Foundation)
    participant T2 as 📊 Tier 2 Agents<br/>(Causal Analytics)
    participant CE as ⚡ Causal Engine<br/>(Layer 2)
    participant RAG as 🔍 CausalRAG
    participant DB as 🗄️ Database<br/>(28 tables)
    participant SYN as 🔗 Synthesizer
    participant VER as ✅ Verification
    participant VIZ as 📈 Viz Selector
    participant DASH as 🖥️ Dashboard

    Note over U,DASH: === PHASE 1: QUERY INGESTION ===

    U->>QP: Raw NL query string<br/>"Why did Kisqali trigger acceptance drop in Q3?"
    
    QP->>QP: Clean & normalize text
    
    par Intent Classification
        QP->>IC: Cleaned query
        IC->>IC: Classify into 5 types:<br/>CAUSAL|GAP|DRIFT|ML_SCOPE|VALIDATION
        IC-->>QP: IntentType.CAUSAL
    and Entity Extraction
        QP->>EE: Cleaned query
        EE->>EE: Fuzzy match against<br/>domain_vocabulary.yaml
        Note right of EE: NO medical NER!<br/>Only: brands, regions,<br/>KPIs, time_periods
        EE-->>QP: ExtractedEntities{<br/>brand: "Kisqali",<br/>metric: "trigger_acceptance",<br/>time: "Q3"}
    end
    
    QP->>QP: Build ParsedQuery object
    
    Note over U,DASH: === PHASE 2: ORCHESTRATION ===
    
    QP->>O: ParsedQuery{intent, entities, query}
    O->>R: Route by tier priority
    R->>R: Map intent → agent(s)
    Note right of R: Lower tier = higher priority<br/>Tier 0 > Tier 1 > ... > Tier 5
    
    R-->>O: AgentPlan[causal_impact,<br/>gap_analyzer, drift_monitor]
    
    O->>O: Multi-step execution plan
    
    Note over U,DASH: === PHASE 3: AGENT EXECUTION ===
    
    alt ML Query (needs Tier 0)
        O->>T0: ML_SCOPE intent
        T0->>T0: scope_definer → data_preparer
        
        rect rgb(236, 72, 153, 0.1)
            Note over T0: 🚦 QC GATE CHECK
            T0->>T0: Great Expectations validation
            alt QC Pass
                T0->>T0: model_selector → model_trainer
                T0->>T0: feature_analyzer (SHAP)
                T0-->>SYN: MLResult + SHAP values
            else QC Fail
                T0-->>O: status="blocked"
            end
        end
    end
    
    O->>T2: CAUSAL intent dispatch
    
    par Causal Impact Agent (5-node workflow)
        T2->>CE: Variables + constraints
        CE->>CE: 1️⃣ DAG Builder (NetworkX)
        CE->>DB: Check expert_reviews
        DB-->>CE: DAG approval status
        CE->>CE: 2️⃣ Effect Estimator (DoWhy)
        CE->>CE: ATE/CATE calculation
        
        rect rgb(239, 68, 68, 0.1)
            Note over CE: 🚦 REFUTATION GATE
            CE->>CE: 3️⃣ RefutationRunner.run_suite()
            Note right of CE: 5 tests:<br/>• placebo_treatment<br/>• random_common_cause<br/>• data_subset<br/>• bootstrap<br/>• sensitivity_e_value
            CE->>DB: INSERT causal_validations
            CE->>CE: 4️⃣ Sensitivity analysis
            alt gate_decision = "proceed"
                CE->>CE: 5️⃣ Interpretation
                CE-->>T2: CausalResult + RefutationSuite
            else gate_decision = "block"
                CE-->>T2: BLOCKED + reasons
            end
        end
        T2-->>SYN: CausalImpactOutput
        
    and RAG Context Retrieval
        T2->>RAG: Query + entities
        RAG->>RAG: Hybrid retrieval:<br/>dense + sparse + graph
        RAG->>DB: Query indexed tables
        Note right of RAG: Sources:<br/>• causal_paths<br/>• agent_activities<br/>• business_metrics<br/>• causal_validations (V4.1)
        RAG->>RAG: Cross-encoder rerank
        RAG-->>T2: Retrieved context chunks
        
    and Gap Analyzer
        T2->>T2: gap_analyzer
        T2->>DB: Query business_metrics
        T2-->>SYN: GapAnalysis{gaps, ROI}
        
    and Monitoring Agents (Tier 3)
        T2->>T2: drift_monitor
        T2->>DB: Query ml_predictions
        T2-->>SYN: DriftReport{PSI, alerts}
    end
    
    Note over U,DASH: === PHASE 4: SYNTHESIS ===
    
    SYN->>SYN: Merge multi-agent outputs
    SYN->>SYN: Deduplicate insights
    SYN->>SYN: Rank by confidence
    
    SYN->>VER: MergedResponse
    VER->>VER: Confidence scoring
    VER->>VER: Compliance check
    VER->>VER: Hallucination detection
    VER->>DB: Log agent_activities
    
    VER->>VIZ: VerifiedResponse
    VIZ->>VIZ: Rules-based chart selection
    Note right of VIZ: visualization_rules.yaml:<br/>causal → DAG + waterfall<br/>comparison → bar + heatmap<br/>trend → line + area
    
    Note over U,DASH: === PHASE 5: DASHBOARD POPULATION ===
    
    par Chat Response
        VIZ->>DASH: Streaming text + badges
        Note right of DASH: Agent badges show<br/>tier colors
    and Causal DAG
        VIZ->>DASH: DAGSpec{nodes, edges}
        DASH->>DASH: D3.js rendering
    and KPI Cards
        VIZ->>DASH: KPIData[46 metrics]
        DASH->>DASH: Render with causal insights
    and CATE Heatmap
        VIZ->>DASH: CATEMatrix{segments, effects}
        DASH->>DASH: Plotly heatmap
    and Validation Badge
        VIZ->>DASH: RefutationSuite
        Note right of DASH: V4.1: Shows<br/>proceed/review/block
    end
    
    DASH-->>U: Complete response with visualizations
```

---

## Data Handoff Reference Table

| Step | From | To | Data Object | Key Fields |
|------|------|-----|-------------|------------|
| 1 | User | Query Processor | `string` | Raw NL query |
| 2 | Query Processor | Intent Classifier | `CleanedQuery` | normalized_text, tokens |
| 3 | Query Processor | Entity Extractor | `CleanedQuery` | normalized_text, tokens |
| 4 | Intent Classifier | Query Processor | `IntentType` | enum: CAUSAL\|GAP\|DRIFT\|ML_SCOPE\|VALIDATION |
| 5 | Entity Extractor | Query Processor | `ExtractedEntities` | brands[], regions[], kpis[], time_periods[] |
| 6 | Query Processor | Orchestrator | `ParsedQuery` | intent, entities, rewritten_query, confidence |
| 7 | Router | Orchestrator | `AgentPlan` | agents[], priority_order, execution_mode |
| 8 | Orchestrator | Tier 0 Agents | `MLRequest` | scope, constraints, data_requirements |
| 9 | Data Preparer | Model Trainer | `QCResult` | status: pass\|block, failures[], baseline_metrics |
| 10 | Orchestrator | Tier 2 Agents | `CausalRequest` | variables, treatment, outcome, confounders |
| 11 | Causal Engine | Database | `ValidationRecord` | estimate_id, test_type, status, gate_decision |
| 12 | CausalRAG | Agents | `RetrievedContext` | chunks[], sources[], relevance_scores[] |
| 13 | All Agents | Synthesizer | `AgentOutput` | result_type, content, confidence, visualizations[] |
| 14 | Synthesizer | Verification | `MergedResponse` | insights[], conflicts[], citations[] |
| 15 | Verification | Viz Selector | `VerifiedResponse` | content, compliance_status, confidence_score |
| 16 | Viz Selector | Dashboard | `ChatResponse` | text, agent_badges[], stream_tokens |
| 17 | Viz Selector | Dashboard | `DAGSpec` | nodes[], edges[], layout_hints |
| 18 | Viz Selector | Dashboard | `KPIData` | metric_id, value, trend, causal_insight |
| 19 | Viz Selector | Dashboard | `CATEMatrix` | segments[], time_periods[], effects[][] |
| 20 | Viz Selector | Dashboard | `ValidationBadge` | gate_decision, test_results[], confidence_score |

---

## Processing Step Details

### Layer 1: NLP Processing

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUERY PROCESSOR                              │
├─────────────────────────────────────────────────────────────────┤
│  Input: "Why did Kisqali trigger acceptance drop in Q3?"        │
│                                                                  │
│  Step 1: Normalization                                          │
│    → lowercase, remove punctuation, expand contractions         │
│                                                                  │
│  Step 2: Intent Classification (5 types)                        │
│    → Pattern: "why did X" + metric change = CAUSAL              │
│    → Output: IntentType.CAUSAL                                  │
│                                                                  │
│  Step 3: Entity Extraction (domain_vocabulary.yaml)             │
│    → "Kisqali" fuzzy match → brand: "Kisqali" (score: 1.0)      │
│    → "trigger acceptance" → metric: "trigger_acceptance"         │
│    → "Q3" → time_period: "2024-Q3"                              │
│                                                                  │
│  Step 4: Query Rewriting (for RAG optimization)                 │
│    → "causal factors trigger acceptance decline Kisqali Q3"     │
│                                                                  │
│  Output: ParsedQuery{                                           │
│    intent: CAUSAL,                                              │
│    entities: {brand, metric, time},                             │
│    rewritten: "causal factors...",                              │
│    confidence: 0.92                                             │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 2: Causal Engine (5-Node Workflow)

```
┌─────────────────────────────────────────────────────────────────┐
│                 CAUSAL IMPACT AGENT WORKFLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Node 1: GraphBuilder                                           │
│    → Input: variables from ParsedQuery                          │
│    → Process: NetworkX DAG construction                         │
│    → Check: expert_reviews table for DAG approval               │
│    → Output: CausalGraph{nodes, edges, confounders}             │
│                                                                  │
│  Node 2: Estimation                                             │
│    → Input: CausalGraph + treatment + outcome                   │
│    → Process: DoWhy/EconML effect estimation                    │
│    → Output: EffectEstimate{ATE, CI, p_value}                   │
│                                                                  │
│  Node 3: Refutation (V4.1) 🚦 GATE                              │
│    → Input: EffectEstimate                                      │
│    → Process: RefutationRunner.run_suite()                      │
│      ├── placebo_treatment test                                 │
│      ├── random_common_cause test                               │
│      ├── data_subset test                                       │
│      ├── bootstrap test                                         │
│      └── sensitivity_e_value test                               │
│    → Persist: INSERT INTO causal_validations                    │
│    → Output: RefutationSuite{tests[], gate_decision}            │
│    → Gate: if gate_decision == "block" → STOP                   │
│                                                                  │
│  Node 4: Sensitivity                                            │
│    → Input: EffectEstimate + RefutationSuite                    │
│    → Process: Sensitivity analysis for unobserved confounders   │
│    → Output: SensitivityResult{e_value, robustness_score}       │
│                                                                  │
│  Node 5: Interpretation                                         │
│    → Input: All previous outputs                                │
│    → Process: LLM-based narrative generation                    │
│    → Output: CausalResult{effect, explanation, confidence}      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Dashboard End States

```
┌─────────────────────────────────────────────────────────────────┐
│                    DASHBOARD COMPONENTS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Chat Response Panel                                         │
│     ├── Streaming text (WebSocket)                              │
│     ├── Agent badges (tier-colored)                             │
│     └── Inline citations to sources                             │
│                                                                  │
│  2. Causal DAG Visualization (D3.js)                            │
│     ├── Nodes: variables (treatment, outcome, confounders)      │
│     ├── Edges: causal relationships with strength               │
│     └── Interactive: click to see effect details                │
│                                                                  │
│  3. KPI Cards (46 metrics)                                      │
│     ├── Value + trend indicator                                 │
│     ├── Sparkline chart (Chart.js)                              │
│     └── Causal insight badge ("↑ caused by X")                  │
│                                                                  │
│  4. CATE Heatmap (Plotly)                                       │
│     ├── X-axis: Time periods                                    │
│     ├── Y-axis: HCP segments                                    │
│     └── Color: Treatment effect magnitude                       │
│                                                                  │
│  5. Resource Allocation Sankey (Plotly)                         │
│     ├── Left: Current budget allocation                         │
│     ├── Right: Optimal allocation                               │
│     └── Flows: Budget movement recommendations                  │
│                                                                  │
│  6. Validation Badge (V4.1)                                     │
│     ├── Status: proceed | review | block                        │
│     ├── Tests passed: 4/5 ✓                                     │
│     └── Confidence score: 87%                                   │
│                                                                  │
│  7. Health Radar Chart (Plotly)                                 │
│     ├── 8 dimensions: coverage, AUC, fairness, etc.             │
│     ├── Current state (solid)                                   │
│     └── Target state (dashed)                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Critical Flow Constraints

| Constraint | Location | Behavior |
|------------|----------|----------|
| **NO Medical NER** | Entity Extractor | Only extracts from domain_vocabulary.yaml. Never uses scispaCy, BioBERT. |
| **QC Gate Blocking** | Data Preparer → Model Trainer | Training blocked with status="blocked" if Great Expectations validation fails. |
| **Refutation Required** | Causal Engine Node 3 | All causal effects must pass 5 DoWhy tests. Results persisted to causal_validations. |
| **ML Split Enforcement** | All data access | Same patient always in same split. Test/holdout never exposed in production. |
| **Operational Data Only** | RAG Retrieval | Only indexes: causal_paths, agent_activities, business_metrics, triggers. Never: clinical trials, medical literature. |
| **Tier Priority** | Router | Lower tier = higher priority. Tier 0 requests handled before Tier 5. |

---

## New in V4.1: Validation Infrastructure

### New Tables
- `causal_validations`: Stores refutation test results with gate decisions
- `expert_reviews`: Tracks DAG approval by domain experts

### New ENUMs
- `refutation_test_types`: placebo_treatment, random_common_cause, data_subset, bootstrap, sensitivity_e_value
- `validation_statuses`: passed, failed, warning, skipped
- `gate_decisions`: proceed, review, block

### New Dashboard Component
- **Validation Badge**: Shows refutation status with proceed/review/block indicator and confidence score

---

*Generated from E2I Causal Analytics V4.1 Architecture Documentation*
