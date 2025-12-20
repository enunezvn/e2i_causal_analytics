# Product Features & Specifications

## Core Feature Set

### 1. Natural Language Query Interface

**Description**: Users ask questions in plain English; the system routes to appropriate AI agents and returns causally-grounded answers with visualizations.

**Key Capabilities**:
- Typo-tolerant NLP with fastText subword embeddings
- Domain-specific vocabulary (46+ KPIs, 100+ pharma terms)
- Multi-faceted query decomposition for complex questions
- Conversational follow-up with context retention

**User Experience**:
```
User: "Why did Kisqali TRx drop 15% in Q3?"
System: [Orchestrator routes to Causal Impact Agent]
Output:
- Primary driver: 25% decrease in HCP rep visits (-12% TRx impact)
- Secondary driver: Competitor launch in same segment (-5% TRx impact)
- Tertiary driver: Payer coverage changes (-3% TRx impact)
[Visual causal graph showing relationships]
[Confidence intervals and validation results]
```

**Acceptance Criteria**:
- 95% query understanding accuracy for pharma domain
- <2s end-to-end latency for simple queries
- <10s for complex multi-agent queries
- Graceful clarification requests for ambiguous queries

---

### 2. 18-Agent Multi-Agent System

**Description**: Specialized AI agents organized in 6 tiers, each with specific expertise, collaborating to deliver comprehensive analytics.

**Agent Architecture**:

**Tier 0: ML Foundation (7 agents)**
- `scope_definer`: Translates business questions to ML scopes
- `data_preparer`: Data cleaning, validation, feature engineering
- `feature_analyzer`: Feature importance, selection, interaction analysis
- `model_selector`: Algorithm selection based on data characteristics
- `model_trainer`: Training with cross-validation, hyperparameter tuning
- `model_deployer`: Model deployment with versioning
- `observability_connector`: Metrics tracking, logging integration

**Tier 1: Coordination (2 agents)**
- `orchestrator`: 4-stage query classification, agent routing, response synthesis
- `tool_composer`: Multi-faceted query decomposition, tool orchestration

**Tier 2: Causal Analytics (3 agents)**
- `causal_impact`: Causal effect estimation with DoWhy/EconML + 5 refutation tests
- `gap_analyzer`: ROI opportunity identification, performance gap analysis
- `heterogeneous_optimizer`: Treatment effect heterogeneity (CATE analysis)

**Tier 3: Monitoring (3 agents)**
- `drift_monitor`: Data/model drift detection with statistical tests
- `experiment_designer`: A/B test design with digital twin pre-screening
- `health_score`: System health metrics, SLA monitoring

**Tier 4: ML Predictions (2 agents)**
- `prediction_synthesizer`: Multi-model ensemble predictions
- `resource_optimizer`: Resource allocation optimization

**Tier 5: Self-Improvement (2 agents)**
- `explainer`: Natural language explanations with SHAP integration
- `feedback_learner`: User feedback incorporation, model retraining triggers

**Acceptance Criteria**:
- 94%+ successful multi-agent orchestration rate
- <5s coordination overhead for multi-agent queries
- Full audit trail of agent interactions
- Graceful degradation if individual agents fail

---

### 3. Causal Inference Engine

**Description**: Rigorous causal analysis using industry-standard libraries (DoWhy, EconML) with statistical validation.

**Key Capabilities**:
- **Causal Graph Discovery**: Automated or user-defined causal DAGs
- **Effect Estimation**: ATE (Average Treatment Effect), CATE (Conditional ATE), ITE (Individual Treatment Effect)
- **Methods Supported**:
  - Propensity Score Matching
  - Inverse Propensity Weighting
  - Doubly Robust Estimation
  - Instrumental Variables
  - Regression Discontinuity
  - Difference-in-Differences

**5-Stage Validation Pipeline**:
1. **Placebo Treatment Test**: Replace treatment with random variable
2. **Random Common Cause Test**: Add random confounder
3. **Data Subset Validation**: Estimate on random subsets
4. **Bootstrap Estimation**: Confidence intervals via resampling
5. **Sensitivity Analysis**: E-value for unmeasured confounding

**Gate Decisions**:
- **PROCEED**: All 5 tests pass → High confidence estimate
- **REVIEW**: 3-4 tests pass → Flag for analyst review
- **BLOCK**: <3 tests pass → Do not report estimate

**Acceptance Criteria**:
- 87%+ of causal estimates achieve "PROCEED" status
- <30s for simple causal queries (single treatment/outcome)
- <2min for complex heterogeneous treatment effects
- Full statistical reporting (p-values, confidence intervals, effect sizes)

---

### 4. Real-Time Model Interpretability (SHAP)

**Description**: Instant explanations for any model prediction using SHAP (SHapley Additive exPlanations) with REST API access.

**Key Capabilities**:
- **Prediction Explanation**: "Why was this patient predicted as high-risk?"
- **Feature Importance**: "Which factors drove this prediction?"
- **Comparison Mode**: "Why did model A predict differently than model B?"
- **Natural Language Integration**: Explainer agent translates SHAP values to English

**Performance SLAs**:
- **P50 Latency**: <100ms (tree models), <200ms (kernel methods)
- **P95 Latency**: <300ms (all model types)
- **P99 Latency**: <500ms (complex models)

**REST API Endpoints**:
```
POST /api/v1/explain/predict      # Single prediction + explanation
POST /api/v1/explain/batch        # Batch predictions
GET  /api/v1/explain/history      # Explanation history for audit
GET  /api/v1/explain/models       # Available explainable models
GET  /api/v1/explain/health       # Explainer service health
```

**Visualization Support**:
- Waterfall charts (cumulative SHAP values)
- Force plots (interactive feature contributions)
- Bar charts (top features)
- Dependence plots (feature interactions)

**Acceptance Criteria**:
- <300ms P95 latency for all model types
- Audit trail for every explanation (compliance requirement)
- Natural language summaries for non-technical users
- Visual exports (PNG, SVG) for presentations

---

### 5. Digital Twin Engine (NEW v4.2)

**Description**: ML-based simulations of HCPs, patients, or territories to pre-screen marketing interventions before real-world deployment.

**Key Capabilities**:
- **Twin Generation**: Create digital twins from historical behavior data
- **Intervention Simulation**: Test "what if" scenarios on 10,000+ digital twins
- **Fidelity Tracking**: Validate twin predictions against real A/B test outcomes
- **Smart Recommendations**: Deploy/Skip/Refine based on simulated results

**Workflow**:
```
1. User: "Should we increase rep visits by 20% in Northeast region?"
2. System: Generates digital twins for Northeast HCPs
3. System: Simulates 20% visit increase on 10,000 twins
4. System: Reports:
   - Simulated ATE: +8.2% TRx increase
   - Confidence: 85% (based on twin fidelity)
   - Recommendation: DEPLOY (high ROI, low risk)
   - Estimated cost: $450K, Est. revenue lift: $2.1M
```

**Twin Fidelity Targets**:
- **Accuracy**: >80% match between twin predictions and real outcomes
- **Precision**: >75% for identifying high-responders
- **Recall**: >70% for identifying low-responders

**Acceptance Criteria**:
- Simulate 10,000 scenarios in <60 seconds
- Twin fidelity >80% on validation experiments
- MLflow tracking for all twin models
- A/B test comparison reports for fidelity validation

---

### 6. Tool Composer (NEW v4.2)

**Description**: Intelligent decomposition of complex, multi-faceted queries into orchestrated tool sequences.

**Key Capabilities**:
- **4-Stage Classification**:
  1. Intent feature extraction
  2. Domain mapping (causal, prediction, monitoring, etc.)
  3. Dependency detection (parallel vs sequential)
  4. Pattern selection (single agent, parallel delegation, tool composition)

- **4-Phase Pipeline**:
  1. **Decompose**: Break query into sub-questions
  2. **Plan**: Identify tools and execution order
  3. **Execute**: Run tools with dependency management
  4. **Synthesize**: Combine results into coherent answer

**Example Query**:
```
User: "Show me ROI for Q3 campaigns, predict Q4 impact if we double spend,
       and explain why certain HCPs respond better"

Decomposition:
1. ROI calculation (Gap Analyzer)
2. Q4 prediction (Prediction Synthesizer + Digital Twin)
3. Heterogeneous effect analysis (Heterogeneous Optimizer)
4. Explanation generation (Explainer + SHAP)

Execution: 1→2 (sequential), 3||4 (parallel)
```

**Acceptance Criteria**:
- Handle queries with 2-5 sub-questions
- Correctly identify parallel vs sequential dependencies
- <15s total execution for 3-sub-question queries
- Coherent synthesis (not just concatenated responses)

---

### 7. Tri-Memory Architecture

**Description**: Intelligent memory system across 4 types for context retention, pattern learning, and semantic understanding.

**Memory Types**:

**Working Memory (Redis)**
- Session state, active messages, evidence board
- TTL: 3600 seconds (1 hour)
- Supports rapid read/write for real-time interactions

**Episodic Memory (Supabase + pgvector)**
- User queries, agent actions, events
- Enables: "What did I ask last week about Fabhalta?"
- Vector similarity search for contextual recall

**Procedural Memory (Supabase + pgvector)**
- Tool sequences, successful query patterns
- Enables: Learning optimal agent workflows
- Reinforcement of high-success patterns

**Semantic Memory (FalkorDB - Graph Database)**
- Entities (HCPs, patients, drugs, KPIs)
- Relationships (causal chains, hierarchies)
- Enables: "Show me the causal path from rep visit to prescription"

**Acceptance Criteria**:
- <50ms read latency for working memory
- <200ms for episodic/procedural vector search
- <500ms for semantic graph traversal
- 30-day retention for episodic/procedural
- Persistent semantic memory

---

### 8. 46+ Pharma-Native KPIs

**Description**: Pre-configured key performance indicators specific to pharmaceutical commercial operations.

**KPI Categories**:

**Prescription Metrics**:
- TRx (Total Prescriptions)
- NRx (New Prescriptions)
- Refill Rate
- Persistence (days on therapy)
- Discontinuation Rate

**HCP Engagement**:
- Rep Visit Frequency
- Sample Drops
- Speaker Program Attendance
- Digital Engagement (email opens, portal usage)
- Peer Influence Score

**Patient Journey**:
- Awareness → Trial Conversion
- Trial → Adoption Rate
- Patient Abandonment Rate
- Prior Authorization Success Rate
- Copay Assistance Utilization

**Market Dynamics**:
- Market Share (NBRx, TRx)
- Competitive Win/Loss Ratio
- Brand Switching Patterns
- Formulary Coverage %

**ROI & Financial**:
- Marketing ROI (by channel)
- Cost per Acquisition
- Lifetime Value (LTV)
- Promotional Response Rate

**Acceptance Criteria**:
- 100% coverage of critical commercial KPIs
- Definitions aligned with IQVIA/Symphony standards
- Configurable thresholds per brand/market
- Trend analysis (MoM, QoQ, YoY)

---

### 9. Hybrid RAG System

**Description**: Retrieval-Augmented Generation combining vector search, full-text search, and graph traversal for contextual responses.

**Retrieval Strategies**:
- **Vector Search** (pgvector): Semantic similarity for unstructured queries
- **Full-Text Search** (PostgreSQL FTS): Exact matches for technical terms
- **Graph Traversal** (FalkorDB): Causal chain exploration

**RAG Indexes**:
- `causal_paths`: Discovered causal relationships
- `agent_activities`: Historical agent analyses
- `business_metrics`: Performance trend summaries
- `triggers`: Trigger explanations (why alerts fired)
- `conversations`: Historical Q&A for learning

**Acceptance Criteria**:
- <300ms retrieval latency (P95)
- >85% relevance for top-3 results
- Hybrid ranking combines all 3 strategies
- Source attribution for every retrieved fact

---

## Non-Functional Requirements

### Performance

- **Query Latency**:
  - Simple queries: <2s (P95)
  - Complex multi-agent queries: <10s (P95)
  - SHAP explanations: <300ms (P95)
  - Digital twin simulations: <60s for 10K scenarios

- **Throughput**:
  - Support 100 concurrent users
  - 1,000 queries/hour sustained load
  - Graceful degradation beyond capacity

### Scalability

- **Horizontal Scaling**: Stateless API layer, scale with containers
- **Database**: Supabase scales to 100M+ rows
- **Memory**: Redis cluster for distributed working memory
- **Agents**: Independent scaling per agent tier

### Reliability

- **Uptime**: 99.5% SLA (excluding planned maintenance)
- **Recovery Time Objective (RTO)**: <1 hour
- **Recovery Point Objective (RPO)**: <15 minutes
- **Graceful Degradation**: Core query functionality maintained if individual agents fail

### Security & Compliance

- **Data Privacy**: De-identified data only, no PHI
- **Access Control**: Role-based access (RBAC)
- **Audit Logging**: Full audit trail for all queries and predictions
- **Encryption**: TLS 1.3 in transit, AES-256 at rest
- **Compliance**: GDPR, HIPAA-aligned (commercial data only), 21 CFR Part 11 (e-signatures)

### Monitoring & Observability

- **Logging**: Structured JSON logs (loguru)
- **Metrics**: Prometheus-compatible metrics
- **Tracing**: OpenTelemetry for distributed tracing
- **Alerting**: Health score agent triggers alerts for drift, degradation, errors

---
