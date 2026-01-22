# E2I Causal Analytics User Guide

A step-by-step guide to using the E2I Causal Analytics dashboard and its agentic system for pharmaceutical commercial operations.

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Dashboard Navigation](#dashboard-navigation)
4. [Using the AI Chat Interface](#using-the-ai-chat-interface)
5. [Key Workflows](#key-workflows)
6. [Understanding the Agent System](#understanding-the-agent-system)
7. [Visualizations Guide](#visualizations-guide)
8. [KPI Reference](#kpi-reference)
9. [Troubleshooting](#troubleshooting)
10. [Glossary](#glossary)

---

## Overview

E2I Causal Analytics is a Natural Language Visualization platform designed for pharmaceutical commercial operations. It combines:

- **18 AI Agents** organized in 6 tiers for intelligent query processing
- **Causal Inference** powered by DoWhy and EconML
- **Self-Improving RAG** (Retrieval-Augmented Generation)
- **Digital Twin Simulations** for A/B test pre-screening

### Supported Brands

| Brand | Indication | Abbreviation |
|-------|------------|--------------|
| **Remibrutinib** | Chronic Spontaneous Urticaria | CSU |
| **Fabhalta** | Paroxysmal Nocturnal Hemoglobinuria | PNH |
| **Kisqali** | HR+/HER2- Breast Cancer | CDK4/6 inhibitor |

### What This System Does

- Analyzes **commercial operations** data (TRx, NRx, market share)
- Identifies **causal relationships** between interventions and outcomes
- Detects **ROI opportunities** and performance gaps
- Designs and pre-screens **A/B tests** using digital twins
- Provides **natural language explanations** of complex analytics

### What This System Does NOT Do

- Clinical decision support
- Medical literature search
- Drug safety monitoring
- Patient treatment recommendations

---

## Getting Started

### Accessing the Dashboard

**Production Environment:**
- **Dashboard URL:** http://138.197.4.36/
- **API Documentation:** http://138.197.4.36:8000/docs

**Local Development:**
- **Dashboard URL:** http://localhost:5174/
- **API URL:** http://localhost:8000/

### First-Time Setup

1. **Open the dashboard** in your browser
2. **Select your brand** from the dropdown in the header (Remibrutinib, Fabhalta, or Kisqali)
3. **Choose your region** (5 US regions available)
4. **Set the date range** (Quarter, YTD, or Rolling 12 months)

### System Requirements

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Stable internet connection
- Screen resolution: 1280x720 minimum (1920x1080 recommended)

---

## Dashboard Navigation

### Main Navigation Menu

The sidebar contains the following sections:

#### Executive Views
| Page | Description |
|------|-------------|
| **Home** | Executive dashboard with KPI overview, alerts, and agent insights |
| **Knowledge Graph** | Interactive visualization of entity relationships |
| **KPI Dictionary** | All 46 KPIs with definitions and methodologies |

#### Analytics
| Page | Description |
|------|-------------|
| **Causal Discovery** | DAG visualization and causal chain analysis |
| **Causal Analysis** | Heterogeneous treatment effects by segment |
| **Gap Analysis** | ROI opportunities and benchmarking |
| **Intervention Impact** | Treatment effects and causal inference results |

#### Predictions
| Page | Description |
|------|-------------|
| **Predictive Analytics** | Risk scores and ML predictions |
| **Time Series** | Trend analysis and forecasting |
| **Resource Optimization** | Budget allocation recommendations |
| **Segment Analysis** | HCP and patient cohort analysis |

#### Experimentation
| Page | Description |
|------|-------------|
| **Experiments** | A/B test management and results |
| **Digital Twin** | Pre-screen tests with ML simulations |

#### Model & System Health
| Page | Description |
|------|-------------|
| **Model Performance** | Metrics, SHAP explanations, model health |
| **Feature Importance** | SHAP waterfall, beeswarm, bar charts |
| **Data Quality** | Data profiling and validation |
| **System Health** | Infrastructure monitoring |
| **Monitoring** | Logs, API usage, error tracking |

#### Agent System
| Page | Description |
|------|-------------|
| **Agent Orchestration** | 18-agent tier visualization and status |
| **Feedback Learning** | Agent self-improvement metrics |
| **Memory Architecture** | Tri-memory system status |
| **Audit Chain** | Compliance and traceability logs |

---

## Using the AI Chat Interface

The AI chat interface is your primary way to interact with the E2I agent system using natural language.

### Opening the Chat

**Option 1:** Click the chat icon in the bottom-right corner of the screen

**Option 2:** Use the keyboard shortcut `Cmd + /` (Mac) or `Ctrl + /` (Windows/Linux)

### Query Types You Can Ask

#### Causal Analysis Questions
```
"What is the causal effect of HCP engagement on TRx for Remibrutinib?"
"What causes conversion rate drops in the Northeast region?"
"How does speaker program attendance affect prescription behavior?"
```

#### Gap & Opportunity Detection
```
"Where are the biggest ROI opportunities for Kisqali?"
"What performance gaps exist compared to benchmarks?"
"Which territories are underperforming?"
```

#### Predictions & Targeting
```
"Who should I target for the next campaign?"
"Which HCPs are most likely to convert?"
"What is the predicted TRx for Q2?"
```

#### Experiment Design
```
"Should we run a speaker program test for Fabhalta?"
"Design an A/B test for digital engagement"
"Pre-screen this intervention before deployment"
```

#### Segment Analysis
```
"Which segments respond best to digital campaigns?"
"Compare high-value vs. low-value HCP segments"
"What are the characteristics of our best responders?"
```

#### Explanations & Why Questions
```
"Why is patient X flagged as high-risk?"
"Explain the model's prediction for territory A"
"What factors drive NRx growth?"
```

#### System Health
```
"Is the system healthy?"
"Are there any data drift alerts?"
"What is the current model performance?"
```

### Chat Features

- **Real-time streaming:** See responses as they're generated token-by-token
- **Agent indicators:** See which agents are processing your query
- **Tool invocations:** Watch as the system calls specialized tools
- **Follow-up suggestions:** Get recommendations for related queries
- **Message history:** Conversations are saved for future reference

### Tips for Better Results

1. **Be specific:** Include brand name, region, and time period when relevant
2. **Use natural language:** The system understands conversational queries
3. **Ask follow-ups:** Build on previous answers for deeper analysis
4. **Request explanations:** Ask "why" to get detailed reasoning

---

## Key Workflows

### Workflow 1: Causal Impact Analysis

**Goal:** Understand the causal effect of an intervention on business outcomes

**Steps:**
1. Open the chat interface
2. Ask: `"What is the causal effect of [intervention] on [outcome] for [brand]?"`
3. Review the results:
   - **Effect estimate** (Average Treatment Effect)
   - **Confidence interval**
   - **Refutation test results** (5 validation tests)
   - **Causal DAG** visualization

**Example:**
```
You: "What is the causal effect of speaker programs on TRx for Remibrutinib in Q4?"

E2I: Based on causal analysis:
- Average Treatment Effect: +12.3 TRx per HCP (95% CI: [8.1, 16.5])
- Refutation tests: 5/5 passed
- Key mediators: HCP engagement score, formulary status
- Recommendation: Continue speaker programs with focus on high-engagement HCPs
```

### Workflow 2: ROI Opportunity Detection

**Goal:** Find the highest-impact opportunities for investment

**Steps:**
1. Navigate to **Gap Analysis** page or use chat
2. Ask: `"Where are the biggest ROI opportunities for [brand]?"`
3. Review prioritized opportunities:
   - **Gap description**
   - **Estimated ROI**
   - **Actionable recommendations**

**Example:**
```
You: "Where are the biggest ROI opportunities for Kisqali?"

E2I: Top 3 ROI Opportunities:

1. Digital Engagement Gap (Southeast Region)
   - Gap: 23% below benchmark
   - Estimated ROI: $2.4M annually
   - Action: Increase digital touchpoints by 40%

2. HCP Coverage Gap (Academic Medical Centers)
   - Gap: 18% of key accounts under-reached
   - Estimated ROI: $1.8M annually
   - Action: Deploy dedicated AMC specialists

3. Conversion Funnel Drop-off (Trial to Prescription)
   - Gap: 34% conversion vs. 48% benchmark
   - Estimated ROI: $1.2M annually
   - Action: Implement targeted follow-up program
```

### Workflow 3: Digital Twin A/B Test Pre-Screening

**Goal:** Simulate an intervention before deploying to production

**Steps:**
1. Navigate to **Digital Twin** page
2. Select intervention type and parameters
3. Click "Run Simulation" or ask via chat
4. Review simulation results:
   - **Expected lift**
   - **Confidence level**
   - **Recommendation** (Deploy / Skip / Refine)

**Example:**
```
You: "Should we run a speaker program test for Fabhalta in the Northeast?"

E2I: Digital Twin Simulation Results:

Simulated Scenarios: 10,000 iterations
Expected Lift: +8.7% TRx (95% CI: [5.2%, 12.1%])
Statistical Power: 82%
Sample Size Required: 150 HCPs per arm

Recommendation: DEPLOY
Rationale: Expected effect exceeds minimum detectable effect (5%)
with sufficient power. Similar historical tests showed 87% accuracy.

Suggested refinements:
- Extend test duration from 8 to 12 weeks for higher confidence
- Focus on oncology specialists (highest predicted response)
```

### Workflow 4: Segment-Level Treatment Effects

**Goal:** Identify which customer segments respond best to interventions

**Steps:**
1. Navigate to **Causal Analysis** or **Segment Analysis**
2. Ask: `"Which segments respond best to [intervention]?"`
3. Review CATE (Conditional Average Treatment Effects) by segment

**Example:**
```
You: "Which HCP segments respond best to digital campaigns for Kisqali?"

E2I: Heterogeneous Treatment Effects Analysis:

Top Responding Segments:
1. Community Oncologists (CATE: +18.2 TRx, p<0.01)
   - Profile: 10-20 years experience, medium-sized practice

2. Academic Researchers (CATE: +14.7 TRx, p<0.01)
   - Profile: Heavy publication activity, clinical trial involvement

3. Urban Specialists (CATE: +11.3 TRx, p<0.05)
   - Profile: High patient volume, tech-savvy

Low Responding Segments:
- Rural Practitioners (CATE: +2.1 TRx, not significant)
- Recommendation: Reallocate budget from rural digital to in-person

Targeting Strategy: Focus 70% of digital spend on segments 1-3
```

### Workflow 5: Model Interpretability

**Goal:** Understand why a model made a specific prediction

**Steps:**
1. Navigate to **Feature Importance** page
2. Select a specific prediction to explain
3. Review SHAP visualizations:
   - **Waterfall chart:** Feature contributions for this prediction
   - **Force plot:** Interactive contribution visualization
   - **Beeswarm:** Feature importance across all predictions

**Example:**
```
You: "Why is HCP Dr. Smith flagged as high-conversion probability?"

E2I: SHAP Explanation for Dr. Smith (Conversion Probability: 87%)

Top Contributing Factors:
1. Recent formulary addition (+0.23 impact)
   - Her hospital just added the drug to formulary

2. High engagement score (+0.18 impact)
   - Attended 3 speaker programs, 12 digital touchpoints

3. Specialty match (+0.15 impact)
   - Oncology specialty aligns with indication

4. Peer influence (+0.12 impact)
   - 4 colleagues already prescribing

Detractors:
1. No prior prescriptions (-0.08 impact)
   - New to the brand

[View Waterfall Chart] [View Force Plot]
```

### Workflow 6: Providing Feedback for System Improvement

**Goal:** Help the system learn from your corrections

**Steps:**
1. After receiving a response, use the feedback buttons (thumbs up/down)
2. Optionally provide detailed feedback via chat
3. The system incorporates feedback to improve future responses

**Example:**
```
You: "This recommendation about speaker programs was incorrect -
we already tried this and it didn't work in that region."

E2I: Thank you for the feedback. I've logged this correction:
- Added negative example for speaker programs in Northeast region
- Updated regional context with your institutional knowledge
- This feedback will improve future recommendations (GEPA optimization)

Would you like to provide more details about what actually worked
in that region?
```

---

## Understanding the Agent System

### The 18-Agent Architecture

E2I uses a tiered multi-agent architecture where specialized agents collaborate to answer your queries.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 0: ML FOUNDATION                        │
│  scope_definer | data_preparer | feature_analyzer | model_      │
│  selector | model_trainer | model_deployer | observability_     │
│  connector                                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 1: ORCHESTRATION                        │
│              orchestrator | tool_composer                       │
│         (Entry point for all queries - routes to specialists)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 2: CAUSAL ANALYTICS                     │
│       causal_impact | gap_analyzer | heterogeneous_optimizer    │
│              (Core analytical capabilities)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 3: MONITORING                           │
│       drift_monitor | experiment_designer | health_score        │
│           (System health and experiment design)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 4: ML PREDICTIONS                       │
│          prediction_synthesizer | resource_optimizer            │
│           (Predictions and resource allocation)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 5: SELF-IMPROVEMENT                     │
│                 explainer | feedback_learner                    │
│         (Natural language and continuous learning)              │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Descriptions

| Agent | Tier | What It Does |
|-------|------|--------------|
| **Orchestrator** | 1 | Routes queries to appropriate agents, synthesizes responses |
| **Tool Composer** | 1 | Breaks complex queries into sub-tasks, coordinates execution |
| **Causal Impact** | 2 | Discovers causal relationships, estimates treatment effects |
| **Gap Analyzer** | 2 | Detects performance gaps and ROI opportunities |
| **Heterogeneous Optimizer** | 2 | Analyzes treatment effects by customer segment |
| **Drift Monitor** | 3 | Detects data and model drift over time |
| **Experiment Designer** | 3 | Designs A/B tests with statistical rigor |
| **Health Score** | 3 | Monitors system and model health |
| **Prediction Synthesizer** | 4 | Aggregates predictions from multiple models |
| **Resource Optimizer** | 4 | Recommends optimal resource allocation |
| **Explainer** | 5 | Generates natural language explanations |
| **Feedback Learner** | 5 | Improves system based on user feedback |

### How Query Processing Works

1. **You submit a query** via chat or API
2. **Orchestrator classifies intent** using a 4-stage classifier
3. **Router determines dispatch plan:**
   - Single agent for focused queries
   - Multiple agents in parallel for complex queries
   - Tool Composer for multi-faceted analysis
4. **Agents execute** with access to data, models, and tools
5. **Results synthesized** into coherent response
6. **RAG enrichment** adds relevant context from knowledge base
7. **Response streamed** back to you in real-time

### Viewing Agent Activity

Navigate to **Agent Orchestration** page to see:
- Active vs. idle agents
- Query processing in real-time
- Agent performance metrics
- Tier visualization

---

## Visualizations Guide

### Causal Visualizations

#### DAG (Directed Acyclic Graph)
- **What it shows:** Causal relationships between variables
- **How to read it:** Arrows show direction of causation (A → B means A causes B)
- **Interaction:** Click nodes for details, hover for variable names

#### Effects Table
- **What it shows:** Estimated treatment effects with confidence intervals
- **Columns:** Effect estimate, standard error, 95% CI, p-value
- **Color coding:** Green = significant positive, Red = significant negative

#### Refutation Tests
- **What it shows:** Validation of causal estimates
- **5 tests:** Placebo treatment, Random cause, Subset data, Bootstrap, Sensitivity
- **Interpretation:** All tests should pass for reliable estimates

### SHAP Visualizations

#### Waterfall Chart
- **What it shows:** How each feature contributes to a single prediction
- **How to read:** Bar length shows impact, color shows positive (red) vs. negative (blue)
- **Use case:** Explaining individual predictions

#### Beeswarm Plot
- **What it shows:** Feature importance distribution across all predictions
- **How to read:** Each dot is a prediction, color shows feature value
- **Use case:** Understanding overall model behavior

#### Bar Chart
- **What it shows:** Mean absolute SHAP values by feature
- **How to read:** Longer bars = more important features
- **Use case:** Quick feature importance ranking

### Dashboard Visualizations

#### KPI Cards
- **Components:** Current value, sparkline trend, status indicator, target
- **Colors:** Green (on target), Yellow (warning), Red (alert)

#### Time Series Charts
- **Features:** Multi-axis support, zoom, pan, annotations
- **Interaction:** Hover for values, click to drill down

#### Knowledge Graph
- **Node types:** HCPs, Products, Regions, Metrics (color-coded)
- **Edge types:** Relationships like "prescribes", "influences", "belongs_to"
- **Filters:** Filter by entity type, time period, relationship type

---

## KPI Reference

### Commercial KPIs
| KPI | Definition |
|-----|------------|
| TRx | Total prescriptions dispensed |
| NRx | New prescriptions (new-to-brand patients) |
| Market Share | Brand TRx / Total category TRx |
| Revenue | Net revenue from prescription sales |

### HCP Engagement KPIs
| KPI | Definition |
|-----|------------|
| Reach | % of target HCPs contacted |
| Frequency | Average contacts per reached HCP |
| Conversion Rate | % of engaged HCPs who prescribe |
| Specialty Penetration | Prescribers per specialty / Total specialists |

### Patient Journey KPIs
| KPI | Definition |
|-----|------------|
| Patient Starts | New patients starting therapy |
| Adherence Rate | % of patients compliant with dosing |
| Discontinuation Rate | % of patients stopping therapy |
| Refill Rate | % of patients refilling prescriptions |

### Causal Metrics
| KPI | Definition |
|-----|------------|
| ATE | Average Treatment Effect across population |
| CATE | Conditional ATE for specific segments |
| ROI | Return on investment from interventions |
| Lift | % improvement vs. control |

Access the complete KPI dictionary with detailed methodology at **KPI Dictionary** page.

---

## Troubleshooting

### Common Issues

#### Chat Not Responding
1. Check your internet connection
2. Refresh the page
3. Try a simpler query first
4. Check System Health page for service status

#### Slow Response Times
1. Complex queries with multiple agents take longer
2. Check if drift monitor is running background jobs
3. Peak usage times may have longer queues

#### Unexpected Results
1. Verify your filters (brand, region, date range)
2. Check data quality page for anomalies
3. Provide feedback via chat to improve the system

#### Visualization Not Loading
1. Try refreshing the page
2. Check browser console for errors
3. Ensure you have the latest browser version

### Getting Help

- **In-app:** Use the chat to ask "help" or "what can you do?"
- **Documentation:** Check docs/ folder for detailed guides
- **API Docs:** Visit /api/docs for endpoint reference
- **GitHub Issues:** Report bugs at the project repository

---

## Glossary

| Term | Definition |
|------|------------|
| **ATE** | Average Treatment Effect - mean causal effect across population |
| **CATE** | Conditional Average Treatment Effect - effect for specific segments |
| **DAG** | Directed Acyclic Graph - visualization of causal relationships |
| **Digital Twin** | Simulated environment for testing interventions |
| **DoWhy** | Python library for causal inference |
| **EconML** | Machine learning library for heterogeneous treatment effects |
| **GEPA** | Generative Evolutionary Prompting with AI - prompt optimization |
| **HCP** | Healthcare Professional |
| **NRx** | New prescriptions |
| **RAG** | Retrieval-Augmented Generation |
| **SHAP** | SHapley Additive exPlanations - model interpretability |
| **TRx** | Total prescriptions |

---

## Quick Reference Card

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Cmd/Ctrl + /` | Open/close chat |
| `Cmd/Ctrl + K` | Quick search |
| `Escape` | Close dialogs |

### Common Chat Commands
```
"What is the causal effect of [X] on [Y]?"
"Where are the ROI opportunities for [brand]?"
"Which segments respond best to [intervention]?"
"Explain why [prediction/metric]"
"Is the system healthy?"
"Design an A/B test for [intervention]"
"Pre-screen [test] with digital twin"
```

### API Quick Reference
```bash
# Get all KPIs
GET /api/kpis

# Run causal analysis
POST /api/causal/analyze

# Get gap analysis
POST /api/gaps/detect

# Design experiment
POST /api/experiments/design

# Get system health
GET /api/health
```

---

*Last Updated: January 2026*
*Version: 4.3 (GEPA Prompt Optimization)*
