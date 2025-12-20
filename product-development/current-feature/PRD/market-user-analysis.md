# Market & User Analysis

## Target Users

### Primary Personas

**1. Commercial Analytics Leader**
- **Role**: VP/Director of Commercial Analytics
- **Goals**: Strategic insights, ROI optimization, resource allocation
- **Pain Points**:
  - Manual analysis takes weeks
  - Correlational tools miss causal drivers
  - Cannot answer "what if" questions proactively
- **Job to be Done**: "Help me identify the highest-ROI commercial investments and prove causality to executives"

**2. Field Operations Director**
- **Role**: Regional Sales Director, Field Force Operations
- **Goals**: Optimize territory coverage, improve HCP targeting
- **Pain Points**:
  - Poor visibility into what's actually driving prescriptions
  - Resource allocation based on gut feel, not data
  - Cannot test strategies before full deployment
- **Job to be Done**: "Help me understand which HCP engagement activities actually drive prescribing behavior"

**3. Brand Manager**
- **Role**: Product Manager for specific drug brand
- **Goals**: Maximize TRx/NRx, understand competitive dynamics
- **Pain Points**:
  - Reports show what happened, not why
  - Cannot predict impact of marketing campaigns
  - Slow to detect market shifts
- **Job to be Done**: "Help me understand the causal path from marketing investment to prescription behavior"

**4. Data Analyst**
- **Role**: Commercial Analytics Analyst
- **Goals**: Generate insights, support decision-making
- **Pain Points**:
  - Repetitive query requests
  - Difficult to perform causal analysis at scale
  - Manual validation of statistical assumptions
- **Job to be Done**: "Help me automate routine analysis and focus on strategic questions"

### Secondary Personas

**5. C-Suite Executive**
- **Needs**: High-level dashboards, strategic recommendations, ROI visibility
- **Usage Pattern**: Occasional, summary-level queries

**6. Medical Science Liaison**
- **Needs**: Understand HCP engagement patterns, educational gaps
- **Usage Pattern**: Regular, HCP-focused queries

## Market Landscape

### Competitive Analysis

| Competitor | Strengths | Weaknesses | E2I Advantage |
|------------|-----------|------------|---------------|
| **Traditional BI Tools** (Tableau, PowerBI) | Visualization, adoption | No causal inference, manual analysis | Automated causal analysis, NL queries |
| **Pharma-Specific Analytics** (Veeva CRM, IQVIA) | Industry-specific data | Correlational only, rigid dashboards | True causal inference, flexible AI agents |
| **Data Science Platforms** (DataRobot, H2O.ai) | ML capabilities | Requires data science expertise, no pharma domain | Natural language, pharma-native KPIs |
| **Causal Inference Tools** (DoWhy, CausalML) | Statistical rigor | Requires coding, not enterprise-ready | Enterprise platform with embedded causal engine |

### Market Opportunity

- **Total Addressable Market (TAM)**: $4.8B (pharma analytics by 2028)
- **Serviceable Addressable Market (SAM)**: $1.2B (large pharma commercial analytics)
- **Serviceable Obtainable Market (SOM)**: $180M (Year 3 target with 15% market penetration)

## User Needs & Jobs to be Done

### Core Jobs to be Done

**1. Understand Causal Drivers**
- **Job**: "When I see a change in prescriptions, I need to understand what actually caused it, not just what correlated with it"
- **Current Alternative**: Manual analysis with Excel, correlational BI tools
- **Success Criteria**: Validated causal estimates with statistical confidence in <2 minutes

**2. Optimize Resource Allocation**
- **Job**: "When planning commercial investments, I need to know which interventions have the highest causal impact on prescriptions"
- **Current Alternative**: Historical ROI analysis, A/B testing (slow and expensive)
- **Success Criteria**: Causal ROI estimates with scenario analysis in real-time

**3. Test Before Deploying**
- **Job**: "Before launching a new campaign, I need to predict its impact without risking budget on failed experiments"
- **Current Alternative**: Small pilot tests, expert judgment
- **Success Criteria**: Digital twin simulations with >80% fidelity to real-world outcomes

**4. Explain to Stakeholders**
- **Job**: "When presenting insights to executives or regulators, I need to explain exactly how the model arrived at its conclusion"
- **Current Alternative**: Black-box models with no transparency
- **Success Criteria**: Plain-English explanations with SHAP values in <500ms

**5. Monitor Performance Continuously**
- **Job**: "When market conditions change, I need to know immediately if my models or strategies are still valid"
- **Current Alternative**: Quarterly manual reviews
- **Success Criteria**: Real-time drift detection with automated alerts

---
