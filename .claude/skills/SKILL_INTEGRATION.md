# E2I Domain Skills Framework: Pharma Commercial Analytics

**Date**: 2025-01-25
**Version**: 1.0
**Purpose**: Encode procedural knowledge for the E2I 21-agent architecture

---

## Overview

This document expands on the recommendation to **"Build domain skills: Encode pharma commercial analytics procedures"** from the Skills vs MCP evaluation. Skills are markdown-based procedural knowledge that teach agents **how** to perform domain-specific tasks.

### Why Skills for Pharma Commercial Analytics?

| Challenge | How Skills Address It |
|-----------|----------------------|
| Agents lose domain context over long conversations | Skills provide on-demand domain procedures |
| KPI calculations require specific business logic | Skills encode exact formulas and thresholds |
| Causal inference needs pharma-specific confounders | Skills list standard confounders per analysis type |
| Experiment design requires validity threat awareness | Skills provide threat taxonomy and mitigations |
| User expertise levels vary (exec vs analyst vs data scientist) | Skills define framing rules per audience |

---

## Skill Directory Structure

```
.claude/skills/
├── SKILL.md                           # Master skill index
├── pharma-commercial/
│   ├── SKILL.md                       # Category index
│   ├── kpi-calculation.md             # TRx, NRx, conversion, market share procedures
│   ├── brand-analytics.md             # Remibrutinib, Fabhalta, Kisqali specifics
│   ├── hcp-targeting.md               # HCP segmentation and prioritization
│   ├── patient-journey.md             # 7-stage funnel analysis
│   ├── trigger-attribution.md         # Trigger-to-outcome attribution rules
│   └── competitive-analysis.md        # Competitor displacement analysis
├── causal-inference/
│   ├── SKILL.md                       # Category index
│   ├── confounder-identification.md   # Standard confounders for pharma
│   ├── dowhy-workflow.md              # DoWhy estimation procedures
│   ├── refutation-testing.md          # Robustness test interpretation
│   ├── sensitivity-analysis.md        # E-value and unobserved confounding
│   └── cate-analysis.md               # Heterogeneous effects with EconML
├── experiment-design/
│   ├── SKILL.md                       # Category index
│   ├── power-analysis.md              # MDE and sample size calculations
│   ├── validity-threats.md            # 6-threat taxonomy
│   ├── design-selection.md            # RCT vs Cluster vs Quasi selection
│   └── pre-registration.md            # OSF-format pre-registration
├── gap-analysis/
│   ├── SKILL.md                       # Category index
│   ├── gap-detection.md               # Multi-comparison methodology
│   ├── roi-estimation.md              # Revenue impact and cost-to-close
│   └── opportunity-prioritization.md  # Quick wins vs strategic bets
└── data-quality/
    ├── SKILL.md                       # Category index
    ├── drift-detection.md             # PSI and concept drift procedures
    ├── label-quality.md               # Labeling edge cases
    └── data-source-lag.md             # Source-specific lag handling
```

---

## Master Skill Index

### File: `.claude/skills/SKILL.md`

```yaml
---
name: E2I Pharma Commercial Analytics Skills
version: 1.0
description: Domain procedures for pharmaceutical commercial analytics
author: E2I Team
triggers:
  - pharma analytics
  - commercial operations
  - KPI calculation
  - causal analysis
  - experiment design
  - gap analysis
categories:
  - pharma-commercial
  - causal-inference
  - experiment-design
  - gap-analysis
  - data-quality
---

# E2I Pharma Commercial Analytics Skills

This skill collection encodes procedural knowledge for pharmaceutical commercial analytics operations. Use these skills when analyzing:

- **Prescription metrics** (TRx, NRx, NBRx, market share)
- **HCP engagement** (targeting, reach, conversion)
- **Patient journeys** (awareness → maintenance funnel)
- **Causal effects** (treatment impacts, intervention effects)
- **Experiments** (A/B tests, quasi-experiments)
- **Opportunities** (gaps, ROI estimation)

## When to Load

Load skills based on the task:

| Task Type | Load Skills |
|-----------|-------------|
| KPI analysis | `pharma-commercial/kpi-calculation.md` |
| Brand-specific analysis | `pharma-commercial/brand-analytics.md` |
| Causal impact estimation | `causal-inference/dowhy-workflow.md` |
| Experiment design | `experiment-design/power-analysis.md`, `experiment-design/validity-threats.md` |
| Gap/opportunity analysis | `gap-analysis/gap-detection.md`, `gap-analysis/roi-estimation.md` |
| Data quality issues | `data-quality/drift-detection.md` |

## Domain Constraints

**This system IS**:
- Pharmaceutical commercial operations analytics ✅
- Business KPIs: TRx, NRx, conversion rates, market share ✅
- HCP targeting and patient journey analysis ✅

**This system IS NOT**:
- Clinical decision support ❌
- Medical literature search ❌
- Drug safety monitoring ❌
```

---

## Pharma Commercial Skills

### File: `.claude/skills/pharma-commercial/kpi-calculation.md`

```yaml
---
name: KPI Calculation Procedures
version: 1.0
description: Standard procedures for calculating pharma commercial KPIs
triggers:
  - calculate TRx
  - calculate NRx
  - market share
  - conversion rate
  - prescription metrics
agents:
  - gap_analyzer
  - prediction_synthesizer
  - explainer
---

# KPI Calculation Procedures

## Prescription Volume Metrics

### TRx (Total Prescriptions)
**Definition**: Total prescriptions dispensed for a brand within a time period.

**Calculation**:
```
TRx = COUNT(prescriptions WHERE brand = target_brand AND dispense_date IN period)
```

**Data Sources**: IQVIA APLD (12-day lag), HealthVerity (7-day lag)

**Revenue Multiplier**: $500 per TRx (use for ROI calculations)

### NRx (New Prescriptions)
**Definition**: Prescriptions for patients new to the brand within the period.

**Calculation**:
```
NRx = COUNT(prescriptions WHERE
  brand = target_brand AND
  dispense_date IN period AND
  patient NOT IN (prior_patients_180_days)
)
```

**Revenue Multiplier**: $400 per NRx

### NBRx (New-to-Brand Prescriptions)
**Definition**: First prescription for patients switching from a competitor.

**Calculation**:
```
NBRx = COUNT(prescriptions WHERE
  brand = target_brand AND
  is_first_fill = TRUE AND
  prior_brand IN competitor_brands
)
```

### Market Share
**Definition**: Brand TRx as percentage of total category TRx.

**Calculation**:
```
Market Share = (Brand TRx / Category TRx) × 100
```

**Targets by Brand**:
- Kisqali: ≥35% of CDK4/6 inhibitor category
- Remibrutinib: ≥25% of CSU biologics (Year 1)
- Fabhalta: Target specialty share (rare disease dynamics)

**Revenue Multiplier**: $500,000 per 1% market share point

---

## Engagement Metrics

### Conversion Rate
**Definition**: Percentage of triggered HCPs who write a prescription.

**Calculation**:
```
Conversion Rate = (HCPs with NRx / HCPs with Trigger) × 100
```

**Attribution Window**: 21 days from trigger delivery

**Target**: ≥8%

**Revenue Multiplier**: $50,000 per 1% conversion point

### HCP Reach
**Definition**: Percentage of priority HCPs with any engagement.

**Calculation**:
```
HCP Reach = (Engaged HCPs / Priority HCPs) × 100
```

**Target**: ≥75%

### Patient Touch Rate
**Definition**: Percentage of eligible patients with at least one trigger.

**Calculation**:
```
Patient Touch Rate = (Patients with Trigger / Eligible Patients) × 100
```

**Target**: ≥40%

---

## Adherence Metrics

### PDC (Proportion of Days Covered)
**Definition**: Days with medication available / Days in observation period.

**Calculation**:
```
PDC = (Days Covered / Observation Days) × 100
```

**Adherence Threshold**: PDC ≥ 80% = Adherent

**Observation Window**: 180 days

### Persistence
**Definition**: Duration from first fill to discontinuation.

**Calculation**:
```
Persistence = Discontinuation Date - First Fill Date
```

**Max Gap**: 60 days between fills before counting as discontinued

---

## Trigger Performance Metrics

### Trigger Precision
**Definition**: Percentage of triggered patients who convert.

**Calculation**:
```
Precision = TP / (TP + FP)
```

Where:
- TP = Triggers where patient converted
- FP = Triggers where patient did not convert

### Lead Time
**Definition**: Median days from trigger to outcome.

**Target**: ≤14 days

### Acceptance Rate
**Definition**: Percentage of triggers acted upon by reps.

**Calculation**:
```
Acceptance Rate = (Triggers with Rep Action / Triggers Delivered) × 100
```

**Target**: ≥60%

---

## ROI Calculation

**Formula**:
```
ROI = (Revenue Impact - Cost to Close) / Cost to Close
```

**Target**: ≥3.0 (300% return)

**Revenue Impact by Metric**:
| Metric | Revenue per Unit |
|--------|------------------|
| TRx | $500 |
| NRx | $400 |
| Conversion (1%) | $50,000 |
| Market Share (1%) | $500,000 |

**Cost Estimation**:
| Intervention Type | Cost Formula |
|-------------------|--------------|
| HCP Reach | $150 × gap × 2 |
| Conversion Improvement | $1,000 × gap × 400 |
| General | $150 × gap |

**Payback Period** (months):
```
Payback = Cost / (Revenue / 12)
```
Cap at 24 months maximum.
```

---

### File: `.claude/skills/pharma-commercial/brand-analytics.md`

```yaml
---
name: Brand-Specific Analytics
version: 1.0
description: Brand context and analytics rules for Kisqali, Fabhalta, Remibrutinib
triggers:
  - Kisqali analysis
  - Fabhalta analysis
  - Remibrutinib analysis
  - brand context
agents:
  - causal_impact
  - gap_analyzer
  - experiment_designer
  - explainer
---

# Brand-Specific Analytics

## Kisqali (Ribociclib) - HR+/HER2- Breast Cancer

### Patient Population
- Adult women with advanced/metastatic HR+/HER2- breast cancer
- Treatment setting: First-line and subsequent lines
- Combined with aromatase inhibitors or fulvestrant

### HCP Segments
| Segment | Count | Characteristics |
|---------|-------|-----------------|
| High-volume oncologists | 2,500 | Drive majority of volume |
| Community oncologists | 8,000 | Broader reach needed |
| Nurse navigators | 3,500 | Influence patient journey |
| Emerging HCPs | 5,000 | Growth potential |

### Causal DAG Drivers
```
HCP Targeting → Rep Engagement → Brand Perception → NRx
Patient Support → Adherence Program Enrollment → PDC → TRx
Early Detection → Diagnosis → Treatment Starts → NRx
```

### Key Competitors
- **Ibrance (palbociclib)**: First-to-market CDK4/6
- **Verzenio (abemaciclib)**: Aggressive positioning

### Historical Experiment Benchmarks
| Experiment | Effect Size | 95% CI |
|------------|-------------|--------|
| Q2 2024 HCP targeting pilot | +18% NRx | [12%, 24%] |
| Q3 2024 nurse navigator program | +8% persistence | [5%, 11%] |

### KPI Targets
| KPI | Target |
|-----|--------|
| NRx Growth YoY | +15% |
| Adherence (PDC) | >80% |
| Market Share (CDK4/6) | >35% |

---

## Fabhalta (Iptacopan) - Paroxysmal Nocturnal Hemoglobinuria (PNH)

### Patient Population
- Adults with PNH (~5,000 US patients total)
- Includes C5 inhibitor-experienced patients
- Mechanism: Factor B inhibitor (proximal complement)

### HCP Segments
| Segment | Count | Characteristics |
|---------|-------|-----------------|
| PNH specialists | 200 | High volume, KOLs |
| Transplant centers | 150 | Referral hubs |
| Community hematologists | 1,500 | Diagnosis source |
| Rare disease centers | 100 | Academic expertise |

### Specialty Concentration Rule
**Top 20% of specialists drive 80% of rare disease volume.**

Use this when:
- Prioritizing HCP targeting
- Designing experiments (cluster by specialist)
- Interpreting market dynamics

### Causal DAG Drivers
```
Disease Awareness → Diagnosis Rate → Referral → PNH Specialist
HCP Education → Treatment Selection → Fabhalta Start
Adherence → Hemoglobin Response → Transfusion Avoidance
```

### Key Competitors
- **Soliris (eculizumab)**: Established C5 inhibitor
- **Ultomiris (ravulizumab)**: Long-acting C5 inhibitor

### Statistical Considerations
- Small population requires:
  - Bayesian methods for underpowered studies
  - Cluster randomization by center
  - Careful multiple comparison adjustment

### KPI Targets
| KPI | Target |
|-----|--------|
| Patient starts (Year 1) | 500+ |
| Switch rate from C5i | 30% |
| Hemoglobin normalization | >70% |
| Transfusion avoidance | >80% |

---

## Remibrutinib - Chronic Spontaneous Urticaria (CSU)

### Patient Population
- Adults with CSU inadequately controlled on H1 antihistamines
- Treatment setting: Second-line after antihistamine failure
- Mechanism: BTK inhibitor

### HCP Segments
| Segment | Count | Characteristics |
|---------|-------|-----------------|
| Allergists/immunologists | 3,000 | Primary prescribers |
| Dermatologists | 5,000 | Significant volume |
| Primary care | 20,000 | Diagnosis source |
| Academic centers | 100 | KOL influence |

### Causal DAG Drivers
```
HCP Education → Disease Awareness → Patient Identification
Antihistamine Failure → Biologic Consideration → Treatment Decision
Access/Coverage → Payer Approval → Prescribing Decision
Efficacy Experience → Persistence → Long-term TRx
```

### Key Competitors
- **Xolair (omalizumab)**: Current standard, established
- **Fenebrutinib**: Pipeline BTK competitor
- **Ligelizumab**: Pending approval

### Launch Phase Dynamics
Apply these rules for launch-phase analysis:

1. **Early Adopter Effect**: KOLs and academic centers adopt 2× faster
2. **Payer Dynamics**: Prior authorization creates natural experiments
3. **Competitive Stockouts**: Temporary competitor shortages cause shifts (confound)

### Brand-Specific KPIs
| KPI | Target |
|-----|--------|
| AH uncontrolled % | ≤40% |
| Intent-to-prescribe change | ≥0.5 points |
| Market penetration (CSU biologics) | 25% Year 1 |
| UAS7 control | >40% |
| Persistence | >70% |
```

---

### File: `.claude/skills/pharma-commercial/patient-journey.md`

```yaml
---
name: Patient Journey Analysis
version: 1.0
description: 7-stage patient journey funnel analysis procedures
triggers:
  - patient journey
  - funnel analysis
  - conversion funnel
  - patient stages
agents:
  - gap_analyzer
  - prediction_synthesizer
  - explainer
---

# Patient Journey Analysis

## 7-Stage Patient Journey Funnel

```
┌─────────────────────────────────────────────────────────────────┐
│                        PATIENT JOURNEY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐   ┌─────────────┐   ┌────────────┐   ┌──────────┐ │
│  │ AWARE   │ → │ CONSIDERING │ → │ PRESCRIBED │ → │FIRST FILL│ │
│  │         │   │             │   │            │   │          │ │
│  └─────────┘   └─────────────┘   └────────────┘   └──────────┘ │
│       ↓                                               ↓         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                        ADHERENT                              ││
│  │                  (PDC ≥ 80%, active therapy)                 ││
│  └─────────────────────────────────────────────────────────────┘│
│       │                                                 │       │
│       ↓                                                 ↓       │
│  ┌────────────┐                              ┌──────────────┐   │
│  │DISCONTINUED│                              │  MAINTAINED  │   │
│  │            │                              │   (>12 mo)   │   │
│  └────────────┘                              └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Stage Definitions

| Stage | Definition | Key Metrics |
|-------|------------|-------------|
| **Aware** | Patient knows about brand/condition | Awareness surveys, HCP mentions |
| **Considering** | Actively evaluating treatment | HCP discussions logged, samples |
| **Prescribed** | Has active prescription | NRx, prescription written |
| **First Fill** | Prescription dispensed | Fill rate, days to first fill |
| **Adherent** | Active therapy, PDC ≥ 80% | PDC, MPR, refill rates |
| **Discontinued** | Stopped therapy | Discontinuation rate, reason codes |
| **Maintained** | Long-term use (>12 months) | Persistence, long-term TRx |

## Stage Transition Analysis

### Key Transitions to Measure

| Transition | Metric | Target |
|------------|--------|--------|
| Aware → Considering | Consideration Rate | Varies by brand |
| Considering → Prescribed | Prescription Rate | ≥30% |
| Prescribed → First Fill | Fill Rate | ≥85% |
| First Fill → Adherent | Adherence Rate | ≥80% |
| Adherent → Maintained | Persistence Rate | ≥70% at 12mo |

### Drop-off Analysis

For each stage transition, calculate:

```python
drop_off_rate = 1 - (next_stage_patients / current_stage_patients)
```

Prioritize interventions where:
- Drop-off rate > 20%
- Revenue impact × drop-off rate is highest
- Intervention is actionable

## Treatment Line Classification

Overlay clinical treatment lines on patient journey:

| Clinical Stage | Patient Journey Mapping |
|----------------|------------------------|
| Diagnosis | → Aware |
| Treatment-naive | Aware → Considering |
| First-line | Considering → Prescribed → First Fill |
| Second-line | (after switch) Prescribed → First Fill |
| Maintenance | Adherent → Maintained |
| Discontinuation | → Discontinued |
| Switch | Discontinued → (competitor) First Fill |

## Attribution Windows by Stage

| Transition | Attribution Window | Rationale |
|------------|-------------------|-----------|
| Trigger → NRx | 21 days | Rep influence period |
| HCP Visit → Consideration | 14 days | Immediate impact |
| Prescription → Fill | 30 days | Standard fulfillment |
| Fill → Adherence | 180 days | Standard observation |

## Segment-Specific Journey Analysis

Different segments have different journey patterns:

### By HCP Type
| HCP Type | Journey Acceleration | Notes |
|----------|---------------------|-------|
| Specialists | Faster Rx → Fill | Familiar with process |
| Primary Care | Slower consideration | May refer out |
| Academic | Very fast early stages | KOL influence |

### By Patient Type
| Patient Type | Key Bottleneck | Intervention |
|--------------|----------------|--------------|
| Treatment-naive | Considering → Prescribed | Education |
| Switch patients | Fill rate | Payer navigation |
| C5i-experienced (PNH) | Trust building | Clinical data |

## Funnel Visualization Requirements

When generating journey visualizations:

1. **Sankey diagrams** for stage-to-stage flow
2. **Bar charts** for drop-off rates by stage
3. **Time-series** for cohort progression
4. Label each stage with:
   - Patient count
   - Transition rate
   - Median days in stage
```

---

## Causal Inference Skills

### File: `.claude/skills/causal-inference/confounder-identification.md`

```yaml
---
name: Confounder Identification for Pharma Analytics
version: 1.0
description: Standard confounders and instrumental variables for pharma causal analysis
triggers:
  - identify confounders
  - causal analysis setup
  - confounding variables
  - instrumental variables
agents:
  - causal_impact
  - experiment_designer
---

# Confounder Identification for Pharma Analytics

## Standard Confounders by Analysis Type

### HCP Targeting → Prescription Impact

Always control for these confounders when analyzing HCP targeting effects:

| Confounder | Type | Rationale |
|------------|------|-----------|
| Territory potential | Continuous | High-potential territories get more targeting AND more Rx |
| HCP specialty | Categorical | Specialists vs generalists differ in targeting and prescribing |
| HCP volume (baseline) | Continuous | High-volume HCPs are targeted more AND write more Rx |
| Practice type | Categorical | Academic vs community affects access and prescribing |
| Payer mix | Continuous | Favorable payer mix → easier access → more targeting AND Rx |
| Geographic region | Categorical | Regional variations in both targeting and prescribing |
| Prior brand usage | Binary | Historical users get more attention AND continue prescribing |

### Patient Journey → Outcome Analysis

| Confounder | Type | Rationale |
|------------|------|-----------|
| Disease severity | Ordinal | Affects treatment selection and outcomes |
| Comorbidities | Count/categorical | Influence both treatment choice and adherence |
| Age | Continuous | Affects engagement, adherence, outcomes |
| Insurance type | Categorical | Access affects journey and outcomes |
| Prior treatments | Count | Treatment history affects next choice and response |
| Socioeconomic factors | Ordinal | Influence adherence and outcome measurement |

### Trigger → Conversion Analysis

| Confounder | Type | Rationale |
|------------|------|-----------|
| Patient severity score | Continuous | Severe patients get more triggers AND convert more |
| Time since last visit | Continuous | Recent visits → more triggers AND more conversion |
| HCP relationship strength | Ordinal | Strong relationships → more triggers AND better response |
| Trigger type | Categorical | Different trigger types have different baseline rates |
| Day of week | Categorical | Timing affects both triggering and rep action |
| Competing triggers | Count | Multiple triggers compete for attention |

---

## Instrumental Variables

Use these when confounding is severe and randomization is impossible:

| Instrument | Affects | Does NOT Directly Affect | Use Case |
|------------|---------|-------------------------|----------|
| Rep turnover | HCP visit frequency | Prescription decisions | HCP engagement → Rx |
| Weather events | Patient visits | Treatment efficacy | Visit → Adherence |
| Policy changes | Coverage availability | Clinical outcomes | Access → Outcomes |
| Competitor stockouts | Competitor availability | Brand preference | Competitor → Switch |
| Distance to HCP | Access to care | Disease severity | Access → Outcomes |
| Conference timing | HCP education | Patient disease state | Education → Prescribing |

### Validity Checks for Instruments

Before using an instrument, verify:

1. **Relevance**: Strong correlation with treatment (F-stat > 10)
2. **Exclusion restriction**: No direct effect on outcome
3. **Independence**: Uncorrelated with unmeasured confounders

---

## Confounder Selection Process

### Step 1: List All Plausible Confounders

Start with the standard lists above, then add domain-specific confounders.

### Step 2: Check Data Availability

For each confounder, verify:
- [ ] Available in data sources
- [ ] Measured at correct time point (pre-treatment)
- [ ] Sufficient variation for adjustment

### Step 3: Assess Importance

Prioritize confounders by:
- **Effect size** on treatment assignment
- **Effect size** on outcome
- **Correlation** with other confounders (avoid over-adjustment)

### Step 4: Check for Colliders

**Never** adjust for:
- Variables caused by both treatment and outcome (colliders)
- Post-treatment variables
- Mediators (unless doing mediation analysis)

**Collider Example**:
```
Treatment → Side Effects ← Outcome
```
Adjusting for side effects opens a biasing path.

### Step 5: Document Assumptions

Record:
- Confounders included and rationale
- Confounders excluded and why
- Unmeasured confounders that may remain

---

## Brand-Specific Confounder Considerations

### Kisqali (Oncology)
Additional confounders:
- Prior CDK4/6 inhibitor use
- Line of therapy
- ECOG performance status
- Tumor characteristics

### Fabhalta (Rare Disease)
Additional confounders:
- Prior C5 inhibitor exposure
- Transfusion history
- Specialist access (geography)
- Clinical trial participation

### Remibrutinib (Launch Phase)
Additional confounders:
- Early adopter bias
- Payer coverage status
- Competitor availability
- KOL influence (proximity to academic center)

---

## Sensitivity to Unmeasured Confounding

After adjustment, always report:

### E-value
```
E-value = RR + sqrt(RR × (RR - 1))
```
Where RR is the risk ratio of the observed effect.

**Interpretation**: How strong would an unmeasured confounder need to be to explain away the effect?

### Reporting Template
```
The observed effect of [treatment] on [outcome] was [effect size].
E-value = [X], meaning an unmeasured confounder would need to be
associated with both treatment and outcome by a factor of [X] to
fully explain away this effect. Given the observed effect of known
confounders (max RR = [Y]), this threshold [is/is not] plausible.
```
```

---

### File: `.claude/skills/causal-inference/dowhy-workflow.md`

```yaml
---
name: DoWhy Causal Estimation Workflow
version: 1.0
description: End-to-end DoWhy/EconML workflow with pharma-specific procedures
triggers:
  - causal estimation
  - DoWhy analysis
  - effect estimation
  - causal impact
agents:
  - causal_impact
---

# DoWhy Causal Estimation Workflow

## Overview

```
Query → DAG Construction → Estimation → Refutation → Sensitivity → Interpretation
```

---

## Phase 1: DAG Construction

### Input Requirements
- Treatment variable (what we're testing)
- Outcome variable (what we're measuring)
- Identified confounders (from confounder-identification.md)
- Time ordering (cause must precede effect)

### DoWhy Graph Specification

```python
from dowhy import CausalModel

# Build causal graph
model = CausalModel(
    data=df,
    treatment='hcp_targeting_intensity',
    outcome='nrx_volume',
    common_causes=['territory_potential', 'hcp_specialty',
                   'baseline_volume', 'payer_mix'],
    instruments=['rep_turnover'],  # If using IV
    effect_modifiers=['hcp_segment'],  # For CATE analysis
)
```

### Graph Validation Checklist
- [ ] All confounders are pre-treatment
- [ ] No colliders included
- [ ] No mediators (unless mediation analysis)
- [ ] Time ordering is correct
- [ ] Effect modifiers identified for heterogeneity

---

## Phase 2: Estimation with Fallback Chain

### Energy Score Selection (V4.2)

Instead of first-success, use energy score to select best estimator:

```
Energy Score = 0.35 × Treatment Balance + 0.45 × Outcome Fit + 0.20 × Propensity Calibration
```

### Quality Tiers

| Score Range | Quality | Action |
|-------------|---------|--------|
| ≤0.25 | Excellent | Use with high confidence |
| ≤0.45 | Good | Use with normal confidence |
| ≤0.65 | Acceptable | Use with caveats |
| ≤0.80 | Poor | Consider alternative methods |
| >0.80 | Unreliable | Do not use without redesign |

### Estimator Priority Chain

Run all estimators, score each, select best:

1. **Causal Forest DML (EconML)** - Best for heterogeneous effects
   ```python
   from econml.dml import CausalForestDML
   cf = CausalForestDML(model_y=model_y, model_t=model_t)
   cf.fit(Y, T, X, W)
   ate = cf.ate(X)
   ```

2. **Linear DML** - Good for linear relationships
   ```python
   from econml.dml import LinearDML
   ldml = LinearDML(model_y=model_y, model_t=model_t)
   ldml.fit(Y, T, X, W)
   ```

3. **Backdoor Linear Regression** - Simple, interpretable
   ```python
   estimate = model.estimate_effect(
       identified_estimand,
       method_name="backdoor.linear_regression"
   )
   ```

4. **Propensity Score Weighting** - When overlap is good
   ```python
   estimate = model.estimate_effect(
       identified_estimand,
       method_name="backdoor.propensity_score_weighting"
   )
   ```

### Positivity Check

Before estimation, verify treatment overlap:
```python
# Check propensity score distribution
ps = model.estimate_propensity()
overlap_ok = (ps.min() > 0.05) and (ps.max() < 0.95)
```

If overlap is poor, consider:
- Trimming extreme propensity scores
- Using matching instead of weighting
- Restricting to common support region

---

## Phase 3: Refutation Testing

Run ALL refutation tests in parallel:

### 1. Placebo Treatment Test
```python
res_placebo = model.refute_estimate(
    identified_estimand, estimate,
    method_name="placebo_treatment_refuter"
)
# Effect should be ~0 with random treatment
```
**Pass criteria**: |effect| < 0.1 × original effect

### 2. Random Common Cause Test
```python
res_random = model.refute_estimate(
    identified_estimand, estimate,
    method_name="random_common_cause"
)
# Effect should be stable with random confounder
```
**Pass criteria**: |change| < 10% of original effect

### 3. Data Subset Test
```python
res_subset = model.refute_estimate(
    identified_estimand, estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8
)
# Effect should be stable across subsets
```
**Pass criteria**: Effect within original 95% CI

### 4. Bootstrap Validation
```python
res_bootstrap = model.refute_estimate(
    identified_estimand, estimate,
    method_name="bootstrap_refuter",
    num_simulations=100
)
```

### Refutation Summary

| Test | Pass | Fail Action |
|------|------|-------------|
| Placebo | Effect ≈ 0 | Check for data leakage |
| Random cause | Stable | Add more confounders |
| Subset | Stable | Check for outliers |
| Bootstrap | Narrow CI | Get more data |

---

## Phase 4: Sensitivity Analysis

### E-value Calculation

```python
from dowhy.causal_estimators.sensitivity_analysis import EValueSensitivityAnalysis

sens = EValueSensitivityAnalysis()
e_value = sens.compute_e_value(estimate)
```

### Interpretation Framework

| E-value | Interpretation |
|---------|----------------|
| < 1.5 | Easily explained by unmeasured confounding |
| 1.5-2.0 | Moderate robustness |
| 2.0-3.0 | Good robustness |
| > 3.0 | Strong robustness |

### Benchmark Against Observed Confounders

Compare E-value to strongest observed confounder:
```
If E-value > max(observed confounder effects) × 1.5:
    → Effect is likely robust
Else:
    → Unmeasured confounding is plausible concern
```

---

## Phase 5: Interpretation by Audience

### For Executives
```
Key Finding: [Treatment] causes a [X%] increase in [Outcome].
Business Impact: This represents [$Y] in additional revenue.
Confidence: [High/Medium/Low] based on robustness checks.
Recommendation: [Specific action to take].
```

### For Analysts
```
Causal Effect: ATE = [X] (95% CI: [lower, upper])
Method: [Estimator used] selected via energy score ([score])
Robustness: [X/4] refutation tests passed
Sensitivity: E-value = [Y], suggesting [interpretation]
Limitations: [Key assumption concerns]
```

### For Data Scientists
```
Estimation: ATE = [X] ± [SE], p < [p-value]
Estimator: [Method] with energy score [score]
  - Treatment balance: [score component]
  - Outcome fit: [score component]
  - Propensity calibration: [score component]
Identification: Backdoor criterion satisfied via [confounders]
Refutation Results:
  - Placebo: effect = [X] (expected ~0)
  - Random cause: Δ = [X]% (expected <10%)
  - Subset: within CI = [Yes/No]
  - Bootstrap: SE = [X]
Sensitivity: E-value = [Y]
  - Required confounder strength: RR > [Z]
  - Observed max confounder: RR = [W]
  - Assessment: [Robust/Concerning]
Assumptions at risk: [Specific concerns]
```

---

## Validation Gates (V4.1)

Before proceeding to interpretation, verify:

| Gate | Criterion | Action if Failed |
|------|-----------|------------------|
| Positivity | Propensity ∈ [0.05, 0.95] | Trim or match |
| Refutation | ≥3/4 tests pass | Investigate failures |
| E-value | > 1.5 | Report with strong caveats |
| Energy score | ≤ 0.65 | Consider alternative design |
```

---

## Experiment Design Skills

### File: `.claude/skills/experiment-design/validity-threats.md`

```yaml
---
name: Experiment Validity Threat Assessment
version: 1.0
description: 6-threat taxonomy for experiment design validation
triggers:
  - validity threats
  - experiment validation
  - internal validity
  - external validity
agents:
  - experiment_designer
---

# Experiment Validity Threat Assessment

## 6-Threat Taxonomy

### 1. Selection Bias

**Definition**: Non-random assignment creates pre-existing differences between groups.

**Manifestations in Pharma**:
- High-volume HCPs self-select into programs
- Early adopters differ from mainstream
- Healthier patients enroll in studies

**Detection Methods**:
- Compare baseline characteristics (t-tests, chi-square)
- Check balance on key confounders
- Examine enrollment patterns

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Randomization | Gold standard, when feasible |
| Stratified randomization | When key confounders are known |
| Matching | When randomization isn't possible |
| Propensity score weighting | Observational settings |
| Regression discontinuity | Natural thresholds exist |

**Mitigation Specificity Required**: Don't just say "randomize." Specify:
- Randomization unit (patient, HCP, territory)
- Stratification variables
- Block size
- Allocation ratio

---

### 2. Confounding

**Definition**: Unmeasured variables affect both treatment and outcome.

**Manifestations in Pharma**:
- Territory potential affects targeting AND outcomes
- HCP motivation affects engagement AND prescribing
- Market dynamics affect interventions AND results

**Detection Methods**:
- Sensitivity analysis (E-value)
- Negative control outcomes
- Instrumental variable tests

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Measure and adjust | Confounder is measurable |
| Instrumental variables | Valid instrument available |
| Difference-in-differences | Pre-post data available |
| Regression discontinuity | Sharp threshold exists |
| Synthetic controls | Single treated unit |

---

### 3. Measurement Error

**Definition**: Outcome or treatment is measured with error.

**Manifestations in Pharma**:
- Data source lag creates incomplete outcomes
- Attribution windows miss delayed effects
- Self-reported outcomes have bias

**Detection Methods**:
- Compare across data sources
- Test-retest reliability
- Examine measurement timing

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Multiple data sources | Cross-validation possible |
| Longer observation windows | Effects may be delayed |
| Objective outcomes | Avoid self-report bias |
| Standardized definitions | Ensure consistency |

---

### 4. Contamination

**Definition**: Control group is exposed to treatment.

**Manifestations in Pharma**:
- Control HCPs learn from treated colleagues
- Patients switch between HCPs
- National campaigns affect both groups

**Detection Methods**:
- Measure treatment exposure in control
- Geographic spillover analysis
- Network analysis for information spread

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Geographic separation | Clusters are distinct |
| Cluster randomization | Prevent within-cluster spillover |
| Waitlist control | Eventual treatment for all |
| Intent-to-treat analysis | Despite contamination |

---

### 5. Temporal Effects

**Definition**: Time-varying factors affect outcomes independent of treatment.

**Manifestations in Pharma**:
- Seasonality in respiratory conditions
- Competitor launches during study
- Policy changes affecting access
- COVID-19 effects on healthcare

**Detection Methods**:
- Examine outcome trends pre-intervention
- Identify concurrent events
- Test for time × treatment interactions

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Concurrent control | Separates time from treatment |
| Difference-in-differences | Pre-post comparison |
| Interrupted time series | Long baseline available |
| Event study design | Sharp intervention timing |

---

### 6. Attrition

**Definition**: Differential dropout between groups.

**Manifestations in Pharma**:
- Non-responders drop out faster
- Side effects cause selective attrition
- Loss to follow-up in rare disease

**Detection Methods**:
- Compare attrition rates between groups
- Analyze attrition by baseline characteristics
- Test for differential attrition on outcomes

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Intent-to-treat analysis | Primary analysis |
| Multiple imputation | Handle missing data |
| Sensitivity analysis | Bound effects under attrition |
| Per-protocol secondary | Complier effects |

---

## Validity Scoring Framework

### For Each Threat, Assess:

1. **Likelihood** (1-5)
   - 1: Very unlikely
   - 3: Possible
   - 5: Very likely

2. **Severity** (1-5)
   - 1: Minor bias
   - 3: Moderate bias
   - 5: Severe bias, could invalidate results

3. **Mitigation Quality** (1-5)
   - 1: No mitigation
   - 3: Partial mitigation
   - 5: Fully mitigated

### Validity Score Calculation

```
Threat Score = Likelihood × Severity / Mitigation Quality

Overall Validity = Σ(Threat Scores) / Number of Threats
```

### Interpretation

| Overall Score | Validity | Recommendation |
|---------------|----------|----------------|
| < 5 | Strong | Proceed |
| 5-10 | Moderate | Proceed with caution |
| 10-15 | Weak | Redesign recommended |
| > 15 | Unacceptable | Do not proceed |

---

## Validity Audit Template

```markdown
## Validity Threat Assessment: [Experiment Name]

### 1. Selection Bias
- Likelihood: [1-5] - [Rationale]
- Severity: [1-5] - [Rationale]
- Mitigation: [Description]
- Mitigation Quality: [1-5]
- Residual Concern: [None/Low/Medium/High]

### 2. Confounding
[Same structure]

### 3. Measurement Error
[Same structure]

### 4. Contamination
[Same structure]

### 5. Temporal Effects
[Same structure]

### 6. Attrition
[Same structure]

### Overall Assessment
- Validity Score: [X]
- Classification: [Strong/Moderate/Weak]
- Recommendation: [Proceed/Caution/Redesign]
- Key Risks: [Top 2-3 concerns]
```
```

---

## Gap Analysis Skills

### File: `.claude/skills/gap-analysis/roi-estimation.md`

```yaml
---
name: ROI Estimation Procedures
version: 1.0
description: Revenue impact and cost-to-close calculation methodology
triggers:
  - ROI calculation
  - revenue impact
  - cost to close
  - opportunity sizing
agents:
  - gap_analyzer
  - resource_optimizer
---

# ROI Estimation Procedures

## Revenue Impact Calculation

### Standard Revenue Multipliers

| Metric | Revenue per Unit | Rationale |
|--------|------------------|-----------|
| TRx | $500 | Average prescription value |
| NRx | $400 | Slightly lower (first fills) |
| NBRx | $600 | High-value brand switch |
| Conversion Rate (1%) | $50,000 | Based on funnel math |
| Market Share (1%) | $500,000 | Category-level impact |
| HCP Reach (1%) | $5,000 | Incremental engagement |
| Persistence (1 month) | $500 × patient_count | Extended TRx |

### Revenue Impact Formula

```
Revenue Impact = Gap Size × Metric Multiplier × Confidence Adjustment
```

Where:
- Gap Size: Current vs Target/Benchmark/Potential
- Metric Multiplier: From table above
- Confidence Adjustment: 0.7 base, modified by evidence strength

### Confidence Adjustments

| Factor | Adjustment |
|--------|------------|
| Gap vs target (strategic) | +0.10 |
| Gap > 20% | +0.10 |
| Prior evidence exists | +0.15 |
| Execution risk high | -0.10 |
| Market uncertainty | -0.15 |

---

## Cost-to-Close Calculation

### Cost Formulas by Intervention Type

| Intervention | Cost Formula | Components |
|--------------|--------------|------------|
| HCP Reach | $150 × gap × 2 | Rep time + materials |
| Conversion Improvement | $1,000 × gap × 400 | Training + incentives + tools |
| Market Share Gain | $10,000 × gap | Competitive displacement cost |
| Adherence Program | $200 × patients × gap | Program costs + support |
| General Improvement | $150 × gap | Default estimate |

### Cost Components

**Fixed Costs** (per initiative):
- Program design: $5,000 - $20,000
- Technology setup: $2,000 - $10,000
- Training: $1,000 - $5,000

**Variable Costs** (per unit):
- Rep time: $100/hour
- Materials: $50/HCP
- Patient support: $200/patient/month

### Implementation Difficulty

| Difficulty | Thresholds | Multiplier |
|------------|------------|------------|
| Low | Cost < $10K, gap < 10% | 1.0 |
| Medium | Cost $10K-$50K, gap 10-20% | 1.3 |
| High | Cost > $50K, gap > 20% | 1.6 |

Apply multiplier to base cost:
```
Adjusted Cost = Base Cost × Difficulty Multiplier
```

---

## ROI Calculation

### Standard Formula

```
ROI = (Revenue Impact - Cost to Close) / Cost to Close
```

### Interpretation

| ROI | Assessment | Action |
|-----|------------|--------|
| > 5.0 | Excellent | High priority |
| 3.0 - 5.0 | Strong | Recommended |
| 1.5 - 3.0 | Good | Consider |
| 1.0 - 1.5 | Marginal | Low priority |
| < 1.0 | Negative | Do not pursue |

### Target ROI
**Minimum acceptable**: 3.0 (300% return)

---

## Payback Period

### Formula

```
Payback (months) = Cost to Close / (Revenue Impact / 12)
```

### Thresholds

| Payback | Assessment |
|---------|------------|
| < 6 months | Quick win |
| 6-12 months | Standard |
| 12-18 months | Strategic bet |
| > 18 months | Long-term |
| > 24 months | Cap at 24 (too uncertain) |

---

## Opportunity Categorization

### Quick Wins

**Criteria**:
- Implementation difficulty = Low
- ROI > 1.0
- Cost < $10,000
- Gap < 10%

**Characteristics**:
- Fast to implement (< 3 months)
- Low risk
- Immediate visibility

### Strategic Bets

**Criteria**:
- Implementation difficulty = High
- ROI > 2.0
- Cost > $50,000
- Gap > 20%

**Characteristics**:
- Significant investment
- Longer payback
- Transformational potential

### Optimization Plays

**Criteria**:
- Implementation difficulty = Medium
- ROI 1.5 - 3.0
- Moderate cost and gap

**Characteristics**:
- Incremental improvement
- Manageable risk
- Steady returns

---

## ROI Output Format

### For Each Opportunity

```markdown
### Opportunity: [Name]

**Gap Analysis**:
- Current: [X]
- Target: [Y]
- Gap: [Z] ([Z%])
- Gap Type: vs_target | vs_benchmark | vs_potential | vs_prior

**Revenue Impact**:
- Metric: [TRx/NRx/etc.]
- Multiplier: $[X]
- Raw Impact: $[Y]
- Confidence Adjustment: [Z]
- Adjusted Impact: $[Final]

**Cost to Close**:
- Intervention: [Description]
- Base Cost: $[X]
- Difficulty: [Low/Medium/High]
- Multiplier: [X]
- Adjusted Cost: $[Final]

**ROI Metrics**:
- ROI: [X%]
- Payback: [X months]
- Category: [Quick Win/Strategic Bet/Optimization]

**Recommendation**: [Specific action with context]
```

---

## Segment-Level Analysis

For each gap, calculate segment-specific ROI:

```python
for segment in segments:
    segment_gap = current[segment] - target[segment]
    segment_revenue = segment_gap × multiplier × segment_weight
    segment_cost = base_cost × segment_difficulty[segment]
    segment_roi = (segment_revenue - segment_cost) / segment_cost
```

### Prioritization Matrix

| Segment | Gap Size | ROI | Priority |
|---------|----------|-----|----------|
| High-volume specialists | [X%] | [Y] | [1-5] |
| Community HCPs | [X%] | [Y] | [1-5] |
| Emerging HCPs | [X%] | [Y] | [1-5] |

Rank by: ROI × Gap Size × Strategic Importance
```

---

## Integration Pattern

Now let me create the integration pattern document showing how agents load and use these skills.
<function_calls>
<invoke name="Write">
<parameter name="content"># Skill Integration Patterns for E2I Agents

**Date**: 2025-01-25
**Version**: 1.0
**Purpose**: How E2I agents load and use domain skills

---

## Overview

This document describes how E2I agents integrate with the domain skills system. Skills provide procedural knowledge that agents load on-demand to maintain context efficiency.

---

## Skill Loading Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Execution                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  Task Input  │ →  │ Skill Matcher│ →  │ Skill Loader │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                              │                    │             │
│                              ▼                    ▼             │
│                       ┌──────────────┐    ┌──────────────┐     │
│                       │ Skill Index  │    │ Full Skill   │     │
│                       │ (metadata)   │    │ (procedures) │     │
│                       │ ~50 tokens   │    │ ~500 tokens  │     │
│                       └──────────────┘    └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent-Skill Mapping

### Deep Agents (Tier 5)

#### `feedback_learner`
**Primary Skills**:
| Skill | Load Trigger | Purpose |
|-------|--------------|---------|
| `pharma-commercial/kpi-calculation.md` | KPI-related feedback | Validate feedback against KPI definitions |
| `causal-inference/refutation-testing.md` | Causal feedback | Evaluate causal claim quality |
| `data-quality/label-quality.md` | Label disputes | Apply edge case taxonomy |

**Skill Usage Pattern**:
```python
class FeedbackLearnerAgent:
    async def process_feedback(self, feedback: Feedback):
        # Load skill based on feedback domain
        if feedback.relates_to_kpi():
            skill = await self.load_skill("pharma-commercial/kpi-calculation.md")
            validation_rules = skill.get_section("Prescription Volume Metrics")

        # Apply skill procedures
        validated = self.apply_validation(feedback, validation_rules)
```

#### `explainer`
**Primary Skills**:
| Skill | Load Trigger | Purpose |
|-------|--------------|---------|
| `pharma-commercial/brand-analytics.md` | Brand-specific explanation | Get brand context |
| `causal-inference/dowhy-workflow.md` | Causal explanation | Use interpretation templates |
| `pharma-commercial/patient-journey.md` | Funnel explanation | Journey stage context |

**Skill Usage Pattern**:
```python
class ExplainerAgent:
    async def generate_explanation(self, analysis: Analysis, user_level: str):
        # Load brand context
        brand_skill = await self.load_skill("pharma-commercial/brand-analytics.md")
        brand_context = brand_skill.get_section(analysis.brand)

        # Load interpretation template
        causal_skill = await self.load_skill("causal-inference/dowhy-workflow.md")
        template = causal_skill.get_section(f"For {user_level.title()}")

        # Generate explanation using skill procedures
        return self.apply_template(template, analysis, brand_context)
```

---

### Hybrid Agents (Tiers 2-3)

#### `causal_impact`
**Primary Skills**:
| Skill | Load Trigger | Purpose |
|-------|--------------|---------|
| `causal-inference/confounder-identification.md` | DAG construction | Standard confounders |
| `causal-inference/dowhy-workflow.md` | Estimation | Procedure guide |
| `pharma-commercial/brand-analytics.md` | Brand analysis | Brand-specific confounders |

**Skill Usage Pattern**:
```python
class CausalImpactAgent:
    async def build_dag(self, treatment: str, outcome: str, brand: str):
        # Load confounder skill
        conf_skill = await self.load_skill("causal-inference/confounder-identification.md")

        # Get analysis-type confounders
        if "hcp" in treatment.lower():
            confounders = conf_skill.get_section("HCP Targeting → Prescription Impact")
        elif "patient" in treatment.lower():
            confounders = conf_skill.get_section("Patient Journey → Outcome Analysis")

        # Add brand-specific confounders
        brand_skill = await self.load_skill("pharma-commercial/brand-analytics.md")
        brand_confounders = brand_skill.get_brand_confounders(brand)

        return confounders + brand_confounders
```

#### `experiment_designer`
**Primary Skills**:
| Skill | Load Trigger | Purpose |
|-------|--------------|---------|
| `experiment-design/validity-threats.md` | Validity audit | Threat taxonomy |
| `experiment-design/power-analysis.md` | Sample sizing | Power procedures |
| `pharma-commercial/brand-analytics.md` | Brand context | Historical benchmarks |

**Skill Usage Pattern**:
```python
class ExperimentDesignerAgent:
    async def audit_validity(self, design: ExperimentDesign):
        # Load threat taxonomy
        validity_skill = await self.load_skill("experiment-design/validity-threats.md")

        # Assess each threat
        assessments = []
        for threat in ["Selection Bias", "Confounding", "Measurement Error",
                       "Contamination", "Temporal Effects", "Attrition"]:
            threat_spec = validity_skill.get_section(threat)
            assessment = self.assess_threat(design, threat_spec)
            assessments.append(assessment)

        # Calculate overall validity score
        scoring = validity_skill.get_section("Validity Scoring Framework")
        return self.score_validity(assessments, scoring)
```

#### `gap_analyzer`
**Primary Skills**:
| Skill | Load Trigger | Purpose |
|-------|--------------|---------|
| `gap-analysis/roi-estimation.md` | ROI calculation | Multipliers and formulas |
| `pharma-commercial/kpi-calculation.md` | Gap detection | KPI definitions |
| `pharma-commercial/brand-analytics.md` | Brand context | Brand targets |

**Skill Usage Pattern**:
```python
class GapAnalyzerAgent:
    async def estimate_roi(self, gap: Gap):
        # Load ROI procedures
        roi_skill = await self.load_skill("gap-analysis/roi-estimation.md")

        # Get multiplier for metric
        multipliers = roi_skill.get_section("Standard Revenue Multipliers")
        multiplier = multipliers.get(gap.metric, multipliers["General"])

        # Calculate revenue impact
        revenue_section = roi_skill.get_section("Revenue Impact Formula")
        revenue = self.apply_formula(revenue_section, gap, multiplier)

        # Calculate cost
        cost_section = roi_skill.get_section("Cost Formulas by Intervention Type")
        cost = self.apply_formula(cost_section, gap)

        # Categorize
        categorization = roi_skill.get_section("Opportunity Categorization")
        return self.categorize(revenue, cost, categorization)
```

---

### Standard Agents (Tier 0)

#### `cohort_constructor`
**Primary Skills**:
| Skill | Load Trigger | Purpose |
|-------|--------------|---------|
| `pharma-commercial/patient-journey.md` | Stage filtering | Journey definitions |
| `data-quality/label-quality.md` | Edge cases | Labeling rules |

#### `data_preparer`
**Primary Skills**:
| Skill | Load Trigger | Purpose |
|-------|--------------|---------|
| `data-quality/data-source-lag.md` | Source selection | Lag handling |
| `data-quality/drift-detection.md` | Data validation | PSI procedures |

---

## Skill Loading Implementation

### SkillLoader Class

```python
# src/skills/loader.py

from pathlib import Path
from typing import Optional
import yaml

class SkillLoader:
    """Loads and caches domain skills for agents."""

    SKILLS_DIR = Path(".claude/skills")
    _cache: dict[str, Skill] = {}

    @classmethod
    async def load(cls, skill_path: str) -> Skill:
        """Load a skill by path, using cache if available."""
        if skill_path in cls._cache:
            return cls._cache[skill_path]

        full_path = cls.SKILLS_DIR / skill_path
        if not full_path.exists():
            raise SkillNotFoundError(f"Skill not found: {skill_path}")

        content = full_path.read_text()
        skill = cls._parse_skill(content)
        cls._cache[skill_path] = skill
        return skill

    @classmethod
    def _parse_skill(cls, content: str) -> Skill:
        """Parse skill markdown with YAML frontmatter."""
        # Split frontmatter from content
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            body = parts[2]
        else:
            frontmatter = {}
            body = content

        return Skill(
            metadata=SkillMetadata(**frontmatter),
            content=body,
            sections=cls._parse_sections(body)
        )

    @classmethod
    def _parse_sections(cls, body: str) -> dict[str, str]:
        """Parse markdown sections into a dictionary."""
        sections = {}
        current_section = None
        current_content = []

        for line in body.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = line[3:].strip()
                current_content = []
            elif line.startswith("### "):
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = line[4:].strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content)

        return sections


@dataclass
class SkillMetadata:
    name: str
    version: str
    description: str
    triggers: list[str]
    agents: list[str]


@dataclass
class Skill:
    metadata: SkillMetadata
    content: str
    sections: dict[str, str]

    def get_section(self, name: str) -> str:
        """Get a specific section by name."""
        return self.sections.get(name, "")

    def matches_trigger(self, query: str) -> bool:
        """Check if query matches any skill trigger."""
        query_lower = query.lower()
        return any(trigger.lower() in query_lower
                   for trigger in self.metadata.triggers)
```

### Agent Base Class Integration

```python
# src/agents/base.py

class BaseAgent:
    """Base class for all E2I agents with skill loading."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.skill_loader = SkillLoader()
        self._loaded_skills: list[str] = []

    async def load_skill(self, skill_path: str) -> Skill:
        """Load a skill and track it."""
        skill = await self.skill_loader.load(skill_path)
        self._loaded_skills.append(skill_path)
        return skill

    async def find_matching_skills(self, query: str) -> list[Skill]:
        """Find all skills matching a query."""
        matching = []
        for skill_path in self._get_relevant_skill_paths():
            skill = await self.skill_loader.load(skill_path)
            if skill.matches_trigger(query):
                matching.append(skill)
        return matching

    def _get_relevant_skill_paths(self) -> list[str]:
        """Get skill paths relevant to this agent."""
        # Override in subclasses
        return []
```

---

## Skill Matching Heuristics

### Query-to-Skill Mapping

```python
SKILL_KEYWORDS = {
    "pharma-commercial/kpi-calculation.md": [
        "trx", "nrx", "prescription", "market share", "conversion",
        "roi", "adherence", "persistence", "pdc"
    ],
    "pharma-commercial/brand-analytics.md": [
        "kisqali", "fabhalta", "remibrutinib", "brand", "competitor",
        "ibrance", "verzenio", "soliris", "xolair"
    ],
    "causal-inference/confounder-identification.md": [
        "confounder", "confounding", "adjustment", "control for",
        "causal", "bias"
    ],
    "causal-inference/dowhy-workflow.md": [
        "causal effect", "ate", "cate", "dowhy", "estimation",
        "refutation", "sensitivity"
    ],
    "experiment-design/validity-threats.md": [
        "validity", "bias", "selection", "contamination", "attrition",
        "experiment design", "threats"
    ],
    "gap-analysis/roi-estimation.md": [
        "roi", "revenue", "cost", "opportunity", "gap", "payback"
    ],
}

def match_skills(query: str) -> list[str]:
    """Match query to relevant skills."""
    query_lower = query.lower()
    scores = {}

    for skill_path, keywords in SKILL_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[skill_path] = score

    # Return top 3 by score
    sorted_skills = sorted(scores.items(), key=lambda x: -x[1])
    return [path for path, _ in sorted_skills[:3]]
```

---

## Token Efficiency

### Metadata-First Loading

Skills are designed for two-phase loading:

1. **Phase 1: Metadata Only** (~50 tokens)
   - Skill name, triggers, agent list
   - Used for matching and routing

2. **Phase 2: Full Content** (~500-1000 tokens)
   - Complete procedures
   - Loaded only when matched

### Section-Based Access

Instead of loading entire skills, agents can load specific sections:

```python
# Full skill load (~1000 tokens)
skill = await self.load_skill("causal-inference/dowhy-workflow.md")

# Section-only load (~200 tokens)
section = skill.get_section("Phase 3: Refutation Testing")
```

### Context Window Budget

| Agent Type | Max Skill Tokens | Typical Skills Loaded |
|------------|------------------|----------------------|
| Deep (Tier 5) | 3,000 | 3-5 full skills |
| Hybrid (Tiers 2-3) | 2,000 | 2-3 full skills |
| Standard (Tier 0-1) | 1,000 | 1-2 skills or sections |

---

## MCP + Skills Integration

When using MCP tools alongside skills:

```python
class CausalImpactAgent:
    async def run(self, query: Query):
        # 1. Load procedural skills
        confounder_skill = await self.load_skill(
            "causal-inference/confounder-identification.md"
        )

        # 2. Get external data via MCP
        if self.needs_external_evidence(query):
            pubmed_results = await self.mcp_gateway.call(
                server="pubmed",
                tool="search_literature",
                params={"query": query.topic}
            )

        # 3. Apply skill procedures with MCP data
        confounders = confounder_skill.get_section(
            "Standard Confounders by Analysis Type"
        )

        # 4. Execute analysis
        return self.execute_with_context(
            confounders=confounders,
            external_evidence=pubmed_results
        )
```

---

## Testing Skills

### Skill Validation Tests

```python
# tests/unit/test_skills.py

import pytest
from src.skills.loader import SkillLoader

class TestSkillLoading:
    async def test_load_kpi_calculation(self):
        skill = await SkillLoader.load("pharma-commercial/kpi-calculation.md")

        assert skill.metadata.name == "KPI Calculation Procedures"
        assert "TRx" in skill.get_section("Prescription Volume Metrics")
        assert "gap_analyzer" in skill.metadata.agents

    async def test_skill_matching(self):
        skill = await SkillLoader.load("pharma-commercial/brand-analytics.md")

        assert skill.matches_trigger("Kisqali analysis")
        assert skill.matches_trigger("What about Fabhalta?")
        assert not skill.matches_trigger("general question")

    async def test_section_extraction(self):
        skill = await SkillLoader.load("causal-inference/dowhy-workflow.md")

        refutation = skill.get_section("Phase 3: Refutation Testing")
        assert "Placebo Treatment Test" in refutation
        assert "Random Common Cause Test" in refutation
```

### Agent-Skill Integration Tests

```python
# tests/integration/test_agent_skills.py

class TestCausalImpactSkills:
    async def test_confounder_loading(self):
        agent = CausalImpactAgent()

        confounders = await agent.get_confounders(
            treatment="hcp_targeting",
            outcome="nrx",
            brand="kisqali"
        )

        # Should include standard HCP confounders
        assert "territory_potential" in confounders
        assert "hcp_specialty" in confounders

        # Should include brand-specific confounders
        assert "prior_cdk46_use" in confounders
```

---

## Summary

The skill integration pattern provides:

1. **On-demand loading**: Skills loaded only when needed
2. **Token efficiency**: Metadata-first, section-based access
3. **Agent-specific mapping**: Each agent knows its relevant skills
4. **MCP complementarity**: Skills provide "how", MCP provides "what"
5. **Testability**: Skills can be validated independently

This approach encodes pharma commercial analytics procedures in a maintainable, efficient format that agents can leverage without context bloat.
