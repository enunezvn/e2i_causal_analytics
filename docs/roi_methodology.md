# E2I ROI Calculation Methodology

**Version:** 1.0
**Date:** 2025-12-15
**Purpose:** Define the methodology for calculating Return on Investment (ROI) for E2I initiatives

---

## Overview

This document specifies the formulas, value drivers, cost inputs, and confidence interval calculations used by the **Gap Analyzer Agent** to estimate ROI for causal insights and implementation recommendations.

---

## Core ROI Formula

### Basic Formula

```
ROI = (Incremental Value - Implementation Cost) / Implementation Cost
```

**Expressed as percentage:**
```
ROI% = ((Incremental Value - Implementation Cost) / Implementation Cost) × 100
```

### Example

- Incremental Value: $120,000
- Implementation Cost: $15,000
- ROI = ($120,000 - $15,000) / $15,000 = **7.0x** or **700%**

---

## Value Drivers

The **Incremental Value** is calculated based on the specific type of improvement identified. Each value driver has an associated dollar value per unit of improvement.

### 1. TRx Lift Value

**Definition:** Value generated per incremental total prescription (TRx)

**Formula:**
```
TRx Lift Value = Incremental TRx × $850
```

**Parameters:**
- **$850 per TRx:** Average net revenue per prescription after rebates, discounts, and channel costs
- Source: Brand-specific P&L models (updated quarterly)

**Example:**
- Causal analysis identifies potential for +200 TRx/month
- Annual value = 200 × 12 × $850 = **$2,040,000**

---

### 2. Patient Identification Value

**Definition:** Value generated per patient correctly identified for brand-appropriate therapy

**Formula:**
```
Patient ID Value = Identified Patients × $1,200
```

**Parameters:**
- **$1,200 per patient:** Lifetime value of patient identification (covers diagnostic confirmation, HCP engagement, patient support program enrollment)
- Assumes 60% conversion to TRx within 6 months
- Includes downstream adherence and refill value

**Example:**
- Improved algorithm identifies +300 eligible patients
- Value = 300 × $1,200 = **$360,000**

---

### 3. Action Rate Improvement Value

**Definition:** Value generated per percentage point improvement in HCP trigger acceptance

**Formula:**
```
Action Rate Value = Δ Action Rate (pp) × $45
```

**Parameters:**
- **$45 per percentage point:** Based on conversion funnel analysis
  - Baseline: 1,000 triggers/month × 25% acceptance = 250 actions
  - +1pp improvement: 1,000 × 26% = 260 actions = +10 actions/month
  - 10 actions × $450 avg value per action = $4,500/month = $54,000/year
  - Normalized: $54,000 / 12 months / 100 triggers = ~$45/pp

**Example:**
- Trigger redesign improves acceptance from 25% → 32% (+7pp)
- Annual value = 7 × $45 × 1,000 triggers × 12 = **$3,780,000**

---

### 4. Intent-to-Prescribe (ITP) Lift Value

**Definition:** Value generated per percentage point increase in HCP intent to prescribe

**Formula:**
```
ITP Lift Value = Δ ITP (pp) × $320 × HCP Count
```

**Parameters:**
- **$320 per HCP per percentage point:** Based on correlation between ITP surveys and subsequent prescribing behavior
  - 1pp ITP increase correlates with +0.4 TRx/HCP/year
  - 0.4 TRx × $850 = $340 gross value
  - Discounted for survey noise and lag: $340 × 0.94 = **$320**

**Example:**
- Omnichannel campaign increases average ITP by 3pp across 500 HCPs
- Annual value = 3 × $320 × 500 = **$480,000**

---

### 5. Data Quality Improvement Value

**Definition:** Value generated from reducing false positives/negatives in predictions

**Formula:**
```
DQ Value = (FP Reduction × $200) + (FN Reduction × $650)
```

**Parameters:**
- **$200 per false positive avoided:** Wasted rep time, customer annoyance, channel fatigue
- **$650 per false negative avoided:** Missed opportunity cost (patient not treated, competitor prescribes)

**Example:**
- Quality improvements reduce FP by 100/month and FN by 50/month
- Monthly value = (100 × $200) + (50 × $650) = $52,500
- Annual value = $52,500 × 12 = **$630,000**

---

### 6. Drift Prevention Value

**Definition:** Value generated from early detection and correction of model degradation

**Formula:**
```
Drift Prevention Value = Prevented Degradation (AUC drop) × Baseline Model Value × 2
```

**Parameters:**
- **2x multiplier:** Reflects cost of retraining, lost predictions during downtime, and business disruption

**Example:**
- Drift detection prevents 0.05 AUC drop on trigger model
- Baseline model generates $5M/year value
- Value = 0.05 × $5,000,000 × 2 = **$500,000**

---

## Cost Inputs

Implementation costs vary by initiative type. The following cost categories and unit costs are used:

### 1. Engineering Costs

**Day Rate:** $2,500/day (fully loaded)

**Typical Effort by Initiative Type:**

| Initiative Type | Engineering Days | Cost |
|----------------|------------------|------|
| Data source integration | 15-20 days | $37,500 - $50,000 |
| New ML model | 25-35 days | $62,500 - $87,500 |
| Algorithm optimization | 10-15 days | $25,000 - $37,500 |
| Dashboard enhancement | 5-10 days | $12,500 - $25,000 |
| Trigger redesign | 8-12 days | $20,000 - $30,000 |
| A/B test implementation | 12-18 days | $30,000 - $45,000 |

---

### 2. Data Acquisition Costs

**Costs vary by source:**

| Data Source | Annual Cost | Monthly Cost |
|-------------|-------------|--------------|
| IQVIA APLD | $150,000 | $12,500 |
| IQVIA LAAD | $120,000 | $10,000 |
| HealthVerity | $180,000 | $15,000 |
| Komodo Health | $200,000 | $16,667 |
| Veeva OCE | $80,000 | $6,667 |
| Specialty Pharmacy | $100,000 | $8,333 |

**Incremental Cost Calculation:**
- If initiative requires NEW data source: Full annual cost
- If initiative requires MORE data from existing source: 20% of annual cost
- If initiative uses existing data differently: $0 incremental cost

---

### 3. Training & Change Management Costs

**Cost per stakeholder group:**

| Stakeholder Group | Training Cost | Rollout Cost | Total |
|-------------------|---------------|--------------|-------|
| Sales reps (500) | $50/rep | $25/rep | $37,500 |
| HQ analytics (30) | $500/analyst | $200/analyst | $21,000 |
| Field managers (50) | $300/manager | $150/manager | $22,500 |

**Total Change Management Budget (major initiative):** ~$80,000

**Minor updates/enhancements:** $5,000 - $15,000

---

### 4. Infrastructure & Hosting Costs

**Monthly costs (annualize for ROI calculation):**

| Component | Monthly Cost | Annual Cost |
|-----------|--------------|-------------|
| Additional Supabase capacity | $500 - $2,000 | $6,000 - $24,000 |
| MLflow compute | $300 - $800 | $3,600 - $9,600 |
| Opik observability | $200 - $600 | $2,400 - $7,200 |
| Additional Anthropic API usage | $1,000 - $3,000 | $12,000 - $36,000 |

**Total Infrastructure:** $24,000 - $77,000/year for major initiative

---

### 5. Opportunity Cost

For initiatives that require pausing other work:

**Formula:**
```
Opportunity Cost = (Delayed Initiative Value / 12) × Delay Months
```

**Example:**
- Initiative A delays Initiative B (valued at $600K/year) by 2 months
- Opportunity cost = ($600,000 / 12) × 2 = **$100,000**

---

## Confidence Intervals

All ROI estimates include **95% confidence intervals** calculated using bootstrap sampling.

### Bootstrap Methodology

**Process:**

1. **Identify Key Uncertain Parameters:**
   - TRx lift estimate
   - Acceptance rate improvement
   - Implementation timeline
   - Adoption rate

2. **Define Distributions:**
   - Use historical data to parameterize distributions
   - For TRx lift: Normal distribution (μ = point estimate, σ = 0.15μ)
   - For acceptance rate: Beta distribution fit to historical A/B tests
   - For timeline: Gamma distribution (to capture right skew)

3. **Run Bootstrap Simulations:**
   - n = 1,000 simulations
   - Sample from each parameter distribution
   - Calculate ROI for each simulation
   - Sort results

4. **Calculate Confidence Interval:**
   - Lower bound: 2.5th percentile
   - Point estimate: 50th percentile (median)
   - Upper bound: 97.5th percentile

### Example Output

**Initiative:** Improve trigger acceptance through personalization

**Point Estimates:**
- Incremental Value: $3,780,000
- Implementation Cost: $45,000
- ROI: **83x**

**Bootstrap Results (1,000 simulations):**
- Median ROI: **82x**
- 95% CI: [**58x**, **112x**]
- Probability of positive ROI (>1x): **99.8%**
- Probability of target ROI (>50x): **91.2%**

---

## Sensitivity Analysis

For high-value initiatives (ROI > 10x or value > $1M), perform sensitivity analysis on key assumptions.

### Tornado Diagram Inputs

Vary each parameter ±20% and recalculate ROI to identify drivers:

| Parameter | -20% | Base | +20% | Impact Range |
|-----------|------|------|------|--------------|
| TRx Lift | 160 | 200 | 240 | ROI: 5.6x → 9.4x |
| Acceptance Rate Δ | 5.6pp | 7pp | 8.4pp | ROI: 6.1x → 8.9x |
| Eng Day Rate | $2,000 | $2,500 | $3,000 | ROI: 8.1x → 6.9x |
| TRx Unit Value | $680 | $850 | $1,020 | ROI: 5.7x → 9.3x |

**Interpretation:** TRx lift estimate is the primary driver of ROI uncertainty.

---

## Time Value of Money

For initiatives with benefits realized over multiple years, apply discounting.

### Discount Rate

**Corporate discount rate:** 10% annually

**Formula:**
```
PV = FV / (1 + r)^t

Where:
- PV = Present Value
- FV = Future Value
- r = Discount rate (0.10)
- t = Time in years
```

### Multi-Year ROI Calculation

**Example:** Initiative with 3-year benefit stream

| Year | Nominal Value | Discount Factor | Present Value |
|------|---------------|-----------------|---------------|
| 1 | $500,000 | 1.00 | $500,000 |
| 2 | $500,000 | 0.91 | $455,000 |
| 3 | $500,000 | 0.83 | $415,000 |
| **Total** | **$1,500,000** | - | **$1,370,000** |

**Implementation Cost:** $150,000 (Year 1)

**NPV ROI = ($1,370,000 - $150,000) / $150,000 = 8.1x**

---

## Causal Attribution

ROI calculations must account for causal attribution to avoid overclaiming.

### Attribution Framework

**Full Attribution (100%):**
- Initiative is sole driver
- Randomized controlled trial validates effect
- No confounding factors

**Partial Attribution (50-80%):**
- Initiative is primary driver
- Observational causal inference with strong identification
- Some confounding possible but controlled

**Shared Attribution (20-50%):**
- Multiple initiatives contribute
- Causal effects estimated with uncertainty
- External factors present

**Minimal Attribution (<20%):**
- Initiative is minor contributor
- Correlation-based evidence only
- High uncertainty

### Example

**Observed lift:** +200 TRx/month after trigger redesign

**Attribution analysis:**
1. **Trend analysis:** Market growing +50 TRx/month baseline
2. **Seasonality:** Q4 typically +30 TRx/month higher
3. **Confounding:** Competitor recall occurred (impact: +40 TRx/month)
4. **Attributed to initiative:** 200 - 50 - 30 - 40 = **80 TRx/month**

**Attribution rate:** 80 / 200 = **40%** (Shared Attribution)

**ROI calculation uses 80 TRx/month**, not 200.

---

## Risk Adjustment

For initiatives with execution risk, apply risk-adjusted ROI.

### Risk Factors

| Risk Factor | Adjustment |
|-------------|-----------|
| **Technical Complexity** | |
| - Low (proven tech) | 0% |
| - Medium (integration challenges) | -15% |
| - High (novel ML approach) | -30% |
| **Organizational Change** | |
| - Low (backend only) | 0% |
| - Medium (process change) | -20% |
| - High (behavioral change required) | -40% |
| **Data Dependencies** | |
| - Low (existing data) | 0% |
| - Medium (new data source) | -25% |
| - High (multiple new sources) | -50% |
| **Timeline Uncertainty** | |
| - Low (<3 months) | 0% |
| - Medium (3-6 months) | -10% |
| - High (>6 months) | -25% |

### Risk-Adjusted ROI Formula

```
Risk-Adjusted ROI = Base ROI × (1 - Total Risk Adjustment)

Total Risk Adjustment = 1 - ∏(1 - Individual Adjustments)
```

### Example

**Base ROI:** 8.0x

**Risk Factors:**
- Technical complexity: Medium (-15%)
- Organizational change: High (-40%)
- Data dependencies: Low (0%)
- Timeline: Medium (-10%)

**Total Risk Adjustment:**
= 1 - [(1 - 0.15) × (1 - 0.40) × (1 - 0) × (1 - 0.10)]
= 1 - [0.85 × 0.60 × 1.0 × 0.90]
= 1 - 0.459
= **54.1%**

**Risk-Adjusted ROI:** 8.0 × (1 - 0.541) = **3.7x**

---

## Reporting Standards

All ROI estimates presented to stakeholders must include:

### Required Components

1. **Point Estimate:** Median ROI from bootstrap
2. **Confidence Interval:** 95% CI bounds
3. **Time Horizon:** 1-year, 3-year, or lifetime
4. **Attribution Rate:** Percentage causally attributed
5. **Risk Adjustment:** Applied risk factors and adjustments
6. **Assumptions:** Key value drivers and costs
7. **Sensitivity:** Tornado diagram for top 3-5 drivers

### Example Summary

**Initiative:** Improve Remibrutinib patient identification algorithm

**ROI Summary:**
- **Base ROI:** 6.2x (5.1x - 7.8x at 95% CI)
- **Time Horizon:** 1-year
- **Attribution:** 60% (partial attribution)
- **Risk-Adjusted ROI:** 4.1x
- **Probability of Success:** 87%

**Key Assumptions:**
- +300 patients identified/year
- $1,200 value per patient
- 60% conversion to TRx
- $58,000 implementation cost

**Sensitivity:**
- Most sensitive to patient identification rate (±25% = ROI 3.8x to 4.4x)
- Moderately sensitive to conversion rate (±15% = ROI 3.9x to 4.3x)

---

## Update Schedule

This methodology is reviewed and updated:

- **Quarterly:** Value drivers updated based on actual performance
- **Annually:** Cost inputs refreshed with budget data
- **Ad-hoc:** When major business model changes occur (e.g., pricing changes, new indications)

---

## Governance

**Owner:** VP Analytics & Data Science
**Reviewers:** CFO, CMO, Head of Commercial Analytics
**Approval Required For:** Changes to value drivers >10%, new cost categories

---

## References

- Brand P&L models (Confidential)
- Historical A/B test results database
- Commercial analytics ROI tracker
- Finance cost accounting standards

---

*Last Updated: 2025-12-15 | E2I Causal Analytics V4.1 | Gap 9 Resolution*
