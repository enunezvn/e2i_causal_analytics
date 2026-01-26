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
  - kpi calculation
  - revenue multiplier
agents:
  - gap_analyzer
  - prediction_synthesizer
  - explainer
  - health_score
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
