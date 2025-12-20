# E2I Confidence Score Methodology

**Version:** 1.0
**Date:** 2025-12-15
**Purpose:** Define the methodology for calculating confidence scores for causal impact estimates

---

## Overview

This document specifies the mathematical foundations, calculation procedures, and interpretation guidelines for **confidence scores** assigned to causal impact estimates in the E2I Causal Analytics platform.

Confidence scores quantify the reliability of causal effect estimates by integrating:
1. **Sample size adequacy** - Statistical power considerations
2. **Temporal stability** - Consistency over time
3. **Effect consistency** - Homogeneity across subgroups
4. **Validation strength** - Robustness to sensitivity tests

**Range:** Confidence scores are normalized to [0, 1] and displayed as percentages (0% - 98%).

**Integration:** Confidence scores feed into ROI calculations, alert thresholds, and result display decisions.

---

## Core Confidence Formula

### Weighted Average Method (Default)

The confidence score is calculated as a **weighted average** of four components:

```
confidence = (w₁ × sample_size_factor) +
             (w₂ × temporal_stability) +
             (w₃ × effect_consistency) +
             (w₄ × validation_strength)

where: w₁ = w₂ = w₃ = w₄ = 0.25 (equal weights)
```

**Example:**
- Sample size factor: 0.85
- Temporal stability: 0.78
- Effect consistency: 0.82
- Validation strength: 0.90

**Base confidence** = (0.25 × 0.85) + (0.25 × 0.78) + (0.25 × 0.82) + (0.25 × 0.90)
                    = 0.2125 + 0.1950 + 0.2050 + 0.2250
                    = **0.8375** or **83.8%**

### Geometric Mean Method (Alternative)

For situations requiring all components to be reasonably strong:

```
confidence = (sample_size_factor × temporal_stability ×
              effect_consistency × validation_strength)^(1/4)
```

This method is **more conservative** - a single weak component has greater impact.

**Same example with geometric mean:**
- Base confidence = (0.85 × 0.78 × 0.82 × 0.90)^0.25 = (0.4907)^0.25 = **0.833** or **83.3%**

---

## Component 1: Sample Size Factor

### Purpose

Quantifies whether the sample is large enough to reliably detect the causal effect.

### Formula

```
sample_size_factor = min(1.0, √(effective_n / n_required))

where:
  effective_n = (n_treatment × n_control) / (n_treatment + n_control)
  n_required = threshold based on entity type (HCP, patient, prescription, territory)
```

### Entity-Specific Thresholds

| Entity Type | Minimum | Recommended | High Confidence |
|-------------|---------|-------------|-----------------|
| HCP | 30 | 100 | 300 |
| Patient | 100 | 500 | 2,000 |
| Prescription | 500 | 2,000 | 10,000 |
| Territory | 20 | 50 | 150 |

**Interpretation:**
- **effective_n ≥ n_required** → sample_size_factor = 1.0
- **effective_n < n_required** → sample_size_factor = √(effective_n / n_required)

### Balance Penalty

If treatment and control groups are **imbalanced**, apply a penalty:

```
balance_ratio = min(n_treatment, n_control) / max(n_treatment, n_control)

if balance_ratio < 0.5:
    sample_size_factor *= (0.5 + 0.5 × balance_ratio)
```

**Example:**
- n_treatment = 150, n_control = 50
- balance_ratio = 50 / 150 = 0.33
- Penalty multiplier = 0.5 + (0.5 × 0.33) = 0.665
- If base sample_size_factor = 0.80, adjusted = 0.80 × 0.665 = **0.53**

### Examples

**Example 1: HCP analysis with balanced groups**
- n_treatment = 200 HCPs
- n_control = 180 HCPs
- effective_n = (200 × 180) / (200 + 180) = 94.7
- n_required = 300 (high confidence threshold)
- sample_size_factor = √(94.7 / 300) = √0.316 = **0.56**

**Example 2: Patient analysis with large sample**
- n_treatment = 2,500 patients
- n_control = 2,300 patients
- effective_n = (2,500 × 2,300) / (2,500 + 2,300) = 1,198
- n_required = 2,000
- sample_size_factor = √(1,198 / 2,000) = √0.599 = 0.774
- **Capped at 1.0** → **1.00** (sample exceeds requirement)

---

## Component 2: Temporal Stability

### Purpose

Measures consistency of the causal effect across time periods. Highly variable effects reduce confidence.

### Formula

```
temporal_stability = 1 / (1 + CV_temporal)

where:
  CV_temporal = σ / |μ|  (Coefficient of Variation)
  σ = standard deviation of effects across time periods
  μ = mean effect across time periods
```

### Time Windows

Analysis is split into time periods based on granularity:

| Analysis Granularity | Time Window | Minimum Periods |
|---------------------|-------------|-----------------|
| Daily | Weekly | 4 |
| Weekly | Monthly | 3 |
| Monthly | Quarterly | 3 |
| Quarterly | Yearly | 2 |

### Interpretation Thresholds

| Temporal Stability | CV Range | Interpretation |
|-------------------|----------|----------------|
| ≥ 0.80 | CV < 0.25 | High stability |
| 0.60 - 0.79 | CV 0.25 - 0.67 | Moderate stability |
| 0.40 - 0.59 | CV 0.67 - 1.5 | Low stability |
| < 0.40 | CV ≥ 1.5 | Unstable |

### Examples

**Example 1: Stable effect over 4 quarters**
- Q1: +12% TRx lift
- Q2: +14% TRx lift
- Q3: +11% TRx lift
- Q4: +13% TRx lift
- μ = 12.5%, σ = 1.29%
- CV = 1.29 / 12.5 = 0.103
- temporal_stability = 1 / (1 + 0.103) = **0.91** (high stability)

**Example 2: Volatile effect over 4 quarters**
- Q1: +5% TRx lift
- Q2: +18% TRx lift
- Q3: -2% TRx lift
- Q4: +15% TRx lift
- μ = 9.0%, σ = 8.87%
- CV = 8.87 / 9.0 = 0.985
- temporal_stability = 1 / (1 + 0.985) = **0.50** (low stability)

### Trend Adjustment

If effect shows a **clear linear trend** (e.g., effect increasing over time), adjust stability upward:

```
if R² > 0.70 (strong linear trend):
    temporal_stability += 0.10
```

**Rationale:** A trending effect is predictable, even if it varies period-to-period.

---

## Component 3: Effect Consistency

### Purpose

Quantifies homogeneity of the causal effect across subgroups (region, HCP tier, specialty, etc.). Heterogeneous effects reduce confidence.

### Formula

Based on the **Higgins I² statistic** from meta-analysis:

```
effect_consistency = exp(-I² / 100)

where:
  I² = ((Q - df) / Q) × 100
  Q = Cochran's Q statistic = Σ wᵢ(θᵢ - θ̄)²
  df = degrees of freedom = k - 1 (k = number of subgroups)
  wᵢ = inverse variance weight for subgroup i
  θᵢ = effect estimate for subgroup i
  θ̄ = weighted average effect
```

### Subgroups Analyzed

**Required subgroups:**
- **Region:** Northeast, South, Midwest, West
- **HCP Tier:** Tier 1, Tier 2, Tier 3, Tier 4

**Optional subgroups:**
- **Specialty:** Allergy, Hematology, Oncology, Cardiology, Rheumatology, Immunology
- **Patient Segment:** New vs existing patients
- **Seasonality:** Q1, Q2, Q3, Q4

### Interpretation Thresholds

| I² Value | Effect Consistency | Interpretation |
|----------|-------------------|----------------|
| < 25% | ≥ 0.85 | Homogeneous (low heterogeneity) |
| 25% - 50% | 0.65 - 0.84 | Moderate heterogeneity |
| 50% - 75% | 0.45 - 0.64 | Substantial heterogeneity |
| ≥ 75% | < 0.45 | High heterogeneity |

### Examples

**Example 1: Consistent effect across regions**
- Northeast: +10% TRx lift (SE = 2%)
- South: +12% TRx lift (SE = 2.5%)
- Midwest: +11% TRx lift (SE = 2.2%)
- West: +9% TRx lift (SE = 2.8%)
- I² calculated = 18% (low heterogeneity)
- effect_consistency = exp(-0.18) = **0.84**

**Example 2: Heterogeneous effect across HCP tiers**
- Tier 1: +25% TRx lift (high prescribers)
- Tier 2: +8% TRx lift
- Tier 3: +2% TRx lift
- Tier 4: -1% TRx lift (no effect)
- I² calculated = 82% (high heterogeneity)
- effect_consistency = exp(-0.82) = **0.44**

**Implication:** In Example 2, the effect is real for Tier 1 HCPs but not generalizable. Confidence is reduced, and heterogeneous effects should be reported explicitly.

---

## Component 4: Validation Strength

### Purpose

Aggregates results from **DoWhy refutation tests** (Gap 3) to quantify robustness of the causal estimate to bias.

### Formula

```
validation_strength = Σ(test_weight × test_score) / Σ(test_weight)

where:
  test_score ∈ {1.0, 0.5, 0.0} for {proceed, review, block}
```

### Test Weights

Based on `config/causal_validation_config.yaml`:

| Refutation Test | Weight |
|----------------|--------|
| placebo_treatment_refuter | 0.30 |
| random_common_cause_refuter | 0.25 |
| data_subset_refuter | 0.20 |
| bootstrap_refuter | 0.15 |
| sensitivity_e_value | 0.10 |

### Outcome Scores

| Validation Outcome | Score | Meaning |
|-------------------|-------|---------|
| Proceed | 1.0 | Test passed - effect robust |
| Review | 0.5 | Borderline - manual review needed |
| Block | 0.0 | Test failed - effect not robust |

### Examples

**Example 1: All tests passed**
- placebo_treatment_refuter: proceed (1.0 × 0.30 = 0.30)
- random_common_cause_refuter: proceed (1.0 × 0.25 = 0.25)
- data_subset_refuter: proceed (1.0 × 0.20 = 0.20)
- bootstrap_refuter: proceed (1.0 × 0.15 = 0.15)
- sensitivity_e_value: proceed (1.0 × 0.10 = 0.10)
- **validation_strength = 1.00**

**Example 2: Mixed results**
- placebo_treatment_refuter: proceed (1.0 × 0.30 = 0.30)
- random_common_cause_refuter: review (0.5 × 0.25 = 0.125)
- data_subset_refuter: proceed (1.0 × 0.20 = 0.20)
- bootstrap_refuter: review (0.5 × 0.15 = 0.075)
- sensitivity_e_value: block (0.0 × 0.10 = 0.00)
- **validation_strength = 0.70**

**Example 3: Multiple failures**
- placebo_treatment_refuter: block (0.0 × 0.30 = 0.00)
- random_common_cause_refuter: review (0.5 × 0.25 = 0.125)
- data_subset_refuter: block (0.0 × 0.20 = 0.00)
- bootstrap_refuter: proceed (1.0 × 0.15 = 0.15)
- sensitivity_e_value: block (0.0 × 0.10 = 0.00)
- **validation_strength = 0.275**

**Implication:** Example 3 should likely not be shown to users (confidence below 30% threshold after reductions).

---

## Confidence Reduction Rules

After calculating the **base confidence**, apply **multiplicative penalties** for known issues.

### Application Formula

```
adjusted_confidence = base_confidence × ∏(1 - reduction_i)

capped at: max cumulative reduction = 60%
```

### Reduction Rules

| Condition | Reduction | Reason |
|-----------|-----------|--------|
| High heterogeneity (effect_consistency < 0.45) | -20% | Substantial heterogeneity across subgroups |
| Missing critical covariates (coverage < 80%) | -15% | Key covariates missing or incomplete |
| Failed sensitivity analysis (E-value blocked) | -25% | Effect not robust to unmeasured confounding |
| Small sample (effective_n < 0.5 × n_required) | -20% | Sample size below recommended threshold |
| Imbalanced groups (balance_ratio < 0.33) | -15% | Severe imbalance between treatment/control |
| High temporal variation (stability < 0.40) | -20% | Effect varies substantially over time |
| Low data quality (quality_score < 70) | -15% | Low quality score from source data |
| Excessive missing data (missing_rate > 15%) | -10% | High proportion of missing values |
| Short observation period (< 90 days) | -15% | Observation period too short |
| Multiple testing (n_hypotheses > 5, no correction) | -10% | Multiple comparisons without adjustment |

### Examples

**Example 1: Single reduction**
- base_confidence = 0.80
- small_sample = true (reduction = 0.20)
- adjusted_confidence = 0.80 × (1 - 0.20) = **0.64**

**Example 2: Multiple reductions**
- base_confidence = 0.75
- high_heterogeneity = true (reduction = 0.20)
- low_data_quality = true (reduction = 0.15)
- Total reduction = 1 - [(1 - 0.20) × (1 - 0.15)] = 1 - 0.68 = 0.32
- adjusted_confidence = 0.75 × (1 - 0.32) = **0.51**

**Example 3: Hitting maximum reduction cap**
- base_confidence = 0.85
- Four major issues trigger reductions: -20%, -25%, -20%, -15%
- Total reduction calculated = 60.8%, but capped at 60%
- adjusted_confidence = 0.85 × (1 - 0.60) = **0.34**
- Floor applied: max(0.34, 0.10) = **0.34**

---

## Confidence Boost Rules

After applying reductions, apply **additive bonuses** for particularly strong evidence.

### Application Formula

```
boosted_confidence = reduced_confidence + Σ(boost_i)

capped at: maximum ceiling = 0.98
```

### Boost Rules

| Condition | Boost | Reason |
|-----------|-------|--------|
| Randomized design (RCT) | +15% | Randomization eliminates selection bias |
| Perfect validation (strength ≥ 0.95) | +10% | All refutation tests passed |
| Large sample (effective_n > 3 × n_required) | +10% | Sample size well exceeds requirements |
| Perfect consistency (consistency ≥ 0.90) | +8% | Effect homogeneous across subgroups |
| Perfect stability (stability ≥ 0.90) | +8% | Effect highly stable over time |
| External validation (independent replication) | +12% | Effect replicated in external dataset |
| Strong theory (causal mechanism known) | +5% | Established causal mechanism supports finding |

### Examples

**Example 1: RCT with large sample**
- reduced_confidence = 0.72
- randomized_design = true (+0.15)
- large_sample = true (+0.10)
- boosted_confidence = 0.72 + 0.15 + 0.10 = **0.97**

**Example 2: Observational study with replication**
- reduced_confidence = 0.65
- external_validation = true (+0.12)
- perfect_stability = true (+0.08)
- boosted_confidence = 0.65 + 0.12 + 0.08 = **0.85**

**Example 3: Exceeding maximum ceiling**
- reduced_confidence = 0.88
- Multiple boosts totaling +0.25
- boosted_confidence = 0.88 + 0.25 = 1.13, capped at **0.98**

---

## Final Confidence Score

After computing base confidence, applying reductions, and applying boosts:

```
final_confidence = clamp(boosted_confidence, minimum=0.10, maximum=0.98)

Display as: final_confidence × 100 (percentage, 1 decimal place)
```

**Why cap at 98%?**
Never claim absolute certainty in observational causal inference. Even RCTs have implementation issues, compliance problems, and measurement error.

**Why floor at 10%?**
Below 10% confidence, the estimate is effectively noise and should not be shown to users (see display thresholds).

---

## Confidence Tiers and Display

### Tier Classification

| Tier | Confidence Range | Color | Interpretation |
|------|-----------------|-------|----------------|
| **High Confidence** | ≥ 75% | Green (#22c55e) | Actionable - proceed with implementation |
| **Moderate Confidence** | 50% - 74% | Amber (#f59e0b) | Consider carefully - validate assumptions |
| **Low Confidence** | 30% - 49% | Red (#ef4444) | Exploratory only - collect more data |
| **Below Threshold** | < 30% | Gray (hidden) | Do not display to users |

### Display Thresholds

**Minimum to display:** 30%
Results with confidence < 30% are **not shown** in the dashboard.

**Alert thresholds (Gap 11):**
- Confidence drop ≥ 15 points → Trigger alert
- Any result with confidence < 40% → Trigger low confidence alert

### Badge Display

High and moderate confidence results display a **badge** with the confidence tier and percentage.

**Example:**
```
┌─────────────────────────────────────────────┐
│ Causal Impact: +12% TRx Lift                │
│                                             │
│ [High Confidence 83.5%] ✓                   │
│                                             │
│ Sample Size: 250 HCPs                       │
│ Validation: 4/5 tests passed                │
└─────────────────────────────────────────────┘
```

Low confidence results display a **warning badge**:
```
┌─────────────────────────────────────────────┐
│ Causal Impact: +8% TRx Lift                 │
│                                             │
│ [Low Confidence 42.1%] ⚠                    │
│ Consider collecting more data               │
│                                             │
│ Sample Size: 45 HCPs (below recommended)    │
└─────────────────────────────────────────────┘
```

---

## Integration with Other Systems

### Causal Validation System (Gap 3)

**Source:** `causal_validations` table
**Field:** validation results for 5 DoWhy refutation tests
**Integration:** `validation_strength` component uses weighted test outcomes

**Gate Respect:**
If validation_gate_decision = "block", confidence is **automatically capped at 0.25** regardless of other factors.

### Data Sources Quality (Gap 7)

**Source:** `data_sources` table
**Field:** `quality_score` (0-100)
**Integration:** If quality_score < 70, apply -15% reduction

**Example:**
- Using IQVIA APLD (quality_score = 92.5): No penalty
- Using Komodo Health (quality_score = 65.3): -15% penalty applied

### ROI Methodology (Gap 9)

**Integration:** Confidence scores affect ROI confidence intervals

**Mechanism:**
- **High confidence** → Narrow CI in bootstrap simulations
- **Moderate confidence** → Normal CI
- **Low confidence** → Wide CI (2× spread)

**Example ROI adjustment:**
```
Base ROI: 5.2x
High confidence (85%): 95% CI = [4.5x, 6.1x]
Low confidence (38%): 95% CI = [2.8x, 8.9x]
```

### Alert System (Gap 11)

**Triggers:**
- Confidence drops by ≥15 points from previous analysis
- Any causal impact with confidence < 40%

**Alert severity:**
- Confidence < 30%: Critical alert
- Confidence 30-40%: Warning alert
- Confidence drop ≥20 points: Urgent alert

---

## Calculation Pipeline

### Step-by-Step Process

```
1. Calculate Components
   ├─ sample_size_factor(n_treatment, n_control, entity_type)
   ├─ temporal_stability(effects_by_period)
   ├─ effect_consistency(effects_by_subgroup)
   └─ validation_strength(validation_id)

2. Compute Base Confidence
   └─ base = (0.25 × SSF) + (0.25 × TS) + (0.25 × EC) + (0.25 × VS)

3. Apply Reductions
   └─ reduced = base × ∏(1 - reduction_i), capped at 60% total

4. Apply Boosts
   └─ boosted = reduced + Σ(boost_i), capped at +30% total

5. Finalize
   └─ final = clamp(boosted, min=0.10, max=0.98)

6. Classify Tier
   └─ tier = {High, Moderate, Low} based on thresholds

7. Store Results
   └─ INSERT INTO causal_impacts (confidence_score, confidence_tier, ...)
```

### Database Storage

**Table:** `causal_impacts`

**Fields:**
- `confidence_score` (DECIMAL) - Final confidence (0.10 - 0.98)
- `confidence_tier` (TEXT) - "high", "moderate", "low"
- `sample_size_factor` (DECIMAL) - Component score
- `temporal_stability` (DECIMAL) - Component score
- `effect_consistency` (DECIMAL) - Component score
- `validation_strength` (DECIMAL) - Component score
- `reductions_applied` (JSONB) - List of reductions and values
- `boosts_applied` (JSONB) - List of boosts and values

---

## Testing and Validation

### Unit Test Cases

**Test 1: Sample size factor with balanced groups**
```
Input: n_treatment=100, n_control=100, entity=hcp
Expected: 0.58
Actual: (100×100)/(100+100) = 50, √(50/300) = 0.58 ✓
```

**Test 2: Temporal stability with low CV**
```
Input: effects=[0.10, 0.12, 0.09, 0.11], mean=0.105
CV = 0.0133 / 0.105 = 0.127
Expected: 1/(1+0.127) = 0.89
```

**Test 3: Effect consistency with I²=15%**
```
Input: I²=15
Expected: exp(-0.15) = 0.86
```

### Integration Test

**Full pipeline test:**
```
Input:
  n_treatment=200, n_control=180
  effects_by_period=[0.12, 0.14, 0.13, 0.11]
  I²=22
  validation_strength=0.85

Expected range: [0.75, 0.85]

Calculation:
  SSF = √(94.7/300) = 0.56
  TS = 1/(1+0.103) = 0.91
  EC = exp(-0.22) = 0.80
  VS = 0.85
  base = 0.25(0.56+0.91+0.80+0.85) = 0.78
  (no reductions/boosts)
  final = 0.78 ✓
```

---

## Error Handling

### Missing Data

| Issue | Action | Fallback |
|-------|--------|----------|
| Sample size missing | Set to minimum | confidence = 0.20 |
| Temporal data missing | Use default | temporal_stability = 0.70 |
| Subgroup data missing | Skip consistency | effect_consistency = 0.60 |
| Validation data missing | Use default | validation_strength = 0.50 |

### Calculation Errors

| Error | Action |
|-------|--------|
| Division by zero | Return 0.0 |
| Negative confidence | Clamp to 0.10 |
| Confidence > 1.0 | Clamp to 0.98 |

### Logging

All confidence calculations are logged with:
- Input parameters
- Component scores
- Reductions/boosts applied
- Final confidence
- Timestamp and analysis_id

---

## Interpretation Guidelines

### For Data Scientists

**High Confidence (≥75%):**
- Sample size adequate for entity type
- Effect stable over time and consistent across subgroups
- Passed most/all validation tests
- Ready for production use and ROI forecasting

**Moderate Confidence (50-74%):**
- Some concerns about sample size, stability, or consistency
- Most validation tests passed, but 1-2 in "review" status
- Suitable for hypothesis generation and targeted experiments
- Recommend additional data collection or longer observation period

**Low Confidence (30-49%):**
- Significant issues with one or more components
- Failed sensitivity tests or high heterogeneity
- Exploratory analysis only - NOT actionable
- Requires investigation into root causes (confounding, data quality, etc.)

**Below Threshold (<30%):**
- Not displayed to users
- Likely spurious association or severe methodological issues
- Should not inform decision-making

### For Business Stakeholders

**High Confidence:**
- "We are confident this effect is real and can be acted upon"
- Green light for implementation and resource allocation
- Include in quarterly business reviews

**Moderate Confidence:**
- "We see evidence of an effect, but recommend caution"
- Yellow light - pilot programs or A/B tests recommended before full rollout
- Monitor closely in initial deployment

**Low Confidence:**
- "This finding is exploratory and should not drive decisions"
- Red light - do not allocate resources based on this result
- Consider as hypothesis for future research

---

## Example: Full Confidence Calculation

### Scenario

**Analysis:** Effect of personalized trigger redesign on HCP TRx prescribing

**Data:**
- Treatment group: 220 HCPs (received personalized triggers)
- Control group: 200 HCPs (received standard triggers)
- Observation period: 12 months (analyzed by quarter)
- Subgroups: 4 regions, 4 HCP tiers
- Validation: 5 DoWhy refutation tests

### Step 1: Calculate Components

**1.1 Sample Size Factor**
```
effective_n = (220 × 200) / (220 + 200) = 104.8
n_required = 300 (HCP high confidence threshold)
sample_size_factor = √(104.8 / 300) = 0.59

Balance check:
  balance_ratio = 200/220 = 0.91 (no penalty)
```

**1.2 Temporal Stability**
```
Effects by quarter:
  Q1: +8.2% TRx lift
  Q2: +10.1% TRx lift
  Q3: +9.4% TRx lift
  Q4: +7.8% TRx lift

Mean = 8.875%, SD = 1.02%
CV = 1.02 / 8.875 = 0.115

temporal_stability = 1 / (1 + 0.115) = 0.90
```

**1.3 Effect Consistency**
```
Subgroup analysis shows:
  - Region: I² = 18% (low heterogeneity)
  - HCP Tier: I² = 35% (moderate heterogeneity)

Average I² = 26.5%

effect_consistency = exp(-0.265) = 0.77
```

**1.4 Validation Strength**
```
DoWhy test results:
  - placebo_treatment: proceed (1.0 × 0.30 = 0.30)
  - random_common_cause: proceed (1.0 × 0.25 = 0.25)
  - data_subset: review (0.5 × 0.20 = 0.10)
  - bootstrap: proceed (1.0 × 0.15 = 0.15)
  - sensitivity_e_value: proceed (1.0 × 0.10 = 0.10)

validation_strength = 0.30 + 0.25 + 0.10 + 0.15 + 0.10 = 0.90
```

### Step 2: Compute Base Confidence

```
base_confidence = 0.25(0.59 + 0.90 + 0.77 + 0.90)
                = 0.25 × 3.16
                = 0.79 (79%)
```

### Step 3: Apply Reductions

**Check reduction rules:**
- High heterogeneity? No (effect_consistency = 0.77 > 0.45)
- Missing covariates? No
- Failed sensitivity? No
- Small sample? Yes (effective_n = 104.8 < 150) → **-20% reduction**
- Other issues? No

```
reduced_confidence = 0.79 × (1 - 0.20) = 0.632 (63.2%)
```

### Step 4: Apply Boosts

**Check boost rules:**
- RCT? No
- Perfect validation? No (0.90 < 0.95)
- Large sample? No
- Perfect stability? Yes (0.90 ≥ 0.90) → **+8% boost**
- Other boosts? No

```
boosted_confidence = 0.632 + 0.08 = 0.712 (71.2%)
```

### Step 5: Finalize

```
final_confidence = clamp(0.712, 0.10, 0.98) = 0.712 (71.2%)
```

### Step 6: Classify Tier

```
71.2% falls in [50%, 75%) → Moderate Confidence
```

### Output

```
┌─────────────────────────────────────────────────────────────┐
│ Causal Impact Analysis: Personalized Trigger Redesign      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Effect: +8.9% TRx Lift (95% CI: +6.2% to +11.8%)          │
│                                                             │
│ [Moderate Confidence 71.2%] ⚠                              │
│ Consider pilot testing before full rollout                  │
│                                                             │
│ Components:                                                 │
│  • Sample Size: 420 HCPs (below ideal threshold)           │
│  • Temporal Stability: High (90%)                          │
│  • Effect Consistency: Moderate (77%)                      │
│  • Validation Strength: Strong (90%)                       │
│                                                             │
│ Recommendation: Proceed with caution. Effect is real but   │
│ smaller sample size introduces uncertainty. Consider        │
│ expanding to more HCPs to reach high confidence.            │
└─────────────────────────────────────────────────────────────┘
```

---

## Update Schedule

This methodology is reviewed and updated:

- **Quarterly:** Validate thresholds against actual prediction performance
- **Annually:** Recalibrate component weights if needed
- **Ad-hoc:** When new validation methods are added or academic best practices change

---

## Governance

**Owner:** VP Analytics & Data Science
**Reviewers:** Lead Data Scientist, Head of Causal Analytics
**Approval Required For:** Changes to component weights, threshold adjustments >5%, new reduction/boost rules

---

## References

### Academic Literature

- Higgins, J. P. T., & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539-1558.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.
- Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing Type S (sign) and Type M (magnitude) errors. *Perspectives on Psychological Science*, 9(6), 641-651.

### Related Documentation

- `config/confidence_logic.yaml` - Configuration file
- `config/causal_validation_config.yaml` - Validation test specifications (Gap 3)
- `e2i_mlops/012_data_sources.sql` - Data quality tracking (Gap 7)
- `docs/roi_methodology.md` - ROI calculation (Gap 9)

---

*Last Updated: 2025-12-15 | E2I Causal Analytics V4.1 | Gap 12 Resolution*
