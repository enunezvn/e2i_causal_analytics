---
name: ROI Estimation Procedures
version: 1.0
description: Revenue impact and cost-to-close calculation methodology
triggers:
  - ROI calculation
  - revenue impact
  - cost to close
  - opportunity sizing
  - payback period
  - investment justification
agents:
  - gap_analyzer
  - resource_optimizer
categories:
  - financial
  - prioritization
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

---

## Brand-Specific Multipliers

### Kisqali (Oncology)
| Metric | Multiplier | Rationale |
|--------|------------|-----------|
| NRx | $800 | Premium oncology |
| Market Share | $750,000 | CDK4/6 market |
| Persistence | $800/mo | Long-term therapy |

### Fabhalta (Rare Disease)
| Metric | Multiplier | Rationale |
|--------|------------|-----------|
| Patient Start | $15,000 | High-value rare disease |
| Switch from C5i | $12,000 | Conversion value |
| Transfusion Avoidance | $5,000 | Cost offset |

### Remibrutinib (CSU - Launch)
| Metric | Multiplier | Rationale |
|--------|------------|-----------|
| NRx | $600 | Specialty pricing |
| Intent-to-Prescribe | $10,000 | Per survey point |
| Market Penetration | $400,000 | Biologic category |

---

## Common Pitfalls

### 1. Overestimating Revenue Impact
**Problem**: Using optimistic multipliers
**Solution**: Use conservative estimates, apply confidence adjustments

### 2. Underestimating Costs
**Problem**: Missing hidden costs (training, change management)
**Solution**: Apply difficulty multipliers, include all cost components

### 3. Ignoring Execution Risk
**Problem**: Assuming perfect implementation
**Solution**: Factor execution track record into confidence adjustment

### 4. Double Counting
**Problem**: Same revenue counted in multiple opportunities
**Solution**: Identify overlap and adjust for cannibalization
