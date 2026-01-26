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
| Confounder selection | `causal-inference/confounder-identification.md` |
| Experiment design | `experiment-design/power-analysis.md`, `experiment-design/validity-threats.md` |
| Gap/opportunity analysis | `gap-analysis/gap-detection.md`, `gap-analysis/roi-estimation.md` |

## Domain Constraints

**This system IS**:
- Pharmaceutical commercial operations analytics ✅
- Business KPIs: TRx, NRx, conversion rates, market share ✅
- HCP targeting and patient journey analysis ✅

**This system IS NOT**:
- Clinical decision support ❌
- Medical literature search ❌
- Drug safety monitoring ❌

## Skill Categories

### pharma-commercial/
KPI calculations, brand analytics, patient journey analysis, HCP targeting.

### causal-inference/
Confounder identification, DoWhy workflow, refutation testing, CATE analysis.

### experiment-design/
Power analysis, validity threats, design selection, pre-registration.

### gap-analysis/
Gap detection, ROI estimation, opportunity prioritization.

### data-quality/
Drift detection, label quality, data source lag handling.
