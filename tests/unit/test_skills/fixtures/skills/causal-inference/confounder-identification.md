---
name: Confounder Identification for Pharma Analytics
version: "1.0"
description: Systematic confounder identification for causal analysis in pharma
triggers:
  - identify confounders
  - confounder analysis
  - confounding variables
  - causal confounders
agents:
  - causal_impact
categories:
  - causal-inference
---

# Confounder Identification for Pharma Analytics

Systematic approach to identifying and controlling for confounders in pharmaceutical causal analysis.

## HCP Targeting → Prescription Impact

Common confounders when analyzing HCP targeting effectiveness:

- Territory potential (market size)
- HCP specialty (oncologist vs general practitioner)
- HCP prescribing volume baseline
- Payer mix in territory
- Competitive activity level

## Patient Demographics → Treatment Outcomes

Confounders in patient-level outcome analysis:

- Age and comorbidities
- Disease severity at baseline
- Prior treatment history
- Insurance coverage
- Geographic access to care

## Marketing Spend → Brand Performance

Confounders in marketing ROI analysis:

- Seasonal prescription patterns
- Competitor promotional activity
- Formulary changes
- Generic entry timing
- Sales force coverage changes

## Identification Methods

### Backdoor Criterion

Use DAG-based identification via the backdoor criterion to find sufficient adjustment sets.

### Instrumental Variables

When confounding is unmeasured, use instrumental variable approaches.

### Sensitivity Analysis

Always perform sensitivity analysis to assess robustness to unmeasured confounding.
