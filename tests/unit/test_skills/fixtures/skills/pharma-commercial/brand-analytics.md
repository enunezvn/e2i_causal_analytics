---
name: Brand-Specific Analytics
version: "1.0"
description: Brand-specific analytics procedures for Novartis pharma portfolio
triggers:
  - Kisqali analysis
  - Fabhalta analysis
  - Remibrutinib analysis
  - brand context
  - brand analytics
agents:
  - causal_impact
  - gap_analyzer
  - experiment_designer
categories:
  - oncology
  - rare-disease
  - immunology
---

# Brand-Specific Analytics

Analytics procedures tailored to specific brands in the Novartis pharma portfolio.

## Kisqali (Ribociclib)

Kisqali is a CDK4/6 inhibitor for HR+/HER2- metastatic breast cancer.

### Key Metrics

- TRx volume and market share in CDK4/6 inhibitor class
- New patient starts (NRx)
- Duration of therapy

### Brand-Specific Confounders

Key confounder variables for Kisqali causal analysis:
- Line of therapy
- Menopausal status
- Prior treatment history
- Payer mix

## Fabhalta (Iptacopan)

Fabhalta is a complement factor B inhibitor for PNH (paroxysmal nocturnal hemoglobinuria).

### Key Metrics

- Patient enrollment and market penetration
- Hemoglobin response rates
- Transfusion avoidance

### Brand-Specific Confounders

- Disease severity at baseline
- Prior complement inhibitor use
- Transfusion history

## Remibrutinib

Remibrutinib is a BTK inhibitor for CSU (chronic spontaneous urticaria).

### Key Metrics

- UAS7 score improvement
- Patient quality of life
- Market share in CSU category

### Brand-Specific Confounders

- Prior antihistamine response
- Disease duration
- Comorbidities
