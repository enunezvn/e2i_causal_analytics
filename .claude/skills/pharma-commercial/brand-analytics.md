---
name: Brand-Specific Analytics
version: 1.0
description: Brand context and analytics rules for Kisqali, Fabhalta, Remibrutinib
triggers:
  - Kisqali analysis
  - Fabhalta analysis
  - Remibrutinib analysis
  - brand context
  - brand performance
  - competitive analysis
agents:
  - causal_impact
  - gap_analyzer
  - experiment_designer
  - explainer
categories:
  - oncology
  - rare-disease
  - immunology
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

| Competitor | Brand | Positioning |
|------------|-------|-------------|
| **Ibrance** | palbociclib | First-to-market CDK4/6, established |
| **Verzenio** | abemaciclib | Aggressive positioning, continuous dosing |

### Historical Experiment Benchmarks

| Experiment | Effect Size | 95% CI |
|------------|-------------|--------|
| Q2 2024 HCP targeting pilot | +18% NRx | [12%, 24%] |
| Q3 2024 nurse navigator program | +8% persistence | [5%, 11%] |
| Q4 2024 digital engagement | +12% reach | [8%, 16%] |

### KPI Targets

| KPI | Target | Current | Gap |
|-----|--------|---------|-----|
| NRx Growth YoY | +15% | Variable | Monitor quarterly |
| Adherence (PDC) | >80% | ~75% | Priority improvement |
| Market Share (CDK4/6) | >35% | ~32% | Competitive focus |

### Brand-Specific Confounders

When analyzing Kisqali, always control for:
- Prior CDK4/6 inhibitor use
- Line of therapy (1L vs 2L+)
- ECOG performance status
- Menopausal status
- Tumor characteristics (ER/PR expression level)

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

| Competitor | Brand | Positioning |
|------------|-------|-------------|
| **Soliris** | eculizumab | Established C5 inhibitor, IV |
| **Ultomiris** | ravulizumab | Long-acting C5 inhibitor, IV |

### Statistical Considerations

Small population requires special methods:
- Bayesian methods for underpowered studies
- Cluster randomization by center
- Careful multiple comparison adjustment
- Consider external controls from registry data

### KPI Targets

| KPI | Target | Rationale |
|-----|--------|-----------|
| Patient starts (Year 1) | 500+ | Market penetration goal |
| Switch rate from C5i | 30% | Conversion opportunity |
| Hemoglobin normalization | >70% | Clinical efficacy marker |
| Transfusion avoidance | >80% | Key patient outcome |

### Brand-Specific Confounders

When analyzing Fabhalta, always control for:
- Prior C5 inhibitor exposure
- Hemoglobin level at baseline
- LDH level
- Transfusion history (last 12 months)
- Clone size (GPI-deficient cells)
- Specialist access (geography)

---

## Remibrutinib - Chronic Spontaneous Urticaria (CSU)

### Patient Population
- Adults with CSU inadequately controlled on H1 antihistamines
- Treatment setting: Second-line after antihistamine failure
- Mechanism: BTK inhibitor (oral)

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

| Competitor | Brand | Positioning |
|------------|-------|-------------|
| **Xolair** | omalizumab | Current standard, established, injectable |
| **Fenebrutinib** | Pipeline | BTK competitor |
| **Ligelizumab** | Pending | Anti-IgE, pending approval |

### Launch Phase Dynamics

Apply these rules for launch-phase analysis:

1. **Early Adopter Effect**: KOLs and academic centers adopt 2x faster
2. **Payer Dynamics**: Prior authorization creates natural experiments
3. **Competitive Stockouts**: Temporary competitor shortages cause shifts (confound)
4. **Formulary Timing**: Coverage rollout affects regional uptake

### KPI Targets

| KPI | Target | Measurement |
|-----|--------|-------------|
| AH uncontrolled % | ≤40% | Addressable market |
| Intent-to-prescribe change | ≥0.5 points | HCP surveys |
| Market penetration (CSU biologics) | 25% Year 1 | Competitive share |
| UAS7 control | >40% | Patient outcomes |
| Persistence | >70% | Adherence tracking |

### Brand-Specific Confounders

When analyzing Remibrutinib, always control for:
- Antihistamine response history
- UAS7 score at baseline
- Prior biologic use
- Angioedema history
- Comorbid allergic conditions
- Early adopter bias
- Payer coverage status

---

## Cross-Brand Considerations

### Common Confounders Across All Brands

| Confounder | Applies To | Rationale |
|------------|------------|-----------|
| Territory potential | All | High-potential = more targeting + more Rx |
| HCP specialty | All | Specialists vs generalists differ |
| Practice type | All | Academic vs community affects access |
| Payer mix | All | Coverage affects prescribing |
| Geographic region | All | Regional variations exist |

### Brand-Specific Analysis Triggers

| Query Contains | Load Brand Section |
|----------------|-------------------|
| "kisqali", "ribociclib", "cdk4/6", "breast cancer" | Kisqali |
| "fabhalta", "iptacopan", "pnh", "hemoglobinuria" | Fabhalta |
| "remibrutinib", "csu", "urticaria", "btk" | Remibrutinib |

### Competitive Intelligence Integration

When analyzing brand performance:
1. Check competitor activity in same period
2. Control for market-level trends
3. Account for launch timing effects
4. Consider payer landscape changes
