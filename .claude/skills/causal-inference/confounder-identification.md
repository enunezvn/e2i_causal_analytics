---
name: Confounder Identification for Pharma Analytics
version: 1.0
description: Standard confounders and instrumental variables for pharma causal analysis
triggers:
  - identify confounders
  - causal analysis setup
  - confounding variables
  - instrumental variables
  - control variables
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
| Rep territory assignment | HCP targeting | Prescribing (except through targeting) | HCP targeting impact |
| Physician graduation year | Practice patterns | Patient outcomes (except through practice) | Practice style effects |
| Distance to specialist | Referral likelihood | Disease severity | Referral impact |
| Insurance formulary timing | Drug availability | Patient health status | Formulary impact |

### Instrument Validity Checks

Before using an instrument, verify:

1. **Relevance**: Instrument strongly predicts treatment
   - F-statistic > 10 in first stage
   - Partial R² > 0.10

2. **Exclusion**: Instrument only affects outcome through treatment
   - Cannot directly test (assumption)
   - Use domain knowledge and sensitivity analysis

3. **Independence**: Instrument uncorrelated with unmeasured confounders
   - Check balance on observables
   - Use falsification tests

---

## Confounder Selection Checklist

When setting up a causal analysis:

1. **Identify treatment and outcome**
2. **List all potential confounders** using tables above
3. **Check data availability** for each confounder
4. **Prioritize by:**
   - Strength of confounding (backdoor paths)
   - Data quality
   - Measurement reliability
5. **Document missing confounders** as limitations
6. **Plan sensitivity analysis** for unobserved confounding

---

## Brand-Specific Confounders

### Kisqali Analysis
Additional confounders:
- Prior CDK4/6 inhibitor use
- Line of therapy
- ECOG performance status
- Menopausal status

### Fabhalta Analysis
Additional confounders:
- Prior C5 inhibitor exposure
- Hemoglobin level at baseline
- LDH level
- Transfusion history

### Remibrutinib Analysis
Additional confounders:
- Antihistamine response
- UAS7 score at baseline
- Prior biologic use
- Angioedema history
