---
name: Opportunity Sizing Methodology
version: 1.0
description: TAM/SAM/SOM framework and market gap quantification for pharmaceutical commercial analytics
triggers:
  - opportunity sizing
  - market sizing
  - TAM SAM SOM
  - addressable market
  - patient pool estimation
  - HCP targeting sizing
  - geographic opportunity
agents:
  - gap_analyzer
  - heterogeneous_optimizer
  - resource_optimizer
categories:
  - market-analysis
  - strategic-planning
---

# Opportunity Sizing Methodology

## TAM/SAM/SOM Framework for Pharma Markets

### Total Addressable Market (TAM)

**Definition**: Total market demand for all products/services in the category.

**Formula**:
```
TAM = Total Patient Population × Average Annual Treatment Value
```

**For E2I Brands**:
- **Remibrutinib (CSU)**: All CSU patients eligible for biologics
- **Fabhalta (PNH)**: All diagnosed PNH patients globally
- **Kisqali (Breast Cancer)**: All HR+/HER2- advanced breast cancer patients

**Data Sources**:
- Epidemiological data
- Disease registries
- Claims databases
- Academic literature

### Serviceable Addressable Market (SAM)

**Definition**: Segment of TAM targeted by your products/services.

**Formula**:
```
SAM = TAM × Geographic Coverage × Specialty Penetration × Insurance Access
```

**Filters Applied**:
1. **Geographic Coverage**: Markets where brand is approved/launched
2. **Specialty Penetration**: HCPs treating the indication
3. **Insurance Access**: Patients with reimbursement coverage
4. **Brand-Appropriate Severity**: Patients matching indication criteria

**Example (Fabhalta)**:
```
TAM: 10,000 PNH patients (global)
× Geographic: 0.60 (US + EU)
× Specialty: 0.80 (hematology access)
× Insurance: 0.85 (covered plans)
= SAM: 4,080 patients
```

### Serviceable Obtainable Market (SOM)

**Definition**: Realistic share of SAM you can capture.

**Formula**:
```
SOM = SAM × Market Share Potential × Competitive Position × Execution Capability
```

**Factors**:
1. **Market Share Potential**: Category dynamics (e.g., switching barriers)
2. **Competitive Position**: Differentiation vs alternatives
3. **Execution Capability**: Sales force size, marketing budget, channel access

**Conservative Approach**:
- Year 1: 5-10% of SAM (new launch)
- Year 2-3: 15-25% of SAM (established)
- Year 4+: 30-40% of SAM (mature)

---

## Market Gap Quantification

### Gap Types

| Gap Type | Definition | Calculation |
|----------|------------|-------------|
| **Penetration Gap** | Addressable patients not on therapy | SAM - Current Patient Count |
| **Share Gap** | Market share vs target/competitor | Target Share - Current Share |
| **HCP Coverage Gap** | Target HCPs not engaged | Target HCP Count - Reached HCPs |
| **Geographic Gap** | Untapped territories | High-Potential Regions - Active Regions |
| **Switch Gap** | Patients on competitor therapy | Competitor Patients × Switch Probability |

### Brand-Specific Gap Quantification

#### Kisqali (HR+/HER2- Breast Cancer)

**Market Context**:
- TAM: ~150,000 patients (US advanced HR+/HER2-)
- SAM: ~90,000 patients (CDK4/6-eligible)
- Key Competitors: Ibrance, Verzenio

**Gap Calculation**:
```python
penetration_gap = sam_patients - current_patients
share_gap = target_share - current_share
switch_opportunity = competitor_patients × switch_rate × clinical_fit
```

**Example**:
```
SAM: 90,000 patients
Current on Kisqali: 25,000 (28% share)
Target share: 35%
Share gap: 7% → 6,300 patients
Revenue opportunity: 6,300 × $60,000/patient = $378M
```

#### Fabhalta (PNH - Rare Disease)

**Market Context**:
- TAM: ~10,000 diagnosed PNH patients (global)
- SAM: ~6,000 patients (US + EU approval)
- Key Competitor: C5 inhibitors (Soliris, Ultomiris)

**Gap Calculation**:
```python
# Unique to rare disease: patient identification gap
undiagnosed_gap = estimated_prevalence - diagnosed_patients
untreated_gap = diagnosed_patients - treated_patients
switch_gap = c5_treated × (inadequate_responders + dissatisfied)
```

**Example**:
```
Diagnosed patients: 6,000
Currently treated: 4,500 (75%)
Untreated gap: 1,500 patients
C5-treated inadequate responders: 900 patients (20% of 4,500)
Total addressable: 2,400 patients
Revenue opportunity: 2,400 × $450,000/patient = $1.08B
```

#### Remibrutinib (CSU - Launch Phase)

**Market Context**:
- TAM: ~300,000 CSU patients (US)
- SAM: ~60,000 biologic-eligible (moderate-severe, H1-refractory)
- Key Competitors: Xolair, prior-generation biologics

**Gap Calculation**:
```python
# Launch-specific: awareness and trial gaps
awareness_gap = target_hcps × (1 - awareness_rate)
trial_gap = aware_hcps × (intent_to_prescribe - actual_prescribers)
market_creation_gap = sam - current_biologic_users
```

**Example**:
```
SAM: 60,000 biologic-eligible patients
Current biologic users: 25,000 (any brand)
Market creation opportunity: 35,000 patients
Remibrutinib target share: 20% of biologic users
Share opportunity: 12,000 patients (20% of 60,000)
Revenue opportunity: 12,000 × $30,000/patient = $360M
```

---

## Addressable Value Calculation

### Value Formula

```
Addressable Value = Opportunity Size × Average Revenue per Unit × Capture Probability × Time Horizon Discount
```

**Components**:

1. **Opportunity Size**: From gap quantification (patients, prescriptions, HCPs)
2. **Average Revenue per Unit**:
   - Oncology: $50,000-$100,000/patient/year
   - Rare disease: $300,000-$500,000/patient/year
   - Specialty chronic: $20,000-$40,000/patient/year
3. **Capture Probability**: Realistic conversion estimate (20-40% for most interventions)
4. **Time Horizon Discount**: Adjust for time to capture (0.9 for Year 1, 0.7 for Year 2+)

### Example Calculation

**Gap**: 5,000 HCPs with low engagement in Northeast

```
Opportunity Size: 5,000 HCPs × 15 patients/HCP = 75,000 patients
Average Revenue: $500/TRx × 12 TRx/patient/year = $6,000/patient
Capture Probability: 30% (moderate intervention)
Time Horizon: Year 1 (0.9 discount)

Addressable Value = 75,000 × $6,000 × 0.30 × 0.9
                  = $121.5M
```

---

## Patient Pool Estimation

### Epidemiological Approach

**Formula**:
```
Patient Pool = Population × Prevalence × Diagnosed Rate × Treatment Rate × Brand-Appropriate Rate
```

**Example (CSU)**:
```
US Population: 330M
CSU Prevalence: 0.5% → 1.65M CSU patients
Moderate-Severe (biologic-eligible): 20% → 330,000 patients
Diagnosed & H1-refractory: 60% → 198,000 patients
Seeking biologic therapy: 30% → 59,400 patients
SAM: ~60,000 patients
```

### Claims-Based Approach

**Data Sources**:
- Symphony Health
- IQVIA
- Komodo Health

**Method**:
1. Identify patients with diagnosis codes (ICD-10)
2. Filter for treatment history (current therapy)
3. Apply eligibility criteria (severity, prior treatments)
4. Project to national estimates

**Validation**:
- Cross-reference with registry data
- Validate with KOL estimates
- Compare to competitor disclosures

---

## Geographic Opportunity Mapping

### Regional Sizing Model

**Factors**:
1. **Population Density**: Patients per 100K population
2. **HCP Concentration**: Specialists per 100K population
3. **Healthcare Access**: Insurance coverage rates
4. **Competitive Intensity**: Market share of competitors
5. **Channel Penetration**: Sales force coverage

### Regional Tiers

| Tier | Criteria | Opportunity Size | Priority |
|------|----------|------------------|----------|
| **Tier 1 (Major Metro)** | Population > 5M, high HCP density | 30-40% of SAM | High |
| **Tier 2 (Regional Hubs)** | Population 1-5M, moderate density | 25-35% of SAM | Medium |
| **Tier 3 (Secondary Markets)** | Population < 1M, lower density | 20-30% of SAM | Low-Medium |
| **Tier 4 (Rural/Frontier)** | Low population, sparse HCP | 5-15% of SAM | Low |

### Example (Fabhalta PNH Opportunity)

| Region | Diagnosed PNH Patients | Current Penetration | Gap | Opportunity Value |
|--------|------------------------|---------------------|-----|-------------------|
| Northeast | 1,200 | 15% (180) | 1,020 | $459M |
| South | 1,800 | 10% (180) | 1,620 | $729M |
| Midwest | 1,400 | 12% (168) | 1,232 | $554M |
| West | 1,600 | 18% (288) | 1,312 | $590M |

**Prioritization**: Northeast (high density, moderate gap) > South (largest gap) > West (high engagement, lower gap) > Midwest

---

## HCP Targeting Opportunity Sizing

### HCP Segmentation

**Decile Analysis**:
```
Decile 1 (Top 10%): 50-60% of category volume
Decile 2-3 (Next 20%): 25-30% of volume
Decile 4-10 (Bottom 70%): 15-20% of volume
```

**Opportunity Targeting**:
- **High-Volume HCPs**: Retention and share-of-voice
- **Medium-Volume HCPs**: Conversion and engagement
- **Emerging HCPs**: Early adoption and relationship building

### HCP Opportunity Matrix

| Segment | HCP Count | Avg Patients/HCP | Current Brand Rx | Potential Brand Rx | Gap |
|---------|-----------|------------------|------------------|--------------------|-----|
| Academic Leaders | 150 | 50 | 25 | 40 | 15/HCP |
| High-Volume Community | 800 | 30 | 10 | 20 | 10/HCP |
| Engaged Mid-Volume | 2,000 | 15 | 5 | 10 | 5/HCP |
| Low-Engagement | 5,000 | 8 | 1 | 4 | 3/HCP |

**Calculation**:
```python
for segment in hcp_segments:
    gap = potential_rx[segment] - current_rx[segment]
    opportunity = gap × hcp_count[segment] × revenue_per_rx
    priority_score = gap × revenue_per_rx × engagement_probability[segment]
```

**Example (Academic Leaders)**:
```
HCP Count: 150
Gap: 15 patients/HCP
Revenue: $500/TRx × 12 TRx/patient = $6,000/patient
Opportunity = 150 × 15 × $6,000 = $13.5M
```

---

## Opportunity Sizing Output Format

### Executive Summary

```markdown
## Market Opportunity Summary: [Brand/Indication]

### TAM/SAM/SOM Analysis
- **Total Addressable Market (TAM)**: [X] patients / $[Y]
- **Serviceable Addressable Market (SAM)**: [X] patients / $[Y]
- **Serviceable Obtainable Market (SOM)**: [X] patients / $[Y]

### Market Gap Quantification
| Gap Type | Size | Value |
|----------|------|-------|
| Penetration Gap | [X patients] | $[Y] |
| Share Gap | [X%] | $[Y] |
| HCP Coverage Gap | [X HCPs] | $[Y] |
| Geographic Gap | [X regions] | $[Y] |

### Addressable Value by Segment
1. **High-Priority Segment**: $[X]M (Capture Probability: [Y%])
2. **Medium-Priority Segment**: $[X]M (Capture Probability: [Y%])
3. **Long-Term Segment**: $[X]M (Capture Probability: [Y%])

**Total Addressable Value (3 Years)**: $[X]M
```

### Detailed Opportunity Profile

```markdown
### Opportunity: [Specific Gap Name]

**Sizing Methodology**:
- Approach: [Epidemiological / Claims-Based / HCP Survey]
- Data Sources: [List sources]
- Validation: [Cross-checks performed]

**Market Context**:
- TAM: [X]
- SAM: [Y]
- Current Penetration: [Z%]
- Gap Identified: [Description]

**Opportunity Quantification**:
- Gap Size: [X patients / HCPs / prescriptions]
- Average Revenue per Unit: $[Y]
- Capture Probability: [Z%]
- Time Horizon: [N years]
- **Addressable Value**: $[Total]

**Geographic Distribution**:
| Region | Gap Size | Opportunity Value | Priority |
|--------|----------|-------------------|----------|
| Northeast | [X] | $[Y] | High |
| South | [X] | $[Y] | Medium |
| Midwest | [X] | $[Y] | Low |
| West | [X] | $[Y] | Medium |

**HCP Targeting**:
| Segment | HCP Count | Gap/HCP | Total Opportunity |
|---------|-----------|---------|-------------------|
| Academic | [X] | [Y] | $[Z] |
| High-Vol Community | [X] | [Y] | $[Z] |
| Mid-Vol Engaged | [X] | [Y] | $[Z] |

**Strategic Recommendation**: [Specific action based on sizing]
```

---

## Validation Checkpoints

### Data Quality Checks

1. **Triangulation**: Cross-validate with 2+ data sources
2. **Reasonability**: Compare to industry benchmarks
3. **Consistency**: Ensure SAM ≤ TAM, SOM ≤ SAM
4. **Sensitivity Analysis**: Test assumptions (+/- 20%)

### Common Sizing Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Overestimating SAM** | Including non-addressable patients | Apply strict eligibility filters |
| **Ignoring Competition** | Assuming no competitive response | Model share-of-voice impact |
| **Static Assumptions** | No adjustment for market dynamics | Build time-based scenarios |
| **Undercounting Barriers** | Missing access/coverage issues | Layer insurance + formulary data |

### Confidence Levels

| Confidence | Data Quality | Sizing Method | Use Case |
|------------|--------------|---------------|----------|
| **High (±10%)** | Claims + Registry | Epidemiological + HCP survey | Investment decisions |
| **Medium (±20%)** | Claims or Survey | Single robust source | Strategic planning |
| **Low (±40%)** | Literature + KOL | Extrapolation | Early exploration |

---

## Integration with ROI Estimation

**Flow**:
1. **Opportunity Sizing** → Quantify addressable market gap
2. **ROI Estimation** → Calculate revenue impact and cost-to-close
3. **Prioritization** → Rank opportunities by ROI × Opportunity Size

**Example**:
```
Opportunity: Increase Fabhalta penetration in Northeast PNH patients

Opportunity Sizing:
- Gap: 1,020 patients (from SAM of 1,200 - current 180)
- Addressable Value: $459M (1,020 × $450K/patient)

ROI Estimation:
- Revenue Impact: $459M × 30% capture = $137.7M
- Cost to Close: $15M (HCP education + patient support)
- ROI: ($137.7M - $15M) / $15M = 8.2

Priority: High (ROI > 5.0, large opportunity size)
```

---

## Brand-Specific Sizing Models

### Kisqali (Oncology)

**Key Metrics**:
- New patient starts (NRx)
- Duration of therapy (persistence)
- Line of therapy mix (1L vs 2L+)

**Opportunity Drivers**:
1. Earlier line adoption (1L vs 2L)
2. Guideline-preferred status
3. Oncologist preference vs competitors

### Fabhalta (Rare Disease)

**Key Metrics**:
- Diagnosed patient identification
- C5i inadequate responders
- Switch probability from C5i

**Opportunity Drivers**:
1. Patient identification programs
2. Hematologist education on Factor D
3. Payer coverage expansion

### Remibrutinib (Launch Phase)

**Key Metrics**:
- Biologic-eligible patient pool
- HCP awareness and intent
- Market creation (non-biologic users)

**Opportunity Drivers**:
1. HCP education (dermatology + allergy)
2. Patient demand generation
3. Formulary access
