"""Brand configuration factory for CohortConstructor agent.

Provides pre-built CohortConfig instances for supported pharmaceutical brands
following FDA/EMA label criteria.

Supported Brands:
- Remibrutinib (CSU - Chronic Spontaneous Urticaria)
- Fabhalta (PNH - Paroxysmal Nocturnal Hemoglobinuria / C3G)
- Kisqali (HR+/HER2- Advanced Breast Cancer)
"""

from typing import Any, Dict, Optional

from .constants import SUPPORTED_BRANDS
from .types import (
    CohortConfig,
    Criterion,
    CriterionType,
    Operator,
    TemporalRequirements,
)

# ==============================================================================
# REMIBRUTINIB CSU CONFIGURATION
# ==============================================================================


def _get_remibrutinib_csu_config() -> CohortConfig:
    """Create cohort configuration for Remibrutinib CSU indication.

    Clinical Context:
    - Indication: Chronic Spontaneous Urticaria (CSU)
    - Mechanism: BTK (Bruton's Tyrosine Kinase) inhibitor
    - Target Population: Adults with moderate-to-severe CSU inadequately
      controlled by H1-antihistamines

    FDA/EMA Label Criteria:
    - Age ≥18 years
    - Confirmed CSU diagnosis (L50.1)
    - UAS7 score ≥16 (moderate-to-severe disease activity)
    - Prior antihistamine therapy failure
    - No active autoimmune conditions
    - No concurrent immunosuppressive therapy
    """
    inclusion_criteria = [
        Criterion(
            field="age_at_diagnosis",
            operator=Operator.GREATER_EQUAL,
            value=18,
            criterion_type=CriterionType.INCLUSION,
            description="Adult patients (≥18 years)",
            clinical_rationale="FDA/EMA approval restricted to adult population",
        ),
        Criterion(
            field="diagnosis_code",
            operator=Operator.IN,
            value=["L50.1", "L50.8", "L50.9"],
            criterion_type=CriterionType.INCLUSION,
            description="Confirmed CSU diagnosis (ICD-10)",
            clinical_rationale="L50.1 (idiopathic urticaria), L50.8/L50.9 (other/unspecified urticaria)",
        ),
        Criterion(
            field="urticaria_severity_uas7",
            operator=Operator.GREATER_EQUAL,
            value=16,
            criterion_type=CriterionType.INCLUSION,
            description="Moderate-to-severe disease activity (UAS7 ≥16)",
            clinical_rationale="UAS7 score 16-27 indicates moderate disease; ≥28 indicates severe",
        ),
        Criterion(
            field="prior_antihistamine_therapy",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.INCLUSION,
            description="Prior H1-antihistamine therapy",
            clinical_rationale="Remibrutinib indicated for patients inadequately controlled by antihistamines",
        ),
    ]

    exclusion_criteria = [
        Criterion(
            field="active_autoimmune_condition",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Active autoimmune conditions",
            clinical_rationale="BTK inhibition may exacerbate autoimmune pathology",
        ),
        Criterion(
            field="concurrent_immunosuppressive",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Concurrent immunosuppressive therapy",
            clinical_rationale="Drug interaction and compounded immunosuppression risk",
        ),
        Criterion(
            field="pregnancy_status",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Pregnancy or planning pregnancy",
            clinical_rationale="Fetal risk category - contraindicated in pregnancy",
        ),
        Criterion(
            field="severe_hepatic_impairment",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Severe hepatic impairment (Child-Pugh C)",
            clinical_rationale="Remibrutinib primarily metabolized by liver; dose adjustment required",
        ),
    ]

    return CohortConfig(
        cohort_name="Remibrutinib CSU Eligible Population",
        brand="remibrutinib",
        indication="csu",
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        temporal_requirements=TemporalRequirements(
            lookback_days=180,
            followup_days=90,
            index_date_field="diagnosis_date",
        ),
        required_fields=[
            "patient_journey_id",
            "age_at_diagnosis",
            "diagnosis_code",
            "diagnosis_date",
            "urticaria_severity_uas7",
            "prior_antihistamine_therapy",
            "first_observation_date",
            "last_observation_date",
        ],
        version="1.0.0",
        status="active",
        clinical_rationale="Adults with moderate-to-severe CSU inadequately controlled by H1-antihistamines",
        regulatory_justification="Based on FDA/EMA label for Remibrutinib in CSU",
    )


# ==============================================================================
# FABHALTA PNH/C3G CONFIGURATION
# ==============================================================================


def _get_fabhalta_pnh_config() -> CohortConfig:
    """Create cohort configuration for Fabhalta PNH indication.

    Clinical Context:
    - Indication: Paroxysmal Nocturnal Hemoglobinuria (PNH)
    - Mechanism: Factor B inhibitor (complement pathway)
    - Target Population: Adults with PNH requiring complement inhibitor therapy

    FDA/EMA Label Criteria:
    - Age ≥18 years
    - Confirmed PNH diagnosis (D59.5)
    - Elevated LDH (≥1.5x ULN indicating hemolysis)
    - Prior or current complement inhibitor therapy
    - No active serious infection
    - Meningococcal vaccination current
    """
    inclusion_criteria = [
        Criterion(
            field="age_at_diagnosis",
            operator=Operator.GREATER_EQUAL,
            value=18,
            criterion_type=CriterionType.INCLUSION,
            description="Adult patients (≥18 years)",
            clinical_rationale="FDA/EMA approval restricted to adult population",
        ),
        Criterion(
            field="diagnosis_code",
            operator=Operator.IN,
            value=["D59.5", "D59.50", "D59.51", "D59.59"],
            criterion_type=CriterionType.INCLUSION,
            description="Confirmed PNH diagnosis (ICD-10)",
            clinical_rationale="D59.5 codes cover PNH spectrum",
        ),
        Criterion(
            field="ldh_ratio",
            operator=Operator.GREATER_EQUAL,
            value=1.5,
            criterion_type=CriterionType.INCLUSION,
            description="Elevated LDH (≥1.5x ULN)",
            clinical_rationale="LDH elevation indicates ongoing intravascular hemolysis",
        ),
        Criterion(
            field="complement_inhibitor_status",
            operator=Operator.IN,
            value=["current", "prior"],
            criterion_type=CriterionType.INCLUSION,
            description="Prior or current complement inhibitor therapy",
            clinical_rationale="Fabhalta indicated for patients on established complement inhibition",
        ),
    ]

    exclusion_criteria = [
        Criterion(
            field="active_serious_infection",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Active serious infection",
            clinical_rationale="Complement inhibition increases infection susceptibility",
        ),
        Criterion(
            field="meningococcal_vaccination_current",
            operator=Operator.EQUAL,
            value=False,
            criterion_type=CriterionType.EXCLUSION,
            description="Meningococcal vaccination not current",
            clinical_rationale="REMS requirement - vaccination mandatory before complement inhibitor therapy",
        ),
        Criterion(
            field="bone_marrow_transplant_recent",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Recent bone marrow transplant (<6 months)",
            clinical_rationale="Engraftment period requires immunologic stability",
        ),
        Criterion(
            field="severe_renal_impairment",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Severe renal impairment (eGFR <30)",
            clinical_rationale="Iptacopan renally excreted; dose adjustment required",
        ),
    ]

    return CohortConfig(
        cohort_name="Fabhalta PNH Eligible Population",
        brand="fabhalta",
        indication="pnh",
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        temporal_requirements=TemporalRequirements(
            lookback_days=365,  # Longer lookback for complement therapy history
            followup_days=180,  # Extended followup for hemolysis monitoring
            index_date_field="diagnosis_date",
        ),
        required_fields=[
            "patient_journey_id",
            "age_at_diagnosis",
            "diagnosis_code",
            "diagnosis_date",
            "ldh_ratio",
            "complement_inhibitor_status",
            "meningococcal_vaccination_current",
            "first_observation_date",
            "last_observation_date",
        ],
        version="1.0.0",
        status="active",
        clinical_rationale="Adults with PNH on established complement inhibitor therapy with ongoing hemolysis",
        regulatory_justification="Based on FDA/EMA label for Fabhalta (iptacopan) in PNH",
    )


def _get_fabhalta_c3g_config() -> CohortConfig:
    """Create cohort configuration for Fabhalta C3 Glomerulopathy indication.

    Clinical Context:
    - Indication: C3 Glomerulopathy (C3G)
    - Mechanism: Factor B inhibitor (complement pathway)
    - Target Population: Adults with C3G requiring complement modulation

    FDA/EMA Label Criteria:
    - Age ≥18 years
    - Confirmed C3G diagnosis (biopsy-proven)
    - Proteinuria ≥1g/day
    - eGFR ≥30 mL/min/1.73m²
    - No active serious infection
    """
    inclusion_criteria = [
        Criterion(
            field="age_at_diagnosis",
            operator=Operator.GREATER_EQUAL,
            value=18,
            criterion_type=CriterionType.INCLUSION,
            description="Adult patients (≥18 years)",
            clinical_rationale="FDA/EMA approval restricted to adult population",
        ),
        Criterion(
            field="diagnosis_code",
            operator=Operator.IN,
            value=["N03.6", "N04.6", "N05.6"],
            criterion_type=CriterionType.INCLUSION,
            description="Confirmed C3 Glomerulopathy diagnosis (ICD-10)",
            clinical_rationale="Dense deposit disease and C3 glomerulonephritis codes",
        ),
        Criterion(
            field="proteinuria_g_day",
            operator=Operator.GREATER_EQUAL,
            value=1.0,
            criterion_type=CriterionType.INCLUSION,
            description="Significant proteinuria (≥1g/day)",
            clinical_rationale="Indicates active glomerular disease requiring intervention",
        ),
        Criterion(
            field="egfr",
            operator=Operator.GREATER_EQUAL,
            value=30,
            criterion_type=CriterionType.INCLUSION,
            description="Adequate renal function (eGFR ≥30)",
            clinical_rationale="Sufficient renal function for drug metabolism and efficacy assessment",
        ),
    ]

    exclusion_criteria = [
        Criterion(
            field="active_serious_infection",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Active serious infection",
            clinical_rationale="Complement inhibition increases infection susceptibility",
        ),
        Criterion(
            field="meningococcal_vaccination_current",
            operator=Operator.EQUAL,
            value=False,
            criterion_type=CriterionType.EXCLUSION,
            description="Meningococcal vaccination not current",
            clinical_rationale="REMS requirement for complement inhibitor therapy",
        ),
        Criterion(
            field="dialysis_dependent",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Dialysis-dependent ESRD",
            clinical_rationale="End-stage renal disease precludes meaningful efficacy assessment",
        ),
    ]

    return CohortConfig(
        cohort_name="Fabhalta C3G Eligible Population",
        brand="fabhalta",
        indication="c3g",
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        temporal_requirements=TemporalRequirements(
            lookback_days=365,
            followup_days=180,
            index_date_field="diagnosis_date",
        ),
        required_fields=[
            "patient_journey_id",
            "age_at_diagnosis",
            "diagnosis_code",
            "diagnosis_date",
            "proteinuria_g_day",
            "egfr",
            "meningococcal_vaccination_current",
            "first_observation_date",
            "last_observation_date",
        ],
        version="1.0.0",
        status="active",
        clinical_rationale="Adults with biopsy-proven C3G with significant proteinuria and preserved renal function",
        regulatory_justification="Based on FDA/EMA label for Fabhalta (iptacopan) in C3G",
    )


# ==============================================================================
# KISQALI HR+/HER2- BREAST CANCER CONFIGURATION
# ==============================================================================


def _get_kisqali_hr_her2_bc_config() -> CohortConfig:
    """Create cohort configuration for Kisqali HR+/HER2- breast cancer indication.

    Clinical Context:
    - Indication: HR+/HER2- Advanced or Metastatic Breast Cancer
    - Mechanism: CDK4/6 inhibitor
    - Target Population: Pre/postmenopausal women with HR+/HER2- advanced BC

    FDA/EMA Label Criteria:
    - Age ≥18 years
    - Confirmed breast cancer diagnosis
    - HR-positive (ER+ and/or PR+)
    - HER2-negative
    - Advanced or metastatic disease
    - ECOG performance status 0-1
    - Adequate organ function
    """
    inclusion_criteria = [
        Criterion(
            field="age_at_diagnosis",
            operator=Operator.GREATER_EQUAL,
            value=18,
            criterion_type=CriterionType.INCLUSION,
            description="Adult patients (≥18 years)",
            clinical_rationale="FDA/EMA approval restricted to adult population",
        ),
        Criterion(
            field="diagnosis_code",
            operator=Operator.IN,
            value=[
                "C50.0",
                "C50.1",
                "C50.2",
                "C50.3",
                "C50.4",
                "C50.5",
                "C50.6",
                "C50.8",
                "C50.9",
            ],
            criterion_type=CriterionType.INCLUSION,
            description="Confirmed breast cancer diagnosis (ICD-10)",
            clinical_rationale="C50.x codes cover breast malignancies",
        ),
        Criterion(
            field="hr_status",
            operator=Operator.EQUAL,
            value="positive",
            criterion_type=CriterionType.INCLUSION,
            description="Hormone receptor positive (ER+ and/or PR+)",
            clinical_rationale="CDK4/6 inhibitors work synergistically with endocrine therapy",
        ),
        Criterion(
            field="her2_status",
            operator=Operator.EQUAL,
            value="negative",
            criterion_type=CriterionType.INCLUSION,
            description="HER2-negative",
            clinical_rationale="Indication restricted to HER2-negative disease",
        ),
        Criterion(
            field="disease_stage",
            operator=Operator.IN,
            value=["advanced", "metastatic", "locally_advanced", "stage_iv"],
            criterion_type=CriterionType.INCLUSION,
            description="Advanced or metastatic disease",
            clinical_rationale="Kisqali indicated for advanced/metastatic setting",
        ),
        Criterion(
            field="ecog_performance_status",
            operator=Operator.LESS_EQUAL,
            value=1,
            criterion_type=CriterionType.INCLUSION,
            description="ECOG performance status 0-1",
            clinical_rationale="Adequate functional status for CDK4/6 inhibitor therapy",
        ),
    ]

    exclusion_criteria = [
        Criterion(
            field="prior_cdk46_inhibitor",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Prior CDK4/6 inhibitor therapy",
            clinical_rationale="First-line CDK4/6 inhibitor indication; prior exposure excluded",
        ),
        Criterion(
            field="qtc_prolongation",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="QTc prolongation (>480ms) or risk factors",
            clinical_rationale="Ribociclib associated with QT prolongation; cardiac monitoring required",
        ),
        Criterion(
            field="severe_hepatic_impairment",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Severe hepatic impairment (Child-Pugh C)",
            clinical_rationale="Ribociclib hepatically metabolized; dose adjustment required",
        ),
        Criterion(
            field="active_cns_metastases",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Active CNS metastases",
            clinical_rationale="Untreated/symptomatic brain metastases require stabilization first",
        ),
        Criterion(
            field="concurrent_strong_cyp3a_inhibitor",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.EXCLUSION,
            description="Concurrent strong CYP3A4 inhibitors",
            clinical_rationale="Significant drug interaction risk with ribociclib metabolism",
        ),
    ]

    return CohortConfig(
        cohort_name="Kisqali HR+/HER2- BC Eligible Population",
        brand="kisqali",
        indication="hr_her2_bc",
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        temporal_requirements=TemporalRequirements(
            lookback_days=365,  # Longer lookback for treatment history
            followup_days=180,
            index_date_field="diagnosis_date",
        ),
        required_fields=[
            "patient_journey_id",
            "age_at_diagnosis",
            "diagnosis_code",
            "diagnosis_date",
            "hr_status",
            "her2_status",
            "disease_stage",
            "ecog_performance_status",
            "first_observation_date",
            "last_observation_date",
        ],
        version="1.0.0",
        status="active",
        clinical_rationale="Pre/postmenopausal women with HR+/HER2- advanced breast cancer",
        regulatory_justification="Based on FDA/EMA label for Kisqali (ribociclib) in HR+/HER2- ABC",
    )


# ==============================================================================
# BRAND CONFIGURATION FACTORY
# ==============================================================================

# Registry of brand/indication configurations
_CONFIG_REGISTRY: Dict[str, CohortConfig] = {}


def _initialize_registry() -> None:
    """Initialize the configuration registry with all brand configs."""
    global _CONFIG_REGISTRY
    if not _CONFIG_REGISTRY:
        _CONFIG_REGISTRY = {
            "remibrutinib": _get_remibrutinib_csu_config(),
            "remibrutinib_csu": _get_remibrutinib_csu_config(),
            "fabhalta": _get_fabhalta_pnh_config(),
            "fabhalta_pnh": _get_fabhalta_pnh_config(),
            "fabhalta_c3g": _get_fabhalta_c3g_config(),
            "kisqali": _get_kisqali_hr_her2_bc_config(),
            "kisqali_hr_her2_bc": _get_kisqali_hr_her2_bc_config(),
        }


def get_brand_config(brand: str, indication: Optional[str] = None) -> CohortConfig:
    """Get pre-built cohort configuration for a pharmaceutical brand.

    Args:
        brand: Brand name (remibrutinib, fabhalta, kisqali)
        indication: Optional indication specifier (csu, pnh, c3g, hr_her2_bc)
                   If not provided, uses default indication for brand.

    Returns:
        CohortConfig with FDA/EMA label criteria

    Raises:
        ValueError: If brand or indication not supported

    Examples:
        >>> config = get_brand_config("remibrutinib")
        >>> config = get_brand_config("fabhalta", "pnh")
        >>> config = get_brand_config("fabhalta", "c3g")
        >>> config = get_brand_config("kisqali")
    """
    _initialize_registry()

    brand_lower = brand.lower()

    # Try brand_indication key first
    if indication:
        key = f"{brand_lower}_{indication.lower()}"
        if key in _CONFIG_REGISTRY:
            return _CONFIG_REGISTRY[key]

    # Fall back to brand-only key
    if brand_lower in _CONFIG_REGISTRY:
        return _CONFIG_REGISTRY[brand_lower]

    # Error: unsupported brand/indication
    supported = list(SUPPORTED_BRANDS.keys())
    raise ValueError(
        f"Unsupported brand '{brand}'. Supported brands: {supported}. "
        f"For Fabhalta, specify indication='pnh' or indication='c3g'."
    )


def list_available_configs() -> Dict[str, Dict[str, Any]]:
    """List all available brand configurations.

    Returns:
        Dictionary mapping brand keys to configuration summaries
    """
    _initialize_registry()

    return {
        key: {
            "cohort_name": config.cohort_name,
            "brand": config.brand,
            "indication": config.indication,
            "version": config.version,
            "inclusion_count": len(config.inclusion_criteria),
            "exclusion_count": len(config.exclusion_criteria),
        }
        for key, config in _CONFIG_REGISTRY.items()
    }


def get_config_for_brand_indication(brand: str, indication: str) -> CohortConfig:
    """Get configuration for specific brand and indication combination.

    This is an alias for get_brand_config with explicit indication parameter.

    Args:
        brand: Brand name
        indication: Indication name

    Returns:
        CohortConfig for the brand/indication combination
    """
    return get_brand_config(brand, indication)
