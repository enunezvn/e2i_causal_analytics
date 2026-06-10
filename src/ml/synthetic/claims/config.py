"""Configuration for the claim-level CSU/Remibrutinib synthetic generator.

All constants are SOURCE-VERIFIED against ``scripts/convert_optum_rwd.py`` so
the embedded DGP is recoverable through the *real* converter, not a mock.
"""

from dataclasses import dataclass

# Enrollment regime — verified against convert_optum_rwd.py:116-118
# (ENROLLMENT_REGIMES["production"] = {pre_days: 360, post_days: 180}).
ENROLLMENT_PRE_DAYS = 360
ENROLLMENT_POST_DAYS = 180
LOOKBACK_DAYS = 180  # mirrors convert_optum_rwd.py:68
PREDICTION_DAYS = 180  # mirrors convert_optum_rwd.py:69

# CSU L50.x dx codes the converter's demographics gate accepts.
# Verified CSU_DX_PREFIXES = ("L501", "L508", "L509") at convert_optum_rwd.py:260.
CSU_DX_CODES = ("L501", "L508", "L509")

# Comorbidity dx codes the converter maps into has_<name>/charlson/elixhauser
# features. Chosen to (a) hit COMORBIDITY_CODES / QUAN_* mappings and (b) NOT
# collide with EXCLUSION_DX_PREFIXES (convert_optum_rwd.py:261 — O*, C*, B20,
# D8*, T78*, L502/4/5/6, Q822/D4702 are all excluded).
#   E119  -> diabetes (Quan Charlson "diabetes_without_complication")
#   J449  -> COPD (Quan Charlson "chronic_pulmonary")
#   I10   -> hypertension (Quan Elixhauser "hypertension_uncomplicated")
#   F329  -> depression (COMORBIDITY_CODES["depression"] = F32/F33 + Elixhauser)
#   J459  -> asthma (COMORBIDITY_CODES["asthma"] = J45 -> atopy_score)
COMORBIDITY_DX = ("E119", "J449", "I10", "F329", "J459")

# CSU biologics — verified XOLAIR/DUPIXENT brands, omalizumab/dupilumab
# generics, J2357/J0517 HCPCS, 50242/00024/0024 NDC prefixes
# (convert_optum_rwd.py:274-277). The DGP uses XOLAIR + NDC 50242* so the
# converter's _csu_biologic_mask (:1969) fires.
BIOLOGIC_BRAND = "XOLAIR"
BIOLOGIC_GENERIC = "omalizumab"
BIOLOGIC_NDC = "50242004"  # 50242 prefix per CSU_BIOLOGIC_NDC_PREFIXES
BIOLOGIC_HCPCS = "J2357"
BIOLOGIC_DAYS_SUP = 28  # Xolair q4w dosing -> ~28-day supply

# Non-biologic prior-therapy generics the converter recognises as NON_TARGET
# drug classes (convert_optum_rwd.py:1060). These feed _fill_count /
# _days_supply_total pre-index features WITHOUT touching the biologic mask.
PRIOR_THERAPY_GENERICS = ("cetirizine", "loratadine", "hydroxyzine", "montelukast")

# Discontinuation/persistence gap thresholds the converter uses to build the
# B/C *labels* (convert_optum_rwd.py:128-129). The DGP encodes post-index fill
# gaps relative to these so adherence_propensity is statistically recoverable.
BIOLOGIC_DISCONT_GAP_DAYS = 90
BIOLOGIC_PERSISTENCE_GAP_DAYS = 60


@dataclass
class ClaimsDGPConfig:
    """Knobs for the claim-level CSU DGP.

    ``signal_scale`` is the honest-band tuning knob (mirrors the spirit of
    ``RwdRealisticConfig.signal_scale``): it scales the latent-state effect on
    initiation/adherence so the recovered ``val_AUC`` lands in [0.62, 0.68]
    rather than a degenerate 0.9+.
    """

    n_patients: int = 2000
    seed: int = 42
    signal_scale: float = 1.0
    prevalence: float = 0.024  # CSU biologic-initiation calibration target
    panel_fragmentation_rate: float = 0.50  # ~half violate enrollment gate
    n_hcps: int = 0  # 0 => derived from n_patients in the CLI / events emit
    pre_days: int = ENROLLMENT_PRE_DAYS
    post_days: int = ENROLLMENT_POST_DAYS
