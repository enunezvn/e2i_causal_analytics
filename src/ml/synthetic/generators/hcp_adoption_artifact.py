"""optum_hcp-grain adoption CAUSAL cohort: HCP treatment arm + EXOGENOUS centrality.

Causal structure (no circularity):
    centrality_score  ~ exogenous draw (lognormal network topology)
    hcp_segment       = centrality tier {high,medium,low}_influence (effect modifier)
    treatment_arm     ~ rep/trigger engagement intensity (confounded by centrality)
    cate_estimate     = segment-scaled per-HCP treatment effect on adoption prob
    adoption_logit    = a + b*standardized(centrality) + tau(segment)*treatment + noise
    adopted           ~ Bernoulli(sigmoid(adoption_logit))
    adoption_category = "ADOPTER" if adopted else "NON_ADOPTER"  (canonical column)

The label is written to the canonical hcp_profiles.adoption_category column
(ADOPTER/NON_ADOPTER), which convert_optum_hcp_adoption.py's target derivation and
the runtime resolver both consume. It is DISTINCT from Shard 10's per-prescriber
peer_influence_score parquet: this is the topology COHORT (entity_type='optum_hcp');
Shard 10 is the claim-level CSU track.

Recoverable CATE (gate 10): the treatment arm RAISES adoption with a heterogeneous
effect by HCP segment (high-influence HCPs respond most), so hcp_adoption
participates in propensity/uplift-CATE recovery, not just label resolution.

Leak-safe: the columns convert_optum_hcp_adoption.py marks _LEAKY_HCP_COLS
(e.g. days_to_first / first_adoption_dt / adopter_rank) are NEVER emitted — adoption
is derived ONLY from exogenous centrality + the treatment arm, so topology features
do not encode the label.

The _compute_adoption core is SHARED with hcp_generator (which populates the
hcp_profiles DB grain) so both grains use the identical DGP (single SSOT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Must equal convert_optum_hcp_adoption.py:_ADOPTER_VALUE.
ADOPTER_VALUE = "ADOPTER"
_NON_ADOPTER_VALUE = "NON_ADOPTER"

# Intercept tuned so marginal adoption ~ 0.30 (band centre); slope makes centrality
# the exogenous driver of adoption.
_ADOPT_INTERCEPT = -0.95
_ADOPT_CENTRALITY_SLOPE = 0.95

# Designed treatment effect on the adoption LOGIT, heterogeneous by HCP segment
# (high-influence HCPs benefit MOST from rep/trigger engagement). Positive = RAISES.
_ADOPT_TREATMENT_LOGIT = {
    "high_influence": 1.30,
    "medium_influence": 0.80,
    "low_influence": 0.40,
}

# Brand-distinct adoption scale so a Kisqali probe differs from Remibrutinib.
_BRAND_ADOPT_SCALE = {"Remibrutinib": 1.0, "Kisqali": 0.8, "Fabhalta": 1.2}

# ---------------------------------------------------------------------------
# #1551: per-(brand, specialty) adoption affinity on the LOGIT scale.
#
# The gold-standard champion (hcp_adoption_{brand}_goldstd_lr_v1) trains on
# hcp_profiles.specialty JOINed to these labels. Without a specialty term the
# LR's specialty coefficients fit pure noise — measured 2026-08-11 on the
# canonical seed-42/427 cohort: Kisqali served propensity ranked rheumatology
# 0.454 > dermatology 0.421 > oncology 0.410, clinically backwards for a CDK4/6
# HR+/HER2- breast-cancer brand. These shifts make the DGP's specialty ordering
# clinically sensible per brand:
#   Kisqali      (ribociclib, CDK4/6; breast cancer)  -> oncology-led (hem/onc adjacent)
#   Fabhalta     (iptacopan, complement inh.; PNH/IgAN) -> hematology-led
#   Remibrutinib (BTK inh.; CSU)                       -> dermatology/allergy-led
#
# "*" is the default shift for unlisted specialties. Values are chosen so the
# population-weighted mean shift on the mixed-brand 5000-HCP cohort (~33% onc,
# 20% hem, 17% derm, 12% allergy, 10% IM, 5% rheum, 3% neuro) is ~0, keeping
# each brand's marginal adoption inside the designed (0.10, 0.60) band, and so
# the faithful goldstd-LR holdout AUC stays inside the accepted [0.70, 0.86]
# band (measured: see PR for #1551).
#
# RNG stream discipline (#1524/#1542): the affinity is a DETERMINISTIC logit
# shift — it consumes NO rng draws, so all seeded streams (treatment arms,
# consideration-date months, downstream columns) are bit-identical whether or
# not a specialty vector is supplied.
_AFFINITY_DEFAULT_KEY = "*"
_BRAND_SPECIALTY_AFFINITY: Dict[str, Dict[str, float]] = {
    "Kisqali": {
        "oncology": 0.9,
        "hematology": 0.1,  # hem/onc practice adjacency
        _AFFINITY_DEFAULT_KEY: -0.7,
    },
    "Fabhalta": {
        "hematology": 1.0,  # PNH
        "internal_medicine": 0.3,  # IgAN/C3G generalist overlap (no nephrology enum)
        "neurology": 0.2,  # rare-disease overlap (mirrors BRAND_SPECIALTY_DIST)
        _AFFINITY_DEFAULT_KEY: -0.4,
    },
    "Remibrutinib": {
        "dermatology": 0.9,  # CSU
        "allergy_immunology": 0.8,  # CSU
        "rheumatology": 0.3,  # urticaria/autoimmune adjacency
        _AFFINITY_DEFAULT_KEY: -0.5,
    },
}


def _specialty_affinity(brand: str, specialty: Optional[Sequence[str]], n: int) -> np.ndarray:
    """Vector of per-HCP logit shifts for ``brand``; zeros when no specialty
    vector is supplied (legacy callers / specialty-free grains stay bit-identical)."""
    if specialty is None:
        return np.zeros(n, dtype=float)
    shifts = _BRAND_SPECIALTY_AFFINITY.get(brand)
    if not shifts:
        return np.zeros(n, dtype=float)
    default = shifts.get(_AFFINITY_DEFAULT_KEY, 0.0)
    return np.array([shifts.get(s, default) for s in specialty], dtype=float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def _compute_adoption(
    rng: np.random.Generator,
    centrality_z: np.ndarray,
    brand: str,
    specialty: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Shared adoption DGP on a standardized centrality vector. Returns
    hcp_segment, treatment_arm, adopted (0/1), and the per-HCP probability-scale
    cate_estimate. Used by BOTH the standalone optum_hcp frame and the hcp_generator
    (hcp_profiles grain) so the two grains share one DGP.

    ``specialty`` (optional, #1551): per-HCP specialty strings aligned to
    ``centrality_z``. When supplied, a deterministic per-(brand, specialty)
    logit shift makes the specialty ordering clinically sensible (see
    ``_BRAND_SPECIALTY_AFFINITY``). When None, behaviour is bit-identical to the
    pre-#1551 DGP — the shift consumes no RNG draws either way."""
    n = len(centrality_z)
    hcp_segment = np.where(
        centrality_z > 0.5,
        "high_influence",
        np.where(centrality_z > -0.5, "medium_influence", "low_influence"),
    )
    # treatment_arm ~ rep/trigger engagement intensity, CONFOUNDED by centrality
    # (central HCPs get more rep attention) -> propensity is estimable.
    p_treat = _sigmoid(0.8 * centrality_z + rng.normal(0, 0.5, n))
    treatment_arm = (rng.random(n) < p_treat).astype(int)

    scale = _BRAND_ADOPT_SCALE.get(brand, 1.0)
    seg_treat = np.array([_ADOPT_TREATMENT_LOGIT[s] for s in hcp_segment], dtype=float)
    # #1551: deterministic specialty-affinity shift (zeros when specialty=None);
    # a pure table lookup — consumes NO rng draws, so the seeded stream is
    # bit-identical with or without a specialty vector.
    affinity = _specialty_affinity(brand, specialty, n)
    logit = (
        _ADOPT_INTERCEPT
        + _ADOPT_CENTRALITY_SLOPE * centrality_z
        + affinity
        + scale * seg_treat * treatment_arm
        + rng.normal(0.0, 0.6, n)
    )
    adopted = (rng.random(n) < _sigmoid(logit)).astype(int)

    # per-HCP CATE on the PROBABILITY scale (P(adopt) at T=1 vs T=0, centrality
    # and specialty fixed).
    base_logit = _ADOPT_INTERCEPT + _ADOPT_CENTRALITY_SLOPE * centrality_z + affinity
    cate_estimate = _sigmoid(base_logit + scale * seg_treat) - _sigmoid(base_logit)
    return {
        "hcp_segment": hcp_segment,
        "treatment_arm": treatment_arm,
        "adopted": adopted,
        "cate_estimate": cate_estimate,
    }


def generate_hcp_adoption_frame(*, seed: int, n_hcps: int, brand: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # EXOGENOUS centrality: lognormal degree (heavy-tailed network topology).
    network_size = rng.lognormal(mean=3.0, sigma=1.1, size=n_hcps)
    centrality = np.log1p(network_size)
    centrality_z = (centrality - centrality.mean()) / (centrality.std() + 1e-9)

    dgp = _compute_adoption(rng, centrality_z, brand)
    adopted = dgp["adopted"]
    return pd.DataFrame(
        {
            "hcp_id": [f"ohcp_{i:06d}" for i in range(n_hcps)],
            "entity_type": "optum_hcp",
            "centrality_score": centrality,
            "influence_network_size": network_size.round().astype(int),
            "hcp_segment": dgp["hcp_segment"],
            "treatment_arm": dgp["treatment_arm"],
            "adoption_category": np.where(adopted == 1, ADOPTER_VALUE, _NON_ADOPTER_VALUE),
            "cate_estimate": dgp["cate_estimate"],
            "is_synthetic": True,
        }
    )


def write_per_hcp_cate_artifact(
    df: pd.DataFrame, *, brand: str, out_dir: Union[Path, str] = "data/synthetic"
) -> Path:
    """Write the per-HCP CATE artifact Shard 08's allocation builder consumes.

    Output: <out_dir>/per_hcp_cate_hcp_adoption_<brand>.parquet with exactly
    [hcp_id, cate_estimate, is_synthetic] (one row per HCP, all is_synthetic=True).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"per_hcp_cate_hcp_adoption_{brand}.parquet"
    artifact = df[["hcp_id", "cate_estimate"]].copy()
    artifact["is_synthetic"] = True
    artifact.to_parquet(out, index=False)
    return out
