"""Feature manifest for the Optum MART ``optum_hcp`` grain (commercial targeting).

The entity-stacked Optum drop (``data/rwd/Optum_Parquet/Optum.parquet``) carries
an ``optum_hcp`` entity (2.75M provider rows) that co-locates an HCP brand-adoption
target with a rich claims-derived practice profile. The HCP-adoption-propensity
cohort (``scripts/convert_optum_hcp_adoption.py``) is the deployable commercial-
HCP-targeting deliverable — the patient grain is feature-bound and non-deployable
(see ``docs/results/deployable_cohort_decision_20260607.md``).

This manifest is the authority for which ``optum_hcp`` columns are admissible
PRE-ADOPTION predictors vs adoption-DERIVED leakage. As with the patient mart
manifest, the adapter's positive-enumeration allow-list is the structural
barrier; this manifest is the runtime cross-check that also grants declared-safe
immunity to the admissible features so the statistical leakage layer does not
over-drop legitimate referral-network / volume predictors.

Admissibility rationale (measured against the live mart, 2026-06-07):
- The target ``adopted_target_brand`` = ``adoption_status == 'ADOPTER'`` (HCP
  prescribed the target brand XOLAIR in the observation window;
  ROGERS_CUMULATIVE_SHARE_BY_BRAND, NDC/HCPCS match).
- The admissible features are TOTAL (all-cause / indication-level) practice
  profile — claims referral-network position, all-cause patient/claim volume,
  specialty, geography — NOT brand-specific counts. ``target_patient_count`` /
  ``target_event_count`` / ``adopter_rank`` / ``days_to_first`` /
  ``adoption_*`` are adoption-DERIVED and declared post-index forbidden.
- A leakage ablation (``docs/results/hcp_adoption_ablation_20260607.py``) shows
  the signal is dominated by referral-network diffusion (network features
  standalone AUC 0.81) + specialty; the tautology-risk volume feature is NOT
  load-bearing (dropping it moves AUC 0.845 -> 0.837).

WINDOWING CAVEAT (honest, not hidden): the network/volume features are
cross-sectional aggregates in this pre-built mart (no clean pre/post window), so
``knowable_at=index_date`` declares the *contract intent* (these are pre-adoption
practice attributes, NOT derived from the adoption event). A strict
forward-causal deployment should recompute features over a pre-index baseline
window upstream; network position is structurally stable and known at targeting
time, so this is a feature-window design step, not a leakage of the label.

Authoring rules (mirror ``optum_mart_feature_manifest.py``):
- ``OPTUM_HCP_FEATURES`` is a statically-declared literal list (AST-traceable;
  NO comprehension rebuild).
- ``aggregation=None``: every mart column is PRECOMPUTED upstream.
"""

from src.data.feature_contract import FeatureContract, KnowableAt

_INDEX = KnowableAt(reference="index_date")
_POST = KnowableAt(reference="post_index")

# Statically-declared literal registry (AST-traceable; do NOT rebuild via comprehension).
OPTUM_HCP_FEATURES: list[FeatureContract] = [
    # ===== PRE-ADOPTION claims referral-network position =====
    FeatureContract(name="influence_network_size", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="shared_patient_edge_count", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="shared_patient_weight", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(
        name="max_shared_patient_edge_weight", knowable_at=_INDEX, source="hcp_network"
    ),
    FeatureContract(name="shared_patient_kol_score_pct", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="referral_in_degree", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="referral_in_patient_count", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="max_referral_in_edge_weight", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="referral_out_degree", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="referral_out_patient_count", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="referral_kol_score_pct", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="kol_score_100pt", knowable_at=_INDEX, source="hcp_network"),
    FeatureContract(name="kol_score", knowable_at=_INDEX, source="hcp_network"),
    # ===== PRE-ADOPTION all-cause / indication-level volume =====
    FeatureContract(name="medical_claim_count", knowable_at=_INDEX, source="hcp_volume"),
    FeatureContract(name="medical_patient_count", knowable_at=_INDEX, source="hcp_volume"),
    FeatureContract(name="treated_patient_count", knowable_at=_INDEX, source="hcp_volume"),
    # ===== PRE-ADOPTION provider attributes (specialty / geography) =====
    FeatureContract(name="specialty_group", knowable_at=_INDEX, source="hcp_provider"),
    FeatureContract(name="prov_type", knowable_at=_INDEX, source="hcp_provider"),
    FeatureContract(name="prov_state", knowable_at=_INDEX, source="hcp_provider"),
    FeatureContract(name="kol_category", knowable_at=_INDEX, source="hcp_provider"),
    FeatureContract(name="cred_type", knowable_at=_INDEX, source="hcp_provider"),
    # ===== POST-INDEX target =====
    FeatureContract(name="adopted_target_brand", knowable_at=_POST, source="hcp_target"),
    # ===== POST-INDEX adoption-DERIVED aliases (defense-in-depth) =====
    # The adapter never emits these (positive enumeration); declared forbidden
    # only as the runtime cross-check / leakage backstop. Each is computed FROM
    # the adoption event, so any of them would (near-)deterministically leak the
    # target.
    FeatureContract(name="adoption_status", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="adoption_category", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="adoption_category_method", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="adopter_rank", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="adopter_count", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="adoption_cumulative_share", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="days_to_first", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="first_adoption_dt", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="target_event_count", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="target_patient_count", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="distinct_target_code_count", knowable_at=_POST, source="hcp_adoption"),
    # Target-match provenance (how/where the target brand event was matched).
    # Adoption-derived (post-index); declared forbidden so the Layer-3
    # defense-in-depth pass catches them should raw optum_hcp data ever reach the
    # data_preparer without the converter's positive-enumeration allow-list (codex L2).
    FeatureContract(name="target_match_methods", knowable_at=_POST, source="hcp_adoption"),
    FeatureContract(name="event_sources", knowable_at=_POST, source="hcp_adoption"),
]


def optum_hcp_contract_for(name: str) -> FeatureContract | None:
    """Return the FeatureContract for a named optum_hcp column, or None if absent."""
    for contract in OPTUM_HCP_FEATURES:
        if contract.name == name:
            return contract
    return None


OPTUM_HCP_SAFE_FEATURES: list[str] = [
    c.name for c in OPTUM_HCP_FEATURES if c.knowable_at.is_pre_or_at_index()
]

OPTUM_HCP_FORBIDDEN_AS_FEATURES: list[str] = [
    c.name for c in OPTUM_HCP_FEATURES if not c.knowable_at.is_pre_or_at_index()
]

# Supervised label: forbidden-as-feature (post-index) but preserved at cohort-build
# so the pipeline can extract the target via prediction_target.
OPTUM_HCP_TARGETS: frozenset[str] = frozenset({"adopted_target_brand"})

OPTUM_HCP_FORBIDDEN_NON_TARGET: list[str] = [
    f for f in OPTUM_HCP_FORBIDDEN_AS_FEATURES if f not in OPTUM_HCP_TARGETS
]
