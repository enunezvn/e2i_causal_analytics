"""Synthetic causal_paths substrate (Shard 09) for CM-003 (causal_impact) and
CM-005 (mediation).

The causal_paths table exists on the faithful DB (50 stale rows, max discovery_date
2025-10-01). We INSERT minimal is_synthetic=true rows so causal_effect_size and
mediators_identified are non-NULL for those 2 KPIs; we do NOT touch the real rows.

NOT-NULL columns verified on the faithful DB: path_id, discovery_date, causal_chain
(jsonb), data_split (enum), created_at, confirmation_count, is_synthetic. data_split
is enum-exact (data_split_type: train/validation/test/holdout/unassigned).
"""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd

from .base import BaseGenerator, GeneratorConfig

_BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]
_REGIONS = ["northeast", "south", "midwest", "west"]
_MEDIATORS = ["adherence", "engagement_score", "prior_therapy", "disease_severity"]

# Gold-standard patient cohorts (src/mlops/gold_standard_eval/cohort_spec.py):
# treatment_arm -> <label>, each a VALIDATED causal relationship
# (scripts/validate_synthetic_causal.py). The KG previously carried ONLY the
# initiation chain, so "Discover chains in KG" returned nothing for the
# persistent_180d default outcome (and discontinued_180d). Emit all three so the
# FalkorDB sync seeds (:Variable treatment_arm)-[:CAUSES]->(:Variable <label>)
# for every cohort. Confounders mirror the specs: initiation stays byte-identical
# to the original Shard-09 substrate; persistence/discontinuation use the
# gold-standard base confounders (disease_severity, academic_hcp, geographic_region).
_COHORT_OUTCOMES = ("treatment_initiated", "persistent_180d", "discontinued_180d")
_COHORT_CONFOUNDERS = {
    "treatment_initiated": ["disease_severity", "age_at_diagnosis"],
    "persistent_180d": ["disease_severity", "academic_hcp", "geographic_region"],
    "discontinued_180d": ["disease_severity", "academic_hcp", "geographic_region"],
}

# HCP-grain adoption edges (Shard 06.3 cohort: hcp_brand_adoption JOIN
# hcp_profiles). TWO questions per brand, ADDITIVE to the patient cohort edges so
# the leaderboard enumerates the HCP grain from the same causal_paths SSOT:
#   peer_influence_score -> adopted : EXOGENOUS centrality, EMPTY backdoor.
#   treatment_arm        -> adopted : rep engagement, confounded by centrality_z
#                                     (= log1p(influence_network_size)).
# centrality_z is the modeled backdoor for the rep-engagement arm; the loader
# derives it from hcp_profiles. A single non-treatment/non-outcome mediator keeps
# every chain a clean 2-hop path AND non-empty (the generator's mediator
# invariant). HCP edges are brand-replicated for all three gold-standard brands.
_HCP_QUESTIONS: Tuple[Tuple[str, str, List[str], str], ...] = (
    ("peer_influence_score", "adopted", [], "centrality_diffusion"),
    ("treatment_arm", "adopted", ["centrality_z"], "rep_engagement_path"),
)

# Trigger-grain edges (the NBA RCT). The triggers table carries the only TRUE
# randomized experiment in the gold standard: control_group_flag is a randomized
# holdout, so control_group_flag -> action_taken has an EMPTY backdoor set (no
# confounder to adjust for — randomization breaks every back-door path). The
# second edge, acceptance_status -> conversion_flag, changed meaning with
# COMM-ARMS Phase 4: conversion_flag is now the REAL "prescription landed in the
# 30d window" outcome (tracked triggers of EVERY acceptance status), and
# acceptance is driven by the patient-level trigger_accepted arm, which is
# CONFOUNDED on disease_severity + engagement_score (treatment_arm.ARM_REGISTRY
# SSOT) — so the edge now carries a real backdoor set instead of the pre-P4
# "empty by construction" (outcome_value was only set for accepted triggers).
_TRIGGER_EDGES: Tuple[Tuple[str, str, List[str]], ...] = (
    # (start_node, end_node, confounders_controlled)
    ("control_group_flag", "action_taken", []),
    ("acceptance_status", "conversion_flag", ["disease_severity", "engagement_score"]),
)

# Patient-grain COMMERCIAL-ARM edges (2026-07-23). The DGP plants recoverable,
# confounded effects for five commercial levers on the patient cohort
# (copay_support / psp_enrolled / rep_detailing_high / sample_dropped /
# trigger_accepted — the treatment_arm.ARM_REGISTRY SSOT), and
# ``_CAUSAL_DATASET_SPECS['patient_journeys']`` (src/api/routes/causal.py)
# ALREADY allowlists every arm as a treatment AND its backdoor confounders. But
# the discovery leaderboard enumerates ONLY (treatment, outcome) pairs present in
# ``causal_paths`` (``_discover_candidate_questions``), and until now only
# treatment_arm had edges — so the strong planted levers were INVISIBLE on the
# leaderboard (reachable only via the manual "Pose your own question" panel).
# These edges surface them. Each carries the arm's EXACT DGP backdoor set (a
# subset of the dataset-spec covariate allowlist, so the confounder-contract
# guard is satisfied and estimation de-biases correctly instead of reporting the
# confounded naive diff). Direct 1-hop edges (no mediator), mirroring
# _TRIGGER_EDGES. Content-addressed like _COMMERCIAL_EDGES → idempotent targeted
# upsert (no full cohort reseed; the patient_journeys arm data already exists).
# The stored causal_effect_size is a registry/KG DISPLAY value in the arm's
# design band; the leaderboard RE-ESTIMATES on real rows.
_COMM_ARM_EDGES: Tuple[Tuple[str, str, Tuple[str, ...], float, float], ...] = (
    # (arm, outcome, confounders_controlled, effect_band_lo, effect_band_hi)
    ("copay_support", "adherent_180d", ("insurance_access_score", "disease_severity"), 0.08, 0.14),
    ("copay_support", "low_gap_180d", ("insurance_access_score", "disease_severity"), 0.06, 0.12),
    (
        "copay_support",
        "persistent_180d",
        ("insurance_access_score", "disease_severity"),
        0.06,
        0.11,
    ),
    (
        "psp_enrolled",
        "adherent_180d",
        ("disease_severity", "engagement_score", "academic_hcp"),
        0.06,
        0.11,
    ),
    (
        "psp_enrolled",
        "persistent_180d",
        ("disease_severity", "engagement_score", "academic_hcp"),
        0.05,
        0.10,
    ),
    ("rep_detailing_high", "treatment_initiated", ("academic_hcp", "engagement_score"), 0.03, 0.06),
    ("sample_dropped", "treatment_initiated", ("academic_hcp", "engagement_score"), 0.02, 0.05),
    (
        "trigger_accepted",
        "treatment_initiated",
        ("disease_severity", "engagement_score"),
        0.04,
        0.07,
    ),
)

N_COMM_ARM_ROWS = len(_COMM_ARM_EDGES) * len(_BRANDS)

# Commercial-KPI grain (2026-07-07). The registry modeled patient/HCP/trigger
# outcomes only, so "what drives TRx?" was a genuine substrate-coverage gap in
# chat and on every strategic-insight surface. These are CURATED synthetic
# driver chains (not estimated effects) for the most impactful commercial KPIs
# — TRx/NRx/NBRx/TRx Share (WS3-BI-005..008), ROI (WS3-BI-010) and
# intent-to-prescribe (BR-002's leading indicator) — surfaced only behind the
# platform provenance gate, labeled data_source="synthetic".
#
# Two contracts:
# * TOKEN MATCH — the chat read path matches 6-char token prefixes as ILIKE
#   substrings on start_node/end_node (outcome_match_tokens), so every
#   end_node carries its KPI's searchable token (trx/nrx/nbrx/share/roi/
#   intent). Pinned by tests/unit/test_synthetic/test_causal_paths_commercial.
# * CONTENT-ADDRESSED — path_id and every numeric value derive from
#   (brand, start, end) via a per-edge rng, independent of the generator seed
#   and n_records, so the targeted apply script upserts idempotently and a
#   later full reseed cannot silently rewrite the values (PR #1105/#1106
#   reseed-idempotency lesson).
#
# Driver vocabulary aligns with the causal_impact agent's DAG builder
# (rep detailing, formulary status, competitor activity) and the DGP
# commercial-arms spec (copay support, PSP/samples); persistent_180d and
# treatment_initiated link the patient-journey grain into volume so the
# registry tells one story. competitor_activity bands are NEGATIVE (pressure,
# not uplift). The leaderboard's grain-scope guard (causal.py
# _discover_candidate_questions) keeps these out of estimation runs because
# the nodes are not in any dataset spec's allowlists.
_COMMERCIAL_EDGES: Tuple[Tuple[str, str, str, Tuple[str, ...], float, float], ...] = (
    # (start_node, end_node, mediator, confounders_controlled, band_lo, band_hi)
    (
        "rep_detailing_frequency",
        "trx_volume",
        "hcp_engagement",
        ("academic_hcp", "geographic_region"),
        0.10,
        0.30,
    ),
    ("formulary_status", "trx_volume", "patient_access", ("payer_mix",), 0.15, 0.40),
    ("copay_support_program", "trx_volume", "adherence", ("disease_severity",), 0.08, 0.25),
    (
        "persistent_180d",
        "trx_volume",
        "refill_continuity",
        ("disease_severity", "academic_hcp", "geographic_region"),
        0.20,
        0.45,
    ),
    ("intent_to_prescribe", "nrx_volume", "new_patient_starts", ("academic_hcp",), 0.15, 0.40),
    (
        "sample_dropped",
        "nrx_volume",
        "trial_experience",
        ("academic_hcp", "geographic_region"),
        0.05,
        0.20,
    ),
    (
        "treatment_initiated",
        "nrx_volume",
        "patient_onboarding",
        ("disease_severity", "age_at_diagnosis"),
        0.20,
        0.45,
    ),
    ("hcp_coverage", "nbrx_volume", "prescriber_breadth", ("geographic_region",), 0.10, 0.30),
    ("competitor_activity", "nbrx_volume", "switch_pressure", ("geographic_region",), -0.30, -0.08),
    (
        "competitor_activity",
        "trx_market_share",
        "share_of_voice",
        ("geographic_region",),
        -0.25,
        -0.05,
    ),
    (
        "hcp_coverage",
        "trx_market_share",
        "prescriber_base",
        ("academic_hcp", "geographic_region"),
        0.08,
        0.25,
    ),
    ("rep_detailing_frequency", "roi", "trx_volume", ("academic_hcp",), 0.05, 0.20),
    ("copay_support_program", "roi", "adherence", ("disease_severity",), 0.05, 0.18),
    (
        "rep_detailing_frequency",
        "intent_to_prescribe",
        "message_recall",
        ("academic_hcp",),
        0.10,
        0.35,
    ),
    (
        "speaker_program_attendance",
        "intent_to_prescribe",
        "peer_validation",
        ("academic_hcp",),
        0.08,
        0.30,
    ),
)

N_COMMERCIAL_ROWS = len(_COMMERCIAL_EDGES) * len(_BRANDS)


def _commercial_edge_rng(brand: str, start: str, end: str) -> np.random.Generator:
    """Per-edge rng keyed on content, so every value is reproducible from the
    edge identity alone (idempotent apply; stable across reseeds). sha1 is
    content addressing, not security (usedforsecurity=False, Bandit B324)."""
    digest = hashlib.sha1(f"{brand}|{start}|{end}".encode(), usedforsecurity=False).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _commercial_path_id(brand: str, start: str, end: str) -> str:
    """Content-addressed id, namespaced scp_c*, 16 chars (varchar(20) cap)."""
    digest = hashlib.sha1(f"{brand}|{start}|{end}".encode(), usedforsecurity=False)
    return "scp_c" + digest.hexdigest()[:11]


def _comm_arm_path_id(brand: str, start: str, end: str) -> str:
    """Content-addressed id for a patient-grain commercial-arm edge, namespaced
    scp_a*, 16 chars (varchar(20) cap). The ``arm|`` prefix keeps the id space
    disjoint from _commercial_path_id even for a shared (brand, start, end)."""
    digest = hashlib.sha1(f"arm|{brand}|{start}|{end}".encode(), usedforsecurity=False)
    return "scp_a" + digest.hexdigest()[:11]


def commercial_rows_for_upsert() -> List[dict]:
    """The commercial grain as DB-shaped records for the targeted apply script
    (scripts/seed_commercial_causal_paths.py).

    Projected to the loader's causal_paths column list (the generator-only
    'grain' column would 400 the insert — the DB has no such column) and safe
    to upsert on path_id: every id and value is content-addressed, so re-runs
    are no-ops apart from the discovery_date/created_at freshness stamps.
    """
    # Lazy import: batch_loader imports the generators package (registry).
    import json

    from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS

    df = CausalPathsGenerator(GeneratorConfig(n_records=0)).generate()
    com = df[df["grain"] == "commercial"]
    cols = [c for c in TABLE_COLUMNS["causal_paths"] if c in com.columns]
    # json round-trip strips the numpy scalars a DataFrame leaves in records
    # (np.int64/np.bool_ break postgrest's stdlib-json serializer, PR #1098).
    records: List[dict] = json.loads(com[cols].to_json(orient="records"))
    return records


def comm_arm_rows_for_upsert() -> List[dict]:
    """The patient-grain commercial-arm edges as DB-shaped records for the
    targeted apply script (scripts/seed_comm_arm_causal_paths.py).

    Content-addressed (scp_a*), so the upsert on path_id is idempotent and needs
    NO full cohort reseed — the ``patient_journeys`` arm data already exists; this
    only adds the leaderboard-enumeration edges pointing at it. Same column
    projection as ``commercial_rows_for_upsert`` (the generator-only 'grain'
    column is dropped — the DB has no such column)."""
    import json

    from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS

    df = CausalPathsGenerator(GeneratorConfig(n_records=0)).generate()
    arm = df[df["grain"] == "patient_arm"]
    cols = [c for c in TABLE_COLUMNS["causal_paths"] if c in arm.columns]
    records: List[dict] = json.loads(arm[cols].to_json(orient="records"))
    return records


class CausalPathsGenerator(BaseGenerator[pd.DataFrame]):
    @property
    def entity_type(self) -> str:
        return "causal_paths"

    def generate(self) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        rows = []
        # Full cross-product so every (brand × outcome) cell is represented.
        # The old i%3 diagonal only emitted 3 of the 9 cells; decoupling the
        # indices seeds all 9 for the leaderboard and KG sync.
        cells = [(b, o) for b in _BRANDS for o in _COHORT_OUTCOMES]
        for i in range(self.config.n_records):
            brand, outcome = cells[i % len(cells)]
            effect = round(float(self._rng.uniform(0.10, 0.55)), 4)  # recoverable band
            direct = round(effect * float(self._rng.uniform(0.4, 0.8)), 4)
            indirect = round(effect - direct, 4)
            n_med = int(self._rng.integers(1, 3))
            mediators = [str(m) for m in self._rng.choice(_MEDIATORS, size=n_med, replace=False)]
            disc = (now - timedelta(days=int(self._rng.integers(0, 25)))).date()
            rows.append(
                {
                    # path_id is varchar(20) on the faithful DB -> a full uuid4 (36)
                    # overflows (22001). Use a short collision-safe synthetic id.
                    "path_id": f"scp_{uuid.uuid4().hex[:13]}",
                    "discovery_date": disc.isoformat(),
                    "causal_chain": {"nodes": ["treatment_arm", *mediators, outcome]},
                    "start_node": "treatment_arm",
                    "end_node": outcome,
                    "intermediate_nodes": mediators,
                    "path_length": n_med + 1,
                    "causal_effect_size": effect,
                    "confidence_level": round(float(self._rng.uniform(0.80, 0.95)), 3),
                    "method_used": "backdoor.linear_regression",
                    "confounders_controlled": list(_COHORT_CONFOUNDERS[outcome]),
                    "mediators_identified": mediators,
                    "time_lag_days": int(self._rng.integers(7, 60)),
                    "validation_status": "validated",
                    "business_impact_estimate": round(
                        effect * float(self._rng.uniform(1e5, 5e5)), 2
                    ),
                    "data_split": "unassigned",
                    "direct_effect": direct,
                    "indirect_effect": indirect,
                    "brand": brand,
                    "region": str(self._rng.choice(_REGIONS)),
                    "confirmation_count": int(self._rng.integers(1, 5)),
                    "created_at": now.isoformat(),
                    "is_synthetic": True,
                    "grain": "patient",
                }
            )
        # HCP-grain adoption edges — ADDITIVE, fixed 6-row block (2 questions x 3
        # brands), independent of n_records, so the SSOT always carries every HCP
        # question for the hcp_adoption-dataset leaderboard.
        for brand in _BRANDS:
            for start_node, end_node, confounders, mediator in _HCP_QUESTIONS:
                effect = round(float(self._rng.uniform(0.10, 0.55)), 4)
                direct = round(effect * float(self._rng.uniform(0.4, 0.8)), 4)
                indirect = round(effect - direct, 4)
                disc = (now - timedelta(days=int(self._rng.integers(0, 25)))).date()
                rows.append(
                    {
                        "path_id": f"scp_{uuid.uuid4().hex[:13]}",
                        "discovery_date": disc.isoformat(),
                        "causal_chain": {"nodes": [start_node, mediator, end_node]},
                        "start_node": start_node,
                        "end_node": end_node,
                        "intermediate_nodes": [mediator],
                        "path_length": 2,
                        "causal_effect_size": effect,
                        "confidence_level": round(float(self._rng.uniform(0.80, 0.95)), 3),
                        "method_used": "backdoor.linear_regression",
                        "confounders_controlled": list(confounders),
                        "mediators_identified": [mediator],
                        "time_lag_days": int(self._rng.integers(7, 60)),
                        "validation_status": "validated",
                        "business_impact_estimate": round(
                            effect * float(self._rng.uniform(1e5, 5e5)), 2
                        ),
                        "data_split": "unassigned",
                        "direct_effect": direct,
                        "indirect_effect": indirect,
                        "brand": brand,
                        "region": str(self._rng.choice(_REGIONS)),
                        "confirmation_count": int(self._rng.integers(1, 5)),
                        "created_at": now.isoformat(),
                        "is_synthetic": True,
                        "grain": "hcp",
                    }
                )
        # Trigger grain: emit each RCT/effect-modifier edge for every brand so a
        # brand-scoped leaderboard surfaces the trigger questions too. Empty
        # confounders_controlled (randomized / effect-modifier — no backdoor set);
        # direct two-node causal_chain; no mediators (mediators_identified=[] so the
        # FalkorDB sync builds a clean direct (:Variable start)-[:CAUSES]->(:Variable end)).
        for brand in _BRANDS:
            for start_node, end_node, confounders in _TRIGGER_EDGES:
                effect = round(float(self._rng.uniform(0.05, 0.25)), 4)
                disc = (now - timedelta(days=int(self._rng.integers(0, 25)))).date()
                rows.append(
                    {
                        "path_id": f"scp_{uuid.uuid4().hex[:13]}",
                        "discovery_date": disc.isoformat(),
                        "causal_chain": {"nodes": [start_node, end_node]},
                        "start_node": start_node,
                        "end_node": end_node,
                        "intermediate_nodes": [],
                        "path_length": 1,
                        "causal_effect_size": effect,
                        "confidence_level": round(float(self._rng.uniform(0.80, 0.95)), 3),
                        "method_used": "backdoor.linear_regression",
                        "confounders_controlled": list(confounders),
                        "mediators_identified": [],
                        "time_lag_days": int(self._rng.integers(7, 60)),
                        "validation_status": "validated",
                        "business_impact_estimate": round(
                            effect * float(self._rng.uniform(1e5, 5e5)), 2
                        ),
                        "data_split": "unassigned",
                        "direct_effect": effect,
                        "indirect_effect": 0.0,
                        "brand": brand,
                        "region": str(self._rng.choice(_REGIONS)),
                        "confirmation_count": int(self._rng.integers(1, 5)),
                        "created_at": now.isoformat(),
                        "is_synthetic": True,
                        "grain": "trigger",
                    }
                )
        # Commercial-KPI grain — ADDITIVE fixed block (15 edges x 3 brands),
        # content-addressed (see _COMMERCIAL_EDGES contract comment above).
        for brand in _BRANDS:
            for start_node, end_node, mediator, edge_confounders, lo, hi in _COMMERCIAL_EDGES:
                rng = _commercial_edge_rng(brand, start_node, end_node)
                effect = round(float(rng.uniform(lo, hi)), 4)
                direct = round(effect * float(rng.uniform(0.4, 0.8)), 4)
                indirect = round(effect - direct, 4)
                disc = (now - timedelta(days=int(rng.integers(0, 25)))).date()
                rows.append(
                    {
                        "path_id": _commercial_path_id(brand, start_node, end_node),
                        "discovery_date": disc.isoformat(),
                        "causal_chain": {"nodes": [start_node, mediator, end_node]},
                        "start_node": start_node,
                        "end_node": end_node,
                        "intermediate_nodes": [mediator],
                        "path_length": 2,
                        "causal_effect_size": effect,
                        "confidence_level": round(float(rng.uniform(0.75, 0.92)), 3),
                        "method_used": "backdoor.linear_regression",
                        "confounders_controlled": list(edge_confounders),
                        "mediators_identified": [mediator],
                        "time_lag_days": int(rng.integers(14, 90)),
                        "validation_status": "validated",
                        "business_impact_estimate": round(effect * float(rng.uniform(1e5, 5e5)), 2),
                        "data_split": "unassigned",
                        "direct_effect": direct,
                        "indirect_effect": indirect,
                        "brand": brand,
                        "region": str(rng.choice(_REGIONS)),
                        "confirmation_count": int(rng.integers(1, 5)),
                        "created_at": now.isoformat(),
                        "is_synthetic": True,
                        "grain": "commercial",
                    }
                )
        # Patient-grain commercial-arm edges — ADDITIVE fixed block (8 edges x 3
        # brands), content-addressed (idempotent targeted upsert; see
        # _COMM_ARM_EDGES). Direct 1-hop edges (no mediator, like _TRIGGER_EDGES)
        # so the FalkorDB sync builds (:Variable arm)-[:CAUSES]->(:Variable outcome).
        for brand in _BRANDS:
            for arm, outcome, arm_confounders, lo, hi in _COMM_ARM_EDGES:
                rng = _commercial_edge_rng(brand, arm, outcome)
                effect = round(float(rng.uniform(lo, hi)), 4)
                disc = (now - timedelta(days=int(rng.integers(0, 25)))).date()
                rows.append(
                    {
                        "path_id": _comm_arm_path_id(brand, arm, outcome),
                        "discovery_date": disc.isoformat(),
                        "causal_chain": {"nodes": [arm, outcome]},
                        "start_node": arm,
                        "end_node": outcome,
                        "intermediate_nodes": [],
                        "path_length": 1,
                        "causal_effect_size": effect,
                        "confidence_level": round(float(rng.uniform(0.80, 0.95)), 3),
                        "method_used": "backdoor.linear_regression",
                        "confounders_controlled": list(arm_confounders),
                        "mediators_identified": [],
                        "time_lag_days": int(rng.integers(7, 60)),
                        "validation_status": "validated",
                        "business_impact_estimate": round(effect * float(rng.uniform(1e5, 5e5)), 2),
                        "data_split": "unassigned",
                        "direct_effect": effect,
                        "indirect_effect": 0.0,
                        "brand": brand,
                        "region": str(rng.choice(_REGIONS)),
                        "confirmation_count": int(rng.integers(1, 5)),
                        "created_at": now.isoformat(),
                        "is_synthetic": True,
                        "grain": "patient_arm",
                    }
                )
        return pd.DataFrame(rows)
