"""Synthetic causal_paths substrate (Shard 09) for CM-003 (causal_impact) and
CM-005 (mediation).

The causal_paths table exists on the faithful DB (50 stale rows, max discovery_date
2025-10-01). We INSERT minimal is_synthetic=true rows so causal_effect_size and
mediators_identified are non-NULL for those 2 KPIs; we do NOT touch the real rows.

NOT-NULL columns verified on the faithful DB: path_id, discovery_date, causal_chain
(jsonb), data_split (enum), created_at, confirmation_count, is_synthetic. data_split
is enum-exact (data_split_type: train/validation/test/holdout/unassigned).
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pandas as pd

from .base import BaseGenerator

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
        return pd.DataFrame(rows)
