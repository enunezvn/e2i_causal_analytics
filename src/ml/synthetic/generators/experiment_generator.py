"""Synthetic experiment + A/B substrate (Shard 09).

Feeds ml_experiments (running shape like the 621 real), ab_experiment_assignments/
enrollments/results with a KNOWN, recoverable uplift. is_synthetic=true on all rows.

The faithful read path: experiment_monitor selects ml_experiments WHERE
status='running' then counts ab_experiment_assignments per experiment. We mirror
the "all running" shape and attach assignments/enrollments/results so the monitor
reads real synthetic rows rather than fabricating health.

Enum-exact values (22P02 landmine): brand_type (Remibrutinib/Kisqali/Fabhalta),
region_type, ab_unit_type, randomization_method, ab_analysis_type/method,
enrollment_status, and the ml_experiments status CHECK (running). minimum_auc and
minimum_precision_at_k respect the valid_auc / valid_precision CHECKs.

IDEMPOTENT (reseed-safe): all ids are DETERMINISTIC uuid5 from their natural keys
(see ``_exp_id``) — experiment id from experiment_name; assignment id from
(experiment_id, unit_id) (the DB UNIQUE); enrollment id from assignment_id; result id
from (experiment_id, analysis_type). So a reseed UPDATES the prior rows in place
instead of accumulating fresh-uuid rows every run.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import Brand
from .base import BaseGenerator, GeneratorConfig

# Fixed namespace for DETERMINISTIC ids (cf. mlops_generator._MLOPS_ID_NS). uuid.uuid4()
# ignored the seed, so every reseed INSERTed fresh-id rows that the loader's upsert-on-PK
# could never match -> the entity substrate ACCUMULATED (ml_experiments 6x the intended
# 360; ab_experiment_assignments 864k). On an include-synthetic showcase instance that
# inflated the read-side counts (e.g. the "Active Campaigns" tile). Deriving each id by
# uuid5 from its NATURAL KEY makes the upsert UPDATE in place (idempotent) and keeps the
# FK chain (experiment_id / assignment_id) stable across runs. ab_experiment_assignments
# additionally carries UNIQUE(experiment_id, unit_id): once experiment_id is deterministic
# that key is stable, so the assignment id MUST key on the same natural key or the upsert
# would collide (23505) instead of updating.
_EXP_ID_NS = uuid.UUID("5d3a8c14-6e2b-4f70-9a18-2b7c4e9f01a3")


def _exp_id(*parts: str) -> str:
    """Deterministic uuid5 from a natural key (stable across runs)."""
    return str(uuid.uuid5(_EXP_ID_NS, "|".join(str(p) for p in parts)))


_REGIONS = ["northeast", "south", "midwest", "west"]
_TARGETS = {
    Brand.REMIBRUTINIB: "csu_treatment_initiation",
    Brand.KISQALI: "kisqali_dx_adoption",
    Brand.FABHALTA: "pnh_persistence",
}


class ExperimentGenerator(BaseGenerator[pd.DataFrame]):
    @property
    def entity_type(self) -> str:
        return "ml_experiments"

    def generate(self) -> pd.DataFrame:
        n = self.config.n_records
        brand = self.config.brand or Brand.KISQALI
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(n):
            experiment_name = f"synth_{brand.value.lower()}_exp_{i:04d}"
            rows.append(
                {
                    # Deterministic PK from the experiment_name natural key -> reseed
                    # UPDATES in place instead of accumulating a fresh-uuid row.
                    "id": _exp_id(experiment_name),
                    "experiment_name": experiment_name,
                    "description": "Synthetic causal-validation experiment (Shard 09).",
                    "prediction_target": _TARGETS[brand],
                    "observation_window_days": int(self._rng.choice([90, 180, 365])),
                    "prediction_horizon_days": int(self._rng.choice([30, 60, 90])),
                    # valid_auc CHECK requires [0.5, 1.0]; valid_precision [0,1]
                    "minimum_auc": round(float(self._rng.uniform(0.65, 0.80)), 3),
                    "minimum_precision_at_k": round(float(self._rng.uniform(0.10, 0.40)), 3),
                    "maximum_fpr": round(float(self._rng.uniform(0.05, 0.20)), 3),
                    "brand": brand.value,
                    "region": str(self._rng.choice(_REGIONS)),
                    "created_by": "synthetic_loader",
                    "created_at": now.isoformat(),
                    "status": "running",  # mirrors the 621 real running experiments
                    "is_synthetic": True,
                }
            )
        return pd.DataFrame(rows)


class ABExperimentGenerator(BaseGenerator[pd.DataFrame]):
    """Builds assignments/enrollments/results referencing the experiments_df ids."""

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        experiments_df: Optional[pd.DataFrame] = None,
        units_per_experiment: int = 60,
        true_uplift: float = 0.15,
    ):
        super().__init__(config)
        if experiments_df is None or experiments_df.empty:
            raise ValueError("ABExperimentGenerator requires a non-empty experiments_df")
        self.experiments_df = experiments_df
        self.units_per_experiment = units_per_experiment
        self.true_uplift = true_uplift

    @property
    def entity_type(self) -> str:
        return "ab_experiment_assignments"

    def generate(self) -> Dict[str, pd.DataFrame]:  # type: ignore[override]
        now = datetime.now(timezone.utc)
        asn_rows, enr_rows, res_rows = [], [], []
        for _, exp in self.experiments_df.iterrows():
            eid = exp["id"]
            base_rate = float(self._rng.uniform(0.20, 0.45))  # control mean in recoverable band
            control_outcomes: list[float] = []
            treatment_outcomes: list[float] = []
            for u in range(self.units_per_experiment):
                variant = "treatment" if u % 2 == 0 else "control"
                unit_id = f"hcp_{u:05d}"
                # Deterministic id from the UNIQUE(experiment_id, unit_id) natural key so a
                # reseed UPDATES in place (eid is itself deterministic, so this is stable).
                aid = _exp_id("asn", eid, unit_id)
                p = base_rate + (self.true_uplift if variant == "treatment" else 0.0)
                y = float(self._rng.binomial(1, min(0.99, max(0.01, p))))
                (treatment_outcomes if variant == "treatment" else control_outcomes).append(y)
                asn_rows.append(
                    {
                        "id": aid,
                        "experiment_id": eid,
                        "unit_id": unit_id,
                        "unit_type": "hcp",
                        "variant": variant,
                        "assigned_at": now.isoformat(),
                        "randomization_method": "stratified",
                        "stratification_key": {"region": exp["region"]},
                        "assignment_hash": uuid.uuid5(uuid.NAMESPACE_OID, aid).hex,
                        "created_by": "synthetic_loader",
                        "is_synthetic": True,
                    }
                )
                enr_rows.append(
                    {
                        "id": _exp_id("enr", aid),
                        "assignment_id": aid,
                        "enrolled_at": now.isoformat(),
                        "enrollment_status": "active",
                        "eligibility_criteria_met": {"min_volume": True},
                        "eligibility_check_timestamp": now.isoformat(),
                        "is_synthetic": True,
                    }
                )
            c, t = np.array(control_outcomes), np.array(treatment_outcomes)
            effect = float(t.mean() - c.mean())
            res_rows.append(
                {
                    "id": _exp_id("res", eid, "final"),
                    "experiment_id": eid,
                    "analysis_type": "final",
                    "analysis_method": "itt",
                    "computed_at": now.isoformat(),
                    "primary_metric": "conversion_rate",
                    "control_mean": float(c.mean()),
                    "control_std": float(c.std()),
                    "control_n": int(c.size),
                    "treatment_mean": float(t.mean()),
                    "treatment_std": float(t.std()),
                    "treatment_n": int(t.size),
                    "effect_estimate": effect,
                    "effect_type": "absolute_difference",
                    "effect_ci_lower": effect - 0.05,
                    "effect_ci_upper": effect + 0.05,
                    "confidence_level": 0.95,
                    "p_value": 0.01,
                    "is_significant": True,
                    "observed_power": 0.80,
                    "is_synthetic": True,
                }
            )
        return {
            "ab_experiment_assignments": pd.DataFrame(asn_rows),
            "ab_experiment_enrollments": pd.DataFrame(enr_rows),
            "ab_experiment_results": pd.DataFrame(res_rows),
        }
