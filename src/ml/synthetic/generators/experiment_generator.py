"""Synthetic experiment + A/B substrate (Shard 09).

Feeds ml_experiments (running shape like the 621 real), ab_experiment_assignments/
enrollments/results with KNOWN, recoverable per-channel uplifts. is_synthetic=true
on all rows.

The faithful read path: experiment_monitor selects ml_experiments WHERE
status='running' then counts ab_experiment_assignments per experiment. We mirror
the "all running" shape and attach assignments/enrollments/results so the monitor
reads real synthetic rows rather than fabricating health.

MEANINGFUL PORTFOLIO (2026-07-11, /experiments usefulness review): each experiment
is a self-explanatory in-silico A/B test of ONE intervention channel from the
user-approved digital-twin taxonomy (INTERVENTION_CATALOG in
src/digital_twin/effect/provider.py — mirrored here as _CHANNELS because importing
it would drag the whole twin stack through src/digital_twin/__init__; a sync test
asserts the two stay identical) against the brand's primary commercial outcome,
on a named HCP cohort and region. Channels carry HETEROGENEOUS ground-truth
uplifts — including a deliberate null (digital_engagement) — so downstream
effect-recovery and the strategic-insight ranking are honest, not a single
global constant. Start dates are staggered and enrollment rolls forward to the
generation frontier, so the monitor sees a living portfolio (varied enrollment,
information fractions, and freshness) instead of 360 clones stamped in one burst.

Enum-exact values (22P02 landmine): brand_type (Remibrutinib/Kisqali/Fabhalta),
region_type, ab_unit_type, randomization_method, ab_analysis_type/method,
enrollment_status, and the ml_experiments status CHECK (running). minimum_auc and
minimum_precision_at_k respect the valid_auc / valid_precision CHECKs.

IDEMPOTENT (reseed-safe): all ids are DETERMINISTIC uuid5 from their natural keys
(see ``_exp_id``) — the experiment id stays keyed on the LEGACY slug
``synth_<brand>_exp_NNNN`` even though the display name is now meaningful, so the
360 rows (and their entire FK fan-out: assignments, enrollments, results, and the
MLOps registry rows generated off the same ids) UPDATE in place across reseeds
instead of accumulating. Do NOT re-key the id on the display name.
"""

import math
import uuid
from datetime import datetime, timedelta, timezone
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
_TARGET_LABELS = {
    Brand.REMIBRUTINIB: "CSU treatment initiation",
    Brand.KISQALI: "Kisqali diagnostic adoption",
    Brand.FABHALTA: "PNH therapy persistence",
}

# Intervention taxonomy — MIRROR of the user-approved digital-twin catalog
# (src/digital_twin/effect/provider.py INTERVENTION_CATALOG; a sync test asserts
# value/label equality so the two vocabularies cannot drift).
# Per channel: (value, label, treatment action phrase, hypothesis rationale,
# GROUND-TRUTH uplift in absolute conversion points). digital_engagement is a
# deliberate NULL channel so effect recovery / insight ranking must earn its
# significance calls instead of finding uplift everywhere.
_CHANNELS: tuple = (
    (
        "email_campaign",
        "Email Campaign",
        "a targeted, brand-approved email sequence",
        "low-cost reach at scale; expected lift is modest and must clear "
        "engagement-fatigue effects",
        0.03,
    ),
    (
        "call_frequency_increase",
        "Increased Call Frequency",
        "an increased field-rep call frequency",
        "more face time with the rep should compound detailing effects, but "
        "risks diminishing returns in saturated territories",
        0.08,
    ),
    (
        "speaker_program_invitation",
        "Speaker Program Invitation",
        "invitations to peer-led speaker programs",
        "high-touch scientific exchange with peer KOLs historically shifts "
        "prescribing behavior more than any remote channel",
        0.16,
    ),
    (
        "sample_distribution",
        "Sample Distribution",
        "additional product sample allocations",
        "lowering the first-prescription barrier lets HCPs trial the therapy "
        "before committing formulary-constrained patients",
        0.10,
    ),
    (
        "peer_influence_activation",
        "Peer Influence Activation",
        "structured peer-to-peer KOL engagement",
        "local-network influence is the strongest observed adoption correlate; "
        "this tests whether activating it CAUSES adoption",
        0.19,
    ),
    (
        "digital_engagement",
        "Digital Engagement",
        "enhanced interactive digital content",
        "digital impressions correlate with adoption in observational data — "
        "this tests whether the relationship is causal or pure selection",
        0.00,
    ),
    (
        "patient_support_program",
        "Patient Support Program",
        "enrollment support into the patient-support program",
        "reducing patient out-of-pocket friction and onboarding burden should "
        "lift persistence and, indirectly, HCP confidence to prescribe",
        0.14,
    ),
    (
        "rep_training_quality",
        "Rep Training Quality",
        "coverage by reps who completed the enhanced training curriculum",
        "better-trained reps deliver higher-quality clinical dialogue; the "
        "expected lift is real but smaller than peer-led channels",
        0.05,
    ),
)

CHANNEL_TRUE_UPLIFT: Dict[str, float] = {c[0]: c[4] for c in _CHANNELS}
# value -> human label (consumed by src/insights/experiments.py so the insight
# narrates "Speaker Program Invitation", not the enum value).
CHANNEL_LABELS: Dict[str, str] = {c[0]: c[1] for c in _CHANNELS}

_COHORTS = [
    "new-writer HCPs",
    "lapsed prescribers",
    "high-volume specialists",
    "low-access territories",
]


class ExperimentGenerator(BaseGenerator[pd.DataFrame]):
    @property
    def entity_type(self) -> str:
        return "ml_experiments"

    def generate(self) -> pd.DataFrame:
        n = self.config.n_records
        brand = self.config.brand or Brand.KISQALI
        now = datetime.now(timezone.utc)
        outcome_label = _TARGET_LABELS[brand]
        rows = []
        for i in range(n):
            # LEGACY slug — the id's natural key. Keeping the id keyed on this
            # (not the display name) is what lets a redesigned portfolio UPDATE
            # the same 360 rows + FK fan-out in place. Do not change.
            legacy_slug = f"synth_{brand.value.lower()}_exp_{i:04d}"
            value, label, action, rationale, _uplift = _CHANNELS[i % len(_CHANNELS)]
            cohort = _COHORTS[(i // len(_CHANNELS)) % len(_COHORTS)]
            region = _REGIONS[(i // (len(_CHANNELS) * len(_COHORTS))) % len(_REGIONS)]
            experiment_name = (
                f"{brand.value}: {label} → {outcome_label} — {cohort}, {region} (#{i:03d})"
            )
            # Staggered starts (10–90 days before the generation frontier) so the
            # portfolio shows varied maturity instead of one same-instant burst.
            # The frontier rolls with each weekly refresh, keeping ages bounded.
            days_back = int(self._rng.integers(10, 91))
            created_at = now - timedelta(days=days_back)
            # Enrollment PLAN (migration 101): every experiment is planned at a
            # nominal 10 units/day over its planned window, while the REALIZED
            # rate varies 2.5–15/day (ABExperimentGenerator._units_for). The gap
            # between plan and reality is what makes the monitor's plan-relative
            # health checks honest: slow enrollers fall behind plan, fast ones
            # reach target early, and experiments older than their window show
            # a genuine overrun — no status is fabricated from a default target.
            planned_duration_days = int(self._rng.integers(45, 121))
            target_enrollment = planned_duration_days * 10
            description = (
                f"In-silico A/B test: does {label.lower()} increase {outcome_label} "
                f"among {cohort} in the {region} region for {brand.value}? "
                f"Treatment-arm HCPs receive {action}; controls keep standard "
                f"engagement. Hypothesis: {rationale}. "
                f"Simulated on the synthetic HCP panel (1:1 randomized) with a "
                f"known ground-truth effect so estimator recovery is verifiable. "
                f"Primary endpoint: {outcome_label} ({_TARGETS[brand]})."
            )
            rows.append(
                {
                    # Deterministic PK from the LEGACY natural key -> reseed
                    # UPDATES in place instead of accumulating a fresh-uuid row.
                    "id": _exp_id(legacy_slug),
                    "experiment_name": experiment_name,
                    "description": description,
                    "prediction_target": _TARGETS[brand],
                    "target_population": f"{cohort}, {region} region",
                    "intervention_channel": value,
                    "observation_window_days": int(self._rng.choice([90, 180, 365])),
                    "prediction_horizon_days": int(self._rng.choice([30, 60, 90])),
                    # valid_auc CHECK requires [0.5, 1.0]; valid_precision [0,1]
                    "minimum_auc": round(float(self._rng.uniform(0.65, 0.80)), 3),
                    "minimum_precision_at_k": round(float(self._rng.uniform(0.10, 0.40)), 3),
                    "maximum_fpr": round(float(self._rng.uniform(0.05, 0.20)), 3),
                    "brand": brand.value,
                    "region": region,
                    "created_by": "synthetic_loader",
                    "created_at": created_at.isoformat(),
                    "status": "running",  # the A/B portfolio is actively enrolling by design
                    "target_enrollment": target_enrollment,
                    "planned_duration_days": planned_duration_days,
                    "is_synthetic": True,
                }
            )
        return pd.DataFrame(rows)


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


class ABExperimentGenerator(BaseGenerator[pd.DataFrame]):
    """Builds assignments/enrollments/results referencing the experiments_df ids.

    Per-experiment design is derived from the experiment row itself:
    - true uplift = CHANNEL_TRUE_UPLIFT[intervention_channel] (falls back to the
      ``true_uplift`` ctor arg for frames without the channel column);
    - unit count scales with the experiment's age (enrollment-rate ~5–15/day,
      with a deterministic slow-enroller minority so the monitor's
      under-enrollment checks have something honest to catch);
    - assignments roll forward from the experiment start to the generation
      frontier (rolling enrollment), so freshness reflects the weekly refresh
      cadence instead of a single same-instant stamp.

    Result rows carry HONESTLY computed statistics: a two-proportion z-test
    p-value (erfc-based, no scipy), a CI from the unpooled standard error, and
    is_significant = p < 0.05 — the null channel really does come out null.
    """

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
        # Back-compat fallbacks for frames lacking channel/created_at metadata.
        self.units_per_experiment = units_per_experiment
        self.true_uplift = true_uplift

    @property
    def entity_type(self) -> str:
        return "ab_experiment_assignments"

    def _units_for(self, exp: pd.Series, now: datetime) -> int:
        """Unit count from the experiment's age × a deterministic daily rate."""
        created = exp.get("created_at")
        if not created:
            return self.units_per_experiment
        start = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        days = max(1, (now - start).days)
        # Deterministic from the (stable) experiment id, NOT the rng stream, so
        # unit counts never depend on generation order.
        h = int(str(exp["id"]).replace("-", ""), 16)
        if h % 17 == 0:
            rate = 2.5  # honest slow-enroller minority (~6%)
        else:
            rate = 5.0 + (h % 11)  # 5–15 units/day
        return int(min(1400, max(120, rate * days)))

    def generate(self) -> Dict[str, pd.DataFrame]:  # type: ignore[override]
        now = datetime.now(timezone.utc)
        asn_rows, enr_rows, res_rows = [], [], []
        for _, exp in self.experiments_df.iterrows():
            eid = exp["id"]
            channel = exp.get("intervention_channel")
            uplift = CHANNEL_TRUE_UPLIFT.get(channel, self.true_uplift)
            created = exp.get("created_at")
            if created:
                start = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                n_units = self._units_for(exp, now)
            else:
                start = now
                n_units = self.units_per_experiment
            span_s = max(0.0, (now - start).total_seconds())
            base_rate = float(self._rng.uniform(0.20, 0.45))  # control mean in recoverable band
            control_outcomes: list[float] = []
            treatment_outcomes: list[float] = []
            for u in range(n_units):
                variant = "treatment" if u % 2 == 0 else "control"
                unit_id = f"hcp_{u:05d}"
                # Deterministic id from the UNIQUE(experiment_id, unit_id) natural key so a
                # reseed UPDATES in place (eid is itself deterministic, so this is stable).
                aid = _exp_id("asn", eid, unit_id)
                # Rolling enrollment: unit u enrolls at its fraction of the
                # start→frontier span (minus a small jitter), so the NEWEST
                # assignment sits within hours of the frontier and staleness
                # reads the refresh cadence, not a frozen batch stamp.
                frac = (u + 1) / n_units
                jitter_s = float(self._rng.uniform(0, 6 * 3600))
                assigned_at = start + timedelta(seconds=max(0.0, frac * span_s - jitter_s))
                assigned_iso = assigned_at.isoformat()
                p = base_rate + (uplift if variant == "treatment" else 0.0)
                y = float(self._rng.binomial(1, min(0.99, max(0.01, p))))
                (treatment_outcomes if variant == "treatment" else control_outcomes).append(y)
                asn_rows.append(
                    {
                        "id": aid,
                        "experiment_id": eid,
                        "unit_id": unit_id,
                        "unit_type": "hcp",
                        "variant": variant,
                        "assigned_at": assigned_iso,
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
                        "enrolled_at": assigned_iso,
                        "enrollment_status": "active",
                        "eligibility_criteria_met": {"min_volume": True},
                        "eligibility_check_timestamp": assigned_iso,
                        "is_synthetic": True,
                    }
                )
            c, t = np.array(control_outcomes), np.array(treatment_outcomes)
            effect = float(t.mean() - c.mean())
            # Two-proportion z-test (unpooled SE): honest p-value/CI so the null
            # channel is NOT reported significant by construction.
            se = math.sqrt(
                max(1e-12, t.mean() * (1 - t.mean()) / t.size + c.mean() * (1 - c.mean()) / c.size)
            )
            z = effect / se
            p_value = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided
            observed_power = _norm_cdf(abs(z) - 1.959964)  # post-hoc, at alpha=0.05
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
                    "effect_ci_lower": effect - 1.959964 * se,
                    "effect_ci_upper": effect + 1.959964 * se,
                    "confidence_level": 0.95,
                    "p_value": round(p_value, 6),
                    "is_significant": bool(p_value < 0.05),
                    "observed_power": round(observed_power, 4),
                    "is_synthetic": True,
                }
            )
        return {
            "ab_experiment_assignments": pd.DataFrame(asn_rows),
            "ab_experiment_enrollments": pd.DataFrame(enr_rows),
            "ab_experiment_results": pd.DataFrame(res_rows),
        }
