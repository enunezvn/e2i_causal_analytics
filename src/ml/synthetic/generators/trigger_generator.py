"""
Trigger Generator.

Generates synthetic triggers for patient/HCP targeting actions.
"""

from typing import Dict, Optional, cast

import numpy as np
import pandas as pd

from ..config import Brand
from .base import BaseGenerator, GeneratorConfig

# Known-ground-truth trigger->prescription conversion lift (accepted-arm minus
# rejected-arm), in [0,1]. The conversion frame / kpi_query COMPUTES the realized
# lift; this constant only SEEDS the data (Shard 05). Mid-band of the +10-20pp target.
DESIGNED_CONVERSION_LIFT = 0.15
_P_REJECTED = 0.25  # base rejected-arm conversion (in the 0.20-0.50 recoverable band)
_P_ACCEPTED = _P_REJECTED + DESIGNED_CONVERSION_LIFT  # 0.40
# Per-priority lift scale (all > 0 => sign-stable CATE-by-priority heterogeneity).
_PRIORITY_LIFT_FACTOR = {"critical": 1.3, "high": 1.3, "medium": 1.0, "low": 0.7}

# #1118 WS2-TR-005 (False Alert Rate): P(field review marks a trigger as a
# false alert | outcome tracked AND no positive outcome materialized). A false
# positive can only be MARKED when the outcome was tracked and demonstrably
# did not pan out (outcome_value NULL or <= 0 — the exact complement of the
# TR-001 precision numerator `outcome_tracked AND outcome_value > 0`), and
# field review catches most-but-not-all of those. Realized rate over ALL
# triggers (the TR-005 denominator):
#   P(tracked)=0.40 x P(no positive outcome | tracked)~=1-P(accepted)~0.575
#   x 0.60 ~= 0.14  -> WARNING band (target 0.10 / warning 0.20), coherent
# with TR-001 precision ~0.38-0.43 CRITICAL from the same table+window.
_P_FALSE_POSITIVE_MARKED = 0.60


class TriggerGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for triggers.

    Generates trigger records with:
    - Priority levels (critical, high, medium, low)
    - Delivery channels and status
    - Causal chain and supporting evidence
    - Outcome tracking
    """

    # Trigger types based on agent actions
    TRIGGER_TYPES = [
        "prescription_opportunity",
        "adherence_risk",
        "churn_prevention",
        "cross_sell",
        "engagement_gap",
        "competitive_threat",
        "treatment_switch",
        "reactivation",
    ]

    # Priority distribution (weighted toward actionable)
    PRIORITY_DIST = {
        "critical": 0.10,
        "high": 0.30,
        "medium": 0.40,
        "low": 0.20,
    }

    # Delivery channels
    DELIVERY_CHANNELS = ["email", "crm", "mobile", "portal", "rep_alert"]

    # Delivery status values
    DELIVERY_STATUS_VALUES = ["pending", "delivered", "viewed", "failed"]

    # Acceptance status values. 'overridden' (#1119 WS2-TR-006): rep actively
    # overrode the recommendation with their own judgment — distinct from
    # 'rejected' (dismissed outright). Only delivered/viewed triggers can carry
    # a non-pending disposition (delivery gates acceptance, see below).
    ACCEPTANCE_STATUS_VALUES = ["pending", "accepted", "rejected", "expired", "overridden"]

    # P(acceptance_status | delivered or viewed), aligned with
    # ACCEPTANCE_STATUS_VALUES order. The 'overridden' mass (0.14 — just under
    # the TR-006 target 0.15: GOOD, but honestly earned and non-degenerate) is
    # carved out of pending/rejected/expired; 'accepted' stays at 0.50 so
    # TR-001 precision (~P(accepted)) and the designed trigger->prescription
    # conversion lift substrate (accepted-vs-rest arms) are unperturbed.
    ACCEPTANCE_STATUS_P = [0.12, 0.50, 0.15, 0.09, 0.14]

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "triggers"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        patient_df: Optional[pd.DataFrame] = None,
        hcp_df: Optional[pd.DataFrame] = None,
        treatment_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the trigger generator.

        Args:
            config: Generator configuration.
            patient_df: Patient DataFrame for foreign key integrity.
            hcp_df: HCP DataFrame for foreign key integrity.
            treatment_df: Existing treatment_events (the prescription substrate).
                Reserved for future de-duplication against baseline prescriptions;
                the conversion lift is realized via injected prescriptions exposed
                on `injected_prescriptions` (Shard 05).
        """
        super().__init__(config)
        self.patient_df = patient_df
        self.hcp_df = hcp_df
        self.treatment_df = treatment_df
        # treatment_events 'prescription' rows this generator appends to encode the
        # known accepted-vs-rejected conversion lift. The loader caller merges these
        # into datasets["treatment_events"] (scripts/load_synthetic_data.py).
        self.injected_prescriptions: pd.DataFrame = pd.DataFrame()

    def generate(self) -> pd.DataFrame:
        """
        Generate triggers.

        Returns:
            DataFrame with triggers matching schema.
        """
        n = self.config.n_records
        self._log(f"Generating {n} triggers...")

        if self.patient_df is not None:
            # Generate triggers linked to patients (hcp_id is read off the patient
            # row, so hcp_df is not required for the linked path — passing only
            # patient_df keeps the patient's brand/journey flowing into the trigger).
            records = []
            triggers_per_patient = max(1, n // len(self.patient_df))

            for _, patient in self.patient_df.iterrows():
                n_triggers = self._rng.integers(1, triggers_per_patient + 2)
                for _ in range(n_triggers):
                    record = self._generate_trigger_record(patient)
                    records.append(record)

            df = pd.DataFrame(records)
        else:
            df = self._generate_standalone_triggers(n)

        # Add IDs
        df["trigger_id"] = self._generate_ids("trg", len(df))

        # Assign splits based on trigger timestamps
        if "trigger_timestamp" in df.columns:
            df["data_split"] = self._assign_splits(df["trigger_timestamp"].tolist())

        # Encode the KNOWN trigger->prescription conversion lift: for each trigger,
        # draw "prescription lands in [trigger_ts, trigger_ts+30d]?" with a per-arm
        # probability (accepted > rejected), scaled by priority. The injected rows are
        # exposed via self.injected_prescriptions and MUST be merged into
        # treatment_events by the loader caller (load_synthetic_data.py, Task 4).
        self.injected_prescriptions = self._inject_conversion_prescriptions(df)

        self._log(f"Generated {len(df)} triggers")
        return df

    def _inject_conversion_prescriptions(self, triggers: pd.DataFrame) -> pd.DataFrame:
        """Build treatment_events 'prescription' rows that realize the designed
        accepted-vs-rejected conversion lift, each landing inside the trigger's
        30-day conversion window. Deterministic via the generator RNG.

        Returns a DataFrame with post-rename treatment_events columns (patient_id,
        brand, event_date, event_type, duration_days) self-stamped is_synthetic=True.
        Empty when there are no triggers.
        """
        if triggers is None or len(triggers) == 0:
            return pd.DataFrame()

        ts = pd.to_datetime(triggers["trigger_timestamp"])
        accepted = triggers["acceptance_status"].astype(str).str.lower().eq("accepted")
        if "priority" in triggers.columns:
            priority = triggers["priority"].astype(str).str.lower()
            factor = priority.map(_PRIORITY_LIFT_FACTOR).fillna(1.0).to_numpy(dtype=float)
        else:
            factor = np.ones(len(triggers), dtype=float)

        # Per-trigger injection probability: arm base * priority factor, clipped <=1.
        base = np.where(accepted.to_numpy(), _P_ACCEPTED, _P_REJECTED)
        p_inject = np.clip(base * factor, 0.0, 1.0)
        draw = self._rng.random(len(triggers))
        inject = draw < p_inject

        if not inject.any():
            return pd.DataFrame()

        sel = triggers[inject].reset_index(drop=True)
        sel_ts = ts[inject].reset_index(drop=True)
        # Offset 1..27 days after the trigger => inside the 30d conversion window.
        offsets = self._rng.integers(1, 28, size=len(sel))
        event_dates = [
            (t + pd.Timedelta(days=int(o))).strftime("%Y-%m-%d")
            for t, o in zip(sel_ts, offsets, strict=False)
        ]
        # #853: the trigger_timestamp is already capped at the rolling reference, but
        # adding the 1..27d conversion offset pushes the FINAL injected event_date past
        # the reference (up to ref+27d future-dated treatment_events). Cap the derived
        # event_date itself — the recency the source carries is preserved, only the
        # future tail collapses onto the reference. No-op when anchor_to_now is off.
        # The window invariant still holds: trigger_timestamp is also <= ref, so a
        # capped row stays event_date >= trigger_timestamp and within 0..27d (a tail
        # row may collapse to same-day — no longer STRICTLY after, but inside the 30d
        # conversion window the realized lift is computed over).
        event_dates = self._shift_dates_to_window(event_dates)
        if "brand" in sel.columns:
            brand_vals = sel["brand"].to_numpy()
        elif "brand_id" in sel.columns:
            brand_vals = sel["brand_id"].to_numpy()
        else:
            brand_vals = np.array([Brand.REMIBRUTINIB.value] * len(sel))
        return pd.DataFrame(
            {
                "treatment_event_id": self._generate_ids("trxc", len(sel)),
                "patient_id": sel["patient_id"].to_numpy(),
                "brand": brand_vals,
                "event_date": event_dates,
                "event_type": ["prescription"] * len(sel),
                "duration_days": self._rng.integers(7, 90, size=len(sel)),
                # data_split is NOT NULL on treatment_events; without it the concat
                # leaves NaN -> the loader sends explicit null -> 23502 and the injected
                # rows silently fail to load (the lift then never reaches the DB).
                "data_split": self._assign_splits(event_dates),
                # APPENDED to treatment_events AFTER the central is_synthetic stamp,
                # so self-stamp here or these rows leak into real-mode KPIs.
                "is_synthetic": True,
            }
        )

    def _generate_trigger_record(self, patient: pd.Series) -> Dict:
        """Generate a trigger record linked to patient."""
        # Select trigger type based on patient state
        engagement_score = patient.get("engagement_score", 5.0)
        treatment_initiated = patient.get("treatment_initiated", 0)

        trigger_type = self._select_trigger_type(engagement_score, treatment_initiated)

        # Priority based on engagement and treatment status
        priority = self._select_priority(engagement_score, treatment_initiated)

        # Confidence score (higher for clear-cut cases)
        confidence = self._calculate_confidence(engagement_score, trigger_type)

        # Timestamps. Cap the trigger time at the rolling-window reference under
        # anchoring (a recent journey + offset must not fire in the future); the
        # expiration is derived from the capped time so it stays a future expiry,
        # which is correct. No-op when anchor_to_now is off (Shard 04).
        journey_start = pd.to_datetime(patient.get("journey_start_date", "2023-01-01"))
        days_offset = self._rng.integers(7, 90)
        trigger_timestamp = self._anchor_cap_timestamp(
            journey_start + pd.Timedelta(days=int(days_offset))
        )

        # Lead time and expiration
        lead_time_days = self._rng.integers(3, 30)
        expiration_date = trigger_timestamp + pd.Timedelta(days=int(lead_time_days))

        # Delivery information
        delivery_channel = self._rng.choice(self.DELIVERY_CHANNELS)
        delivery_status = self._rng.choice(
            self.DELIVERY_STATUS_VALUES,
            p=[0.10, 0.60, 0.25, 0.05],
        )

        # Acceptance (only if delivered/viewed)
        if delivery_status in ["delivered", "viewed"]:
            acceptance_status = self._rng.choice(
                self.ACCEPTANCE_STATUS_VALUES,
                p=self.ACCEPTANCE_STATUS_P,
            )
        else:
            acceptance_status = "pending"

        # Outcome tracking (some triggers have measured outcomes)
        outcome_tracked = self._rng.random() < 0.40
        outcome_value = None
        if outcome_tracked and acceptance_status == "accepted":
            # Positive outcome more likely with high engagement
            outcome_value = round(self._rng.beta(2 + engagement_score / 5, 3) * 1.0, 3)

        # #1118 WS2-TR-005: false-positive marking. Only an outcome-tracked
        # trigger whose outcome demonstrably failed to materialize (no value,
        # or <= 0 — the complement of the TR-001 precision numerator) can be
        # marked a false alert; field review marks _P_FALSE_POSITIVE_MARKED of
        # those. Unconditional draw keeps the per-record RNG stream shape
        # stable (deterministic/seeded => reseed-idempotent).
        fp_draw = self._rng.random()
        false_positive_flag = bool(
            outcome_tracked
            and (outcome_value is None or outcome_value <= 0)
            and fp_draw < _P_FALSE_POSITIVE_MARKED
        )

        # Generate causal chain and evidence
        causal_chain = self._generate_causal_chain(trigger_type, engagement_score)
        supporting_evidence = self._generate_supporting_evidence(trigger_type)

        brand_value = patient.get("brand", Brand.REMIBRUTINIB.value)

        # #577 WS2-TR-003: randomized control-arm holdout + arm-conditioned
        # action_taken. This generator is the LOADER OF RECORD for triggers (via
        # scripts/load_synthetic_data.py), so it must mirror migration 051 +
        # data_generator.py or a fresh load reverts action_taken to all-NULL and
        # re-breaks the metric. control_group_flag=True => CONTROL (NBA withheld);
        # False => TREATMENT (NBA shown). Treatment draws a higher P(action
        # present) than control so a real incrementality signal exists; the
        # registry query COMPUTES the realized uplift — these P's only seed data.
        control_group_flag = bool(self._rng.random() < 0.28)
        p_action = 0.30 if control_group_flag else 0.38
        action_taken = (
            str(self._rng.choice(["called_patient", "scheduled_visit", "sent_info"]))
            if self._rng.random() < p_action
            else None
        )

        return {
            "patient_id": patient.get("patient_id", ""),
            "hcp_id": patient.get("hcp_id", ""),
            "trigger_timestamp": trigger_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "trigger_type": trigger_type,
            "priority": priority,
            "confidence_score": round(confidence, 3),
            "lead_time_days": lead_time_days,
            "expiration_date": expiration_date.strftime("%Y-%m-%d"),
            "delivery_channel": delivery_channel,
            "delivery_status": delivery_status,
            "acceptance_status": acceptance_status,
            "outcome_tracked": outcome_tracked,
            "outcome_value": outcome_value,
            "false_positive_flag": false_positive_flag,
            "trigger_reason": self._generate_trigger_reason(trigger_type),
            "causal_chain": causal_chain,
            "supporting_evidence": supporting_evidence,
            "recommended_action": self._generate_recommended_action(trigger_type),
            "brand": brand_value,
            "brand_id": brand_value,
            "action_taken": action_taken,
            "control_group_flag": control_group_flag,
        }

    def _select_trigger_type(
        self,
        engagement_score: float,
        treatment_initiated: int,
    ) -> str:
        """Select trigger type based on patient state."""
        if treatment_initiated == 1:
            # Patient already on treatment - focus on adherence/retention
            probs = {
                "adherence_risk": 0.35,
                "churn_prevention": 0.25,
                "cross_sell": 0.15,
                "treatment_switch": 0.10,
                "engagement_gap": 0.10,
                "reactivation": 0.05,
            }
        else:
            # Not yet on treatment - focus on acquisition
            probs = {
                "prescription_opportunity": 0.40,
                "engagement_gap": 0.25,
                "competitive_threat": 0.15,
                "cross_sell": 0.10,
                "reactivation": 0.10,
            }

        # Adjust for engagement
        if engagement_score < 4:
            # Low engagement → more gap/reactivation triggers
            if "engagement_gap" in probs:
                probs["engagement_gap"] *= 1.5
            if "reactivation" in probs:
                probs["reactivation"] *= 1.5

        # Normalize
        options = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights = weights / weights.sum()

        return cast(str, self._rng.choice(options, p=weights))

    def _select_priority(
        self,
        engagement_score: float,
        treatment_initiated: int,
    ) -> str:
        """Select priority based on patient state."""
        # Base distribution
        probs = list(self.PRIORITY_DIST.values())

        # High-value patients (low engagement + not initiated) get higher priority
        if engagement_score < 4 and treatment_initiated == 0:
            probs = [0.20, 0.40, 0.30, 0.10]  # Shift toward higher priority
        elif engagement_score > 7 and treatment_initiated == 1:
            probs = [0.05, 0.20, 0.45, 0.30]  # Lower priority (already engaged)

        return cast(str, self._rng.choice(list(self.PRIORITY_DIST.keys()), p=probs))

    def _calculate_confidence(
        self,
        engagement_score: float,
        trigger_type: str,
    ) -> float:
        """Calculate confidence score for trigger."""
        # Base confidence
        base_confidence = 0.70

        # Clearer signal for certain trigger types
        type_adjustment = {
            "adherence_risk": 0.10,
            "prescription_opportunity": 0.08,
            "churn_prevention": 0.05,
            "engagement_gap": 0.03,
        }.get(trigger_type, 0.0)

        # Engagement extremes are clearer signals
        engagement_factor = abs(engagement_score - 5) / 10.0 * 0.15

        confidence = base_confidence + type_adjustment + engagement_factor
        noise = self._rng.normal(0, 0.05)

        return float(np.clip(confidence + noise, 0.50, 0.99))

    def _generate_causal_chain(
        self,
        trigger_type: str,
        engagement_score: float,
    ) -> Dict:
        """Generate causal chain JSON."""
        chains = {
            "prescription_opportunity": {
                "root_cause": "high_prescriber_fit",
                "intermediate_factors": ["engagement_pattern", "treatment_gap"],
                "confidence": round(0.7 + engagement_score * 0.02, 2),
            },
            "adherence_risk": {
                "root_cause": "declining_engagement",
                "intermediate_factors": ["refill_pattern", "support_interaction"],
                "confidence": round(0.65 + (10 - engagement_score) * 0.03, 2),
            },
            "churn_prevention": {
                "root_cause": "competitor_activity",
                "intermediate_factors": ["price_sensitivity", "satisfaction_score"],
                "confidence": round(0.60 + self._rng.random() * 0.2, 2),
            },
        }
        return chains.get(trigger_type, {"root_cause": "model_prediction", "confidence": 0.70})

    def _generate_supporting_evidence(self, trigger_type: str) -> Dict:
        """Generate supporting evidence JSON."""
        return {
            "data_sources": ["claims_data", "engagement_logs", "prescription_history"],
            "model_version": f"v{self._rng.integers(1, 4)}.{self._rng.integers(0, 10)}",
            "feature_importance": {
                "engagement_recency": round(self._rng.uniform(0.1, 0.4), 2),
                "prescription_history": round(self._rng.uniform(0.1, 0.3), 2),
                "hcp_relationship": round(self._rng.uniform(0.05, 0.2), 2),
            },
        }

    def _generate_trigger_reason(self, trigger_type: str) -> str:
        """Generate human-readable trigger reason."""
        reasons = {
            "prescription_opportunity": "High prescriber fit score with treatment gap identified",
            "adherence_risk": "Declining engagement pattern suggests potential non-adherence",
            "churn_prevention": "Competitor activity detected in territory",
            "cross_sell": "Patient profile matches additional indication criteria",
            "engagement_gap": "Below-average engagement compared to similar HCPs",
            "competitive_threat": "Recent competitive detailing detected",
            "treatment_switch": "Patient may benefit from therapy adjustment",
            "reactivation": "Lapsed patient with high historical value",
        }
        return reasons.get(trigger_type, "Model-generated recommendation")

    def _generate_recommended_action(self, trigger_type: str) -> str:
        """Generate recommended action text."""
        actions = {
            "prescription_opportunity": "Schedule detail visit to discuss treatment benefits",
            "adherence_risk": "Initiate patient support program outreach",
            "churn_prevention": "Deploy competitive positioning materials",
            "cross_sell": "Present expanded indication data",
            "engagement_gap": "Increase touchpoint frequency",
            "competitive_threat": "Prioritize relationship-building activities",
            "treatment_switch": "Discuss alternative treatment options with HCP",
            "reactivation": "Re-engage with updated clinical evidence",
        }
        return actions.get(trigger_type, "Follow up with appropriate engagement")

    def _generate_standalone_triggers(self, n: int) -> pd.DataFrame:
        """Generate triggers without patient/HCP linkage."""
        patient_ids = self._generate_ids("pt", n, width=6)
        hcp_ids = self._generate_ids("hcp", max(100, n // 10))

        # Generate engagement scores for trigger selection
        engagement_scores = self._random_normal(5.0, 2.0, n, clip_min=0, clip_max=10)
        treatment_initiated = (engagement_scores > 5).astype(int)

        # Generate trigger types
        trigger_types = [
            self._select_trigger_type(eng, treat)
            for eng, treat in zip(engagement_scores, treatment_initiated, strict=False)
        ]

        # Generate priorities
        priorities = [
            self._select_priority(eng, treat)
            for eng, treat in zip(engagement_scores, treatment_initiated, strict=False)
        ]

        # Confidence scores
        confidences = [
            self._calculate_confidence(eng, ttype)
            for eng, ttype in zip(engagement_scores, trigger_types, strict=False)
        ]

        # Timestamps and dates
        trigger_timestamps = self._random_dates(n)
        lead_times = self._rng.integers(3, 30, size=n)
        expiration_dates = [
            (pd.to_datetime(ts) + pd.Timedelta(days=int(lt))).strftime("%Y-%m-%d")
            for ts, lt in zip(trigger_timestamps, lead_times, strict=False)
        ]

        # Delivery information
        delivery_channels = self._random_choice(self.DELIVERY_CHANNELS, n).tolist()
        delivery_statuses = self._random_choice(
            self.DELIVERY_STATUS_VALUES,
            n,
            p=[0.10, 0.60, 0.25, 0.05],
        ).tolist()

        # Acceptance statuses
        acceptance_statuses = []
        for ds in delivery_statuses:
            if ds in ["delivered", "viewed"]:
                acceptance_statuses.append(
                    self._rng.choice(self.ACCEPTANCE_STATUS_VALUES, p=self.ACCEPTANCE_STATUS_P)
                )
            else:
                acceptance_statuses.append("pending")

        # Outcome tracking
        outcome_tracked = self._rng.random(n) < 0.40
        outcome_values = [
            round(self._rng.beta(3, 3) * 1.0, 3) if tracked and acc == "accepted" else None
            for tracked, acc in zip(outcome_tracked, acceptance_statuses, strict=False)
        ]

        # #1118 WS2-TR-005: false-positive marking (mirrors
        # _generate_trigger_record — tracked AND no positive outcome AND field
        # review marks it, all off the seeded RNG => reseed-idempotent).
        unproductive = np.array(
            [v is None or v <= 0 for v in outcome_values],
            dtype=bool,
        )
        fp_draws = self._rng.random(n)
        false_positive_flags = (
            outcome_tracked & unproductive & (fp_draws < _P_FALSE_POSITIVE_MARKED)
        )

        # Brands
        brands_list: list[str]
        if self.config.brand:
            brands_list = [self.config.brand.value] * n
        else:
            brands_list = list(self._random_choice([b.value for b in Brand], n))

        # #577 WS2-TR-003: randomized control-arm holdout + arm-conditioned action_taken
        # (mirrors _generate_trigger_record + migration 051). control_group_flag=True =>
        # CONTROL (NBA withheld); False => TREATMENT (NBA shown). Treatment draws a higher
        # P(action present) than control so a real incrementality signal exists.
        control_group_flags = self._rng.random(n) < 0.28
        action_present = self._rng.random(n) < np.where(control_group_flags, 0.30, 0.38)
        action_choices = self._rng.choice(
            ["called_patient", "scheduled_visit", "sent_info"], size=n
        )
        action_taken_vals: list[str | None] = [
            str(choice) if present else None
            for present, choice in zip(action_present, action_choices, strict=False)
        ]

        return pd.DataFrame(
            {
                "patient_id": patient_ids,
                "hcp_id": self._random_choice(hcp_ids, n).tolist(),
                "trigger_timestamp": trigger_timestamps,
                "trigger_type": trigger_types,
                "priority": priorities,
                "confidence_score": np.round(confidences, 3),
                "lead_time_days": lead_times,
                "expiration_date": expiration_dates,
                "delivery_channel": delivery_channels,
                "delivery_status": delivery_statuses,
                "acceptance_status": acceptance_statuses,
                "outcome_tracked": outcome_tracked,
                "outcome_value": outcome_values,
                "false_positive_flag": false_positive_flags.tolist(),
                "trigger_reason": [self._generate_trigger_reason(tt) for tt in trigger_types],
                "causal_chain": [
                    self._generate_causal_chain(tt, eng)
                    for tt, eng in zip(trigger_types, engagement_scores, strict=False)
                ],
                "supporting_evidence": [
                    self._generate_supporting_evidence(tt) for tt in trigger_types
                ],
                "recommended_action": [
                    self._generate_recommended_action(tt) for tt in trigger_types
                ],
                "brand": brands_list,
                "brand_id": brands_list,
                "action_taken": action_taken_vals,
                "control_group_flag": control_group_flags.tolist(),
            }
        )
