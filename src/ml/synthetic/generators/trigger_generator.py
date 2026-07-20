"""
Trigger Generator.

Generates synthetic triggers for patient/HCP targeting actions.
"""

from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from ..config import (
    TRIGGER_ACCEPTANCE_STATUS_VALUES,
    TRIGGER_DELIVERY_CHANNELS,
    TRIGGER_DELIVERY_STATUS_VALUES,
    TRIGGER_PRIORITY_VALUES,
    Brand,
)
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
# did not pan out (outcome_value <= 0 — the exact complement of the
# conversion_flag numerator), and field review catches most-but-not-all of
# those. COMM-ARMS Phase 4 (2026-07-20): "no positive outcome" now means "no
# REAL prescription landed in the 30d window" (the decoupled substrate), so the
# realized rate over ALL triggers (the TR-005 denominator) becomes
#   P(tracked)=0.40 x P(no Rx in-window | tracked)~0.42 x 0.60 ~= 0.10
# -> the GOOD/WARNING boundary (target 0.10 / warning 0.20). The pre-P4 level
# was ~0.14 (its complement was the acceptance coin, ~0.575) — the shift is a
# DEFINITION-DRIVEN discontinuity, disclosed in the Phase 4 PR body.
_P_FALSE_POSITIVE_MARKED = 0.60

# #1188 (WS2-TR-003 + nba_triggers RCT): arm-conditioned, baseline-PROGNOSTIC
# action probability. Arm assignment stays a pure coin flip (~28% control) —
# the RCT's empty backdoor is untouched — but P(action | arm) now also depends
# on PRE-TREATMENT patient baselines (disease_severity primary,
# age_at_diagnosis secondary). This plants the prognostic outcome signal an
# ANCOVA-style baseline adjustment exploits for variance reduction (measured
# on the planted DGP: baseline R^2 ~0.1 -> the adjusted ATE interval is ~10%
# narrower than the unadjusted diff-in-means while both stay unbiased).
# Balanced across arms by construction => the ~8pp arm contrast is preserved
# (clipping erodes it slightly; the realized WS2-TR-003 relative uplift stays
# above the 0.15 YAML target). Mirrored by migration 106 for live rows.
_P_ACTION_CONTROL = 0.30
_P_ACTION_TREATMENT = 0.38
_ACTION_SEVERITY_SLOPE = 0.12  # per severity unit, centered at 5.0 (~N(5,2))
_ACTION_SEVERITY_CENTER = 5.0
_ACTION_AGE_SLOPE = -0.002  # per year, centered at 50 (uniform 18-85)
_ACTION_AGE_CENTER = 50.0
_ACTION_P_FLOOR = 0.02
_ACTION_P_CEIL = 0.95


def _prognostic_action_probability(
    arm_base: float, disease_severity: Any, age_at_diagnosis: Any
) -> float:
    """P(action present | arm, baselines) for one trigger.

    Missing / non-numeric baselines fall back to the arm base (pre-#1188
    callers pass patient frames without baseline columns). Only the THRESHOLD
    changes with baselines — callers must keep their single RNG draw so the
    per-record stream shape stays reseed-deterministic.
    """
    p = arm_base
    try:
        if disease_severity is not None and not pd.isna(disease_severity):
            p += _ACTION_SEVERITY_SLOPE * (float(disease_severity) - _ACTION_SEVERITY_CENTER)
    except (TypeError, ValueError):
        pass
    try:
        if age_at_diagnosis is not None and not pd.isna(age_at_diagnosis):
            p += _ACTION_AGE_SLOPE * (float(age_at_diagnosis) - _ACTION_AGE_CENTER)
    except (TypeError, ValueError):
        pass
    return float(min(max(p, _ACTION_P_FLOOR), _ACTION_P_CEIL))


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

    # Priority distribution (weighted toward actionable). Keys are the shared
    # TRIGGER_PRIORITY_VALUES (config.py — matches the DB enum priority_type,
    # #1125); their ORDER is load-bearing for this and the positional
    # probability vectors in _select_priority. zip preserves that order, so
    # emission behavior is byte-identical to the previous dict literal.
    PRIORITY_DIST = dict(zip(TRIGGER_PRIORITY_VALUES, [0.10, 0.30, 0.40, 0.20], strict=True))

    # Delivery channels (single source of truth in config.py, shared with the
    # pandera TriggerSchema — #1125)
    DELIVERY_CHANNELS = TRIGGER_DELIVERY_CHANNELS

    # Delivery status values (shared with TriggerSchema via config.py — #1125)
    DELIVERY_STATUS_VALUES = TRIGGER_DELIVERY_STATUS_VALUES

    # Acceptance status values (shared via config.py). 'overridden' (#1119
    # WS2-TR-006): rep actively overrode the recommendation with their own
    # judgment — distinct from 'rejected' (dismissed outright). Only
    # delivered/viewed triggers can carry a non-pending disposition (delivery
    # gates acceptance, see below).
    ACCEPTANCE_STATUS_VALUES = TRIGGER_ACCEPTANCE_STATUS_VALUES

    # P(acceptance_status | delivered or viewed), aligned with
    # ACCEPTANCE_STATUS_VALUES order. The 'overridden' mass (0.14 — just under
    # the TR-006 target 0.15: GOOD, but honestly earned and non-degenerate) is
    # carved out of pending/rejected/expired; 'accepted' stays at 0.50.
    # COMM-ARMS Phase 4: on the linked path with a patient-level trigger_accepted
    # arm, these are the TARGET MARGINALS the arm-conditional mixture
    # (_acceptance_mixture) reproduces — arm=1 patients draw from q1
    # (accepted-rich), arm=0 from q0 (accepted mass ZERO), with
    # share*q1 + (1-share)*q0 == this vector by construction, so TR-004/TR-006
    # do not move. Standalone / arm-less paths still draw from this vector
    # directly.
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
            # COMM-ARMS Phase 4: when the patient frame carries the trigger_accepted
            # arm, acceptance_status is drawn from the arm-conditional mixture and
            # per-patient consistency is enforced (arm=1 <=> >=1 accepted trigger).
            mixture = self._acceptance_mixture()
            records = []
            triggers_per_patient = max(1, n // len(self.patient_df))

            for _, patient in self.patient_df.iterrows():
                n_triggers = self._rng.integers(1, triggers_per_patient + 2)
                arm = None
                if mixture is not None:
                    raw_arm = patient.get("trigger_accepted")
                    if raw_arm is not None and not pd.isna(raw_arm):
                        arm = int(raw_arm)
                start = len(records)
                for _ in range(n_triggers):
                    record = self._generate_trigger_record(
                        patient,
                        acceptance_p=mixture[arm] if mixture and arm is not None else None,
                    )
                    records.append(record)
                if arm == 1:
                    self._enforce_arm_consistency(records, start)

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
        inject_mask, self.injected_prescriptions = self._inject_conversion_prescriptions(df)

        # COMM-ARMS Phase 4: outcome_value is DECOUPLED from acceptance — the outcome
        # is the REAL downstream prescription inside the 30d window (injected OR
        # baseline treatment_events), for triggers of EVERY acceptance status.
        df = self._finalize_outcomes(df, inject_mask)

        self._log(f"Generated {len(df)} triggers")
        return df

    def _acceptance_mixture(self) -> Optional[Dict[int, list]]:
        """Arm-conditional acceptance distributions q1 (arm=1) / q0 (arm=0) whose
        share-weighted mixture reproduces ACCEPTANCE_STATUS_P EXACTLY, with
        q0[accepted] = 0 (an arm=0 patient can never carry an accepted trigger).

        q1[accepted] = p[accepted] / share (capped 0.97 for degenerate tiny shares);
        the residual q1 mass is split over the non-accepted statuses proportionally
        to their target masses; q0 solves the mixture equation per status. Returns
        None when the patient frame has no populated trigger_accepted column (legacy
        callers keep the base draw)."""
        if self.patient_df is None or "trigger_accepted" not in self.patient_df.columns:
            return None
        arm_col = self.patient_df["trigger_accepted"]
        if arm_col.isna().all():
            return None
        share = float(arm_col.fillna(0).astype(float).mean())
        statuses = list(self.ACCEPTANCE_STATUS_VALUES)
        p = np.asarray(self.ACCEPTANCE_STATUS_P, dtype=float)
        acc_idx = statuses.index("accepted")
        if share <= p[acc_idx]:
            # Not enough arm mass to carry the whole accepted marginal — q1 goes
            # fully accepted and the realized marginal degrades to `share` (loud in
            # the marginal-preservation test rather than a silent negative q0).
            q1 = np.zeros(len(p))
            q1[acc_idx] = 1.0
        else:
            q1 = np.zeros(len(p))
            q1[acc_idx] = min(p[acc_idx] / share, 0.97)
            rest = 1.0 - q1[acc_idx]
            non_acc_mass = 1.0 - p[acc_idx]
            for i in range(len(p)):
                if i != acc_idx:
                    q1[i] = rest * p[i] / non_acc_mass
        q0 = np.zeros(len(p))
        for i in range(len(p)):
            if i != acc_idx:
                q0[i] = max(p[i] - share * q1[i], 0.0) / max(1.0 - share, 1e-9)
        total = q0.sum()
        if total > 0:
            q0 = q0 / total
        return {1: q1.tolist(), 0: q0.tolist()}

    def _enforce_arm_consistency(self, records: list, start: int) -> None:
        """arm=1 patients MUST carry >=1 accepted trigger. Deterministic
        post-processing (no RNG draws — the per-record stream shape is untouched):
        if none of the patient's delivered/viewed triggers drew 'accepted', promote
        the first delivered/viewed one; if the patient has NO delivered/viewed
        trigger at all, force the first trigger delivered+accepted."""
        block = records[start:]
        if any(r["acceptance_status"] == "accepted" for r in block):
            return
        for r in block:
            if r["delivery_status"] in ("delivered", "viewed"):
                r["acceptance_status"] = "accepted"
                return
        block[0]["delivery_status"] = "delivered"
        block[0]["acceptance_status"] = "accepted"

    def _baseline_rx_in_window(self, triggers: pd.DataFrame) -> np.ndarray:
        """Per-trigger flag: a BASELINE treatment_events prescription (the journey
        substrate, NOT the injected conversions) for the same patient+brand lands
        inside [trigger_ts, trigger_ts + 30d]."""
        hit = np.zeros(len(triggers), dtype=bool)
        tx = self.treatment_df
        if tx is None or len(tx) == 0 or "patient_id" not in tx.columns:
            return hit
        if "event_type" in tx.columns:
            tx = tx[tx["event_type"].astype(str) == "prescription"]
        if len(tx) == 0:
            return hit
        brand_col = "brand" if "brand" in tx.columns else None
        dates_by_key: Dict[Any, np.ndarray] = {}
        ev_dates = pd.to_datetime(tx["event_date"], errors="coerce")
        keys = zip(tx["patient_id"], tx[brand_col], strict=False) if brand_col else tx["patient_id"]
        for key, d in zip(keys, ev_dates, strict=False):
            if pd.isna(d):
                continue
            dates_by_key.setdefault(key, []).append(d.to_datetime64())
        dates_by_key = {k: np.sort(np.asarray(v)) for k, v in dates_by_key.items()}
        ts = pd.to_datetime(triggers["trigger_timestamp"]).to_numpy()
        tbrand = triggers["brand"] if "brand" in triggers.columns else triggers.get("brand_id")
        for i, (pid, t) in enumerate(zip(triggers["patient_id"], ts, strict=False)):
            key = (pid, tbrand.iloc[i]) if brand_col and tbrand is not None else pid
            dates = dates_by_key.get(key)
            if dates is None or len(dates) == 0:
                continue
            lo = np.searchsorted(dates, t, side="left")
            if lo < len(dates) and dates[lo] <= t + np.timedelta64(30, "D"):
                hit[i] = True
        return hit

    def _finalize_outcomes(self, df: pd.DataFrame, inject_mask: np.ndarray) -> pd.DataFrame:
        """Resolve the TRI-STATE outcome_value from the REAL prescription substrate,
        acceptance-independent (COMM-ARMS Phase 4 decoupling):

        * NULL  iff NOT outcome_tracked (outcome not observable in CRM);
        * > 0   when tracked AND a prescription landed in the 30d window (injected
                conversion OR baseline treatment_events) — magnitude from the
                per-record beta draw (clamped positive so the DB stored-generated
                conversion_flag = outcome_value > 0 reads true);
        * 0.0   when tracked AND no prescription landed (a tracked miss —
                conversion_flag false, false-alert candidate).

        false_positive_flag keeps its #1118 semantics (tracked AND unproductive AND
        field review marks _P_FALSE_POSITIVE_MARKED of those) — computed here from
        the per-record fp draw so it sees the FINAL outcome."""
        rx_in_window = np.asarray(inject_mask, dtype=bool) | self._baseline_rx_in_window(df)
        tracked = df["outcome_tracked"].to_numpy(dtype=bool)
        mag = df.pop("_outcome_magnitude").to_numpy(dtype=float)
        fp_draw = df.pop("_fp_draw").to_numpy(dtype=float)
        mag = np.where(mag > 0, mag, 0.001)  # a landed Rx must read converted
        df["outcome_value"] = [
            (round(float(m), 3) if hit else 0.0) if t else None
            for t, hit, m in zip(tracked, rx_in_window, mag, strict=False)
        ]
        df["false_positive_flag"] = (
            tracked & ~rx_in_window & (fp_draw < _P_FALSE_POSITIVE_MARKED)
        ).tolist()
        return df

    def _inject_conversion_prescriptions(
        self, triggers: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Build treatment_events 'prescription' rows that realize the designed
        accepted-vs-rejected conversion lift, each landing inside the trigger's
        30-day conversion window. Deterministic via the generator RNG.

        Returns ``(inject_mask, frame)``: the per-trigger boolean injection mask
        (consumed by _finalize_outcomes — Phase 4 derives outcome_value from the
        REAL prescription substrate, so the mask is the injection channel's half of
        rx_in_window) and a DataFrame with post-rename treatment_events columns
        (patient_id, brand, event_date, event_type, duration_days) self-stamped
        is_synthetic=True. Empty mask/frame when there are no triggers.
        """
        if triggers is None or len(triggers) == 0:
            return np.zeros(0, dtype=bool), pd.DataFrame()

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
            return inject, pd.DataFrame()

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
        return inject, pd.DataFrame(
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

    def _generate_trigger_record(
        self, patient: pd.Series, acceptance_p: Optional[list] = None
    ) -> Dict:
        """Generate a trigger record linked to patient.

        ``acceptance_p`` (COMM-ARMS Phase 4) overrides the acceptance-status
        distribution with the patient's arm-conditional mixture (q1/q0 from
        _acceptance_mixture); None keeps the legacy base draw."""
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

        # Acceptance (only if delivered/viewed). Phase 4: the linked path draws
        # from the patient's arm-conditional mixture when supplied.
        if delivery_status in ["delivered", "viewed"]:
            acceptance_status = self._rng.choice(
                self.ACCEPTANCE_STATUS_VALUES,
                p=acceptance_p if acceptance_p is not None else self.ACCEPTANCE_STATUS_P,
            )
        else:
            acceptance_status = "pending"

        # Outcome tracking (CRM observability coin). Phase 4 DECOUPLING: the
        # outcome VALUE is no longer assigned here — it is resolved in
        # _finalize_outcomes from the REAL prescription substrate (injected
        # conversions + baseline treatment_events inside the 30d window),
        # acceptance-independent. The per-record magnitude + false-positive
        # draws stay HERE (unconditional) so the record-level RNG stream shape
        # is deterministic/seeded => reseed-idempotent.
        outcome_tracked = self._rng.random() < 0.40
        outcome_magnitude = round(self._rng.beta(2 + engagement_score / 5, 3) * 1.0, 3)
        fp_draw = self._rng.random()

        # Generate causal chain and evidence
        causal_chain = self._generate_causal_chain(trigger_type, engagement_score)
        supporting_evidence = self._generate_supporting_evidence(trigger_type)

        brand_value = patient.get("brand", Brand.REMIBRUTINIB.value)

        # #577 WS2-TR-003: randomized control-arm holdout + arm-conditioned
        # action_taken. This generator is the LOADER OF RECORD for triggers (via
        # scripts/load_synthetic_data.py), so it must mirror migrations 051/106 +
        # data_generator.py or a fresh load reverts action_taken to all-NULL and
        # re-breaks the metric. control_group_flag=True => CONTROL (NBA withheld);
        # False => TREATMENT (NBA shown). Treatment draws a higher P(action
        # present) than control so a real incrementality signal exists; the
        # registry query COMPUTES the realized uplift — these P's only seed data.
        # #1188: the probability is additionally PROGNOSTIC on the patient's
        # pre-treatment baselines (same single draw — stream shape unchanged;
        # assignment stays a pure coin flip).
        control_group_flag = bool(self._rng.random() < 0.28)
        p_action = _prognostic_action_probability(
            _P_ACTION_CONTROL if control_group_flag else _P_ACTION_TREATMENT,
            patient.get("disease_severity"),
            patient.get("age_at_diagnosis"),
        )
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
            # temp columns consumed + dropped by _finalize_outcomes (Phase 4)
            "_outcome_magnitude": outcome_magnitude,
            "_fp_draw": fp_draw,
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

        # Outcome tracking. Phase 4 DECOUPLING: values are resolved in
        # _finalize_outcomes from the real prescription substrate; here only the
        # observability coin + the per-record magnitude/false-positive draws
        # (unconditional, seeded => reseed-idempotent).
        outcome_tracked = self._rng.random(n) < 0.40
        outcome_magnitudes = np.round(self._rng.beta(3, 3, size=n) * 1.0, 3)
        fp_draws = self._rng.random(n)

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
        # #1188: deliberately NOT baseline-prognostic here — standalone triggers
        # fabricate patient_ids with no patient_journeys row to join baselines
        # from, so a prognostic term would be unobservable noise. The prognostic
        # substrate lives on the LINKED path (the load path of record).
        control_group_flags = self._rng.random(n) < 0.28
        action_present = self._rng.random(n) < np.where(
            control_group_flags, _P_ACTION_CONTROL, _P_ACTION_TREATMENT
        )
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
                # temp columns consumed + dropped by _finalize_outcomes (Phase 4)
                "_outcome_magnitude": outcome_magnitudes,
                "_fp_draw": fp_draws,
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
