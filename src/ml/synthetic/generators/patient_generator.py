"""
Patient Journey Generator.

Generates synthetic patient journeys with embedded causal effects.
This is the core generator for causal validation.
"""

from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from scipy.special import expit

from src.ml.synthetic.dgp.adherence_outcomes import generate_adherence_outcomes
from src.ml.synthetic.dgp.initiation_outcomes import generate_initiation_outcome

from ..clinical_codes import BRAND_ELIGIBILITY_FIELDS, brand_codes
from ..config import (
    DGP_CONFIGS,
    Brand,
    DGPType,
    InsuranceTypeEnum,
    RegionEnum,
)
from ..dgp.treatment_arm import (
    _BIOLOGIC_SPAWN_KEY,
    _BRAND_CATE_SCALE,
    ARM_REGISTRY,
    assign_arm_from_spec,
    assign_segment,
    assign_treatment_arm,
    biologic_cate_modifier,
    brand_scaled_cate,
    initiation_prognostic_offset,
    insurance_access_from_type,
    rd_map_from_tau,
)
from .base import BaseGenerator, GeneratorConfig
from .cohort_outcomes import generate_discontinuation_outcomes

# Independent SeedSequence spawn_key for the copay arm, so wiring Phase 1 does
# NOT shift the generator's main self._rng stream (mirrors Phase 3's
# _BIOLOGIC_SPAWN_KEY). Every pre-existing column stays byte-identical; only the
# adherence OUTCOMES change, because copay genuinely enters their latent.
_COPAY_SPAWN_KEY = 0xC0FA
# Phase 2: an INDEPENDENT substream for the psp arm (distinct from _COPAY_SPAWN_KEY),
# so wiring psp shifts NO pre-existing column — only the adherence/persistence OUTCOMES
# move, because psp genuinely enters both latents. Distinct key => copay and psp draw
# from non-overlapping streams (no shared entropy that would correlate the arms).
_PSP_SPAWN_KEY = 0x9527
# Phase 3: INDEPENDENT substreams for the rep_detailing_high + sample_dropped arms.
# Distinct from copay/psp/biologic keys so all commercial arms draw from non-overlapping
# streams. Unlike copay/psp (assigned AFTER the initiation outcome because they feed the
# LATER adherence/persistence latents), rep/sample feed the INITIATION latent and so are
# assigned BEFORE it — but from these substreams, so the main self._rng stream (and every
# column not causally downstream of who-initiates) stays byte-identical to pre-Phase-3.
_REP_SPAWN_KEY = 0x8EE9
_SAMPLE_SPAWN_KEY = 0x5A3D


class PatientGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for patient journeys with embedded causal effects.

    This generator creates patient records with:
    - Confounders (disease_severity, academic_hcp)
    - Treatment (engagement_score)
    - Outcome (treatment_initiated)
    - TRUE CAUSAL EFFECT embedded per DGP

    The causal structure follows:
        Confounders → Treatment
        Confounders → Outcome
        Treatment → Outcome (TRUE CAUSAL EFFECT)
    """

    # Insurance distribution (US commercial market)
    # Only using enum values that exist in schema
    INSURANCE_DIST = {
        InsuranceTypeEnum.COMMERCIAL: 0.60,
        InsuranceTypeEnum.MEDICARE: 0.30,
        InsuranceTypeEnum.MEDICAID: 0.10,
    }

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "patient_journeys"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        hcp_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the patient generator.

        Args:
            config: Generator configuration.
            hcp_df: Optional HCP DataFrame for foreign key integrity.
        """
        super().__init__(config)
        self.hcp_df = hcp_df
        self._dgp_config = None
        if self.config.dgp_type:
            self._dgp_config = DGP_CONFIGS.get(self.config.dgp_type)

    def generate(self) -> pd.DataFrame:
        """
        Generate patient journeys with embedded causal effects.

        Returns:
            DataFrame with patient journeys matching schema.
        """
        n = self.config.n_records
        dgp_type = self.config.dgp_type or DGPType.CONFOUNDED
        self._log(f"Generating {n} patient journeys with DGP: {dgp_type.value}")

        # Get DGP config
        dgp_config = DGP_CONFIGS.get(dgp_type)
        true_ate = dgp_config.true_ate if dgp_config else 0.25

        # Generate confounders FIRST (they affect both treatment and outcome)
        confounders = self._generate_confounders(n, dgp_type)

        # Generate treatment (engagement) based on confounders. engagement_score is
        # retained as an EMITTED covariate (continuous), but the causal treatment for
        # arm-based estimators is now the confounded BINARY arm below (Task 03.1).
        engagement_scores = self._generate_treatment(confounders, dgp_type)

        # Confounded binary treatment ARM + estimable propensity with overlap
        # (Task 03.1). Confounders carry disease_severity + academic_hcp.
        treatment_arm, propensity = assign_treatment_arm(confounders, self._rng)

        # Per-unit segment + brand-scaled latent CATE (Task 03.2). Distinct per
        # brand so a Kisqali probe yields a different structure than Remibrutinib.
        brand_enum = self.config.brand or Brand.REMIBRUTINIB
        segment = assign_segment(confounders["disease_severity"])
        latent_cate_map = brand_scaled_cate(brand_enum)

        # T9/T11: prognostic drivers — drawn INDEPENDENTLY of treatment_arm so they
        # raise predictive AUC WITHOUT changing the true ATE/CATE. Hoisted above BOTH
        # outcome calls: T11 feeds them into the initiation eqn (treatment_initiated)
        # via initiation_prognostic_offset, and T9 feeds them into the discontinuation
        # eqn below. Reused verbatim in the record dict (single SSOT, no re-draw).
        insurance_type = self._random_choice(
            [i.value for i in InsuranceTypeEnum],
            n,
            p=[self.INSURANCE_DIST[i] for i in InsuranceTypeEnum],
        )
        age_at_diagnosis = self._random_int(18, 85, n)
        comorbidity_burden = self._rng.poisson(1.3, n).clip(0, 5)
        prior_therapy_lines = self._rng.integers(0, 4, n)

        # Prevalence-banded binary outcome carrying E[tau]=TRUE_ATE (Task 03.3).
        # REPLACES the expit(...)>0.5 outcome for treatment_initiated so the label
        # is recoverable (in-band [0.20,0.50]) rather than degenerate. tau_i is the
        # per-unit RD-scale (de-confounded, recoverable) segment CATE. T11: the 4
        # prognostic drivers enter via prognostic_offset (⊥ arm) so the goldstd
        # initiation model gains real signal while ATE/CATE recovery is preserved.
        # Phase 3 (COMM-ARMS): rep_detailing_high + sample_dropped fold into the
        # INITIATION latent, so they are assigned HERE — BEFORE the treatment_initiated
        # outcome that consumes them — from independent substreams (no main-stream shift).
        # Both confound on academic_hcp + engagement_score (drawn above).
        _rep_rng = np.random.default_rng(
            np.random.SeedSequence(self.config.seed, spawn_key=(_REP_SPAWN_KEY,))
        )
        rep_spec = ARM_REGISTRY["rep_detailing_high"]
        rep_detailing_high, rep_propensity = assign_arm_from_spec(
            rep_spec,
            {
                "academic_hcp": confounders["academic_hcp"],
                "engagement_score": np.asarray(engagement_scores, dtype=float),
            },
            _rep_rng,
        )
        _sample_rng = np.random.default_rng(
            np.random.SeedSequence(self.config.seed, spawn_key=(_SAMPLE_SPAWN_KEY,))
        )
        sample_spec = ARM_REGISTRY["sample_dropped"]
        sample_dropped, sample_propensity = assign_arm_from_spec(
            sample_spec,
            {
                "academic_hcp": confounders["academic_hcp"],
                "engagement_score": np.asarray(engagement_scores, dtype=float),
            },
            _sample_rng,
        )
        # Brand-scaled latent CATE maps for the two arms (same _BRAND_CATE_SCALE SSOT the
        # arm/copay/psp maps use) — NOT the _INIT_LATENT_CATE_BOOST (that is treatment_arm's).
        _rep_scale = _BRAND_CATE_SCALE.get(brand_enum, 1.0)
        rep_cate = {
            seg: round(val * _rep_scale, 4) for seg, val in rep_spec.cate_by_segment.items()
        }
        sample_cate = {
            seg: round(val * _rep_scale, 4) for seg, val in sample_spec.cate_by_segment.items()
        }

        prognostic_offset = initiation_prognostic_offset(
            insurance_type, age_at_diagnosis, comorbidity_burden, prior_therapy_lines
        )
        # Phase 3: treatment_arm + rep + sample fold into ONE shared initiation latent.
        # With rep/sample absent this is byte-identical to the old binary_outcome_with_cate
        # call (same coefs/boost/single noise draw), so self._rng advances identically.
        _init = generate_initiation_outcome(
            treatment_arm=treatment_arm,
            disease_severity=confounders["disease_severity"],
            academic_hcp=confounders["academic_hcp"],
            segment=segment,
            cate_map=latent_cate_map,
            rng=self._rng,
            prognostic_offset=prognostic_offset,
            rep_detailing_high=rep_detailing_high,
            rep_cate=rep_cate,
            sample_dropped=sample_dropped,
            sample_cate=sample_cate,
        )
        treatment_initiated = _init["treatment_initiated"]
        tau_i = _init["tau_i"]
        # RD-scale ground-truth CATE map (what the estimators recover) — persisted.
        cate_map = _init["arm_rd_by_segment"]
        # Per-arm RD ground truth for the recovery gate (may be REVISED for Remibrutinib by
        # the biologic-differential rebuild below, which re-draws initiation on that brand).
        rep_rd_by_segment = _init["rep_rd_by_segment"]
        sample_rd_by_segment = _init["sample_rd_by_segment"]

        # Hoist region generation here so it can be passed into the DGP (region now
        # carries real leakage-safe signal via _DISC_REGION_LOGIT) and reused in the
        # record dict below — single SSOT, no second random draw.
        geographic_region = self._random_choice([r.value for r in RegionEnum], n)

        # Phase 1 (COMM-ARMS): the copay_support commercial arm. Assigned AFTER
        # insurance_type exists (its backdoor covariate) and BEFORE both outcome
        # calls below, which now consume it — the discontinuation/persistence eqn
        # (Task 10) and the adherence latent. Drawn from an INDEPENDENT substream,
        # so its position in this function does NOT shift the main RNG stream.
        insurance_access = insurance_access_from_type(np.asarray(insurance_type))
        _copay_rng = np.random.default_rng(
            np.random.SeedSequence(self.config.seed, spawn_key=(_COPAY_SPAWN_KEY,))
        )
        copay_spec = ARM_REGISTRY["copay_support"]
        copay_support, copay_propensity = assign_arm_from_spec(
            copay_spec,
            {
                "insurance_access_score": insurance_access,
                "disease_severity": confounders["disease_severity"],
            },
            _copay_rng,
        )
        # Brand-scaled copay CATE, reusing the arm-scale SSOT so a Kisqali probe
        # differs from a Remibrutinib one (same pattern as brand_scaled_cate).
        _copay_scale = _BRAND_CATE_SCALE.get(brand_enum, 1.0)
        copay_cate = {
            seg: round(val * _copay_scale, 4) for seg, val in copay_spec.cate_by_segment.items()
        }

        # Phase 2 (COMM-ARMS): the psp_enrolled commercial arm. Confounded on
        # disease_severity + engagement_score + academic_hcp (all already-allowlisted
        # covariates, all drawn ABOVE this point), from its OWN independent substream so
        # its position here shifts no pre-existing column. Consumed by both outcome
        # calls below (persistence logit + adherence latent), exactly like copay.
        _psp_rng = np.random.default_rng(
            np.random.SeedSequence(self.config.seed, spawn_key=(_PSP_SPAWN_KEY,))
        )
        psp_spec = ARM_REGISTRY["psp_enrolled"]
        psp_enrolled, psp_propensity = assign_arm_from_spec(
            psp_spec,
            {
                "disease_severity": confounders["disease_severity"],
                "engagement_score": np.asarray(engagement_scores, dtype=float),
                "academic_hcp": confounders["academic_hcp"],
            },
            _psp_rng,
        )
        _psp_scale = _BRAND_CATE_SCALE.get(brand_enum, 1.0)
        psp_cate = {
            seg: round(val * _psp_scale, 4) for seg, val in psp_spec.cate_by_segment.items()
        }

        # Shard 06: disc/persist cohort outcomes from the Shard-03 CANONICAL arm +
        # segment (single SSOT — no second arm/segment source). brand_cate_scale reuses
        # Shard 03's _BRAND_CATE_SCALE so a Kisqali probe differs from a Remibrutinib one.
        _coh = generate_discontinuation_outcomes(
            rng=self._rng,
            treatment_arm=np.asarray(treatment_arm, dtype=int),
            disease_severity=confounders["disease_severity"],
            academic_hcp=confounders["academic_hcp"],
            geographic_region=np.asarray(geographic_region),
            insurance_type=np.asarray(insurance_type),
            age_at_diagnosis=np.asarray(age_at_diagnosis),
            comorbidity_burden=np.asarray(comorbidity_burden),
            prior_therapy_lines=np.asarray(prior_therapy_lines),
            segment=np.asarray(segment),
            brand_cate_scale=_BRAND_CATE_SCALE.get(brand_enum, 1.0),
            copay_support=copay_support,
            psp_enrolled=psp_enrolled,
        )

        # Phase 0 (commercial-arms enrichment): binarized adherence outcomes of the
        # EXISTING treatment_arm, on the SAME segment/CATE map (single SSOT). The
        # binary is authoritative + recoverable; adherence_rate/gap_days are proxies.
        # Phase 1: copay_support now also enters the shared adherence latent.
        _adh = generate_adherence_outcomes(
            treatment_arm=np.asarray(treatment_arm, dtype=int),
            disease_severity=confounders["disease_severity"],
            academic_hcp=confounders["academic_hcp"],
            segment=np.asarray(segment),
            cate_map=latent_cate_map,
            rng=self._rng,
            copay_support=copay_support,
            copay_cate=copay_cate,
            psp_enrolled=psp_enrolled,
            psp_cate=psp_cate,
        )

        # days_to_treatment only for initiators (preserve prior shape)
        days_to_treatment: Any = np.where(
            treatment_initiated == 1,
            self._random_int(7, 90, n),
            np.nan,
        )

        # Generate dates and assign splits
        journey_dates = self._random_dates(n)
        data_splits = self._assign_splits(journey_dates)

        # Generate HCP assignments
        hcp_ids = self._assign_hcps(n, confounders)

        # Generate patient IDs
        patient_ids = self._generate_ids("pt", n, width=6)
        journey_ids = self._generate_ids("patient", n, width=6)

        # Determine brand
        if self.config.brand:
            brands = [self.config.brand.value] * n
        else:
            brands = self._random_choice([b.value for b in Brand], n).tolist()

        # Brand-specific eligibility (Shard 04 M5). Generated to pass the
        # cohort_constructor inclusion gates (configs.py required_fields) so each
        # brand's cohort is populated (non-empty); the OUTCOME prevalence within the
        # cohort is the DGP's banded treatment_initiated (Shard 03).
        #
        # Phase 2 brand-gating: every field is DRAWN for every row (so the RNG
        # stream is byte-for-byte identical to the pre-gating generator — this is
        # the LAST RNG consumer in generate(), so draw-then-discard preserves the
        # whole causal substrate + every other column), but each value is then kept
        # ONLY for the brand whose indication it belongs to (BRAND_ELIGIBILITY_FIELDS)
        # and NULLed otherwise. So a Kisqali row no longer carries a fabricated CSU
        # UAS7, a Remibrutinib row no longer carries a fabricated renal eGFR, etc.
        # cohort_constructor already reads only the brand-relevant subset;
        # segments/causal select the brand-relevant effect-modifiers so a now-NULL
        # off-brand column never reaches EconML as NaN. primary_diagnosis_code stays
        # the row's own brand-correct diagnosis (never gated).
        primary_dx: List[str] = []
        uas7: List[Optional[int]] = []
        prior_ah: List[Optional[bool]] = []
        hr_status: List[Optional[str]] = []
        her2_status: List[Optional[str]] = []
        stage: List[Optional[str]] = []
        ecog: List[Optional[int]] = []
        ldh: List[Optional[float]] = []
        comp_inh: List[Optional[str]] = []
        proteinuria: List[Optional[float]] = []
        egfr: List[Optional[float]] = []
        _stage_pool = ["advanced", "metastatic", "locally_advanced", "stage_iv"]
        for b in brands:
            codes = (
                brand_codes(b)
                if b in ("Remibrutinib", "Kisqali", "Fabhalta")
                else {"icd10": ["L50.9"]}
            )
            fields = BRAND_ELIGIBILITY_FIELDS.get(b, frozenset())
            # DRAW every field unconditionally (identical order/consumption to the
            # pre-gating generator) …
            dx_v = str(self._rng.choice(cast("list[str]", codes["icd10"])))
            uas7_v = int(self._rng.integers(16, 43))  # Remi: UAS7 16-42 (inclusion >=16)
            stage_v = str(self._rng.choice(_stage_pool))
            ecog_v = int(self._rng.integers(0, 2))
            ldh_v = round(float(self._rng.uniform(1.5, 5.0)), 2)
            comp_v = str(self._rng.choice(["current", "prior"]))
            prot_v = round(float(self._rng.uniform(1.0, 6.0)), 2)
            egfr_v = round(float(self._rng.uniform(30.0, 110.0)), 2)
            # … then KEEP only the brand-relevant columns (draw-then-discard).
            primary_dx.append(dx_v)
            uas7.append(uas7_v if "urticaria_severity_uas7" in fields else None)
            prior_ah.append(True if "prior_antihistamine_therapy" in fields else None)
            hr_status.append("positive" if "hr_status" in fields else None)
            her2_status.append("negative" if "her2_status" in fields else None)
            stage.append(stage_v if "disease_stage" in fields else None)
            ecog.append(ecog_v if "ecog_performance_status" in fields else None)
            ldh.append(ldh_v if "ldh_ratio" in fields else None)
            comp_inh.append(comp_v if "complement_inhibitor_status" in fields else None)
            proteinuria.append(prot_v if "proteinuria_g_day" in fields else None)
            egfr.append(egfr_v if "egfr" in fields else None)

        # Phase 2: biologic-experience + baseline serum IgE — the real anti-IgE
        # clinical axis for CSU/Remibrutinib (the axis the chatbot used to invent).
        # Drawn in a SEPARATE loop AFTER the eligibility loop (the previously-last
        # RNG consumer) so the existing per-row stream is untouched; drawn for every
        # row so the stream itself is brand-independent, then kept for Remibrutinib
        # only (correlational Phase 2 columns — differential causal effects are
        # Phase 3). ~40% biologic-experienced; IgE lognormal (median ~150 IU/mL).
        biologic_experienced: List[Optional[int]] = []
        ige_level: List[Optional[float]] = []
        for b in brands:
            bio_v = int(self._rng.random() < 0.40)
            ige_v = round(float(np.clip(self._rng.lognormal(mean=5.0, sigma=0.8), 2.0, 2000.0)), 1)
            if "biologic_experienced" in BRAND_ELIGIBILITY_FIELDS.get(b, frozenset()):
                biologic_experienced.append(bio_v)
                ige_level.append(ige_v)
            else:
                biologic_experienced.append(None)
                ige_level.append(None)

        # Build DataFrame
        df = pd.DataFrame(
            {
                "patient_journey_id": journey_ids,
                "patient_id": patient_ids,
                "hcp_id": hcp_ids,
                "brand": brands,
                "journey_start_date": journey_dates,
                "data_split": data_splits,
                "disease_severity": confounders["disease_severity"],
                "academic_hcp": confounders["academic_hcp"],
                "engagement_score": engagement_scores,
                "treatment_initiated": treatment_initiated,
                "days_to_treatment": days_to_treatment,
                "geographic_region": geographic_region,
                "insurance_type": insurance_type,
                "age_at_diagnosis": age_at_diagnosis,
                # Brand eligibility columns (Shard 04 M5). primary_diagnosis_code is
                # an existing column; the other 10 are added by migration 068.
                "primary_diagnosis_code": primary_dx,
                "urticaria_severity_uas7": uas7,
                "prior_antihistamine_therapy": prior_ah,
                "hr_status": hr_status,
                "her2_status": her2_status,
                "disease_stage": stage,
                "ecog_performance_status": ecog,
                "ldh_ratio": ldh,
                "complement_inhibitor_status": comp_inh,
                "proteinuria_g_day": proteinuria,
                "egfr": egfr,
                # Phase 2 anti-IgE axis (migration 107) — Remibrutinib/CSU only,
                # NULL for the oncology/PNH brands (gated above).
                "biologic_experienced": biologic_experienced,
                "ige_level": ige_level,
                # Causal substrate columns (Shard 01 M2 DDL). Populated by Shard
                # 03's DGP: treatment_arm/propensity_score/segment_assignment are
                # the confounded arm + estimable propensity + per-unit CATE segment;
                # treatment_effect_estimate is the per-unit brand-scaled tau (the
                # recoverable ground truth). discontinued_180d/persistent_180d stay
                # NULL here — Shard 06's outcome builders own those.
                "treatment_arm": treatment_arm,
                "propensity_score": np.round(propensity, 4),
                "segment_assignment": segment,
                "treatment_effect_estimate": np.round(tau_i, 4),
                "discontinued_180d": _coh["discontinued_180d"],
                "persistent_180d": _coh["persistent_180d"],
                # Phase 0 adherence outcomes + raw proxies (migration 088).
                "adherent_180d": _adh["adherent_180d"],
                "low_gap_180d": _adh["low_gap_180d"],
                "adherence_rate": _adh["adherence_rate"],
                "gap_days": _adh["gap_days"],
                # Phases 2-3 commercial arms — NULL placeholders so the loader
                # carries them; populated by their phase's generator wiring.
                # Phase 1/2 (COMM-ARMS): copay_support + psp_enrolled are POPULATED.
                "copay_support": copay_support,
                "psp_enrolled": psp_enrolled,
                # Phase 1/2/3 (COMM-ARMS): all four commercial arms are POPULATED.
                "rep_detailing_high": rep_detailing_high,
                "sample_dropped": sample_dropped,
                "copay_support_propensity": np.round(copay_propensity, 4),
                "psp_enrolled_propensity": np.round(psp_propensity, 4),
                "rep_detailing_high_propensity": np.round(rep_propensity, 4),
                "sample_dropped_propensity": np.round(sample_propensity, 4),
                "insurance_access_score": np.round(insurance_access, 4),
                "comorbidity_burden": comorbidity_burden,
                "prior_therapy_lines": prior_therapy_lines,
            }
        )

        # Phase 3 (CLIN-SEG-P3): plant the biologic-experience differential CATE on
        # the Remibrutinib initiation outcome. Post-hoc + an independent substream so
        # the main self._rng stream — and every non-Remibrutinib row — stays
        # byte-identical (only Remibrutinib treatment_initiated /
        # treatment_effect_estimate / days_to_treatment are rewritten). No-op (returns
        # tau_i unchanged, cate_by_biologic=None) when the frame carries no
        # Remibrutinib rows with a populated biologic_experienced.
        tau_i, cate_by_biologic, rep_rd_by_segment, sample_rd_by_segment = (
            self._apply_biologic_differential(
                df,
                np.asarray(segment),
                confounders,
                prognostic_offset,
                latent_cate_map,
                tau_i,
                rep_detailing_high=rep_detailing_high,
                rep_cate=rep_cate,
                sample_dropped=sample_dropped,
                sample_cate=sample_cate,
                rep_rd_by_segment=rep_rd_by_segment,
                sample_rd_by_segment=sample_rd_by_segment,
            )
        )
        cate_map = rd_map_from_tau(np.asarray(segment), tau_i)

        # Store ground truth metadata (realized values from the wired DGP)
        df.attrs["true_ate"] = float(np.mean(tau_i))
        df.attrs["cate_by_segment"] = cate_map
        df.attrs["brand"] = brand_enum.value
        df.attrs["dgp_type"] = dgp_type.value
        df.attrs["prevalence"] = float(np.mean(df["treatment_initiated"].to_numpy()))
        df.attrs["confounders"] = dgp_config.confounders if dgp_config else []
        if cate_by_biologic is not None:
            df.attrs["cate_by_biologic"] = cate_by_biologic

        # Per-arm/outcome recoverable ground truth (commercial-arms enrichment).
        # Existing arm: treatment_initiated (scalar above) + the Phase 0 adherence
        # outcomes. Later phases extend this dict with their arm keys.
        df.attrs["true_ate_by_arm"] = {
            "treatment_arm": {
                "treatment_initiated": {
                    "ate": float(np.mean(tau_i)),
                    "cate_by_segment": cate_map,
                    **({"cate_by_biologic": cate_by_biologic} if cate_by_biologic else {}),
                },
                "adherent_180d": {
                    "ate": float(
                        np.mean([_adh["adherent_rd_by_segment"][str(s)] for s in segment])
                    ),
                    "cate_by_segment": _adh["adherent_rd_by_segment"],
                },
                "low_gap_180d": {
                    "ate": float(np.mean([_adh["low_gap_rd_by_segment"][str(s)] for s in segment])),
                    "cate_by_segment": _adh["low_gap_rd_by_segment"],
                },
            }
        }
        df.attrs["true_ate_by_arm"]["copay_support"] = {
            "adherent_180d": {
                "ate": float(np.mean(list(_adh["copay_adherent_rd_by_segment"].values()))),
                "cate_by_segment": _adh["copay_adherent_rd_by_segment"],
            },
            "low_gap_180d": {
                "ate": float(np.mean(list(_adh["copay_low_gap_rd_by_segment"].values()))),
                "cate_by_segment": _adh["copay_low_gap_rd_by_segment"],
            },
            "persistent_180d": {
                "ate": float(
                    np.mean([_coh["copay_persistent_rd_by_segment"][str(s)] for s in segment])
                ),
                "cate_by_segment": _coh["copay_persistent_rd_by_segment"],
            },
        }
        df.attrs["true_ate_by_arm"]["psp_enrolled"] = {
            "adherent_180d": {
                "ate": float(np.mean(list(_adh["psp_adherent_rd_by_segment"].values()))),
                "cate_by_segment": _adh["psp_adherent_rd_by_segment"],
            },
            "persistent_180d": {
                "ate": float(
                    np.mean([_coh["psp_persistent_rd_by_segment"][str(s)] for s in segment])
                ),
                "cate_by_segment": _coh["psp_persistent_rd_by_segment"],
            },
        }
        # Phase 3 (COMM-ARMS): rep_detailing_high + sample_dropped target ONLY
        # treatment_initiated. Their per-segment RD is the fold-step ground truth
        # (RE-DERIVED against the biologic-rebuilt score for Remibrutinib frames).
        # ate = population-mean of the per-segment RD map (segment-marginal). Both maps
        # are always populated here — generate() unconditionally supplies both arms to the
        # initiation folder — so narrow the folder's Optional return for the indexing below.
        assert rep_rd_by_segment is not None and sample_rd_by_segment is not None
        df.attrs["true_ate_by_arm"]["rep_detailing_high"] = {
            "treatment_initiated": {
                "ate": float(np.mean([rep_rd_by_segment[str(s)] for s in segment])),
                "cate_by_segment": rep_rd_by_segment,
            },
        }
        df.attrs["true_ate_by_arm"]["sample_dropped"] = {
            "treatment_initiated": {
                "ate": float(np.mean([sample_rd_by_segment[str(s)] for s in segment])),
                "cate_by_segment": sample_rd_by_segment,
            },
        }

        self._log(f"Generated {len(df)} patient journeys (TRUE_ATE={true_ate})")
        return df

    def _apply_biologic_differential(
        self,
        df: pd.DataFrame,
        segment: np.ndarray,
        confounders: Dict[str, np.ndarray],
        prognostic_offset: np.ndarray,
        latent_cate_map: Dict[str, float],
        tau_i: np.ndarray,
        *,
        rep_detailing_high: np.ndarray,
        rep_cate: Dict[str, float],
        sample_dropped: np.ndarray,
        sample_cate: Dict[str, float],
        rep_rd_by_segment: Optional[Dict[str, float]],
        sample_rd_by_segment: Optional[Dict[str, float]],
    ) -> tuple[
        np.ndarray,
        Optional[Dict[str, float]],
        Optional[Dict[str, float]],
        Optional[Dict[str, float]],
    ]:
        """Phase 3 (CLIN-SEG-P3): plant the biologic-experience differential CATE on
        the Remibrutinib initiation outcome.

        Rebuilds ``treatment_initiated`` + ``treatment_effect_estimate`` for the
        Remibrutinib rows (where ``biologic_experienced`` is populated) with a
        per-unit CATE modifier (biologic-experienced ~0.625x the naive effect — a
        mean-preserving 2x spread, see ``treatment_arm._BIOLOGIC_*``). Runs POST-HOC
        on the already-built frame, drawing its fresh noise from an INDEPENDENT
        deterministic substream, so:
          * the generator's main ``self._rng`` stream is never perturbed, and
          * every non-Remibrutinib row (biologic NULL) is byte-identical to pre-P3.

        COMM-ARMS Phase 3: the rebuild goes through the SAME multi-arm initiation folder
        (``generate_initiation_outcome``) as the main call, folding rep_detailing_high +
        sample_dropped into the Remibrutinib initiation latent — otherwise this rebuild,
        which overwrites the Remibrutinib outcome, would CLOBBER their planted effect.
        rep/sample per-segment RD is RE-DERIVED against the rebuilt (biologic-noise) score
        so the recovery gate compares the estimate against the ground truth of the SAME
        outcome the estimate is fit on. A Remibrutinib frame is single-brand (mask covers
        all biologic-populated rows), so the subset RD IS the frame RD.

        Returns (frame-wide per-unit tau with Remibrutinib rows updated, {naive,experienced}
        biologic RD map, rep RD map, sample RD map). When no Remibrutinib rows carry
        biologic-experience it is a no-op that passes the inputs straight back.
        """
        biologic = df["biologic_experienced"].to_numpy()
        mask = (df["brand"].to_numpy() == Brand.REMIBRUTINIB.value) & ~pd.isna(biologic)
        if not mask.any():
            return tau_i, None, rep_rd_by_segment, sample_rd_by_segment
        idx = np.where(mask)[0]
        bio = biologic[idx].astype(float)  # 0=naive, 1=experienced
        modifier = biologic_cate_modifier(bio)

        # Independent substream keyed off the frame seed — reproducible AND orthogonal
        # to the main stream (spawn_key guarantees non-overlap with self._rng).
        bio_rng = np.random.default_rng(
            np.random.SeedSequence(self.config.seed, spawn_key=(_BIOLOGIC_SPAWN_KEY,))
        )
        _bio = generate_initiation_outcome(
            treatment_arm=df["treatment_arm"].to_numpy()[idx].astype(int),
            disease_severity=confounders["disease_severity"][idx],
            academic_hcp=confounders["academic_hcp"][idx],
            segment=segment[idx],
            cate_map=latent_cate_map,
            rng=bio_rng,
            prognostic_offset=np.asarray(prognostic_offset)[idx],
            rep_detailing_high=np.asarray(rep_detailing_high)[idx],
            rep_cate=rep_cate,
            sample_dropped=np.asarray(sample_dropped)[idx],
            sample_cate=sample_cate,
            cate_modifier=modifier,
        )
        y_new = _bio["treatment_initiated"]
        tau_new = _bio["tau_i"]

        # Reconcile days_to_treatment ONLY for rows whose initiation flipped, so
        # unchanged initiators keep their original day and non-initiators stay NaN.
        old_init = df["treatment_initiated"].to_numpy()[idx].astype(int)
        days = df["days_to_treatment"].to_numpy(dtype=float).copy()
        up = (old_init == 0) & (y_new == 1)
        down = (old_init == 1) & (y_new == 0)
        d_sub = days[idx]
        d_sub[down] = np.nan
        if up.any():
            d_sub[up] = bio_rng.integers(7, 91, int(up.sum())).astype(float)
        days[idx] = d_sub

        # Write the rebuilt columns back (Remibrutinib rows only).
        ti = df["treatment_initiated"].to_numpy().copy()
        ti[idx] = y_new
        df["treatment_initiated"] = ti
        tee = df["treatment_effect_estimate"].to_numpy(dtype=float).copy()
        tee[idx] = np.round(tau_new, 4)
        df["treatment_effect_estimate"] = tee
        df["days_to_treatment"] = days

        final_tau = np.asarray(tau_i, dtype=float).copy()
        final_tau[idx] = tau_new
        cate_by_biologic = {
            "naive": float(np.mean(tau_new[bio < 0.5])),
            "experienced": float(np.mean(tau_new[bio >= 0.5])),
        }
        # rep/sample RD re-derived against the biologic-rebuilt score (fall back to the
        # fold-step map only if the arm wasn't supplied to the folder). The biologic map
        # is derived from the biologic SUBSET (rows with populated biologic_experienced);
        # for a real Remibrutinib frame that is ALL rows (100% populated), so it covers
        # every segment. But a partial-biologic frame (e.g. small-n / defaulted-brand test
        # frames) can leave a segment out of the subset — so MERGE the subset map OVER the
        # main-call (full-coverage) map: biologic values win where present, main-call values
        # backfill any uncovered segment, guaranteeing the true_ate_by_arm segment-mean can
        # never KeyError on a segment the full frame contains.
        rep_rd_out = (
            {**(rep_rd_by_segment or {}), **_bio["rep_rd_by_segment"]}
            if _bio["rep_rd_by_segment"] is not None
            else rep_rd_by_segment
        )
        sample_rd_out = (
            {**(sample_rd_by_segment or {}), **_bio["sample_rd_by_segment"]}
            if _bio["sample_rd_by_segment"] is not None
            else sample_rd_by_segment
        )
        return final_tau, cate_by_biologic, rep_rd_out, sample_rd_out

    def _generate_confounders(
        self,
        n: int,
        dgp_type: DGPType,
    ) -> Dict[str, np.ndarray]:
        """
        Generate confounding variables.

        These affect both treatment and outcome.
        """
        # Disease severity: 0-10 scale, normally distributed
        disease_severity = self._random_normal(5.0, 2.0, n, clip_min=0, clip_max=10)

        # Academic HCP: binary, ~30% are academic
        academic_hcp = (self._rng.random(n) < 0.30).astype(int)

        # For heterogeneous DGP, add segment-specific variation
        if dgp_type == DGPType.HETEROGENEOUS:
            # Create segments based on disease severity
            # High: severity > 7, Medium: 4-7, Low: < 4
            pass  # Segment effects handled in outcome generation

        return {
            "disease_severity": disease_severity,
            "academic_hcp": academic_hcp,
        }

    def _generate_treatment(
        self,
        confounders: Dict[str, np.ndarray],
        dgp_type: DGPType,
    ) -> np.ndarray:
        """
        Generate treatment (engagement_score) with confounding.

        Treatment propensity is influenced by confounders:
        - Higher disease severity → more engagement
        - Academic HCP → more engagement
        """
        n = len(confounders["disease_severity"])

        if dgp_type == DGPType.SIMPLE_LINEAR:
            # No confounding - pure random treatment
            engagement = self._random_float(0, 10, n)
        elif dgp_type == DGPType.SELECTION_BIAS:
            # Strong selection bias based on disease severity
            propensity = (
                2.0
                + 0.8 * confounders["disease_severity"]  # Strong severity effect
                + self._rng.normal(0, 0.5, n)
            )
            engagement = expit(propensity / 3) * 10
        else:
            # Standard confounding structure
            propensity = (
                3.0
                + 0.3 * confounders["disease_severity"]
                + 2.0 * confounders["academic_hcp"]
                + self._rng.normal(0, 1, n)
            )
            engagement = expit(propensity / 3) * 10

        return np.asarray(np.clip(engagement, 0, 10))

    def _generate_outcome(
        self,
        treatment: np.ndarray,
        confounders: Dict[str, np.ndarray],
        true_ate: float,
        dgp_type: DGPType,
    ) -> Dict[str, np.ndarray]:
        """
        Generate outcome with TRUE causal effect.

        Outcome = f(confounders) + TRUE_ATE * treatment + noise

        This is the key function for causal validation.
        """
        n = len(treatment)

        if dgp_type == DGPType.SIMPLE_LINEAR:
            # Simple linear: Y = TRUE_ATE * T + noise
            outcome_propensity = -2.0 + true_ate * treatment + self._rng.normal(0, 1, n)
        elif dgp_type == DGPType.HETEROGENEOUS:
            # Heterogeneous treatment effects by segment
            # Segment assignment based on disease severity
            segments = np.where(
                confounders["disease_severity"] > 7,
                "high",
                np.where(confounders["disease_severity"] > 4, "medium", "low"),
            )

            # CATE by segment
            cate = np.where(
                segments == "high",
                0.50,  # High severity: strong effect
                np.where(segments == "medium", 0.30, 0.15),  # Medium: moderate, Low: weak
            )

            outcome_propensity = (
                -2.0
                + cate * treatment  # Heterogeneous effect
                + 0.4 * confounders["disease_severity"]
                + 0.6 * confounders["academic_hcp"]
                + self._rng.normal(0, 1, n)
            )
        elif dgp_type == DGPType.TIME_SERIES:
            # Time series: effect with lag
            # Simulated by adding temporal decay
            lag_effect = 0.85 ** np.arange(n)  # Decay over time
            effective_treatment = treatment * (0.5 + 0.5 * lag_effect)

            outcome_propensity = (
                -2.0
                + true_ate * effective_treatment
                + 0.4 * confounders["disease_severity"]
                + 0.6 * confounders["academic_hcp"]
                + self._rng.normal(0, 1, n)
            )
        elif dgp_type == DGPType.SELECTION_BIAS:
            # Selection bias: outcome affected by selection mechanism
            # Higher baseline for high-severity patients
            selection_baseline = 0.3 * confounders["disease_severity"]

            outcome_propensity = (
                -2.0
                + selection_baseline
                + true_ate * treatment
                + 0.2 * confounders["disease_severity"]  # Residual confounding
                + 0.6 * confounders["academic_hcp"]
                + self._rng.normal(0, 1, n)
            )
        else:
            # Default: Confounded DGP
            outcome_propensity = (
                -2.0
                + true_ate * treatment  # TRUE CAUSAL EFFECT
                + 0.4 * confounders["disease_severity"]  # Confounding
                + 0.6 * confounders["academic_hcp"]  # Confounding
                + self._rng.normal(0, 1, n)
            )

        # Convert to binary outcome
        treatment_initiated = (expit(outcome_propensity) > 0.5).astype(int)

        # Generate days to treatment (only for those who initiated)
        days_to_treatment: Any = np.where(
            treatment_initiated == 1,
            self._random_int(7, 90, n),
            np.nan,  # Use np.nan instead of None for numpy compatibility
        )

        return {
            "treatment_initiated": treatment_initiated,
            "days_to_treatment": days_to_treatment,
        }

    def _assign_hcps(
        self,
        n: int,
        confounders: Dict[str, np.ndarray],
    ) -> List[str]:
        """
        Assign HCPs to patients.

        If HCP DataFrame provided, maintains referential integrity.
        Otherwise generates placeholder IDs.
        """
        if self.hcp_df is not None and len(self.hcp_df) > 0:
            # Match academic patients to academic HCPs when possible
            academic_hcps = self.hcp_df[self.hcp_df["academic_hcp"] == 1]["hcp_id"].values
            non_academic_hcps = self.hcp_df[self.hcp_df["academic_hcp"] == 0]["hcp_id"].values

            hcp_ids = []
            for is_academic in confounders["academic_hcp"]:
                if is_academic == 1 and len(academic_hcps) > 0:
                    hcp_ids.append(self._rng.choice(academic_hcps))
                elif len(non_academic_hcps) > 0:
                    hcp_ids.append(self._rng.choice(non_academic_hcps))
                else:
                    hcp_ids.append(self._rng.choice(self.hcp_df["hcp_id"].values))

            return hcp_ids
        else:
            # Generate placeholder HCP IDs
            n_hcps = max(100, n // 10)  # ~10 patients per HCP
            hcp_ids = self._generate_ids("hcp", n_hcps)
            return cast(List[str], self._random_choice(hcp_ids, n).tolist())
