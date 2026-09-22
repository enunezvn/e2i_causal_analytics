"""Gold-standard dataset registry for the causal routes package.

Which columns each dataset offers as treatment / outcome / covariate, how they
are coerced and derived, which are categorical, and which brands a cohort holds.

Import rule: may import ``_common`` and non-package modules only; never
``loaders`` or the package root.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from src.data.manifests import MART_SAFE_FEATURES

# Re-exported under the historical names (segments.py + six test modules import
# them from this route) — see the pointer comment where the dicts used to live.
from src.insights.column_labels import (  # noqa: F401 — re-export
    COLUMN_DEFINITIONS as _COLUMN_DEFINITIONS,
)
from src.insights.column_labels import (  # noqa: F401 — re-export
    COLUMN_LABELS as _COLUMN_LABELS,
)
from src.insights.column_labels import (  # noqa: F401 — re-export
    column_label as _column_label,
)
from src.repositories.provenance import apply_provenance_filter

logger = logging.getLogger(__name__)

# patient_journeys is the gold-standard causal frame: a fully-populated,
# patient-level cohort (treatment_arm -> persistent_180d, controlling for
# disease_severity / engagement_score / age_at_diagnosis) — the same cohort the
# gold-standard models use, with a known TRUE_ATE. (business_metrics is sparse:
# its causal columns are mostly NULL, so it is not offered here.)
#
# Covariate candidates are NUMERIC confounders the executors consume directly.
# Categorical confounders (geographic_region, brand) are still excluded — they
# would need server-side encoding the DoWhy/EconML executors don't do here (brand
# is offered instead as a cohort FILTER via the brand dropdown, not a covariate).
# The covariate list was expanded (#1027) with the additional numeric clinical
# markers that are 100%-populated WITH variance in the gold-standard cohort
# (verified against the live table: academic_hcp, egfr, proteinuria_g_day,
# ldh_ratio, urticaria_severity_uas7, ecog_performance_status) so the analyst has
# a richer adjustment set. Columns that LOOK like confounders but are 100% NULL
# (risk_score, refill_count) are deliberately NOT offered — they would
# fail-close every run.
#
# Phase 0 of the commercial-arms enrichment POPULATES adherence_rate and gap_days
# (raw continuous proxies) and the binary outcomes adherent_180d / low_gap_180d.
# The binary outcomes ARE offered below (outcome list). However, adherence_rate
# and gap_days are POST-TREATMENT DESCENDANTS of treatment_arm — they are
# generated from a latent that includes arm * tau — and are near-deterministic
# proxies of the outcomes (adherent_180d = 1{adherence_rate>=0.8}). Adjusting on
# them as covariates would OVERCONTROL: they block the very causal path being
# estimated, collapsing the treatment coefficient toward zero (measured: +0.228
# with clinical confounders → +0.022 under default route adjustment set including
# proxies — a fake "no effect"). They are therefore deliberately NOT offered as
# covariate/adjustment candidates. They remain populated DB columns and
# feature-store inputs. (Caught in adversarial review, 2026-06-29.)
# treatment/outcome stay the curated causal columns (the synthetic gold-standard
# only wires those relationships).
_CAUSAL_DATASET_SPECS: Dict[str, Dict[str, List[str]]] = {
    "patient_journeys": {
        "treatment": [
            "treatment_arm",
            "treatment_initiated",
            "copay_support",
            "psp_enrolled",
            # COMM-ARMS Phase 3: rep_detailing_high + sample_dropped, two arms that fold
            # into the treatment_initiated (initiation) latent. Their backdoor
            # {academic_hcp, engagement_score} is already in the covariate list below, so
            # the confounder-contract guard is satisfied without a new covariate.
            "rep_detailing_high",
            "sample_dropped",
            # COMM-ARMS Phase 4: trigger_accepted (NBA trigger acceptance), the fourth
            # arm in the initiation latent. Backdoor {disease_severity, engagement_score}
            # is already in the covariate list below.
            "trigger_accepted",
            # #1321 Fabhalta pilot: prior C5-inhibitor exposure (the FIRST
            # brand-DISTINCT treatment). A text column ("current"/"prior") derived
            # to 1.0 for "prior" (the switch population) by _derive_is_prior_c5
            # below. Its causal_paths edge is emitted for Fabhalta ONLY, so
            # _discover_candidate_questions surfaces the leaderboard question only
            # on the Fabhalta cohort (the row's brand="Fabhalta" scopes the data
            # load via q_brand); Kisqali/Remibrutinib carry no such edge and the
            # column is 100% NULL for them. Also a Fabhalta effect-MODIFIER (in the
            # covariate list + _BRAND_CLINICAL_COVARIATES below — dual-role, like
            # treatment_initiated is treatment+outcome).
            "complement_inhibitor_status",
            # #1321 rollout: two more brand-DISTINCT axes, same recipe as
            # complement_inhibitor_status. Each is an OBSERVATIONAL treatment (NOT
            # randomized — so it keeps the unmeasured-confounding gate) whose
            # causal_paths edge is emitted for its brand ONLY, so the leaderboard
            # question surfaces only on that brand's cohort:
            #   * Kisqali — advanced-line CDK4/6 burden, derived to 1.0 for
            #     disease_stage in {metastatic, stage_iv} by _derive_is_advanced_line.
            #   * Remibrutinib — uncontrolled CSU, derived to 1.0 for
            #     urticaria_severity_uas7 >= 28 by _derive_is_uncontrolled_csu.
            # Both are ALSO brand-scoped effect-MODIFIERS (covariate list +
            # _BRAND_CLINICAL_COVARIATES below — dual-role).
            "disease_stage",
            "urticaria_severity_uas7",
        ],
        "outcome": [
            "persistent_180d",
            "discontinued_180d",
            "treatment_initiated",
            "adherent_180d",
            "low_gap_180d",
        ],
        "covariate": [
            "disease_severity",
            "engagement_score",
            "age_at_diagnosis",
            "academic_hcp",
            "geographic_region",
            "egfr",
            "proteinuria_g_day",
            "ldh_ratio",
            "urticaria_severity_uas7",
            "ecog_performance_status",
            # Phase 3 (CLIN-SEG-P3): biologic-experience is a Remibrutinib-only
            # pre-treatment EFFECT MODIFIER (the DGP plants a differential CATE on
            # it — biologic-experienced patients respond less). Brand-scoped below
            # so it is offered only when Remibrutinib is the row filter (NULL for the
            # other brands). ige_level is NOT here — it stays a descriptive KPI axis
            # (Remibrutinib is a BTK inhibitor, not anti-IgE; no IgE causal effect).
            "biologic_experienced",
            # #1321 Fabhalta pilot: prior C5-inhibitor exposure is ALSO a
            # Fabhalta-only pre-treatment EFFECT MODIFIER (the DGP plants a
            # differential CATE — prior-C5-experienced patients respond less to
            # iptacopan). Brand-scoped below so it is offered only when Fabhalta is
            # the row filter (NULL for the other brands). Derived to 1.0/0.0 by
            # _derive_is_prior_c5, exactly like biologic_experienced's 0/1 flag.
            "complement_inhibitor_status",
            # #1321 rollout: Kisqali advanced-line burden is ALSO a Kisqali-only
            # pre-treatment effect MODIFIER (the DGP plants a differential CATE —
            # advanced-line patients respond less). Brand-scoped below so it is offered
            # only when Kisqali is the row filter (NULL for the other brands). Derived to
            # 1.0/0.0 by _derive_is_advanced_line. urticaria_severity_uas7 is ALREADY
            # above (Remibrutinib's continuous covariate); its axis derivation
            # (_derive_is_uncontrolled_csu) dichotomizes it in the causal loaders only —
            # segments.py / KPIs read the raw continuous column via their own loaders.
            "disease_stage",
            # Phase 1 (COMM-ARMS): the NUMERIC access gradient derived from
            # insurance_type. This is copay_support's backdoor — it MUST stay
            # allowlisted or a copay estimate reports the confounded naive
            # diff-in-means (locked by test_arm_confounder_contract.py). The raw
            # categorical insurance_type stays a cohort FILTER, never a covariate.
            "insurance_access_score",
            # adherence_rate and gap_days are NOT listed here: they are
            # post-treatment descendants of treatment_arm (near-deterministic
            # proxies of adherent_180d / low_gap_180d). Adjusting on them
            # overcontrols and blocks the causal path. They remain DB columns
            # and feature-store inputs but are excluded from the adjustment set.
        ],
    },
    # HCP grain: hcp_brand_adoption (treatment_arm, adopted, brand) JOIN
    # hcp_profiles (peer_influence_score, influence_network_size) on hcp_id. The
    # JOIN loader derives centrality_z = zscore(log1p(influence_network_size)) as
    # the modeled backdoor for the rep-engagement arm. peer_influence_score is the
    # EXOGENOUS-centrality treatment (empty backdoor); treatment_arm is the rep
    # engagement arm (adjust centrality_z). adopted is the binary outcome.
    "hcp_adoption": {
        "treatment": ["peer_influence_score", "treatment_arm"],
        "outcome": ["adopted"],
        "covariate": ["centrality_z"],
    },
    "nba_triggers": {
        # The triggers table — the ONLY true RCT in the gold standard.
        # Treatments: the randomized holdout flag (control_group_flag) and the
        # "trigger accepted" indicator (acceptance_status). Outcomes: action_taken
        # (an action was taken) and conversion_flag (the DB STORED-GENERATED
        # outcome_value>0).
        "treatment": ["control_group_flag", "acceptance_status"],
        "outcome": ["action_taken", "conversion_flag"],
        # #1872: the OBSERVATIONAL acceptance_status -> conversion_flag edge
        # carries a REAL backdoor since COMM-ARMS Phase 4 (the trigger_accepted
        # arm is confounded on disease_severity + engagement_score —
        # treatment_arm.ARM_REGISTRY, mirrored on the causal_paths SSOT edge).
        # The pre-P4 "no confounder is offered" comment survived the semantics
        # change and silently emptied every registry-derived adjustment set:
        # measured +0.0145 upward bias on Kisqali (naive +0.0676 vs OLS-adjusted
        # +0.0531, prod DB 2026-08-31). Both are patient_journeys columns riding
        # the #1188 patient JOIN — NOT triggers columns. engagement_score is
        # pre-treatment HERE (the DGP draws it once, before arm assignment); its
        # #1188 baseline-role exclusion below guards the RCT edge's ANCOVA
        # channel, a different concern. The RCT edge's default stays UNADJUSTED
        # via the randomized_treatment guard at the submit endpoint — this offer
        # never leaks into a randomized design by default.
        "covariate": ["disease_severity", "engagement_score"],
        # #1188: curated PRE-TREATMENT baselines, joined from patient_journeys
        # via triggers.patient_id, for OPT-IN variance reduction (ANCOVA-style
        # efficiency adjustment — NOT de-confounding; the empty backdoor above
        # stays correct). Only columns fixed at/before diagnosis qualify:
        # disease_severity + age_at_diagnosis are the prognostic pair the DGP
        # plants (#1188 rev), academic_hcp + geographic_region are balanced
        # strata. Post-treatment descendants (adherence_rate, gap_days) and the
        # journey-accumulating engagement_score are EXCLUDED — adjusting on a
        # post-trigger measure would reintroduce the very bias an RCT removes
        # (see the 2026-06-29 overcontrol note above).
        "baseline_covariate": [
            "disease_severity",
            "age_at_diagnosis",
            "academic_hcp",
            "geographic_region",
        ],
        # DESIGN declaration, per-TREATMENT: only the holdout flag is a
        # randomized assignment (the DGP draws it as a pure coin flip);
        # acceptance_status is a rep CHOICE and keeps the full observational
        # unmeasured-confounding gate. Consumed by _is_randomized_treatment →
        # the agent state's ``randomized_design`` channel → the refutation
        # E-value gate + sensitivity/interpretation wording.
        "randomized_treatment": ["control_group_flag"],
    },
    # Lane A (spec 2026-09-22 §3A.3): the REAL Optum claims causal cohort —
    # Dupixent vs Xolair CSU escalation-therapy initiators (n = 15,209; XOLAIR
    # 11,009 / DUPIXENT 4,200), migration 148, loaded by
    # scripts/load_optum_causal_cohort.py from the persistence_causal export.
    # Observational (no randomized_treatment): the unmeasured-confounding gate,
    # the E-value and the refutation suite run unchanged.
    #   * treatment_dupixent: 1 = DUPIXENT, 0 = XOLAIR — the only contrast the
    #     drop observes (remibrutinib absent).
    #   * outcomes, PRIMARY first: persistent_at_180d_g28 (covered through day
    #     152 AND no gap > 60 d; brand-invariant across a 28-60 d grace),
    #     discontinued_180d (brand-robust secondary), biologic_switch_180d_flag,
    #     and the SHIPPED persistent_at_180d — days-supply sensitive per brand
    #     (14-d Dupixent vs 28-45-d Xolair fills; the -17.6 pp raw gap is a
    #     measurement artefact), reported only alongside the grace sweep
    #     (docs/demos/results/2026-09-22_persistence_definition_disproof/).
    #   * covariates: the 64 pre-index baseline features the mart manifest
    #     allow-lists (measured at the diagnosis index, which precedes treatment
    #     start). Seven are text and one-hot (see _CAUSAL_CATEGORICAL_COLUMNS).
    # No negative control is declared: the omitted-confounder experiment the
    # _CAUSAL_NEGATIVE_CONTROL_OUTCOMES note mandates per data source has not
    # been run here, so the runner emits SKIPPED no_negative_control_declared.
    # Structure discovery is OFF by default (_CAUSAL_DISCOVERY_DEFAULT_OFF):
    # measured singular on this frame (rank 45/59) and ~230 s per PC fit; the
    # curated common-cause DAG is the run shape until Lane D lands.
    "optum_biologic_persistence": {
        "treatment": ["treatment_dupixent"],
        "outcome": [
            "persistent_at_180d_g28",
            "discontinued_180d",
            "biologic_switch_180d_flag",
            "persistent_at_180d",
        ],
        "covariate": list(MART_SAFE_FEATURES),
    },
}

_DEFAULT_CAUSAL_DATASET = "patient_journeys"

# Lane A: datasets whose API default is auto_discover=False. Guided discovery
# was MEASURED to fail on the real claims frame (singular correlation matrix,
# rank 45/59; ~230 s per PC fit on 43 covariates —
# docs/demos/results/2026-09-22_discovery_real_claims_disproof/). Until Lane D's
# pre-flight lands, the default run uses the curated common-cause DAG. A caller
# that sets auto_discover=True explicitly is honored (PR #2203 then reports
# "could not run: singular…" instead of an empty DAG). The request schema's
# field default stays True (changing it would alter the generated api.ts).
_CAUSAL_DISCOVERY_DEFAULT_OFF: frozenset = frozenset({"optum_biologic_persistence"})


def _default_auto_discover(dataset: Optional[str]) -> bool:
    """The ``auto_discover`` value a request gets when the caller did not set it."""
    return (dataset or _DEFAULT_CAUSAL_DATASET) not in _CAUSAL_DISCOVERY_DEFAULT_OFF


# #1872: every nba_triggers covariate is JOINED from patient_journeys via
# triggers.patient_id (the triggers table itself carries NO covariate columns).
# Consumers that read exactly one physical table (the raw estimation-data
# endpoint, the /variables physical-table probe) must treat these like the
# #1188 baselines: never probe/select them off the triggers table.
_NBA_JOINED_COVARIATES: frozenset = frozenset(_CAUSAL_DATASET_SPECS["nba_triggers"]["covariate"])

# Row cap for the discovery leaderboard's per-question estimate (2026-07-23). The
# patient outcomes are quantile-thresholded binaries whose latent CATE attenuates
# to a small risk difference, and brand-scoping cuts rows to ~1/3, so the prior
# 1500 cap left the medium-severity commercial-arm effects (recovery-gated at
# n=8000) pulling to the estimator's noise floor. 5000 (~1650/brand scoped)
# resolves the strong + most medium effects AND is what lets them clear the
# refutation gate: MEASURED 2026-07-23 in the prod container, copay_support->
# adherent_180d is BLOCKED (E-value sensitivity, 2/3 tests) at 1500 but PASSES at
# 5000. The cost is real and accepted (owner decision): each full agent estimation
# is ~158s at 5000 (~82s at 1500), so the 30-question patient_journeys leaderboard
# is a ~79-min BACKGROUND job — NOT interactive. That is acceptable because the run
# is async (submit->poll, 8h TTL) and fills STRONGEST-FIRST (see _prerank_questions),
# so the top effects surface within minutes. The cheap FWL / partial-correlation
# PRE-RANK screens stay at 1500 (they only order candidates; not the reported estimate).
_DISCOVERY_ROW_CAP = 5000


def _is_randomized_treatment(dataset: Optional[str], treatment_var: str) -> bool:
    """Whether this question's treatment is RANDOMIZED by design.

    Reads the curated spec's ``randomized_treatment`` list — a per-TREATMENT
    design declaration (nba_triggers carries both the randomized holdout and
    the rep-chosen acceptance_status; only the former qualifies). Fail-closed:
    unknown dataset / unlisted treatment → False, and this must NEVER be
    inferred from an empty discovered backdoor (an observational question where
    discovery found nothing still deserves the unmeasured-confounding gate).
    """
    spec = _CAUSAL_DATASET_SPECS.get(dataset or _DEFAULT_CAUSAL_DATASET, {})
    return treatment_var in spec.get("randomized_treatment", [])


# #2007 (Lane G): NEGATIVE-CONTROL OUTCOMES, per dataset, per TREATMENT.
#
# A negative-control outcome is an outcome the treatment cannot causally affect
# but that shares the treatment's confounders (Lipsitch, Tchetgen Tchetgen &
# Cohen 2010, "Negative controls: a tool for detecting confounding and bias in
# observational studies"). Re-running the SAME adjusted fit with the control as
# the outcome should give an interval containing zero; a non-null control
# effect is direct evidence that the adjustment set left confounding behind.
#
# This registry is DECLARED, never inferred from discovery, and lists ONLY the
# pairs MEASURED to respond on the synthetic generator: the control's omitted-
# confounder fit leaves its 95% CI on >= 5 of 6 seeds (42, 7, 123, 2024, 99,
# 314) at PRODUCTION's nuisance config (LinearDML, RF leaf 50 from
# src/causal_engine/nuisance_config.py, X = W, #2031; seed-21 frame, n = 1500;
# tests/unit/test_causal_engine/test_negative_control_calibration_2007.py):
#   copay_support -> treatment_initiated  omitted 6/6 seeds, seed-mean +0.062
#   psp_enrolled  -> treatment_initiated  omitted 6/6 seeds, seed-mean +0.086
# (adjusted CI contains 0 on 6/6 seeds for both; 0/9 adjusted false positives
# on the structural nulls; 11/11 planted truths detected by the adjusted fit).
# The 2026-09-11 registry (docs/demos/results/2026-09-11_negative_control_
# disproof/disproof.md) was measured on ONE seed at RF leaf 5 and also declared
# rep_detailing_high -> persistent_180d (omitted +0.058): over 6 seeds that
# candidate leaks a STABLE ~ +0.04 (~ 1.6 SE) and never clears the bar (0/6 at
# leaf 50, 2/6 at leaf 5) — the single-seed declaration was a high draw
# (+0.058 vs 6-seed mean +0.042), so it was dropped (#2031, 2026-09-12).
#
# DELIBERATELY ABSENT: rep_detailing_high, sample_dropped and trigger_accepted
# — none of their candidate controls responds (0/6 seeds each at n = 1500), so
# a declared control would PASS under confounding and read as false assurance.
# treatment_arm has no structural-null outcome in the generator (it moves every
# outcome); hcp_adoption and nba_triggers declare none. For every undeclared
# pair the runner emits SKIPPED ``no_negative_control_declared`` — a null is a
# finding, never a fabricated PASS.
#
# When Optum / CSU (or any non-synthetic) data arrive, this registry MUST be
# re-verified PER DATA SOURCE with the same omitted-confounder experiment before
# the test is allowed to score: a control that responds on the generator is
# not evidence it responds on real claims data.
#
# SHAPE (why the control is NOT a column of ``estimation_data``): the submit
# endpoint fetches the declared control as a loader PASSTHROUGH column (same
# rows, same select), and ``_run_agent_analysis_task`` splits it off into
# ``data_cache["negative_control_data"]`` — a one-column frame sharing the
# estimation frame's index. Measured disproof of "an extra frame column is
# inert": two existing consumers treat EVERY non-question column of the
# estimation frame as a covariate —
#   * src/agents/causal_impact/nodes/graph_builder.py:787-800 (guided
#     discovery) tiers all of them as candidate pre-treatment covariates and
#     hands the whole frame to discover_dag, so the control would become a
#     DAG node and could enter the DAG-derived adjustment set;
#   * src/agents/causal_impact/nodes/estimation.py:286-295 (no-backdoor
#     fallback) adjusts on all of them.
# Only the static PROVENANCE_DROP_COLS is excluded on those paths; a per-
# question column cannot go there. Keeping the control OUT of the frame holds
# "the estimate conditions on exactly the declared covariates" by construction
# instead of by two nodes each remembering to drop a dynamic column. The
# refutation node aligns by index (the #1419 subsample is ``frame.iloc[...]``,
# which keeps the original labels): ``negative_control_data.loc[frame.index]``.
_CAUSAL_NEGATIVE_CONTROL_OUTCOMES: Dict[str, Dict[str, str]] = {
    "patient_journeys": {
        "copay_support": "treatment_initiated",
        "psp_enrolled": "treatment_initiated",
    },
}


def _negative_control_outcome(
    dataset: Optional[str], treatment_var: str, outcome_var: str
) -> Optional[str]:
    """The declared negative-control outcome column for this question, or None.

    Fail-closed like :func:`_is_randomized_treatment`: None when the dataset
    or treatment is undeclared, when the mapped column IS the outcome under
    test (a control cannot be the outcome it is meant to check — e.g.
    psp_enrolled -> treatment_initiated), or when the mapped column is not in
    the dataset spec's ``outcome`` list (an unlisted column is never fetched).
    Never inferred from discovery.
    """
    dataset_key = dataset or _DEFAULT_CAUSAL_DATASET
    control = _CAUSAL_NEGATIVE_CONTROL_OUTCOMES.get(dataset_key, {}).get(treatment_var)
    if control is None or control == outcome_var:
        return None
    spec = _CAUSAL_DATASET_SPECS.get(dataset_key)
    if spec is None or control not in spec.get("outcome", []):
        return None
    return control


# --- Brand-aware clinical covariate gating (Phase 2) --------------------------------
# After the DGP brand-gating (src.ml.synthetic.clinical_codes.BRAND_ELIGIBILITY_FIELDS)
# the indication-specific clinical columns are populated ONLY for their own brand's
# rows and NULL off-brand. Adjusting on a column that is NULL across the cohort feeds
# NaN straight into EconML (`ValueError: Input contains NaN`) — so the adjustment set
# must be brand-aware:
#   * UNIVERSAL covariates are populated for EVERY brand's rows (never NULL) -> always safe.
#   * a brand's own clinical covariates are safe ONLY when that brand is the row filter;
#     for the all-brands cohort (brand=None) they are ~2/3 NULL, so they are excluded.
# _BRAND_CLINICAL_COVARIATES is the numeric-adjustment SUBSET of the DGP SSOT
# BRAND_ELIGIBILITY_FIELDS (test_brand_covariate_consistency locks the two together so
# this map cannot silently drift from the generator's gating).
_UNIVERSAL_COVARIATES: frozenset = frozenset(
    {
        "disease_severity",
        "engagement_score",
        "age_at_diagnosis",
        "academic_hcp",
        "geographic_region",
    }
)

_BRAND_CLINICAL_COVARIATES: Dict[str, frozenset] = {
    # biologic_experienced (Phase 3) is Remibrutinib's planted CATE effect-modifier;
    # it is in BRAND_ELIGIBILITY_FIELDS["Remibrutinib"] so the subset-consistency
    # gate stays green. ige_level is intentionally NOT here (descriptive only).
    # urticaria_severity_uas7 is Remibrutinib's #1321 axis (uncontrolled CSU, its
    # _derive_is_uncontrolled_csu dichotomization is ALSO its planted CATE modifier).
    "Remibrutinib": frozenset({"urticaria_severity_uas7", "biologic_experienced"}),
    # disease_stage (#1321 rollout) is Kisqali's planted CATE effect-modifier
    # (advanced-line patients respond less); it is in BRAND_ELIGIBILITY_FIELDS["Kisqali"]
    # so test_brand_covariate_consistency stays green. Brand-scoped so it is dropped from
    # the all-brands adjustment set (NULL for non-Kisqali rows).
    "Kisqali": frozenset({"ecog_performance_status", "disease_stage"}),
    # complement_inhibitor_status (#1321 pilot) is Fabhalta's planted CATE
    # effect-modifier (prior-C5-experienced respond less to iptacopan); it is in
    # BRAND_ELIGIBILITY_FIELDS["Fabhalta"] so test_brand_covariate_consistency
    # stays green. Brand-scoped so it is dropped from the all-brands adjustment
    # set (NULL for non-Fabhalta rows), exactly like biologic_experienced.
    "Fabhalta": frozenset(
        {"egfr", "proteinuria_g_day", "ldh_ratio", "complement_inhibitor_status"}
    ),
}

# Union of every indication-specific clinical covariate — the columns that must be
# dropped from the adjustment set unless the selected brand vouches for them.
_ALL_CLINICAL_COVARIATES: frozenset = frozenset().union(*_BRAND_CLINICAL_COVARIATES.values())


def _brand_scoped_covariates(covariates: List[str], brand: Optional[str]) -> List[str]:
    """Drop indication-specific clinical covariates that are NULL for the selected
    brand cohort (Phase 2 brand-gating). Universals and any non-clinical column pass
    through unchanged; a brand's own clinical columns survive only when that brand is
    the row filter. Order-preserving. brand=None (all-brands) keeps ONLY the universals
    among the clinical set, since each clinical column is populated for just one brand.
    """
    valid_clinical = _BRAND_CLINICAL_COVARIATES.get(brand or "", frozenset())
    return [c for c in covariates if c not in _ALL_CLINICAL_COVARIATES or c in valid_clinical]


# Datasets that are NOT a single physical table — built by a JOIN-aware loader
# (e.g. hcp_adoption = hcp_brand_adoption ⋈ hcp_profiles, centrality_z derived).
# Endpoints that issue a single-table client.table(dataset) read MUST special-case
# these (P3 adds its grain here if it is also non-single-table).
_JOIN_DATASETS: frozenset = frozenset({"hcp_adoption"})

# Lane A: the seven TEXT baseline columns of the Optum mart (payer / geography /
# gender / the two comorbidity risk bands), one-hot encoded by the loader.
_OPTUM_BASELINE_CATEGORICALS: frozenset = frozenset(
    {
        "gdr_cd",
        "payer_category",
        "payer_product",
        "payer_bus",
        "charlson_risk_band",
        "elixhauser_risk_band",
        "geographic_region",
    }
)

# Columns coerced to float before handing the frame to the executors. Every
# curated candidate above is numeric, so all are coerced; a value that cannot
# be coerced becomes None and (for treatment/outcome) drops the row.
_CAUSAL_NUMERIC_COLUMNS: Dict[str, set] = {
    "patient_journeys": {
        "treatment_arm",
        "treatment_initiated",
        "persistent_180d",
        "discontinued_180d",
        "disease_severity",
        "engagement_score",
        "age_at_diagnosis",
        "academic_hcp",
        "egfr",
        "proteinuria_g_day",
        "ldh_ratio",
        "urticaria_severity_uas7",
        "ecog_performance_status",
        "biologic_experienced",  # Phase 3: 0/1, float-coerced like the other flags
        "adherent_180d",
        "low_gap_180d",
        "adherence_rate",
        "gap_days",
        "copay_support",
        "psp_enrolled",
        "rep_detailing_high",  # Phase 3: 0/1 initiation-latent arm, float-coerced
        "sample_dropped",  # Phase 3: 0/1 initiation-latent arm, float-coerced
        "trigger_accepted",  # Phase 4: 0/1 initiation-latent arm, float-coerced (SMALLINT in DB)
        "insurance_access_score",
        # #1321 Fabhalta pilot: text "current"/"prior" -> 1.0/0.0 via
        # _derive_is_prior_c5 (below), then float-coerced like acceptance_status
        # on nba_triggers (both derivation + numeric membership, belt-and-braces).
        "complement_inhibitor_status",
        # #1321 rollout: Kisqali advanced-line, text disease_stage -> 1.0/0.0 via
        # _derive_is_advanced_line (below). urticaria_severity_uas7 is already numeric
        # above; its axis derivation coerces the same float, so no new entry is needed.
        "disease_stage",
    },
    "hcp_adoption": {
        "peer_influence_score",
        "treatment_arm",
        "adopted",
        "centrality_z",
    },
    "nba_triggers": {
        # Every trigger question column coerces to numeric 0/1 (booleans via
        # float(bool); acceptance_status/action_taken via the derivations below).
        "control_group_flag",
        "action_taken",
        "conversion_flag",
        "acceptance_status",
        # #1872: patient-JOINED backdoor confounders of the acceptance edge —
        # membership here keeps them through the discovery-path intersection
        # (spec covariate ∩ numeric∪categorical in _discover_candidate_questions).
        "disease_severity",
        "engagement_score",
    },
    # Lane A: the treatment, the four outcomes and every NON-text baseline feature
    # float-coerce (ints / 0-1 flags / the age). test_causal_optum_dataset_registry
    # locks numeric ∪ categorical == MART_SAFE_FEATURES so a manifest change
    # cannot silently null-coerce a text column.
    "optum_biologic_persistence": {
        "treatment_dupixent",
        "persistent_at_180d_g28",
        "discontinued_180d",
        "biologic_switch_180d_flag",
        "persistent_at_180d",
        *(c for c in MART_SAFE_FEATURES if c not in _OPTUM_BASELINE_CATEGORICALS),
    },
}

# Per-dataset brand-filter column. The triggers table has NO `brand` column — it
# carries `brand_id` (text, NOT NULL). Datasets absent here default to "brand"
# (patient_journeys). Used by _list_dataset_brands + the loaders' brand filter.
_CAUSAL_BRAND_COLUMN: Dict[str, str] = {
    "nba_triggers": "brand_id",
    # Lane A: the brand filter IS the treatment label. Scoping to one brand makes
    # the treatment constant — NOT a loud estimation-time failure (DoWhy still
    # returns a finite estimate on a constant treatment, and refutation.py's
    # nunique()==2 check silently switches to the continuous-treatment path
    # instead of refusing). The loader's constant-treatment guard
    # (_load_agent_estimation_frame, right after the frame is built) refuses
    # a one-arm scope with a 400 before that can happen. The dropdown offers
    # the brand filter because the table has it; the analyst's all-brands
    # default is the causal contrast.
    "optum_biologic_persistence": "index_biologic_brand",
}


# Per-dataset value derivations applied BEFORE float-coercion, for columns whose
# raw value is not directly float()-able into the modeled 0/1 (categorical /
# text / bool). Only allowlisted columns are reachable (the allowlist gate runs
# first), so a derivation can never read an arbitrary column. Each maps a raw cell
# (possibly None) to a float.
def _derive_is_accepted(value: Any) -> float:
    """acceptance_status -> 1.0 when 'accepted' (case-insensitive), else 0.0."""
    if value is None:
        return 0.0
    return 1.0 if str(value).strip().lower() == "accepted" else 0.0


def _derive_presence(value: Any) -> float:
    """A nullable text/flag column -> 1.0 when a non-empty value is present."""
    if value is None:
        return 0.0
    text = str(value).strip()
    return 0.0 if text == "" or text.lower() in {"none", "false", "0"} else 1.0


def _derive_is_prior_c5(value: Any) -> float:
    """complement_inhibitor_status -> 1.0 when 'prior' (the eculizumab/ravulizumab
    switch population), else 0.0 (#1321 Fabhalta pilot). The treatment contrast is
    prior-C5-experienced vs C5-naive; None (off-brand) is NOT reached here because
    the derivation only runs on non-None values (a NULL complement_inhibitor_status
    stays None and drops the row as a missing treatment — but the question is
    Fabhalta-scoped, where the column is always populated)."""
    if value is None:
        return 0.0
    return 1.0 if str(value).strip().lower() == "prior" else 0.0


# #1321 rollout: the two additional brand-distinct axes. Both mirror _derive_is_prior_c5
# — a text/continuous eligibility column dichotomized to the 0/1 treatment contrast.
_ADVANCED_LINE_STAGES = {"metastatic", "stage_iv"}


def _derive_is_advanced_line(value: Any) -> float:
    """disease_stage -> 1.0 for the advanced-line CDK4/6 burden (metastatic / stage_iv),
    else 0.0 (#1321 Kisqali axis). Kisqali-scoped: disease_stage is populated only for
    Kisqali rows, so None (off-brand) is not reached."""
    if value is None:
        return 0.0
    return 1.0 if str(value).strip().lower() in _ADVANCED_LINE_STAGES else 0.0


# UAS7 >= 28 == uncontrolled CSU (high disease activity; the governing urticaria
# guideline puts "severe" at 28-42 of the 0-42 UAS7). A brand/demo threshold, not a
# recalculation of UAS7_UNCONTROLLED_THRESHOLD (that is the weekly-score cutoff).
_UNCONTROLLED_UAS7_THRESHOLD = 28.0


def _derive_is_uncontrolled_csu(value: Any) -> float:
    """urticaria_severity_uas7 -> 1.0 for uncontrolled CSU (UAS7 >= 28), else 0.0 (#1321
    Remibrutinib axis). Remibrutinib-scoped: uas7 is populated only for Remibrutinib rows.
    A non-numeric value is treated as 0.0 (never the axis) rather than raising."""
    if value is None:
        return 0.0
    try:
        return 1.0 if float(value) >= _UNCONTROLLED_UAS7_THRESHOLD else 0.0
    except (TypeError, ValueError):
        return 0.0


_CAUSAL_NUMERIC_DERIVATIONS: Dict[str, Dict[str, Callable[[Any], float]]] = {
    "patient_journeys": {
        "complement_inhibitor_status": _derive_is_prior_c5,
        # #1321 rollout: the causal loaders see these as the 0/1 axis. urticaria_severity_uas7
        # is ALSO a Remibrutinib covariate — dichotomizing it there is immaterial (it adjusts
        # a RANDOMIZED treatment_arm, a precision covariate, not a confounder). KPIs read the
        # raw continuous column via their OWN loaders. segments.py applies this map to its
        # QUESTION slots only (wave 53): the treatment axis runs as the same 0/1 contrast the
        # page labels and the registry validated, while the column stays raw as an effect
        # modifier.
        "disease_stage": _derive_is_advanced_line,
        "urticaria_severity_uas7": _derive_is_uncontrolled_csu,
    },
    "nba_triggers": {
        "acceptance_status": _derive_is_accepted,
        "action_taken": _derive_presence,
    },
}

# Per-dataset outcome columns whose NULL is a DESIGNED zero (not missing data), so
# a NULL value fills to 0.0 rather than dropping the row. On triggers, action_taken
# is NULL when no action was taken (= 0) and conversion_flag is NULL when not
# converted (the STORED-GENERATED outcome_value>0 is NULL-not-false); dropping those
# rows would discard the RCT control arm / the non-converters and bias the estimate.
_CAUSAL_FILL_ZERO_OUTCOMES: Dict[str, set] = {
    "nba_triggers": {"action_taken", "conversion_flag"},
}

# Logical-dataset -> physical-table name. Datasets whose dataset key differs from
# their real table go here (nba_triggers -> the triggers table). Absent => itself.
_CAUSAL_PHYSICAL_TABLE: Dict[str, str] = {
    "nba_triggers": "triggers",
    # Lane A: migration 148.
    "optum_biologic_persistence": "optum_biologic_persistence_causal",
}


# Categorical covariates ONE-HOT ENCODED before the frame reaches the executors
# (DoWhy/EconML require numeric inputs). DELIBERATELY absent from
# _CAUSAL_NUMERIC_COLUMNS so the loader does NOT float-coerce them to None;
# _one_hot_categoricals expands each into stable <col>=<level> 0/1 float dummies
# (drop_first reference level). geographic_region is the modeled RETENTION
# confounder: an unordered 4-level region (midwest/south/northeast/west).
_CAUSAL_CATEGORICAL_COLUMNS: Dict[str, set] = {
    "patient_journeys": {"geographic_region"},
    "optum_biologic_persistence": set(_OPTUM_BASELINE_CATEGORICALS),
}


async def _list_dataset_brands(dataset: str) -> List[str]:
    """Distinct, non-null brand values present in ``dataset``'s live table.

    Data-driven (mirrors how /variables intersects with the live schema): the
    dropdown only ever offers a brand that actually has rows. Returns [] if the
    table has no ``brand`` column or the store is unavailable (the FE then shows
    only 'All brands'). Bounded select — the cohort is small and a few thousand
    rows reliably cover every brand.
    """
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        return []
    try:
        # hcp_adoption is a JOIN dataset; its brand column lives on hcp_brand_adoption.
        brand_table = (
            "hcp_brand_adoption"
            if dataset == "hcp_adoption"
            else _CAUSAL_PHYSICAL_TABLE.get(dataset, dataset)
        )
        brand_col = _CAUSAL_BRAND_COLUMN.get(dataset, "brand")
        query = client.table(brand_table).select(brand_col)
        query = apply_provenance_filter(query)
        result = await query.limit(20000).execute()
    except Exception as e:  # noqa: BLE001 — missing column / store hiccup => no brands
        logger.warning(f"causal brands: could not enumerate brands for '{dataset}': {e}")
        return []
    seen = {
        str(row[brand_col])
        for row in (result.data or [])
        if isinstance(row, dict) and row.get(brand_col)
    }
    return sorted(seen)
