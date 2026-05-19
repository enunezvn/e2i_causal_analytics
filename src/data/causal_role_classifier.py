"""Layer 4 — DSPy-compiled CausalRoleClassifier.

Classifies a proposed feature derivation as ancestor / confounder / mediator /
collider / descendant / instrument with respect to the prediction target.
Compiled on a curated subset (21 of 18+) of documented past-leakage incidents
from this codebase's history (`.claude/state/leakage_compile_set_20260507.md`)
plus 8 domain-expert collider / instrument exemplars added under issue #198
plus 1 explicit negative-direction confounder exemplar
(baseline_severity_score_preindex) added on codex pass-5 to teach the
discrimination boundary between confounder (arrows OUT of severity) and
collider (arrows IN to V), plus 12 paired (T, Y)-explicit demos added on
Phase-4 S12 Option C recompile (2026-05-19) to teach the classifier to
read ``treatment=X; outcome=Y`` as first-class semicolon-delimited fields
in ``dataset_context``.

Compile-set role coverage:
- Legacy 21 (cohort-only context): ancestor=1, confounder=3, mediator=1,
  descendant=8, collider=4, instrument=4.
- Phase-4 S12 Option C 12 paired demos ((T, Y)-explicit context): 1 ancestor,
  3 confounder, 3 mediator, 1 descendant, 2 collider, 2 instrument.
- Combined 33: ancestor=2, confounder=6, mediator=4, descendant=9, collider=6,
  instrument=6.

``dataset_context`` schema (Phase-4 S12 Option C — backward-compatible):
the field remains ``str``; recognized semicolon-delimited keys are
``cohort=``, ``target=``, ``anchor=``, plus newly optional
``treatment=`` and ``outcome=``. Production callers (cohort-only) continue
working unchanged; S12 callers may supply explicit (T, Y) to enable
formal instrument-recall identification on a per-(T, Y)-pair basis.

All six declared ``CausalRole`` Literal values are
represented (previously: 4 of 6; collider + instrument were deferred to issue
#198 pending domain-expert labeling). All 4 collider examples are
confounder-collider / M-structures per Greenland-Pearl-Robins 1999 (the
dominant collider failure mode in observational pharmacoepi, where baseline
severity is itself a T-Y confounder with arrowheads into BOTH T and V). The
examples vary in derivation MECHANISM to teach the LM that the
confounder-collider pattern transfers across feature shapes: count of
utilization events (hospitalizations_total), count of medication events
(concomitant_steroid_burst_count_followup), count of workup events
(diagnostic_test_count_followup), and binary sample-inclusion gate
(alive_at_180d_observation_window). The compile set explicitly rejects
"T AND (post-T event)" framings (e.g., discontinuation_flag =
treatment_initiated AND fill-gap) as colliders because the second "arrow"
there is itself a downstream of T, making the variable a descendant, not a
collider. The instrument examples span two pharmacoepi IV families:
supply-side geographic (urban_rural_code, geographic_region) and
preference/volume-based provider IVs (provider_preference_score,
index_provider_biologic_volume_prior_year). Each instrument rationale
phrases the exclusion restriction as an ASSUMPTION TO AUDIT (with a named
validity-check step), not as a confessed violation — so the LM learns
IV-validity discipline rather than a "name a violation and still call it
an IV" pattern.

Why DSPy: replace ad-hoc Claude prompts (which hallucinated feature names per
the documented synthetic_v2 incident) with a STRUCTURED PROGRAM that:
- Has a typed input/output schema (no free-form LLM hallucination)
- Optimizes its prompts against labeled examples (BootstrapFewShot)
- Provides reproducible reasoning (CoT with self-consistency)
- Improves over time as new labeled incidents are added

Disease-agnostic by construction: the classification VOCABULARY (ancestor /
confounder / mediator / collider / descendant / instrument) is from causal
inference theory (Pearl), not specific to any disease. The compile set spans
CSU and Optum, teaching cross-disease patterns ("unwindowed event aggregations
are descendants").

Reference: .claude/plans/adaptive_temporal_validity_redesign.md (Layer 4).
"""

from __future__ import annotations

from typing import Any, Literal

import dspy

CausalRole = Literal[
    "ancestor",  # Pre-prediction-time, causally upstream of target
    "confounder",  # Influences both prediction-time-knowable factors and target
    "instrument",  # Affects target only through known mediators (rare)
    "mediator",  # On the causal path FROM treatment TO outcome
    "collider",  # Influenced by both target and other features
    "descendant",  # Causally downstream of treatment (typical leak pattern)
]

Remediation = Literal[
    "drop",  # Remove feature entirely
    "window",  # Apply temporal windowing to make pre-prediction-time
    "transform",  # Re-derive without post-prediction-time inputs
    "keep_with_caveat",  # Keep with documented audit-trail justification
]


class CausalRoleSignature(dspy.Signature):
    """Classify the causal role of a proposed feature relative to the prediction target.

    The model must reason from the feature's derivation steps and the dataset
    context, NOT from prior knowledge of feature names. Hallucinated feature
    names (proposing replacements not in the column list) are explicitly
    forbidden.
    """

    feature_name: str = dspy.InputField(desc="Name of the feature being classified.")
    derivation_pseudocode: str = dspy.InputField(
        desc="Plain-English or pseudo-code description of how the feature is derived. "
        "Must specify the SOURCE TABLE (e.g., medication_events, lab_events, demo) "
        "and any temporal filtering."
    )
    dataset_context: str = dspy.InputField(
        desc="Context about the dataset: target variable, prevalence, prediction-time "
        "anchor (e.g., 'index_date'), data source (e.g., 'ConcertAI CSU claims')."
    )
    causal_role: CausalRole = dspy.OutputField(
        desc="The feature's causal role with respect to the target."
    )
    mechanism: str = dspy.OutputField(
        desc="Concise explanation of WHY the feature has this causal role (1-2 sentences). "
        "Reference SPECIFIC mechanism (e.g., 'aggregates events post-prediction without "
        "temporal filter', 'pre-existing demographic measured before prediction time')."
    )
    recommended_remediation: Remediation = dspy.OutputField(
        desc="What to do with this feature in the ML pipeline."
    )


class CausalRoleClassifier(dspy.Module):
    """DSPy module wrapping the CausalRoleSignature with chain-of-thought reasoning.

    Usage:
        classifier = CausalRoleClassifier()
        result = classifier(
            feature_name="journey_duration_days",
            derivation_pseudocode="end_date - index_date; end_date = max(eligend, last_med+supply, ...)",
            dataset_context="ConcertAI CSU; target=treatment_initiated; prevalence=0.024; "
                            "prediction_anchor=index_date",
        )
        print(result.causal_role, result.mechanism, result.recommended_remediation)
    """

    def __init__(self):
        super().__init__()
        # ChainOfThought adds a hidden 'reasoning' field that improves accuracy
        # for nuanced classification tasks per DSPy v3+ best practices.
        self.classify = dspy.ChainOfThought(CausalRoleSignature)

    def forward(
        self,
        feature_name: str,
        derivation_pseudocode: str,
        dataset_context: str,
    ) -> dspy.Prediction:
        return self.classify(
            feature_name=feature_name,
            derivation_pseudocode=derivation_pseudocode,
            dataset_context=dataset_context,
        )


def build_compile_set() -> list[dspy.Example]:
    """Build the DSPy compile set: 33 curated examples covering all 6 roles.

    Composition:
    - 21 legacy demos (cohort-only ``dataset_context``).
    - 12 Phase-4 S12 Option C paired (T, Y)-explicit demos (2026-05-19;
      see ``.claude/plans/option_c_dspy_recompile_for_s12_FINAL.md``).

    Of the 18 incidents catalogued at
    ``.claude/state/leakage_compile_set_20260507.md``, 12 have been distilled
    into typed ``dspy.Example`` objects below (the remaining 6 are either
    duplicates of an already-represented mechanism or have been folded into
    the broader exemplars). 8 additional ``collider`` and ``instrument``
    exemplars were added under issue #198 from domain-expert review, plus 1
    explicit negative-direction confounder exemplar
    (baseline_severity_score_preindex) added on codex pass-5 to teach the
    discrimination boundary between confounder (arrows OUT) and collider
    (arrows IN). Phase-4 S12 Option C adds 12 paired demos covering 6
    feature/derivation pairs × 2 (T, Y) variants each, designed for
    falsifiability: same Z + same derivation, different (T, Y) =>
    different graph-theoretic role. The 12 paired demos are pinned by
    quadruple in
    ``tests/unit/test_data/test_causal_role_classifier.py::test_persisted_artifact_emits_role_conditional_on_treatment_outcome``.

    All 4 collider examples are confounder-collider M-structures (per
    Greenland-Pearl-Robins 1999) — the dominant collider failure mode
    in observational pharmacoepi, where baseline severity is itself a
    T-Y confounder with an arrowhead into V. They differ in derivation
    MECHANISM to teach the LM that the confounder-collider pattern
    transfers across feature shapes: count of utilization
    (hospitalizations_total), count of medication
    (concomitant_steroid_burst_count_followup), count of workup
    (diagnostic_test_count_followup), and binary sample-inclusion gate
    (alive_at_180d_observation_window). Each rationale exposes BOTH
    parents in the derivation: baseline severity (T-Y confounder with
    arrowhead into V) AND a T-driven second arrow (AE, non-response,
    protocol-monitoring, or T-mediated survival respectively).
    `discontinuation_flag` / `discontinued_180d` / `persistent_at_180d`
    are intentionally NOT used as colliders because their derivation
    reduces to ``T AND (post-T event)`` — the second "arrow" is
    downstream of T rather than an independent cause.

    The 4 instrument examples span two pharmacoepi IV families: supply-
    side geographic (urban_rural_code, geographic_region) and
    preference-based provider IVs (provider_preference_score,
    index_provider_biologic_volume_prior_year). Each rationale phrases
    the exclusion restriction as an ASSUMPTION-TO-AUDIT with a named
    validity check step (per Brookhart 2010 IV-validity audit), NOT as
    a confessed Z -> Y violation. The LM thereby learns to FLAG
    candidate IVs whose exclusion restriction has not been validated,
    rather than to accept "exclusion violation named -> still an IV" as
    a pattern. `plan_type` was considered but rejected on codex pass-2
    because as an enrollment-time payer feature it duplicates the
    `insurance_product` confounder exemplar and creates contradictory
    training signal for an access/coverage variable.

    Coverage by role: ancestor=2 (1 legacy + 1 Option C), confounder=6
    (3 legacy + 3 Option C), mediator=4 (1 legacy + 3 Option C),
    descendant=9 (8 legacy + 1 Option C), collider=6 (4 legacy + 2 Option C),
    instrument=6 (4 legacy + 2 Option C). Total: 33.

    Source: .claude/state/leakage_compile_set_20260507.md + issue #198
    + .claude/plans/option_c_dspy_recompile_for_s12_FINAL.md (Phase-4 S12).
    """
    examples = [
        # Incident 1
        dspy.Example(
            feature_name="disease_severity",
            derivation_pseudocode=(
                "min(3.0, sum(med.fill_count) * 0.5) + min(2.0, count(proc.code='J2357')) "
                "+ count(lab.flag='abnormal')  # over entire patient panel, NO pre-index filter"
            ),
            dataset_context=(
                "ConcertAI CSU claims; target=treatment_initiated; prevalence=0.024; "
                "prediction_anchor=index_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Aggregates medication fills, J2357 procedures, and abnormal labs across the "
                "ENTIRE patient panel without filtering to events before the prediction time. "
                "Treated patients accumulate post-index events that drive the score, making it "
                "structurally a post-treatment indicator."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 2
        dspy.Example(
            feature_name="engagement_score",
            derivation_pseudocode=(
                "len(unique(med.npi)) + count(med.fills) + count(lab.tests)  # over entire panel"
            ),
            dataset_context="ConcertAI CSU; target=treatment_initiated; anchor=index_date",
            causal_role="descendant",
            mechanism=(
                "Counts unique HCPs, medication fills, and lab tests across the patient panel "
                "without temporal filtering. Score is deterministically zero for untreated "
                "patients (no medication events) and non-zero for treated patients."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 3
        dspy.Example(
            feature_name="days_on_therapy",
            derivation_pseudocode="sum(med.days_supply) over entire panel",
            dataset_context="CSU; target=treatment_initiated; anchor=index_date",
            causal_role="descendant",
            mechanism=(
                "Sum of medication days_supply directly measures therapy duration, which is "
                "structurally post-treatment. For untreated patients this is identically zero."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 4
        dspy.Example(
            feature_name="medication_claim_count",
            derivation_pseudocode="count(rows in medication_events where patient_id=p)",
            dataset_context="CSU/Optum; target=treatment_initiated; anchor=index_date",
            causal_role="descendant",
            mechanism=(
                "Total count of medication claims is tautologically equivalent to the target: "
                "len > 0 if and only if patient was treated. Perfect class separator."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 6 — journey_duration_days (the famous escape-the-threshold case)
        dspy.Example(
            feature_name="journey_duration_days",
            derivation_pseudocode=(
                "(journey_end_date - journey_start_date).days where journey_end_date = "
                "max(eligend, last_med_date + days_supply, last_proc_date, last_lab_date)"
            ),
            dataset_context="CSU; target=treatment_initiated; anchor=index_date",
            causal_role="mediator",
            mechanism=(
                "Journey end date is computed from max of clinical event dates without a "
                "pre-prediction-time filter. For treated patients the medication events "
                "extend end_date past index, making journey_duration depend on treatment. "
                "Single-feature AUC=0.689 escapes hardcoded thresholds but is detected by "
                "permutation-baseline z-score."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 8 — composite/derived from target itself
        dspy.Example(
            feature_name="journey_status",
            derivation_pseudocode=(
                "categorical from (treatment_initiated, discontinuation_flag) tuple"
            ),
            dataset_context="synthetic; target=treatment_initiated",
            causal_role="descendant",
            mechanism=(
                "Derived directly from the target variable. Cramér's V to target = 1.0, "
                "perfect class separation."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 13 — composite charlson_score
        dspy.Example(
            feature_name="charlson_score",
            derivation_pseudocode=(
                "weighted sum over comorbidity diagnoses across entire patient diagnosis "
                "history (no pre-prediction-time filter)"
            ),
            dataset_context="Optum; target=treatment_initiated",
            causal_role="descendant",
            mechanism=(
                "Comorbidity scores aggregate diagnoses across the entire patient timeline. "
                "Diagnoses recorded post-treatment contaminate the score. Without temporal "
                "filtering the feature is post-prediction-time."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 14-17 (collapsed): boolean diagnosis flags
        dspy.Example(
            feature_name="has_angioedema",
            derivation_pseudocode="any(diagnosis_events where dx_code='T78.3')",
            dataset_context="Optum; target=treatment_initiated",
            causal_role="descendant",
            mechanism=(
                "Boolean any-time-in-history diagnosis flag captures post-prediction-time "
                "diagnoses. Class_0 patients have it never observed; class_1 patients had "
                "diagnosis recorded at any time including post-treatment."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 12 — atopy_score composite
        dspy.Example(
            feature_name="atopy_score",
            derivation_pseudocode=(
                "weighted composite of atopic-disease diagnosis indicators "
                "(eczema, asthma, allergic_rhinitis) across patient timeline"
            ),
            dataset_context="Optum; target=treatment_initiated",
            causal_role="descendant",
            mechanism=(
                "Composite of multiple post-prediction-time diagnosis booleans inherits "
                "their post-hoc semantics. Without temporal filtering at component level "
                "the composite leaks post-treatment information."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Incident 7 — prior_treatments (correctly windowed; ambiguous case)
        dspy.Example(
            feature_name="prior_treatments",
            derivation_pseudocode=(
                "list of medication_event.med_class where medication_date < index_date"
            ),
            dataset_context="CSU; target=treatment_initiated",
            causal_role="confounder",
            mechanism=(
                "Pre-index medication history is genuinely pre-prediction-time and reflects "
                "prior care patterns that influence both observable patient attributes and "
                "treatment decision. Properly windowed."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Legitimate ancestor: age
        dspy.Example(
            feature_name="age_at_index",
            derivation_pseudocode="(index_date - birth_date).years",
            dataset_context="CSU/Optum; target=treatment_initiated; anchor=index_date",
            causal_role="ancestor",
            mechanism=(
                "Age at the prediction time is a stable demographic measured before the "
                "prediction time. Causally upstream of both treatment decisions and outcomes."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Legitimate confounder: insurance product
        dspy.Example(
            feature_name="insurance_product",
            derivation_pseudocode="demo.insurance_product (one row per patient at enrollment)",
            dataset_context="CSU/Optum; target=treatment_initiated",
            causal_role="confounder",
            mechanism=(
                "Insurance product type is set at enrollment, before the prediction time. "
                "Influences both healthcare access (affecting observable patient attributes) "
                "and treatment authorization decisions."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Issue #198 codex pass-5: NEGATIVE-DIRECTION DISCRIMINATION
        # exemplar. A pure pre-index baseline severity score is a
        # CONFOUNDER (arrows OUT: severity -> T via prescriber decision
        # and severity -> Y via uncontrolled disease activity). It is NOT
        # a collider (arrows go OUT of severity, not INTO it). This
        # exemplar pairs with the 4 confounder-collider M-structures in
        # the collider section below: when severity is itself the
        # variable being classified and its arrows go OUT to T and Y,
        # the role is confounder; when severity is a PARENT of a
        # downstream count/binary variable V (with arrows from severity
        # AND from T converging into V), V is the collider.
        # Without this exemplar the LM has no positive example of
        # "pre-index severity = confounder", risking spurious collider
        # labels on legitimate severity confounders after seeing the 4
        # collider examples that all name severity as a parent of V.
        dspy.Example(
            feature_name="baseline_severity_score_preindex",
            derivation_pseudocode=(
                "weighted sum of pre-index diagnoses + lab abnormalities + "
                "prior medication intensity WHERE event_date < index_date - 30d "
                "(strict pre-index window with a 30-day washout buffer)"
            ),
            dataset_context=(
                "Optum/CSU claims; target=initiated_biologic_180d; "
                "anchor=index_date; STRICT pre-index temporal filter; severity "
                "score is a composite of baseline disease activity"
            ),
            causal_role="confounder",
            mechanism=(
                "Pre-index severity score is a CONFOUNDER: arrows go OUT of "
                "severity into BOTH T (severity -> T via prescriber decision "
                "to escalate to biologic) and Y (severity -> Y via "
                "uncontrolled disease activity). The variable itself has no "
                "incoming arrowheads from T or downstream events because "
                "the derivation strictly filters to pre-index. Severity is "
                "NOT a collider: collider DAG requires arrowheads INTO V, "
                "but severity's only arrowheads are OUTGOING. Standard "
                "remediation is keep_with_caveat (condition on severity in "
                "downstream models to close the backdoor T <- severity -> Y "
                "path; this is the canonical confounder-adjustment use case)."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # Issue #198 — collider exemplars (4)
        # ---------------------------------------------------------------------
        # A COLLIDER (Pearl, `Causality` 2009) is any variable V with TWO
        # ARROWHEADS pointing into it from distinct sources. Conditioning
        # on V opens a non-causal path between V's parents. Two important
        # sub-families:
        #
        # (i)  CLASSIC BERKSON COLLIDER: V <- T and V <- X with X causally
        #      INDEPENDENT of T. Conditioning opens an artificial T-X
        #      dependence in the conditioned subpopulation.
        #
        # (ii) CONFOUNDER-COLLIDER (confounded-collider, "M-structure"
        #      collider per Greenland-Pearl-Robins 1999): V <- T and
        #      V <- X where X also CONFOUNDS T-Y (i.e., X -> T and
        #      X -> Y). V still has two arrowheads in, so it remains a
        #      collider. Conditioning on V opens a backdoor X <-> Y path
        #      and is the dominant collider failure mode in observational
        #      pharmacoepi (baseline severity is the canonical X).
        #
        # The four exemplars below are confounder-collider (type ii). The
        # rationale in each example NAMES the structure explicitly so the
        # LM does not learn "Berkson-only" framing and reject realistic
        # pharmacoepi colliders. Each derivation exposes BOTH parents in
        # the data: a PRE-INDEX activity / severity stream AND a POST-T
        # AE / non-response stream feeding the same total.
        #
        # IMPORTANT codex pass-1 correction (preserved): post-treatment
        # outcome flags whose derivation reduces to `T AND (post-T event)`
        # (e.g., `discontinuation_flag = T AND (gap_in_fills)`) are
        # DESCENDANTS, not colliders — the "second arrow" they appeal to
        # is itself a consequence of T, not an independent or confounding
        # cause.
        #
        # Remediation is `drop` because conditioning on a collider — of
        # EITHER sub-family — induces selection bias on the T->Y
        # relationship.
        # =====================================================================
        # Collider 1 — hospitalizations_total (unwindowed): the canonical
        # CONFOUNDER-COLLIDER in pharmacoepi (sub-family ii). Two arrows
        # in from distinct sources: (a) baseline disease severity (which
        # is itself a T-Y CONFOUNDER — severity -> T via prescriber
        # decision, severity -> Y via uncontrolled disease activity) and
        # (b) treatment-related adverse events (which is a path from T:
        # T -> AE -> hospitalization). V = hospitalizations_total has
        # arrowheads from BOTH severity and AE. Conditioning on V opens
        # the severity <-> Y backdoor that the unconditioned (T, severity,
        # Y) graph leaves blocked.
        dspy.Example(
            feature_name="hospitalizations_total",
            derivation_pseudocode=(
                "count(encounter_events where admit_date in [journey_start, "
                "journey_end]) — NO pre-index temporal filter; admissions "
                "include both pre-index severity-driven admits AND post-index "
                "biologic-AE-driven admits"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; "
                "encounter_events panel; admissions can be driven by "
                "pre-index severity OR by post-treatment adverse events"
            ),
            causal_role="collider",
            mechanism=(
                "Confounder-collider (sub-family ii per Greenland-Pearl-"
                "Robins 1999 M-structure): two arrowheads in from distinct "
                "sources. (a) Baseline disease severity is a T-Y confounder "
                "(severity -> T via prescriber decision; severity -> Y via "
                "uncontrolled disease activity) and ALSO has an arrow into "
                "the hospitalization count (severity drives admissions). "
                "(b) Treatment-related adverse events form a path T -> AE "
                "-> hospitalization. V has arrowheads from both severity "
                "and AE, so V is a collider on the (severity, T) backdoor. "
                "Conditioning on V opens the severity <-> Y path, biasing "
                "the treatment-effect estimate. (NOTE: a properly windowed "
                "pre-index hospitalization count collapses to a pure "
                "confounder with NO post-index arrow; the collider semantics "
                "depend on the derivation crossing the index date.)"
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 2 — concomitant_steroid_burst_count_followup: post-
        # index rescue-therapy count. Cleaner confounder-collider example
        # than the parallel ER/admission counts because the two arrowheads
        # come from CAUSALLY DIFFERENT mechanisms: (a) baseline disease
        # severity (a T-Y confounder; severity -> T; severity -> Y; severity
        # -> bursts) and (b) biologic-treatment NON-RESPONSE — non-response
        # is a path from T but NOT driven by severity (a non-responding
        # biologic patient gets bursts whether or not their baseline
        # severity was high or low).
        dspy.Example(
            feature_name="concomitant_steroid_burst_count_followup",
            derivation_pseudocode=(
                "count(medication_events where med_class='oral_steroid_burst' AND "
                "fill_date in [index_date, index_date+180d]) — driven by "
                "both pre-index severity and post-treatment non-response"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; "
                "medication panel; oral steroid burst is a rescue therapy that "
                "fires from both baseline severity and biologic non-response"
            ),
            causal_role="collider",
            mechanism=(
                "Confounder-collider on the rescue-therapy count: V has "
                "two arrowheads from distinct sources. (a) Baseline disease "
                "severity is a classical T-Y confounder (severity -> T via "
                "prescriber escalation; severity -> Y via uncontrolled "
                "disease) AND drives bursts (severity -> V) for patients "
                "with high baseline activity regardless of biologic "
                "exposure. (b) Biologic-treatment non-response forms a "
                "path T -> non-response -> burst-prescription. Non-response "
                "is downstream of T but is NOT itself caused by baseline "
                "severity in a deterministic way (responders and non-"
                "responders are distributed across the severity spectrum). "
                "V has arrowheads from both severity (confounder) and "
                "non-response (T-driven). Conditioning on V opens the "
                "severity <-> Y backdoor."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 3 — alive_at_180d_observation_window: SAMPLE-SELECTION
        # confounder-collider (sub-family ii). V = patient alive AND
        # continuously enrolled at index+180d, used as a complete-
        # follow-up sample-inclusion filter. Two arrowheads from distinct
        # mechanism families: (a) baseline mortality/dropout risk (a
        # T-Y confounder via the severity-of-illness pathway — sicker
        # patients are less likely to be prescribed biologics AND less
        # likely to survive; this is the standard pharmacoepi reality)
        # and (b) T -> survival -> V (treatment itself affects survival).
        # Distinct from the utilization/medication/workup counts above:
        # V here is a BINARY SAMPLE-INCLUSION indicator, not a count.
        # Conditioning on V=1 (the common complete-follow-up filter) is
        # what causes the M-bias. Pinned separately so the LM learns the
        # confounder-collider pattern transfers across BOTH count
        # features AND binary inclusion-filter features.
        dspy.Example(
            feature_name="alive_at_180d_observation_window",
            derivation_pseudocode=(
                "(death_date IS NULL OR death_date > index_date + 180d) AND "
                "(enrollment_active(index_date + 180d) = 1)  # binary "
                "complete-follow-up flag used as a SAMPLE-INCLUSION filter"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; "
                "death + enrollment panels; analysts commonly filter to V=1 "
                "to get complete-follow-up cohorts"
            ),
            causal_role="collider",
            mechanism=(
                "Sample-selection confounder-collider (sub-family ii). V "
                "has two arrowheads. (a) Baseline mortality/dropout risk "
                "(age, comorbidity burden, frailty) is a T-Y confounder "
                "via the severity-of-illness pathway (severity -> T via "
                "prescriber declining biologics for frail patients; "
                "severity -> Y via uncontrolled disease) AND drives the "
                "survival/enrollment arrow into V. (b) T -> survival -> V: "
                "the treatment itself affects mortality and dropout, so T "
                "has an arrow into V independent of the baseline-risk arm. "
                "When the analysis is RESTRICTED to V=1 patients (the "
                "common complete-follow-up filter), conditioning opens "
                "the T <-> baseline-risk backdoor on the Y path. "
                "Structurally distinct from the count colliders above "
                "because V is BINARY and used as a sample-inclusion gate."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 4 — diagnostic_test_count_followup: diagnostic-workup
        # test count in the post-index window. Confounder-collider
        # (sub-family ii) with a mechanically different second-parent
        # path than the utilization-based hospitalization/burst counts:
        # the post-T arrow is AE-specific monitoring, not severity-
        # driven utilization. Pinned separately so the LM learns the
        # pattern transfers across COUNT-feature derivations (utilization
        # vs workup vs medication).
        dspy.Example(
            feature_name="diagnostic_test_count_followup",
            derivation_pseudocode=(
                "count(lab_events U procedure_events where category='diagnostic' "
                "AND date in [index_date, index_date+180d]) — driven by both "
                "pre-index workup intensity and post-treatment AE-specific monitoring"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; "
                "lab+procedure panels filtered to diagnostic codes; pre-index "
                "severity drives workup, post-T monitoring fires conditional on T"
            ),
            causal_role="collider",
            mechanism=(
                "Confounder-collider on the workup count: V has two "
                "arrowheads. (a) Baseline disease activity / diagnostic "
                "uncertainty (a T-Y confounder via severity-of-illness; "
                "drives both the prescriber's T decision AND the workup "
                "stream). (b) Treatment-specific monitoring forms a path "
                "T -> protocol-driven labs -> V (e.g., LFT/CBC monitoring "
                "in biologic protocols; the test request comes from the "
                "treatment protocol, not from underlying disease activity). "
                "Conditioning on the total count opens the activity <-> Y "
                "backdoor."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # Issue #198 — instrument exemplars (4)
        # ---------------------------------------------------------------------
        # An INSTRUMENT (instrumental variable, IV) is a variable Z such that:
        # (i)  Z -> T (the instrument affects the treatment), AND
        # (ii) Z -> Y only through T (exclusion restriction — no direct effect
        #      on the outcome bypassing treatment), AND
        # (iii) Z is unconfounded with Y (no common cause of Z and Y).
        #
        # In pharmacoepi the canonical IV families are (a) preference-based
        # provider-level IVs (Brookhart et al. 2006, "Preference-Based
        # Instrumental Variable Methods for the Estimation of Treatment
        # Effects", Stat Med 25:1907; Brookhart and Schneeweiss 2007,
        # "Evaluating Short-Term Drug Effects Using a Physician-Specific
        # Prescribing Preference as an IV", Epidemiology 18:589) and
        # (b) supply-side geographic-access IVs (regional formulary
        # heterogeneity, RUCA-tier specialist density).
        #
        # Critical codex pass-1 framing correction: the exclusion-restriction
        # rationale below is phrased as an ASSUMPTION-TO-AUDIT, NOT as a
        # confessed violation. The compile set must NOT teach the LM that
        # "the rationale already acknowledged Z -> Y, so it's still an
        # instrument" — that would corrupt the role discrimination. Each
        # rationale instead reads: "exclusion holds under the standard
        # IV assumption; verification step X must be run before relying on
        # this Z; refuse the IV interpretation if X fails." This matches
        # the standard pharmacoepi IV-validity audit (Brookhart 2010,
        # "Instrumental Variable Analysis"; Garabedian et al. 2014, "Potential
        # Bias of Instrumental Variable Analyses for Observational
        # Comparative Effectiveness Research", Ann Intern Med).
        #
        # Remediation: `keep_with_caveat` because instruments are LEGITIMATE
        # pre-index features useful for causal identification when the
        # exclusion restriction holds; they shouldn't be dropped reflexively.
        # =====================================================================
        # Instrument 1 — urban_rural_code: classic geographic IV per
        # Brookhart-style pharmacoepi. Rural vs urban code reflects
        # supply-side access to specialist-driven biologic prescribing
        # without directly influencing disease biology. Static at
        # enrollment. Exclusion restriction is the standard assumption to
        # audit, not a confessed violation.
        dspy.Example(
            feature_name="urban_rural_code",
            derivation_pseudocode=("rural_urban_commuting_area_code(zip3)  # static at enrollment"),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; "
                "geographic feature derived from zip3; static at enrollment"
            ),
            causal_role="instrument",
            mechanism=(
                "Supply-side geographic instrument (Brookhart-style "
                "pharmacoepi IV): RUCA code reflects access to specialist-"
                "driven biologic prescribing (Z -> T arrow via specialist "
                "density and biologic-friendly provider mix). Exclusion "
                "restriction: Z -> Y holds only through T under the "
                "standard IV assumption that geographic classification "
                "does not directly affect disease pathophysiology. "
                "Static at enrollment so no confounding with post-index "
                "outcomes. IV-VALIDITY AUDIT STEP (required before use): "
                "test for direct Z -> Y on a placebo outcome that should "
                "be invariant to T; reject the IV interpretation if the "
                "placebo test fails."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Instrument 2 — geographic_region: 4-region prescribing-intensity
        # IV. Coarse-grained complement to urban_rural_code; ships as a
        # distinct example because the Z -> T mechanism (regional
        # formulary policy) is differently shaped from the RUCA-tier
        # specialist-density mechanism.
        dspy.Example(
            feature_name="geographic_region",
            derivation_pseudocode=("{NE,MW,South,West} from zip3  # static at enrollment"),
            dataset_context=(
                "CSU/Optum claims; target=treatment_initiated; "
                "anchor=index_date; static at enrollment"
            ),
            causal_role="instrument",
            mechanism=(
                "Regional prescribing-rate variation is a canonical "
                "supply-side IV in observational pharmacoepi (Brookhart "
                "et al. 2006). Z -> T arrow: regional formulary policy and "
                "specialist density drive the biologic-vs-non-biologic mix. "
                "Exclusion restriction: Z -> Y holds only through T under "
                "the standard IV assumption that region does not directly "
                "cause disease activity or modify drug pharmacokinetics. "
                "Z is set at enrollment, before treatment decision, so "
                "unconfounded with post-index outcomes. IV-VALIDITY AUDIT "
                "STEP (required before use): verify exchangeability on "
                "pre-index covariates across region levels; if covariate "
                "imbalance is large, condition on the imbalanced covariate "
                "or drop the IV interpretation."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Instrument 3 — provider_preference_score: the canonical Brookhart-
        # Schneeweiss 2007 preference-based IV. Replaces the original `zip3`
        # IV (which was a third nested zip-geography variable, contributing
        # collinear training signal). Provider preference is a DIFFERENT IV
        # family from supply-side geography, broadening the LM's training
        # signal across the two pharmacoepi IV idioms.
        dspy.Example(
            feature_name="provider_preference_score",
            derivation_pseudocode=(
                "fraction(prior_patients_of(index_provider) where "
                "biologic_initiation=1) over the 12 months BEFORE this "
                "patient's index_date; patient excluded from own provider's "
                "denominator"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; "
                "anchor=index_date; provider-level prescribing-pattern feature "
                "derived from index_provider's other patients' prior history"
            ),
            causal_role="instrument",
            mechanism=(
                "Preference-based provider IV per Brookhart-Schneeweiss 2007 "
                "(Epidemiology 18:589). Z -> T arrow: a high-biologic-"
                "preference provider is mechanically more likely to "
                "initiate a biologic for the index patient. Exclusion "
                "restriction: Z -> Y holds only through T under the "
                "standard IV assumption that provider preference acts on "
                "the patient solely via prescribing choice, with no "
                "direct effect on disease biology. Patient excluded from "
                "own denominator so Z is genuinely pre-index. IV-VALIDITY "
                "AUDIT STEP (required before use): test for unbalanced "
                "patient-level baseline covariates across high- vs low-"
                "preference providers; condition on imbalanced baselines "
                "or reject the IV if the imbalance is structural."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Instrument 4 — index_provider_biologic_volume_prior_year:
        # provider-volume IV. Distinct from the preference-fraction IV
        # (which captures prescribing PROPORTION) — this captures the
        # ABSOLUTE VOLUME of biologic initiations the provider performed
        # in the year before the patient's index. High-volume biologic
        # prescribers initiate biologics more readily (operational
        # familiarity, established prior-auth workflows). Replaces the
        # original `plan_type` example because plan_type is functionally
        # equivalent to the existing `insurance_product` confounder
        # exemplar (both enrollment-time payer features that affect
        # outcomes via access/monitoring/SES paths). Codex pass-2 MED-2
        # flagged the contradictory plan_type-vs-insurance_product
        # training signal — this replacement preserves the IV slot
        # without the access/coverage path that breaks exclusion.
        dspy.Example(
            feature_name="index_provider_biologic_volume_prior_year",
            derivation_pseudocode=(
                "count(distinct patients of index_provider where "
                "biologic_initiation_date in [index_date - 365d, index_date - 1d]) "
                "EXCLUDING this patient — provider-level volume in the year "
                "BEFORE this patient's index"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; "
                "anchor=index_date; provider-volume IV measured strictly "
                "pre-index from other patients"
            ),
            causal_role="instrument",
            mechanism=(
                "Provider-volume IV — a VOLUME-based variant of the "
                "Brookhart 2006 / Brookhart-Schneeweiss 2007 preference-"
                "based IV family (the canonical Brookhart 2006 used the "
                "PREFERENCE FRACTION on the previous prescription, not "
                "an absolute count; the volume operationalization "
                "captures the same underlying provider-level supply-side "
                "lever but via raw count rather than ratio). Z -> T "
                "arrow: high-volume biologic prescribers have higher "
                "initiation rates per patient due to operational "
                "familiarity, established prior-auth workflows, and "
                "established formulary navigation. Exclusion restriction: "
                "Z -> Y holds only through T under the standard IV "
                "assumption that the provider's PRIOR volume on OTHER "
                "patients does not directly affect THIS patient's outcome "
                "except through T (the provider's skill / care quality may "
                "violate this assumption — see audit step). Measured "
                "pre-index from other patients so Z is exogenous to this "
                "patient's covariates. IV-VALIDITY AUDIT STEP (required "
                "before use): test for provider-level care-quality "
                "differences in NON-biologic management across volume "
                "tiers; if high-volume providers also differ in non-"
                "biologic care quality, reject the IV interpretation."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # Phase-4 S12 Option C recompile (2026-05-19) — 12 paired (T, Y)-
        # explicit demos. Plan: `.claude/plans/option_c_dspy_recompile_for_s12_FINAL.md`.
        # ---------------------------------------------------------------------
        # These 12 demos teach the classifier to read `treatment={T};
        # outcome={Y}` as first-class semicolon-delimited fields in
        # `dataset_context` and to condition role classification on the
        # explicit (T, Y) pair rather than inferring T implicitly from
        # cohort/target convention. The legacy 21 demos above continue
        # to drive the cohort-only-context production path (the sole
        # production caller at adaptive_validity_check.py:892-893 emits
        # cohort-only contexts); the 12 paired demos below extend the
        # classifier to S12-style callers that supply explicit (T, Y).
        # Backward-compatible by construction: the input field is still
        # `str`; production behavior on cohort-only contexts is pinned
        # by tests/unit/test_data/test_causal_role_classifier.py::
        # test_persisted_artifact_preserves_legacy_demo_roles.
        #
        # Falsifiability design (codex iter-2 redesign): the 12 demos
        # are 6 paired fixtures. Each pair shares ONE feature_name AND
        # ONE derivation_pseudocode, varying only (T, Y) in
        # dataset_context — and the graph-theoretic-correct causal role
        # FLIPS across the variants. This is the strongest possible
        # falsifiability signal: classification cannot be by feature_name
        # alone (else role would not flip); the classifier MUST be
        # reading the (T, Y) fields. The 12 quadruples are pinned by
        # `test_persisted_artifact_emits_role_conditional_on_treatment_outcome`.
        #
        # Pair 4 (baseline_oncotype_dx_recurrence_score) carries a
        # d-separation assumption (Oncotype ⊥ tumor_size | pre-diagnosis
        # covariates) flagged for expert review in Option C plan §3.5
        # + §9. If the assumption is disputed, swap the pair for one
        # with a less-contestable ancestor — see plan §5 row 2 recovery
        # procedure (`git checkout artifacts/dspy/causal_role_classifier.json`,
        # redesign pair, recompile).
        #
        # Pairs 3 and 5 share feature_name with legacy demos
        # (concomitant_steroid_burst_count_followup, provider_preference_score).
        # The shared feature is intentional — same Z, same derivation,
        # the only delta is the dataset_context's (T, Y) fields. This is
        # the cleanest falsifiability anchor: a classifier that ignores
        # (T, Y) would re-emit the legacy role label; a classifier that
        # reads (T, Y) flips correctly.
        # =====================================================================
        # Pair 1a — instrument: provider omalizumab volume IV.
        # Z = high-volume omalizumab prescriber proxy; (T = omalizumab
        # initiation, Y = remission within 180d). Z->T arrow: operational
        # familiarity + established prior-auth workflows drive higher
        # initiation rates. Exclusion restriction (Z->Y only through T)
        # holds under the standard IV assumption that prior-year volume
        # on OTHER patients does not directly affect THIS patient's
        # remission. IV audit step: test for provider-level care quality
        # in non-omalizumab CSU management across volume tiers.
        dspy.Example(
            feature_name="index_provider_omalizumab_volume_prior_year",
            derivation_pseudocode=(
                "count(distinct patients of index_provider where "
                "omalizumab_initiation_date in [index_date - 365d, index_date - 1d]) "
                "EXCLUDING this patient — provider-level omalizumab initiation "
                "volume measured strictly pre-index from OTHER patients"
            ),
            dataset_context=(
                "ConcertAI CSU claims; cohort=CSU; target=remission_180d; "
                "anchor=index_date; treatment=omalizumab_init; outcome=remission_180d"
            ),
            causal_role="instrument",
            mechanism=(
                "Provider-volume IV for treatment=omalizumab_init. Z->T arrow: "
                "high-omalizumab-volume providers initiate omalizumab more "
                "readily for this patient (operational familiarity, prior-auth "
                "workflows). Exclusion restriction: Z->Y holds only through T "
                "under the IV assumption that the provider's PRIOR-year "
                "omalizumab volume on OTHER patients has no direct path to "
                "THIS patient's remission_180d outcome beyond the omalizumab "
                "initiation decision. Z is exogenous (other-patient-derived) "
                "and pre-index. IV-VALIDITY AUDIT STEP: test for provider-"
                "level differences in non-omalizumab CSU management quality "
                "across volume tiers; reject IV interpretation if care quality "
                "differs systematically (the exclusion restriction would then "
                "fail via direct Z -> care quality -> Y path)."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 1b — confounder: same Z, same T, different Y. Now Y =
        # hospitalization_180d. Z -> Y now has a direct path through
        # care-quality / adverse-event management: high-volume providers
        # manage AEs better, reducing hospitalization independent of T
        # assignment. Z has both Z -> T (volume effect) and Z -> Y
        # (care quality), so Z is a confounder of (T, Y_b) rather than
        # an instrument.
        dspy.Example(
            feature_name="index_provider_omalizumab_volume_prior_year",
            derivation_pseudocode=(
                "count(distinct patients of index_provider where "
                "omalizumab_initiation_date in [index_date - 365d, index_date - 1d]) "
                "EXCLUDING this patient — provider-level omalizumab initiation "
                "volume measured strictly pre-index from OTHER patients"
            ),
            dataset_context=(
                "ConcertAI CSU claims; cohort=CSU; target=hospitalization_180d; "
                "anchor=index_date; treatment=omalizumab_init; outcome=hospitalization_180d"
            ),
            causal_role="confounder",
            mechanism=(
                "Confounder of (T=omalizumab_init, Y=hospitalization_180d). "
                "Same Z -> T arrow as Pair 1a (volume drives initiation), but "
                "the exclusion restriction FAILS for this outcome: high-volume "
                "providers also manage CSU adverse events (anaphylaxis, "
                "infusion reactions, secondary infections) more competently, "
                "so Z -> hospitalization runs through a care-quality path "
                "independent of the omalizumab treatment decision. With both "
                "Z -> T and Z -> Y arrows present, Z is a confounder of the "
                "(T, Y_b) relationship, NOT an instrument. Standard remediation "
                "is condition on Z in downstream models to close the backdoor "
                "T <- Z -> Y_b path."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 2a — mediator: acute kidney injury post-index on the
        # T -> AKI -> CV-death path. ACE inhibitors are known to induce
        # AKI in susceptible patients (renal-dose-response mechanism);
        # AKI is then on the path from ACE -> AKI -> cardiovascular
        # death. Standard mediation pattern.
        dspy.Example(
            feature_name="acute_kidney_injury_event_count_followup",
            derivation_pseudocode=(
                "count(diagnosis_events where dx_code in ['N17.x'] AND "
                "event_date in [index_date, index_date + 5y]) — followup "
                "AKI count; no temporal exclusion of post-T events"
            ),
            dataset_context=(
                "Optum claims; cohort=Hypertension; target=cv_death_5y; "
                "anchor=index_date; treatment=ace_inhibitor_init; outcome=cv_death_5y"
            ),
            causal_role="mediator",
            mechanism=(
                "Mediator on the (T=ace_inhibitor_init, Y=cv_death_5y) path. "
                "ACE inhibitors induce acute kidney injury in susceptible "
                "patients via afferent-arteriole vasodilation (renal-dose-"
                "response mechanism). AKI events post-T then mediate "
                "cardiovascular outcomes: T -> AKI -> renal compromise -> "
                "cardiac strain -> Y. AKI is structurally ON the causal path "
                "from T to Y, not a confounder or descendant. Standard "
                "mediation remediation is windowing (use pre-index AKI "
                "history as a confounder; exclude post-T AKI from the model)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 2b — collider: same Z, different (T, Y). Now T = baseline
        # eGFR category (a pre-index renal-function bin) and Y = ACE
        # initiation. AKI in followup is downstream of BOTH baseline
        # renal function (low-eGFR patients have more AKI events) AND
        # downstream of ACE initiation (ACE-induced AKI). Two arrowheads
        # into V from distinct sources = collider on the (T, Y) backdoor.
        dspy.Example(
            feature_name="acute_kidney_injury_event_count_followup",
            derivation_pseudocode=(
                "count(diagnosis_events where dx_code in ['N17.x'] AND "
                "event_date in [index_date, index_date + 5y]) — followup "
                "AKI count; no temporal exclusion of post-T events"
            ),
            dataset_context=(
                "Optum claims; cohort=Hypertension; target=ace_inhibitor_init; "
                "anchor=index_date; treatment=baseline_egfr_category; "
                "outcome=ace_inhibitor_init"
            ),
            causal_role="collider",
            mechanism=(
                "Confounder-collider (M-structure per Greenland-Pearl-Robins "
                "1999) on the (T=baseline_egfr_category, Y=ace_inhibitor_init) "
                "relationship. AKI in followup has TWO arrowheads from "
                "distinct sources: (a) baseline renal function (low-eGFR "
                "patients have more spontaneous AKI; baseline_egfr -> AKI) "
                "AND (b) any ACE initiation that follows (T -> AKI). V has "
                "arrowheads from both T and a downstream/confounding source, "
                "so V is a collider on the (T, Y) backdoor. Conditioning on "
                "V opens a non-causal egfr <-> ace-init path. Drop the "
                "feature; do not condition on it for this (T, Y) analysis."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 3a — collider: SAME feature as legacy collider
        # `concomitant_steroid_burst_count_followup`, but now with
        # explicit (T = biologic_init, Y = hospitalization_180d). Same
        # confounder-collider pattern as the legacy demo, but the (T, Y)
        # fields make the role inference explicit-rather-than-implicit.
        # Falsifiability for the new schema: a classifier that ignores
        # (T, Y) would re-emit the legacy role (`collider`) by feature_name
        # alone — but the EXPLICIT (T, Y) here happens to agree, so role
        # remains collider. Pair 3b (below) flips the role with a
        # different T, proving the classifier reads (T, Y).
        dspy.Example(
            feature_name="concomitant_steroid_burst_count_followup",
            derivation_pseudocode=(
                "count(medication_events where med_class='oral_steroid_burst' AND "
                "fill_date in [index_date, index_date+180d]) — driven by "
                "both pre-index severity and post-treatment non-response"
            ),
            dataset_context=(
                "Optum claims; cohort=CSU; target=hospitalization_180d; "
                "anchor=index_date; treatment=biologic_init; outcome=hospitalization_180d"
            ),
            causal_role="collider",
            mechanism=(
                "Confounder-collider for (T=biologic_init, Y=hospitalization_180d). "
                "Two arrowheads in from distinct sources: (a) baseline disease "
                "severity is a (T, Y) confounder (severity -> T via prescriber "
                "escalation to biologic; severity -> Y via uncontrolled disease "
                "activity) AND drives bursts. (b) Biologic non-response forms "
                "the path T -> non-response -> burst-prescription. V has "
                "arrowheads from both severity (confounder) and non-response "
                "(T-driven). Conditioning on V opens the severity <-> Y "
                "backdoor for this specific (T, Y) pair. Drop the feature."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 3b — descendant: SAME feature, NEW T. When T_b is the
        # steroid-burst policy indicator itself (a payer/protocol policy
        # that drives steroid burst rates), the steroid-burst count V
        # is downstream of T_b by definition. No second arrowhead from
        # an independent source — V is a pure descendant of T_b on the
        # T_b -> V path. Different role than Pair 3a despite the same
        # feature + same derivation, proving (T, Y) drives the
        # classification.
        dspy.Example(
            feature_name="concomitant_steroid_burst_count_followup",
            derivation_pseudocode=(
                "count(medication_events where med_class='oral_steroid_burst' AND "
                "fill_date in [index_date, index_date+180d]) — driven by "
                "both pre-index severity and post-treatment non-response"
            ),
            dataset_context=(
                "Optum claims; cohort=CSU; target=biologic_init; "
                "anchor=index_date; treatment=steroid_burst_policy_indicator; "
                "outcome=biologic_init"
            ),
            causal_role="descendant",
            mechanism=(
                "Descendant of T=steroid_burst_policy_indicator. The policy "
                "indicator (e.g., payer-mandated step-therapy requiring a "
                "documented steroid-burst failure before authorizing biologic) "
                "directly causes steroid-burst prescriptions. V is downstream "
                "of T_b on the T_b -> V path, with no independent second "
                "parent for this (T_b, Y_b) framing. Differs from Pair 3a's "
                "collider role because T_b is now the policy that drives V "
                "rather than the biologic decision that converges on V with "
                "severity. Window or drop the feature; do not use it as a "
                "covariate in (T_b, Y_b) analysis."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 4a — confounder: baseline Oncotype DX recurrence score on
        # (T = cdk46i_init, Y = recurrence_5y). Oncotype score is measured
        # at diagnosis (pre-index) and drives BOTH the cdk4/6 inhibitor
        # decision (high-score patients are escalated) AND the recurrence
        # outcome (high-score patients recur more from baseline biology).
        # Classic (T, Y) confounder.
        dspy.Example(
            feature_name="baseline_oncotype_dx_recurrence_score",
            derivation_pseudocode=(
                "Oncotype DX 21-gene recurrence score from tumor RNA "
                "expression panel; measured at diagnosis BEFORE index_date "
                "(pre-index, used in treatment-decision pathway)"
            ),
            dataset_context=(
                "ConcertAI Breast Cancer claims; cohort=HR+_BreastCancer; "
                "target=recurrence_5y; anchor=index_date; treatment=cdk46i_init; "
                "outcome=recurrence_5y"
            ),
            causal_role="confounder",
            mechanism=(
                "Classical (T=cdk46i_init, Y=recurrence_5y) confounder. "
                "Z -> T arrow: high Oncotype score (>25) escalates patients "
                "to CDK4/6 inhibitor combination therapy per NCCN guidelines. "
                "Z -> Y arrow: high Oncotype score reflects tumor biology "
                "(proliferation, ER/PR status, HER2 expression) directly "
                "associated with recurrence risk independent of treatment. "
                "Z is measured at diagnosis (pre-index) so arrows are "
                "outgoing only; not a collider. Standard remediation: "
                "condition on Z in causal estimation to close the backdoor "
                "T <- Z -> Y path."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 4b — ancestor: SAME Z, NEW (T, Y). When T_b is tumor size
        # at diagnosis (an earlier-in-the-causal-chain measurement) and
        # Y_b is cdk46i_init (a downstream treatment decision), Oncotype
        # is upstream of cdk46i_init (Z causes T_b via downstream
        # treatment-decision algorithm). The ancestor claim requires Z⊥T_b
        # given the pre-diagnosis covariate set (patient demographics,
        # screening history). PLAN §3.5 + §9: this d-separation assumption
        # is flagged for expert review; Oncotype is computed from tumor
        # biology and could share unobserved upstream tumor-genomic causes
        # with tumor size, making Z a confounder rather than ancestor of
        # (T_b, Y_b). Recovery procedure: swap Pair 4 for a cleaner
        # ancestor (e.g., provider-specialty indicator) if domain expert
        # rejects the d-separation assumption.
        dspy.Example(
            feature_name="baseline_oncotype_dx_recurrence_score",
            derivation_pseudocode=(
                "Oncotype DX 21-gene recurrence score from tumor RNA "
                "expression panel; measured at diagnosis BEFORE index_date "
                "(pre-index, used in treatment-decision pathway)"
            ),
            dataset_context=(
                "ConcertAI Breast Cancer claims; cohort=HR+_BreastCancer; "
                "target=cdk46i_init; anchor=index_date; treatment=tumor_size_at_diagnosis; "
                "outcome=cdk46i_init"
            ),
            causal_role="ancestor",
            mechanism=(
                "Ancestor of (T_b=tumor_size_at_diagnosis, Y_b=cdk46i_init) "
                "under the d-separation assumption Oncotype ⊥ tumor_size | "
                "{pre-diagnosis demographics, screening history}. Oncotype "
                "is biology-derived from tumor RNA expression (proliferation "
                "markers, ER/PR pathway); tumor_size is morphologic. Both "
                "feed into the cdk46i treatment-decision algorithm but the "
                "biology-vs-morphology streams are conditionally independent "
                "given the dataset's standard covariate set. NOTE (Option C "
                "plan §3.5 + §9): if domain reviewer disputes the "
                "conditional-independence assumption (e.g., upstream tumor-"
                "genomic causes link RNA expression and tumor size), Z would "
                "become a confounder of (T_b, Y_b) rather than ancestor; "
                "swap this pair per plan §5 risk row 2."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 5a — instrument: SAME feature as legacy
        # `provider_preference_score`, NEW explicit (T, Y) = (biologic_init,
        # remission_180d). Same preference-based IV pattern as the legacy
        # demo, but with explicit (T, Y) fields making the role assignment
        # context-aware. Role agrees with legacy (instrument) because the
        # (T, Y) framing matches the legacy's implicit biologic + remission
        # framing. Pair 5b (below) flips the role.
        dspy.Example(
            feature_name="provider_preference_score",
            derivation_pseudocode=(
                "fraction(prior_patients_of(index_provider) where "
                "biologic_initiation=1) over the 12 months BEFORE this "
                "patient's index_date; patient excluded from own provider's "
                "denominator"
            ),
            dataset_context=(
                "Optum claims; cohort=CSU; target=remission_180d; "
                "anchor=index_date; treatment=biologic_init; outcome=remission_180d"
            ),
            causal_role="instrument",
            mechanism=(
                "Preference-based provider IV for (T=biologic_init, "
                "Y=remission_180d) per Brookhart-Schneeweiss 2007. Z -> T "
                "arrow: high-preference providers initiate biologics more "
                "frequently. Exclusion restriction: Z -> Y holds only through "
                "T under the standard IV assumption that provider preference "
                "acts on remission solely via the biologic prescribing "
                "decision, with no direct effect on disease biology. Patient "
                "excluded from own denominator so Z is genuinely pre-index "
                "and exogenous. IV-VALIDITY AUDIT STEP: test for unbalanced "
                "patient-level baseline covariates across high- vs low-"
                "preference providers."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 5b — mediator: SAME feature, NEW T. When T_b is provider's
        # geographic region (a system-level upstream variable), the
        # preference score is downstream of region (region -> provider
        # mix -> preference) AND upstream of biologic_init (preference ->
        # init). Provider preference is on the textbook region -> provider
        # -> prescribing mediator path — NOT a descendant of T_b alone
        # but a true mediator on the (T_b, Y_b) path. Codex iter-2 caught
        # the v2 "descendant" label as wrong; mediator is correct because
        # preference has an arrow OUT to Y_b (biologic_init).
        dspy.Example(
            feature_name="provider_preference_score",
            derivation_pseudocode=(
                "fraction(prior_patients_of(index_provider) where "
                "biologic_initiation=1) over the 12 months BEFORE this "
                "patient's index_date; patient excluded from own provider's "
                "denominator"
            ),
            dataset_context=(
                "Optum claims; cohort=CSU; target=biologic_init; "
                "anchor=index_date; treatment=provider_geographic_region; "
                "outcome=biologic_init"
            ),
            causal_role="mediator",
            mechanism=(
                "Mediator on the (T_b=provider_geographic_region, "
                "Y_b=biologic_init) path. Geographic region determines "
                "provider mix (regional formulary policy, specialist "
                "density, payer-mix-driven prescribing norms), which "
                "shapes per-provider biologic preference, which directly "
                "drives biologic initiation. The textbook region -> "
                "provider-attribute -> prescribing pathway places "
                "preference ON the causal path from T_b to Y_b. Codex "
                "iter-2 correction: this is mediator, not descendant — "
                "the feature has an outgoing arrow to Y_b. Standard "
                "mediation remediation is windowing to use only pre-T_b "
                "provider attributes (but T_b here is exogenous region, "
                "so windowing is trivially satisfied)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 6a — confounder: prior treatment count pre-index on
        # (T = biologic_init, Y = remission_180d). More prior failed
        # treatments drives both the biologic escalation decision (T)
        # AND lower remission rates (Y) via treatment-refractory biology.
        # Classical (T, Y) confounder.
        dspy.Example(
            feature_name="prior_treatment_count_preindex",
            derivation_pseudocode=(
                "count(distinct medication_events where therapy_line < "
                "index_therapy_line AND fill_date < index_date) — pre-index "
                "count of distinct prior therapy lines, strictly pre-index"
            ),
            dataset_context=(
                "Optum claims; cohort=CSU; target=remission_180d; "
                "anchor=index_date; treatment=biologic_init; outcome=remission_180d"
            ),
            causal_role="confounder",
            mechanism=(
                "Classical (T=biologic_init, Y=remission_180d) confounder. "
                "Z -> T arrow: patients with more failed prior treatments are "
                "escalated to biologic per step-therapy protocols. Z -> Y "
                "arrow: more prior failures reflects treatment-refractory "
                "underlying disease biology, depressing remission rates "
                "regardless of current treatment. Z is pre-index (strict "
                "fill_date < index_date filter) so arrows are outgoing only; "
                "not a collider. Remediation: condition on Z in causal "
                "estimation to close the backdoor T <- Z -> Y path."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Pair 6b — mediator: SAME feature, NEW T. When T_b is time since
        # diagnosis (a temporal upstream variable), prior treatment count
        # is downstream of time (longer disease duration -> more
        # treatment attempts) AND upstream of biologic_init (more prior
        # attempts -> biologic eligibility -> biologic_init). The
        # T_b -> Z -> Y_b path makes Z a mediator. Codex iter-3 noted
        # partial mediation is still mediation; no swap needed.
        dspy.Example(
            feature_name="prior_treatment_count_preindex",
            derivation_pseudocode=(
                "count(distinct medication_events where therapy_line < "
                "index_therapy_line AND fill_date < index_date) — pre-index "
                "count of distinct prior therapy lines, strictly pre-index"
            ),
            dataset_context=(
                "Optum claims; cohort=CSU; target=biologic_init; "
                "anchor=index_date; treatment=time_since_diagnosis_years; "
                "outcome=biologic_init"
            ),
            causal_role="mediator",
            mechanism=(
                "Mediator on the (T_b=time_since_diagnosis_years, "
                "Y_b=biologic_init) path. Longer disease duration causally "
                "drives more prior treatment attempts (more time = more "
                "step-therapy trials), and accumulated prior failures drive "
                "step-therapy eligibility for biologic. The T_b -> Z -> Y_b "
                "path places Z structurally on the mediation pathway. "
                "Codex iter-3 noted: partial mediation (some of T_b's "
                "effect on Y_b runs through other paths like disease "
                "severity progression) is still mediation. Standard "
                "remediation is windowing — but since Z is already strictly "
                "pre-index by construction, the active concern is whether "
                "to include Z in the (T_b -> Y_b) effect estimate (excludes "
                "the mediated effect)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
    ]
    return examples


def get_compile_set_summary() -> dict[str, Any]:
    """Summary statistics for the compile set."""
    examples = build_compile_set()
    role_counts: dict[str, int] = {}
    for ex in examples:
        role_counts[ex.causal_role] = role_counts.get(ex.causal_role, 0) + 1
    return {
        "n_examples": len(examples),
        "role_distribution": role_counts,
    }
