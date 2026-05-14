"""Layer 4 — DSPy-compiled CausalRoleClassifier.

Classifies a proposed feature derivation as ancestor / confounder / mediator /
collider / descendant / instrument with respect to the prediction target.
Compiled on a curated subset (20 of 18+) of documented past-leakage incidents
from this codebase's history (`.claude/state/leakage_compile_set_20260507.md`)
plus 8 domain-expert collider / instrument exemplars added under issue #198.

Compile-set role coverage: ancestor=1, confounder=2, mediator=1, descendant=8,
collider=4, instrument=4. All six declared ``CausalRole`` Literal values are
represented (previously: 4 of 6; collider + instrument were deferred to issue
#198 pending domain-expert labeling). The collider examples emphasize the
*Berkson-bias DAG structure* with TWO DISTINCT CAUSAL PARENTS — typically
(baseline disease severity / activity, treatment-related adverse event). That
is what separates a collider from a descendant: a descendant has only one
arrow in (T -> V); a collider has T -> V AND X -> V where X is causally
independent of T (e.g., pre-index severity). This compile set explicitly
rejects "T AND (post-T event)" framings (e.g., discontinuation_flag =
treatment_initiated AND fill-gap) as colliders because the second "arrow"
there is itself a downstream of T, not an independent cause. The instrument
examples follow the canonical pharmacoepidemiology IV families: Brookhart
preference-based provider IVs and supply-side geographic IVs. Each instrument
rationale phrases the exclusion restriction as an ASSUMPTION TO AUDIT (with
a named validity-check step), not as a confessed violation — so the LM
learns IV-validity discipline rather than a "name a violation and still call
it an IV" pattern.

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
    """Build the DSPy compile set: 20 curated examples covering all 6 roles.

    Of the 18 incidents catalogued at
    ``.claude/state/leakage_compile_set_20260507.md``, 12 have been distilled
    into typed ``dspy.Example`` objects below (the remaining 6 are either
    duplicates of an already-represented mechanism or have been folded into
    the broader exemplars). 8 additional ``collider`` and ``instrument``
    exemplars were added under issue #198 from domain-expert review.

    The 4 collider examples are Berkson-bias structures with two DISTINCT
    causal parents: hospitalizations_total / er_visit_count_followup /
    concomitant_steroid_burst_count_followup /
    diagnostic_test_count_followup. Each rationale exposes BOTH parents in
    the derivation (pre-index severity AND post-index treatment-AE).
    `discontinuation_flag` / `discontinued_180d` / `persistent_at_180d` are
    intentionally NOT used as colliders because their derivation reduces to
    ``T AND (post-T event)`` — the second "arrow" there is itself
    downstream of T, making them descendants, not colliders.

    The 4 instrument examples are the canonical pharmacoepi IV pattern:
    urban_rural_code / geographic_region / provider_preference_score /
    plan_type. Each rationale phrases the exclusion restriction as an
    ASSUMPTION-TO-AUDIT with a named validity check step (per Brookhart
    2010 IV-validity audit), NOT as a confessed Z -> Y violation. The
    LM thereby learns to FLAG candidate IVs whose exclusion restriction
    has not been validated, rather than to accept "exclusion violation
    named -> still an IV" as a pattern.

    Coverage by role: ancestor=1, confounder=2, mediator=1, descendant=8,
    collider=4, instrument=4.

    Source: .claude/state/leakage_compile_set_20260507.md + issue #198.
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
        # =====================================================================
        # Issue #198 — collider exemplars (4)
        # ---------------------------------------------------------------------
        # A COLLIDER is a variable V with TWO ARROWS POINTING INTO IT from
        # DISTINCT causes (Pearl, `Causality` 2009): V <- T and V <- X
        # where X is a causal factor INDEPENDENT of T (not merely a
        # downstream property of T). Conditioning on V opens a non-causal
        # backdoor path between T and X. This is what separates a
        # collider from a descendant: descendants have only T -> V (the
        # variable is downstream of treatment); colliders have arrows from
        # T AND from a SECOND, INDEPENDENT cause (typically a pre-treatment
        # baseline characteristic OR an outcome-proxy that is itself
        # caused by something other than T).
        #
        # IMPORTANT codex pass-1 correction: post-treatment outcome flags
        # whose derivation reduces to `T AND (post-T event)` (e.g.,
        # `discontinuation_flag = T AND (gap_in_fills)`) are DESCENDANTS,
        # not colliders — the "second arrow" they appeal to is itself a
        # consequence of T, not an independent cause. The exemplars below
        # are restricted to the canonical Berkson-bias family where the
        # second parent is genuinely INDEPENDENT of T: baseline disease
        # severity (set pre-index) or a non-disease comorbidity flow.
        #
        # Remediation is `drop` because conditioning on a collider induces
        # selection bias on the T->Y relationship — same family of bias
        # as Berkson's bias in case-control studies.
        # =====================================================================
        # Collider 1 — hospitalizations_total (unwindowed): the canonical
        # Berkson-bias collider in pharmacoepi. Two distinct parents made
        # explicit in the derivation: (a) baseline disease severity
        # (PRE-index, independent of T at the patient-level; influences
        # both T-selection AND admission probability), (b) treatment-
        # related adverse events (POST-T, but caused by T not by
        # severity). Hospitalizations of either parent type both count
        # into the same total. Conditioning opens the
        # severity <-> AE backdoor.
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
                "Berkson-style two-arrow-in collider with DISTINCT parents: "
                "(a) baseline disease severity (set pre-index, drives the "
                "treatment-selection arm AND independently drives admissions "
                "via disease activity), and (b) treatment-related adverse "
                "events (post-initiation biologic AE causally drives "
                "admissions on a path NOT mediated by severity). Severity "
                "and AE are causally distinct sources. Conditioning on the "
                "total opens a non-causal severity <-> AE path. (NOTE: a "
                "properly windowed pre-index hospitalization count collapses "
                "to a confounder; the collider semantics depend on the "
                "derivation crossing the index date.)"
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 2 — er_visit_count_followup: same two-distinct-parent
        # structure as hospitalizations_total but at the ER-visit level
        # (POS-23 in Optum inpatient panel). Pre-index severity AND
        # treatment AE both drive ER admissions. Pinned separately so the
        # LM learns the Berkson-bias pattern transfers across encounter
        # types (inpatient vs ER vs observation).
        dspy.Example(
            feature_name="er_visit_count_followup",
            derivation_pseudocode=(
                "count(inpatient_events where POS=23 AND admit_date in "
                "[index_date, index_date+180d]) — both pre-index severity "
                "and post-treatment AE feed this stream"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; "
                "inpatient panel filtered to ER place-of-service; window "
                "includes both severity-driven and AE-driven admissions"
            ),
            causal_role="collider",
            mechanism=(
                "ER-visit Berkson collider: (a) baseline disease severity "
                "(pre-index, independent of T; drives ER visits via "
                "uncontrolled disease activity), and (b) treatment-related "
                "adverse events (post-initiation biologic AE such as "
                "anaphylaxis/injection-site reactions drives ER visits "
                "via a path NOT mediated by severity). The two parents "
                "are causally distinct (severity is not caused by T; AE "
                "is caused by T but not by severity). Conditioning on the "
                "total ER count opens the severity <-> AE backdoor."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 3 — concomitant_steroid_burst_count_followup: post-
        # index steroid-burst prescription count. Two distinct parents:
        # (a) baseline disease severity (drives rescue therapy
        # independently of biologic exposure — even untreated severe
        # patients receive bursts), (b) biologic-treatment failure
        # (post-T; non-responders receive more bursts). Severity and
        # biologic-response are causally distinct sources.
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
                "Steroid-burst rescue is a Berkson-bias collider with two "
                "DISTINCT parents: (a) baseline disease severity (pre-index, "
                "independent of biologic exposure; severe patients receive "
                "bursts regardless of biologic), (b) biologic-treatment "
                "non-response (post-T, biologic-induced; non-responders "
                "escalate to bursts). The two arrows feed the same count "
                "stream but originate in independent causal factors. "
                "Conditioning on the burst count opens severity <-> "
                "non-response, biasing the treatment-effect estimate."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 4 — diagnostic_test_count_followup: diagnostic-workup
        # test count in the post-index window. Two distinct parents:
        # (a) baseline disease activity / diagnostic uncertainty (pre-
        # index, drives ongoing workup independently of T), (b) treatment-
        # related adverse events (post-T, drives AE-specific lab/imaging
        # workup). Severity and AE are causally distinct, but both feed
        # the test-count total.
        dspy.Example(
            feature_name="diagnostic_test_count_followup",
            derivation_pseudocode=(
                "count(lab_events ∪ procedure_events where category='diagnostic' "
                "AND date in [index_date, index_date+180d]) — driven by both "
                "pre-index workup intensity and post-treatment AE workup"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; "
                "lab+procedure panels filtered to diagnostic codes; pre-index "
                "severity and post-T AE both drive test ordering"
            ),
            causal_role="collider",
            mechanism=(
                "Diagnostic-workup count is a Berkson collider: "
                "(a) baseline disease activity / diagnostic uncertainty "
                "(pre-index, independent of T; drives ongoing imaging and "
                "labs to characterize disease), and (b) treatment-related "
                "adverse events (post-T; AE-specific labs such as liver "
                "function for biologic monitoring fire conditional on T). "
                "The two parents are causally distinct: activity is not "
                "caused by T; AE-workup is caused by T but not by baseline "
                "activity. Joint contribution to the same count makes the "
                "total a collider on the T -> Y path."
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
        # Instrument 4 — plan_type: HMO vs PPO vs POS as an IV via step-
        # therapy, biologic-formulary placement, and prior-auth tiering.
        # Distinct mechanism from geography and provider preference.
        dspy.Example(
            feature_name="plan_type",
            derivation_pseudocode=("demo.product  # HMO/PPO/POS/other, set at enrollment"),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; "
                "anchor=index_date; static at enrollment"
            ),
            causal_role="instrument",
            mechanism=(
                "Plan-design IV: HMO vs PPO vs POS varies in step-therapy "
                "requirements, biologic-formulary placement, and prior-auth "
                "tiering. Z -> T arrow: plan design affects whether a "
                "biologic prescription is approved at first attempt. "
                "Exclusion restriction: Z -> Y holds only through T under "
                "the standard IV assumption that plan type does not "
                "directly modify disease biology. Static at enrollment so "
                "unconfounded with post-index outcomes. IV-VALIDITY AUDIT "
                "STEP (required before use): test exchangeability on "
                "patient-level baseline covariates (age, comorbidity, "
                "income proxy) across plan_type levels; condition on or "
                "stratify by any imbalanced baseline characteristic, OR "
                "reject the IV interpretation if a baseline is plausibly "
                "on a Z -> Y path independent of T."
            ),
            recommended_remediation="keep_with_caveat",
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
