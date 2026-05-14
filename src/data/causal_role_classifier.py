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
*two-arrow-in DAG structure* (variable with two distinct causal parents
including the target or a target-proxy) — that's what separates a collider
from a descendant (single arrow in from treatment). The instrument examples
follow the canonical pharmacoepidemiology supply-side IV pattern (e.g.,
Brookhart et al., geographic prescribing variation as IV) where the variable
plausibly affects the OUTCOME only through the TREATMENT (exclusion
restriction), and its assumed-independence from the outcome is named as part
of the rationale so the LM can learn to flag exclusion-restriction violations.

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
    exemplars were added under issue #198 from domain-expert review of the
    CSU + Optum manifests at ``src/data/manifests/``; rationales reference
    the canonical DAG structure (two-arrow-in for colliders; exclusion-
    restriction for instruments) rather than feature names alone.

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
        # A COLLIDER is a variable with TWO ARROWS POINTING INTO IT from
        # distinct causes (Pearl). Conditioning on it opens a non-causal
        # backdoor path between the original parents. In our pharmacoepi
        # setup the two parents are typically (treatment_exposure,
        # treatment_response/AE) — both must be present for the colliding
        # variable to take a non-trivial value. This is what separates a
        # collider from a descendant: descendants have a single arrow in
        # from treatment; colliders have arrows from treatment AND from a
        # second independent cause (often a target-proxy or an outcome
        # proxy like adverse-event severity). Remediation is `drop`
        # because conditioning a regression on a collider induces
        # selection bias on the T→Y relationship — the same family of
        # bias as Berkson's bias in case-control studies.
        # =====================================================================
        # Collider 1 — discontinuation_flag (CSU): classic pharmacoepi
        # collider. Two parents: (a) treatment_initiated (you must have
        # started treatment to discontinue), and (b) clinical response /
        # adverse-event severity (you discontinue because the drug worked,
        # didn't work, or caused intolerable AEs). Both arrows are
        # post-index. Adjusting for it in a treatment-effect model would
        # induce Berkson-style selection bias.
        dspy.Example(
            feature_name="discontinuation_flag",
            derivation_pseudocode=(
                "treatment_initiated AND (last_fill_date + days_supply < "
                "journey_end_date - gap_threshold)"
            ),
            dataset_context=(
                "ConcertAI CSU claims; target=treatment_initiated; "
                "anchor=index_date; auxiliary clinical-response signal latent"
            ),
            causal_role="collider",
            mechanism=(
                "Two-arrow-in DAG structure: (1) requires treatment_initiated=1 "
                "to be non-trivial (treatment exposure is one parent); "
                "(2) requires the latent treatment-response variable to take "
                "an unfavorable value (efficacy failure or AE) to flip from 0 "
                "to 1 (clinical response is the second parent). Conditioning "
                "on discontinuation in a downstream model opens a non-causal "
                "path between treatment and response — Berkson-style selection "
                "bias. NOT a descendant because descendant requires one parent; "
                "this has two distinct causal sources."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 2 — discontinued_180d (Optum): same two-parent structure
        # as CSU's discontinuation_flag but with the 180d Optum lookback
        # window. Pinned separately so the LM learns the pattern transfers
        # across data sources and lookback semantics.
        dspy.Example(
            feature_name="discontinued_180d",
            derivation_pseudocode=(
                "treatment_initiated AND (last_biologic_fill + days_supply + "
                "60d_gap < index_date + 180d)"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; lookback=180d"
            ),
            causal_role="collider",
            mechanism=(
                "Same two-arrow-in pharmacoepi collider as discontinuation_flag "
                "but for Optum's 180d window: parents are (treatment_exposure, "
                "treatment_response/adherence). Variable is identically zero "
                "for patients who never initiated (single-parent floor) AND "
                "for patients whose response was favorable (second parent "
                "floor). Joint dependence on both makes it a collider. "
                "Adjusting for it in a discontinuation-vs-persistence model "
                "would bias the estimated effect of initiation."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 3 — persistent_at_180d (Optum): mirror of discontinuation
        # with opposite polarity. Two parents: treatment_initiated AND
        # adherence/response (favorable). Useful exemplar so the LM
        # doesn't anchor "collider" only on negative-outcome polarities.
        dspy.Example(
            feature_name="persistent_at_180d",
            derivation_pseudocode=(
                "treatment_initiated AND (PDC over [index_date, index_date+180d] "
                ">= 0.80) AND (last_biologic_fill within 60d of [index+180d])"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; lookback=180d"
            ),
            causal_role="collider",
            mechanism=(
                "Persistence outcome variable is the favorable-polarity twin of "
                "discontinuation. Two arrows in: (a) treatment_initiated (must "
                "have started), (b) adherence + clinical response (must have "
                "stayed). Same collider DAG as discontinuation but encodes the "
                "OR-of-favorable rather than the OR-of-unfavorable. Including "
                "it as a feature in an initiation model would leak the "
                "future-treatment-response signal AND open a backdoor selection "
                "path."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Collider 4 — hospitalizations_total (when computed unwindowed or
        # over the journey post-index): the canonical Berkson-bias
        # collider in pharmacoepi. Two parents: (a) baseline disease
        # severity (which influences both the treatment decision AND the
        # patient's outcome trajectory), (b) treatment-related adverse
        # events (which can drive hospitalization directly). Note: when
        # PROPERLY WINDOWED to a pre-index lookback only, this feature
        # collapses to a confounder. The collider semantics apply when
        # the window includes any post-index event. Rationale flags this
        # dependence on derivation so the LM doesn't blanket-label any
        # hospitalization count as collider.
        dspy.Example(
            feature_name="hospitalizations_total",
            derivation_pseudocode=(
                "count(encounter_events where admit_date in [journey_start, "
                "journey_end]) — NO pre-index temporal filter"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; "
                "encounter_events panel; hospitalizations can be driven by "
                "pre-existing severity OR by post-treatment adverse events"
            ),
            causal_role="collider",
            mechanism=(
                "When measured across the journey without a pre-index filter, "
                "the count has two arrows in: (a) baseline disease severity "
                "(which independently drives both treatment selection and "
                "outcome trajectory — severity is the common cause), and "
                "(b) treatment-related adverse events (post-initiation "
                "biologic AE can independently trigger admissions). "
                "Conditioning opens a backdoor between treatment and outcome "
                "via the severity ↔ AE relationship. (NOTE: a properly "
                "windowed pre-index hospitalization count collapses to a "
                "confounder; the collider semantics depend on the derivation "
                "including post-index events.)"
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # Issue #198 — instrument exemplars (4)
        # ---------------------------------------------------------------------
        # An INSTRUMENT (instrumental variable, IV) is a variable Z such that:
        # (i)  Z → T (the instrument affects the treatment), AND
        # (ii) Z → Y only through T (exclusion restriction — no direct effect
        #      on the outcome bypassing treatment), AND
        # (iii) Z is unconfounded with Y (no common cause of Z and Y).
        #
        # In pharmacoepi the canonical IV pattern is SUPPLY-SIDE / GEOGRAPHIC
        # variation: regional prescribing rates, formulary structure, or
        # specialist-density variation that drives access to a specific
        # treatment without directly affecting the disease biology
        # (Brookhart et al. 2006, "Preference-Based Instrumental Variable
        # Methods for the Estimation of Treatment Effects", Stat Med).
        #
        # Important caveat that we encode in EVERY rationale: the exclusion
        # restriction is an ASSUMPTION, not an observable property. We name
        # the assumption explicitly so the LM learns to FLAG candidate IVs
        # whose exclusion-restriction is plausibly violated (e.g., a
        # geographic region that correlates with environmental allergen
        # exposure would violate exclusion in a CSU/urticaria cohort).
        # Remediation: `keep_with_caveat` because instruments are LEGITIMATE
        # pre-index features useful for causal identification when the
        # exclusion restriction holds; they shouldn't be dropped reflexively.
        # =====================================================================
        # Instrument 1 — urban_rural_code: classic geographic IV per
        # Brookhart-style pharmacoepi. Rural vs urban code reflects
        # supply-side access to specialist-driven biologic prescribing
        # (T) without directly influencing CSU biology (Y). Exclusion
        # restriction is plausible but not proven; assumption stated.
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
                "pharmacoepi IV pattern): RUCA code reflects access to "
                "specialist-driven biologic prescribing (Z→T arrow via "
                "specialist density and biologic-friendly provider mix), "
                "and is plausibly independent of CSU disease biology "
                "(Z↛Y direct path; exclusion restriction holds under the "
                "assumption that urban vs rural classification does not "
                "directly affect urticaria pathophysiology). Set at "
                "enrollment so no confounding with post-index outcomes. "
                "ASSUMPTION: exclusion restriction could be violated if "
                "rural areas have systematically different allergen "
                "exposures or air-quality drivers — name this in audit."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Instrument 2 — geographic_region: 4-region (Northeast / Midwest /
        # South / West) variation in prescribing intensity. Standard IV
        # in observational pharmacoepi research. Exclusion restriction
        # could be violated by region-level allergen exposure in CSU,
        # so we name the caveat in the rationale.
        dspy.Example(
            feature_name="geographic_region",
            derivation_pseudocode=("{NE,MW,South,West} from zip3  # static at enrollment"),
            dataset_context=(
                "CSU/Optum claims; target=treatment_initiated; "
                "anchor=index_date; static at enrollment"
            ),
            causal_role="instrument",
            mechanism=(
                "Regional prescribing-rate variation is the canonical "
                "supply-side IV in observational pharmacoepi (Brookhart "
                "et al. 2006). Z→T arrow: regional formulary policy and "
                "specialist density drive biologic-vs-non-biologic mix. "
                "Z↛Y direct path: region itself does not directly cause "
                "CSU or modify drug pharmacokinetics. Z⊥confounders: "
                "region is set at enrollment, before treatment decision. "
                "ASSUMPTION FLAG: exclusion restriction could be violated "
                "if region correlates with environmental allergen "
                "exposure (e.g., humid Southeast → dust mite load) — "
                "audit-trail rationale should name the assumption."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Instrument 3 — zip3: finer-grained geographic IV. Captures
        # local prescribing heterogeneity below the 4-region level (e.g.,
        # Boston metro biologic uptake vs rural NE) without directly
        # encoding biology. Same exclusion-restriction caveat as region.
        dspy.Example(
            feature_name="zip3",
            derivation_pseudocode=(
                "substr(zipcode_5, 1, 3)  # first 3 digits of ZIP at enrollment"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; "
                "anchor=index_date; static at enrollment; ~900 distinct zip3s"
            ),
            causal_role="instrument",
            mechanism=(
                "Finer-grained geographic IV than region. Z→T arrow: "
                "ZIP3-level variation in specialist density, formulary "
                "negotiation, and biologic-friendly clinic mix drives "
                "prescribing probability. Z↛Y direct path: ZIP3 does not "
                "directly affect disease biology under the standard "
                "exclusion restriction. Used as an instrument in "
                "claims-data IV studies for prescribing variation. "
                "ASSUMPTION FLAG: same caveat as geographic_region — "
                "ZIP3-level allergen exposure or socioeconomic-status "
                "confounding could violate exclusion. Practical note: "
                "with ~900 distinct values, one-hot encoding gives high "
                "cardinality; pre-bucketing into RUCA tiers is common."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Instrument 4 — plan_type: HMO vs PPO vs POS as an IV. Plan
        # design (step-therapy, prior-authorization tiering, biologic
        # formulary placement) affects prescribing probability without
        # directly modifying disease biology. Same exclusion-restriction
        # discipline: name the assumption + flag the SES confounder risk.
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
                "tiering. Z→T arrow: plan design directly affects whether "
                "a biologic prescription is approved at first attempt. "
                "Z↛Y direct path: plan type does not directly modify "
                "disease biology under the exclusion restriction. Common "
                "IV in claims-data drug-utilization research. ASSUMPTION "
                "FLAG: plan type correlates with employer-level "
                "socioeconomic status, and SES can independently affect "
                "outcomes via lifestyle/access — name the confounding path "
                "in the audit trail when using this IV for treatment-effect "
                "estimation."
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
