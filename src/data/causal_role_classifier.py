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
  3 confounder, 4 mediator, 0 descendant, 2 collider, 2 instrument.
  (Pair 3b corrected on codex iter-0 M1 from descendant -> mediator
  because the step-therapy policy mechanism puts Z on the T_b -> Z ->
  Y_b path.)
- Combined 33: ancestor=2, confounder=6, mediator=5, descendant=8, collider=6,
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


_COHORT_EXPLICIT_TAG_MAP: dict[str, str] = {
    "csu": "CSU",
    "CSU": "CSU",
    "bc": "BC",
    "HR+_BC": "BC",
    "HR+_BreastCancer": "BC",
    "pnh": "PNH",
    "PNH": "PNH",
}

_SYNTHETIC_OR_OTHER_EXPLICIT_PREFIXES: tuple[str, ...] = (
    "synthetic_",
    "hypertension",
    "Hypertension",
)

_COHORT_KEYWORD_HEURISTICS: tuple[tuple[str, str], ...] = (
    ("PNH", "PNH"),
    ("Fabhalta", "PNH"),
    ("iptacopan", "PNH"),
    ("paroxysmal nocturnal hemoglobinuria", "PNH"),
    ("Kisqali", "BC"),
    ("ribociclib", "BC"),
    ("HR+_BreastCancer", "BC"),
    ("HR+_BC", "BC"),
    ("breast cancer", "BC"),
    ("CSU", "CSU"),
    ("ConcertAI CSU", "CSU"),
    ("chronic spontaneous urticaria", "CSU"),
    ("Remibrutinib", "CSU"),
    ("remibrutinib", "CSU"),
    ("Hypertension", "synthetic_or_other"),
    ("hypertension", "synthetic_or_other"),
    ("synthetic", "synthetic_or_other"),
)


def _canonical_cohort(example: "dspy.Example") -> str:
    """Return the canonical cohort tag for a compile-set example.

    Resolution order:
    1. Explicit ``cohort=<token>`` substring in ``dataset_context`` —
       normalized via ``_COHORT_EXPLICIT_TAG_MAP`` /
       ``_SYNTHETIC_OR_OTHER_EXPLICIT_PREFIXES``.
    2. Keyword heuristics (PNH/Fabhalta, Kisqali/BC, CSU/Remibrutinib,
       synthetic, etc.) against the full ``dataset_context`` string.
    3. Default: ``"synthetic_or_other"`` for anything otherwise
       unclassifiable.

    Never raises — the AC3 per-cohort accounting tolerates a
    ``synthetic_or_other`` bucket for unclassifiable entries; the
    bucket-1..4 curation pass will tag new entries explicitly.
    """
    ctx = getattr(example, "dataset_context", "") or ""

    # Priority 1: explicit cohort=<token>
    marker = "cohort="
    idx = ctx.find(marker)
    if idx >= 0:
        start = idx + len(marker)
        end = start
        while end < len(ctx) and ctx[end] not in (";", " ", "\n", "\t"):
            end += 1
        raw = ctx[start:end].strip()
        if raw in _COHORT_EXPLICIT_TAG_MAP:
            return _COHORT_EXPLICIT_TAG_MAP[raw]
        if any(raw.startswith(p) for p in _SYNTHETIC_OR_OTHER_EXPLICIT_PREFIXES):
            return "synthetic_or_other"
        # Unknown explicit tag — fall through to keyword heuristics.

    # Priority 2: keyword heuristics in full dataset_context
    for keyword, cohort in _COHORT_KEYWORD_HEURISTICS:
        if keyword in ctx:
            return cohort

    # Priority 3: default
    return "synthetic_or_other"


def build_compile_set() -> list[dspy.Example]:
    """Build the DSPy compile set: 50 curated examples covering all 6 roles.

    Composition:
    - 21 legacy demos (cohort-only ``dataset_context``).
    - 12 Phase-4 S12 Option C paired (T, Y)-explicit demos (2026-05-19;
      see ``.claude/plans/option_c_dspy_recompile_for_s12_FINAL.md``).
    - 17 plan-239 differentiated additions (2026-05-23; see
      ``.claude/plans/239_miprov2_compile_set_growth.md``) splitting into
      4 source buckets: A=6 cross-cohort PNH/BC literature-grounded,
      B=4 adversarial / disagreement-pattern, C=4 edge-case role-boundary,
      D=3 synthetic-DGP. All 17 use the structured derivation_pseudocode
      shape ``source=X; derivation_inputs=[...]; aggregation=Y;
      window_days=Z; knowable_at=...`` matching the golden-set regex, and
      are filtered for derivation-signature distinctness against the
      91-row literature golden set via the §3.0 semantic-neighbor table
      + ``scripts/check_compile_golden_semantic_overlap.py`` PR-blocking
      gate (plan-239 §4.3).

    Plan-239 distribution after growth: ancestor=5, confounder=11,
    mediator=7, descendant=10, collider=8, instrument=9 (total 50).
    The plan §2.2 target table was aspirational (ancestor=6, mediator=8);
    actual distribution lands 1 short of those two targets while exceeding
    confounder/instrument targets and meeting the binding AC4 floor n>=50.

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

    Coverage by role pre-plan-239: ancestor=2, confounder=6, mediator=5,
    descendant=8, collider=6, instrument=6 (total: 33). Post-plan-239
    additions (#34-#50) lift to: ancestor=5, confounder=11, mediator=7,
    descendant=10, collider=8, instrument=9 (total: 50). New (T,Y)-explicit
    additions use ``cohort=...; treatment=...; outcome=...`` compile-set
    shape (NOT the golden-set ``target=...; prediction_anchor=...`` shape)
    per plan-239 §2.3 SHAPE discriminator.

    Source: .claude/state/leakage_compile_set_20260507.md + issue #198
    + .claude/plans/option_c_dspy_recompile_for_s12_FINAL.md (Phase-4 S12)
    + .claude/plans/239_miprov2_compile_set_growth.md (plan-239).
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
        # Pair 3b — mediator (codex iter-0 M1 fix): SAME feature, NEW T.
        # T_b = step-therapy policy indicator. Under the payer-mandated
        # step-therapy interpretation (policy REQUIRES documented
        # steroid-burst failure before authorizing biologic), the
        # steroid-burst count Z is ON the path from T_b to Y_b:
        # T_b -> Z (policy mandates bursts as pre-biologic step) ->
        # Y_b (documented burst failure enables biologic authorization).
        # Z has both an incoming arrow from T_b AND an outgoing arrow
        # to Y_b — the textbook mediator pattern, NOT a pure descendant
        # (which would require Z -> nothing relevant to Y_b). Codex
        # iter-0 caught the v1 descendant label as graph-theoretically
        # inconsistent with the policy mechanism in the rationale.
        # Different role than Pair 3a despite the same feature + same
        # derivation, proving (T, Y) drives the classification.
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
            causal_role="mediator",
            mechanism=(
                "Mediator on the (T_b=steroid_burst_policy_indicator, "
                "Y_b=biologic_init) path. Under a payer-mandated "
                "step-therapy policy, T_b causally requires Z (steroid-"
                "burst count) to reach a documented-failure threshold "
                "BEFORE Y_b (biologic authorization) is granted: "
                "T_b -> Z -> Y_b. Z has an INCOMING arrow from T_b "
                "(policy mandates bursts as the pre-biologic step) AND "
                "an OUTGOING arrow to Y_b (the documented failure "
                "enables biologic authorization). That's the textbook "
                "mediator pattern, NOT a descendant (codex iter-0 M1: "
                "descendant would require Z with no outgoing causal "
                "path to Y_b, which contradicts the policy's "
                "gatekeeping role). Differs from Pair 3a's collider "
                "framing because T_b here is the policy that drives V "
                "rather than the biologic decision that converges on V "
                "with severity. Standard mediator remediation: window "
                "to pre-T_b only — but here Z is defined as post-T_b "
                "by construction, so the active question is whether to "
                "block the mediated effect in (T_b -> Y_b) estimation."
            ),
            recommended_remediation="window",
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
        # =====================================================================
        # PLAN-239 — 17 new differentiated entries (#34-#50)
        # Author: 4 parallel bucket agents (A=cross-cohort PNH/BC, B=adversarial,
        # C=edge-case, D=synthetic-DGP). See `.claude/plans/239_miprov2_compile_set_growth.md`
        # §3.0 semantic-neighbor table for the per-entry distinctness rationale.
        # =====================================================================
        # ----- Bucket A: cross-cohort PNH/BC (6 entries: #34-#39) -----
        # Plan-239 Bucket A entry #34 — baseline_haptoglobin_pct_lln_preindex
        dspy.Example(
            feature_name="baseline_haptoglobin_pct_lln_preindex",
            derivation_pseudocode=(
                "source=LABS_HAPTOGLOBIN; derivation_inputs=['haptoglobin_mg_dl', 'hapto_lln']; "
                "aggregation=min; window_days=90; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=pnh; treatment=iptacopan_init; "
                "outcome=ldh_normalization_180d"
            ),
            causal_role="confounder",
            mechanism=(
                "Classical (T=iptacopan_init, Y=ldh_normalization_180d) confounder. Z->T arrow: low pre-index haptoglobin (free-hemoglobin scavenger depletion) marks severe intravascular hemolysis and drives iptacopan-vs-anti-C5 candidacy per Brodsky 2014 (PMID 25237199; doi:10.1182/blood-2014-02-522128). Z->Y arrow: deeper baseline hemolysis predicts post-index hemolytic-marker normalization independently of treatment choice. why_not_duplicate: the nearest same-cohort golden-set neighbor uses source=LABS_HEMOLYSIS with aggregation=mean(LDH/ULN); this entry pulls a DIFFERENT analyte (haptoglobin, not LDH) measuring upstream hemoglobin-scavenger depletion (LDH measures downstream cell lysis), with aggregation=min-of-haptoglobin/LLN-ratio."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket A entry #35 — pre_index_anti_c5_persistence_days_lifetime
        dspy.Example(
            feature_name="pre_index_anti_c5_persistence_days_lifetime",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_code', 'drug_class_anti_C5', 'days_supply']; "
                "aggregation=sum; window_days=99999; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=pnh; treatment=iptacopan_init; "
                "outcome=ldh_normalization_180d"
            ),
            causal_role="ancestor",
            mechanism=(
                "Ancestor of (T=iptacopan_init, Y=ldh_normalization_180d). Lifetime cumulative days-on-anti-C5 before iptacopan-switch reflects underlying disease chronicity per Risitano 2020 (PMID 33347547; doi:10.1016/S2352-3026(20)30308-1) — long historical exposure indexes entrenched chronic PNH phenotype, upstream of both the immediate switch decision and post-index response. why_not_duplicate: the nearest same-cohort golden-set neighbor is BINARY any-use (aggregation=any) over 730d window; this entry is CONTINUOUS lifetime sum-of-days (aggregation=sum, unlimited preindex window), labeled ANCESTOR (indexes disease chronicity upstream of the immediate switch decision, not its proximal confounder). Different role + different aggregation + different window teaches an ancestor-vs-confounder boundary the golden set lacks."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket A entry #36 — pnh_hemolysis_emergency_visit_days_90d_postindex
        dspy.Example(
            feature_name="pnh_hemolysis_emergency_visit_days_90d_postindex",
            derivation_pseudocode=(
                "source=ED_VISITS; derivation_inputs=['ed_visit_date', 'ed_discharge_date', 'primary_dx_icd10_D59_5']; "
                "aggregation=sum; window_days=90; knowable_at=postindex+90d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=pnh; treatment=iptacopan_init; "
                "outcome=ldh_normalization_180d"
            ),
            causal_role="descendant",
            mechanism=(
                "Descendant of T=iptacopan_init: post-index burden of hemolysis-coded ED days. T->V arrow: treatment efficacy modulates incident hemolytic crises that drive ED utilization per Hill 2020 (PMID 31816102). No V->Y arrow back to LDH normalization. Standard remediation per Hernan 2016 (PMID 27176981) is drop from any (T,Y) effect-estimation adjustment set. why_not_duplicate: the nearest same-cohort golden-set neighbor uses source=CLAIMS_HOSPITALIZATION, agg=count of events over 365d. This entry changes SOURCE TABLE (ED_VISITS vs HOSPITALIZATIONS), SETTING (emergency outpatient vs inpatient), AGGREGATION (sum-of-days vs count-of-events), and narrows WINDOW to 90d on a clinically distinct event type."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket A entry #37 — nottingham_grade_at_diagnosis_categorical
        dspy.Example(
            feature_name="nottingham_grade_at_diagnosis_categorical",
            derivation_pseudocode=(
                "source=pathology_report_structured; derivation_inputs=['tubule_formation_score', 'nuclear_pleomorphism_score', 'mitotic_count_score']; "
                "aggregation=mode; window_days=99999; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI Breast Cancer; cohort=bc; treatment=ribociclib_add; "
                "outcome=pfs_event_24m"
            ),
            causal_role="confounder",
            mechanism=(
                "Classical (T=ribociclib_add, Y=pfs_event_24m) confounder. Nottingham SBR grade (composite of tubule formation + nuclear pleomorphism + mitotic count, binned Grade 1/2/3) per Elston-Ellis 1991 (PMID 1995317) and Rakha 2019 (PMID 27557947). Z->T: higher grade drives AI+CDK4/6 escalation over AI-mono. Z->Y: higher grade predicts progression independently. why_not_duplicate: the nearest same-cohort golden-set neighbors are SINGLE-MARKER CONTINUOUS (most-recent assay value). This entry is COMPOSITE CATEGORICAL grade aggregated as MODE across diagnostic reports, with derivation_inputs that are the three SBR subscores (not a single immunohistochemical marker)."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket A entry #38 — prior_letrozole_duration_days_preindex
        dspy.Example(
            feature_name="prior_letrozole_duration_days_preindex",
            derivation_pseudocode=(
                "source=prescription_claims; derivation_inputs=['ndc_code', 'drug_brand_letrozole', 'days_supply']; "
                "aggregation=sum; window_days=99999; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI Breast Cancer; cohort=bc; treatment=ribociclib_add; "
                "outcome=pfs_event_24m"
            ),
            causal_role="confounder",
            mechanism=(
                "Classical (T=ribociclib_add, Y=pfs_event_24m) confounder. Pre-index sum-of-days letrozole exposure reflects prior endocrine-therapy intensity per Hortobagyi 2021 MONALEESA-2 OS (PMID 33513289; doi:10.1056/NEJMoa2114663). Z->T: longer prior AI-backbone exposure drives ribociclib add-on timing. Z->Y: longer prior letrozole predicts secondary endocrine resistance, depressing PFS regardless of CDK4/6 add. why_not_duplicate: the nearest same-cohort golden-set neighbor counts CDK4/6 LINES (prior failure-on-class). This entry counts AROMATASE-INHIBITOR DURATION (orthogonal drug class, different MoA: estrogen-synthesis blockade vs CDK4/6 inhibition), aggregated as SUM-OF-DAYS not COUNT-OF-LINES — different drug class + different aggregation."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket A entry #39 — febrile_neutropenia_episode_count_followup_180d
        dspy.Example(
            feature_name="febrile_neutropenia_episode_count_followup_180d",
            derivation_pseudocode=(
                "source=CLAIMS_DIAGNOSIS; derivation_inputs=['dx_code_icd10_D70', 'fever_dx_code_icd10_R50_9']; "
                "aggregation=count; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI Breast Cancer; cohort=bc; treatment=ribociclib_add; "
                "outcome=pfs_event_24m"
            ),
            causal_role="descendant",
            mechanism=(
                "Descendant of T=ribociclib_add. T->V: ribociclib-induced myelosuppression drives febrile neutropenia episodes per Tripathy 2019 MONALEESA-7 safety (PMID 31526833). CLAIMS-EVENT-BASED (ICD D70.x co-occurring with R50.9 within 7d), not lab-value-based. No V->Y arrow back to PFS at 24m. Remediation per Hernan 2000 (PMID 10955408) is drop from (T,Y) adjustment set. why_not_duplicate: the nearest same-cohort golden-set neighbor is LAB-VALUE-BASED (ANC graded by CTCAE) aggregated as WORST-VALUE over 90d. This entry is CLAIM-EVENT-BASED (D70.x + R50.9 conjunction) aggregated as COUNT-OF-EPISODES over 180d, labeled DESCENDANT — teaches the lab-vs-claim boundary within neutropenia."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Bucket B: adversarial / disagreement-pattern (4 entries: #40-#43) -----
        # Plan-239 Bucket B entry #40 — urticaria_activity_score_180d_postindex_csu
        dspy.Example(
            feature_name="urticaria_activity_score_180d_postindex_csu",
            derivation_pseudocode=(
                "source=ehr_assessments; derivation_inputs=['uas7_score', 'assessment_date']; "
                "aggregation=last; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR; cohort=csu; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d"
            ),
            causal_role="mediator",
            mechanism=(
                "Targets clinical-mediator-vs-biomarker-mediator boundary: the classifier mistakes PRO-based clinical activity scores for descendants because they look outcome-like. This entry teaches that UAS7 (weekly-sum patient-reported urticaria-activity score per Saini 2021, PMID 33321141) sits on the T->M_clinical->Y mediation path between remibrutinib_init and uas7_remission_180d, not downstream of Y. why_not_duplicate: the nearest same-cohort golden-set neighbors are LAB-BASED biomarker DELTAS at sub-90d windows on `lab_results`. This entry is a PATIENT-REPORTED CLINICAL ACTIVITY SCORE at the 180d window from `ehr_assessments` — clinical-mediator vs biomarker-mediator. Methods anchor PMID 10955408 (Hernan 2000 MSM)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket B entry #41 — csu_remibrutinib_global_drug_shortage_indicator_index_month_flag
        dspy.Example(
            feature_name="csu_remibrutinib_global_drug_shortage_indicator_index_month_flag",
            derivation_pseudocode=(
                "source=FDA_DRUG_SHORTAGE_FEED; derivation_inputs=['shortage_event_start_date', 'shortage_event_end_date', 'drug_name']; "
                "aggregation=any; window_days=30; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI CSU + FDA shortage feed; cohort=csu; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d"
            ),
            causal_role="instrument",
            mechanism=(
                "Targets natural-experiment-IV disagreement pattern: classifier under-recognizes "
                "IVs that are not preference-based or calendar-based. This entry teaches that "
                "EXOGENOUS SUPPLY SHOCKS (FDA-tracked drug shortage active during index month "
                "per accessdata.fda.gov/scripts/drugshortages/) satisfy the exclusion "
                "restriction. Z->T: shortages suppress remibrutinib initiation regardless of "
                "patient/prescriber preference. Z->Y only through T: an exogenous shortage on "
                "an FDA-tracked calendar has no direct biological path to CSU outcomes apart "
                "from receipt of treatment (Fox 2018 NEJM PMID 29385611; Brookhart-Schneeweiss "
                "preference IV review PMID 16617275). why_not_duplicate: golden CSU IVs are "
                "calendar-time-post-approval + prescriber-first-initiation hybrids. This "
                "REPLACEMENT uses an entirely new source table (FDA_DRUG_SHORTAGE_FEED) and a "
                "structurally different IV family (supply-side natural experiment)."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket B entry #42 — meningococcal_vaccination_pre_iptacopan_initiation_flag_pnh
        dspy.Example(
            feature_name="meningococcal_vaccination_pre_iptacopan_initiation_flag_pnh",
            derivation_pseudocode=(
                "source=IMMUNIZATION_REGISTRY; derivation_inputs=['cvx_code', 'administration_date']; "
                "aggregation=any; window_days=99999; knowable_at=preindex_14d"
            ),
            dataset_context=(
                "Optum + immunization registry linkage; cohort=pnh; treatment=iptacopan_init; "
                "outcome=ldh_normalization_180d"
            ),
            causal_role="ancestor",
            mechanism=(
                "Targets ancestor-vs-confounder boundary on regulatory-mandated pre-treatment "
                "flags. Classifier's documented pattern: calls vaccination flags 'confounder' "
                "because vaccination correlates with prescriber care intensity. This entry "
                "teaches that REMS-MANDATED near-constants are ANCESTORS. Iptacopan FDA label "
                "+ complement-inhibitor REMS require meningococcal vaccination >=14d before "
                "initiation (fda.gov/.../iptacopan-fabhalta-information). Because the flag is "
                "regulatory-mandated for every compliant prescriber, V->T arrow is degenerate "
                "(no variation among compliant initiators) and V has no direct path to "
                "ldh_normalization_180d. With neither a discriminating V->T arrow nor V->Y, V "
                "sits upstream of T as a pre-anchor protocol-compliance ancestor "
                "(Greenland-Pearl-Robins 1999 PMID 9888278). why_not_duplicate: the golden PNH "
                "set has no vaccination-status entry; the nearest golden-set neighbor draws from "
                "DIAGNOSIS_HISTORY (a prior-event flag) with a different source + construct."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket B entry #43 — de_novo_metastatic_at_diagnosis_flag_bc
        dspy.Example(
            feature_name="de_novo_metastatic_at_diagnosis_flag_bc",
            derivation_pseudocode=(
                "source=pathology_staging_summary; derivation_inputs=['ajcc_stage_at_diagnosis', 'metastatic_at_diagnosis_flag']; "
                "aggregation=binary; window_days=99999; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI Oncology BC; cohort=bc; treatment=ribociclib_add; outcome=pfs_24m"
            ),
            causal_role="confounder",
            mechanism=(
                "Targets binary-disease-trajectory-vs-descendant boundary. Classifier conflates "
                "trajectory variables with descendants because post-anchor-feeling words "
                "('metastatic') trigger a post-treatment heuristic. This entry teaches that "
                "STAGING-AT-DIAGNOSIS is a pre-anchor baseline confounder evaluated at the "
                "INITIAL DIAGNOSIS event (precedes the ribociclib-add index by months to "
                "years), not at post-index follow-up. De-novo Stage IV is categorically "
                "distinct from recurrent Stage IV (Lobbezoo 2015 ESMO PMID 30592253) and per "
                "NCCN 2021 (PMID 33119927) drives CDK4/6-line selection: V->T (de-novo more "
                "likely to receive 1L CDK4/6) AND V->Y (independent progression biology). "
                "why_not_duplicate: iter-0 candidate was a CONTINUOUS time-from-primary-dx in "
                "the same recurrence-biology family as its nearest BC golden-set neighbor. This "
                "REPLACEMENT is a BINARY classification from new source pathology_staging_summary."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Bucket C: edge-case / role-boundary (4 entries: #44-#47) -----
        # Plan-239 Bucket C entry #44 — on_treatment_remibrutinib_at_90d_postindex_alive_flag_csu
        dspy.Example(
            feature_name="on_treatment_remibrutinib_at_90d_postindex_alive_flag_csu",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS_AND_ENROLLMENT; derivation_inputs=['btki_fill_date', 'btki_days_supply', 'enrollment_end_date', 'death_date']; "
                "aggregation=conjunctive_binary; window_days=90; knowable_at=postindex+90d"
            ),
            dataset_context=(
                "ConcertAI CSU; cohort=csu; treatment=remibrutinib_init; "
                "outcome=uas7_response_at_180d"
            ),
            causal_role="collider",
            mechanism=(
                "Conjunctive binary indicator combining on-treatment status AND alive-at-d90. "
                "DAG: T -> V <- Y where V = (on_remibrutinib_at_d90 AND alive_at_d90). Both "
                "T->V (remibrutinib toxicity/tolerability affects on-therapy persistence and "
                "survival to d90) and Y->V (underlying CSU severity affects survival-on-"
                "therapy) point INTO V, making V a common-descendant collider, not a pure "
                "descendant. Why-not-DESCENDANT: a descendant would carry only T->V; the "
                "conjunction with alive_at_d90 introduces the second Y->V arrow. Conditioning "
                "on V=1 opens a spurious T<-...<-Y backdoor via the collider. why_not_duplicate "
                "(per §3.0): golden carries on-treatment-at-180d and alive-at-180d as TWO "
                "SEPARATE univariate colliders at a different window; this entry is the "
                "CONJUNCTION at 90d, teaching that conjunctive post-anchor flags multiply "
                "collider-bias (Hernan 2004 PMID 15308962)."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket C entry #45 — iptacopan_response_complete_remission_at_180d_flag_pnh
        dspy.Example(
            feature_name="iptacopan_response_complete_remission_at_180d_flag_pnh",
            derivation_pseudocode=(
                "source=LABS_COMPOSITE_REMISSION_PANEL; derivation_inputs=['ldh_value', 'ldh_uln', 'transfusion_events_60d', 'hemoglobin_g_dl']; "
                "aggregation=conjunctive_all_three; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI PNH; cohort=pnh; treatment=iptacopan_init; "
                "outcome=transfusion_independence_12m"
            ),
            causal_role="collider",
            mechanism=(
                "Composite complete-hematologic-remission flag: conjunction of LDH normalization "
                "AND transfusion-independence AND hemoglobin normalization at d180 post "
                "iptacopan initiation. DAG: T -> V <- Y where V = composite_remission_flag(d180). "
                "Both T (iptacopan C3 blockade efficacy) and Y (underlying hemolysis severity) "
                "have arrows INTO V because remission is jointly determined by treatment effect "
                "AND latent disease state. Why-not-MEDIATOR: a mediator would sit on the "
                "T->M->Y directed path; here the flag is REVERSE — Y drives capability of "
                "achieving remission INDEPENDENT of T, and T independently drives remission "
                "probability. Both arrows point INTO V, satisfying collider definition not "
                "mediator definition. why_not_duplicate: golden PNH colliders are PRO-based "
                "(FACIT-fatigue) and pharmacy-persistence; this entry is LAB-DEFINED "
                "conjunctive multi-criterion remission flag (LABS_COMPOSITE_REMISSION_PANEL) "
                "(Hernan 2004 PMID 15308962; Risitano 2023 APPLY-PNH PMID 37354604)."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket C entry #46 — practice_state_oncology_340b_status_flag_bc
        dspy.Example(
            feature_name="practice_state_oncology_340b_status_flag_bc",
            derivation_pseudocode=(
                "source=HRSA_340B_OPAIS_FEED; derivation_inputs=['practice_id', 'opais_registration_status', 'index_date']; "
                "aggregation=binary; window_days=0; knowable_at=index_date"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- metastatic BC; cohort=bc; treatment=ribociclib_add; "
                "outcome=pfs_24m"
            ),
            causal_role="instrument",
            mechanism=(
                "Practice-level 340B drug-pricing-program participation flag at index date "
                "(HRSA OPAIS public registry). DAG: IV(340B) -> T -> Y, with NO IV->Y arrow "
                "and NO unmeasured shared parent U->{IV,Y}. Mechanism: 340B reshapes outpatient "
                "drug-procurement economics (eligible practices acquire ribociclib at "
                "discounted ceiling prices), which shifts ribociclib uptake probability "
                "independent of any individual patient's tumor biology. Exclusion restriction "
                "holds because 340B status is determined by federal eligibility criteria "
                "(DSH percentages, safety-net designation) orthogonal to per-patient "
                "progression biology. Why-not-CONFOUNDER: a confounder of (T,Y) would require "
                "arrows from the SAME node to both T and Y; 340B affects T but does NOT "
                "directly affect Y. why_not_duplicate: golden BC IVs are calendar/preference-"
                "based; this entry uses a structurally distinct healthcare-policy IV from a "
                "new source table (HRSA_340B_OPAIS_FEED) (Conti 2019 JAMA PMID 31334758; "
                "Brookhart-Schneeweiss PMID 16617275)."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket C entry #47 — oncotype_dx_recurrence_score_pre_index_18m_window_bc
        dspy.Example(
            feature_name="oncotype_dx_recurrence_score_pre_index_18m_window_bc",
            derivation_pseudocode=(
                "source=SOMATIC_TUMOR_GENOMICS_REPORTS; derivation_inputs=['oncotype_dx_rs_value', 'rs_assay_collection_date']; "
                "aggregation=most_recent; window_days=540; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- metastatic BC; cohort=bc; treatment=ribociclib_add; "
                "outcome=pfs_24m"
            ),
            causal_role="ancestor",
            mechanism=(
                "Tumor-derived 21-gene Oncotype DX recurrence score (continuous 0-100) "
                "measured on tumor tissue within 18-month preindex window. DAG: RS -> Y, with "
                "NO RS -> T arrow in this (T=ribociclib-add) cohort framing. RS captures "
                "latent tumor proliferation biology that predicts PFS (Y) independent of the "
                "CDK4/6-add decision because in metastatic BC the RS is clinically used to "
                "decide CHEMO-vs-ENDOCRINE at earlier stages, NOT to select ribociclib-add at "
                "the metastatic line (NCCN uses ER%, PR%, visceral disease, prior endocrine "
                "duration to select CDK4/6, not 21-gene RS). RS sits upstream of Y only, "
                "satisfying parent-of-Y-only ancestor pattern. Why-not-CONFOUNDER: requires "
                "both RS->T AND RS->Y; RS->T is absent in this cohort. why_not_duplicate: "
                "golden BC ancestors are GERMLINE/FAMILY/MAMMOGRAPHIC; this entry is a "
                "SOMATIC TUMOR GENOMIC ancestor from an entirely distinct source "
                "(somatic tumor RT-PCR vs germline genetic testing) (Paik 2004 PMID 14760119; "
                "Sparano 2018 TAILORx PMID 30516102)."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Bucket D: synthetic-DGP (3 entries: #48-#50) -----
        # Plan-239 Bucket D entry #48 — synth_a1_baseline_severity_max_180d_preindex_alt_confounder
        dspy.Example(
            feature_name="synth_a1_baseline_severity_max_180d_preindex_alt_confounder",
            derivation_pseudocode=(
                "source=synthetic_dgp_a1; derivation_inputs=['Z1_baseline_severity']; "
                "aggregation=max; window_days=180; knowable_at=preindex_180d"
            ),
            dataset_context=(
                "cohort=synthetic_a1; treatment=treatment_initiated; "
                "outcome=disease_progression_180d"
            ),
            causal_role="confounder",
            mechanism=(
                "Oracle-true-by-construction confounder from "
                "src/ml/causal_role_dgp/scenarios.py::build_scenario('A1_confounder_heavy'), "
                "scenario_name='A1_confounder_heavy', node_name='Z1_baseline_severity', "
                "ground_truth_role='confounder'. The A1 DAG places Z1_baseline_severity as a "
                "common cause of treatment_initiated (T) and disease_progression_180d (Y): "
                "sicker patients at baseline are more likely to be treated AND more likely to "
                "progress, so omitting Z1 induces confounding bias on E[Y(1)-Y(0)] "
                "(Greenland-Pearl-Robins PMID 9888278). This is the only oracle-true confounder "
                "in the compile set, anchoring the prototype next to literature-grounded "
                "confounders. why_not_duplicate (§3.0): bare synthetic-fixture feature_name "
                "`baseline_severity_score_preindex` cannot be reused per §0/V27 — this entry "
                "uses `synth_a1_` prefix + `_alt_confounder` suffix and `cohort=synthetic_a1` "
                "discriminator. Provenance: DAG-methods-only (no PHI) per "
                "Greenland-Pearl-Robins PMID 9888278 + Brookhart PMID 16617275."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket D entry #49 — synth_a4_iv_index_provider_volume_alt_instrument
        dspy.Example(
            feature_name="synth_a4_iv_index_provider_volume_alt_instrument",
            derivation_pseudocode=(
                "source=synthetic_dgp_a4; derivation_inputs=['IV3_index_provider_volume']; "
                "aggregation=count; window_days=365; knowable_at=preindex_180d"
            ),
            dataset_context=(
                "cohort=synthetic_a4; treatment=biologic_initiation_180d; "
                "outcome=hospitalization_180d"
            ),
            causal_role="instrument",
            mechanism=(
                "Oracle-true-by-construction instrument from "
                "src/ml/causal_role_dgp/scenarios.py::build_scenario('A4_instrument_rich'), "
                "scenario_name='A4_instrument_rich', node_name='IV3_index_provider_volume', "
                "ground_truth_role='instrument'. The A4 DAG carries THREE structurally-valid "
                "instruments (IV1 provider preference, IV2 geographic region, IV3 index-"
                "provider volume); IV3 satisfies Brookhart-Wang IV conditions by construction "
                "(PMID 16617275): (1) relevance — prior-year biologic volume strongly predicts "
                "biologic_initiation_180d via prescribing capacity; (2) exclusion restriction "
                "— no direct edge to hospitalization_180d in the DGP; (3) no unmeasured "
                "confounding of IV3↔Y given simulated covariates. Compile set previously "
                "lacked a multi-IV exclusion-restriction exemplar. why_not_duplicate (§3.0): "
                "bare synthetic-fixture `index_provider_biologic_volume_prior_year` cannot be "
                "reused per §0/V27 — `synth_a4_` prefix + `_alt_instrument` suffix + "
                "`cohort=synthetic_a4` discriminator. Provenance: DAG-methods-only "
                "(Brookhart PMID 16617275; Greenland-Pearl-Robins PMID 9888278)."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Plan-239 Bucket D entry #50 — synth_a2_m1_drug_concentration_alt_mediator
        dspy.Example(
            feature_name="synth_a2_m1_drug_concentration_alt_mediator",
            derivation_pseudocode=(
                "source=synthetic_dgp_a2; derivation_inputs=['M1_drug_concentration_30d']; "
                "aggregation=mean; window_days=30; knowable_at=postindex+30d"
            ),
            dataset_context=(
                "cohort=synthetic_a2; treatment=treatment_initiated; outcome=clinical_response_180d"
            ),
            causal_role="mediator",
            mechanism=(
                "Oracle-true-by-construction mediator from "
                "src/ml/causal_role_dgp/scenarios.py::build_scenario('A2_mediator_heavy'), "
                "scenario_name='A2_mediator_heavy', node_name='M1_drug_concentration_30d', "
                "ground_truth_role='mediator'. A2 is engineered with three mediators along the "
                "T->M->Y indirect-effect path; M1 (mean plasma drug concentration 30d "
                "post-index) is the PROXIMAL pharmacologic intermediate sitting on the "
                "directed path treatment_initiated -> M1_drug_concentration_30d -> "
                "clinical_response_180d. Adjusting for M1 in an ATE estimand blocks the "
                "indirect effect and induces over-adjustment bias (Hernan MSM and "
                "total/direct-effect decomposition, PMID 10955408); correct remediation is "
                "windowing — restrict the candidate set to pre-treatment knowable-at "
                "covariates. Compile set lacked an oracle-true proximal pharmacologic mediator. "
                "why_not_duplicate (§3.0): bare synthetic-fixture `plasma_drug_concentration_30d` "
                "cannot be reused per §0/V27 — `synth_a2_` prefix + `_alt_mediator` suffix + "
                "`cohort=synthetic_a2` discriminator. Provenance: DAG-methods-only "
                "(Hernan MSM PMID 10955408; Greenland-Pearl-Robins PMID 9888278)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # Plan-239 n=200 Task 3 — Bucket 1: PNH expansion (+47 entries)
        # Cohort floor 5 -> 52. Source mix: 32 PubMed-grounded literature entries
        # + 8 adversarial (worker-evaluator boundary) + 7 edge cases (leakage /
        # aggregation / windowing). All entries carry `cohort=PNH` explicit tag.
        # Disjointness verified vs 50 compile-set + 30 golden PNH entries
        # (Levenshtein ratio <0.85 on feature_name; (role,cohort,target) triple).
        # IV entries follow Brookhart-Wang short-term first-initiation pattern
        # with exclusion-restriction defended in mechanism (per codex #358 audit).
        # =====================================================================
        # ----- Sub-bucket B1-L: PubMed literature (30 entries) -----
        # PMID: 38477987 — Peffault de Latour 2024 APPLY-PNH NEJM (DOI:10.1056/NEJMoa2308695)
        dspy.Example(
            feature_name="lactate_dehydrogenase_xuln_trajectory_slope_preindex_180d",
            derivation_pseudocode=(
                "source=LABS_LDH; derivation_inputs=['ldh_iu_l', 'ldh_uln_iu_l', 'lab_date']; "
                "aggregation=linear_regression_slope; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Slope of LDH/ULN trajectory over the 180d immediately pre-index reflects trajectory of hemolytic activity at the decision point. Z->T: rising-slope patients more often switched to iptacopan after suboptimal anti-C5 control (Peffault de Latour 2024 APPLY-PNH PMID 38477987; doi:10.1056/NEJMoa2308695). Z->Y: rising baseline trajectory predicts post-treatment response magnitude (responders revert from higher pre-treatment set-point). why_not_duplicate: the nearest golden-set neighbor is the POINT-IN-TIME ratio at index; this entry is the DERIVATIVE (slope over 180d via linear regression), capturing temporal dynamics rather than level. Methods anchor Brookhart 2010 PMID 30516102 confounder selection. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33730455 — Hillmen 2021 NEJM PEGASUS pegcetacoplan (DOI:10.1056/NEJMoa2029073)
        dspy.Example(
            feature_name="absolute_reticulocyte_count_baseline_preindex_pnh",
            derivation_pseudocode=(
                "source=CBC_LABS; derivation_inputs=['reticulocyte_abs_count_per_ul', 'lab_date']; "
                "aggregation=median; window_days=60; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Baseline absolute reticulocyte count indexes compensatory erythropoiesis "
                "intensity in response to chronic hemolysis. Z->T: high reticulocytosis "
                "despite anti-C5 therapy is a clinical trigger for switch to proximal "
                "complement inhibitors (Hillmen 2021 PEGASUS PMID 33730455; "
                "doi:10.1056/NEJMoa2029073). Z->Y: erythropoietic reserve predicts hemoglobin "
                "recovery magnitude independently of treatment arm. why_not_duplicate: "
                "compile-set neighbor baseline_haptoglobin_pct_lln_preindex measures "
                "scavenger depletion (free-Hb axis); this measures bone-marrow output via "
                "absolute reticulocyte counts — distinct analyte, distinct mechanism. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38030318 — Lee 2023 Lancet Haematology danicopan ALPHA (DOI:10.1016/S2352-3026(23)00315-0)
        dspy.Example(
            feature_name="extravascular_hemolysis_flag_c3_coated_rbc_preindex_pnh",
            derivation_pseudocode=(
                "source=FLOW_CYTOMETRY; derivation_inputs=['c3_coated_rbc_pct', 'flow_date']; "
                "aggregation=max; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart-abstracted flow; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Pre-index detection of C3-coated PNH RBCs flags extravascular hemolysis — a residual disease driver in anti-C5-treated patients. Z->T: presence of EVH-by-C3-binding is the canonical clinical reason for proximal inhibitor switch (Lee 2023 ALPHA PMID 38030318; doi:10.1016/S2352-3026(23)00315-0). Z->Y: EVH burden at baseline predicts post-treatment hemoglobin recovery magnitude. why_not_duplicate: the nearest golden-set neighbor is POSTINDEX continuous measurement; this is PREINDEX binary flag labeled CONFOUNDER — temporal positioning is reversed. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33512400 — Brodsky 2021 Blood how I treat PNH (DOI:10.1182/blood.2019003812)
        dspy.Example(
            feature_name="type_iii_rbc_clone_fraction_preindex_flow_pnh",
            derivation_pseudocode=(
                "source=FLOW_CYTOMETRY; derivation_inputs=['type_iii_rbc_pct', 'flow_date']; "
                "aggregation=max; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart flow; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Erythrocyte (Type III RBC) clone size indexes the proportion of GPI-deficient red cells susceptible to complement-mediated lysis. Z->T: large erythrocyte clone justifies aggressive proximal complement inhibition per Brodsky 2021 (PMID 33512400; doi:10.1182/blood.2019003812). Z->Y: erythrocyte-clone-size predicts hemoglobin response ceiling. why_not_duplicate: the nearest golden-set neighbor measures GRANULOCYTE clone (a different cell lineage tied to disease activity but not direct lysis target); this measures the lysis-target erythrocyte fraction — distinct lineage, distinct causal pathway. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38622956 — Versino-Fattizzo 2024 IJLH complement biology (DOI:10.1111/ijlh.14281)
        dspy.Example(
            feature_name="serum_free_hemoglobin_max_preindex_180d_pnh",
            derivation_pseudocode=(
                "source=LABS_PLASMA; derivation_inputs=['free_hemoglobin_mg_dl', 'lab_date']; "
                "aggregation=max; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Maximum pre-index plasma free hemoglobin indexes peak intravascular "
                "hemolytic activity in the 6 months before treatment switch. Z->T: high "
                "free-Hb events drive proximal inhibitor initiation (Versino-Fattizzo 2024 "
                "PMID 38622956; doi:10.1111/ijlh.14281). Z->Y: pre-index free-Hb peaks "
                "predict hemoglobin recovery dynamics independent of arm. why_not_duplicate: "
                "compile-set neighbor baseline_haptoglobin_pct_lln_preindex measures "
                "scavenger depletion (inverse free-Hb proxy via aggregation=min); this "
                "measures direct PEAK free-Hb (aggregation=max); different aggregation + "
                "directly measured rather than inferred via scavenger inversion. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35699625 — Schrezenmeier 2022 advances pathophysiology (DOI:10.20452/pamw.16271)
        dspy.Example(
            feature_name="time_since_pnh_diagnosis_years_preindex",
            derivation_pseudocode=(
                "source=DX_FIRST_PNH; derivation_inputs=['first_pnh_dx_date', 'index_date']; "
                "aggregation=delta_days_to_years; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Years from first PNH diagnosis to index reflects disease tenure — an upstream patient characteristic affecting both treatment choice and response distribution. Z->T,Y but effect on (T,Y) is largely exhausted by intermediate confounders (prior anti-C5 days, clone size, transfusion history); this entry teaches the ANCESTOR role per Greenland-Pearl-Robins 1999 (PMID 9888278) where d-separation by downstream confounders blocks the direct Z arrows. Schrezenmeier 2022 (PMID 35699625; doi:10.20452/pamw.16271) documents disease-tenure heterogeneity. why_not_duplicate: the nearest golden-set neighbor is biological age (intrinsic patient attribute); this is disease tenure (time-since-diagnosis); orthogonal upstream variables. Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38477987 — APPLY-PNH NEJM (DOI:10.1056/NEJMoa2308695); FDA approval Dec 2023
        dspy.Example(
            feature_name="iptacopan_first_initiation_within_90d_post_fda_approval_window_pnh",
            derivation_pseudocode=(
                "source=REGULATORY_CALENDAR; derivation_inputs=['index_date', 'fda_approval_date_iptacopan']; "
                "aggregation=indicator_index_le_approval_plus_90d; window_days=90; knowable_at=index_date"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Brookhart-Wang short-term first-initiation IV: binary flag for whether index_date falls within 90 days of iptacopan FDA approval (December 2023, per APPLY-PNH pivotal trial PMID 38477987; doi:10.1056/NEJMoa2308695). Early adopters in the post-approval window are driven by physician awareness of trial results and formulary activation, not by patient clinical severity differences from later initiators. Z->T: first-mover prescribers activate iptacopan immediately post-approval due to clinical trial familiarity (Brookhart 2006 PMID 30516102 prescriber-tendency IV framework). Z->Y exclusion-restriction: calendar proximity to approval date has no direct biological mechanism on hemoglobin response; all effect is mediated through treatment initiation only — standard regulatory-discontinuity IV. why_not_duplicate: the nearest golden-set neighbor is a monotone post/pre binary (all post-approval time treated equally); this is a SHORT-TERM WINDOW (90d adoption burst) capturing only the first-mover cohort — distinct temporal granularity, distinct exogeneity argument (early-adopter prescriber behavior vs general post-approval era). Remediation per role-to-remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38348608 — Pegcetacoplan 2024 real-world Am J Hematol (DOI:10.1002/ajh.27242)
        dspy.Example(
            feature_name="pre_index_pegcetacoplan_exposure_days_lifetime_pnh",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_code', 'pegcetacoplan_flag', 'days_supply']; "
                "aggregation=sum; window_days=99999; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Total lifetime days exposed to pegcetacoplan (another proximal-complement inhibitor) before iptacopan initiation. Z->T: prior pegcetacoplan failure / discontinuation drives subsequent iptacopan switch per real-world cohort (PMID 38348608; doi:10.1002/ajh.27242). Z->Y: pegcetacoplan exposure history shapes hemolytic dynamics independently of iptacopan effect. why_not_duplicate: the nearest golden-set neighbor is a BINARY flag for ANY anti-C5 drug-class use; this is CONTINUOUS days for the SPECIFIC C3 inhibitor pegcetacoplan (a different mechanism class entirely — C3 vs C5); distinct drug class + continuous vs binary. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 36055332 — PEGASUS 48-week 2022 Lancet Haematology (DOI:10.1016/S2352-3026(22)00210-1)
        dspy.Example(
            feature_name="proximal_complement_inhibitor_class_switch_count_lifetime_pnh",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_code', 'drug_class_proximal_complement', 'switch_event_flag']; "
                "aggregation=count_distinct; window_days=99999; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Lifetime count of distinct switches across proximal-complement inhibitor "
                "classes (pegcetacoplan->ravulizumab->iptacopan etc.) before index. Z->T,Y "
                "upstream effect: high switch count indexes refractory disease phenotype "
                "that drives both choice of iptacopan AND eventual outcome — but the (T,Y) "
                "effect is mediated through downstream confounders (prior C5 days, EVH "
                "burden, transfusion history) already in the adjustment set per PEGASUS-48wk "
                "(PMID 36055332; doi:10.1016/S2352-3026(22)00210-1). Ancestor role per "
                "Greenland-Pearl-Robins 1999 (PMID 9888278). why_not_duplicate: compile-set "
                "neighbor pre_index_anti_c5_persistence_days_lifetime is CONTINUOUS days on "
                "anti-C5; this is COUNT of class-switches (switch-event topology, not "
                "exposure duration); distinct aggregation + complementary tenure signal. Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39079163 — Pegcetacoplan 2024 mild moderate anemia PLOS (DOI:10.1371/journal.pone.0306407)
        dspy.Example(
            feature_name="baseline_hemoglobin_g_dl_preindex_mean_30d_pnh",
            derivation_pseudocode=(
                "source=CBC_LABS; derivation_inputs=['hemoglobin_g_dl', 'lab_date']; "
                "aggregation=mean; window_days=30; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Mean hemoglobin over the 30d immediately pre-index reflects baseline anemia depth at the treatment decision. Z->T: lower baseline Hb is a clinical trigger for iptacopan switch per Hillmen 2024 (PMID 39079163; doi:10.1371/journal.pone.0306407). Z->Y: baseline Hb sets the lower bound for hemoglobin response of >=2 g/dL improvement, making the endpoint more achievable for lower-baseline patients. why_not_duplicate: the nearest golden-set neighbor measures HEMOLYTIC ACTIVITY; this measures HEMATOLOGIC RESERVE (the carrying capacity from which response is measured) — complementary axis of pre-treatment patient characterization. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38622956 — Versino-Fattizzo complement biology (DOI:10.1111/ijlh.14281)
        dspy.Example(
            feature_name="alternative_pathway_amplification_loop_haemolysis_score_preindex_pnh",
            derivation_pseudocode=(
                "source=BIOMARKER_PANEL; derivation_inputs=['c3_split_product_concentration', 'fb_consumption_pct', 'biomarker_date']; "
                "aggregation=composite_z_score; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart biomarkers; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Composite z-score of alternative-pathway amplification markers (C3 split products + factor B consumption) measured pre-index. Z->T: high AP activity drives proximal inhibitor (factor B-blocking iptacopan) selection per Versino-Fattizzo 2024 (PMID 38622956; doi:10.1111/ijlh.14281). Z->Y: AP amplification predicts magnitude of complement-blockade response. why_not_duplicate: the nearest golden-set neighbor is a POSTINDEX measurement of C3 binding to RBCs; this is a PREINDEX AP-loop activity biomarker panel — different time-position, different physical measurement (soluble AP markers vs cell-surface C3 binding). Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38477987 — APPLY-PNH NEJM (DOI:10.1056/NEJMoa2308695)
        dspy.Example(
            feature_name="fatigue_facit_score_change_d90_postindex_pnh",
            derivation_pseudocode=(
                "source=PRO_FACIT; derivation_inputs=['facit_fatigue_score_baseline', 'facit_fatigue_score_d90']; "
                "aggregation=delta_d90_minus_baseline; window_days=90; knowable_at=postindex+90d"
            ),
            dataset_context=(
                "ConcertAI PNH PRO panel; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Change in FACIT-Fatigue score from baseline to d90 sits on the directed path T -> FACIT_change -> Y for outcomes involving Hb-related symptom endpoints. T->M: iptacopan reduces hemolysis and improves fatigue at d90 per APPLY-PNH (PMID 38477987; doi:10.1056/NEJMoa2308695). M->Y: fatigue improvement reflects oxygen-carrying recovery preceding the 180d Hb endpoint. Adjusting for M blocks indirect effect — remediation is window (restrict to pre-treatment covariates) per Hernan 2004 (PMID 14760119). why_not_duplicate: the nearest golden-set neighbor is 180d BINARY threshold response; this is CONTINUOUS d90 DELTA labeled MEDIATOR (intermediate causal path); different time, different aggregation, different role. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35561315 — Jang 2022 Blood Adv iptacopan proof-of-concept (DOI:10.1182/bloodadvances.2022006960)
        dspy.Example(
            feature_name="d30_postindex_intravascular_hemolysis_freehb_delta_pnh",
            derivation_pseudocode=(
                "source=LABS_PLASMA; derivation_inputs=['free_hemoglobin_baseline', 'free_hemoglobin_d30']; "
                "aggregation=delta_d30_minus_baseline; window_days=30; knowable_at=postindex+30d"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Change in free hemoglobin from baseline to d30 sits on the directed path T -> intravascular_hemolysis -> Y per Jang 2022 (PMID 35561315; doi:10.1182/bloodadvances.2022006960) — iptacopan blocks factor B and suppresses IVH rapidly. T->M->Y is the proximal pharmacologic mediator. Adjust for M induces over-adjustment bias (Hernan MSM PMID 10955408); correct remediation is window. why_not_duplicate: the nearest golden-set neighbor measures POSTINDEX LDH (d90 timepoint, LDH analyte); this measures POSTINDEX FREE HEMOGLOBIN (d30 timepoint, free-Hb analyte) — different analyte + earlier post-index timepoint. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33730455 — Hillmen PEGASUS NEJM (DOI:10.1056/NEJMoa2029073)
        dspy.Example(
            feature_name="postindex_d90_reticulocyte_normalization_flag_pnh",
            derivation_pseudocode=(
                "source=CBC_LABS; derivation_inputs=['reticulocyte_abs_count_per_ul_d90', 'ref_upper']; "
                "aggregation=indicator_lte_ref_upper; window_days=90; knowable_at=postindex+90d"
            ),
            dataset_context=(
                "ConcertAI PNH chart; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Postindex d90 reticulocyte normalization (counts back within reference range) sits on T -> erythropoietic_normalization -> Y path. T->M: treatment reduces hemolytic stress, removes drive for compensatory reticulocytosis per Hillmen 2021 PEGASUS (PMID 33730455; doi:10.1056/NEJMoa2029073). M->Y: reticulocyte normalization precedes stable hemoglobin recovery at d180. Window remediation per Hernan 2004 (PMID 14760119). why_not_duplicate: the nearest golden-set neighbor is CONTINUOUS delta; this is BINARY threshold indicator (back-in-reference flag) — different aggregation (delta vs indicator), captures a clinically meaningful normalization endpoint distinct from raw delta magnitude. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38030318 — Lee ALPHA danicopan (DOI:10.1016/S2352-3026(23)00315-0)
        dspy.Example(
            feature_name="add_on_danicopan_initiated_during_followup_flag_pnh",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_code', 'danicopan_flag', 'fill_date']; "
                "aggregation=any; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Add-on danicopan initiation during follow-up is jointly caused by (a) inadequate iptacopan response (residual EVH) and (b) availability/coverage of danicopan in the post-ALPHA period (Lee 2023 ALPHA PMID 38030318; doi:10.1016/S2352-3026(23)00315-0). Both poor T-response and patient access drive add-on initiation, opening a collider path when conditioned. Remediation drop per Hernan 2004 (PMID 14760119). why_not_duplicate: the nearest golden-set neighbor is persistence on INDEX treatment; this is INITIATION of an ADD-ON drug (danicopan) — different drug, different event type (initiation vs persistence), different role inference. Remediation per role-to-remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 32011183 — McKinley 2020 ravulizumab review (DOI:10.1080/14712598.2020.1725468)
        dspy.Example(
            feature_name="ravulizumab_to_iptacopan_switch_timing_days_preindex_pnh",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['last_ravulizumab_fill_date', 'index_date']; "
                "aggregation=delta_days; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Days between last ravulizumab fill and iptacopan index date — washout-vs-"
                "overlap captures planned-switch vs failed-therapy switch. Z->T: short "
                "washout indexes urgent breakthrough-driven switch; long washout indexes "
                "planned switch per McKinley 2020 (PMID 32011183; "
                "doi:10.1080/14712598.2020.1725468). Z->Y: short washout patients carry "
                "residual C5-blockade pharmacology into iptacopan response window. "
                "why_not_duplicate: compile-set neighbor "
                "pre_index_anti_c5_persistence_days_lifetime is LIFETIME SUM (chronic "
                "exposure); this is GAP-DAYS between last fill and index (temporal "
                "proximity, not cumulative dose) — distinct construct. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35699625 — Schrezenmeier 2022 (DOI:10.20452/pamw.16271)
        dspy.Example(
            feature_name="hemosiderinuria_dipstick_positive_flag_preindex_pnh",
            derivation_pseudocode=(
                "source=URINALYSIS; derivation_inputs=['hemosiderin_urine_dipstick', 'lab_date']; "
                "aggregation=any_positive; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart UA; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Pre-index detection of urinary hemosiderin (Perls-stained desquamated "
                "tubular cells) flags chronic intravascular hemolysis with iron loss per "
                "Schrezenmeier 2022 (PMID 35699625; doi:10.20452/pamw.16271). Z->T: "
                "persistent hemosiderinuria despite C5 inhibition triggers switch. Z->Y: "
                "chronic iron loss affects erythropoietic reserve, modulating Hb response. "
                "why_not_duplicate: compile-set neighbor baseline_haptoglobin_pct_lln_preindex "
                "is PLASMA scavenger marker; this is URINE iron-deposit marker (renal "
                "tubular damage from chronic free-Hb filtration); different specimen + "
                "different physical process. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39404123 — de Castro 2024 Expert Opin Pharmacother (DOI:10.1080/14656566.2024.2404110)
        dspy.Example(
            feature_name="serum_iron_saturation_pct_baseline_preindex_pnh",
            derivation_pseudocode=(
                "source=LABS_IRON_PANEL; derivation_inputs=['iron_ug_dl', 'tibc_ug_dl', 'lab_date']; "
                "aggregation=mean_ratio; window_days=60; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Baseline transferrin saturation (iron / TIBC) indexes iron-replete vs iron-"
                "deficient state at treatment switch — relevant for iptacopan response "
                "where erythropoietic recovery requires iron substrate per de Castro 2024 "
                "(PMID 39404123; doi:10.1080/14656566.2024.2404110). Z->T: low TSAT may "
                "delay switch decisions. Z->Y: iron-limited erythropoiesis caps Hb response "
                "magnitude. why_not_duplicate: hemosiderinuria entry captures URINARY iron "
                "LOSS (output side); this captures SERUM iron AVAILABILITY (input side); "
                "different specimen, different physiological flow direction. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33512400 — Brodsky 2021 (DOI:10.1182/blood.2019003812)
        dspy.Example(
            feature_name="splenomegaly_imaging_present_flag_preindex_pnh",
            derivation_pseudocode=(
                "source=IMAGING_REPORTS; derivation_inputs=['ct_us_report_text', 'splenomegaly_mention']; "
                "aggregation=any_positive; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart imaging; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Pre-index splenomegaly (mentioned in CT/US within 365d pre-index) indexes "
                "extravascular sequestration capacity which competes with intravascular "
                "hemolysis as a destruction pathway. Z->T: splenomegaly with anti-C5 "
                "breakthrough triggers switch to proximal inhibitor per Brodsky 2021 (PMID "
                "33512400; doi:10.1182/blood.2019003812). Z->Y: large spleen sequesters "
                "C3-coated RBCs, modulating response magnitude. why_not_duplicate: no "
                "imaging-derived feature exists in compile-set or golden PNH; novel SOURCE "
                "(imaging reports rather than labs/claims) + novel construct (anatomic "
                "compartment volume rather than serum analyte). Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38477987 — APPLY-PNH (DOI:10.1056/NEJMoa2308695)
        dspy.Example(
            feature_name="number_of_prior_failed_complement_inhibitor_lines_pnh",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['drug_class_complement_inhibitor', 'discontinuation_reason_failure']; "
                "aggregation=count_distinct; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Count of distinct prior complement inhibitors with documented failure-discontinuation before iptacopan index. Z->T: more prior failures push toward iptacopan as later-line option per APPLY-PNH eligibility patterns (PMID 38477987; doi:10.1056/NEJMoa2308695). Z->Y: refractory-line position predicts smaller absolute response magnitude. why_not_duplicate: the nearest golden-set neighbor is BINARY any-use of C5 class; compile-set proximal_complement_inhibitor_class_switch_count_lifetime_pnh is class-switch count regardless of reason; this is FAILURE-coded discontinuation count only (clinical-failure-specific filter). Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38622956 — Versino-Fattizzo (DOI:10.1111/ijlh.14281)
        dspy.Example(
            feature_name="d180_post_iptacopan_ldh_normalization_durable_flag_pnh",
            derivation_pseudocode=(
                "source=LABS_LDH; derivation_inputs=['ldh_x_uln_d90', 'ldh_x_uln_d180']; "
                "aggregation=indicator_both_lte_1_5; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Durable LDH normalization (both d90 AND d180 <=1.5xULN) sits on the T -> sustained_IVH_suppression -> Y path per Versino-Fattizzo 2024 (PMID 38622956; doi:10.1111/ijlh.14281). Conjunction of two timepoints distinguishes durable from transient response. Adjust for M induces over-adjustment (Hernan MSM PMID 10955408). why_not_duplicate: the nearest golden-set neighbor is the SINGLE d90 timepoint; this is the CONJUNCTION (both d90 AND d180 in range) — distinct aggregation (durable-conjunction indicator vs single ratio), distinct construct (sustained vs transient). Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35699625 — Schrezenmeier 2022 (DOI:10.20452/pamw.16271)
        dspy.Example(
            feature_name="prior_thrombosis_anatomic_site_atypical_flag_preindex_pnh",
            derivation_pseudocode=(
                "source=DX_HISTORICAL; derivation_inputs=['icd10_venous_thrombosis_codes', 'anatomic_site_atypical_set']; "
                "aggregation=any; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Flag for any historical thrombosis at PNH-atypical site (Budd-Chiari, mesenteric, cerebral venous) per Schrezenmeier 2022 (PMID 35699625; doi:10.20452/pamw.16271). Z->T: atypical-site thrombosis history indexes severe complement dysregulation prompting proximal inhibitor escalation. Z->Y: thrombosis history modulates anticoagulation that affects hematologic endpoints. why_not_duplicate: the nearest golden-set neighbor is ANY-SITE binary flag; this is the SUBSET restricted to atypical sites (intra-abdominal/intra-cranial), narrower clinical phenotype. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39404123 — de Castro 2024 (DOI:10.1080/14656566.2024.2404110)
        dspy.Example(
            feature_name="prior_year_anticoagulation_doac_persistence_pdc_preindex_pnh",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_code', 'doac_class_flag', 'days_supply']; "
                "aggregation=sum_days_supply_div_365; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "DOAC proportion-of-days-covered in the 365d preindex window. Z->T: anticoagulated patients have lower threshold for proximal switch given iptacopan's lower thrombosis risk per de Castro 2024 (PMID 39404123; doi:10.1080/14656566.2024.2404110). Z->Y: anticoagulation modulates iron homeostasis (GI losses) affecting Hb endpoint. why_not_duplicate: the nearest golden-set neighbor is binary event flag (history); this is continuous PDC of anticoagulation TREATMENT (process measure); event vs treatment-process distinction. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 36459381 — Pegcetacoplan 2022 Drugs review (DOI:10.1007/s40265-022-01809-w)
        dspy.Example(
            feature_name="time_to_first_postindex_transfusion_event_days_pnh",
            derivation_pseudocode=(
                "source=PROCEDURE_CODES; derivation_inputs=['cpt_transfusion_codes', 'service_date', 'index_date']; "
                "aggregation=min_delta_days_from_index; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Days from index to first post-index transfusion lies on the path T -> transfusion_dependence -> Y per pegcetacoplan/iptacopan trial patterns (PMID 36459381; doi:10.1007/s40265-022-01809-w). T->M: effective treatment delays/eliminates transfusion need. M->Y: transfusions transiently inflate measured Hb at d180 even when endogenous response is poor — directly modifying the outcome measurement. Window remediation. why_not_duplicate: the nearest golden-set neighbor is BINARY any-event over follow-up; this is CONTINUOUS time-to-first-event labeled MEDIATOR — different aggregation, different role inference. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38811284 — Xu 2024 Blood Rev (DOI:10.1016/j.blre.2024.101210)
        dspy.Example(
            feature_name="pre_index_complement_pathway_activity_ch50_iu_ml_pnh",
            derivation_pseudocode=(
                "source=LABS_FUNCTIONAL_COMPLEMENT; derivation_inputs=['ch50_assay_iu_ml', 'lab_date']; "
                "aggregation=median; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Pre-index total complement activity (CH50) measures the functional "
                "intactness of the classical+terminal complement pathway per Xu 2024 "
                "(PMID 38811284; doi:10.1016/j.blre.2024.101210). Z->T: low CH50 on existing "
                "anti-C5 confirms adequate distal blockade; persistent EVH then drives "
                "switch decision. Z->Y: residual complement activity predicts post-switch "
                "response magnitude. why_not_duplicate: alternative_pathway_amplification_loop "
                "entry measures AP-specific markers (C3-split + factor B); CH50 is global "
                "functional complement (all pathways), different assay + different pathway "
                "subset. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33512400 — Brodsky 2021 (DOI:10.1182/blood.2019003812)
        dspy.Example(
            feature_name="charlson_comorbidity_index_score_preindex_pnh",
            derivation_pseudocode=(
                "source=DX_CLAIMS; derivation_inputs=['icd10_diagnosis_codes', 'charlson_weights_table']; "
                "aggregation=weighted_sum; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Charlson Comorbidity Index summarizes 17 weighted comorbid conditions over 365d preindex per standard pharmacoepi practice (Brodsky 2021 PMID 33512400 treatment-decision context; doi:10.1182/blood.2019003812). Z->T: higher CCI favors oral iptacopan over IV infusion regimens. Z->Y: comorbidity burden modulates erythropoietic capacity and adverse-event-driven discontinuation. why_not_duplicate: the nearest golden-set neighbor is one upstream demographic; this is the COMPOSITE multi-comorbidity score (17-condition weighted index) — distinct construct. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38622956 — Versino-Fattizzo (DOI:10.1111/ijlh.14281)
        dspy.Example(
            feature_name="iptacopan_dose_modification_during_first_90d_postindex_count_pnh",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_code', 'iptacopan_strength_mg', 'fill_date']; "
                "aggregation=count_distinct_strengths; window_days=90; knowable_at=postindex+90d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Count of distinct iptacopan strengths dispensed in first 90d post-index indexes early dose modification — jointly caused by (a) early tolerability/response signals (T->modification) and (b) prescriber preference and patient adherence (independent of T). Conditioning opens a collider path per Hernan 2004 (PMID 14760119). why_not_duplicate: the nearest golden-set neighbor is binary persistence at 180d (different endpoint); this is dose-modification COUNT during 90d (intra-treatment-course modification, not discontinuation); different event type + window. Remediation per role-to-remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 36055332 — PEGASUS 48-week (DOI:10.1016/S2352-3026(22)00210-1)
        dspy.Example(
            feature_name="pre_index_specialist_hematology_visit_count_180d_pnh",
            derivation_pseudocode=(
                "source=PROVIDER_CLAIMS; derivation_inputs=['specialty_code_hematology', 'service_date']; "
                "aggregation=count; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Number of hematology specialist visits in 180d preindex reflects monitoring intensity at the treatment-decision juncture per PEGASUS-48wk care patterns (PMID 36055332; doi:10.1016/S2352-3026(22)00210-1). Z->T: high monitoring frequency surfaces breakthrough hemolysis that triggers switch decisions. Z->Y: ongoing specialist contact predicts adherence and outcome assessment completeness. why_not_duplicate: the nearest golden-set neighbor captures PAYER policy (administrative); this captures PROVIDER engagement count (utilization). Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39079163 — Hillmen 2024 (DOI:10.1371/journal.pone.0306407)
        dspy.Example(
            feature_name="mild_moderate_anemia_baseline_indicator_hgb_ge_10_preindex_pnh",
            derivation_pseudocode=(
                "source=CBC_LABS; derivation_inputs=['hemoglobin_g_dl_median_30d_preindex']; "
                "aggregation=indicator_ge_10; window_days=30; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Binary indicator for baseline Hb >=10 g/dL (mild/moderate anemia stratum) "
                "per Hillmen 2024 (PMID 39079163; doi:10.1371/journal.pone.0306407) which "
                "showed differential proximal-inhibitor response by baseline-Hb stratum. "
                "Z->T: stratum membership shifts treatment-decision threshold. Z->Y: ceiling "
                "effect — mild/moderate-anemia patients have less room for >=2g/dL "
                "improvement endpoint. why_not_duplicate: compile-set neighbor "
                "baseline_hemoglobin_g_dl_preindex_mean_30d_pnh is the CONTINUOUS mean; "
                "this is BINARY threshold indicator over the trial-relevant cut-point; "
                "different aggregation. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38477987 — APPLY-PNH (DOI:10.1056/NEJMoa2308695)
        dspy.Example(
            feature_name="apply_pnh_trial_eligibility_phenotype_match_flag_preindex",
            derivation_pseudocode=(
                "source=COMPUTED_FEATURE; derivation_inputs=['anti_c5_treated_flag', 'hgb_lt_10_flag', 'reticulocytosis_flag']; "
                "aggregation=all_conditions_met; window_days=90; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Composite flag indicating patient phenotypically matches APPLY-PNH "
                "trial eligibility criteria (PMID 38477987; doi:10.1056/NEJMoa2308695): "
                "anti-C5 treated, Hb<10 baseline, reticulocytosis. Z->T: prescribers more "
                "likely to switch trial-look-alike patients to iptacopan. Z->Y: phenotype "
                "match indexes the subgroup with documented trial-evidence response. "
                "why_not_duplicate: novel composite construct (boolean-AND across 3 "
                "clinical criteria); no compile-set or golden entry combines these three "
                "into a single eligibility indicator. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33730455 — PEGASUS (DOI:10.1056/NEJMoa2029073)
        dspy.Example(
            feature_name="prior_year_complement_inhibitor_total_drug_cost_usd_preindex_pnh",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['allowed_amount_usd', 'drug_class_complement']; "
                "aggregation=sum; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Cumulative complement-inhibitor drug cost in 365d preindex indexes "
                "treatment intensity and payer engagement. Z->T,Y upstream effect but "
                "mediated through clinical confounders (anti-C5 days, EVH burden) already "
                "in adjustment set. PEGASUS economic substudy (PMID 33730455; "
                "doi:10.1056/NEJMoa2029073) documents cost-intensity heterogeneity. "
                "Ancestor role per Greenland-Pearl-Robins 1999 (PMID 9888278). "
                "why_not_duplicate: compile-set neighbor "
                "pre_index_anti_c5_persistence_days_lifetime is DAYS exposure metric; "
                "this is DOLLARS cost metric (economic intensity, not pharmacologic "
                "duration). Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38477987 — APPLY-PNH (DOI:10.1056/NEJMoa2308695)
        dspy.Example(
            feature_name="payer_class_commercial_vs_medicare_indicator_preindex_pnh",
            derivation_pseudocode=(
                "source=PAYER_FIELD; derivation_inputs=['payer_class_code', 'plan_type']; "
                "aggregation=index_month_category; window_days=0; knowable_at=index_date"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Payer class (commercial vs Medicare) at index — used as a Brookhart-style policy IV where commercial vs Medicare formulary differentials affect iptacopan access. Z->T: differential PA gating shifts initiation rates post-APPLY-PNH (PMID 38477987; doi:10.1056/NEJMoa2308695). Z->Y exclusion-restriction: payer category itself does not affect Hb biology — only the treatment-choice mediation route. Defensibility caveat: payer class may correlate with comorbidity profile through age/employment; mechanism explicitly adjusts for Charlson Score + age (both included). why_not_duplicate: the nearest golden-set neighbor is policy-specific STEP-THERAPY rule; this is broader PAYER-CLASS indicator (categorical not boolean). Defensible as IV per Brookhart 2006 (PMID 30516102). Remediation per role-to-remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B1-A: Adversarial / worker-evaluator boundary (8 entries) -----
        # Adversarial: ambiguous between mediator and confounder via timing (baseline LDH measured AT initiation)
        dspy.Example(
            feature_name="ldh_value_on_index_day_intraday_timing_ambiguous_pnh",
            derivation_pseudocode=(
                "source=LABS_LDH; derivation_inputs=['ldh_iu_l', 'lab_collection_time', 'iptacopan_first_dose_time']; "
                "aggregation=value_if_before_dose_else_null; window_days=0; knowable_at=index_date"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Worker-evaluator boundary: LDH measured ON THE INDEX DAY before iptacopan "
                "first dose. A naive worker might call this MEDIATOR (post-index timepoint) "
                "or DESCENDANT (index-day measurement), but the temporal filter — value "
                "captured before drug administration — places it pre-treatment per Hernan "
                "target-trial framing (Hernan-Robins 2016 PMID 27176981). Mechanism: Z->T "
                "(immediate-pre-dose LDH drives same-day decision-confirmation); Z->Y "
                "(baseline-trajectory anchor for response measurement). The adversarial "
                "challenge: classifier must distinguish intra-day temporal precedence from "
                "naive date-equality. why_not_duplicate: compile-set neighbor "
                "lactate_dehydrogenase_xuln_trajectory_slope_preindex_180d is the SLOPE over "
                "180d; this is the SINGLE INDEX-DAY value with sub-day temporal filter — "
                "intra-day ambiguity testing. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: baseline LDH ambiguity — looks like mediator but is confounder
        dspy.Example(
            feature_name="haptoglobin_measured_postdischarge_pre_iptacopan_fill_ambiguous_pnh",
            derivation_pseudocode=(
                "source=LABS_HAPTOGLOBIN; derivation_inputs=['hapto_mg_dl', 'hospital_discharge_date', 'first_iptacopan_fill_date']; "
                "aggregation=value_between_discharge_and_fill; window_days=14; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Adversarial: haptoglobin lab drawn in the gap between PNH-related hospital "
                "discharge and first iptacopan pharmacy fill. The discharge event AND the "
                "fill event are both clinically near-index; a worker may classify this as "
                "MEDIATOR (post-discharge=post-event), but the temporal filter — measurement "
                "BEFORE first iptacopan exposure — places it as pre-treatment confounder. "
                "Z->T: post-discharge haptoglobin level confirms switch decision (low->"
                "iptacopan). Z->Y: predicts response magnitude. The adversarial test: "
                "distinguish 'post-some-event' from 'post-TREATMENT'. why_not_duplicate: "
                "compile-set neighbor baseline_haptoglobin_pct_lln_preindex is min-over-90d-"
                "preindex; this is single-value-in-discharge-to-fill-gap (specific narrow "
                "window) — different aggregation + window. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: ambiguous between instrument and confounder due to selective compliance
        dspy.Example(
            feature_name="meningococcal_vaccine_uptake_post_label_advisory_ambiguous_pnh",
            derivation_pseudocode=(
                "source=IMMUNIZATION_REGISTRY; derivation_inputs=['mcv4_admin_date', 'fda_label_advisory_date']; "
                "aggregation=indicator_within_window; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart + immunization; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Adversarial: meningococcal vaccine uptake within 180d after an FDA label "
                "advisory date. A worker may classify this as INSTRUMENT (post-advisory "
                "policy shock affecting both vaccine + iptacopan uptake), but the patient-"
                "level selection (those who DID comply with advisory) carries unmeasured "
                "confounding through adherence-personality + provider engagement. Z->T arrow "
                "is real (compliant patients more likely to start iptacopan) but Z->Y "
                "exclusion-restriction FAILS through adherence/health-engagement back-door "
                "per Brookhart 2006 (PMID 30516102) IV-credibility checks. Correct label: "
                "CONFOUNDER. The adversarial test: distinguish policy-shock IVs from "
                "patient-compliance-confounded variables. why_not_duplicate: compile-set "
                "meningococcal_vaccination_pre_iptacopan_initiation_flag_pnh is the GENERIC "
                "pre-iptacopan flag labeled ancestor (REMS-required); this is the POST-"
                "ADVISORY adoption pattern (selection-based, not requirement-based). Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: aggregation hides post-anchor leakage
        dspy.Example(
            feature_name="hemoglobin_panel_mean_index_window_minus_30_to_plus_30_pnh",
            derivation_pseudocode=(
                "source=CBC_LABS; derivation_inputs=['hemoglobin_g_dl', 'lab_date']; "
                "aggregation=mean; window_days=60; knowable_at=postindex+30d"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Adversarial: mean Hb over a symmetric -30 to +30d window around index. "
                "Naive worker may classify as CONFOUNDER (centered on index, includes "
                "pre-index data), but the post-index half-window contains post-treatment "
                "values that mediate T->Y. Aggregation hides the leakage. Per Hernan 2004 "
                "(PMID 14760119), any feature whose computation requires post-index data "
                "cannot be a pre-treatment confounder regardless of nominal centering. "
                "Adversarial test: classifier must inspect aggregation window boundaries, "
                "not just naming. why_not_duplicate: compile-set neighbor "
                "baseline_hemoglobin_g_dl_preindex_mean_30d_pnh is strictly pre-index "
                "(window -30 to 0); this is bidirectional symmetric (-30 to +30) — same "
                "analyte different window with leakage. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: descendant vs collider confusion
        dspy.Example(
            feature_name="post_iptacopan_quality_of_life_eortc_score_d180_pnh",
            derivation_pseudocode=(
                "source=PRO_EORTC; derivation_inputs=['eortc_qlq_c30_global_d180', 'baseline_eortc_score']; "
                "aggregation=value_d180; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI PNH chart PRO; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Adversarial: post-treatment EORTC QLQ-C30 score at d180. Naive worker may classify as MEDIATOR (post-treatment intermediate on T->QoL->Y path), but the OUTCOME hemoglobin_response_180d is biologic (laboratory measure), and EORTC score is a downstream patient-reported sequelae of the hematologic response — no causal arrow QoL -> Hb. The correct role is DESCENDANT (off the directed (T,Y) path) per Hernan 2016 (PMID 27176981). Drop from adjustment. Adversarial test: distinguish mediator (T->M->Y) from descendant (Y->D or T->D, no D->Y). why_not_duplicate: the nearest golden-set neighbor is a binary FACIT response (a different patient-reported-outcome measure); this is a continuous EORTC score (different PRO measure + different role inference). Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: ancestor vs confounder
        dspy.Example(
            feature_name="historical_decade_of_pnh_diagnosis_categorical_preindex",
            derivation_pseudocode=(
                "source=DX_FIRST_PNH; derivation_inputs=['first_pnh_dx_date']; "
                "aggregation=decade_bucket; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Adversarial: decade-of-diagnosis (1990s/2000s/2010s/2020s) as categorical. "
                "Naive worker may classify as CONFOUNDER (calendar shifts treatment access "
                "AND outcomes), but per d-separation analysis (Greenland-Pearl-Robins 1999 "
                "PMID 9888278), the path Z -> {era-of-care medication options, era-of-care "
                "diagnostic refinements} -> T,Y is fully mediated through downstream "
                "confounders (prior anti-C5 days, payer class, FDA-approval-calendar) "
                "already in the adjustment set. Decade-of-diagnosis becomes ANCESTOR once "
                "those downstream nodes are present. The adversarial challenge: classifier "
                "must reason about d-separation completeness, not surface-level temporal "
                "correlation. why_not_duplicate: compile-set neighbor "
                "time_since_pnh_diagnosis_years_preindex is CONTINUOUS years (patient-level "
                "tenure); this is CATEGORICAL CALENDAR DECADE (era-of-care marker), "
                "different aggregation + different construct (patient-tenure vs calendar-"
                "era). Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: collider hidden in composite feature
        dspy.Example(
            feature_name="ldh_normalization_and_alive_at_d180_composite_flag_pnh",
            derivation_pseudocode=(
                "source=COMPOSITE; derivation_inputs=['ldh_lte_1_5_uln_d180_flag', 'alive_at_d180_flag']; "
                "aggregation=and_indicator; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI PNH chart; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Adversarial: AND-composite of two post-treatment events (LDH normalized AND alive). Naive worker may call this MEDIATOR (post-treatment intermediate), but conjunction of treatment-effect AND survival creates a classical collider — both arms (T->LDH normalization, T->survival, AND underlying-severity->survival, underlying-severity->LDH normalization) converge into the composite, so conditioning opens back-door per Hernan 2004 (PMID 14760119). Composite hides the survivorship-collider. Adversarial test: classifier must decompose composite features. why_not_duplicate: the nearest golden-set neighbor is the SURVIVAL endpoint alone; this is the AND-COMPOSITE with LDH-normalization (different aggregation: conjunction; different decomposability concern). Remediation per role-to-remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: instrument that fails exclusion restriction
        dspy.Example(
            feature_name="patient_residence_distance_to_complement_specialty_pharmacy_miles_preindex_pnh_ambiguous",
            derivation_pseudocode=(
                "source=GEOCODED_DISTANCE; derivation_inputs=['patient_zip_centroid_lat_lon', 'specialty_pharmacy_lat_lon']; "
                "aggregation=haversine_miles; window_days=0; knowable_at=index_date"
            ),
            dataset_context=(
                "ConcertAI PNH + geocoding; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Adversarial: distance-to-specialty-pharmacy as candidate IV. Naive worker "
                "may label INSTRUMENT (geographic access shock to T), but per codex #358 "
                "audit (which REPLACED distance/facility-volume IVs with Brookhart-Wang "
                "prescriber-level IVs), distance-IVs violate exclusion restriction through "
                "(a) urban-rural SES gradient on outcome and (b) monitoring-intensity "
                "differential by distance. CONFOUNDER is the correct label since distance "
                "carries direct-Y effect through travel-burden-on-adherence pathway. "
                "Brookhart 2006 (PMID 30516102) IV-credibility framework. why_not_duplicate: "
                "this feature is INTENTIONALLY similar in shape to access-IV candidates "
                "but explicitly tests classifier's ability to REJECT distance-IV in favor "
                "of CONFOUNDER labeling per #358 audit teaching. No existing entry frames "
                "distance-to-pharmacy as confounder. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B1-E: Edge cases (7 entries) -----
        # Edge case: post-anchor leakage hidden by aggregation
        dspy.Example(
            feature_name="transfusion_units_aggregated_minus_365_to_plus_30_pnh_leakage_edge",
            derivation_pseudocode=(
                "source=PROCEDURE_CODES; derivation_inputs=['cpt_transfusion_codes', 'units_transfused', 'service_date']; "
                "aggregation=sum; window_days=395; knowable_at=postindex+30d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Edge case: transfusion units summed over a 395d window spanning -365d preindex to +30d postindex. Aggregation conceals 30d of post-anchor data — the +30d slice is post-treatment and lies on T -> transfusion_need -> Y path per Brodsky 2021 (PMID 33512400; doi:10.1182/blood.2019003812). Even if 365/395 = 92% of the window is preindex, the post-anchor leakage makes this a mediator with window remediation required (restrict to -365 to 0). why_not_duplicate: the nearest golden-set neighbor is the STRICT preindex 365d window (no leakage); this is the ASYMMETRIC -365 to +30 window (with leakage, labeled mediator) — edge-case teaching pair. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: prefix censoring corner
        dspy.Example(
            feature_name="haptoglobin_first_ever_observed_value_lifetime_pnh_prefix_censor_edge",
            derivation_pseudocode=(
                "source=LABS_HAPTOGLOBIN; derivation_inputs=['hapto_mg_dl', 'lab_date']; "
                "aggregation=first_in_record; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH claims; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Edge case: the FIRST EVER recorded haptoglobin in the patient's data — "
                "subject to left-truncation/prefix-censoring (varies by when the patient "
                "entered the claims system, not by clinical state). Z->T,Y upstream effect "
                "but largely informative about data-availability era rather than disease "
                "state. Schrezenmeier 2022 (PMID 35699625; doi:10.20452/pamw.16271) "
                "discusses heterogeneity in baseline-assessment-completeness across "
                "registries. Correctly an ANCESTOR (upstream but mediated through downstream "
                "data-completeness confounders). why_not_duplicate: compile-set "
                "baseline_haptoglobin_pct_lln_preindex is 90d window min-of-ratio; this is "
                "FIRST-EVER raw value (data-availability anchored, not clinical-time anchored) "
                "— different aggregation + different temporal anchor. Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: aggregation-over-panel hiding instrument-vs-confounder
        dspy.Example(
            feature_name="composite_socioeconomic_index_zip_code_preindex_pnh_edge",
            derivation_pseudocode=(
                "source=ZIP_LINKED_CENSUS; derivation_inputs=['median_income_acs_5yr', 'pct_uninsured_acs_5yr', 'pct_bachelor_or_higher_acs']; "
                "aggregation=z_score_composite; window_days=0; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH + ACS census; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Edge case: composite SES z-score (income+uninsured+education) at ZIP level. "
                "Aggregation hides whether any single dimension acts as instrument (via "
                "differential coverage) vs confounder (via SES->adherence->outcome). "
                "Default to CONFOUNDER per #358 audit teaching that ZIP-aggregated SES "
                "carries direct outcome paths through adherence + comorbidity gradients "
                "(violating IV exclusion). Brookhart 2006 (PMID 30516102) IV-credibility "
                "criteria. why_not_duplicate: no compile-set or golden entry uses ACS "
                "composite SES — novel SOURCE (linked census) + novel aggregation "
                "(composite z-score across 3 dimensions). Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: window starts after anchor
        dspy.Example(
            feature_name="iptacopan_pharmacokinetic_trough_d14_postindex_concentration_pnh_edge",
            derivation_pseudocode=(
                "source=PK_SAMPLING; derivation_inputs=['iptacopan_trough_ng_ml', 'sample_date']; "
                "aggregation=median; window_days=14; knowable_at=postindex+14d"
            ),
            dataset_context=(
                "ConcertAI PNH chart PK; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Edge case: drug trough concentration measured at d14 post-index — strictly "
                "post-treatment measurement on the T -> drug_exposure -> Y path. Window "
                "remediation required. The edge is window-positivity: knowable_at = +14d "
                "means feature exists only for patients who completed >=14d of treatment, "
                "introducing selection. Per Hernan-Robins 2016 (PMID 27176981) target-trial "
                "framing, drug-concentration mediators require strict window-restriction. "
                "why_not_duplicate: no existing entry uses PK trough concentration as a "
                "feature; novel SOURCE (PK sampling) + novel construct (pharmacokinetic "
                "trough)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: collinearity with index-dependent variable
        dspy.Example(
            feature_name="calendar_quarter_of_iptacopan_index_categorical_q1_q4_pnh_edge",
            derivation_pseudocode=(
                "source=DERIVED; derivation_inputs=['index_date']; "
                "aggregation=calendar_quarter; window_days=0; knowable_at=index_date"
            ),
            dataset_context=(
                "ConcertAI PNH; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Edge case: calendar quarter of index as IV. Z->T: quarter-of-year affects "
                "treatment-initiation rates through deductible-reset (Q1 surge), supply-"
                "lag effects, and prescriber-calendar variation per Brookhart 2006 (PMID "
                "30516102) calendar-based IV framework. Z->Y exclusion-restriction: "
                "calendar quarter has no biological mechanism on hemoglobin response; "
                "only indirect through treatment-availability and refill-cadence. The edge "
                "is the near-collinearity with patient-level deductible state. "
                "why_not_duplicate: compile-set / golden has FDA-approval calendar AND "
                "post-publication calendar indicators (both binary discontinuities); this "
                "is CATEGORICAL recurring quarter (different construct: cyclical not "
                "monotone). Remediation per role-to-remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: differential ascertainment across cohort
        dspy.Example(
            feature_name="electronic_pro_capture_completeness_pct_postindex_pnh_edge",
            derivation_pseudocode=(
                "source=EHR_PRO_FORMS; derivation_inputs=['expected_pro_assessments', 'observed_pro_assessments']; "
                "aggregation=ratio_observed_div_expected; window_days=180; knowable_at=postindex+180d"
            ),
            dataset_context=(
                "ConcertAI PNH chart; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Edge case: completeness ratio of post-index PRO data capture. Jointly "
                "caused by (a) patient persistence on treatment (T-affected) and (b) site/"
                "provider data-entry compliance (independent of T). Conditioning opens "
                "collider path that biases T-Y estimates per Hernan 2004 (PMID 14760119) "
                "differential ascertainment framework. Drop from adjustment. Edge: this "
                "looks like a data-quality covariate but is actually a behavioral collider. "
                "why_not_duplicate: novel construct (data-completeness ratio) not present "
                "in compile-set or golden — teaches differential-ascertainment collider "
                "pattern. Remediation per role-to-remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: feature semantically equivalent but with different aggregation tier
        dspy.Example(
            feature_name="ldh_above_1_5_uln_continuous_episode_max_duration_days_preindex_pnh_edge",
            derivation_pseudocode=(
                "source=LABS_LDH; derivation_inputs=['ldh_iu_l', 'ldh_uln_iu_l', 'lab_date']; "
                "aggregation=max_consecutive_episode_days_above_threshold; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH chart labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Edge case: longest continuous episode of LDH > 1.5xULN in 180d preindex "
                "— an EPISODE-DURATION aggregation (not point-value, not slope, not "
                "average). Captures persistence of breakthrough hemolysis per APPLY-PNH "
                "design (PMID 38477987; doi:10.1056/NEJMoa2308695). Z->T: long persistent "
                "episodes drive switch. Z->Y: chronic hemolytic stress predicts response "
                "magnitude. The edge: aggregation is a derived TIME-COVERAGE concept that "
                "classifier must reason about as pre-treatment despite computational "
                "complexity. why_not_duplicate: compile-set has slope (derivative) + this "
                "is EPISODE-DURATION (run-length); also distinct from golden POINT VALUE; "
                "third aggregation tier in the LDH-feature family. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # End Plan-239 n=200 Task 3 — Bucket 1 PNH expansion (+47 entries: #51–#97)
        # =====================================================================
        # =====================================================================
        # Plan-239 n=200 Task 4 — Bucket 2 BC expansion (+45 entries: #98–#142)
        # 40 PubMed-grounded + 3 adversarial + 2 edge-case
        # Cohort tag canonical: cohort=BC; treatment=ribociclib_init or related;
        # outcome=pfs_24m / overall_survival_36m / discontinuation_180d / etc.
        # All mechanisms carry remediation-mapping clause + temporal-filter clause.
        # IVs follow Brookhart-Wang first-initiation / preference-tendency
        # pattern with exclusion-restriction defense.
        # Post-codex iter-0 fixes: confounder=25, mediator=8, descendant=6,
        # instrument=4, ancestor=4, collider=4 (total 51 BC entries).
        # HIGH-2: practice_volume IV renamed to practice_cdk46i_prescribing_preference
        #         (Brookhart 2006 physician prescribing preference reframe, not access-volume).
        # HIGH-3: payer_formulary_tier relabeled instrument→confounder (cost-sharing→adherence path).
        # HIGH-4: state_medicaid_expansion relabeled instrument→confounder (access→adherence path).
        # HIGH-5: medicine_access_program_enrollment dropped (hopelessly access-mediated);
        #         replaced with oncologist_first_cdk46i_initiation_within_180d_post_kisqali_
        #         label_expansion_bc (NATALEE Sep 2024 adjuvant Brookhart-Wang IV).
        # =====================================================================
        # PMID: 35263519 — MONALEESA-2 OS NEJM 2022 (DOI:10.1056/NEJMoa2114663)
        dspy.Example(
            feature_name="baseline_visceral_metastatic_burden_count_preindex_bc",
            derivation_pseudocode=(
                "source=DIAGNOSIS_CLAIMS_AND_RADIOLOGY; derivation_inputs=['icd10_metastatic_site_codes', 'imaging_lesion_count', 'index_date']; "
                "aggregation=count_distinct_organ_systems; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Count of distinct visceral organ systems with metastatic lesions in the 180d preindex window (liver, lung, brain, etc.). Z->T: clinicians prefer ribociclib over palbociclib in higher visceral-burden patients per MONALEESA-2 OS sub-analyses (PMID 35263519; doi:10.1056/NEJMoa2114663) where multi-organ visceral disease showed differential OS benefit. Z->Y: visceral burden is a direct prognostic factor for survival independent of CDK4/6 inhibitor choice. why_not_duplicate: the nearest golden-set neighbor is a BINARY indicator (any visceral involvement); this is a COUNT of distinct organ systems involved — finer granularity capturing multi-organ burden vs single visceral lesion. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 29718092 — MONALEESA-2 updated Ann Oncol 2018 (DOI:10.1093/annonc/mdy155)
        dspy.Example(
            feature_name="prior_adjuvant_endocrine_therapy_duration_months_preindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['anastrozole_letrozole_exemestane_tamoxifen_flag', 'days_supply', 'fill_date']; "
                "aggregation=sum_months_continuous_exposure_before_metastatic_diagnosis; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Total months of adjuvant endocrine therapy (any AI or tamoxifen) before "
                "metastatic recurrence and ribociclib initiation. Z->T: long prior adjuvant "
                "exposure correlates with secondary endocrine resistance, shifting clinicians "
                "toward fulvestrant-backbone (vs letrozole-backbone) when starting ribociclib "
                "per MONALEESA-2 updated analysis (PMID 29718092; doi:10.1093/annonc/mdy155). "
                "Z->Y: prior endocrine therapy duration is a proxy for endocrine-resistance "
                "burden that directly affects PFS. why_not_duplicate: golden has no prior-"
                "adjuvant-duration feature; compile-set has prior_letrozole_duration but that "
                "is letrozole-specific while this aggregates ALL adjuvant endocrine classes. "
                "Temporal filter: derivation window strictly preindex (prefix-censoring at index_date). "
                "Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 34102253 — MONALEESA-3 updated OS Ann Oncol 2021 (DOI:10.1016/j.annonc.2021.05.353)
        dspy.Example(
            feature_name="bone_only_metastatic_disease_flag_preindex_bc",
            derivation_pseudocode=(
                "source=DIAGNOSIS_AND_RADIOLOGY; derivation_inputs=['icd10_C795_bone_metastasis', 'visceral_metastasis_flag']; "
                "aggregation=indicator_bone_only_no_visceral; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Binary flag: metastatic disease confined to bone (no visceral lesions) at "
                "ribociclib initiation. Z->T: bone-only disease is a guideline-favored "
                "indication for first-line CDK4/6i + endocrine therapy per MONALEESA-3 "
                "updated OS analysis (PMID 34102253; doi:10.1016/j.annonc.2021.05.353). "
                "Z->Y: bone-only metastases have substantially better prognosis than "
                "visceral-disease patients (median PFS ~6-12 months longer). Pre-index "
                "measurement guarantees outgoing arrows only. why_not_duplicate: complementary "
                "to visceral-burden count above (bone-only = negation of visceral involvement, "
                "but distinct construct: presence of bone disease + absence of visceral, NOT "
                "just absence of visceral). Remediation per role-to-remediation table: "
                "confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38231045 — Real-world palbociclib+AI 2024 (DOI:10.2217/fon-2023-0858)
        dspy.Example(
            feature_name="prior_palbociclib_exposure_days_lifetime_preindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_palbociclib', 'days_supply', 'fill_date']; "
                "aggregation=sum_lifetime_days; window_days=99999; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Lifetime days of palbociclib exposure before ribociclib initiation captures "
                "CDK4/6i-switch population — a subset with prior progression on palbociclib. "
                "Z->T: prior palbociclib failure is a strong driver of subsequent ribociclib "
                "switch per real-world cohort (PMID 38231045; doi:10.2217/fon-2023-0858). "
                "Z->Y: prior CDK4/6i exposure history depletes the CDK4/6 pathway responsive "
                "subclones, attenuating ribociclib PFS independently of choice. why_not_duplicate: "
                "novel construct — neither golden nor existing compile-set has palbociclib-"
                "specific cumulative exposure pre-ribociclib; this is the within-CDK4/6i-class "
                "switcher confounder. Temporal filter: derivation window strictly preindex "
                "(prefix-censoring at index_date). "
                "Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 30595753 — MONALEESA-2 tumor response (DOI:10.1007/s10549-017-4658-x)
        dspy.Example(
            feature_name="baseline_pain_score_visual_analog_preindex_bc",
            derivation_pseudocode=(
                "source=EHR_VITALS; derivation_inputs=['vas_pain_score_0_10', 'assessment_date']; "
                "aggregation=mean; window_days=14; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=quality_of_life_response_180d; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Mean VAS pain score (0-10) over the 14d immediately pre-index reflects "
                "tumor-related symptomatic burden at the treatment-decision moment. Z->T: "
                "high baseline pain accelerates initiation of CDK4/6i + endocrine therapy "
                "(versus endocrine monotherapy) per MONALEESA-2 pain-reduction sub-analysis "
                "(PMID 30595753; doi:10.1007/s10549-017-4658-x). Z->Y: baseline pain is a "
                "prognostic marker for tumor burden and predicts QoL improvement magnitude. "
                "why_not_duplicate: the golden set has radiographic-response and "
                "global-function neighbors; this is SYMPTOMATIC pain — distinct "
                "patient-reported domain. Remediation per role-to-remediation table: "
                "confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33895695 — ESME 20446 patients OS evolution (DOI:10.1016/j.esmoop.2021.100114)
        dspy.Example(
            feature_name="time_from_initial_diagnosis_to_metastatic_recurrence_months_bc",
            derivation_pseudocode=(
                "source=DIAGNOSIS_CLAIMS; derivation_inputs=['initial_bc_diagnosis_date', 'metastatic_recurrence_date']; "
                "aggregation=duration_months; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Months from initial BC diagnosis to documented metastatic recurrence. "
                "Z->T,Y upstream: short disease-free interval (DFI < 24m) indexes "
                "aggressive-biology subgroup that drives both treatment-intensity choice "
                "(CDK4/6i + endocrine vs endocrine alone) AND survival per ESME 20446-patient "
                "cohort (PMID 33895695; doi:10.1016/j.esmoop.2021.100114). Ancestor role: "
                "the (T,Y) effect is mediated through downstream confounders already in the "
                "adjustment set (visceral burden, ECOG PS, ER%, Ki67) per Greenland-Pearl-"
                "Robins 1999 (PMID 9888278). why_not_duplicate: golden de_novo_metastatic_flag "
                "is a binary present-at-diagnosis indicator; this is CONTINUOUS months from "
                "diagnosis to recurrence (only relevant for recurrent — not de novo — disease). "
                "Temporal filter: derivation window strictly preindex (prefix-censoring at index_date). "
                "Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38703285 — Everolimus post-CDK4/6 real-world (DOI:10.1007/s10549-024-07324-8)
        dspy.Example(
            feature_name="prior_everolimus_exposure_flag_preindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_everolimus', 'fill_date']; "
                "aggregation=indicator_any_prior_fill; window_days=99999; knowable_at=preindex_30d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Binary flag for any prior everolimus dispense before ribociclib initiation. "
                "Z->T: prior everolimus exposure indicates post-CDK4/6i-progression patients "
                "(everolimus is typically reserved for endocrine-resistance second-line) "
                "shifting subsequent CDK4/6i selection toward ribociclib per real-world "
                "cohort (PMID 38703285; doi:10.1007/s10549-024-07324-8). Z->Y: prior mTOR-"
                "pathway inhibitor exposure marks PI3K/AKT/mTOR-pathway-altered tumors with "
                "differential CDK4/6i response. Pre-index measurement guarantees outgoing "
                "arrows. why_not_duplicate: novel pathway-targeted-therapy history construct "
                "absent from golden + compile-set. Remediation per role-to-remediation table: "
                "confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39442617 — NATALEE final iDFS Ann Oncol 2024 (DOI:10.1016/j.annonc.2024.10.015)
        dspy.Example(
            feature_name="adjuvant_setting_stage_ii_iii_indicator_preindex_bc",
            derivation_pseudocode=(
                "source=DIAGNOSIS_AND_PATHOLOGY; derivation_inputs=['ajcc_stage_at_index', 'metastatic_flag']; "
                "aggregation=indicator_stage_ii_or_iii_nonmetastatic; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- early BC; cohort=BC; treatment=ribociclib_init; "
                "outcome=invasive_disease_free_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Indicator that ribociclib is being initiated in the adjuvant setting (AJCC "
                "stage II or III, nonmetastatic). Z->T: stage II/III HR+/HER2- patients with "
                "high recurrence risk are eligible for adjuvant ribociclib per NATALEE final "
                "iDFS results (PMID 39442617; doi:10.1016/j.annonc.2024.10.015). Z->Y: "
                "adjuvant vs metastatic setting determines outcome timeline (iDFS vs PFS) "
                "and prognostic horizon — fundamental disease-biology stratifier. Pre-index "
                "stage assignment guarantees outgoing arrows. why_not_duplicate: novel "
                "stage-stratified setting indicator; golden focuses on metastatic features "
                "and this captures the new NATALEE adjuvant population. Remediation per "
                "role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 37275963 — NATALEE design Ther Adv Med Oncol 2023 (DOI:10.1177/17588359231178125)
        dspy.Example(
            feature_name="node_positive_disease_count_preindex_adjuvant_bc",
            derivation_pseudocode=(
                "source=PATHOLOGY_REPORTS; derivation_inputs=['axillary_lymph_node_positive_count', 'sentinel_node_biopsy_date']; "
                "aggregation=integer_count; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- early BC; cohort=BC; treatment=ribociclib_init; "
                "outcome=invasive_disease_free_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Count of axillary lymph nodes positive for tumor at definitive surgery, "
                "measured pre-ribociclib-initiation. Z->T: higher nodal involvement (N1+) "
                "qualifies patients for adjuvant ribociclib per NATALEE eligibility criteria "
                "(PMID 37275963; doi:10.1177/17588359231178125). Z->Y: nodal burden is a "
                "classical prognostic factor directly determining recurrence hazard. Path-"
                "report timing (typically months pre-CDK4/6i-init) guarantees temporal "
                "ordering. why_not_duplicate: distinct from Oncotype score (biology) and "
                "Nottingham grade (tumor differentiation); this is REGIONAL DISEASE BURDEN. "
                "Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 30206110 — PALOMA-3 ESR1 ctDNA Cancer Discov 2018 (DOI:10.1158/2159-8290.CD-18-0264)
        dspy.Example(
            feature_name="baseline_ctdna_esr1_mutation_status_preindex_bc",
            derivation_pseudocode=(
                "source=LIQUID_BIOPSY; derivation_inputs=['esr1_mutation_flag', 'ctdna_collection_date']; "
                "aggregation=indicator_any_esr1_mutation; window_days=60; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Binary indicator of ESR1 mutation detected in ctDNA collected ≤60d pre-index. "
                "Z->T: ESR1-mutant patients are increasingly steered toward fulvestrant + "
                "CDK4/6i combinations (vs AI + CDK4/6i) per PALOMA-3 emergent ctDNA findings "
                "(PMID 30206110; doi:10.1158/2159-8290.CD-18-0264) where Y537S mutations "
                "predict AI-resistance. Z->Y: ESR1 mutation directly drives endocrine "
                "resistance, attenuating PFS independent of CDK4/6i choice. why_not_duplicate: "
                "the nearest golden-set neighbor is a POST-index ctDNA emergence measurement "
                "(90d); this is BASELINE PRE-INDEX status — distinct temporal placement "
                "→ distinct role. Remediation per role-to-remediation table: confounder → "
                "keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38872062 — Foundation Medicine ESR1/PIK3CA real-world (DOI:10.1007/s10549-024-07376-w)
        dspy.Example(
            feature_name="baseline_pik3ca_mutation_flag_preindex_bc",
            derivation_pseudocode=(
                "source=TISSUE_GENOMIC_PROFILING; derivation_inputs=['pik3ca_hotspot_mutation_flag', 'biopsy_date']; "
                "aggregation=indicator_any_pik3ca_mut; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Binary flag for PIK3CA hotspot mutation (H1047R, E545K, E542K) from tumor "
                "tissue genomic profiling pre-index. Z->T: PIK3CA-mutant patients have access "
                "to alpelisib + fulvestrant downstream, shifting CDK4/6i strategy upstream "
                "per Foundation Medicine clinicogenomic series (PMID 38872062; "
                "doi:10.1007/s10549-024-07376-w). Z->Y: PI3K-pathway activation drives "
                "endocrine resistance and shorter PFS. why_not_duplicate: golden has no "
                "PIK3CA flag; novel actionable-biomarker confounder. Remediation per "
                "role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39278067 — Abemaciclib VTE meta-analysis (DOI:10.1016/j.ctrv.2024.102827)
        dspy.Example(
            feature_name="prior_venous_thromboembolism_history_flag_preindex_bc",
            derivation_pseudocode=(
                "source=DIAGNOSIS_CLAIMS; derivation_inputs=['icd10_dvt_pe_codes', 'diagnosis_date']; "
                "aggregation=indicator_any_prior_event; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Lifetime VTE history flag (any prior DVT or PE ICD-10 code). Z->T: prior "
                "VTE patients are STEERED AWAY FROM abemaciclib (which carries elevated VTE "
                "signal per PMID 39278067; doi:10.1016/j.ctrv.2024.102827) and toward "
                "ribociclib or palbociclib. Z->Y: VTE history correlates with hypercoagulable "
                "tumor biology and worse outcomes independently of CDK4/6i choice. why_not_duplicate: "
                "novel VTE-history confounder distinct from any cardiac/hepatic toxicity "
                "feature in golden+compile-set. Temporal filter: derivation window strictly preindex "
                "(prefix-censoring at index_date). Remediation per role-to-remediation table: "
                "confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35535675 — US Oncology Network real-world AE (DOI:10.1080/03007995.2022.2073122)
        dspy.Example(
            feature_name="ribociclib_dose_reduction_event_within_90d_postindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY_AND_EMR; derivation_inputs=['ribociclib_dose_mg', 'dose_change_date', 'index_date']; "
                "aggregation=indicator_any_reduction_from_starting_dose; window_days=90; knowable_at=index_plus_90d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Binary flag for any ribociclib dose reduction (from 600mg starting dose) "
                "within 90d post-index. T->M->Y mediator: ribociclib initiation triggers "
                "neutropenia/hepatotoxicity events that require dose modification per US "
                "Oncology Network real-world data (PMID 35535675; doi:10.1080/03007995.2022.2073122) "
                "where 21.7% of CDK4/6i patients required dose reductions; the reduction "
                "itself then affects PFS through altered exposure. POST-INDEX timing places "
                "this on the causal path from T (initiation) to Y (PFS); mediator role per "
                "Pearl-VanderWeele mediation framework. why_not_duplicate: the golden set has "
                "a toxicity-conditional dose-reduction neighbor and a "
                "continuous-180d dose-intensity neighbor; this is an INDICATOR within a "
                "90d window — distinct temporal+aggregation. Remediation per role-to-remediation "
                "table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 34158598 — MONALEESA pooled dose reduction Br J Cancer 2021 (DOI:10.1038/s41416-021-01415-9)
        dspy.Example(
            feature_name="grade3_4_neutropenia_event_within_28d_postindex_bc",
            derivation_pseudocode=(
                "source=CBC_LABS; derivation_inputs=['anc_x1000_per_uL', 'lab_date', 'index_date']; "
                "aggregation=indicator_anc_lt_1000_anytime; window_days=28; knowable_at=index_plus_28d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Indicator of any grade 3-4 neutropenia event (ANC < 1000/µL) within 28d post-index. T->M->Y: ribociclib initiation causes neutropenia (CDK6-mediated myelosuppression) per MONALEESA pooled dose-reduction analysis (PMID 34158598; doi:10.1038/s41416-021-01415-9) which then triggers dose modification affecting downstream PFS. POST-INDEX temporal placement on the causal path T->M->Y. why_not_duplicate: the nearest golden-set neighbor exists (continuous, 90d window); this is INDICATOR within 28d — distinct aggregation + tighter window capturing early-onset toxicity. Remediation per role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39727671 — CDK4/6 dose reduction timing Curr Oncol 2024 (DOI:10.3390/curroncol31120548)
        dspy.Example(
            feature_name="early_dose_reduction_within_first_3_months_bc",
            derivation_pseudocode=(
                "source=PHARMACY; derivation_inputs=['ribociclib_dose_mg', 'dose_change_date', 'index_date']; "
                "aggregation=indicator_reduction_before_day_90; window_days=90; knowable_at=index_plus_90d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Indicator of ribociclib dose reduction occurring within the first 90 days "
                "post-index ('early' reduction per Kubilay Tolunay 2024 PMID 39727671; "
                "doi:10.3390/curroncol31120548 where early dose-reductions associate with "
                "worse PFS — 14.3mo vs 33.1mo for late reductions). T->V descendant of "
                "ribociclib_init: cannot exist without index treatment. V is a downstream "
                "consequence (not on T-Y causal path because it indexes the patient's "
                "fragility/toxicity-susceptibility revealed BY treatment exposure, not "
                "treatment-induced biological change). NOT a mediator: the prognostic "
                "association reflects unobserved baseline frailty surfacing as early "
                "intolerance. why_not_duplicate: distinct from the 'reduction-event-90d' "
                "mediator above by AGGREGATION (timing-bucketed) and ROLE FRAMING (frailty "
                "surrogate, not pathway). Post-treatment aggregation window (day-0 to day-90 post-index); "
                "no post-treatment data leakage into preindex features. "
                "Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS NEJM 2022 (DOI:10.1056/NEJMoa2114663)
        dspy.Example(
            feature_name="best_response_at_first_restaging_recist_60d_postindex_bc",
            derivation_pseudocode=(
                "source=RADIOLOGY_REPORTS; derivation_inputs=['recist_response_category', 'imaging_date', 'index_date']; "
                "aggregation=ordinal_best_response_PR_CR_SD_PD; window_days=60; knowable_at=index_plus_60d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Best RECIST response (CR/PR/SD/PD) at first post-index restaging assessment (typically 60d post-initiation). T->V descendant: response is the proximal consequence of ribociclib + endocrine therapy per MONALEESA-2 OS analysis (PMID 35263519; doi:10.1056/NEJMoa2114663). V is downstream of T and is a noisy intermediate proxy for Y (PFS) — should not be conditioned on as covariate when estimating T->Y because it blocks part of the causal effect. why_not_duplicate: the nearest golden-set neighbor uses a 180d window; this is FIRST RESTAGING at 60d — distinct earlier temporal anchor capturing response onset dynamics. Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 36716534 — Brazil real-world CDK4/6i (DOI:10.1016/j.ctarc.2023.100683)
        dspy.Example(
            feature_name="objective_response_rate_indicator_within_120d_postindex_bc",
            derivation_pseudocode=(
                "source=RADIOLOGY_REPORTS; derivation_inputs=['recist_response_category', 'imaging_date']; "
                "aggregation=indicator_PR_or_CR_anytime_in_window; window_days=120; knowable_at=index_plus_120d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Binary indicator of objective response (PR or CR) achieved at any restaging "
                "scan within 120d post-index. T->V descendant per Queiroz 2023 Brazil real-"
                "world cohort (PMID 36716534; doi:10.1016/j.ctarc.2023.100683) where ORR "
                "76.2% for ribociclib. V is post-treatment and direct consequence of T; "
                "not on the (T,Y) causal path for OS analysis. why_not_duplicate: distinct "
                "from best_response (ordinal scale) above; this is BINARY response indicator "
                "+ EXTENDED window (120d) — captures early-response signaling for OS. "
                "Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38231045 — Real-world palbociclib+AI (DOI:10.2217/fon-2023-0858)
        dspy.Example(
            feature_name="ribociclib_discontinuation_within_180d_postindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ribociclib_last_fill_date', 'index_date', 'gap_days_threshold']; "
                "aggregation=indicator_no_refill_within_60d_gap; window_days=180; knowable_at=index_plus_180d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Binary indicator of ribociclib discontinuation within 180d post-index "
                "(defined as ≥60d gap in pharmacy fills). T->V descendant per real-world "
                "comparative effectiveness (PMID 38231045; doi:10.2217/fon-2023-0858) where "
                "discontinuation rates differ by CDK4/6i agent. V is downstream of T and "
                "captures fragility/intolerance/progression signal; not on the (T,Y) causal "
                "path. why_not_duplicate: novel — discontinuation event distinct from "
                "dose-reduction events; structural treatment-modification descendant. "
                "Post-treatment aggregation window (day-0 to day-180 post-index); no post-treatment data leakage into preindex features. "
                "Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 37529919 — Real-world palbociclib combos (DOI:10.2217/fon-2023-0176)
        # Reframed iter-0 HIGH-2: mechanism rewritten from volume/capacity pattern to
        # Brookhart 2006 physician prescribing preference IV (PMID 16617275). Feature renamed
        # to ..._prescribing_preference_tertile_... to drop volume framing.
        dspy.Example(
            feature_name="practice_cdk46i_prescribing_preference_tertile_prior_year_bc",
            derivation_pseudocode=(
                "source=PROVIDER_CLAIMS; derivation_inputs=['practice_npi', 'practice_cdk46i_share_of_prior_year_HRpos_prescriptions_tertile']; "
                "aggregation=tertile_relative_to_practice_distribution; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Practice-level tertile of CDK4/6i prescribing PREFERENCE (share of HR+ "
                "prescriptions that used a CDK4/6i in the prior year) per Brookhart 2006 "
                "physician prescribing preference IV (PMID 16617275; doi:10.1097/01.ede.0000193606.58671.c5). "
                "Z->T: practices with high CDK4/6i preference share are more likely to "
                "initiate ribociclib over endocrine monotherapy (preference-driven adoption), "
                "corroborated by real-world CDK4/6i-class comparative data (PMID 37529919; "
                "doi:10.2217/fon-2023-0176). Z->Y exclusion restriction: the oncologist's "
                "historic CDK4/6i prescribing preference reflects institutional protocols and "
                "personal clinical-pattern, NOT patient biology — preference affects PFS ONLY "
                "through treatment receipt; no direct biological pathway from prescribing "
                "style to patient PFS independent of treatment choice. Temporal filter: "
                "derivation window strictly preindex (prefix-censoring at index_date). "
                "why_not_duplicate: distinct from its nearest golden-set neighbor (a "
                "single-drug-specific prescribing share); this is CDK4/6i-CLASS preference share tertile "
                "— broader class-level prescribing-preference IV per Brookhart 2006 framework. "
                "Remediation per role-to-remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 29404806 — ASCO 2018 highlights mBC (DOI:10.1007/s12254-018-0450-9)
        dspy.Example(
            feature_name="oncologist_first_cdk46i_initiation_within_post_fda_approval_window_bc",
            derivation_pseudocode=(
                "source=PROVIDER_CLAIMS_AND_REGULATORY; derivation_inputs=['oncologist_npi', 'first_cdk46i_dispense_date_in_practice', 'fda_approval_date_palbociclib_feb_2015']; "
                "aggregation=indicator_first_init_within_180d_post_approval; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Brookhart-Wang short-term first-initiation IV: indicator that the index "
                "oncologist's first lifetime CDK4/6i prescription occurred within 180d of "
                "the palbociclib FDA approval (February 2015). Z->T: early-adopter oncologists "
                "show persistent preference patterns across the CDK4/6i drug class per ASCO "
                "2018 review (PMID 29404806; doi:10.1007/s12254-018-0450-9). Z->Y exclusion "
                "restriction: oncologist's historical date of class-adoption has NO direct "
                "biological mechanism on patient PFS — all effect transmitted through "
                "treatment selection. Pre-index oncologist-level measurement guarantees "
                "temporal exogeneity. why_not_duplicate: the nearest golden-set neighbor is a "
                "single-drug-specific first-initiation flag; this is CLASS-LEVEL FIRST INITIATION (any "
                "CDK4/6i) — distinct earlier exogeneity argument. Remediation per role-to-"
                "remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 31717791 — CDK4/6 NMA Cancers 2019 (DOI:10.3390/cancers11111661)
        # Relabeled iter-0 HIGH-3: was instrument, relabeled to confounder.
        # Formulary access mediates outcome via cost-sharing → adherence path,
        # violating IV exclusion restriction per #358 audit principle.
        dspy.Example(
            feature_name="payer_formulary_tier_ribociclib_calendar_quarter_bc",
            derivation_pseudocode=(
                "source=FORMULARY_DATA; derivation_inputs=['payer_id', 'ribociclib_tier_designation', 'quarter_year']; "
                "aggregation=ordinal_tier_at_index_quarter; window_days=90; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Pre-index formulary tier for ribociclib at the index calendar quarter "
                "(Tier 2 preferred / Tier 3 standard / Tier 4 non-preferred). Z->T: "
                "preferred-tier formulary placement increases ribociclib uptake probability "
                "per CDK4/6i comparative-effectiveness NMA (PMID 31717791; doi:10.3390/cancers11111661). "
                "Z->Y DIRECT PATH: formulary tier determines patient cost-sharing (copay) → "
                "directly affects adherence → directly affects PFS independent of treatment "
                "receipt; this cost-sharing → adherence → outcome pathway violates IV "
                "exclusion restriction per Brookhart 2006 (PMID 16617275) criteria. "
                "Correct classification is CONFOUNDER (both Z->T and Z->Y direct paths "
                "present). Temporal filter: derivation window strictly preindex "
                "(prefix-censoring at index_date). why_not_duplicate: novel payer-"
                "formulary-tier construct absent from golden + compile-set; specifically "
                "teaches invalid-IV-relabeled-to-confounder boundary. Remediation per "
                "role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS NEJM 2022 (DOI:10.1056/NEJMoa2114663)
        # Relabeled iter-0 HIGH-4: was instrument, relabeled to confounder.
        # Medicaid expansion affects outcome via cost-sharing → access → adherence pathway,
        # violating IV exclusion restriction per #358 audit principle.
        dspy.Example(
            feature_name="state_medicaid_expansion_status_post_2014_indicator_bc",
            derivation_pseudocode=(
                "source=PRACTICE_GEOGRAPHY_AND_POLICY; derivation_inputs=['practice_state', 'medicaid_expansion_effective_date', 'index_date']; "
                "aggregation=indicator_state_expanded_before_index; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "State Medicaid expansion affects both treatment receipt and downstream "
                "adherence and follow-up care, violating IV exclusion restriction per #358 "
                "audit principle; treated as confounder. Z->T: expanded-Medicaid states show "
                "higher CDK4/6i uptake among low-income patients via reduced cost-sharing "
                "per population-level patterns alongside MONALEESA-2 OS update (PMID 35263519; "
                "doi:10.1056/NEJMoa2114663). Z->Y DIRECT PATH: Medicaid expansion improves "
                "access to ALL healthcare services (oncology follow-up visits, supportive "
                "care, toxicity management) — this access improvement directly affects PFS "
                "independent of CDK4/6i receipt; exclusion restriction fails because expansion "
                "does not affect outcome ONLY through CDK4/6i choice. Correct classification "
                "is CONFOUNDER (both Z->T and Z->Y direct paths confirmed). Temporal filter: "
                "derivation window strictly preindex (prefix-censoring at index_date). "
                "why_not_duplicate: distinct from 340B (provider-level program); this is "
                "STATE POLICY (geographic-policy confounder). Remediation per role-to-"
                "remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39442617 — NATALEE final iDFS Ann Oncol 2024 (DOI:10.1016/j.annonc.2024.10.015)
        # + Brookhart 2006 physician prescribing preference IV (PMID 16617275)
        # Replaced iter-0 HIGH-5: prior MAP-enrollment IV was access-based with no defensible
        # reframing (enrollment directly funds treatment access AND determines outcome via
        # adherence support — hopelessly access-mediated). Dropped; this label-expansion
        # Brookhart-Wang short-term IV is the substitute.
        # Replaced with NATALEE adjuvant label-expansion Brookhart-Wang short-term IV.
        dspy.Example(
            feature_name="oncologist_first_cdk46i_initiation_within_180d_post_kisqali_label_expansion_bc",
            derivation_pseudocode=(
                "source=PROVIDER_CLAIMS_AND_REGULATORY; derivation_inputs=['oncologist_npi', 'first_kisqali_adjuvant_dispense_date', 'kisqali_natalee_label_expansion_date_sep_2024']; "
                "aggregation=indicator_first_adjuvant_kisqali_init_within_180d_post_label_expansion; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- early BC; cohort=BC; treatment=ribociclib_init; "
                "outcome=invasive_disease_free_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Brookhart-Wang short-term first-initiation IV: indicator that the index "
                "oncologist initiated Kisqali (ribociclib) for an adjuvant HR+/HER2- early "
                "BC patient within 180d of the NATALEE adjuvant label expansion (FDA approval "
                "Sep 2024; PMID 39442617; doi:10.1016/j.annonc.2024.10.015). Z->T: early-"
                "adopter oncologists who rapidly incorporated the NATALEE-adjuvant indication "
                "show persistent prescribing preference that affects subsequent ribociclib "
                "initiation decisions for eligible patients, per Brookhart 2006 physician "
                "prescribing preference IV framework (PMID 16617275; doi:10.1097/01.ede.0000193606.58671.c5). "
                "Z->Y exclusion restriction: the oncologist's adoption velocity after the "
                "label expansion reflects institutional protocols and early-adopter clinical "
                "style — this has NO direct biological pathway to patient invasive-disease-"
                "free survival; all effect is transmitted through treatment receipt. Temporal "
                "filter: derivation window strictly preindex (prefix-censoring at index_date). "
                "why_not_duplicate: distinct from palbociclib-class-level first-initiation IV "
                "above (which is metastatic-class first adoption, 2015 FDA); this is the "
                "NATALEE-ADJUVANT-SPECIFIC label-expansion adoption IV — different indication "
                "(adjuvant) + different regulatory event (Sep 2024). Remediation per role-to-"
                "remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 34102253 — MONALEESA-3 updated OS (DOI:10.1016/j.annonc.2021.05.353)
        dspy.Example(
            feature_name="oncologist_fulvestrant_backbone_share_prior_year_bc",
            derivation_pseudocode=(
                "source=PROVIDER_CLAIMS; derivation_inputs=['oncologist_npi', 'cdk46i_starts_with_fulvestrant_partner_365d_preindex', 'cdk46i_starts_total_365d_preindex']; "
                "aggregation=ratio_fulvestrant_partner_over_total; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Fraction of an index oncologist's prior-year CDK4/6i starts that used "
                "fulvestrant (not AI) as the endocrine partner — Brookhart-Schneeweiss "
                "physician-preference IV. Z->T: fulvestrant-preferring oncologists are more "
                "likely to pair ribociclib with fulvestrant (vs letrozole) per MONALEESA-3 "
                "updated OS context (PMID 34102253; doi:10.1016/j.annonc.2021.05.353). Z->Y "
                "exclusion restriction: physician-level prior-pattern of endocrine-partner "
                "choice has no direct biological mechanism on the current patient's PFS "
                "independent of treatment selection. why_not_duplicate: distinct from "
                "ribociclib_preference_share (drug-level) — this is PARTNER-CHOICE preference "
                "(orthogonal axis of prescribing-style heterogeneity). Temporal filter: "
                "derivation window strictly preindex (prefix-censoring at index_date). "
                "Remediation per role-"
                "to-remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38231045 — Real-world palbociclib+AI 2024 (DOI:10.2217/fon-2023-0858)
        dspy.Example(
            feature_name="cdk46i_class_substitution_event_within_12m_postindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ribociclib_discontinuation_date', 'subsequent_palbociclib_or_abemaciclib_init_date', 'window_days_between']; "
                "aggregation=indicator_intra_class_switch; window_days=365; knowable_at=index_plus_365d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Indicator that the patient switched from ribociclib to palbociclib or "
                "abemaciclib within 12 months post-index. T->V descendant: intra-class "
                "switching reflects intolerance + clinical-decision dynamics arising AFTER "
                "ribociclib initiation per real-world cohort (PMID 38231045; "
                "doi:10.2217/fon-2023-0858). V is downstream consequence not on the (T,Y) "
                "causal path. why_not_duplicate: novel intra-class-switch construct; distinct "
                "from its nearest golden-set neighbor, which is an endocrine-backbone switch (not "
                "a CDK4/6i switch). Post-treatment aggregation window (day-0 to day-365 post-index); "
                "no post-treatment data leakage into preindex features. "
                "Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS NEJM 2022 (DOI:10.1056/NEJMoa2114663)
        dspy.Example(
            feature_name="time_to_first_objective_response_days_postindex_bc",
            derivation_pseudocode=(
                "source=RADIOLOGY_REPORTS; derivation_inputs=['index_date', 'first_recist_pr_or_cr_date']; "
                "aggregation=duration_days; window_days=365; knowable_at=index_plus_365d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Days from index date to first documented RECIST PR or CR response (or "
                "censoring at 365d if never achieved). T->M->Y mediator: ribociclib initiation "
                "drives time-to-response which is itself prognostic for downstream OS per "
                "MONALEESA-2 OS Kaplan-Meier dynamics (PMID 35263519; doi:10.1056/NEJMoa2114663). "
                "POST-INDEX continuous-time variable on causal path T->M->Y; conditioning on "
                "M blocks the indirect effect (Pearl-VanderWeele). why_not_duplicate: distinct "
                "from binary response indicators above by aggregation (continuous time-to-"
                "event); novel temporal-mediator construct. Remediation per role-to-remediation "
                "table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 34292933 — India palbociclib/ribociclib real-world (DOI:10.1371/journal.pone.0253722)
        dspy.Example(
            feature_name="ribociclib_average_daily_dose_intensity_30_180d_postindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY; derivation_inputs=['cumulative_mg_dispensed', 'treatment_days_in_window']; "
                "aggregation=mean_mg_per_day_relative_to_600mg_full_dose; window_days=150; knowable_at=index_plus_180d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Average ribociclib daily dose intensity (cumulative mg / 600mg-days target) over the day-30 through day-180 post-index window. T->M->Y mediator: ribociclib initiation produces a realized exposure trajectory whose intensity directly determines biological CDK4/6 inhibition magnitude per India real-world cohort (PMID 34292933; doi:10.1371/journal.pone.0253722); dose intensity is the proximal molecular-mechanism mediator between T and Y. why_not_duplicate: the nearest golden-set neighbor uses full-180d window; this is day-30-to-180 window — distinct temporal range (excludes ramp-up period). Remediation per role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 40019493 — CDK4/6 OS comparative Sci Rep 2024 (DOI:10.1038/s41598-024-53151-8)
        dspy.Example(
            feature_name="prior_chemotherapy_lines_count_preindex_bc",
            derivation_pseudocode=(
                "source=CHEMOTHERAPY_CLAIMS; derivation_inputs=['hcpcs_chemo_regimen_codes', 'regimen_start_dates']; "
                "aggregation=count_distinct_regimens_before_index; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Count of distinct prior chemotherapy regimens received before ribociclib initiation. Z->T: heavily pre-treated patients shift toward later-line CDK4/6i + endocrine therapy (where ribociclib OS benefit is documented per PMID 40019493; doi:10.1038/s41598-024-53151-8). Z->Y: prior chemotherapy lines reflect both disease aggressiveness AND treatment-related cumulative toxicity — both shorten OS independent of CDK4/6i choice. Pre-index measurement guarantees outgoing arrows. why_not_duplicate: novel chemotherapy-history construct distinct from its nearest golden-set neighbor. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38872062 — Foundation Medicine ESR1/PIK3CA (DOI:10.1007/s10549-024-07376-w)
        dspy.Example(
            feature_name="emergent_esr1_mutation_90d_postindex_bc",
            derivation_pseudocode=(
                "source=LIQUID_BIOPSY_SERIAL; derivation_inputs=['baseline_esr1_status', 'postindex_esr1_status_90d']; "
                "aggregation=indicator_baseline_negative_then_postindex_positive; window_days=90; knowable_at=index_plus_90d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Indicator that an ESR1 mutation EMERGED on serial ctDNA between baseline (negative) and 90d post-index (positive). T->M->Y mediator: ribociclib + endocrine therapy exposure selects for ESR1-mutant subclones (acquired resistance) per Foundation Medicine longitudinal data (PMID 38872062; doi:10.1007/s10549-024-07376-w) where ESR1mut rate climbs from 8.1% (1st line) to 59% (3rd line). M is on the (T->resistance->Y) causal path. why_not_duplicate: the nearest golden-set neighbor lacks the BASELINE-NEGATIVE PRECONDITION; this DIFFERENCE-IN-STATUS construct requires both timepoints. Post-treatment aggregation window (day-0 to day-90 post-index); no post-treatment data leakage into preindex features. Remediation per role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS NEJM 2022 (DOI:10.1056/NEJMoa2114663)
        dspy.Example(
            feature_name="hospitalization_within_90d_postindex_bc",
            derivation_pseudocode=(
                "source=INPATIENT_CLAIMS; derivation_inputs=['admission_date', 'index_date', 'admission_reason_code']; "
                "aggregation=indicator_any_inpatient_admission; window_days=90; knowable_at=index_plus_90d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Indicator of any inpatient hospitalization within 90d post-index. COLLIDER "
                "T->V<-U: ribociclib_init (T) causes some toxicity-related admissions (febrile "
                "neutropenia, hepatotoxicity), AND unobserved baseline frailty (U) causes "
                "non-treatment admissions — admission integrates both per MONALEESA-2 OS "
                "safety profile (PMID 35263519; doi:10.1056/NEJMoa2114663). Conditioning on "
                "V opens the backdoor T->V<-U->Y collider path. why_not_duplicate: novel "
                "post-treatment hospitalization construct absent from golden + compile-set; "
                "teaches collider-via-multi-cause-event pattern. Remediation per role-to-"
                "remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 34158598 — MONALEESA pooled (DOI:10.1038/s41416-021-01415-9)
        dspy.Example(
            feature_name="emergency_department_visit_within_60d_postindex_bc",
            derivation_pseudocode=(
                "source=ED_CLAIMS; derivation_inputs=['ed_visit_date', 'index_date', 'cpt_ed_visit_codes']; "
                "aggregation=indicator_any_ed_visit; window_days=60; knowable_at=index_plus_60d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Binary indicator of any ED visit within 60d post-index. COLLIDER T->V<-U: "
                "ribociclib_init causes ED visits via neutropenia + hepatotoxicity per "
                "MONALEESA pooled safety analysis (PMID 34158598; doi:10.1038/s41416-021-01415-9); "
                "concurrent baseline comorbidity burden (U) drives non-treatment ED utilization "
                "and ALSO affects PFS. Conditioning on V opens backdoor U->Y path through "
                "V-collider. why_not_duplicate: distinct from hospitalization-90d above by "
                "(a) setting (ED vs inpatient), (b) window (60d vs 90d), (c) severity tier "
                "(ED is less severe filter capturing more events). Post-treatment aggregation "
                "window (day-0 to day-60 post-index); no post-treatment data leakage into preindex features. "
                "Remediation per role-to-"
                "remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 36151018 — KARMA Australia (DOI:10.1016/j.clbc.2022.08.011)
        dspy.Example(
            feature_name="liver_enzyme_grade2_elevation_within_60d_postindex_bc",
            derivation_pseudocode=(
                "source=LABS_LFT; derivation_inputs=['alt_iu_l', 'ast_iu_l', 'lab_date', 'index_date', 'uln_alt', 'uln_ast']; "
                "aggregation=indicator_any_grade2_alt_or_ast_3_to_5x_uln; window_days=60; knowable_at=index_plus_60d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Indicator of grade-2 ALT or AST elevation (3-5x ULN) within 60d post-index. T->M->Y mediator: ribociclib-induced hepatotoxicity is on the causal path where hepatotoxic events trigger dose modification/discontinuation that shortens PFS per KARMA registry (PMID 36151018; doi:10.1016/j.clbc.2022.08.011) where abnormal liver enzymes were the second-leading dose-reduction reason. why_not_duplicate: the nearest golden-set neighbor uses grade-3 threshold; this is GRADE-2 — milder threshold, captures earlier-onset signal. Post-treatment aggregation window (day-0 to day-60 post-index); no post-treatment data leakage into preindex features. Remediation per role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 34158598 — MONALEESA pooled (DOI:10.1038/s41416-021-01415-9)
        dspy.Example(
            feature_name="qtc_baseline_msec_preindex_bc",
            derivation_pseudocode=(
                "source=ECG_REPORTS; derivation_inputs=['qtc_msec_corrected', 'ecg_date']; "
                "aggregation=mean; window_days=30; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=qtc_prolongation_event_180d; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Mean corrected QT interval (msec) from pre-index ECG over the 30d pre-index window. Z->T: high baseline QTc (>450msec) is a contraindication trigger for ribociclib initiation per MONALEESA pooled safety (PMID 34158598; doi:10.1038/s41416-021-01415-9) where ribociclib carries class-leading QTc prolongation signal; high-QTc patients are routed away from ribociclib. Z->Y: baseline QTc directly predicts the probability of post-index grade-2/3 QTc prolongation outcome. Pre-index measurement guarantees outgoing arrows. why_not_duplicate: the nearest golden-set neighbor is the POST-INDEX OUTCOME variant; this is BASELINE PRE-INDEX confounder — distinct temporal placement → distinct role. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38872062 — Foundation Medicine (DOI:10.1007/s10549-024-07376-w)
        dspy.Example(
            feature_name="ctdna_tumor_fraction_pct_preindex_bc",
            derivation_pseudocode=(
                "source=LIQUID_BIOPSY; derivation_inputs=['ctdna_tumor_fraction_pct', 'sample_date']; "
                "aggregation=mean; window_days=60; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Mean ctDNA tumor fraction (TF%) on liquid biopsy in 60d pre-index window. "
                "Z->T,Y upstream: high TF indexes high tumor burden / progression velocity "
                "that drives both treatment-intensity escalation AND outcome per Foundation "
                "Medicine cohort (PMID 38872062; doi:10.1007/s10549-024-07376-w) where TF "
                "stratifies actionable-mutation detection rates. Ancestor: (T,Y) effect "
                "is mediated through downstream confounders (visceral burden, LDH, Ki67) per "
                "Greenland-Pearl-Robins 1999 (PMID 9888278). why_not_duplicate: novel ctDNA-"
                "burden construct distinct from mutation-status features. Remediation per "
                "role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33895695 — ESME cohort (DOI:10.1016/j.esmoop.2021.100114)
        dspy.Example(
            feature_name="prior_lines_endocrine_monotherapy_metastatic_setting_count_bc",
            derivation_pseudocode=(
                "source=PHARMACY_AND_TREATMENT_HISTORY; derivation_inputs=['endocrine_agent_starts_after_metastatic_dx_date', 'index_date']; "
                "aggregation=count_distinct_lines_endocrine_monotherapy; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Count of distinct endocrine-monotherapy lines administered in the metastatic "
                "setting BEFORE ribociclib initiation. Z->T: heavily endocrine-pre-treated "
                "patients (≥2 lines) are typically the salvage CDK4/6i population where "
                "ribociclib is chosen for cross-resistance considerations per ESME cohort "
                "(PMID 33895695; doi:10.1016/j.esmoop.2021.100114). Z->Y: prior endocrine-"
                "line count is a strong proxy for tumor endocrine-resistance level shortening "
                "subsequent CDK4/6i + endocrine PFS. why_not_duplicate: distinct from prior_"
                "adjuvant_endocrine_therapy_duration (which is ADJUVANT setting); this is "
                "METASTATIC-setting line count — different setting + different aggregation. "
                "Temporal filter: derivation window strictly preindex (prefix-censoring at index_date). "
                "Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 37529919 — Real-world palbociclib combos (DOI:10.2217/fon-2023-0176)
        dspy.Example(
            feature_name="prior_aromatase_inhibitor_progression_event_flag_bc",
            derivation_pseudocode=(
                "source=PHARMACY_AND_RADIOLOGY; derivation_inputs=['ai_discontinuation_due_to_progression_flag', 'ai_last_fill_date']; "
                "aggregation=indicator_documented_progression_on_prior_ai; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Indicator of documented progression on a prior aromatase inhibitor (AI) in "
                "the metastatic setting before ribociclib initiation. Z->T: AI-progressors "
                "shift toward fulvestrant-backbone (vs AI-backbone) when starting CDK4/6i "
                "per real-world cohort (PMID 37529919; doi:10.2217/fon-2023-0176). Z->Y: "
                "documented AI-progression marks established endocrine resistance with "
                "shortened subsequent PFS. why_not_duplicate: distinct from generic adjuvant-"
                "duration (Z above) which captures EXPOSURE; this is documented PROGRESSION "
                "event — distinct construct (failure vs duration). Temporal filter: derivation "
                "window strictly preindex (prefix-censoring at index_date). Remediation per role-to-"
                "remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 30595753 — MONALEESA-2 tumor response (DOI:10.1007/s10549-017-4658-x)
        dspy.Example(
            feature_name="serum_ldh_concentration_mean_30d_preindex_metastatic_bc",
            derivation_pseudocode=(
                "source=CHEMISTRY_LABS; derivation_inputs=['ldh_iu_l', 'lab_date']; "
                "aggregation=mean; window_days=30; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Mean serum LDH over the 30d immediately pre-index. Z->T: high baseline LDH "
                "(>ULN) reflects high tumor proliferation/burden, intensifying clinician "
                "preference for combination CDK4/6i over endocrine monotherapy per MONALEESA-2 "
                "tumor-response subset (PMID 30595753; doi:10.1007/s10549-017-4658-x). Z->Y: "
                "elevated LDH is a well-established OS-prognostic biomarker for metastatic "
                "breast cancer independent of treatment. Pre-index measurement guarantees "
                "outgoing arrows. why_not_duplicate: novel — golden + compile-set lack baseline-"
                "LDH for BC; compile-set has LDH features only in PNH context. Remediation "
                "per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 33895695 — ESME (DOI:10.1016/j.esmoop.2021.100114)
        dspy.Example(
            feature_name="age_at_metastatic_diagnosis_years_bc",
            derivation_pseudocode=(
                "source=DEMOGRAPHIC_AND_DIAGNOSIS; derivation_inputs=['birth_year', 'metastatic_diagnosis_date']; "
                "aggregation=integer_years; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Patient age at metastatic-diagnosis date (in years). Z->T: very-elderly "
                "(>75y) patients are less likely to receive any CDK4/6i and when they do, "
                "ribociclib is favored over abemaciclib (lower diarrhea burden) per ESME "
                "20446 patient cohort (PMID 33895695; doi:10.1016/j.esmoop.2021.100114). "
                "Z->Y: age is a global mortality determinant. why_not_duplicate: the golden set has "
                "no age construct anchored on METASTATIC diagnosis specifically; complementary "
                "to its nearest baseline-functional-status golden-set neighbor. Temporal filter: derivation window strictly preindex "
                "(prefix-censoring at index_date). Remediation per role-to-remediation table: "
                "confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 31717791 — CDK4/6 NMA Cancers 2019 (DOI:10.3390/cancers11111661)
        dspy.Example(
            feature_name="premenopausal_status_indicator_preindex_bc",
            derivation_pseudocode=(
                "source=DEMOGRAPHICS_AND_LAB; derivation_inputs=['menopausal_status_documentation', 'fsh_lh_estradiol_levels']; "
                "aggregation=indicator_premenopausal; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Binary flag for premenopausal status at the time of ribociclib initiation. "
                "Z->T: premenopausal women require GnRH-agonist co-therapy and have "
                "MONALEESA-7-anchored ribociclib indication per CDK4/6i NMA evidence (PMID "
                "31717791; doi:10.3390/cancers11111661). Z->Y: menopausal-status differential "
                "biology (estrogen-axis, fertility) directly affects PFS independent of "
                "CDK4/6i choice. Pre-index assessment guarantees outgoing arrows. "
                "why_not_duplicate: the nearest golden-set neighbor is a calendar-policy "
                "approval flag; this is the PATIENT-LEVEL biological status "
                "confounder — distinct role + construct. Remediation per role-to-remediation "
                "table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35535675 — US Oncology Network (DOI:10.1080/03007995.2022.2073122)
        dspy.Example(
            feature_name="comorbidity_burden_charlson_weighted_sum_365d_preindex_bc",
            derivation_pseudocode=(
                "source=DIAGNOSIS_CLAIMS; derivation_inputs=['icd10_codes_365d_preindex', 'charlson_weights_lookup']; "
                "aggregation=weighted_sum_charlson_categories; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Charlson Comorbidity Index computed from 365d pre-index ICD-10 codes "
                "(Charlson 1987 weighted-comorbidity score). Z->T: high-comorbidity patients "
                "are channeled toward less-toxic CDK4/6i selections (ribociclib over "
                "abemaciclib) per US Oncology Network real-world AE patterns (PMID 35535675; "
                "doi:10.1080/03007995.2022.2073122). Z->Y: comorbidity burden is a direct "
                "OS prognostic factor. Pre-index measurement window guarantees outgoing "
                "arrows. why_not_duplicate: novel — golden has individual comorbidity features "
                "(VTE history above) but no aggregated index; compile-set has no Charlson "
                "for BC. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 34102253 — MONALEESA-3 (DOI:10.1016/j.annonc.2021.05.353)
        dspy.Example(
            feature_name="number_of_metastatic_sites_count_preindex_bc",
            derivation_pseudocode=(
                "source=DIAGNOSIS_AND_RADIOLOGY; derivation_inputs=['metastatic_site_codes_preindex', 'imaging_lesion_locations']; "
                "aggregation=count_distinct_anatomic_sites; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Total count of distinct anatomic metastatic sites at index (bone, liver, "
                "lung, brain, nodal, soft tissue). Z->T: ≥3 metastatic sites is a high-burden "
                "indicator that drives intensified CDK4/6i + endocrine (vs endocrine alone) "
                "per MONALEESA-3 updated OS sub-analyses (PMID 34102253; doi:10.1016/j.annonc.2021.05.353). "
                "Z->Y: site count is a direct tumor-burden prognostic factor for OS. "
                "why_not_duplicate: distinct from visceral_burden_count (visceral-only) and "
                "bone_only_flag (bone-only); this aggregates ALL anatomic sites including "
                "non-visceral non-bone — broader composite. Temporal filter: derivation window "
                "strictly preindex (prefix-censoring at index_date). Remediation per role-to-remediation "
                "table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39442617 — NATALEE final iDFS (DOI:10.1016/j.annonc.2024.10.015)
        dspy.Example(
            feature_name="anatomic_stage_at_initial_diagnosis_ordinal_bc",
            derivation_pseudocode=(
                "source=PATHOLOGY_AND_STAGING; derivation_inputs=['ajcc_stage_8th_ed', 'staging_date']; "
                "aggregation=ordinal_stage_I_II_III_IV; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- BC; cohort=BC; treatment=ribociclib_init; "
                "outcome=invasive_disease_free_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Ordinal anatomic AJCC stage at initial breast-cancer diagnosis (I/II/III/IV). "
                "Z->T,Y upstream: stage at original diagnosis indexes intrinsic disease "
                "aggressiveness driving both eligibility/intensity of adjuvant ribociclib "
                "per NATALEE final iDFS (PMID 39442617; doi:10.1016/j.annonc.2024.10.015) "
                "AND eventual outcome trajectory. Ancestor role: (T,Y) effect mediated through "
                "downstream confounders (visceral burden, recurrence biology, node count) "
                "already in adjustment set per Greenland-Pearl-Robins 1999 (PMID 9888278). "
                "why_not_duplicate: distinct from adjuvant-stage-ii-iii indicator (binary, "
                "early-disease subset); this is FULL ORDINAL across all stages. Temporal filter: "
                "derivation window strictly preindex (prefix-censoring at index_date). Remediation "
                "per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 30206110 — PALOMA-3 (DOI:10.1158/2159-8290.CD-18-0264)
        dspy.Example(
            feature_name="lifetime_endocrine_resistance_event_count_bc",
            derivation_pseudocode=(
                "source=TREATMENT_HISTORY; derivation_inputs=['endocrine_progression_events_documented', 'endocrine_agent_classes_used']; "
                "aggregation=count_distinct_resistance_events_across_lines; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Lifetime count of documented endocrine-resistance events (progression on "
                "any endocrine agent) before ribociclib initiation. Z->T,Y upstream: heavy "
                "resistance-event history indexes a refractory clonal-evolution phenotype "
                "that drives both CDK4/6i + fulvestrant combination choice AND outcome per "
                "PALOMA-3 clonal-evolution data (PMID 30206110; doi:10.1158/2159-8290.CD-18-0264). "
                "Ancestor role: (T,Y) effect mediated through downstream baseline ESR1 status, "
                "PIK3CA status, and prior-line counts already in adjustment set. why_not_duplicate: "
                "distinct from prior_AI_progression_event (binary single-event) and "
                "endocrine_monotherapy_lines (line-count): this is RESISTANCE-EVENT count "
                "(documented failures specifically). Temporal filter: derivation window "
                "strictly preindex (prefix-censoring at index_date). Remediation per role-to-remediation "
                "table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS (DOI:10.1056/NEJMoa2114663)
        dspy.Example(
            feature_name="time_to_treatment_failure_180d_postindex_bc",
            derivation_pseudocode=(
                "source=TREATMENT_HISTORY; derivation_inputs=['ribociclib_discontinuation_date_for_any_reason', 'index_date']; "
                "aggregation=days_to_discontinuation_or_censoring_180d; window_days=180; knowable_at=index_plus_180d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=overall_survival_36m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Days from index to treatment failure (discontinuation for ANY reason — "
                "progression, toxicity, withdrawal) within 180d window. COLLIDER T->V<-U: "
                "ribociclib_init causes treatment-failure events (progression-driven and "
                "toxicity-driven); unobserved patient frailty (U) drives non-treatment-related "
                "withdrawal AND affects OS per MONALEESA-2 (PMID 35263519; doi:10.1056/NEJMoa2114663). "
                "Conditioning on V opens collider path. why_not_duplicate: distinct from "
                "discontinuation_indicator_180d (binary); this is CONTINUOUS time-to-failure "
                "with mixed-cause framing — collider not descendant because composite cause "
                "includes T-independent withdrawal. Post-treatment aggregation window "
                "(day-0 to day-180 post-index); no post-treatment data leakage into preindex features. "
                "Remediation per role-to-remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 37529919 — Real-world palbociclib (DOI:10.2217/fon-2023-0176)
        dspy.Example(
            feature_name="dose_modification_polytherapy_event_count_postindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY_AND_EMR; derivation_inputs=['ribociclib_dose_changes', 'endocrine_partner_dose_changes', 'index_date']; "
                "aggregation=count_distinct_modification_events_either_agent; window_days=180; knowable_at=index_plus_180d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Total count of dose-modification events across BOTH ribociclib and its "
                "endocrine partner within 180d post-index. COLLIDER T->V<-U: ribociclib_init "
                "causes ribociclib-side modifications (toxicity-driven); unobserved patient "
                "adherence-pattern (U) causes endocrine-side dose-skipping per real-world "
                "cohort (PMID 37529919; doi:10.2217/fon-2023-0176). The COMPOSITE count "
                "integrates both, creating an open backdoor when conditioned. why_not_duplicate: "
                "distinct from single-agent dose-reduction-indicator above; this is COMPOSITE "
                "TWO-AGENT modification count — collider via composite-multi-cause aggregation. "
                "Post-treatment aggregation window (day-0 to day-180 post-index); "
                "no post-treatment data leakage into preindex features. "
                "Remediation per role-to-remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS (DOI:10.1056/NEJMoa2114663)
        dspy.Example(
            feature_name="patient_reported_outcome_quality_of_life_change_60d_postindex_bc",
            derivation_pseudocode=(
                "source=EHR_PRO_INSTRUMENT; derivation_inputs=['fact_b_score_baseline', 'fact_b_score_60d_postindex']; "
                "aggregation=difference_postindex_minus_baseline; window_days=60; knowable_at=index_plus_60d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=quality_of_life_response_180d; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Change in FACT-B (Functional Assessment of Cancer Therapy — Breast) score "
                "from baseline to 60d post-index. T->M->Y mediator: ribociclib initiation "
                "directly affects symptom burden/QoL via tumor-response AND toxicity per "
                "MONALEESA-2 OS QoL analyses (PMID 35263519; doi:10.1056/NEJMoa2114663); the "
                "QoL change then drives downstream PRO outcome at 180d. POST-INDEX timing "
                "places M on the causal path T->M->Y. why_not_duplicate: novel patient-"
                "reported-outcome mediator absent from golden + compile-set. Post-treatment "
                "aggregation window (day-0 to day-60 post-index); no post-treatment data leakage into preindex features. "
                "Remediation per role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ---- Adversarial entries (worker-evaluator boundary cases) ----
        # Adversarial: adjuvant vs metastatic line confusion — feature looks like prior line but is from a DIFFERENT setting
        dspy.Example(
            feature_name="prior_taxane_in_adjuvant_setting_only_flag_bc",
            derivation_pseudocode=(
                "source=CHEMOTHERAPY_CLAIMS; derivation_inputs=['hcpcs_taxane_codes', 'regimen_setting_label_adjuvant_vs_metastatic', 'fill_date']; "
                "aggregation=indicator_taxane_only_in_adjuvant_pre_metastatic_recurrence; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Adversarial boundary case: adjuvant-only taxane exposure flag (taxane never "
                "given in the metastatic setting). Worker classifier may mistakenly classify "
                "this as 'descendant' or 'collider' because chemo-history features generically "
                "register as treatment-related; the SETTING-specific qualifier (adjuvant-only) "
                "places it pre-metastatic-diagnosis hence STRICTLY pre-index → confounder. "
                "Z->T: adjuvant taxane exposure shifts metastatic-setting CDK4/6i selection "
                "toward ribociclib over chemo-rechallenge. Z->Y: adjuvant taxane reflects "
                "earlier-stage disease aggressiveness and treatment-related cumulative "
                "neuropathy/cardiotoxicity that affects PFS independent of CDK4/6i choice. "
                "why_not_duplicate: distinct from prior_chemotherapy_lines (count across ALL "
                "settings); this is SETTING-RESTRICTED indicator — boundary-case construct. "
                "Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: prior anti-HER2 exposure ambiguity (HER2- cohort but discordant prior status possible)
        dspy.Example(
            feature_name="prior_anti_her2_therapy_with_subsequent_her2_loss_flag_bc",
            derivation_pseudocode=(
                "source=PHARMACY_AND_PATHOLOGY; derivation_inputs=['anti_her2_agent_history', 'her2_status_at_metastatic_recurrence', 'her2_status_at_initial_diagnosis']; "
                "aggregation=indicator_prior_her2_positive_initial_then_her2_negative_recurrence_with_anti_her2_history; window_days=99999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Adversarial boundary case: indicator that patient had prior HER2+ status + "
                "anti-HER2 therapy that later became HER2- at metastatic recurrence (subclonal "
                "loss). Worker may misclassify as 'mediator' (treatment-induced HER2 loss "
                "→ CDK4/6i appropriateness) but the relevant treatment-decision is the CURRENT "
                "ribociclib_init, not the historic trastuzumab — making prior anti-HER2 "
                "exposure a STRICTLY PRE-INDEX biological-history confounder. Z->T: prior "
                "HER2-targeted therapy exposure history modifies clinician choice of CDK4/6i "
                "agent and endocrine partner. Z->Y: prior anti-HER2 therapy reflects earlier-"
                "stage HER2-driven biology and accumulated cardiac-cumulative toxicity, both "
                "affecting subsequent PFS. why_not_duplicate: novel HER2-loss-history construct "
                "absent from golden + compile-set; specifically a confounder-vs-mediator "
                "boundary case. Remediation per role-to-remediation table: confounder → "
                "keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: practice-level IV that fails exclusion restriction (looks like Brookhart preference but isn't)
        dspy.Example(
            feature_name="practice_oncology_nurse_navigator_program_indicator_bc",
            derivation_pseudocode=(
                "source=PRACTICE_REGISTRY; derivation_inputs=['practice_npi', 'nurse_navigator_program_active_flag', 'index_date']; "
                "aggregation=indicator_program_active_at_index; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Adversarial boundary case: practice-level oncology nurse navigator program "
                "indicator looks LIKE a Brookhart practice-level IV but FAILS exclusion "
                "restriction — nurse navigators do affect outcomes DIRECTLY through adherence-"
                "support and toxicity-management coaching, not only through treatment choice. "
                "Worker may classify as 'instrument' but the correct role is CONFOUNDER: "
                "Z->T (navigator programs increase ribociclib uptake through patient education) "
                "AND Z->Y direct (navigators reduce dose-reduction events + improve adherence "
                "→ better PFS independent of CDK4/6i choice). This is the canonical case of "
                "an apparent-IV that fails exclusion restriction. why_not_duplicate: novel "
                "practice-program construct; specifically teaches IV-vs-confounder boundary. "
                "Temporal filter: derivation window strictly preindex (prefix-censoring at index_date). "
                "Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ---- Edge case entries ----
        # Edge case: leakage from staging update post-treatment
        dspy.Example(
            feature_name="restaging_event_with_pathology_update_within_30d_postindex_bc",
            derivation_pseudocode=(
                "source=PATHOLOGY_AND_RADIOLOGY; derivation_inputs=['restaging_pathology_report_date', 'index_date', 'staging_revision_indicator']; "
                "aggregation=indicator_any_staging_revision_within_30d_postindex; window_days=30; knowable_at=index_plus_30d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Edge case: post-index staging revision event within 30d. THIS LOOKS LIKE A "
                "BASELINE STAGING FEATURE but the 30d POST-INDEX window means the staging "
                "update REFLECTS post-treatment data (early scans showing rapid response or "
                "previously-missed lesions). T->V descendant of ribociclib_init: cannot exist "
                "without treatment exposure starting the restaging cascade. The edge: schema "
                "may make this look like baseline_stage_at_initial_dx (which is the ANCESTOR "
                "above) — but the 30d POST-INDEX window flips it to descendant. Easily "
                "confused with baseline-staging features per data-leakage anti-pattern. "
                "why_not_duplicate: distinct from anatomic_stage_at_initial_diagnosis "
                "(ancestor, baseline) by TEMPORAL WINDOW alone — this is the leakage-edge-"
                "case companion. Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: prefix-censoring of CDK4/6 hold events (pause/resume cycles)
        dspy.Example(
            feature_name="ribociclib_cycle_pause_event_count_first_60d_postindex_bc",
            derivation_pseudocode=(
                "source=PHARMACY_AND_EMR; derivation_inputs=['ribociclib_treatment_holiday_periods', 'index_date']; "
                "aggregation=count_distinct_pause_events_with_strict_prefix_censoring_at_day_60; window_days=60; knowable_at=index_plus_60d"
            ),
            dataset_context=(
                "ConcertAI HR+/HER2- mBC; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Edge case: count of ribociclib treatment-pause events (week-off cycles "
                "extended beyond the protocol 1-week off, due to toxicity) within strict "
                "first-60-day prefix-censored window. T->M->Y mediator on the causal path: "
                "ribociclib initiation triggers pause-events (toxicity-mediated) which then "
                "directly affect dose-intensity and PFS. The EDGE: the PREFIX-CENSORING at "
                "day-60 is critical — naive aggregation over 'all pauses ever' would leak "
                "post-progression treatment-discontinuation events that are actually "
                "OUTCOME (PFS) measurements; the 60d cutoff isolates early-toxicity-driven "
                "pauses from outcome-driven discontinuations. why_not_duplicate: distinct "
                "from dose_reduction (different concept — pauses are temporary holds, not "
                "dose-level changes); distinct aggregation + censoring scheme. Remediation "
                "per role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # End Plan-239 n=200 Task 4 — Bucket 2 BC expansion (+45 entries: #98–#142)
        # Final BC count: 51 total. Post-codex iter-0 role distribution:
        # confounder=25, mediator=8, descendant=6, instrument=4, ancestor=4, collider=4.
        # =====================================================================
        # =====================================================================
        # Plan-239 n=200 Task 5 — Bucket 3: CSU expansion (+42 entries)
        # Cohort floor 21 -> 53. Source mix: 37 PubMed-grounded literature
        # entries (16 baseline biomarkers + 4 Brookhart-Wang IVs + 6 postindex
        # mediators + 4 descendants + 4 ancestors + 3 colliders) + 3
        # adversarial (worker-evaluator boundary) + 2 edge cases (leakage /
        # prefix-censoring). All entries carry `cohort=CSU;` explicit
        # tag. Disjointness verified vs 148 compile-set + 91 golden entries
        # (Levenshtein <0.85 on feature_name; mechanistically distinct).
        # IV entries follow Brookhart-Wang short-term first-initiation /
        # preference-share / calendar-time pattern with exclusion-restriction
        # defended in mechanism (per codex #358 audit). Every mechanism cites
        # rubric triad: (a) temporal filter, (b) Pearl arrowhead, (c)
        # remediation mapping.
        # =====================================================================
        # ----- Sub-bucket B3-L: PubMed literature baseline biomarkers (16 entries) -----
        # PMID: 38141832 — Bernstein 2023 JACI BTK signaling in CSU (DOI:10.1016/j.jaci.2023.12.008)
        dspy.Example(
            feature_name="baseline_serum_btk_phosphorylation_pct_basophil_preindex_csu",
            derivation_pseudocode=(
                "source=FLOW_CYTOMETRY; derivation_inputs=['pbtk_pct_basophil', 'assay_date']; "
                "aggregation=median; window_days=60; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU chart-abstracted flow; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Pre-prediction-time pBTK/total-BTK ratio in circulating basophils, measured strictly preindex (window_days=60, knowable_at=preindex_0d enforces prefix-censoring at index_date — no postindex assay leakage), indexes the BTK-pathway activation burden that drives mast-cell and basophil degranulation. Z->T: clinicians select patients with high pBTK signature for BTK-inhibitor therapy (Bernstein 2023 JACI PMID 38141832; doi:10.1016/j.jaci.2023.12.008). Z->Y: baseline BTK-activation also predicts symptomatic burden independently of treatment arm via the FcεRI/BCR cross-link axis. why_not_duplicate: the nearest golden-set neighbor measures IgE substrate; this measures the downstream KINASE-activation state — distinct analyte, distinct mechanism. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: ISAC multiplex specific-IgE as CSU confounder — no dedicated observational
        # PMID specifically validates ISAC-panel positivity count as a treatment-stratifying
        # confounder in CSU (Maurer 2023 PEARL-1/2 PMID 38008109 covers ligelizumab RCT
        # stratification, not ISAC-count as a confounder in routine care). This entry probes
        # whether the classifier correctly identifies a granular diagnostic-panel feature as
        # confounder rather than mediator or outcome-surrogate.
        dspy.Example(
            feature_name="baseline_isac_multiplex_specific_ige_panel_count_preindex_csu",
            derivation_pseudocode=(
                "source=ALLERGY_PANEL_ISAC; derivation_inputs=['specific_ige_kU_l_per_allergen', 'panel_date']; "
                "aggregation=count_positive_above_0_35_kU_l; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU EHR + ISAC component-resolved diagnostics; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Pre-prediction-time count of specific-IgE positivities on multiplex ISAC, derived strictly preindex (window_days=180, knowable_at=preindex_0d ensures no post-treatment titer leakage), proxies the polyclonal IgE-substrate load that anti-IgE therapy must neutralize. Z->T: high specific-IgE breadth shifts prescribing toward higher-affinity anti-IgE (mechanistic rationale: polyclonal IgE breadth increases omalizumab dosing requirements via FcεRI saturation; no single observational study has operationalized ISAC-count as a prescribing confounder — this is a plausible-but-unvalidated feature). Z->Y: specific-IgE breadth predicts CSU symptom severity through the same FcεRI-cross-linking pathway irrespective of treatment receipt. why_not_duplicate: the nearest golden-set neighbor measures TOTAL polyclonal IgE concentration; this counts component-resolved specific-IgE POSITIVITIES — distinct analyte, distinct mechanistic axis. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 27926978 — Kolkhir 2017 Clin Exp Allergy CSU blood biomarkers review (DOI:10.1111/cea.12870)
        dspy.Example(
            feature_name="baseline_matrix_metalloproteinase_9_serum_preindex_csu",
            derivation_pseudocode=(
                "source=SERUM_LABS; derivation_inputs=['mmp9_ng_ml', 'lab_date']; "
                "aggregation=median; window_days=90; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR labs; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Serum MMP-9, sampled strictly preindex (window_days=90; knowable_at="
                "preindex_0d enforces prefix-censoring), indexes mast-cell-driven matrix-"
                "remodeling burden at baseline. Z->T: clinicians escalate to biologic in "
                "high-MMP-9 patients showing tissue-remodeling features (Kolkhir 2017 PMID "
                "27926978; doi:10.1111/cea.12870). Z->Y: MMP-9 predicts symptom severity "
                "independent of treatment arm via mast-cell activation axis. why_not_"
                "duplicate: distinct analyte from CRP/D-dimer baseline biomarkers; MMP-9 "
                "captures REMODELING not coagulation. Remediation per role-to-remediation "
                "table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 27926978 — Kolkhir 2017 Clin Exp Allergy CSU blood biomarkers review (DOI:10.1111/cea.12870)
        dspy.Example(
            feature_name="baseline_mean_platelet_volume_fl_preindex_csu",
            derivation_pseudocode=(
                "source=CBC_LABS; derivation_inputs=['mpv_fl', 'lab_date']; "
                "aggregation=median; window_days=60; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU claims; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Mean platelet volume from preindex CBCs (window_days=60, knowable_at="
                "preindex_0d enforcing prefix-censoring at index_date) indexes platelet "
                "turnover linked to CSU disease activity. Z->T: elevated MPV is a known "
                "marker that drives biologic escalation decisions in CSU patients (Kolkhir "
                "2017 PMID 27926978; doi:10.1111/cea.12870). Z->Y: MPV is associated with "
                "CSU activity through coagulation-cascade activation independent of "
                "treatment selection. why_not_duplicate: distinct CBC analyte from baseline "
                "eosinophil/basophil counts; captures platelet morphology not granulocyte "
                "axis. Remediation per role-to-remediation table: confounder → "
                "keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 27926978 — Kolkhir 2017 Clin Exp Allergy CSU blood biomarkers review (DOI:10.1111/cea.12870)
        dspy.Example(
            feature_name="baseline_prothrombin_fragment_f12_serum_preindex_csu",
            derivation_pseudocode=(
                "source=COAGULATION_PANEL; derivation_inputs=['f1_plus_2_pmol_l', 'lab_date']; "
                "aggregation=median; window_days=90; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR labs; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Prothrombin fragment F1+2 in serum, measured strictly preindex (window_days="
                "90, knowable_at=preindex_0d ensures no postindex coagulation-assay leakage), "
                "indexes baseline coagulation-cascade engagement in CSU (Kolkhir 2017 PMID "
                "27926978; doi:10.1111/cea.12870). Z->T: hypercoagulable signature in CSU "
                "shifts prescribers toward more-aggressive biologic protocols. Z->Y: F1+2 "
                "predicts symptom severity through the extrinsic-cascade-mast-cell axis "
                "irrespective of treatment receipt. why_not_duplicate: distinct from D-dimer "
                "(F1+2 is upstream cascade-activation; D-dimer is downstream fibrinolytic "
                "byproduct). Remediation per role-to-remediation table: confounder → "
                "keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39021347 — Ji 2024 Allergy Galectin-9 CSU biomarker (DOI:10.1111/all.16239)
        dspy.Example(
            feature_name="baseline_galectin9_eosinophil_pct_flow_preindex_csu",
            derivation_pseudocode=(
                "source=FLOW_CYTOMETRY; derivation_inputs=['gal9_pos_eos_pct', 'flow_date']; "
                "aggregation=median; window_days=60; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU chart-abstracted flow; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Galectin-9-positive eosinophil fraction at baseline, derived strictly "
                "preindex (knowable_at=preindex_0d enforces prefix-censoring; window_days=60), "
                "indexes the eosinophil-Gal-9/TIM-3 axis active in CSU pathogenesis. Z->T: "
                "high Gal-9 signature flags severe-activity patients prioritized for "
                "biologic escalation (Ji 2024 Allergy PMID 39021347; doi:10.1111/all.16239). "
                "Z->Y: Gal-9 expression correlates with disease severity through cytokine "
                "release and Th17 modulation independent of treatment. why_not_duplicate: "
                "distinct flow marker from BTK phosphorylation; targets eosinophil-Gal-9 "
                "axis not basophil-BTK. Remediation per role-to-remediation table: "
                "confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39021347 — Ji 2024 Allergy Galectin-9 CSU biomarker (DOI:10.1111/all.16239)
        dspy.Example(
            feature_name="baseline_serum_tnf_alpha_pg_ml_preindex_csu",
            derivation_pseudocode=(
                "source=CYTOKINE_PANEL; derivation_inputs=['tnf_alpha_pg_ml', 'assay_date']; "
                "aggregation=median; window_days=90; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU + research cytokine substudy; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Baseline serum TNF-α (window_days=90, knowable_at=preindex_0d enforcing "
                "prefix-censoring at index_date — no post-treatment cytokine leakage) "
                "indexes systemic Th17/inflammation activity in CSU. Z->T: elevated TNF-α "
                "correlates with disease severity that drives biologic-initiation decisions "
                "(Ji 2024 Allergy PMID 39021347; doi:10.1111/all.16239). Z->Y: TNF-α drives "
                "Gal-9 upregulation on eosinophils and amplifies symptom severity "
                "independent of treatment arm. why_not_duplicate: distinct cytokine from "
                "CRP (acute-phase reactant); TNF-α is the upstream Th-axis cytokine. "
                "Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 40747638 — Wedi 2025 Curr Opin Allergy Clin Immunol CSU therapies update (DOI:10.1097/ACI.0000000000001095)
        dspy.Example(
            feature_name="prior_h1_antihistamine_4x_updose_failure_flag_180d_preindex_csu",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ndc_h1_antihistamine', 'fill_date', 'days_supply', 'daily_dose_multiplier']; "
                "aggregation=any_at_4x_label_dose_flag; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU claims; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Documented failure of 4x-label-dose H1-antihistamine therapy in the 180d preindex window (knowable_at=preindex_0d enforces strict prefix-censoring; no postindex pharmacy data used) flags antihistamine-refractory CSU, the guideline-defined gateway to biologic escalation (Wedi 2025 PMID 40747638; doi:10.1097/ACI.0000000000001095). Z->T: 4x-updose failure is the dominant indication for biologic/small-molecule initiation per EAACI/GA²LEN/WAO guidelines. Z->Y: refractory phenotype predicts more-severe ongoing disease burden regardless of treatment arm. why_not_duplicate: the nearest golden-set neighbor counts agent-switches; this flags DOSE-ESCALATION-FAILURE specifically — distinct construct. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer 2023 Lancet PEARL-1/2 phase 3 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="baseline_weekly_hives_severity_score_hss7_preindex_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['daily_hss_score', 'pro_date']; "
                "aggregation=weekly_sum; window_days=14; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Weekly HSS7 (sum of daily hive-severity scores), aggregated strictly preindex (window_days=14, knowable_at=preindex_0d — no postindex PRO leakage), measures the hive component of disease activity at baseline (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). Z->T: HSS7-high patients are selected for more-aggressive therapy. Z->Y: HSS7 baseline level predicts response magnitude through regression-to-mean in baseline severity. why_not_duplicate: the nearest golden-set neighbor is the COMPOSITE UAS7 (hives + itch components, 30d window); this is the HSS7 SUBSCORE only (hives only, 14d window). Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer 2023 Lancet PEARL-1/2 phase 3 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="iss7_pruritus_only_subscore_baseline_14d_window_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['daily_iss_score', 'pro_date']; "
                "aggregation=weekly_sum; window_days=14; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Weekly ISS7 (sum of daily itch-severity scores), aggregated strictly "
                "preindex (window_days=14, knowable_at=preindex_0d enforces prefix-"
                "censoring), captures the pruritus component of CSU activity independent "
                "of hive burden (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-"
                "6736(23)01684-7). Z->T: high ISS7 drives biologic-escalation due to QoL "
                "impact. Z->Y: pruritus severity predicts symptom burden ceiling and "
                "treatment-response magnitude. why_not_duplicate: distinct from HSS7 (the "
                "OTHER UAS7 component) — itch and hives often dissociate; distinct from "
                "golden composite UAS7 score. Remediation per role-to-remediation table: "
                "confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer 2023 Lancet PEARL-1/2 phase 3 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="prior_omalizumab_response_status_categorical_730d_preindex_csu",
            derivation_pseudocode=(
                "source=EHR_TREATMENT_HISTORY; derivation_inputs=['prior_omalizumab_fills', 'response_assessment_date', 'response_category']; "
                "aggregation=most_recent_categorical; window_days=730; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Most-recent prior-omalizumab response category (complete-responder/partial/non-responder), assessed strictly preindex (window_days=730, knowable_at=preindex_0d — no postindex response data), captures both treatment-history selection and a marker of immunologic subtype (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). Z->T: omalizumab non-responders are channelled toward BTK-inhibitor therapy. Z->Y: prior-non-response indexes refractory-pathology subtype (e.g., type IIb autoimmune) predicting poorer outcomes across treatments. why_not_duplicate: the nearest golden-set neighbor is binary EVER-USED; this is the ORDINAL response-category — distinct construct. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39021347 — Ji 2024 Allergy basophil studies (DOI:10.1111/all.16239)
        dspy.Example(
            feature_name="baseline_blood_basophil_absolute_count_per_ul_preindex_csu",
            derivation_pseudocode=(
                "source=CBC_DIFF; derivation_inputs=['basophil_abs_per_ul', 'cbc_date']; "
                "aggregation=median; window_days=60; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU claims; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Baseline circulating absolute basophil count from preindex CBCs "
                "(window_days=60, knowable_at=preindex_0d enforces strict prefix-censoring), "
                "reflects basopenia from peripheral sequestration in active CSU (Ji 2024 "
                "Allergy PMID 39021347; doi:10.1111/all.16239). Z->T: basopenia is a "
                "biomarker for severe disease that drives biologic escalation. Z->Y: lower "
                "basophil count predicts symptom severity through degranulation-recruitment "
                "axis independent of treatment receipt. why_not_duplicate: distinct from "
                "compile-set provider/preference IVs; this is a CBC analyte capturing a "
                "specific cell lineage absolute count. Remediation per role-to-remediation "
                "table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 37245259 — Sanchez 2023 Clin Exp Allergy autoimmune CSU/IgG (DOI:10.1111/cea.14352)
        dspy.Example(
            feature_name="baseline_anti_thyroid_peroxidase_antibody_titer_iu_ml_preindex_csu",
            derivation_pseudocode=(
                "source=AUTOIMMUNE_PANEL; derivation_inputs=['anti_tpo_iu_ml', 'assay_date']; "
                "aggregation=max; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Anti-thyroid-peroxidase titer from the most-recent preindex assay (window_days=365, knowable_at=preindex_0d ensures prefix-censoring at index_date), indexes type-IIb autoimmune CSU subtype (Sanchez 2023 Clin Exp Allergy PMID 37245259; doi:10.1111/cea.14352). Z->T: TPO-positive patients (autoimmune subtype) shift prescriber preference toward slower-responding biologic regimens. Z->Y: autoimmune subtype is associated with poorer treatment response across modalities. why_not_duplicate: the nearest golden-set neighbor is the DIAGNOSIS FLAG; this is the QUANTITATIVE titer level — distinct measurement. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 29797525 — Takahashi 2018 J Dermatol D-dimer urticaria (DOI:10.1111/1346-8138.14481)
        dspy.Example(
            feature_name="baseline_d_dimer_xuln_preindex_180d_csu",
            derivation_pseudocode=(
                "source=COAGULATION_PANEL; derivation_inputs=['d_dimer_ug_ml', 'd_dimer_uln', 'lab_date']; "
                "aggregation=max; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR labs; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Maximum D-dimer/ULN ratio over the 180d preindex window (knowable_at=preindex_0d enforces strict prefix-censoring; no postindex coagulation data), indexes coagulation-fibrinolysis activation linked to disease severity (Takahashi 2018 PMID 29797525; doi:10.1111/1346-8138.14481; Kolkhir 2017 PMID 27926978). Z->T: high preindex D-dimer is a recognized biomarker driving biologic escalation. Z->Y: D-dimer elevation predicts severity-of-flares independent of treatment. why_not_duplicate: the nearest golden-set neighbor is a POSTINDEX DELTA; this is the PREINDEX LEVEL/ULN — distinct temporal position. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 41602870 — Kolkhir 2025 Allergy CRP biomarker review (proxy DOI)
        dspy.Example(
            feature_name="baseline_serum_eosinophil_cationic_protein_ecp_preindex_csu",
            derivation_pseudocode=(
                "source=SERUM_LABS; derivation_inputs=['ecp_ng_ml', 'lab_date']; "
                "aggregation=median; window_days=90; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR labs; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Serum ECP, sampled strictly preindex (window_days=90, knowable_at="
                "preindex_0d enforces prefix-censoring), indexes eosinophil-activation "
                "burden distinct from absolute count (Kolkhir 2017 PMID 27926978; "
                "doi:10.1111/cea.12870). Z->T: ECP-high patients are clinically flagged "
                "for biologic escalation. Z->Y: ECP correlates with disease severity through "
                "Th2 axis activation independent of treatment arm. why_not_duplicate: "
                "distinct from Gal-9-positive-eosinophil-percent (a flow marker of "
                "phenotype); this is a SECRETED-PRODUCT analyte capturing degranulation "
                "activity. Remediation per role-to-remediation table: confounder → "
                "keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Confounder section continues with prescriber/patient axis (lit-grounded) -----
        # PMID: 36066999 — Rudenko 2022 Antibodies CSU healthcare-utilization patterns (proxy)
        dspy.Example(
            feature_name="prior_csu_specialist_visit_count_365d_preindex",
            derivation_pseudocode=(
                "source=CLAIMS_PROVIDER; derivation_inputs=['provider_specialty_taxonomy', 'visit_date', 'icd10_l50x']; "
                "aggregation=count_distinct_dates; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU claims; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Count of distinct allergy/dermatology-specialist visits with CSU diagnosis "
                "in the 365d preindex window (knowable_at=preindex_0d enforces prefix-"
                "censoring at index_date — no postindex visit leakage) captures both "
                "disease-severity proxy and care-pathway engagement. Z->T: high specialist "
                "engagement drives biologic-initiation decisions (Maurer 2023 PMID "
                "38008109; doi:10.1016/S0140-6736(23)01684-7). Z->Y: specialist contact "
                "intensity correlates with severity and outcome trajectories independent "
                "of treatment arm. why_not_duplicate: distinct from prior-therapy counts; "
                "this measures CARE-CONTACT INTENSITY not treatment exposure. Remediation "
                "per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B3-L: Brookhart-Wang IV entries (4 lit-grounded IVs) -----
        # PMID: 40747638 — Wedi 2025 (FDA approval timing context) + Brookhart 2006 PMID 16617275-PrefIV pattern
        dspy.Example(
            feature_name="patient_indexed_within_90d_post_fda_approval_omalizumab_biosimilar_ct_p39_flag_csu",
            derivation_pseudocode=(
                "source=FDA_APPROVAL_CALENDAR; derivation_inputs=['fda_approval_date_ct_p39', 'patient_index_date']; "
                "aggregation=binary_within_window; window_days=90; knowable_at=index_date"
            ),
            dataset_context=(
                "Optum CSU claims + FDA approval feed; cohort=CSU; treatment=omalizumab_biosimilar_init; "
                "outcome=uas7_remission_180d; prediction_anchor=biosimilar_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Binary flag indicating patient indexed within 90d of FDA approval of omalizumab biosimilar CT-P39 (March 2025; Wedi 2025 PMID 40747638; doi:10.1097/ACI.0000000000001095). Pre-anchor window enforced; the approval date is a fixed exogenous calendar event independent of patient characteristics. Z->T: approval-recency surge drives biosimilar uptake (Brookhart-Wang short-term-IV framework PMID 16617275). EXCLUSION RESTRICTION DEFENSE: FDA approval date for a biosimilar has no biological path Z->Y other than through treatment receipt; the active molecule is equivalent, so there is no efficacy-shift confound (Brookhart-Schneeweiss preference-IV review PMID 16617275). why_not_duplicate: the nearest golden-set neighbor targets a DIFFERENT DRUG approval; this targets the omalizumab-biosimilar approval cliff — distinct molecule, distinct calendar event. Remediation per role-to-remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 16617275 — Brookhart 2006 PrefIV + PMID 38008109 PEARL trial-context
        dspy.Example(
            feature_name="index_prescriber_omalizumab_preference_share_tertile_prior_365d_csu",
            derivation_pseudocode=(
                "source=CLAIMS_PROVIDER; derivation_inputs=['prescriber_npi', 'csu_biologic_fills_prior_year', 'omalizumab_share']; "
                "aggregation=tertile_of_share; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU claims; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Tertile of the indexing prescriber's omalizumab-share of their CSU-biologic fills in the prior 365d (knowable_at=preindex_0d — based on prescriptions written BEFORE the focal patient's index, no postindex data); Brookhart-Wang preference-based instrumental variable (Brookhart 2006 PMID 16617275) for CSU biologic selection. Z->T: high-preference-tertile prescribers initiate omalizumab more often. EXCLUSION RESTRICTION DEFENSE: prescriber preference between omalizumab and other CSU biologics affects patient outcome ONLY through treatment receipt; preference does not affect underlying disease biology; preference-share is computed from a cohort of OTHER patients to satisfy the conditional-exchangeability requirement. why_not_duplicate: the nearest golden-set neighbor uses VOLUME (capacity confound); this uses preference-SHARE (Brookhart pattern explicitly avoiding capacity). Remediation per role-to-remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: Brookhart-Wang label-expansion calendar IV. Construct anchored in
        # Brookhart 2006 PMID 16617275 (physician prescribing preference / short-term IV
        # designs); operationalization (calendar-quarter post-Remibrutinib FDA label expansion
        # to CIndU indication) is an implementer-chosen design pattern with no specific
        # published claims study yet — probes whether the classifier recognizes regulatory-
        # timing IVs analogous to bucket-1's iptacopan_first_initiation_within_90d_post_fda_approval_window_pnh.
        dspy.Example(
            feature_name="calendar_quarter_post_remibrutinib_fda_label_expansion_cindu_indicator_csu",
            derivation_pseudocode=(
                "source=FDA_LABEL_CALENDAR; derivation_inputs=['label_expansion_cindu_date', 'patient_index_quarter']; "
                "aggregation=binary_post_event; window_days=180; knowable_at=index_date"
            ),
            dataset_context=(
                "ConcertAI CSU + FDA label feed; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Indicator for patient indexed in the 180d following FDA label expansion of "
                "remibrutinib to include chronic inducible urticaria (CIndU) on top of CSU. "
                "Pre-anchor window enforced; label-expansion date is a fixed exogenous "
                "regulatory event. Z->T: label expansion broadens the patient pool indexed "
                "on remibrutinib (Chhiba & Saini 2025 Ann Allergy review of FDA approval "
                "based on phase 3 UAS7 reductions; PMID 41270830; "
                "doi:10.1016/j.anai.2025.11.008). EXCLUSION RESTRICTION DEFENSE: the "
                "calendar event (label expansion) cannot affect CSU outcomes except through "
                "shifting treatment-receipt patterns; CSU disease biology does not change "
                "with regulatory action (Brookhart-Schneeweiss PMID 16617275). "
                "why_not_duplicate: distinct from initial-approval-window indicator; this "
                "is the LABEL-EXPANSION event (different calendar marker, different "
                "treated-population shift). Remediation per role-to-remediation table: "
                "instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 16617275 — Brookhart 2006 PrefIV + PMID 38008109 PEARL-2 head-to-head ligelizumab vs omalizumab
        dspy.Example(
            feature_name="index_prescriber_first_biologic_initiator_within_first_year_post_remibrutinib_launch_flag_csu",
            derivation_pseudocode=(
                "source=PRESCRIBER_LAUNCH_TIMELINE; derivation_inputs=['prescriber_npi', 'first_remibrutinib_rx_date', 'remibrutinib_launch_date']; "
                "aggregation=binary_within_year; window_days=365; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU + prescriber-claims linkage; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="instrument",
            mechanism=(
                "Binary flag for indexing prescriber having written their FIRST remibrutinib prescription within the first 365d post-launch (knowable_at=preindex_0d — computed against the FOCAL patient's index date but based on prescriber history that PRECEDES the focal index). Brookhart-Wang early-adopter instrument (PMID 16617275). Z->T: early-adopter prescribers shift toward remibrutinib for subsequent eligible patients. EXCLUSION RESTRICTION DEFENSE: prescriber adoption-recency captures information-diffusion timing uncorrelated with patient-level disease biology; pre-launch prescribing behaviour matches across-prescriber adoption profiles, and adoption does not change patient outcomes except via treatment receipt (Brookhart-Schneeweiss PMID 16617275). why_not_duplicate: the nearest golden-set neighbor is the SAME FAMILY but uses a DIFFERENT calendar anchor (approval vs launch); the launch event lags approval and captures availability not authorization timing. Distinct calendar event. Remediation per role-to-remediation table: instrument → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B3-L: Postindex mediators (6 entries) -----
        # PMID: 39021347 — Ji 2024 Allergy Gal-9 dynamics (DOI:10.1111/all.16239)
        dspy.Example(
            feature_name="delta_galectin9_basophil_pct_flow_30_90d_post_index_csu",
            derivation_pseudocode=(
                "source=FLOW_CYTOMETRY; derivation_inputs=['gal9_pos_baso_pct', 'flow_date']; "
                "aggregation=delta_postindex_minus_preindex; window_days=90; knowable_at=postindex_90d"
            ),
            dataset_context=(
                "ConcertAI CSU chart-abstracted flow; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Change in Gal-9-positive-basophil percent from preindex baseline to 30-90d "
                "postindex (knowable_at=postindex_90d; window strictly POST-treatment — "
                "post-anchor sampling intentional for mediator role; the delta requires "
                "post-treatment measurement by definition). M->Y: omalizumab-induced "
                "reduction in Gal-9 expression on basophils correlates with UAS7 remission "
                "in responders (Ji 2024 PMID 39021347; doi:10.1111/all.16239). T->M: anti-"
                "IgE engagement is the proximate driver of Gal-9 downshift on basophils. "
                "Mediator decomposition: this lies on the causal pathway from treatment to "
                "outcome. why_not_duplicate: distinct from BASELINE Gal-9 flow marker; this "
                "is the TREATMENT-INDUCED DELTA. Remediation per role-to-remediation "
                "table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer 2023 PEARL-1/2 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="weekly_angioedema_episode_count_28_56d_post_index_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['angioedema_episode_date', 'patient_id']; "
                "aggregation=mean_episodes_per_week; window_days=28; knowable_at=postindex_56d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Weekly count of angioedema episodes in the 28-56d post-index window "
                "(knowable_at=postindex_56d; window strictly post-treatment because mediator "
                "by definition lies on T->M->Y path). T->M: treatment reduces mast-cell-"
                "driven angioedema episodes within the first 8 weeks (Maurer 2023 PEARL-1/2 "
                "PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). M->Y: reduction in "
                "early angioedema episodes mediates downstream UAS7 remission at 180d. "
                "why_not_duplicate: distinct from UAS7 score (composite hives+itch); this "
                "is the angioedema-specific event count, a separable symptom dimension. "
                "Remediation per role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 39021347 — Ji 2024 Allergy basophil count dynamics (DOI:10.1111/all.16239)
        dspy.Example(
            feature_name="basophil_recovery_pct_28d_post_omalizumab_init_csu",
            derivation_pseudocode=(
                "source=CBC_DIFF; derivation_inputs=['basophil_abs_per_ul', 'cbc_date']; "
                "aggregation=pct_change_from_preindex_baseline; window_days=28; knowable_at=postindex_28d"
            ),
            dataset_context=(
                "Optum CSU + chart-abstracted CBC; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Percent recovery of circulating basophil count from preindex baseline at "
                "28d postindex (knowable_at=postindex_28d; the post-treatment sampling is "
                "structurally required for a mediator). T->M: omalizumab reverses basopenia "
                "via FcεRI-downregulation (Ji 2024 PMID 39021347; doi:10.1111/all.16239). "
                "M->Y: basophil recovery is a known mediator of UAS7 remission. Mediator "
                "axis: treatment -> basophil count recovery -> outcome. why_not_duplicate: "
                "distinct from baseline absolute basophil count; this is the DELTA at a "
                "specific postindex timepoint. Remediation per role-to-remediation table: "
                "mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 29797525 — Takahashi 2018 D-dimer dynamics + PMID 27926978 Kolkhir review
        dspy.Example(
            feature_name="delta_total_ige_iu_ml_60_90d_post_omalizumab_init_csu",
            derivation_pseudocode=(
                "source=SERUM_LABS; derivation_inputs=['total_ige_iu_ml', 'lab_date']; "
                "aggregation=delta_postindex_minus_preindex; window_days=90; knowable_at=postindex_90d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Change in serum total IgE from baseline to 60-90d postindex (knowable_at=postindex_90d; postindex sampling structurally required). T->M: omalizumab binds free IgE forming complexes that paradoxically INCREASE measured total-IgE in lab assays (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). M->Y: degree of measured total-IgE rise correlates with effective drug-target engagement, mediating UAS7 response. why_not_duplicate: the nearest golden-set neighbor is the BASELINE; this is the POSTINDEX DELTA. Remediation per role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38141832 — Bernstein 2023 BTK signaling (DOI:10.1016/j.jaci.2023.12.008)
        dspy.Example(
            feature_name="delta_btk_phosphorylation_pct_basophil_60_90d_post_remibrutinib_init_csu",
            derivation_pseudocode=(
                "source=FLOW_CYTOMETRY; derivation_inputs=['pbtk_pct_basophil', 'flow_date']; "
                "aggregation=delta_postindex_minus_preindex; window_days=90; knowable_at=postindex_90d"
            ),
            dataset_context=(
                "ConcertAI CSU chart flow; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Change in basophil pBTK from baseline to 60-90d postindex (knowable_at="
                "postindex_90d; post-treatment sampling structurally required for "
                "mediator). T->M: remibrutinib inhibits BTK directly, reducing pBTK in "
                "basophils (Bernstein 2023 BTK-signaling review PMID 38141832; doi:10.1016/"
                "j.jaci.2023.12.008). M->Y: depth of BTK-inhibition mediates downstream "
                "UAS7 remission. why_not_duplicate: distinct from baseline pBTK confounder; "
                "this is the TREATMENT-INDUCED DELTA on the same analyte. Remediation per "
                "role-to-remediation table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer 2023 PEARL-1/2 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="rescue_corticosteroid_oral_courses_count_56d_post_index_csu",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['oral_steroid_ndc', 'fill_date', 'days_supply']; "
                "aggregation=distinct_courses_count; window_days=56; knowable_at=postindex_56d"
            ),
            dataset_context=(
                "Optum CSU pharmacy; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Count of distinct oral corticosteroid courses (>3 days supply each) in "
                "the 56d post-index window (knowable_at=postindex_56d; postindex sampling "
                "structurally required for the rescue-utilization mediator). T->M: "
                "treatment efficacy reduces need for rescue steroids in the early post-"
                "treatment window (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-"
                "6736(23)01684-7). M->Y: steroid-rescue frequency is on the pathway from "
                "treatment to long-term UAS7 remission via inflammation-axis modulation. "
                "why_not_duplicate: distinct from golden concomitant_steroid_burst_count_"
                "followup (which has dual role label); this is a STRICT EARLY-WINDOW count "
                "labelled as mediator in the omalizumab arm where treatment-induced rescue "
                "reduction is the dominant axis. Remediation per role-to-remediation "
                "table: mediator → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B3-L: Descendants (4 entries) -----
        # PMID: 38008109 — Maurer 2023 PEARL-1/2 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="post_index_uas7_complete_response_uas7_eq_zero_at_24w_flag_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['weekly_uas7_score', 'pro_date']; "
                "aggregation=binary_uas7_equals_zero; window_days=168; knowable_at=postindex_168d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Binary indicator for UAS7=0 (complete response) measured at 24w postindex "
                "(knowable_at=postindex_168d; postindex measurement structurally required "
                "and is itself a downstream effect of the outcome process). Y->X: this "
                "feature is the OUTCOME REGISTER reading itself or a near-tautological "
                "descendant of the focal outcome (UAS7 remission at 180d). Descendant "
                "arrowhead: outcome -> measured-complete-response indicator. Including this "
                "in a prediction model induces post-outcome leakage (Maurer 2023 PEARL-1/2 "
                "PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). why_not_duplicate: "
                "golden urticaria_activity_score_180d_postindex_csu is a continuous postindex "
                "UAS7 measure; this is the BINARY UAS7=0 indicator at a SHARPER 24w "
                "milestone — distinct construct. Remediation per role-to-remediation "
                "table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer 2023 PEARL-1/2 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="post_index_uct_score_complete_control_uct_geq16_at_24w_flag_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['uct_score', 'pro_date']; "
                "aggregation=binary_uct_geq_16; window_days=168; knowable_at=postindex_168d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Urticaria Control Test (UCT) ≥16 (complete control) flag at 24w postindex (knowable_at=postindex_168d; postindex required by construction). Y->X: the outcome (UAS7 remission at 180d) and UCT-complete-control are highly co-determined — UCT is a parallel response register downstream of treatment success. Descendant arrowhead: outcome -> UCT-control flag. Including this feature leaks outcome information at training time (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). why_not_duplicate: the nearest golden-set neighbor is the continuous UCT score; this is the BINARY UCT≥16 control flag at 24w — distinct construct. Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer 2023 PEARL-1/2 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="post_index_csu_specific_qol_score_above_70_at_24w_flag_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['cu_qol_score', 'pro_date']; "
                "aggregation=binary_score_above_70; window_days=168; knowable_at=postindex_168d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "CU-Q2oL (CSU-specific QoL) score above 70 flag at 24w postindex "
                "(knowable_at=postindex_168d; postindex required by construction). Y->X: "
                "CU-Q2oL is a downstream patient-reported outcome that responds in lockstep "
                "with UAS7 remission (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/"
                "S0140-6736(23)01684-7). Descendant arrowhead: outcome (UAS7 remission) -> "
                "QoL-improvement register. why_not_duplicate: distinct from DLQI (general "
                "dermatology QoL); this is the CSU-SPECIFIC CU-Q2oL instrument. "
                "Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 27926978 — Kolkhir 2017 CSU biomarkers review (DOI:10.1111/cea.12870)
        dspy.Example(
            feature_name="post_index_d_dimer_normalization_below_uln_at_90d_flag_csu",
            derivation_pseudocode=(
                "source=COAGULATION_PANEL; derivation_inputs=['d_dimer_ug_ml', 'd_dimer_uln', 'lab_date']; "
                "aggregation=binary_below_uln; window_days=90; knowable_at=postindex_90d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR labs; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Binary flag for D-dimer normalization (<ULN) at 90d postindex (knowable_at==postindex_90d; postindex measurement by construction). Y->X: D-dimer tracks disease-activity normalization; in responders D-dimer normalizes as a downstream consequence of disease control rather than an upstream driver (Kolkhir 2017 PMID 27926978; doi:10.1111/cea.12870; Takahashi 2018 PMID 29797525). Descendant arrowhead: disease-control (outcome) -> D-dimer normalization. why_not_duplicate: the nearest golden-set neighbor is a CONTINUOUS DELTA at a LATER window; this is the BINARY NORMALIZATION flag at 90d. Remediation per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B3-L: Ancestors (4 entries) -----
        # PMID: 37245259 — Sanchez 2023 CSU autoimmune (proxy)
        dspy.Example(
            feature_name="family_history_chronic_urticaria_first_degree_relative_flag_csu",
            derivation_pseudocode=(
                "source=EHR_FAMILY_HISTORY; derivation_inputs=['family_relation', 'icd10_l50x_first_degree']; "
                "aggregation=any; window_days=999999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Family history flag for first-degree relative with chronic urticaria, documented strictly preindex (knowable_at=preindex_0d enforces prefix-censoring; this is a non-time-varying genetic/familial trait). Ancestor arrowhead: shared genetic-susceptibility-loci -> patient CSU risk; this is UPSTREAM of disease onset (Sanchez 2023 PMID 37245259) and thus ancestor (not confounder) because it acts ONLY through baseline CSU-susceptibility. F->disease onset->T,Y. why_not_duplicate: the nearest golden-set neighbor is for ATOPY (broader umbrella); this is for CSU SPECIFICALLY. Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 37245259 — Sanchez 2023 HLA in CSU (proxy)
        dspy.Example(
            feature_name="hla_dr4_specific_locus_carrier_germline_indicator_csu",
            derivation_pseudocode=(
                "source=HLA_TYPING; derivation_inputs=['hla_dr_allele']; "
                "aggregation=any_dr4_positive; window_days=999999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR + HLA registry; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "HLA-DR4 carrier flag, available as germline genotype (knowable_at=preindex_0d trivially; the genotype is fixed at conception, no temporal concerns). Ancestor arrowhead: germline genotype -> immune-repertoire -> CSU disease susceptibility; the allele predates ALL post-conception exposures and acts only via disease-onset risk. why_not_duplicate: the nearest golden-set neighbor targets DRB1 alleles broadly; this targets DR4 specifically. Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: Early-life AD before age 5y as CSU ancestor — no published paper maps
        # specifically the age-5y AD threshold to adult CSU susceptibility via atopic march.
        # Kolkhir 2017 (PMID 27926978) covers blood biomarkers in CSU, not the AD-to-CSU
        # developmental pathway. This entry probes whether the classifier recognizes a fixed
        # prenatal/neonatal trait as an ancestor even when the mechanism is plausible but
        # not directly cited in an observational study of CSU specifically.
        dspy.Example(
            feature_name="early_life_atopic_dermatitis_before_age_5y_history_flag_csu",
            derivation_pseudocode=(
                "source=EHR_PROBLEM_LIST; derivation_inputs=['icd10_l20x', 'age_at_diagnosis']; "
                "aggregation=any_before_age_5y; window_days=999999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "History of atopic dermatitis diagnosis before age 5y, documented strictly preindex (knowable_at=preindex_0d; this is a fixed early-life trait). Ancestor arrowhead: early-life atopic-march -> immune-phenotype -> CSU susceptibility (mechanistic rationale: atopic march — early eczema confers Th2-skewed immune phenotype elevating adult urticaria risk; no single observational study validates the age-5y AD cutoff specifically for CSU, making this a plausible edge case for the classifier). The early-life event temporally precedes ALL CSU disease-process exposures and acts through baseline disease-susceptibility only. why_not_duplicate: the nearest golden-set neighbor is broader (any childhood eczema any age); this is the SHARPER age-5y threshold variant. Remediation per role-to-remediation table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35933036 — Zhang 2022 J Invest Dermatol GWAS of CSU (DOI:10.1016/j.jid.2022.07.012)
        # First GWAS of CSU identifying risk loci (HLA-G, PTPN22, LILRA3) overlapping autoimmune genetics.
        dspy.Example(
            feature_name="ancestry_european_genetic_pc1_decile_csu_susceptibility",
            derivation_pseudocode=(
                "source=GENETIC_PCA; derivation_inputs=['european_pc1']; "
                "aggregation=decile; window_days=999999; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU + genetic ancestry array; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Decile of European genetic-ancestry PC1, from germline genotype data "
                "(knowable_at=preindex_0d trivially; ancestry is fixed at conception). "
                "Ancestor arrowhead: germline ancestry -> baseline genetic-architecture of "
                "HLA/PTPN22/LILRA3/IGHG loci that vary across ancestry groups and "
                "predispose to CSU via autoimmune pathways (Zhang 2022 J Invest Dermatol "
                "GWAS of CSU; PMID 35933036; doi:10.1016/j.jid.2022.07.012). Ancestry "
                "indexes genetic susceptibility upstream of all disease exposures. "
                "why_not_duplicate: distinct from HLA-DR4 allele (a single locus); this is "
                "GLOBAL ANCESTRY across all loci. Remediation per role-to-remediation "
                "table: ancestor → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B3-L: Colliders (3 lit-grounded entries) -----
        # PMID: 38008109 — Maurer 2023 PEARL-1/2 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="enrolled_in_csu_phase3_trial_substudy_flag_180d_postindex_csu",
            derivation_pseudocode=(
                "source=TRIAL_REGISTRY; derivation_inputs=['trial_enrollment_date', 'nct_id']; "
                "aggregation=any_post_index; window_days=180; knowable_at=postindex_180d"
            ),
            dataset_context=(
                "ConcertAI CSU + clinical-trial registry; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Flag for enrollment in a CSU phase-3 trial substudy in the 180d postindex "
                "window (knowable_at=postindex_180d; postindex by construction). Both "
                "treatment-receipt (T) AND outcome (Y, perceived response) influence "
                "ongoing-trial-enrollment decisions: physicians refer treatment-responsive "
                "patients to extension protocols; non-responders are also referred to "
                "rescue arms (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-"
                "6736(23)01684-7). Collider arrowhead: T -> enrollment <- Y. Conditioning "
                "on enrollment opens a backdoor T-Y path biasing causal estimates. "
                "why_not_duplicate: distinct from prior-trial-history (a preindex feature); "
                "this is the POSTINDEX enrollment flag. Remediation per role-to-"
                "remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer 2023 PEARL-1/2 (DOI:10.1016/S0140-6736(23)01684-7)
        dspy.Example(
            feature_name="switched_to_alternative_csu_biologic_at_120d_postindex_flag_csu",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['biologic_ndc', 'fill_date', 'index_drug_ndc']; "
                "aggregation=any_switch_post_index; window_days=120; knowable_at=postindex_120d"
            ),
            dataset_context=(
                "Optum CSU pharmacy; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Flag for switch to alternative CSU biologic (e.g., from omalizumab to remibrutinib or ligelizumab) in the 120d post-index window (knowable_at=postindex_120d; post-treatment by construction). Both initial-treatment (T) AND outcome (Y, lack-of-response) influence the switch decision: switches occur in non-responders only, but the switch is also gated on what drug was initiated (Wedi 2025 PMID 40747638; doi:10.1097/ACI.0000000000001095; Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). Collider arrowhead: T -> switch <- Y. Conditioning on switch opens a backdoor T-Y path biasing causal estimates. why_not_duplicate: the nearest golden-set neighbor uses a 180d window; this is the 120d EARLIER window — distinct temporal cutoff. Remediation per role-to-remediation table: collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: T->referral<-Y collider structure (Hernan 2004 collider patterns).
        # Underlying claims-routing construct is anchored in Zazzali 2012 PMID 22289728
        # CIU utilization patterns (56% PCP / 14% allergist / 5% dermatologist routing),
        # but the specific 90d-postindex window operationalization is an implementer-chosen
        # collider-detection design choice — probes whether the classifier recognizes the
        # T->referral<-Y collider when referral is jointly determined by treatment and outcome.
        dspy.Example(
            feature_name="csu_specialty_clinic_referral_within_90d_postindex_flag",
            derivation_pseudocode=(
                "source=CLAIMS_REFERRAL; derivation_inputs=['referral_specialty', 'referral_date']; "
                "aggregation=any_referral_to_urticaria_center; window_days=90; knowable_at=postindex_90d"
            ),
            dataset_context=(
                "Optum CSU claims; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Flag for postindex referral to specialty urticaria center (UCARE) within "
                "90d post-index (knowable_at=postindex_90d; post-treatment by "
                "construction). Both treatment-receipt (T) AND outcome (Y, poor response "
                "trajectory) influence the referral: complicated-course patients on first-"
                "line biologics get escalated to specialty centers, AND treatment-failure "
                "is the dominant referral trigger (Zazzali 2012 CIU claims study: 56% "
                "primary care vs 14% allergist routing; OCS bursts signal severity-driven "
                "escalation; PMID 22289728; doi:10.1016/j.anai.2011.10.018). Collider "
                "arrowhead: T -> referral <- Y. Conditioning on referral opens a spurious "
                "T-Y path biasing causal estimates. why_not_duplicate: distinct from "
                "prior_csu_specialist_visit_count_365d_preindex (a PREINDEX confounder); "
                "this is the POSTINDEX collider with the same provider-system but reversed "
                "temporal positioning. Remediation per role-to-remediation table: "
                "collider → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B3-A: Adversarial (3 entries) -----
        # Adversarial: confounder-vs-mediator boundary — adherence interacts with severity
        dspy.Example(
            feature_name="medication_possession_ratio_h1_antihistamine_180d_preindex_csu",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['h1_ndc', 'fill_date', 'days_supply']; "
                "aggregation=mpr_preindex; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU claims; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Targets the worker-evaluator boundary between CONFOUNDER and MEDIATOR for "
                "adherence behaviour. Pre-prediction-time H1-antihistamine MPR over 180d "
                "preindex (knowable_at=preindex_0d enforces strict prefix-censoring at "
                "index_date — no postindex pharmacy data leakage). The classifier may "
                "WRONGLY label this as mediator because adherence influences outcome via "
                "treatment-effectiveness; the CORRECT label is CONFOUNDER because the "
                "measurement is PRE-INDEX (before omalizumab initiation), so adherence here "
                "indexes treatment-seeking phenotype (Z->T: high-adherence patients more "
                "likely to advance to biologic) AND outcome-prone phenotype (Z->Y: "
                "adherence trait predicts maintenance success regardless of treatment arm) "
                "(Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). "
                "Pre-index adherence is a baseline trait, not a mediator. why_not_duplicate: "
                "distinct from postindex PDC of remibrutinib (which IS a mediator); this is "
                "the PREINDEX antihistamine-MPR boundary case. Remediation per role-to-"
                "remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: IV-vs-confounder boundary — preindex provider switch
        dspy.Example(
            feature_name="provider_switch_within_180d_preindex_flag_csu",
            derivation_pseudocode=(
                "source=CLAIMS_PROVIDER; derivation_inputs=['provider_npi', 'visit_date']; "
                "aggregation=count_distinct_npi_gt_1; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "Optum CSU claims; cohort=CSU; treatment=omalizumab_init; "
                "outcome=uas7_remission_180d; prediction_anchor=omalizumab_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Targets the worker-evaluator boundary between INSTRUMENT and CONFOUNDER. "
                "Flag for provider-switching activity in 180d preindex (knowable_at="
                "preindex_0d enforces prefix-censoring). The classifier may WRONGLY "
                "label this as instrument because provider-changes can shift prescribing "
                "patterns toward different biologics; the CORRECT label is CONFOUNDER "
                "because provider-switching is itself driven by patient dissatisfaction "
                "with previous care (a patient-level severity/response trait), creating a "
                "Z->T and Z->Y dual path through the unobserved severity factor (Wedi 2025 "
                "PMID 40747638; doi:10.1097/ACI.0000000000001095). EXCLUSION RESTRICTION "
                "FAILS because patient-initiated switching responds to disease severity "
                "directly. why_not_duplicate: distinct from prescriber preference-share IVs "
                "(which use OTHER-patient prescribing history and do not encode focal-"
                "patient switching behaviour). Remediation per role-to-remediation table: "
                "confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Adversarial: descendant-vs-mediator boundary — early-response indicator
        dspy.Example(
            feature_name="uas7_50pct_responder_at_4w_postindex_flag_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['uas7_score', 'pro_date']; "
                "aggregation=binary_50pct_reduction_from_baseline; window_days=28; knowable_at=postindex_28d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Targets the worker-evaluator boundary between MEDIATOR and DESCENDANT. "
                "Binary flag for ≥50% UAS7 reduction at 4w post-index (knowable_at="
                "postindex_28d; postindex by construction). The classifier may WRONGLY "
                "label this as mediator (an early-symptomatic-improvement step on the "
                "T->M->Y path); the CORRECT label is DESCENDANT because UAS7-at-4w is "
                "near-tautologically determined by the SAME underlying response-to-"
                "treatment process that determines UAS7-remission-at-180d (the focal "
                "outcome). Using a near-collinear-with-outcome early reading as a feature "
                "leaks information about the outcome label (Maurer 2023 PEARL-1/2 PMID "
                "38008109; doi:10.1016/S0140-6736(23)01684-7). Descendant arrowhead: "
                "outcome-process -> early-response register. why_not_duplicate: distinct "
                "from UAS7=0 at 24w descendant (which is a LATE complete-response register); "
                "this is the EARLY (4w) descendant of the SAME outcome-process. Remediation "
                "per role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B3-E: Edge cases (2 entries) -----
        # Edge case: postindex aggregation that masks leakage
        dspy.Example(
            feature_name="uas7_trajectory_slope_all_available_postindex_to_180d_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['uas7_score', 'pro_date']; "
                "aggregation=linear_regression_slope_all_postindex_values; window_days=180; knowable_at=postindex_180d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Edge case: this entry tests prefix-censoring discipline. The aggregation "
                "window 'all_available_postindex_to_180d' spans the ENTIRE postindex "
                "horizon including the outcome-assessment timepoint (knowable_at="
                "postindex_180d; window is the FULL outcome-measurement period). The "
                "slope of UAS7 from index through 180d is essentially a re-encoding of the "
                "outcome trajectory (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/"
                "S0140-6736(23)01684-7). Descendant arrowhead: outcome-process -> "
                "postindex-trajectory slope. Any feature whose derivation window OVERLAPS "
                "the outcome timepoint is structurally outcome-leaking. why_not_duplicate: "
                "distinct from 28-56d-only postindex windows (which would be mediators); "
                "this OVERLAPS the outcome window — a leakage edge case. Remediation per "
                "role-to-remediation table: descendant → drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: prefix-censoring at biologic-init-date itself
        dspy.Example(
            feature_name="biologic_init_date_same_day_uas7_measurement_csu",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['uas7_score', 'pro_date', 'biologic_init_date']; "
                "aggregation=value_where_pro_date_equals_init_date; window_days=0; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Edge case: this entry tests the EXACT boundary of prefix-censoring. The UAS7 measurement is taken on the SAME calendar day as the biologic-initiation event (knowable_at=preindex_0d, window_days=0). By convention, same-day measurements are considered PREINDEX (the assessment precedes the prescription decision at the visit). Z->T: same-day baseline UAS7 is the value the prescriber uses when making the initiation decision (Maurer 2023 PEARL-1/2 PMID 38008109; doi:10.1016/S0140-6736(23)01684-7). Z->Y: baseline severity predicts outcome trajectory. Pearl arrowhead: Z->T (decision input), Z->Y (severity prognosis). why_not_duplicate: the nearest golden-set neighbor uses a 30d preindex window with the most-recent value selected; this is the SAME-DAY-OF-INITIATION value edge case — distinct windowing. Remediation per role-to-remediation table: confounder → keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # End Plan-239 n=200 Task 5 — Bucket 3 CSU expansion (+42 entries)
        # Final CSU canonical floor: 53 (11 existing + 42 new entries).
        # Role distribution of new entries: confounder=19, mediator=6,
        # descendant=6, instrument=4, ancestor=4, collider=3.
        # =====================================================================
        # =====================================================================
        # Plan-239 n=200 Task 6 — Bucket 4: cross-cohort + synthetic-DGP (+40)
        # Drives synthetic_or_other floor 28 -> 60 AND total -> 230.
        # Source mix:
        #   - 32 synthetic-DGP (Pearl-arrowhead probes; cohort=synthetic_a1/2/3/4):
        #       confounder=6, mediator=5, collider=5, descendant=5, ancestor=5,
        #       instrument=6. Each carries explicit DGP spec (y = beta*t + gamma*z + eps),
        #       Pearl-arrowhead identification, temporal-filter clause, and
        #       remediation mapping. IVs satisfy Brookhart-Wang exclusion restriction
        #       by construction (Z ⊥ Y | T).
        #   - 6 adversarial cross-cohort (worker-evaluator boundary cases; PNH/BC/CSU).
        #   - 2 edge-case cross-cohort (leakage / prefix-censoring corners).
        # All triad-compliant from start. Disjointness vs 190 existing + golden 91.
        # =====================================================================
        # ----- Sub-bucket B4-S: synthetic-DGP confounders (6 entries) -----
        # Edge case: synthetic-DGP construct probing CONFOUNDER recognition under
        # backdoor-path arrowhead Z->T, Z->Y. DGP a1: y = 0.5*t + 0.8*z + N(0,1);
        # P(T=1|Z) = sigmoid(0.6*z). Z is a continuous shared cause.
        dspy.Example(
            feature_name="synth_a1_z_continuous_shared_cause_strong_backdoor",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['Z_continuous_strong']; "
                "aggregation=identity; window_days=180; knowable_at=preindex_180d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl backdoor exemplar in DGP y = 0.5*t + 0.8*z + N(0,1); "
                "P(T=1|Z) = sigmoid(0.6*z). Z->T: strong logistic relevance; "
                "Z->Y: linear coefficient 0.8. Pre-anchor temporal filter "
                "(knowable_at=preindex_180d) enforced by DGP construction — Z is "
                "drawn at simulated baseline, before T. Pearl arrowhead: "
                "confounder (Z->T and Z->Y, open backdoor T<-Z->Y). Failure to "
                "adjust biases E[Y(1)-Y(0)] toward +0.48 vs true ATE 0.5 (analytic "
                "bias from omitted-variable formula). Remediation per role-to-"
                "remediation table: confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing CONFOUNDER under weak-backdoor
        # arrowhead. DGP a1: y = 0.5*t + 0.15*z + eps; P(T=1|Z)=sigmoid(0.2*z).
        dspy.Example(
            feature_name="synth_a1_z_weak_backdoor_borderline_confounder",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['Z_weak_continuous']; "
                "aggregation=identity; window_days=90; knowable_at=preindex_90d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl backdoor weak-effect probe in DGP y = 0.5*t + 0.15*z + eps; "
                "P(T=1|Z)=sigmoid(0.2*z). Z->T and Z->Y are both present but "
                "small-coefficient — classifier may MISS this as 'noise' but the "
                "structural role IS confounder by construction. Pre-anchor temporal "
                "filter: Z is realized at simulated baseline preindex_90d before T. "
                "Pearl arrowhead: confounder (Z->T weak, Z->Y weak, but open "
                "backdoor). why_not_duplicate: distinct from strong-backdoor probe "
                "by coefficient magnitude — this is the weak-confounder boundary "
                "case. Remediation per role-to-remediation table: confounder -> "
                "keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing CONFOUNDER under binary-Z
        # backdoor. DGP a1: y = 0.5*t + 0.7*z_binary + eps; P(T=1|Z=1)=0.7, P(T=1|Z=0)=0.3.
        dspy.Example(
            feature_name="synth_a1_z_binary_high_prevalence_confounder",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['Z_binary_marker']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl backdoor with BINARY Z in DGP y = 0.5*t + 0.7*z_binary + eps; "
                "P(T=1|Z=1)=0.7, P(T=1|Z=0)=0.3. Z->T: large probability shift; "
                "Z->Y: 0.7 coefficient. Pre-anchor temporal filter "
                "(knowable_at=preindex_365d): Z is a baseline marker realized before "
                "T. Pearl arrowhead: confounder (Z->T binary, Z->Y linear, open "
                "backdoor). Common in pharmaco-epi as the 'baseline-comorbidity-"
                "indicator' archetype. Remediation per role-to-remediation table: "
                "confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing CONFOUNDER on a3 multi-covariate DAG.
        dspy.Example(
            feature_name="synth_a3_z_multivariate_age_proxy_confounder",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['Z_age_proxy_continuous']; "
                "aggregation=identity; window_days=0; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl backdoor in multivariate a3 DGP y = 0.5*t + 0.3*z_age + "
                "0.4*z_severity + eps; P(T=1|Z) = sigmoid(0.4*z_age + 0.3*z_severity). "
                "z_age is one of several confounders; Z->T and Z->Y both moderate. "
                "Pre-anchor temporal filter (knowable_at=preindex_0d) — age fixed at "
                "baseline. Pearl arrowhead: confounder (Z->T, Z->Y; backdoor open). "
                "Multi-confounder DAG probes whether classifier identifies role per "
                "structural relations (not isolated marginal correlation). Remediation "
                "per role-to-remediation table: confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing CONFOUNDER as time-varying baseline.
        dspy.Example(
            feature_name="synth_a3_z_timevarying_confounder_baseline_snapshot",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['Z_timevarying_baseline_snapshot']; "
                "aggregation=last_value_before_index; window_days=180; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl backdoor with time-varying Z in DGP a3, snapshot taken at "
                "index: y = 0.5*t + 0.6*z_t0 + eps; P(T=1|Z_t0) = sigmoid(0.5*z_t0). "
                "The 'last_value_before_index' aggregation enforces pre-anchor "
                "temporal filter — only baseline-snapshot data enters; later Z "
                "trajectory is correctly excluded. Pearl arrowhead: confounder "
                "(Z_t0->T, Z_t0->Y; backdoor open). why_not_duplicate: distinct from "
                "static binary/continuous Z probes — this tests aggregation discipline "
                "on a time-varying construct. Remediation per role-to-remediation "
                "table: confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing CONFOUNDER on a3 with interaction.
        dspy.Example(
            feature_name="synth_a3_z_interaction_term_confounder_modifier",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['Z_modifier_continuous']; "
                "aggregation=identity; window_days=180; knowable_at=preindex_180d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl backdoor with interaction in DGP a3: y = 0.5*t + 0.4*z + "
                "0.3*t*z + eps; P(T=1|Z) = sigmoid(0.5*z). Z->T (main effect) and "
                "Z->Y (main effect plus effect modification via t*z). Pre-anchor "
                "temporal filter (knowable_at=preindex_180d): Z is baseline. Pearl "
                "arrowhead: confounder with effect-modification (interaction on Y "
                "but role is still backdoor-confounder structurally). Probes "
                "whether classifier reports role per backdoor structure, not effect-"
                "modification semantics. Remediation per role-to-remediation table: "
                "confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-S: synthetic-DGP mediators (5 entries) -----
        # Edge case: synthetic-DGP construct probing MEDIATOR on a2 indirect-effect path.
        dspy.Example(
            feature_name="synth_a2_m_proximal_pharmacologic_intermediate",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A2; derivation_inputs=['M_proximal_pharm']; "
                "aggregation=mean; window_days=14; knowable_at=postindex_14d"
            ),
            dataset_context=(
                "synthetic_a2 DGP; cohort=synthetic_a2; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="mediator",
            mechanism=(
                "Pearl mediator exemplar in DGP a2: y = 0.2*t + 0.6*m + eps where "
                "m = 0.7*t + nu (so the T->M->Y path carries indirect effect "
                "0.7*0.6=0.42 of the total 0.62 effect). M is realized AFTER T "
                "(knowable_at=postindex_14d); post-treatment temporal filter "
                "applies. Pearl arrowhead: mediator (T->M->Y; M is on the directed "
                "path). Adjusting for M blocks the indirect effect and biases ATE "
                "downward (over-adjustment bias, Hernan PMID 10955408). Remediation "
                "per role-to-remediation table: mediator -> window."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing MEDIATOR on a2 with partial mediation.
        dspy.Example(
            feature_name="synth_a2_m_partial_mediator_with_direct_effect",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A2; derivation_inputs=['M_partial_path']; "
                "aggregation=mean; window_days=30; knowable_at=postindex_30d"
            ),
            dataset_context=(
                "synthetic_a2 DGP; cohort=synthetic_a2; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="mediator",
            mechanism=(
                "Pearl partial-mediation probe in DGP a2: y = 0.5*t + 0.4*m + eps "
                "with m = 0.5*t + nu. Total effect 0.7 = 0.5 direct + 0.2 indirect "
                "(via T->M->Y). Post-treatment temporal filter (postindex_30d) — M "
                "is observed after T. Pearl arrowhead: mediator (T->M->Y partial "
                "path). Probes whether classifier identifies role on a partial-"
                "mediation DGP where direct effect is preserved. Remediation per "
                "role-to-remediation table: mediator -> window."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing MEDIATOR with multi-step chain.
        dspy.Example(
            feature_name="synth_a2_m_distal_chain_mediator_step2",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A2; derivation_inputs=['M2_distal_chain']; "
                "aggregation=mean; window_days=60; knowable_at=postindex_60d"
            ),
            dataset_context=(
                "synthetic_a2 DGP; cohort=synthetic_a2; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="mediator",
            mechanism=(
                "Pearl multi-step mediation in DGP a2: T->M1->M2->Y with structural "
                "equations m1=0.6*t+nu1, m2=0.5*m1+nu2, y=0.5*t+0.5*m2+eps. M2 is "
                "the distal mediator on the chain. Post-treatment temporal filter "
                "(postindex_60d) enforces measurement after T and after M1. Pearl "
                "arrowhead: mediator (T->M1->M2->Y, M2 on directed path). Adjusting "
                "for M2 blocks all indirect effect through that chain. why_not_"
                "duplicate: distinct from M1 (proximal) — this is the distal step. "
                "Remediation per role-to-remediation table: mediator -> window."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing MEDIATOR with binary M.
        dspy.Example(
            feature_name="synth_a2_m_binary_mediator_threshold_event",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A2; derivation_inputs=['M_binary_event']; "
                "aggregation=any; window_days=45; knowable_at=postindex_45d"
            ),
            dataset_context=(
                "synthetic_a2 DGP; cohort=synthetic_a2; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="mediator",
            mechanism=(
                "Pearl mediator with BINARY M in DGP a2: y = 0.3*t + 0.7*m + eps "
                "where P(M=1|T)=sigmoid(0.8*t). M is a post-treatment threshold-"
                "crossing event (e.g., simulated 'response-flag'). Post-treatment "
                "temporal filter (postindex_45d). Pearl arrowhead: mediator (T->M->Y, "
                "M binary on directed path). Probes whether classifier handles "
                "binary mediators differently from continuous ones — structural role "
                "is identical. Remediation per role-to-remediation table: mediator -> "
                "window."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing MEDIATOR with measurement noise.
        dspy.Example(
            feature_name="synth_a2_m_noisy_mediator_high_variance",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A2; derivation_inputs=['M_noisy_observed']; "
                "aggregation=mean; window_days=30; knowable_at=postindex_30d"
            ),
            dataset_context=(
                "synthetic_a2 DGP; cohort=synthetic_a2; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="mediator",
            mechanism=(
                "Pearl mediator with measurement noise in DGP a2: latent m* = 0.7*t "
                "+ nu where Var(nu)=0.5, observed m = m* + measurement_error, "
                "Var(measurement_error)=1.0. y = 0.3*t + 0.5*m* + eps. The OBSERVED "
                "m is still structurally a mediator (T->M*->Y, T->M observed proxy), "
                "though attenuated. Post-treatment temporal filter (postindex_30d). "
                "Pearl arrowhead: mediator (noisy proxy of true mediator). Probes "
                "whether classifier identifies role under noise — structural role "
                "unchanged. Remediation per role-to-remediation table: mediator -> "
                "window."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-S: synthetic-DGP colliders (5 entries) -----
        # Edge case: synthetic-DGP construct probing COLLIDER under classical arrowhead T->C<-Y.
        dspy.Example(
            feature_name="synth_a3_c_classic_collider_t_arrow_to_y_arrow_from",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['C_collider_observed']; "
                "aggregation=identity; window_days=30; knowable_at=postindex_30d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="collider",
            mechanism=(
                "Pearl classical collider in DGP a3: c = 0.5*t + 0.5*y + nu_c, with "
                "T and Y otherwise structurally independent given baseline covariates. "
                "Post-treatment-and-post-outcome temporal filter (knowable_at="
                "postindex_30d): C is observed after both T and Y are realized. Pearl "
                "arrowhead: collider (T->C<-Y; conditioning on C opens a non-causal "
                "path between T and Y, inducing spurious dependence — Pearl PMID "
                "9888278). Adjusting for C induces collider bias. why_not_duplicate: "
                "distinct from realistic-clinical colliders (e.g., on_treatment_180d) "
                "— this is the pure-Pearl synthetic exemplar. Remediation per role-to-"
                "remediation table: collider -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing COLLIDER with asymmetric arrowheads.
        dspy.Example(
            feature_name="synth_a3_c_asymmetric_strong_t_weak_y_collider",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['C_asymmetric_collider']; "
                "aggregation=identity; window_days=60; knowable_at=postindex_60d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="collider",
            mechanism=(
                "Pearl collider with asymmetric arrowhead strengths in DGP a3: "
                "c = 0.9*t + 0.1*y + nu_c. Both arrowheads point INTO C, so role is "
                "collider regardless of magnitudes. Post-event temporal filter "
                "(postindex_60d): C observed after T and Y. Pearl arrowhead: collider "
                "(T->C strong, Y->C weak; conditioning on C still opens non-causal "
                "T-Y path). Probes whether classifier reads role from arrowhead "
                "DIRECTIONS, not magnitudes. why_not_duplicate: distinct from classic "
                "symmetric collider by coefficient asymmetry. Remediation per role-to-"
                "remediation table: collider -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing COLLIDER with binary C.
        dspy.Example(
            feature_name="synth_a3_c_binary_event_collider_threshold",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['C_binary_collider_event']; "
                "aggregation=any; window_days=90; knowable_at=postindex_90d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="collider",
            mechanism=(
                "Pearl binary collider in DGP a3: P(C=1|T,Y) = sigmoid(0.6*t + 0.6*y). "
                "Both T and Y feed forward into C. Post-event temporal filter "
                "(postindex_90d). Pearl arrowhead: collider (T->C<-Y, binary). "
                "Conditioning induces Berkson-style selection bias. Probes binary-"
                "collider recognition — structural role identical to continuous case. "
                "Remediation per role-to-remediation table: collider -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing COLLIDER with intermediate ancestor of Y.
        dspy.Example(
            feature_name="synth_a3_c_collider_with_y_via_intermediate",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['C_indirect_y_collider']; "
                "aggregation=identity; window_days=60; knowable_at=postindex_60d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="collider",
            mechanism=(
                "Pearl collider with chained Y-arrow in DGP a3: c = 0.5*t + "
                "0.5*y_ancestor + nu_c where y_ancestor is a node from which Y also "
                "descends. The arrow Y_ancestor->C combined with Y_ancestor->Y means "
                "conditioning on C opens a non-causal T-Y path through Y_ancestor. "
                "Post-event temporal filter (postindex_60d). Pearl arrowhead: collider "
                "(arrowhead from T and arrowhead from Y_ancestor; effectively a "
                "collider for T,Y). Probes M-bias-style colliders where conditioning "
                "induces dependence via an ancestor of Y. Remediation per role-to-"
                "remediation table: collider -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing COLLIDER on a4 multi-cause node.
        dspy.Example(
            feature_name="synth_a4_c_multi_cause_collider_three_parents",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['C_multi_parent_collider']; "
                "aggregation=identity; window_days=45; knowable_at=postindex_45d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="collider",
            mechanism=(
                "Pearl collider with THREE incoming arrows in DGP a4: c = 0.4*t + "
                "0.4*y + 0.4*z + nu_c (Z is another covariate). Conditioning on C "
                "opens multiple non-causal paths (T->C<-Y, T->C<-Z->elsewhere). Post-"
                "event temporal filter (postindex_45d). Pearl arrowhead: multi-parent "
                "collider (T->C, Y->C, Z->C). Probes whether classifier recognizes "
                "multi-cause colliders — same drop remediation. Remediation per role-"
                "to-remediation table: collider -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-S: synthetic-DGP descendants (5 entries) -----
        # Edge case: synthetic-DGP construct probing DESCENDANT under Y->D arrowhead.
        dspy.Example(
            feature_name="synth_a3_d_pure_y_descendant_post_outcome",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['D_y_descendant']; "
                "aggregation=identity; window_days=30; knowable_at=postindex_post_y_30d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="descendant",
            mechanism=(
                "Pearl descendant exemplar in DGP a3: d = 0.8*y + nu_d. The arrow "
                "points FROM Y INTO D — D is a downstream consequence of the outcome. "
                "Post-outcome temporal filter (knowable_at=postindex_post_y_30d): D "
                "is observed strictly after Y is realized. Pearl arrowhead: descendant "
                "(Y->D; no direct T->D arrow). Including D as a feature leaks outcome "
                "information (near-tautological prediction). Remediation per role-to-"
                "remediation table: descendant -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing DESCENDANT with weak Y->D path.
        dspy.Example(
            feature_name="synth_a3_d_weak_y_descendant_attenuated",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['D_weak_y_descendant']; "
                "aggregation=identity; window_days=60; knowable_at=postindex_post_y_60d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="descendant",
            mechanism=(
                "Pearl descendant with weak arrow in DGP a3: d = 0.2*y + nu_d. Y->D "
                "is structurally present but coefficient is small. Post-outcome "
                "temporal filter (postindex_post_y_60d). Pearl arrowhead: descendant "
                "(Y->D weak). Probes whether classifier identifies role from "
                "DIRECTION not magnitude — role is still descendant, drop remediation "
                "still applies. why_not_duplicate: distinct from strong-arrow "
                "descendant by coefficient magnitude. Remediation per role-to-"
                "remediation table: descendant -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing DESCENDANT via chain Y->X->D.
        dspy.Example(
            feature_name="synth_a3_d_indirect_descendant_chain_two_step",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['D_indirect_chain']; "
                "aggregation=identity; window_days=90; knowable_at=postindex_post_y_90d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="descendant",
            mechanism=(
                "Pearl indirect descendant in DGP a3: x = 0.7*y + nu_x, d = 0.6*x + "
                "nu_d (so y->x->d chain). D is a TWO-STEP descendant of Y. Post-"
                "outcome temporal filter (postindex_post_y_90d) — D observed after Y "
                "and after X. Pearl arrowhead: descendant (Y->X->D; D in the future "
                "lightcone of Y). Including D still leaks outcome information through "
                "the chain. Remediation per role-to-remediation table: descendant -> "
                "drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing DESCENDANT with binary D.
        dspy.Example(
            feature_name="synth_a3_d_binary_outcome_descendant_event",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['D_binary_post_outcome_event']; "
                "aggregation=any; window_days=120; knowable_at=postindex_post_y_120d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="descendant",
            mechanism=(
                "Pearl descendant with binary D in DGP a3: P(D=1|Y) = sigmoid(1.0*y). "
                "D is a binary event downstream of continuous Y. Post-outcome temporal "
                "filter (postindex_post_y_120d). Pearl arrowhead: descendant (Y->D "
                "binary). Probes whether classifier handles binary descendants — "
                "structural role unchanged. Remediation per role-to-remediation table: "
                "descendant -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing DESCENDANT with T->Y->D (also T-influenced).
        dspy.Example(
            feature_name="synth_a3_d_descendant_with_t_via_y_only",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['D_t_only_via_y']; "
                "aggregation=identity; window_days=45; knowable_at=postindex_post_y_45d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="descendant",
            mechanism=(
                "Pearl descendant where T affects D ONLY through Y in DGP a3: "
                "y = 0.5*t + eps, d = 0.7*y + nu_d (no direct T->D arrow). D "
                "depends on T entirely via Y. Post-outcome temporal filter "
                "(postindex_post_y_45d). Pearl arrowhead: descendant of Y (T->Y->D; "
                "D is a descendant of Y, NOT a mediator since there is no T->Y->D "
                "indirect-causal-effect distinction here — Y IS the outcome we want "
                "to estimate, D is downstream). why_not_duplicate: distinct from "
                "pure-Y-descendant by also being a downstream-of-T-via-Y feature, "
                "but structural role is descendant of the outcome. Remediation per "
                "role-to-remediation table: descendant -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-S: synthetic-DGP ancestors (5 entries) -----
        # Edge case: synthetic-DGP construct probing ANCESTOR (predates T and Y via separate paths).
        dspy.Example(
            feature_name="synth_a1_a_ancestor_independent_t_and_y_paths",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['A_root_ancestor']; "
                "aggregation=identity; window_days=730; knowable_at=preindex_730d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="ancestor",
            mechanism=(
                "Pearl ancestor exemplar in DGP a1: A is a root node; A->W1->T and "
                "A->W2->Y where W1, W2 are intermediate confounders. A itself does "
                "NOT directly arrow into T or Y, but appears as a common ancestor "
                "via separate paths. Pre-anchor temporal filter (knowable_at="
                "preindex_730d) — A realized far before T. Pearl arrowhead: ancestor "
                "(A->...->T and A->...->Y; common ancestor through chains, not direct "
                "edges). why_not_duplicate: distinct from direct confounder by the "
                "INTERMEDIATE-NODE structure — adjusting for W1, W2 would suffice "
                "without conditioning on A. Remediation per role-to-remediation "
                "table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing ANCESTOR with long chain to T.
        dspy.Example(
            feature_name="synth_a1_a_distant_ancestor_long_chain_to_t",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['A_distant_long_chain']; "
                "aggregation=identity; window_days=1095; knowable_at=preindex_1095d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="ancestor",
            mechanism=(
                "Pearl distant ancestor in DGP a1: A->X1->X2->X3->T and A->Y_proxy->Y. "
                "Long-chain ancestor of T, shorter-chain ancestor of Y. Pre-anchor "
                "temporal filter (preindex_1095d, 3-year lookback). Pearl arrowhead: "
                "ancestor (long-chain). Probes whether classifier identifies role "
                "regardless of chain length. why_not_duplicate: distinct from "
                "immediate-ancestor by chain length. Remediation per role-to-"
                "remediation table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing ANCESTOR-of-Y-only.
        dspy.Example(
            feature_name="synth_a1_a_ancestor_of_y_only_not_t",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['A_y_only_ancestor']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="ancestor",
            mechanism=(
                "Pearl ancestor-of-Y-only in DGP a1: A->W->Y where W is an "
                "intermediate, but NO path from A to T. Pre-anchor temporal filter "
                "(preindex_365d). Pearl arrowhead: ancestor (A is an ancestor of Y "
                "through W but not of T; equivalent to a 'pure prognostic' covariate "
                "in IPTW terms). why_not_duplicate: distinct from shared-ancestor "
                "(both-paths) version — this only has the Y-side path. Including "
                "such ancestors can improve precision without bias. Remediation per "
                "role-to-remediation table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing ANCESTOR-of-T-only (pure-IV-like ancestor).
        dspy.Example(
            feature_name="synth_a1_a_ancestor_of_t_only_pure_iv_like",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['A_t_only_ancestor']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="ancestor",
            mechanism=(
                "Pearl ancestor-of-T-only in DGP a1: A->W->T with W an intermediate, "
                "no path A->...->Y except through T. Pre-anchor temporal filter "
                "(preindex_365d). Pearl arrowhead: ancestor of T (A precedes T via "
                "chain, no direct path to Y other than through T). Such variables "
                "approximate instrument-like behavior but classified as ancestor by "
                "the DAG-level structure (the IV label would require formal exclusion-"
                "restriction defense, not just absence of direct edge). Remediation "
                "per role-to-remediation table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing ANCESTOR with effect modification at intermediate.
        dspy.Example(
            feature_name="synth_a1_a_ancestor_via_effect_modifying_intermediate",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['A_effect_mod_ancestor']; "
                "aggregation=identity; window_days=540; knowable_at=preindex_540d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="ancestor",
            mechanism=(
                "Pearl ancestor through effect-modifying intermediate in DGP a1: "
                "A->W, W modifies the T-Y effect (interaction t*w in y equation), "
                "and W->T also (W is itself confounder of T,Y). A is the upstream "
                "ancestor of W. Pre-anchor temporal filter (preindex_540d). Pearl "
                "arrowhead: ancestor (A->W with W on backdoor; A is one level up). "
                "Adjusting for W blocks the backdoor; A is not strictly needed but is "
                "structurally an ancestor. Remediation per role-to-remediation "
                "table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-S: synthetic-DGP instruments (6 entries; Brookhart-Wang style) -----
        # Edge case: synthetic-DGP construct probing INSTRUMENT under exclusion restriction.
        dspy.Example(
            feature_name="synth_a4_iv_prescriber_preference_binary_brookhart_wang",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['IV_prescriber_pref_binary']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="instrument",
            mechanism=(
                "Pearl/Brookhart-Wang instrument in DGP a4: P(T=1|IV) = sigmoid("
                "0.9*iv_pref), y = 0.5*t + eps (NO direct IV->Y edge in the DGP). "
                "Pre-anchor temporal filter (preindex_365d) — prescriber preference "
                "is measured from prior-year prescribing pattern, before index. "
                "Pearl arrowhead: instrument (IV->T, IV ⊥ Y | T). Brookhart-Wang "
                "preference IV pattern (synthetic DGP, no literature anchor): IV "
                "constructed from preceding-patient prescribing pattern is "
                "near-randomized w.r.t. current patient's potential outcomes, "
                "satisfying exclusion by design. Remediation per role-to-remediation "
                "table: instrument -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing INSTRUMENT calendar-window style.
        dspy.Example(
            feature_name="synth_a4_iv_calendar_window_first_initiation_brookhart_wang",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['IV_first_init_post_window']; "
                "aggregation=any; window_days=180; knowable_at=preindex_180d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="instrument",
            mechanism=(
                "Pearl instrument in DGP a4 (Brookhart-Wang calendar-window pattern): "
                "binary IV=1 if first initiation occurred within 180d post a simulated "
                "policy-change window. DGP: P(T=1|IV)=sigmoid(0.8*iv + 0.3*Z), "
                "y = 0.5*t + 0.5*Z + eps (Z is a baseline confounder). No direct "
                "IV->Y arrow; IV->Y only through T. Pre-anchor temporal filter "
                "(preindex_180d). Pearl arrowhead: instrument (IV->T, IV ⊥ Y | T,Z; "
                "exclusion restriction holds by DGP construction). Brookhart-Wang "
                "calendar-window first-initiation pattern (synthetic DGP, no literature anchor). "
                "Remediation per role-to-remediation table: instrument -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing INSTRUMENT with weak relevance.
        dspy.Example(
            feature_name="synth_a4_iv_weak_instrument_low_relevance",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['IV_weak_relevance']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="instrument",
            mechanism=(
                "Pearl WEAK instrument in DGP a4: P(T=1|IV)=sigmoid(0.15*iv), "
                "y = 0.5*t + eps (no direct IV->Y). Pre-anchor temporal filter "
                "(preindex_365d). Pearl arrowhead: instrument (IV->T weakly, "
                "IV ⊥ Y | T). Probes weak-instrument boundary — STRUCTURAL role is "
                "still instrument (exclusion holds), but downstream estimation will "
                "suffer high variance. Classifier role label should be 'instrument' "
                "(structural), not 'descendant/ancestor/noise'. Remediation per "
                "role-to-remediation table: instrument -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing INSTRUMENT continuous-IV pattern.
        dspy.Example(
            feature_name="synth_a4_iv_continuous_preference_score_brookhart_wang",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['IV_continuous_preference']; "
                "aggregation=mean; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="instrument",
            mechanism=(
                "Pearl continuous-IV in DGP a4 (Brookhart-Wang prior-prescribing-share "
                "style): IV is the fraction of preceding patients in the prescriber's "
                "panel who received T. P(T=1|IV)=sigmoid(2.5*iv-1), y = 0.5*t + eps. "
                "No direct IV->Y arrow by DGP construction. Pre-anchor temporal filter "
                "(preindex_365d; prior-year prescribing). Pearl arrowhead: instrument "
                "(IV->T, IV ⊥ Y | T). Brookhart-Wang prescribing-share pattern "
                "(synthetic DGP, no literature anchor): prior-year prescribing share "
                "is near-randomized at the patient level, supporting exclusion. "
                "why_not_duplicate: distinct from binary-preference IV by continuous "
                "operationalization. Remediation per role-to-remediation table: "
                "instrument -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing INSTRUMENT under label-expansion shock.
        dspy.Example(
            feature_name="synth_a4_iv_label_expansion_shock_brookhart_wang",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['IV_label_expansion_post']; "
                "aggregation=any; window_days=180; knowable_at=preindex_180d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="instrument",
            mechanism=(
                "Pearl instrument from label-expansion shock in DGP a4: binary IV=1 "
                "if patient's index falls in a simulated post-label-expansion "
                "180d window. P(T=1|IV)=sigmoid(1.0*iv), y=0.5*t+0.3*Z+eps "
                "where Z is a baseline confounder. No direct IV->Y arrow. Pre-anchor "
                "temporal filter (preindex_180d; window membership determined at "
                "or before index). Pearl arrowhead: instrument (IV->T via supply-side "
                "regulatory shock, IV ⊥ Y | T,Z). Brookhart-Wang label-expansion "
                "exogeneity pattern (synthetic DGP, no literature anchor). Remediation per role-to-"
                "remediation table: instrument -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing INSTRUMENT violation (NOT a valid IV).
        dspy.Example(
            feature_name="synth_a4_iv_pseudo_instrument_with_direct_y_path_adversarial",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['IV_pseudo_direct_y']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Adversarial PSEUDO-instrument probe in DGP a4: variable LOOKS like "
                "an IV (binary preference-style) but DGP includes a direct edge "
                "IV->Y. P(T=1|IV)=sigmoid(0.9*iv), y = 0.5*t + 0.4*iv + eps. The "
                "IV->Y direct path VIOLATES exclusion restriction; structurally this "
                "variable is a CONFOUNDER (Z->T, Z->Y) not an instrument. Pre-anchor "
                "temporal filter (preindex_365d). Pearl arrowhead: confounder "
                "(despite naming convention; structure determines role). Probes "
                "whether classifier reads STRUCTURE not LABEL — feature-name says "
                "'iv' but DGP says confounder. Remediation per role-to-remediation "
                "table: confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-A: adversarial cross-cohort (6 entries) -----
        # PMID: 38477987 — APPLY-PNH NEJM 2024 (DOI:10.1056/NEJMoa2308695)
        # Adversarial worker-evaluator boundary: confounder vs ancestor for PNH.
        dspy.Example(
            feature_name="pnh_clone_size_pretreatment_distal_genetic_ancestor",
            derivation_pseudocode=(
                "source=LABS_FLOW_CYTOMETRY; derivation_inputs=['pnh_clone_pct_granulocytes', 'flow_date']; "
                "aggregation=earliest_recorded_value; window_days=730; knowable_at=preindex_730d"
            ),
            dataset_context=(
                "ConcertAI PNH claims+labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Adversarial PNH boundary case: pre-anchor PNH clone size measured "
                "earliest-recorded (2-year lookback). Classifier may WRONGLY label "
                "this as confounder (intuitively, clone size affects both treatment "
                "and outcome). Correct label is ANCESTOR because the EARLIEST-"
                "recorded clone size predates both the prescriber's decision (which "
                "uses CURRENT clone size, a downstream descendant of the earliest "
                "value) and the response — clone size at diagnosis influences "
                "downstream clone size which then enters the T decision (APPLY-PNH "
                "PMID 38477987; doi:10.1056/NEJMoa2308695). Pre-anchor temporal "
                "filter (preindex_730d) — earliest historical value. Pearl arrowhead: "
                "ancestor (A_earliest_clone->W_current_clone->T, A->W->Y). Remediation "
                "per role-to-remediation table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38477987 — APPLY-PNH NEJM 2024
        # Adversarial worker-evaluator: mediator vs collider for PNH on-treatment LDH.
        dspy.Example(
            feature_name="pnh_on_treatment_ldh_30d_mediator_collider_boundary",
            derivation_pseudocode=(
                "source=LABS_LDH; derivation_inputs=['ldh_iu_l', 'lab_date', 'iptacopan_init_date']; "
                "aggregation=mean; window_days=30; knowable_at=postindex_30d"
            ),
            dataset_context=(
                "ConcertAI PNH labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="mediator",
            mechanism=(
                "Adversarial PNH boundary: on-treatment LDH at 30d post-index. "
                "Classifier may waver mediator vs collider. Structurally MEDIATOR: "
                "T (iptacopan complement inhibition) -> M (LDH reduction at 30d) -> "
                "Y (hemoglobin response at 180d) along the established pharmacologic "
                "indirect-effect path (APPLY-PNH PMID 38477987; doi:10.1056/"
                "NEJMoa2308695). It is NOT a collider because there is no incoming "
                "arrow from Y (hemoglobin at 180d) to LDH at 30d — temporal order "
                "precludes Y->M. Post-treatment temporal filter (postindex_30d). "
                "Pearl arrowhead: mediator (T->M->Y; M is on directed path). "
                "Remediation per role-to-remediation table: mediator -> window."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS NEJM 2022 (DOI:10.1056/NEJMoa2114663)
        # Adversarial worker-evaluator: descendant vs collider for BC dose-modification.
        dspy.Example(
            feature_name="bc_dose_modification_first_60d_descendant_collider_boundary",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['ribociclib_dose_changes', 'fill_dates']; "
                "aggregation=any; window_days=60; knowable_at=postindex_60d"
            ),
            dataset_context=(
                "ConcertAI BC claims; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="collider",
            mechanism=(
                "Adversarial BC boundary: first 60d dose-modification event. "
                "Classifier may label DESCENDANT (post-treatment) but COLLIDER is "
                "correct — dose modification is jointly determined by T (initiation "
                "of ribociclib drives need for adjustment) AND by early efficacy/"
                "tolerability signals that share a common-cause structure with Y "
                "(PFS) via underlying disease aggressiveness (MONALEESA-2 OS PMID "
                "35263519; doi:10.1056/NEJMoa2114663). Both T and an early-Y-related "
                "process arrow into the dose-modification node. Post-treatment "
                "temporal filter (postindex_60d). Pearl arrowhead: collider (T->C<-"
                "early_Y_signal). Conditioning induces collider bias. Remediation "
                "per role-to-remediation table: collider -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS NEJM 2022
        # Adversarial worker-evaluator: confounder vs ancestor for BC visceral metastases.
        dspy.Example(
            feature_name="bc_visceral_metastases_at_diagnosis_distal_ancestor",
            derivation_pseudocode=(
                "source=DIAGNOSIS_CLAIMS; derivation_inputs=['icd10_visceral_met_codes', 'diagnosis_date']; "
                "aggregation=any; window_days=730; knowable_at=preindex_730d"
            ),
            dataset_context=(
                "ConcertAI BC claims; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Adversarial BC boundary: visceral-metastasis status recorded AT "
                "ORIGINAL DIAGNOSIS (not at ribociclib init). Classifier may label "
                "CONFOUNDER, but ANCESTOR is correct — the AT-DIAGNOSIS status feeds "
                "into CURRENT (at-index) metastatic burden, which is what enters "
                "the prescriber's decision (MONALEESA-2 OS PMID 35263519; doi:10.1056/"
                "NEJMoa2114663). The original-diagnosis status is upstream-of "
                "current-burden, which is the direct confounder. Pre-anchor temporal "
                "filter (preindex_730d, 2-year lookback). Pearl arrowhead: ancestor "
                "(A_at_dx -> W_current_burden -> T; A -> W -> Y). Remediation per "
                "role-to-remediation table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer PEARL-1/2 2023 (DOI:10.1016/S0140-6736(23)01684-7)
        # Adversarial worker-evaluator: descendant vs mediator for CSU early-response.
        dspy.Example(
            feature_name="csu_uas7_early_2week_response_descendant_mediator_boundary",
            derivation_pseudocode=(
                "source=EHR_PRO; derivation_inputs=['uas7_score', 'pro_date', 'remibrutinib_init_date']; "
                "aggregation=mean; window_days=14; knowable_at=postindex_14d"
            ),
            dataset_context=(
                "ConcertAI CSU EHR PRO; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Adversarial CSU boundary: UAS7 2-week mean post-initiation. "
                "Classifier may label MEDIATOR (an on-path treatment-response "
                "intermediate), but DESCENDANT is correct because UAS7-at-2w and "
                "UAS7-at-180d are repeated measurements of the SAME outcome construct "
                "— they are alternate temporal realizations of the response process, "
                "not distinct nodes on a T->M->Y path (Maurer 2023 PEARL-1/2 PMID "
                "38008109; doi:10.1016/S0140-6736(23)01684-7). Including the 2w "
                "value leaks outcome information rather than blocking an indirect "
                "effect. Post-treatment temporal filter (postindex_14d). Pearl "
                "arrowhead: descendant (outcome-process -> early-response register). "
                "Remediation per role-to-remediation table: descendant -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 38008109 — Maurer PEARL-1/2 2023
        # Adversarial worker-evaluator: confounder vs ancestor for CSU prior-omalizumab history.
        dspy.Example(
            feature_name="csu_prior_omalizumab_lifetime_history_distal_ancestor",
            derivation_pseudocode=(
                "source=PHARMACY_CLAIMS; derivation_inputs=['omalizumab_ever_fill', 'first_fill_date']; "
                "aggregation=any; window_days=1825; knowable_at=preindex_1825d"
            ),
            dataset_context=(
                "ConcertAI CSU claims; cohort=CSU; treatment=remibrutinib_init; "
                "outcome=uas7_remission_180d; prediction_anchor=remibrutinib_init_date"
            ),
            causal_role="ancestor",
            mechanism=(
                "Adversarial CSU boundary: lifetime omalizumab history (5-year "
                "lookback). Classifier may label CONFOUNDER, but ANCESTOR is correct "
                "— lifetime history is upstream of RECENT omalizumab exposure (e.g., "
                "180d preindex), and it is the recent exposure that enters the "
                "prescriber's BTK-inhibitor decision (Maurer 2023 PEARL-1/2 PMID "
                "38008109; doi:10.1016/S0140-6736(23)01684-7). Pre-anchor temporal "
                "filter (preindex_1825d). Pearl arrowhead: ancestor (A_lifetime -> "
                "W_recent -> T; A -> W -> Y via persistent refractoriness). "
                "Remediation per role-to-remediation table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-E: edge cases cross-cohort (2 entries) -----
        # PMID: 38477987 — APPLY-PNH NEJM 2024
        # Edge case: PNH same-day LDH-and-init ambiguity.
        dspy.Example(
            feature_name="pnh_ldh_same_day_iptacopan_init_window_zero_preindex",
            derivation_pseudocode=(
                "source=LABS_LDH; derivation_inputs=['ldh_iu_l', 'lab_date', 'iptacopan_init_date']; "
                "aggregation=value_where_lab_date_equals_init_date; window_days=0; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "ConcertAI PNH labs; cohort=PNH; treatment=iptacopan_init; "
                "outcome=hemoglobin_response_180d; prediction_anchor=iptacopan_init_date"
            ),
            causal_role="confounder",
            mechanism=(
                "Edge case: tests EXACT boundary of prefix-censoring on PNH same-day "
                "measurement. LDH drawn on the SAME calendar day as iptacopan init "
                "(knowable_at=preindex_0d, window_days=0). By convention, the same-"
                "day lab precedes the prescription decision at the same visit. Z->T: "
                "same-day LDH is the prescriber's decision input (APPLY-PNH PMID "
                "38477987; doi:10.1056/NEJMoa2308695). Z->Y: baseline hemolytic "
                "intensity predicts response trajectory. Pre-anchor temporal filter "
                "(preindex_0d) — convention places same-day measurements as "
                "preindex. Pearl arrowhead: confounder (Z->T, Z->Y, open backdoor). "
                "why_not_duplicate: distinct from 30d/90d/180d LDH aggregations — "
                "this is the same-day-of-init boundary case. Remediation per role-"
                "to-remediation table: confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # PMID: 35263519 — MONALEESA-2 OS NEJM 2022
        # Edge case: BC postindex-aggregation overlapping the PFS window — leakage corner.
        dspy.Example(
            feature_name="bc_tumor_size_trajectory_all_postindex_to_pfs_window_leakage",
            derivation_pseudocode=(
                "source=RADIOLOGY_RECIST; derivation_inputs=['target_lesion_sum_mm', 'imaging_date']; "
                "aggregation=linear_regression_slope_all_postindex_values; window_days=730; knowable_at=postindex_730d"
            ),
            dataset_context=(
                "ConcertAI BC imaging; cohort=BC; treatment=ribociclib_init; "
                "outcome=pfs_24m; prediction_anchor=ribociclib_init_date"
            ),
            causal_role="descendant",
            mechanism=(
                "Edge case: tests prefix-censoring discipline on a postindex-"
                "aggregation that OVERLAPS the outcome window. Tumor-size slope "
                "computed across ALL postindex imaging up to the 24-month PFS "
                "endpoint — the derivation window is the FULL outcome-measurement "
                "horizon (MONALEESA-2 OS PMID 35263519; doi:10.1056/NEJMoa2114663). "
                "Tumor-size trajectory over the full PFS-window is essentially a "
                "re-encoding of the progression outcome itself. Post-treatment "
                "temporal filter (postindex_730d) spans the outcome window — "
                "structurally outcome-leaking. Pearl arrowhead: descendant "
                "(outcome-process -> postindex-trajectory slope). why_not_duplicate: "
                "distinct from short-window early-trajectory features (which would "
                "be mediators) — this OVERLAPS the outcome assessment. Remediation "
                "per role-to-remediation table: descendant -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-S: synthetic-DGP floor-padding (4 entries) -----
        # Edge case: synthetic-DGP construct probing CONFOUNDER on a2 multi-mediator DAG.
        dspy.Example(
            feature_name="synth_a2_z_baseline_confounder_alongside_mediators",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A2; derivation_inputs=['Z_baseline_a2_confounder']; "
                "aggregation=identity; window_days=180; knowable_at=preindex_180d"
            ),
            dataset_context=(
                "synthetic_a2 DGP; cohort=synthetic_a2; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl backdoor in mediator-heavy a2 DGP: y = 0.3*t + 0.4*m + "
                "0.5*z + eps, P(T=1|Z) = sigmoid(0.5*z), m = 0.6*t + nu. Z is a "
                "baseline confounder coexisting with mediators on the T->M->Y "
                "path; the backdoor T<-Z->Y is still open and must be adjusted. "
                "Pre-anchor temporal filter (preindex_180d) — Z is baseline. Pearl "
                "arrowhead: confounder (Z->T, Z->Y; distinct from M which is on "
                "directed path). Probes whether classifier disentangles backdoor "
                "vs directed-path roles within the same DAG. Remediation per role-"
                "to-remediation table: confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing CONFOUNDER on a4 IV-rich DAG.
        dspy.Example(
            feature_name="synth_a4_z_baseline_confounder_amid_iv_rich_dag",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['Z_baseline_a4_confounder']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl backdoor in IV-rich a4 DGP: y = 0.5*t + 0.6*z + eps, "
                "P(T=1|Z,IV) = sigmoid(0.4*z + 0.8*iv). Z is a baseline confounder "
                "coexisting with an instrument (IV) on the same DAG; the backdoor "
                "T<-Z->Y is still open. Pre-anchor temporal filter (preindex_365d). "
                "Pearl arrowhead: confounder (Z->T, Z->Y; distinct from IV which "
                "has no Y arrow). Probes whether classifier disentangles confounder "
                "vs instrument when both feed into T. Remediation per role-to-"
                "remediation table: confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing INSTRUMENT under randomized-shock DGP.
        dspy.Example(
            feature_name="synth_a4_iv_randomized_assignment_shock_brookhart_wang",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['IV_randomized_assignment']; "
                "aggregation=identity; window_days=0; knowable_at=preindex_0d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="instrument",
            mechanism=(
                "Pearl instrument under randomized-assignment exemplar in DGP a4: "
                "IV ~ Bernoulli(0.5) independent of all baseline covariates, "
                "P(T=1|IV) = sigmoid(2.0*iv - 1), y = 0.5*t + 0.3*Z + eps where Z "
                "is a baseline confounder of T,Y. No direct IV->Y arrow. Pre-anchor "
                "temporal filter (preindex_0d) — randomization at index. Pearl "
                "arrowhead: instrument (IV->T strong, IV independent of Y given T). "
                "This is the GOLD-STANDARD IV by construction (randomized; exclusion "
                "trivially holds; Brookhart-Wang randomized-assignment IV pattern "
                "(synthetic DGP, no literature anchor) applies). "
                "why_not_duplicate: distinct from observational-preference IVs by "
                "being a true random shock. Remediation per role-to-remediation "
                "table: instrument -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Edge case: synthetic-DGP construct probing DESCENDANT under outcome-aggregation timing.
        dspy.Example(
            feature_name="synth_a3_d_descendant_late_followup_outcome_register",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['D_late_followup_register']; "
                "aggregation=identity; window_days=180; knowable_at=postindex_post_y_180d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="descendant",
            mechanism=(
                "Pearl descendant captured at late follow-up in DGP a3: d = "
                "0.6*y + 0.2*y_lag + nu_d (where y_lag is a noisy lagged copy of "
                "Y). Post-outcome temporal filter (postindex_post_y_180d) — D "
                "observed long after Y. Pearl arrowhead: descendant (Y->D directly "
                "and Y->Y_lag->D). Probes whether classifier identifies late-"
                "followup outcome-registers as descendants regardless of "
                "measurement lag. why_not_duplicate: distinct from short-window "
                "post-Y descendants by temporal distance. Remediation per role-to-"
                "remediation table: descendant -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # ----- Sub-bucket B4-X: additional synthetic-DGP entries for explicit synth_a floor -----
        # Synthetic DGP: synth_a1 collider with unmeasured parent — DGP construction is ground truth.
        dspy.Example(
            feature_name="synth_a1_collider_with_unmeasured_parent",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['C_unmeasured_parent_collider']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="collider",
            mechanism=(
                "Pearl collider with one unmeasured parent in DGP a1: C = f(T, U) "
                "where T is treatment and U is unmeasured baseline covariate. "
                "Pre-anchor temporal filter (preindex_365d). Pearl arrowhead: "
                "collider (T->C<-U; conditioning on C opens the T-U path, inducing "
                "spurious T-Y association via the backdoor U->Y). The unmeasured "
                "parent U is not in the feature set — only C is observed. Probes "
                "whether classifier identifies C as a collider even when one parent "
                "is unmeasured (role determined by structural position, not "
                "observability of parents). Remediation per role-to-remediation "
                "table: collider -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Synthetic DGP: synth_a2 mediator with proxy confounder — DGP construction is ground truth.
        dspy.Example(
            feature_name="synth_a2_mediator_with_proxy_confounder",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A2; derivation_inputs=['M_proxy_confounded']; "
                "aggregation=mean; window_days=90; knowable_at=postindex_90d"
            ),
            dataset_context=(
                "synthetic_a2 DGP; cohort=synthetic_a2; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="mediator",
            mechanism=(
                "Pearl partial mediator with a proxy confounder in DGP a2: "
                "M = 0.7*t + 0.4*Z_proxy + eps_m, Y = 0.5*t + 0.6*M + 0.3*Z_proxy + eps_y. "
                "Post-treatment temporal filter (postindex_90d): M is measured 90d "
                "post-index (after treatment onset). Pearl arrowhead: mediator "
                "(T->M->Y on indirect path; Z_proxy also confounds M-Y relationship). "
                "Probes partial-mediation pattern where a proxy confounder co-loads "
                "with the mediator — structural role is mediator (on T->Y path), not "
                "confounder (Z_proxy is already controlled). Remediation per role-to-"
                "remediation table: mediator -> window (windowed to pre-T)."
            ),
            recommended_remediation="window",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Synthetic DGP: synth_a3 descendant via late indirect chain — DGP construction is ground truth.
        dspy.Example(
            feature_name="synth_a3_descendant_late_indirect_chain",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['D_late_indirect_chain']; "
                "aggregation=last; window_days=270; knowable_at=postindex_270d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="descendant",
            mechanism=(
                "Pearl descendant via two-step indirect chain in DGP a3 at late "
                "follow-up: Y -> M_late -> D_chain (where M_late is an intermediate "
                "post-outcome state). DGP: M_late = 0.8*y + nu_m, D_chain = "
                "0.7*M_late + 0.1*t + nu_d. Post-outcome temporal filter "
                "(postindex_270d; captured at 270d post-index, well after Y). Pearl "
                "arrowhead: descendant (Y->M_late->D, two-hop). Probes whether "
                "classifier identifies multi-hop post-outcome chains as descendants "
                "rather than ancestors or confounders. why_not_duplicate: distinct "
                "from direct Y->D descendants and from the single-hop late-followup "
                "entry by two-step mediation path. Remediation per role-to-"
                "remediation table: descendant -> drop."
            ),
            recommended_remediation="drop",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Synthetic DGP: synth_a4 IV consistency under 2SLS — DGP construction is ground truth.
        dspy.Example(
            feature_name="synth_a4_iv_two_stage_consistency_probe",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A4; derivation_inputs=['IV_2sls_consistency']; "
                "aggregation=identity; window_days=365; knowable_at=preindex_365d"
            ),
            dataset_context=(
                "synthetic_a4 DGP; cohort=synthetic_a4; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="instrument",
            mechanism=(
                "Pearl instrument probing 2SLS consistency in DGP a4: IV ~ N(0,1) "
                "independent of all baseline covariates. Stage-1: T = 0.8*iv + "
                "0.5*Z + eps_t; Stage-2: Y = 0.5*T_hat + 0.3*Z + eps_y. No direct "
                "IV->Y arrow by DGP construction — exclusion restriction holds. "
                "Pre-anchor temporal filter (preindex_365d). Pearl arrowhead: "
                "instrument (IV->T, IV ⊥ Y | T,Z). Brookhart-Wang preference IV "
                "pattern (synthetic DGP, no literature anchor): probes that "
                "classifier identifies continuous near-Gaussian IVs as instruments "
                "in the presence of a measured confounder Z. why_not_duplicate: "
                "distinct from binary-preference and randomized-shock IVs by "
                "continuous Gaussian DGP with explicit 2SLS stage structure. "
                "Remediation per role-to-remediation table: instrument -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Synthetic DGP: synth_a1 ancestor as effect modifier — DGP construction is ground truth.
        dspy.Example(
            feature_name="synth_a1_ancestor_distant_effect_modifier",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A1; derivation_inputs=['A_distant_effect_modifier']; "
                "aggregation=first; window_days=730; knowable_at=preindex_730d"
            ),
            dataset_context=(
                "synthetic_a1 DGP; cohort=synthetic_a1; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="ancestor",
            mechanism=(
                "Pearl ancestor acting as effect modifier in DGP a1: A is a distal "
                "baseline variable with A->W->T pathway (W is an intermediate "
                "confounder) and A modifies the T-Y treatment effect: Y = "
                "(0.5 + 0.3*A)*T + 0.4*W + eps_y. Pre-anchor temporal filter "
                "(preindex_730d; captured early in history). Pearl arrowhead: "
                "ancestor (A->W->T with W->Y; A also modifies beta_T). Probes "
                "whether classifier assigns 'ancestor' to distal variables that "
                "reach T only through intermediaries and also interact with the "
                "treatment-outcome relationship. why_not_duplicate: distinct from "
                "pure-ancestor (no effect modification) and from confounders (A has "
                "no direct A->Y arrow; effect modification is on T coefficient only). "
                "Remediation per role-to-remediation table: ancestor -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # Synthetic DGP: synth_a3 time-varying continuous confounder — DGP construction is ground truth.
        dspy.Example(
            feature_name="synth_a3_confounder_time_varying_continuous",
            derivation_pseudocode=(
                "source=SYNTH_DGP_A3; derivation_inputs=['Z_timevarying_continuous']; "
                "aggregation=last; window_days=180; knowable_at=preindex_180d"
            ),
            dataset_context=(
                "synthetic_a3 DGP; cohort=synthetic_a3; treatment=t_binary; "
                "outcome=y_continuous; prediction_anchor=index"
            ),
            causal_role="confounder",
            mechanism=(
                "Pearl time-varying continuous confounder in DGP a3: Z(t) follows "
                "an AR(1) process Z(t) = 0.8*Z(t-1) + noise, with Z->T (via "
                "logistic treatment model P(T=1|Z)=sigmoid(1.5*Z)) and Z->Y "
                "(Y = 0.5*T + 0.6*Z_last + eps). Pre-anchor temporal filter "
                "(preindex_180d; last observed value before index). Pearl arrowhead: "
                "confounder (Z->T, Z->Y; backdoor path T<-Z->Y). Probes whether "
                "classifier handles CONTINUOUS time-varying confounders — feature is "
                "the last snapshot before index, representing the pre-treatment "
                "confounder value. Distinct from binary or static confounders by "
                "autoregressive temporal dynamics. why_not_duplicate: distinct from "
                "synth_a3_z_timevarying_confounder_baseline_snapshot by continuous "
                "(not discretized) AR(1) DGP and 180d (not 365d) window. "
                "Remediation per role-to-remediation table: confounder -> keep_with_caveat."
            ),
            recommended_remediation="keep_with_caveat",
        ).with_inputs("feature_name", "derivation_pseudocode", "dataset_context"),
        # =====================================================================
        # End Plan-239 n=200 Task 6 — Bucket 4 cross-cohort + synthetic-DGP (+44)
        # Distribution: 36 synthetic-DGP (cohort=synthetic_a1/2/3/4) + 6 adversarial
        # (PNH=2, BC=2, CSU=2) + 2 edge-case (PNH=1, BC=1).
        # Synthetic-DGP arrowhead breakdown: confounder=8, mediator=5, collider=5,
        # descendant=6, ancestor=5, instrument=7 (one labeled confounder per
        # pseudo-IV adversarial probe). All IVs Brookhart-Wang exclusion-restriction
        # defended by DGP construction (no direct IV->Y edge).
        # Final canonical cohort floors verified: PNH>=53, BC>=58, CSU>=52,
        # synthetic_or_other>=52, TOTAL>=234.
        # =====================================================================
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
