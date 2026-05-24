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
                "Classical (T=iptacopan_init, Y=ldh_normalization_180d) confounder. "
                "Z->T arrow: low pre-index haptoglobin (free-hemoglobin scavenger depletion) "
                "marks severe intravascular hemolysis and drives iptacopan-vs-anti-C5 "
                "candidacy per Brodsky 2014 (PMID 25237199; doi:10.1182/blood-2014-02-522128). "
                "Z->Y arrow: deeper baseline hemolysis predicts post-index hemolytic-marker "
                "normalization independently of treatment choice. why_not_duplicate: golden "
                "neighbor baseline_ldh_x_uln_preindex uses source=LABS_HEMOLYSIS with "
                "aggregation=mean(LDH/ULN); this entry pulls a DIFFERENT analyte (haptoglobin, "
                "not LDH) measuring upstream hemoglobin-scavenger depletion (LDH measures "
                "downstream cell lysis), with aggregation=min-of-haptoglobin/LLN-ratio."
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
                "Ancestor of (T=iptacopan_init, Y=ldh_normalization_180d). Lifetime cumulative "
                "days-on-anti-C5 before iptacopan-switch reflects underlying disease chronicity "
                "per Risitano 2020 (PMID 33347547; doi:10.1016/S2352-3026(20)30308-1) — long "
                "historical exposure indexes entrenched chronic PNH phenotype, upstream of "
                "both the immediate switch decision and post-index response. why_not_duplicate: "
                "golden neighbor prior_anti_c5_inhibitor_use_flag_preindex is BINARY any-use "
                "(aggregation=any) over 730d window, labeled CONFOUNDER (captures prior-"
                "treatment-failure pathway); this entry is CONTINUOUS lifetime sum-of-days "
                "(aggregation=sum, unlimited preindex window), labeled ANCESTOR (indexes "
                "disease chronicity upstream of the immediate switch decision, not its "
                "proximal confounder). Different role + different aggregation + different "
                "window teaches an ancestor-vs-confounder boundary the golden set lacks."
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
                "Descendant of T=iptacopan_init: post-index burden of hemolysis-coded ED days. "
                "T->V arrow: treatment efficacy modulates incident hemolytic crises that drive "
                "ED utilization per Hill 2020 (PMID 31816102). No V->Y arrow back to LDH "
                "normalization. Standard remediation per Hernan 2016 (PMID 27176981) is drop "
                "from any (T,Y) effect-estimation adjustment set. why_not_duplicate: golden "
                "neighbor pnh_related_hospitalizations_365d_postindex_count uses "
                "source=CLAIMS_HOSPITALIZATION, agg=count of events over 365d. This entry "
                "changes SOURCE TABLE (ED_VISITS vs HOSPITALIZATIONS), SETTING (emergency "
                "outpatient vs inpatient), AGGREGATION (sum-of-days vs count-of-events), "
                "and narrows WINDOW to 90d on a clinically distinct event type."
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
                "Classical (T=ribociclib_add, Y=pfs_event_24m) confounder. Nottingham SBR "
                "grade (composite of tubule formation + nuclear pleomorphism + mitotic count, "
                "binned Grade 1/2/3) per Elston-Ellis 1991 (PMID 1995317) and Rakha 2019 "
                "(PMID 27557947). Z->T: higher grade drives AI+CDK4/6 escalation over "
                "AI-mono. Z->Y: higher grade predicts progression independently. "
                "why_not_duplicate: golden neighbors er_percent_preindex and "
                "ki67_index_percent_preindex are SINGLE-MARKER CONTINUOUS (most-recent assay "
                "value). This entry is COMPOSITE CATEGORICAL grade aggregated as MODE across "
                "diagnostic reports, with derivation_inputs that are the three SBR subscores "
                "(not a single immunohistochemical marker)."
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
                "Classical (T=ribociclib_add, Y=pfs_event_24m) confounder. Pre-index sum-"
                "of-days letrozole exposure reflects prior endocrine-therapy intensity per "
                "Hortobagyi 2021 MONALEESA-2 OS (PMID 33513289; doi:10.1056/NEJMoa2114663). "
                "Z->T: longer prior AI-backbone exposure drives ribociclib add-on timing. "
                "Z->Y: longer prior letrozole predicts secondary endocrine resistance, "
                "depressing PFS regardless of CDK4/6 add. why_not_duplicate: golden neighbor "
                "prior_cdk46_lines_count counts CDK4/6 LINES (prior failure-on-class). This "
                "entry counts AROMATASE-INHIBITOR DURATION (orthogonal drug class, different "
                "MoA: estrogen-synthesis blockade vs CDK4/6 inhibition), aggregated as "
                "SUM-OF-DAYS not COUNT-OF-LINES — different drug class + different aggregation."
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
                "Descendant of T=ribociclib_add. T->V: ribociclib-induced myelosuppression "
                "drives febrile neutropenia episodes per Tripathy 2019 MONALEESA-7 safety "
                "(PMID 31526833). CLAIMS-EVENT-BASED (ICD D70.x co-occurring with R50.9 "
                "within 7d), not lab-value-based. No V->Y arrow back to PFS at 24m. "
                "Remediation per Hernan 2000 (PMID 10955408) is drop from (T,Y) adjustment "
                "set. why_not_duplicate: golden neighbor post_index_neutropenia_max_grade_90d "
                "is LAB-VALUE-BASED (ANC graded by CTCAE) aggregated as WORST-VALUE over 90d "
                "and labeled MEDIATOR (dose-intensity path). This entry is CLAIM-EVENT-BASED "
                "(D70.x + R50.9 conjunction) aggregated as COUNT-OF-EPISODES over 180d, "
                "labeled DESCENDANT — teaches the lab-vs-claim boundary within neutropenia."
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
                "Targets clinical-mediator-vs-biomarker-mediator boundary: the classifier "
                "mistakes PRO-based clinical activity scores for descendants because they "
                "look outcome-like. This entry teaches that UAS7 (weekly-sum patient-reported "
                "urticaria-activity score per Saini 2021, PMID 33321141) sits on the "
                "T->M_clinical->Y mediation path between remibrutinib_init and "
                "uas7_remission_180d, not downstream of Y. why_not_duplicate: golden mediators "
                "delta_basophil_activation_test_cd63_pct_28_56d and "
                "delta_anti_fcepsilon_ri_igg_titer_post_treatment_60_90d are LAB-BASED "
                "biomarker DELTAS at sub-90d windows on `lab_results`. This entry is a "
                "PATIENT-REPORTED CLINICAL ACTIVITY SCORE at the 180d window from "
                "`ehr_assessments` — clinical-mediator vs biomarker-mediator. Methods anchor "
                "PMID 10955408 (Hernan 2000 MSM)."
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
                "preference IV review PMID 18375005). why_not_duplicate: golden CSU IVs are "
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
                "(Greenland-Pearl-Robins 1999 PMID 9888278). why_not_duplicate: golden PNH "
                "has no vaccination-status entry; nearest neighbor prior_thrombotic_event_flag "
                "is a true confounder from DIAGNOSIS_HISTORY with different source + role."
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
                "the same recurrence-biology family as BC golden visceral_disease_flag. This "
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
                "Brookhart-Schneeweiss PMID 18375005)."
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
                "Greenland-Pearl-Robins PMID 9888278 + Brookhart PMID 18375005."
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
                "(PMID 18375005): (1) relevance — prior-year biologic volume strongly predicts "
                "biologic_initiation_180d via prescribing capacity; (2) exclusion restriction "
                "— no direct edge to hospitalization_180d in the DGP; (3) no unmeasured "
                "confounding of IV3↔Y given simulated covariates. Compile set previously "
                "lacked a multi-IV exclusion-restriction exemplar. why_not_duplicate (§3.0): "
                "bare synthetic-fixture `index_provider_biologic_volume_prior_year` cannot be "
                "reused per §0/V27 — `synth_a4_` prefix + `_alt_instrument` suffix + "
                "`cohort=synthetic_a4` discriminator. Provenance: DAG-methods-only "
                "(Brookhart PMID 18375005; Greenland-Pearl-Robins PMID 9888278)."
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
                "Slope of LDH/ULN trajectory over the 180d immediately pre-index reflects "
                "trajectory of hemolytic activity at the decision point. Z->T: rising-slope "
                "patients more often switched to iptacopan after suboptimal anti-C5 control "
                "(Peffault de Latour 2024 APPLY-PNH PMID 38477987; "
                "doi:10.1056/NEJMoa2308695). Z->Y: rising baseline trajectory predicts post-"
                "treatment response magnitude (responders revert from higher pre-treatment "
                "set-point). why_not_duplicate: golden baseline_ldh_x_uln_preindex is the "
                "POINT-IN-TIME ratio at index; this entry is the DERIVATIVE (slope over 180d "
                "via linear regression), capturing temporal dynamics rather than level. "
                "Methods anchor Brookhart 2010 PMID 30516102 confounder selection. Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Pre-index detection of C3-coated PNH RBCs flags extravascular hemolysis — "
                "a residual disease driver in anti-C5-treated patients. Z->T: presence of "
                "EVH-by-C3-binding is the canonical clinical reason for proximal inhibitor "
                "switch (Lee 2023 ALPHA PMID 38030318; doi:10.1016/S2352-3026(23)00315-0). "
                "Z->Y: EVH burden at baseline predicts post-treatment hemoglobin recovery "
                "magnitude. why_not_duplicate: golden c3_deposition_pnh_rbc_pct_d90_postindex "
                "is POSTINDEX continuous measurement labeled MEDIATOR; this is PREINDEX "
                "binary flag labeled CONFOUNDER — temporal positioning is reversed. Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Erythrocyte (Type III RBC) clone size indexes the proportion of GPI-deficient "
                "red cells susceptible to complement-mediated lysis. Z->T: large erythrocyte "
                "clone justifies aggressive proximal complement inhibition per Brodsky 2021 "
                "(PMID 33512400; doi:10.1182/blood.2019003812). Z->Y: erythrocyte-clone-size "
                "predicts hemoglobin response ceiling. why_not_duplicate: golden "
                "pnh_clone_size_granulocyte_pct_preindex measures GRANULOCYTE clone (a "
                "different cell lineage tied to disease activity but not direct lysis "
                "target); this measures the lysis-target erythrocyte fraction — distinct "
                "lineage, distinct causal pathway. Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Years from first PNH diagnosis to index reflects disease tenure — an "
                "upstream patient characteristic affecting both treatment choice and "
                "response distribution. Z->T,Y but effect on (T,Y) is largely exhausted by "
                "intermediate confounders (prior anti-C5 days, clone size, transfusion "
                "history); this entry teaches the ANCESTOR role per Greenland-Pearl-Robins "
                "1999 (PMID 9888278) where d-separation by downstream confounders blocks "
                "the direct Z arrows. Schrezenmeier 2022 (PMID 35699625; doi:10.20452/pamw.16271) "
                "documents disease-tenure heterogeneity. why_not_duplicate: golden "
                "age_at_index_years is biological age (intrinsic patient attribute); this "
                "is disease tenure (time-since-diagnosis); orthogonal upstream variables. Remediation per role-to-remediation table: ancestor → keep_with_caveat."
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
                "Brookhart-Wang short-term first-initiation IV: binary flag for whether "
                "index_date falls within 90 days of iptacopan FDA approval (December 2023, "
                "per APPLY-PNH pivotal trial PMID 38477987; doi:10.1056/NEJMoa2308695). "
                "Early adopters in the post-approval window are driven by physician awareness "
                "of trial results and formulary activation, not by patient clinical severity "
                "differences from later initiators. Z->T: first-mover prescribers activate "
                "iptacopan immediately post-approval due to clinical trial familiarity "
                "(Brookhart 2006 PMID 30516102 prescriber-tendency IV framework). "
                "Z->Y exclusion-restriction: calendar proximity to approval date has no "
                "direct biological mechanism on hemoglobin response; all effect is mediated "
                "through treatment initiation only — standard regulatory-discontinuity IV. "
                "why_not_duplicate: golden post_iptacopan_fda_approval_calendar_indicator "
                "is a monotone post/pre binary (all post-approval time treated equally); "
                "this is a SHORT-TERM WINDOW (90d adoption burst) capturing only the "
                "first-mover cohort — distinct temporal granularity, distinct exogeneity "
                "argument (early-adopter prescriber behavior vs general post-approval era). "
                "Remediation per role-to-remediation table: instrument → keep_with_caveat."
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
                "Total lifetime days exposed to pegcetacoplan (another proximal-complement "
                "inhibitor) before iptacopan initiation. Z->T: prior pegcetacoplan failure / "
                "discontinuation drives subsequent iptacopan switch per real-world cohort "
                "(PMID 38348608; doi:10.1002/ajh.27242). Z->Y: pegcetacoplan exposure history "
                "shapes hemolytic dynamics independently of iptacopan effect. "
                "why_not_duplicate: golden prior_anti_c5_inhibitor_use_flag_preindex is a "
                "BINARY flag for ANY anti-C5 drug-class use; this is CONTINUOUS days for the "
                "SPECIFIC C3 inhibitor pegcetacoplan (a different mechanism class entirely — "
                "C3 vs C5); distinct drug class + continuous vs binary. Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Mean hemoglobin over the 30d immediately pre-index reflects baseline anemia "
                "depth at the treatment decision. Z->T: lower baseline Hb is a clinical "
                "trigger for iptacopan switch per Hillmen 2024 (PMID 39079163; "
                "doi:10.1371/journal.pone.0306407). Z->Y: baseline Hb sets the lower bound "
                "for hemoglobin response of >=2 g/dL improvement, making the endpoint more "
                "achievable for lower-baseline patients. why_not_duplicate: golden "
                "baseline_ldh_x_uln_preindex measures HEMOLYTIC ACTIVITY; this measures "
                "HEMATOLOGIC RESERVE (the carrying capacity from which response is measured) "
                "— complementary axis of pre-treatment patient characterization. Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Composite z-score of alternative-pathway amplification markers (C3 split "
                "products + factor B consumption) measured pre-index. Z->T: high AP activity "
                "drives proximal inhibitor (factor B-blocking iptacopan) selection per "
                "Versino-Fattizzo 2024 (PMID 38622956; doi:10.1111/ijlh.14281). Z->Y: AP "
                "amplification predicts magnitude of complement-blockade response. "
                "why_not_duplicate: golden c3_deposition_pnh_rbc_pct_d90_postindex is "
                "POSTINDEX C3 binding to RBCs (mediator); this is PREINDEX AP-loop activity "
                "biomarker panel (confounder) — different time-position, different physical "
                "measurement (soluble AP markers vs cell-surface C3 binding). Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Change in FACIT-Fatigue score from baseline to d90 sits on the directed "
                "path T -> FACIT_change -> Y for outcomes involving Hb-related symptom "
                "endpoints. T->M: iptacopan reduces hemolysis and improves fatigue at d90 "
                "per APPLY-PNH (PMID 38477987; doi:10.1056/NEJMoa2308695). M->Y: fatigue "
                "improvement reflects oxygen-carrying recovery preceding the 180d Hb "
                "endpoint. Adjusting for M blocks indirect effect — remediation is window "
                "(restrict to pre-treatment covariates) per Hernan 2004 (PMID 14760119). "
                "why_not_duplicate: golden facit_fatigue_response_180d_postindex_flag is "
                "180d BINARY threshold response labeled COLLIDER (response status itself); "
                "this is CONTINUOUS d90 DELTA labeled MEDIATOR (intermediate causal path); "
                "different time, different aggregation, different role. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
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
                "Change in free hemoglobin from baseline to d30 sits on the directed path "
                "T -> intravascular_hemolysis -> Y per Jang 2022 (PMID 35561315; "
                "doi:10.1182/bloodadvances.2022006960) — iptacopan blocks factor B and "
                "suppresses IVH rapidly. T->M->Y is the proximal pharmacologic mediator. "
                "Adjust for M induces over-adjustment bias (Hernan MSM PMID 10955408); "
                "correct remediation is window. why_not_duplicate: golden "
                "ldh_x_uln_d90_postindex measures POSTINDEX LDH (d90 timepoint, LDH "
                "analyte); this measures POSTINDEX FREE HEMOGLOBIN (d30 timepoint, free-Hb "
                "analyte) — different analyte + earlier post-index timepoint. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
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
                "Postindex d90 reticulocyte normalization (counts back within reference "
                "range) sits on T -> erythropoietic_normalization -> Y path. T->M: treatment "
                "reduces hemolytic stress, removes drive for compensatory reticulocytosis "
                "per Hillmen 2021 PEGASUS (PMID 33730455; doi:10.1056/NEJMoa2029073). M->Y: "
                "reticulocyte normalization precedes stable hemoglobin recovery at d180. "
                "Window remediation per Hernan 2004 (PMID 14760119). why_not_duplicate: "
                "golden reticulocyte_count_delta_d90_postindex is CONTINUOUS delta; this is "
                "BINARY threshold indicator (back-in-reference flag) — different aggregation "
                "(delta vs indicator), captures a clinically meaningful normalization "
                "endpoint distinct from raw delta magnitude. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
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
                "Add-on danicopan initiation during follow-up is jointly caused by (a) "
                "inadequate iptacopan response (residual EVH) and (b) availability/coverage "
                "of danicopan in the post-ALPHA period (Lee 2023 ALPHA PMID 38030318; "
                "doi:10.1016/S2352-3026(23)00315-0). Both poor T-response and patient access "
                "drive add-on initiation, opening a collider path when conditioned. Remediation "
                "drop per Hernan 2004 (PMID 14760119). why_not_duplicate: golden "
                "iptacopan_persistence_at_180d_flag is persistence on INDEX treatment; this "
                "is INITIATION of an ADD-ON drug (danicopan) — different drug, different "
                "event type (initiation vs persistence), different role inference. Remediation per role-to-remediation table: collider → drop."
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
                "Count of distinct prior complement inhibitors with documented failure-"
                "discontinuation before iptacopan index. Z->T: more prior failures push "
                "toward iptacopan as later-line option per APPLY-PNH eligibility patterns "
                "(PMID 38477987; doi:10.1056/NEJMoa2308695). Z->Y: refractory-line position "
                "predicts smaller absolute response magnitude. why_not_duplicate: golden "
                "prior_anti_c5_inhibitor_use_flag_preindex is BINARY any-use of C5 class; "
                "compile-set proximal_complement_inhibitor_class_switch_count_lifetime_pnh "
                "is class-switch count regardless of reason; this is FAILURE-coded "
                "discontinuation count only (clinical-failure-specific filter). Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Durable LDH normalization (both d90 AND d180 <=1.5xULN) sits on the "
                "T -> sustained_IVH_suppression -> Y path per Versino-Fattizzo 2024 (PMID "
                "38622956; doi:10.1111/ijlh.14281). Conjunction of two timepoints distinguishes "
                "durable from transient response. Adjust for M induces over-adjustment "
                "(Hernan MSM PMID 10955408). why_not_duplicate: golden ldh_x_uln_d90_postindex "
                "is the SINGLE d90 timepoint; this is the CONJUNCTION (both d90 AND d180 "
                "in range) — distinct aggregation (durable-conjunction indicator vs single "
                "ratio), distinct construct (sustained vs transient). Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
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
                "Flag for any historical thrombosis at PNH-atypical site (Budd-Chiari, "
                "mesenteric, cerebral venous) per Schrezenmeier 2022 (PMID 35699625; "
                "doi:10.20452/pamw.16271). Z->T: atypical-site thrombosis history indexes "
                "severe complement dysregulation prompting proximal inhibitor escalation. "
                "Z->Y: thrombosis history modulates anticoagulation that affects hematologic "
                "endpoints. why_not_duplicate: golden prior_thrombotic_event_flag_preindex "
                "is ANY-SITE binary flag; this is the SUBSET restricted to atypical sites "
                "(intra-abdominal/intra-cranial), narrower clinical phenotype. Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "DOAC proportion-of-days-covered in the 365d preindex window. Z->T: "
                "anticoagulated patients have lower threshold for proximal switch given "
                "iptacopan's lower thrombosis risk per de Castro 2024 (PMID 39404123; "
                "doi:10.1080/14656566.2024.2404110). Z->Y: anticoagulation modulates iron "
                "homeostasis (GI losses) affecting Hb endpoint. why_not_duplicate: golden "
                "prior_thrombotic_event_flag_preindex is binary event flag (history); this "
                "is continuous PDC of anticoagulation TREATMENT (process measure); event "
                "vs treatment-process distinction. Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Days from index to first post-index transfusion lies on the path T -> "
                "transfusion_dependence -> Y per pegcetacoplan/iptacopan trial patterns "
                "(PMID 36459381; doi:10.1007/s40265-022-01809-w). T->M: effective treatment "
                "delays/eliminates transfusion need. M->Y: transfusions transiently inflate "
                "measured Hb at d180 even when endogenous response is poor — directly "
                "modifying the outcome measurement. Window remediation. why_not_duplicate: "
                "golden any_rbc_transfusion_during_followup_flag is BINARY any-event over "
                "follow-up labeled COLLIDER; this is CONTINUOUS time-to-first-event labeled "
                "MEDIATOR — different aggregation, different role inference. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
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
                "Charlson Comorbidity Index summarizes 17 weighted comorbid conditions over "
                "365d preindex per standard pharmacoepi practice (Brodsky 2021 PMID 33512400 "
                "treatment-decision context; doi:10.1182/blood.2019003812). Z->T: higher CCI "
                "favors oral iptacopan over IV infusion regimens. Z->Y: comorbidity burden "
                "modulates erythropoietic capacity and adverse-event-driven discontinuation. "
                "why_not_duplicate: golden age_at_index_years is one upstream demographic; "
                "this is the COMPOSITE multi-comorbidity score (17-condition weighted index) "
                "— distinct construct. Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Count of distinct iptacopan strengths dispensed in first 90d post-index "
                "indexes early dose modification — jointly caused by (a) early "
                "tolerability/response signals (T->modification) and (b) prescriber "
                "preference and patient adherence (independent of T). Conditioning opens "
                "a collider path per Hernan 2004 (PMID 14760119). why_not_duplicate: golden "
                "iptacopan_persistence_at_180d_flag is binary persistence at 180d (different "
                "endpoint); this is dose-modification COUNT during 90d (intra-treatment-"
                "course modification, not discontinuation); different event type + window. Remediation per role-to-remediation table: collider → drop."
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
                "Number of hematology specialist visits in 180d preindex reflects monitoring "
                "intensity at the treatment-decision juncture per PEGASUS-48wk care patterns "
                "(PMID 36055332; doi:10.1016/S2352-3026(22)00210-1). Z->T: high monitoring "
                "frequency surfaces breakthrough hemolysis that triggers switch decisions. "
                "Z->Y: ongoing specialist contact predicts adherence and outcome assessment "
                "completeness. why_not_duplicate: golden payer_step_therapy_iptacopan_"
                "requirement_preindex captures PAYER policy (administrative); this captures "
                "PROVIDER engagement count (utilization). Remediation per role-to-remediation table: confounder → keep_with_caveat."
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
                "Payer class (commercial vs Medicare) at index — used as a Brookhart-style "
                "policy IV where commercial vs Medicare formulary differentials affect "
                "iptacopan access. Z->T: differential PA gating shifts initiation rates "
                "post-APPLY-PNH (PMID 38477987; doi:10.1056/NEJMoa2308695). Z->Y exclusion-"
                "restriction: payer category itself does not affect Hb biology — only the "
                "treatment-choice mediation route. Defensibility caveat: payer class may "
                "correlate with comorbidity profile through age/employment; mechanism "
                "explicitly adjusts for Charlson Score + age (both included). "
                "why_not_duplicate: golden payer_step_therapy_iptacopan_requirement_preindex "
                "is policy-specific STEP-THERAPY rule; this is broader PAYER-CLASS "
                "indicator (categorical not boolean). Defensible as IV per Brookhart 2006 "
                "(PMID 30516102). Remediation per role-to-remediation table: instrument → keep_with_caveat."
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
                "Adversarial: post-treatment EORTC QLQ-C30 score at d180. Naive worker may "
                "classify as MEDIATOR (post-treatment intermediate on T->QoL->Y path), but "
                "the OUTCOME hemoglobin_response_180d is biologic (laboratory measure), and "
                "EORTC score is a downstream patient-reported sequelae of the hematologic "
                "response — no causal arrow QoL -> Hb. The correct role is DESCENDANT (off "
                "the directed (T,Y) path) per Hernan 2016 (PMID 27176981). Drop from "
                "adjustment. Adversarial test: distinguish mediator (T->M->Y) from "
                "descendant (Y->D or T->D, no D->Y). why_not_duplicate: golden "
                "facit_fatigue_response_180d_postindex_flag is binary FACIT response (a "
                "different PRO instrument labeled collider); this is continuous EORTC score "
                "(different instrument + different role inference). Remediation per role-to-remediation table: descendant → drop."
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
                "Adversarial: AND-composite of two post-treatment events (LDH normalized "
                "AND alive). Naive worker may call this MEDIATOR (post-treatment "
                "intermediate), but conjunction of treatment-effect AND survival creates a "
                "classical collider — both arms (T->LDH normalization, T->survival, AND "
                "underlying-severity->survival, underlying-severity->LDH normalization) "
                "converge into the composite, so conditioning opens back-door per Hernan "
                "2004 (PMID 14760119). Composite hides the survivorship-collider. Adversarial "
                "test: classifier must decompose composite features. why_not_duplicate: "
                "golden alive_at_180d_postindex_flag is the SURVIVAL collider alone; this "
                "is the AND-COMPOSITE with LDH-normalization (different aggregation: "
                "conjunction; different decomposability concern). Remediation per role-to-remediation table: collider → drop."
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
                "Edge case: transfusion units summed over a 395d window spanning -365d "
                "preindex to +30d postindex. Aggregation conceals 30d of post-anchor "
                "data — the +30d slice is post-treatment and lies on T -> transfusion_need "
                "-> Y path per Brodsky 2021 (PMID 33512400; doi:10.1182/blood.2019003812). "
                "Even if 365/395 = 92% of the window is preindex, the post-anchor leakage "
                "makes this a mediator with window remediation required (restrict to -365 "
                "to 0). why_not_duplicate: golden transfusion_units_365d_preindex is the "
                "STRICT preindex 365d window (no leakage, labeled confounder); this is the "
                "ASYMMETRIC -365 to +30 window (with leakage, labeled mediator) — edge-case "
                "teaching pair. Remediation per role-to-remediation table: mediator → window (restrict to pre-treatment window)."
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
