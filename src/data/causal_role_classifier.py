"""Layer 4 — DSPy-compiled CausalRoleClassifier.

Classifies a proposed feature derivation as ancestor / confounder / mediator /
collider / descendant / instrument with respect to the prediction target.
Compiled on a curated set of 18 documented past-leakage incidents from this
codebase's history (`.claude/state/leakage_compile_set_20260507.md`).

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

    feature_name: str = dspy.InputField(
        desc="Name of the feature being classified."
    )
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
    """Build the DSPy compile set from the 18 documented leakage incidents.

    Source: .claude/state/leakage_compile_set_20260507.md
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
