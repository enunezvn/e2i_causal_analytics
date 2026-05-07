"""Tests for Layer 4 — CausalRoleClassifier (DSPy program).

These tests exercise the DSPy program's STRUCTURE (signature, fields, compile
set) without making actual LLM calls. End-to-end LLM-based classification is
tested separately in tests/integration/ where API keys / mocks are managed.
"""

from __future__ import annotations


def test_compile_set_has_diverse_examples():
    """The compile set must cover multiple causal roles, not just one."""
    from src.data.causal_role_classifier import build_compile_set, get_compile_set_summary

    examples = build_compile_set()
    summary = get_compile_set_summary()

    assert len(examples) == summary["n_examples"], (
        f"summary.n_examples ({summary['n_examples']}) must match build_compile_set() len ({len(examples)})"
    )
    assert summary["n_examples"] >= 8, (
        f"Compile set too small: {summary['n_examples']}; need at least 8"
    )
    # Must have at least 2 distinct roles (else DSPy can't learn classification)
    assert len(summary["role_distribution"]) >= 2, (
        f"Compile set role distribution too narrow: {summary['role_distribution']}"
    )

    # Must have descendants (the dominant leak pattern) AND non-descendants
    # (legit features) for class balance
    assert summary["role_distribution"].get("descendant", 0) >= 3
    non_descendant = sum(v for k, v in summary["role_distribution"].items() if k != "descendant")
    assert non_descendant >= 2, (
        f"Compile set must include non-descendant examples for balance; "
        f"got {summary['role_distribution']}"
    )


def test_each_compile_set_example_has_required_fields():
    """Every example must have: feature_name, derivation_pseudocode, dataset_context,
    causal_role, mechanism, recommended_remediation.
    """
    from src.data.causal_role_classifier import build_compile_set

    examples = build_compile_set()
    required_fields = {
        "feature_name",
        "derivation_pseudocode",
        "dataset_context",
        "causal_role",
        "mechanism",
        "recommended_remediation",
    }
    for i, ex in enumerate(examples):
        for field in required_fields:
            assert hasattr(ex, field), f"Example {i} missing field {field}"
            value = getattr(ex, field)
            assert value is not None and value != "", f"Example {i} has empty {field}"


def test_compile_set_journey_duration_classified_as_mediator():
    """The journey_duration_days incident should be labeled `mediator` (not
    descendant) because it's NOT a direct downstream of treatment but rather
    an aggregate that includes post-treatment events on the causal path.
    Used as a difficult example for the LLM to learn from.
    """
    from src.data.causal_role_classifier import build_compile_set

    examples = build_compile_set()
    journey = next((e for e in examples if e.feature_name == "journey_duration_days"), None)
    assert journey is not None, "journey_duration_days example missing from compile set"
    assert journey.causal_role == "mediator", (
        f"Expected journey_duration_days to be classified as 'mediator'; got {journey.causal_role}"
    )
    assert journey.recommended_remediation == "window", (
        f"Expected journey_duration_days remediation = 'window'; "
        f"got {journey.recommended_remediation}"
    )


def test_signature_has_correct_io_fields():
    """Verify DSPy Signature schema."""
    from src.data.causal_role_classifier import CausalRoleSignature

    # Inputs
    assert "feature_name" in CausalRoleSignature.input_fields
    assert "derivation_pseudocode" in CausalRoleSignature.input_fields
    assert "dataset_context" in CausalRoleSignature.input_fields

    # Outputs
    assert "causal_role" in CausalRoleSignature.output_fields
    assert "mechanism" in CausalRoleSignature.output_fields
    assert "recommended_remediation" in CausalRoleSignature.output_fields


def test_classifier_module_constructs():
    """The CausalRoleClassifier module must construct without an LM connection."""
    from src.data.causal_role_classifier import CausalRoleClassifier

    classifier = CausalRoleClassifier()
    assert classifier is not None
    # Has the inner ChainOfThought predictor
    assert hasattr(classifier, "classify")
