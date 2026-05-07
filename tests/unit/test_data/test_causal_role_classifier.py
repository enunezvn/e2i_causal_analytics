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


# --- Codex audit follow-ups (Layer 4 — item F) ------------------------------


def test_compile_set_role_coverage_is_explicit_subset():
    """The compile set covers 4 of 6 declared CausalRole values (ancestor,
    confounder, mediator, descendant). ``collider`` and ``instrument`` are
    deliberately unrepresented pending domain-expert example construction;
    the gap is documented in the module docstring and tracked in backlog
    item #11. This test pins the current state so an unsigned extension to
    those roles (which would change the LM training signal) fires the test
    and prompts a deliberate review.
    """
    from typing import get_args

    from src.data.causal_role_classifier import CausalRole, get_compile_set_summary

    summary = get_compile_set_summary()
    declared_roles = set(get_args(CausalRole))
    covered_roles = set(summary["role_distribution"].keys())

    # Every covered role must be a valid CausalRole Literal value (no typos).
    drift = covered_roles - declared_roles
    assert drift == set(), (
        f"Compile set has roles not in the CausalRole Literal: {drift}. "
        f"Declared: {declared_roles}; covered: {covered_roles}."
    )

    # Pin the current 4-role coverage. If extending to collider/instrument,
    # this test must be updated in the same commit so the docstring claim
    # ('4 of 6') stays true.
    expected_covered = {"ancestor", "confounder", "mediator", "descendant"}
    assert covered_roles == expected_covered, (
        f"Compile-set role coverage changed: covered={covered_roles}, "
        f"expected={expected_covered}. If this is intentional, update the "
        f"module docstring + this assertion together so the documented "
        f"coverage matches the data."
    )


def test_compile_set_remediation_values_are_valid_literals():
    """Every example's recommended_remediation must be one of the four declared
    Remediation Literal values. Otherwise the LLM's compiled output schema
    has fewer enforced labels than designed.
    """
    from typing import get_args

    from src.data.causal_role_classifier import Remediation, build_compile_set

    valid_remediations = set(get_args(Remediation))
    for i, ex in enumerate(build_compile_set()):
        assert ex.recommended_remediation in valid_remediations, (
            f"Example {i} ({ex.feature_name!r}) has remediation "
            f"{ex.recommended_remediation!r} not in {valid_remediations}."
        )


def test_compile_set_role_remediation_pairs_are_consistent():
    """Role/remediation pairs should respect causal semantics:
    - ``descendant`` and ``mediator`` should NOT be ``keep_with_caveat``
      (they require ``drop`` or ``window`` because their values reflect the
      target).
    - ``ancestor`` and ``confounder`` should NOT be ``drop`` (they're
      legitimate pre-prediction-time signal — dropping discards real signal).
    - ``collider`` and ``instrument`` are not currently represented; this
      assertion runs only on the covered subset.
    """
    from src.data.causal_role_classifier import build_compile_set

    POST_INDEX_ROLES = {"descendant", "mediator"}
    PRE_INDEX_ROLES = {"ancestor", "confounder"}
    POST_INDEX_REMEDIATIONS = {"drop", "window", "transform"}
    PRE_INDEX_REMEDIATIONS = {"keep_with_caveat", "transform"}

    for ex in build_compile_set():
        if ex.causal_role in POST_INDEX_ROLES:
            assert ex.recommended_remediation in POST_INDEX_REMEDIATIONS, (
                f"{ex.feature_name!r} role={ex.causal_role!r} should not be "
                f"remediated as {ex.recommended_remediation!r}; post-index roles "
                f"need one of {POST_INDEX_REMEDIATIONS}."
            )
        elif ex.causal_role in PRE_INDEX_ROLES:
            assert ex.recommended_remediation in PRE_INDEX_REMEDIATIONS, (
                f"{ex.feature_name!r} role={ex.causal_role!r} should not be "
                f"remediated as {ex.recommended_remediation!r}; pre-index roles "
                f"need one of {PRE_INDEX_REMEDIATIONS}."
            )
