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
    # Issue #198: compile set extended from 12 → 20 examples to add collider
    # and instrument coverage. Bar raised to 20 so a regression that drops
    # examples (e.g., a refactor that splits the function) fires the test.
    assert summary["n_examples"] >= 20, (
        f"Compile set too small: {summary['n_examples']}; need at least 20"
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


def test_compile_set_role_coverage_is_full_six_roles():
    """The compile set covers ALL 6 declared CausalRole values.

    Issue #198 extended the compile set from 4 of 6 roles (ancestor,
    confounder, mediator, descendant) to all 6 by adding 4 collider +
    4 instrument labeled exemplars from a domain-expert pass over the
    CSU + Optum manifests. This test pins the full-coverage state so a
    regression that drops any role (e.g., a refactor that removes a
    branch of examples) fires the test.
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

    # Pin full 6-role coverage. The set equality also catches typos
    # (e.g., 'collidre') in the example data.
    expected_covered = {
        "ancestor",
        "confounder",
        "mediator",
        "descendant",
        "collider",
        "instrument",
    }
    assert covered_roles == expected_covered, (
        f"Compile-set role coverage changed: covered={covered_roles}, "
        f"expected={expected_covered}. If this is intentional, update the "
        f"module docstring + this assertion together so the documented "
        f"coverage matches the data."
    )


def test_compile_set_has_at_least_four_collider_examples():
    """Issue #198: pin the minimum collider exemplar count at 4 so a
    refactor that silently drops collider examples fires this test.

    All 4 collider examples are confounder-collider M-structures (per
    Greenland-Pearl-Robins 1999) — the dominant collider failure mode
    in observational pharmacoepi. They differ in derivation MECHANISM:
    count of utilization (hospitalizations_total), count of medication
    (concomitant_steroid_burst_count_followup), count of workup
    (diagnostic_test_count_followup), and binary sample-inclusion gate
    (alive_at_180d_observation_window).

    Note: discontinuation_flag / discontinued_180d / persistent_at_180d
    were considered but rejected on codex pass-1 review because their
    derivation ``T AND (post-T event)`` makes them DESCENDANTS, not
    colliders — the second "arrow" is downstream of T rather than an
    independent cause.
    """
    from src.data.causal_role_classifier import get_compile_set_summary

    summary = get_compile_set_summary()
    n_collider = summary["role_distribution"].get("collider", 0)
    assert n_collider >= 4, (
        f"Compile set must have >= 4 collider exemplars (issue #198); "
        f"got {n_collider}. Role distribution: {summary['role_distribution']}."
    )


def test_compile_set_has_at_least_four_instrument_examples():
    """Issue #198: pin the minimum instrument exemplar count at 4 so a
    refactor that silently drops instrument examples fires this test.

    The 4 instrument examples are canonical pharmacoepi supply-side IVs
    (Brookhart et al. 2006 style): urban_rural_code, geographic_region,
    zip3, and plan_type. Each rationale names the exclusion-restriction
    assumption explicitly so the LM learns to flag violations.
    """
    from src.data.causal_role_classifier import get_compile_set_summary

    summary = get_compile_set_summary()
    n_instrument = summary["role_distribution"].get("instrument", 0)
    assert n_instrument >= 4, (
        f"Compile set must have >= 4 instrument exemplars (issue #198); "
        f"got {n_instrument}. Role distribution: {summary['role_distribution']}."
    )


def test_compile_set_collider_examples_have_required_feature_names():
    """Pin the 4 specific collider features so a silent renaming or
    swap is caught. The collider exemplars are confounder-collider
    M-structures varying in derivation mechanism (count vs binary) —
    load-bearing for Layer 4 training signal.
    """
    from src.data.causal_role_classifier import build_compile_set

    collider_features = {
        ex.feature_name for ex in build_compile_set() if ex.causal_role == "collider"
    }
    expected = {
        "hospitalizations_total",
        "concomitant_steroid_burst_count_followup",
        "alive_at_180d_observation_window",
        "diagnostic_test_count_followup",
    }
    missing = expected - collider_features
    assert not missing, (
        f"Compile set is missing pinned collider exemplars: {missing}. Got: {collider_features}."
    )


def test_compile_set_instrument_examples_have_required_feature_names():
    """Pin the 4 specific instrument features. The LM training signal
    for the IV pattern depends on these specific pharmacoepi examples
    spanning two distinct IV families: supply-side geographic
    (urban_rural_code, geographic_region) and preference/volume-based
    provider IVs (provider_preference_score,
    index_provider_biologic_volume_prior_year).

    Note: plan_type was considered but rejected on codex pass-2 review
    because as an enrollment-time payer feature it would duplicate the
    `insurance_product` confounder exemplar (creating contradictory
    confounder-vs-IV training signal on the same feature family).
    """
    from src.data.causal_role_classifier import build_compile_set

    instrument_features = {
        ex.feature_name for ex in build_compile_set() if ex.causal_role == "instrument"
    }
    expected = {
        "urban_rural_code",
        "geographic_region",
        "provider_preference_score",
        "index_provider_biologic_volume_prior_year",
    }
    missing = expected - instrument_features
    assert not missing, (
        f"Compile set is missing pinned instrument exemplars: {missing}. "
        f"Got: {instrument_features}."
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
    - ``collider`` (issue #198) must be ``drop``: conditioning on a collider
      induces non-causal selection bias on the T→Y relationship; the only
      safe remediation is to remove from features (windowing doesn't help
      because the two-arrow-in DAG structure still holds).
    - ``instrument`` (issue #198) must be ``keep_with_caveat``: instruments
      are legitimate pre-prediction-time features that aid causal
      identification under the exclusion restriction. Dropping reflexively
      discards causal-identification value. ``keep_with_caveat`` preserves
      the feature while documenting the exclusion-restriction assumption.
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
        elif ex.causal_role == "collider":
            # Colliders are NEVER safe to condition on — drop is the only
            # remediation that closes the selection-bias backdoor. Windowing
            # doesn't help because the two-arrow-in structure persists.
            assert ex.recommended_remediation == "drop", (
                f"{ex.feature_name!r} role='collider' must be remediated as "
                f"'drop'; got {ex.recommended_remediation!r}. Conditioning on "
                f"a collider induces non-causal selection bias."
            )
        elif ex.causal_role == "instrument":
            # Instruments are LEGITIMATE features under the exclusion
            # restriction. Dropping would discard causal-identification
            # value. keep_with_caveat documents the assumption.
            assert ex.recommended_remediation == "keep_with_caveat", (
                f"{ex.feature_name!r} role='instrument' must be remediated as "
                f"'keep_with_caveat'; got {ex.recommended_remediation!r}. "
                f"Instruments preserve causal-identification value under the "
                f"exclusion restriction."
            )


# --- Issue #198: persisted artifact + bi-directional classifier tests --------


def test_persisted_artifact_contains_collider_and_instrument_demos():
    """The recompiled artifact must contain BOTH a collider exemplar AND an
    instrument exemplar in the BootstrapFewShot-curated demos.

    Issue #198 acceptance: a downstream caller loading the artifact gets a
    classifier that has SEEN both new roles in its few-shot examples. If a
    future compile run accidentally drops EITHER role from the labeled
    demo cap (e.g., ``max_labeled_demos`` lowered below 8), this test fires
    so the artifact regression is caught at unit-test time rather than at
    Layer 4 deploy time.

    Codex pass-1 LOW-1 tightening: the prior version used ``or`` (either
    role suffices) which would silently let half of #198's intent
    disappear. The set-subset assertion below requires both roles.
    """
    import json
    from pathlib import Path

    artifact_path = (
        Path(__file__).resolve().parents[3] / "artifacts" / "dspy" / "causal_role_classifier.json"
    )
    assert artifact_path.exists(), (
        f"Artifact missing at {artifact_path}. Recompile via "
        f"`python scripts/compile_causal_role_classifier.py`."
    )
    data = json.loads(artifact_path.read_text())
    # DSPy persists ChainOfThought-wrapped Predict under ``classify.predict``.
    classify_predict = data.get("classify.predict")
    assert classify_predict is not None, (
        f"Artifact at {artifact_path} has no 'classify.predict' key. "
        f"Top-level keys: {list(data.keys())}."
    )
    demos = classify_predict.get("demos") or []
    assert demos, (
        f"Artifact has 0 demos under classify.predict. Recompile run likely "
        f"degraded to --no-lm. Top-level: {list(data.keys())}."
    )
    demo_roles = {d.get("causal_role") for d in demos if d.get("causal_role")}
    required_roles = {"collider", "instrument"}
    missing_roles = required_roles - demo_roles
    assert not missing_roles, (
        f"Recompiled artifact is missing role(s) {sorted(missing_roles)} "
        f"from its demo causal_role set: got {sorted(demo_roles)}. The "
        f"extended compile set (issue #198) must surface BOTH 'collider' "
        f"AND 'instrument' in the BootstrapFewShot-curated demos "
        f"(max_labeled_demos=8 default). If the role got dropped, the "
        f"compile run likely shrank max_labeled_demos below 8 or the "
        f"BootstrapFewShot teacher rejected the demos via the exact-match "
        f"metric — investigate before re-pinning the artifact."
    )


def test_dummy_lm_classifier_emits_new_roles_on_new_examples():
    """End-to-end: when fed one of the issue-#198 collider or instrument
    feature's derivation, the classifier emits the labeled role.

    Uses ``dspy.utils.dummies.DummyLM`` to stub the LM with a scripted
    response — verifies the WIRING (signature → output parsing → role
    propagation) end-to-end without spending an LLM token. The compiled
    classifier's persisted demos remain present in the prompt; the dummy
    LM's response is the only thing controlled.

    Tests the POSITIVE direction of the discrimination required by issue
    #198: classifier must be able to emit ``collider`` and ``instrument``
    as output roles (not just ancestor/confounder/mediator/descendant).
    """
    import dspy
    from dspy.utils.dummies import DummyLM

    from src.data.causal_role_classifier import CausalRoleClassifier

    # DummyLM scripts the model's response field-by-field. ChainOfThought
    # adds an implicit ``reasoning`` field before the signature outputs.
    dummy = DummyLM(
        [
            {
                "reasoning": "Two-arrow-in DAG structure (treatment + response).",
                "causal_role": "collider",
                "mechanism": "two arrows in from treatment and response",
                "recommended_remediation": "drop",
            },
            {
                "reasoning": "Supply-side geographic IV per Brookhart 2006.",
                "causal_role": "instrument",
                "mechanism": "Z->T via specialist density; Z->Y only through T",
                "recommended_remediation": "keep_with_caveat",
            },
        ]
    )
    with dspy.context(lm=dummy):
        classifier = CausalRoleClassifier()
        collider_prediction = classifier(
            feature_name="hospitalizations_total",
            derivation_pseudocode=(
                "count(encounter_events in [journey_start, journey_end]); "
                "both pre-index severity and post-T AE feed the count"
            ),
            dataset_context="Optum; target=initiated_biologic_180d",
        )
        instrument_prediction = classifier(
            feature_name="urban_rural_code",
            derivation_pseudocode="RUCA(zip3); static at enrollment",
            dataset_context="Optum; target=initiated_biologic_180d",
        )

    assert collider_prediction.causal_role == "collider", (
        f"Expected classifier to emit 'collider' on hospitalizations_total "
        f"derivation; got {collider_prediction.causal_role!r}. The DSPy "
        f"signature must accept 'collider' as a valid output role (issue #198)."
    )
    assert collider_prediction.recommended_remediation == "drop", (
        f"Expected remediation 'drop' for collider; "
        f"got {collider_prediction.recommended_remediation!r}."
    )
    assert instrument_prediction.causal_role == "instrument", (
        f"Expected classifier to emit 'instrument' on urban_rural_code "
        f"derivation; got {instrument_prediction.causal_role!r}. The DSPy "
        f"signature must accept 'instrument' as a valid output role (issue #198)."
    )
    assert instrument_prediction.recommended_remediation == "keep_with_caveat", (
        f"Expected remediation 'keep_with_caveat' for instrument; "
        f"got {instrument_prediction.recommended_remediation!r}."
    )


def test_dummy_lm_classifier_does_not_spuriously_emit_new_roles_on_legacy_examples():
    """End-to-end NEGATIVE direction: when the DummyLM scripts a confounder
    or mediator response, the classifier preserves it — the new
    collider/instrument vocabulary does not corrupt the existing roles.

    This is the discrimination side of issue #198 acceptance (codex review
    item 6e): adding collider/instrument to the compile set MUST NOT cause
    the classifier to spuriously re-label legitimate confounder or mediator
    examples as colliders/instruments. The signature is a Literal type so
    output validation catches typos, but the WIRING-level test confirms the
    end-to-end pipeline preserves whatever role the LM emits.
    """
    import dspy
    from dspy.utils.dummies import DummyLM

    from src.data.causal_role_classifier import CausalRoleClassifier

    dummy = DummyLM(
        [
            # Insurance product — labeled confounder in the compile set.
            {
                "reasoning": "Set at enrollment before treatment decision.",
                "causal_role": "confounder",
                "mechanism": "set at enrollment; influences access and decision",
                "recommended_remediation": "keep_with_caveat",
            },
            # journey_duration_days — labeled mediator in the compile set.
            {
                "reasoning": "On the path from treatment to outcome.",
                "causal_role": "mediator",
                "mechanism": "aggregate over events including post-treatment",
                "recommended_remediation": "window",
            },
        ]
    )
    with dspy.context(lm=dummy):
        classifier = CausalRoleClassifier()
        confounder_pred = classifier(
            feature_name="insurance_product",
            derivation_pseudocode="demo.insurance_product (one row per patient)",
            dataset_context="CSU/Optum; target=treatment_initiated",
        )
        mediator_pred = classifier(
            feature_name="journey_duration_days",
            derivation_pseudocode="(journey_end - journey_start).days; end unfiltered",
            dataset_context="CSU; target=treatment_initiated; anchor=index_date",
        )

    # The classifier must NOT spuriously re-label confounder as collider or
    # mediator as instrument just because the new vocabulary is present.
    assert confounder_pred.causal_role == "confounder", (
        f"insurance_product (a labeled confounder) was re-labeled as "
        f"{confounder_pred.causal_role!r}. Adding collider/instrument to the "
        f"compile set must not cause spurious role drift on legacy examples."
    )
    assert confounder_pred.causal_role not in {"collider", "instrument"}, (
        f"Negative-direction check: confounder must NOT be mis-emitted as "
        f"one of the new roles; got {confounder_pred.causal_role!r}."
    )
    assert mediator_pred.causal_role == "mediator", (
        f"journey_duration_days (a labeled mediator) was re-labeled as "
        f"{mediator_pred.causal_role!r}."
    )
    assert mediator_pred.causal_role not in {"collider", "instrument"}, (
        f"Negative-direction check: mediator must NOT be mis-emitted as "
        f"one of the new roles; got {mediator_pred.causal_role!r}."
    )


def test_persisted_artifact_demos_carry_diverse_new_role_features():
    """Codex pass-3 MED-2 strengthening: pass-1 added a weak "at least one
    new feature" check; pass-3 noted that with 4 collider + 4 instrument
    exemplars in the compile set, the persisted artifact should carry
    multiple new-role features per role — not just the single feature
    picked by the BootstrapFewShot's first-bootstrap-attempt run.

    Requires at least 2 collider FEATURES AND at least 2 instrument
    FEATURES from the issue-#198 set to be present in the saved demos.
    This catches a stale compile run where only one feature per new
    role made it through the teacher metric. The cap is set well below
    the maximum (4 per role) so the test stays robust to BootstrapFewShot's
    teacher-rejection variance across runs.

    Together with ``test_persisted_artifact_contains_collider_and_instrument_demos``
    (which pins role-set membership), this test pins the diversity of
    new-role TRAINING SIGNAL the compiled classifier carries into
    production.
    """
    import json
    from pathlib import Path

    artifact_path = (
        Path(__file__).resolve().parents[3] / "artifacts" / "dspy" / "causal_role_classifier.json"
    )
    assert artifact_path.exists(), f"Artifact missing at {artifact_path}."
    data = json.loads(artifact_path.read_text())
    demos = data.get("classify.predict", {}).get("demos") or []

    new_collider_features = {
        "hospitalizations_total",
        "concomitant_steroid_burst_count_followup",
        "alive_at_180d_observation_window",
        "diagnostic_test_count_followup",
    }
    new_instrument_features = {
        "urban_rural_code",
        "geographic_region",
        "provider_preference_score",
        "index_provider_biologic_volume_prior_year",
    }

    demo_collider_features = {
        d["feature_name"]
        for d in demos
        if d.get("causal_role") == "collider" and d.get("feature_name")
    }
    demo_instrument_features = {
        d["feature_name"]
        for d in demos
        if d.get("causal_role") == "instrument" and d.get("feature_name")
    }

    collider_overlap = demo_collider_features & new_collider_features
    instrument_overlap = demo_instrument_features & new_instrument_features

    assert len(collider_overlap) >= 2, (
        f"Persisted artifact demos carry only {len(collider_overlap)} of "
        f"the 4 issue-#198 collider features ({sorted(collider_overlap)}). "
        f"Expected >= 2 distinct collider features so the compiled "
        f"classifier has diverse training signal for the confounder-collider "
        f"pattern across both count and binary derivations. Recompile with "
        f"the default max_labeled_demos=16 (raised from 8 on codex pass-3)."
    )
    assert len(instrument_overlap) >= 2, (
        f"Persisted artifact demos carry only {len(instrument_overlap)} of "
        f"the 4 issue-#198 instrument features ({sorted(instrument_overlap)}). "
        f"Expected >= 2 distinct instrument features so the compiled "
        f"classifier has diverse training signal across both supply-side "
        f"geographic and preference/volume-based provider IV families."
    )


def test_dummy_lm_pure_severity_confounder_is_not_classified_as_collider():
    """Codex pass-3 (e) negative-direction strengthening: a pure
    pre-index severity confounder must NOT be spuriously emitted as
    collider just because the widened compile-set framing teaches the
    LM about confounder-collider M-structures.

    A pure pre-index severity score has arrowheads into BOTH T (via
    prescriber decision) and Y (via uncontrolled disease) but is NOT
    itself a collider — it's a CONFOUNDER. The arrowheads go OUT of
    severity into both T and Y, not into severity from T. The
    discrimination boundary the compile set must teach is:

    - confounder: severity -> T AND severity -> Y (arrows OUT)
    - collider: T -> V AND severity -> V (arrows IN, V at the bottom)

    This test uses DummyLM to script a "confounder" response on a pure
    severity feature and verifies the WIRING correctly propagates it
    without the new collider vocabulary corrupting the legacy
    confounder path.
    """
    import dspy
    from dspy.utils.dummies import DummyLM

    from src.data.causal_role_classifier import CausalRoleClassifier

    dummy = DummyLM(
        [
            {
                "reasoning": (
                    "Pre-index severity has arrowheads OUT to both T and Y "
                    "(severity -> T via prescriber decision; severity -> Y "
                    "via disease activity); arrows go OUT, not IN, so this "
                    "is a confounder not a collider."
                ),
                "causal_role": "confounder",
                "mechanism": ("pre-index severity score; arrows out to T and Y"),
                "recommended_remediation": "keep_with_caveat",
            },
        ]
    )
    with dspy.context(lm=dummy):
        classifier = CausalRoleClassifier()
        prediction = classifier(
            feature_name="baseline_severity_score",
            derivation_pseudocode=(
                "weighted sum of pre-index dx + lab abnormalities "
                "WHERE event_date < index_date - 30d"
            ),
            dataset_context=(
                "Optum claims; target=initiated_biologic_180d; anchor=index_date; PRE-INDEX ONLY"
            ),
        )

    assert prediction.causal_role == "confounder", (
        f"Pure pre-index severity confounder was emitted as "
        f"{prediction.causal_role!r}. The widened confounder-collider "
        f"framing in the compile set MUST NOT cause a severity-only "
        f"feature (arrows OUT of severity) to be re-labeled as collider "
        f"(arrows IN to V). If the LM gets this wrong, the discrimination "
        f"boundary the compile set is teaching is too loose — the "
        f"compile-set rationales should be tightened so the LM learns "
        f"that 'severity is a confounder when its arrows go OUT, a "
        f"parent of a collider when one of its arrows goes INTO a "
        f"separate variable V'."
    )
    assert prediction.causal_role != "collider", (
        f"Pure pre-index severity must not be classified as collider. "
        f"Got {prediction.causal_role!r}. Confounder-collider M-structures "
        f"in the issue-#198 compile set are colliders BECAUSE the COUNT/"
        f"BINARY variable V has two arrowheads in — not because severity "
        f"itself has any arrowhead in."
    )
