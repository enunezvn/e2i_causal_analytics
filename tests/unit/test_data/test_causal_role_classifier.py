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
    # Issue #198: compile set extended from 12 to 21 examples to add
    # collider + instrument coverage and one explicit confounder
    # negative-direction discrimination exemplar
    # (baseline_severity_score_preindex, added on codex pass-5).
    #
    # Phase-4 S12 Option C recompile (2026-05-19): extended from 21 to
    # 33 to add 12 (T, Y)-explicit paired-fixture demos (6 features x 2
    # (treatment, outcome) variants each) that teach the classifier to
    # read `treatment=X; outcome=Y` fields in `dataset_context` as
    # first-class input. The 12 paired demos enable instrument-role
    # falsifiability (codex iter-2 redesign: same Z, different (T, Y) =>
    # different graph-theoretic-correct role) and are pinned by feature
    # name + (T, Y) + role quadruple in
    # `test_persisted_artifact_emits_role_conditional_on_treatment_outcome`.
    # Bar raised to 33 (not the plan's nominal 35) because the §3.5 binding
    # 12-demo paired-fixture design overrides §3.1's notional 14-demo split
    # — see Option C plan §3.1/§3.5 reconciliation note in the PR body.
    assert summary["n_examples"] >= 33, (
        f"Compile set too small: {summary['n_examples']}; need at least 33"
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
    # Codex pass-4 MED-1: require BOTH IV families in saved demos.
    # Pass-3's count-only check (>=2 instruments) could be satisfied by
    # urban_rural_code + geographic_region (both geography), silently
    # dropping the provider IV family from production training signal.
    geographic_iv_features = {"urban_rural_code", "geographic_region"}
    provider_iv_features = {
        "provider_preference_score",
        "index_provider_biologic_volume_prior_year",
    }
    new_instrument_features = geographic_iv_features | provider_iv_features

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
    geographic_iv_overlap = demo_instrument_features & geographic_iv_features
    provider_iv_overlap = demo_instrument_features & provider_iv_features

    assert len(collider_overlap) >= 2, (
        f"Persisted artifact demos carry only {len(collider_overlap)} of "
        f"the 4 issue-#198 collider features ({sorted(collider_overlap)}). "
        f"Expected >= 2 distinct collider features so the compiled "
        f"classifier has diverse training signal for the confounder-collider "
        f"pattern across both count and binary derivations. Recompile with "
        f"the default max_labeled_demos=24 (raised from 16 on codex pass-4)."
    )
    assert len(instrument_overlap) >= 2, (
        f"Persisted artifact demos carry only {len(instrument_overlap)} of "
        f"the 4 issue-#198 instrument features ({sorted(instrument_overlap)}). "
        f"Expected >= 2 distinct instrument features so the compiled "
        f"classifier has diverse training signal across both supply-side "
        f"geographic and preference/volume-based provider IV families."
    )
    # Codex pass-4 MED-1: family-level diversity. Require at least one
    # geographic IV AND at least one provider IV in the saved demos.
    # Without this, the count-only pin lets the compile pass while
    # silently dropping an entire IV family from production training.
    assert geographic_iv_overlap, (
        f"Persisted artifact demos carry NO geographic IV exemplars from "
        f"{sorted(geographic_iv_features)}; got instrument features "
        f"{sorted(demo_instrument_features)}. The compiled classifier "
        f"needs at least one Brookhart supply-side geographic IV demo to "
        f"teach the geographic-access IV pattern."
    )
    assert provider_iv_overlap, (
        f"Persisted artifact demos carry NO provider IV exemplars from "
        f"{sorted(provider_iv_features)}; got instrument features "
        f"{sorted(demo_instrument_features)}. The compiled classifier "
        f"needs at least one Brookhart-Schneeweiss provider IV demo "
        f"(preference-fraction or volume-based) to teach the preference-"
        f"based IV pattern as distinct from the geographic IV family. "
        f"Recompile with the default max_labeled_demos=24 (raised from "
        f"16 on codex pass-4 so all 20 labeled examples survive)."
    )


def test_compile_set_contains_explicit_severity_confounder_negative_exemplar():
    """Codex pass-5 MED-1 substantive fix: the prior pass-3 DummyLM
    negative-direction test was a wiring test only (scripted LM response,
    so it would pass even if the compile set were teaching the wrong
    discrimination). This test pins the substantive fix: an explicit
    POSITIVE training exemplar of "pre-index severity = confounder" was
    added to the compile set itself, so the LM learns the discrimination
    boundary from labeled data rather than only via the negative-by-
    absence of "no severity-as-collider" exemplars.

    Without this exemplar the compile set has 0 positive instances of
    "pre-index severity = confounder" — and 4 negative instances where
    severity is named as a PARENT of a collider V (in the confounder-
    collider M-structures). The risk: LM learns "severity appears in
    collider rationales -> label severity-shaped features as collider"
    even when severity is the feature being classified directly.
    """
    from src.data.causal_role_classifier import build_compile_set

    severity_confounder = next(
        (ex for ex in build_compile_set() if ex.feature_name == "baseline_severity_score_preindex"),
        None,
    )
    assert severity_confounder is not None, (
        "Compile set must contain the pre-index baseline severity confounder "
        "exemplar (codex pass-5 MED-1). Without it the LM has no positive "
        "example of 'pre-index severity = confounder' and may spuriously "
        "label legitimate severity confounders as colliders after seeing "
        "the 4 collider examples that all name severity as a parent of V."
    )
    assert severity_confounder.causal_role == "confounder", (
        f"baseline_severity_score_preindex must be labeled 'confounder' "
        f"(arrows OUT of severity to T and Y); got "
        f"{severity_confounder.causal_role!r}."
    )
    assert severity_confounder.recommended_remediation == "keep_with_caveat", (
        f"baseline_severity_score_preindex must have remediation "
        f"'keep_with_caveat' (condition on it to close the backdoor); got "
        f"{severity_confounder.recommended_remediation!r}."
    )
    # The rationale must explicitly name OUTGOING arrows so the LM learns
    # the discrimination boundary, not just the role label.
    mechanism = severity_confounder.mechanism.lower()
    assert "out of severity" in mechanism or "arrows out" in mechanism, (
        f"baseline_severity_score_preindex mechanism must explicitly say "
        f"arrows go OUT of severity (the discrimination boundary vs "
        f"collider). Got (truncated): {severity_confounder.mechanism[:200]!r}..."
    )


def test_dummy_lm_pure_severity_confounder_is_not_classified_as_collider():
    """WIRING test only (codex pass-5): scripts a DummyLM 'confounder'
    response and verifies the signature/output-parsing pipeline
    propagates it without the new collider vocabulary corrupting the
    confounder path. Substantive negative-direction discrimination is
    enforced by
    ``test_compile_set_contains_explicit_severity_confounder_negative_exemplar``
    above (which pins the positive training exemplar in the compile
    set).

    Kept as a wiring sanity check that the Literal-typed signature
    still accepts 'confounder' as a valid output and does not silently
    coerce it to one of the new roles.
    """
    import dspy
    from dspy.utils.dummies import DummyLM

    from src.data.causal_role_classifier import CausalRoleClassifier

    dummy = DummyLM(
        [
            {
                "reasoning": (
                    "Pre-index severity has arrowheads OUT to both T and Y; "
                    "arrows go OUT, not IN, so this is a confounder not a "
                    "collider."
                ),
                "causal_role": "confounder",
                "mechanism": "pre-index severity score; arrows out to T and Y",
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

    # Wiring check: signature must accept 'confounder' as a valid output
    # and not silently coerce it to a new role.
    assert prediction.causal_role == "confounder", (
        f"Wiring failure: DSPy signature did not propagate scripted "
        f"'confounder' role; got {prediction.causal_role!r}."
    )
    assert prediction.causal_role != "collider"


# --- Phase-4 S12 Option C recompile (2026-05-19): (T, Y)-explicit demos -----
#
# These tests pin the artifact-level forward contract of the Option C
# recompile (see `.claude/plans/option_c_dspy_recompile_for_s12_FINAL.md`):
#
# 1. `test_persisted_artifact_preserves_legacy_demo_roles` (§3.4) — pins all
#    21 pre-Option-C demos by (feature_name, causal_role) so the recompile
#    cannot silently shift legacy roles. This is the production-safety gate:
#    the sole production caller at adaptive_validity_check.py:892-893 emits
#    cohort-only dataset_context strings, so legacy role labels must be
#    preserved under cohort-only replay.
#
# 2. `test_persisted_artifact_emits_role_conditional_on_treatment_outcome`
#    (§3.5 Stage 1) — pins 12 quadruples (feature_name, treatment, outcome,
#    role) covering the 6 paired (T, Y) fixtures designed to be falsifiable:
#    the same feature with the same derivation_pseudocode is asserted to
#    classify into different causal roles under different (T, Y) variants.
#    Same-feature-different-role under different (T, Y) is the binding
#    behavioural signature that the instrument-role schema gap has closed.
#
# 3. `test_persisted_artifact_has_treatment_outcome_explicit_instruments`
#    (§3.3) — pins ≥2 distinct instrument features whose dataset_context
#    contains explicit `treatment=` markers, ensuring BootstrapFewShot
#    actually persists the (T, Y)-explicit demos rather than dropping the
#    new schema in favour of legacy-cohort-only retention.


# Legacy demo (feature_name -> expected causal_role) pin. These 21 features
# pre-date the Phase-4 S12 Option C recompile and must continue to classify
# unchanged in the persisted artifact under cohort-only dataset_context.
# Refer to `src/data/causal_role_classifier.py::build_compile_set()` for the
# canonical labels. Updating either side requires updating the other.
LEGACY_DEMO_EXPECTED_ROLES: dict[str, str] = {
    "disease_severity": "descendant",
    "engagement_score": "descendant",
    "days_on_therapy": "descendant",
    "medication_claim_count": "descendant",
    "journey_duration_days": "mediator",
    "journey_status": "descendant",
    "charlson_score": "descendant",
    "has_angioedema": "descendant",
    "atopy_score": "descendant",
    "prior_treatments": "confounder",
    "age_at_index": "ancestor",
    "insurance_product": "confounder",
    "baseline_severity_score_preindex": "confounder",
    "hospitalizations_total": "collider",
    "concomitant_steroid_burst_count_followup": "collider",
    "alive_at_180d_observation_window": "collider",
    "diagnostic_test_count_followup": "collider",
    "urban_rural_code": "instrument",
    "geographic_region": "instrument",
    "provider_preference_score": "instrument",
    "index_provider_biologic_volume_prior_year": "instrument",
}


# 12 paired-fixture quadruples (feature_name, treatment, outcome, role) from
# Option C plan §3.5. Two of the six features (concomitant_steroid_burst_-
# count_followup, provider_preference_score) appear in both the legacy
# compile set AND the paired-fixture set: the paired variants share the
# feature name but differ from legacy in dataset_context (legacy lacks the
# `treatment=` field; paired variants embed `treatment={T}; outcome={Y}`).
# Demo matching for this test uses the substring `treatment={T}; outcome={Y}`
# to disambiguate paired variants from legacy ones in the persisted demos.
EXPECTED_TREATMENT_OUTCOME_QUADRUPLES: list[tuple[str, str, str, str]] = [
    # Pair 1 — provider-volume IV vs care-quality confounder
    (
        "index_provider_omalizumab_volume_prior_year",
        "omalizumab_init",
        "remission_180d",
        "instrument",
    ),
    (
        "index_provider_omalizumab_volume_prior_year",
        "omalizumab_init",
        "hospitalization_180d",
        "confounder",
    ),
    # Pair 2 — post-T renal event: mediator vs collider
    (
        "acute_kidney_injury_event_count_followup",
        "ace_inhibitor_init",
        "cv_death_5y",
        "mediator",
    ),
    (
        "acute_kidney_injury_event_count_followup",
        "baseline_egfr_category",
        "ace_inhibitor_init",
        "collider",
    ),
    # Pair 3 — steroid burst: collider vs descendant
    (
        "concomitant_steroid_burst_count_followup",
        "biologic_init",
        "hospitalization_180d",
        "collider",
    ),
    (
        "concomitant_steroid_burst_count_followup",
        "steroid_burst_policy_indicator",
        "biologic_init",
        "descendant",
    ),
    # Pair 4 — Oncotype DX: confounder vs ancestor (d-separation assumption;
    # see Option C plan §3.5 Pair 4 + §9 — domain reviewer may swap if the
    # Oncotype ⊥ tumor_size | pre-diagnosis covariates assumption is disputed).
    (
        "baseline_oncotype_dx_recurrence_score",
        "cdk46i_init",
        "recurrence_5y",
        "confounder",
    ),
    (
        "baseline_oncotype_dx_recurrence_score",
        "tumor_size_at_diagnosis",
        "cdk46i_init",
        "ancestor",
    ),
    # Pair 5 — provider preference: instrument vs mediator on region path
    (
        "provider_preference_score",
        "biologic_init",
        "remission_180d",
        "instrument",
    ),
    (
        "provider_preference_score",
        "provider_geographic_region",
        "biologic_init",
        "mediator",
    ),
    # Pair 6 — prior treatment count: confounder vs mediator on duration path
    (
        "prior_treatment_count_preindex",
        "biologic_init",
        "remission_180d",
        "confounder",
    ),
    (
        "prior_treatment_count_preindex",
        "time_since_diagnosis_years",
        "biologic_init",
        "mediator",
    ),
]


def _load_artifact_demos() -> list[dict]:
    """Read the persisted classifier artifact and return its demo list.

    Raised separately so the asserts below produce focused failure
    messages rather than a tangle of `KeyError` traces.
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
    classify_predict = data.get("classify.predict") or {}
    demos = classify_predict.get("demos") or []
    assert demos, (
        f"Artifact at {artifact_path} has 0 demos under classify.predict; "
        f"recompile may have degraded to --no-lm or the optimizer dropped "
        f"every labeled example. Top-level keys: {list(data.keys())}."
    )
    return demos


def test_persisted_artifact_preserves_legacy_demo_roles():
    """Phase-4 S12 Option C (§3.4): all 21 pre-Option-C demos retain their
    labeled causal_role in the recompiled artifact under cohort-only
    dataset_context.

    Falsifiability: this is the production-safety gate. The sole production
    caller at `adaptive_validity_check.py:892-893` builds cohort-only
    `dataset_context` strings (no `treatment=` / `outcome=` fields), so the
    recompile must preserve the classifier's behavior on cohort-only inputs.
    The assertion has two parts:

    * **Retention**: every legacy feature_name must have at least one demo
      in the persisted artifact whose dataset_context does NOT contain
      `treatment=` (i.e., a cohort-only legacy variant — not a Phase-4
      paired-fixture variant that happens to share the feature_name).
    * **Role match**: every retained legacy demo's causal_role matches
      the labeled role in ``LEGACY_DEMO_EXPECTED_ROLES``.

    Codex iter-1 redesign of the original v1 "live LLM 21/21" gate: this
    test is artifact-level and deterministic (no LM call), aligning with
    repo test practice (DummyLM / no live LM in CI). See Option C plan
    §3.4 for the rationale + falsifiability narrative.
    """
    demos = _load_artifact_demos()

    # Find LEGACY (cohort-only) variant per feature_name. Filter out Phase-4
    # paired-fixture variants (which carry `treatment=` in dataset_context).
    legacy_demos_by_feature: dict[str, list[dict]] = {}
    for d in demos:
        feature_name = d.get("feature_name")
        dataset_context = d.get("dataset_context") or ""
        if not feature_name or feature_name not in LEGACY_DEMO_EXPECTED_ROLES:
            continue
        if "treatment=" in dataset_context:
            continue
        legacy_demos_by_feature.setdefault(feature_name, []).append(d)

    missing_legacy = set(LEGACY_DEMO_EXPECTED_ROLES) - set(legacy_demos_by_feature)
    assert not missing_legacy, (
        f"Recompiled artifact dropped legacy demos: {sorted(missing_legacy)}. "
        f"The Phase-4 S12 Option C recompile must preserve all 21 pre-Option-C "
        f"demos under cohort-only dataset_context (the sole production caller "
        f"at adaptive_validity_check.py:892-893 emits cohort-only contexts). "
        f"Investigate BootstrapFewShot's `random.sample` step at "
        f"max_labeled_demos cap — if some labeled demos were dropped, raise "
        f"DEFAULT_MAX_LABELED_DEMOS in scripts/compile_causal_role_classifier.py."
    )

    role_mismatches: list[tuple[str, str, str]] = []
    for feature_name, expected_role in LEGACY_DEMO_EXPECTED_ROLES.items():
        for demo in legacy_demos_by_feature[feature_name]:
            actual_role = demo.get("causal_role")
            if actual_role != expected_role:
                role_mismatches.append((feature_name, expected_role, actual_role or "?"))

    assert not role_mismatches, (
        f"Recompiled artifact reassigned legacy demo roles "
        f"(feature, expected, actual): {role_mismatches}. The Phase-4 S12 "
        f"Option C recompile must NOT shift legacy roles under cohort-only "
        f"replay. To recover: `git checkout artifacts/dspy/causal_role_classifier.json` "
        f"and inspect the new (T, Y)-explicit demos for context-leak into the "
        f"legacy LLM reasoning (Option C plan §5 risk-register row 1)."
    )


def test_persisted_artifact_emits_role_conditional_on_treatment_outcome():
    """Phase-4 S12 Option C (§3.5 Stage 1): 12 paired-fixture quadruples
    `(feature_name, treatment, outcome, role)` are present in the artifact.

    This is the binding falsifiability for the instrument-role schema gap
    closure: the same feature with the same `derivation_pseudocode` must
    classify into a different causal role under different (T, Y) variants.
    Asserting all 12 quadruples (not 6 feature names) closes the
    optimizer-drop gap (codex iter-2 HIGH-F1): if BootstrapFewShot drops
    one variant of a paired fixture, this test trips.

    Demo matching: dataset_context must contain the literal substring
    `treatment={T}; outcome={Y}` (the schema the new compile demos use)
    AND causal_role must equal the expected role. The Pair 4 (Oncotype DX)
    ancestor assumption is flagged for expert review in Option C plan
    §3.5 + §9; if the d-separation assumption is disputed the pair is
    swappable without affecting the other 5 pairs.
    """
    demos = _load_artifact_demos()

    missing_quadruples: list[tuple[str, str, str, str]] = []
    for feature_name, treatment, outcome, expected_role in EXPECTED_TREATMENT_OUTCOME_QUADRUPLES:
        marker = f"treatment={treatment}; outcome={outcome}"
        match = next(
            (
                d
                for d in demos
                if d.get("feature_name") == feature_name
                and marker in (d.get("dataset_context") or "")
                and d.get("causal_role") == expected_role
            ),
            None,
        )
        if match is None:
            missing_quadruples.append((feature_name, treatment, outcome, expected_role))

    assert not missing_quadruples, (
        f"Phase-4 S12 Option C: the following (feature, T, Y, role) "
        f"quadruples are NOT present in the persisted artifact: "
        f"{missing_quadruples}. The recompile must persist all 12 paired-"
        f"fixture demos so the classifier learns that role label depends "
        f"on (T, Y), not on feature_name alone. Recovery: `git checkout "
        f"artifacts/dspy/causal_role_classifier.json`, verify the 12 paired "
        f"demos in `build_compile_set()` carry the exact "
        f"`treatment={{T}}; outcome={{Y}}` markers in dataset_context, then "
        f"recompile with --force if backlog gate blocks (see Option C plan "
        f"§5 risk-register row 2)."
    )


def test_persisted_artifact_has_treatment_outcome_explicit_instruments():
    """Phase-4 S12 Option C (§3.3 strengthening): the recompile actually
    persists instrument demos with `treatment=` markers — not just retains
    the 4 legacy cohort-only instruments.

    Falsifiability: if BootstrapFewShot drops all new (T, Y)-explicit
    instrument demos (e.g., metric fails on the new schema), the artifact's
    instrument demos collapse to the 4 legacy cohort-only ones. This test
    catches that regression by requiring ≥2 distinct instrument
    feature_names whose persisted demo dataset_context contains
    `treatment=`. Two such features are guaranteed by §3.5 Pair 1
    (index_provider_omalizumab_volume_prior_year, instrument arm) and
    Pair 5 (provider_preference_score, instrument arm under
    biologic_init/remission_180d).
    """
    demos = _load_artifact_demos()
    treatment_explicit_instruments = {
        d["feature_name"]
        for d in demos
        if d.get("causal_role") == "instrument"
        and d.get("feature_name")
        and "treatment=" in (d.get("dataset_context") or "")
    }
    assert len(treatment_explicit_instruments) >= 2, (
        f"Persisted artifact has only {len(treatment_explicit_instruments)} "
        f"distinct (T, Y)-explicit instrument features "
        f"(got: {sorted(treatment_explicit_instruments)}). Phase-4 S12 "
        f"Option C requires >=2 so the classifier carries training signal "
        f"for instrument-role discrimination under explicit (T, Y) input. "
        f"The plan binds two such features (Pair 1 + Pair 5 instrument "
        f"arms): if both are missing, BootstrapFewShot may have failed on "
        f"the new (T, Y) schema. Recompile with the §3.5 demos verified "
        f"present in build_compile_set()."
    )
