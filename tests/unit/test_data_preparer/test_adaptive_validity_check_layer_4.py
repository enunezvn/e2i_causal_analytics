"""Tests for Phase 2.9 Stage 3 — Layer 4 LLM verdict wiring (issue #193).

These tests pin the end-to-end ``adaptive_verdicts`` flow when an LLM
endpoint is configured AND the compiled CausalRoleClassifier artifact at
``artifacts/dspy/causal_role_classifier.json`` is present. CI runs without
API keys; the deterministic-LLM tests use ``dspy.utils.dummies.DummyLM`` so
the LM endpoint is stubbed but the wiring is exercised end-to-end.

What's pinned:

1. The persisted compile-set JSON loads cleanly via
   ``causal_role_classifier_loader.load_compiled_classifier``.
2. With a stubbed LM, ``classify_feature`` returns an ``LLMVerdict`` with a
   role / mechanism / remediation populated from the LM output (not from
   the bootstrapped demos).
3. ``_compose_legacy_verdict`` accepts ``llm_verdict=...`` and the resulting
   verdict carries ``decided_by="llm"`` + ``layer="4"`` in the legacy schema.
4. The full ``adaptive_validity_check`` orchestrator emits
   ``decided_by="llm"`` on a CSU manifest feature when the LLM verdict path
   is active.
5. The loader gracefully returns ``None`` when no LM is configured (the
   developer-laptop / CI-without-key path).
6. Malformed LLM output is sanitised — the voter falls through to the
   non-LLM precedence path rather than crashing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import dspy
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "dspy" / "causal_role_classifier.json"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def reset_dspy_lm() -> Any:
    """Reset dspy.settings.lm after the test so tests don't leak LM state.

    yields nothing; the cleanup happens at teardown.
    """
    prior_lm = getattr(dspy.settings, "lm", None)
    yield
    try:
        if prior_lm is None:
            dspy.settings.configure(lm=None)
        else:
            dspy.configure(lm=prior_lm)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _enable_layer4_llm_decides(monkeypatch):
    """This file pins the Layer-4 LLM-*decides* path (decided_by="llm", the #212
    cap, layer="4"). Plan v4 Phase 1 demoted the LLM to audit-only by DEFAULT
    (ensemble_voter._llm_decides_enabled), so enable the decides flag here — the
    mechanism is preserved behind it for the Phase-3 ramp / back-compat. NOTE:
    the node-level test additionally sets ``adaptive_layer4_enabled=True`` in its
    state to invoke the LLM CALL (the env flag only governs the voter's
    decide-vs-audit behaviour, not whether the node calls the classifier).
    """
    monkeypatch.setenv("ADAPTIVE_LAYER4_LLM_DECIDES", "1")


def _stub_dspy_lm_with_role(role: str, remediation: str = "keep_with_caveat") -> dspy.LM:
    """Return a DummyLM that always answers the CausalRoleSignature with
    ``role`` + ``remediation``.

    DummyLM accepts a list of answer dicts — each dict represents one
    LM call's structured output. We supply a list with one entry that
    contains every output-field name the signature declares (plus the
    hidden ``reasoning`` field that ``ChainOfThought`` injects). DummyLM
    cycles through the list, so a single-entry list is fine for any
    number of calls in the test.
    """
    from dspy.utils.dummies import DummyLM

    return DummyLM(
        [
            {
                "reasoning": "test stub reasoning",
                "causal_role": role,
                "mechanism": f"test stub mechanism for role={role}",
                "recommended_remediation": remediation,
            }
        ]
    )


# --------------------------------------------------------------------------
# Test 1 — persisted artifact loads
# --------------------------------------------------------------------------


def test_persisted_compiled_classifier_loads_from_default_path() -> None:
    """The committed ``artifacts/dspy/causal_role_classifier.json`` loads.

    Acceptance criterion (issue #193 #1): "Persisted compiled program file
    checked into ``artifacts/`` OR generated reproducibly by a script."
    """
    from src.data.causal_role_classifier_loader import (
        DEFAULT_ARTIFACT_PATH,
        load_compiled_classifier,
    )

    assert DEFAULT_ARTIFACT_PATH.exists(), (
        f"Persisted compiled classifier missing at {DEFAULT_ARTIFACT_PATH}; "
        f"run `python scripts/compile_causal_role_classifier.py` to regenerate."
    )

    classifier = load_compiled_classifier()
    assert classifier is not None
    # Inspect the inner ChainOfThought to confirm BootstrapFewShot
    # populated demos (otherwise we shipped an un-compiled program).
    demos = classifier.classify.predict.demos
    assert len(demos) >= 1, (
        f"Compiled classifier has no demos; expected >= 1 from BootstrapFewShot. "
        f"Path: {DEFAULT_ARTIFACT_PATH}"
    )


def test_persisted_artifact_metadata_pins_python_and_dspy_versions() -> None:
    """The persisted JSON declares its toolchain.

    Pin-test so a future DSPy upgrade that changes the artifact schema is
    visible: the JSON's ``metadata`` block must record python+dspy versions
    so a downstream operator can correlate "this artifact came from"
    decisions with the toolchain.
    """
    with DEFAULT_ARTIFACT_PATH.open() as fh:
        artifact = json.load(fh)
    assert "metadata" in artifact
    deps = artifact["metadata"].get("dependency_versions", {})
    assert "python" in deps
    assert "dspy" in deps


# --------------------------------------------------------------------------
# Test 2 — classify_feature returns LLMVerdict under DummyLM
# --------------------------------------------------------------------------


def test_classify_feature_returns_llm_verdict_under_dummy_lm(reset_dspy_lm) -> None:
    """With a DummyLM stub, classify_feature emits a typed LLMVerdict.

    Pins that the loader -> CausalRoleClassifier -> LLMVerdict adapter
    chain works under a deterministic-LLM stub. CI uses this path so no
    real API key is required for the wiring to be exercised.
    """
    from src.data.causal_role_classifier_loader import (
        classify_feature,
        load_compiled_classifier,
    )

    dspy.configure(lm=_stub_dspy_lm_with_role("ancestor", "keep_with_caveat"))
    classifier = load_compiled_classifier()
    assert classifier is not None

    verdict = classify_feature(
        feature_name="age_continuous",
        derivation_pseudocode="(index_date - birth_date).years",
        dataset_context="cohort=csu; target=treatment_initiated; prediction_anchor=index_date",
        classifier=classifier,
    )
    assert verdict is not None
    assert verdict.causal_role == "ancestor"
    assert verdict.recommended_remediation == "keep_with_caveat"


def test_classify_feature_returns_none_when_no_lm_configured(reset_dspy_lm) -> None:
    """classify_feature silently no-ops when no LM is configured.

    Pins acceptance criterion #2 from the issue: when no LM endpoint is
    configured, the Layer 4 path returns None and the caller falls through
    to the non-LLM verdict path — no raise, no crash, just a documented
    skip. Codex review pass-1 deliverable.
    """
    from src.data.causal_role_classifier_loader import classify_feature

    # Force-clear the LM
    dspy.settings.configure(lm=None)
    verdict = classify_feature(
        feature_name="age_continuous",
        derivation_pseudocode="dummy",
        dataset_context="dummy",
    )
    assert verdict is None


def test_classify_feature_coerces_non_string_mechanism(reset_dspy_lm) -> None:
    """A valid role + invalid non-string mechanism still yields a clean
    LLMVerdict (mechanism coerced to empty string) — does NOT raise.

    Codex pass-1 MEDIUM-1 (issue #193): the loader previously fed
    ``mechanism`` directly to ``_extract_pmids`` which calls
    ``re.findall``. A non-string mechanism (list, dict) would raise
    ``TypeError`` outside the LM-call try/except, breaking the
    "malformed LLM output falls back cleanly" contract.
    """
    from dspy.utils.dummies import DummyLM

    from src.data.causal_role_classifier_loader import (
        classify_feature,
        load_compiled_classifier,
    )

    # DummyLM that returns a valid role but a non-string mechanism.
    # The hidden 'reasoning' field is also a string, the mechanism field
    # is a list — exercising the loader's string-coercion path without
    # tripping any DSPy-level type validation upstream.
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "test stub reasoning",
                    "causal_role": "ancestor",
                    # DummyLM emits this as-is; downstream signature
                    # coercion may stringify it, so the test is
                    # belt-and-braces: the loader-side guard runs even
                    # if upstream stringifies first.
                    "mechanism": ["not", "a", "string"],
                    "recommended_remediation": "keep_with_caveat",
                }
            ]
        )
    )
    classifier = load_compiled_classifier()
    assert classifier is not None
    verdict = classify_feature(
        feature_name="age_continuous",
        derivation_pseudocode="dummy",
        dataset_context="dummy",
        classifier=classifier,
    )
    # Whether the upstream parser stringified the list or the loader's
    # type guard fired, the verdict must NOT be None (role is valid) and
    # ``mechanism`` must be a string (the dataclass invariant).
    assert verdict is not None
    assert isinstance(verdict.mechanism, str)


def test_classify_feature_handles_malformed_llm_role(reset_dspy_lm) -> None:
    """An LLM returning an out-of-vocab role yields None, not a crash.

    Pins acceptance criterion #5 from the issue: the wired verdict
    correctly falls back when the LLM returns a malformed answer.
    """
    from src.data.causal_role_classifier_loader import (
        classify_feature,
        load_compiled_classifier,
    )

    dspy.configure(lm=_stub_dspy_lm_with_role("not_a_real_role", "drop"))
    classifier = load_compiled_classifier()
    assert classifier is not None
    verdict = classify_feature(
        feature_name="age_continuous",
        derivation_pseudocode="dummy",
        dataset_context="dummy",
        classifier=classifier,
    )
    assert verdict is None, (
        f"Expected None for out-of-vocab role; got {verdict}. The loader "
        f"must sanitise so the voter does not see an invalid role downstream."
    )


# --------------------------------------------------------------------------
# Test 3 — _compose_legacy_verdict carries decided_by='llm'
# --------------------------------------------------------------------------


def test_compose_legacy_verdict_with_llm_emits_decided_by_llm(reset_dspy_lm) -> None:
    """_compose_legacy_verdict forwards llm_verdict to the voter and the
    resulting legacy dict carries decided_by="llm" + layer="4".

    This is the load-bearing wiring assertion for issue #193's acceptance
    criterion: "wired verdict correctly emits decided_by='llm'".
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import LLMVerdict

    voter = EnsembleVoter()
    # Adversarial moderate input (ambiguous bucket — Layer 4 trigger).
    adversarial_input = {
        "layer": "3",
        "severity": "moderate",
        "remediation": "ambiguous",
        "evidence": "test moderate signal",
        "z_score": 3.5,
        "actual_auc": 0.62,
        "null_mean": 0.50,
        "null_std": 0.035,
        "p_value": 0.001,
        "n_permutations": 200,
        # Per the codex MED-5 routing-guard contract: any
        # ``adversarial_input`` passed in directly must carry this tag.
        "_hblp_classified": True,
    }
    llm_verdict = LLMVerdict(
        causal_role="ancestor",
        mechanism="test mechanism: pre-index demographic",
        recommended_remediation="keep_with_caveat",
        cited_pmids=(),
    )
    verdict = _compose_legacy_verdict(
        "age_continuous",
        voter=voter,
        adversarial_input=adversarial_input,
        llm_verdict=llm_verdict,
    )
    assert verdict["decided_by"] == "llm", (
        f"Expected decided_by='llm', got {verdict.get('decided_by')!r}. Full verdict: {verdict}"
    )
    assert verdict["layer"] == "4"


def test_compose_legacy_verdict_caps_llm_leak_role_when_joint_check_fired(
    reset_dspy_lm,
) -> None:
    """Issue #212 cap — when the joint check fired on the adversarial
    input (``delta_auc_below_floor=True``) AND the LLM verdict path
    selected a leak role (final severity would be 'high' / drop), the
    final ``severity`` / ``remediation`` MUST be capped to the
    joint-clamped adversarial values. The LLM audit fields
    (``decided_by='llm'``, ``layer='4'``, ``llm_role``,
    ``llm_remediation``) MUST be preserved so audit consumers see
    Layer 4 was consulted but its severity was capped.

    Codex pass-1 HIGH-1 (issue #212): without this cap, the LLM path
    can silently relax #194's downstream bar by promoting a
    joint-clamped weak-effect feature back to ``high`` / drop. The cap
    is INWARD only (high → info, drop → keep); Layer 4 cannot relax
    info → high via this path.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import LLMVerdict

    voter = EnsembleVoter()
    # Joint-check-clamped adversarial input — z is in moderate band
    # (severity_pre_joint_check='moderate') but |delta_AUC| <= floor
    # forced severity to 'info'. This is the exact #212 scenario.
    adversarial_input = {
        "layer": "3",
        "severity": "info",  # joint-clamped
        "severity_pre_joint_check": "moderate",  # z-only band
        "remediation": "keep",
        "evidence": "test joint-clamped moderate signal",
        "z_score": 4.0,
        "actual_auc": 0.55,
        "null_mean": 0.50,
        "null_std": 0.0125,
        "p_value": 0.001,
        "n_permutations": 200,
        "delta_auc": 0.05,  # below floor 0.10
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": True,  # joint check fired
        "_hblp_classified": True,
    }
    # LLM verdict that maps to a LEAK role — would normally promote
    # severity to 'high' / drop via _llm_severity. The cap must
    # prevent this promotion since the joint check fired.
    llm_verdict = LLMVerdict(
        causal_role="descendant",  # leak role
        mechanism="hypothetical leak path",
        recommended_remediation="drop",
        cited_pmids=(),
    )
    verdict = _compose_legacy_verdict(
        "weak_feat",
        voter=voter,
        adversarial_input=adversarial_input,
        llm_verdict=llm_verdict,
    )
    # Audit fields preserved.
    assert verdict["decided_by"] == "llm", (
        f"Audit-trail integrity: decided_by must stay 'llm' even when "
        f"the cap fires; got {verdict.get('decided_by')!r}"
    )
    assert verdict["layer"] == "4", (
        f"Audit-trail integrity: layer must stay '4' even when the cap "
        f"fires; got {verdict.get('layer')!r}"
    )
    assert verdict["llm_role"] == "descendant", (
        f"Audit-trail integrity: llm_role must be preserved; got {verdict.get('llm_role')!r}"
    )
    assert verdict["llm_remediation"] == "drop", (
        f"Audit-trail integrity: llm_remediation must be preserved; got "
        f"{verdict.get('llm_remediation')!r}"
    )
    # Final severity capped to joint-clamped 'info' (NOT 'high' as the
    # LLM would have produced). This is the load-bearing #212 cap.
    assert verdict["severity"] == "info", (
        f"Issue #212 cap: final severity MUST be capped to joint-clamped "
        f"'info' when delta_auc_below_floor=True AND decided_by='llm'; "
        f"got {verdict.get('severity')!r}. Full verdict: {verdict}"
    )
    assert verdict["remediation"] == "keep", (
        f"Issue #212 cap: final remediation MUST be capped to "
        f"joint-clamped 'keep'; got {verdict.get('remediation')!r}"
    )
    # The cap annotation must appear in evidence so audit readers see
    # WHY the LLM verdict's severity was capped.
    assert "issue #212" in verdict["evidence"].lower() or "212 cap" in verdict["evidence"], (
        f"Issue #212 cap: evidence string must record the cap rationale; "
        f"got evidence={verdict['evidence']!r}"
    )


def test_compose_legacy_verdict_does_not_cap_when_joint_check_did_not_fire(
    reset_dspy_lm,
) -> None:
    """Issue #212 cap — the cap MUST NOT fire when the joint check did
    NOT fire (``delta_auc_below_floor=False``). The LLM verdict path
    can still promote a verdict to 'high' / drop when the underlying
    Layer 3 signal genuinely passed the joint check.

    This is the symmetric negative pin to
    ``test_compose_legacy_verdict_caps_llm_leak_role_when_joint_check_fired``.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import LLMVerdict

    voter = EnsembleVoter()
    # Adversarial input where joint check did NOT fire — z and
    # delta_AUC both above thresholds, so severity stays moderate.
    adversarial_input = {
        "layer": "3",
        "severity": "moderate",
        "severity_pre_joint_check": "moderate",
        "remediation": "ambiguous",
        "evidence": "real moderate signal with above-floor delta_AUC",
        "z_score": 4.0,
        "actual_auc": 0.70,
        "null_mean": 0.50,
        "null_std": 0.05,
        "p_value": 0.001,
        "n_permutations": 200,
        "delta_auc": 0.20,  # above floor
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": False,  # joint check NOT fired
        "_hblp_classified": True,
    }
    llm_verdict = LLMVerdict(
        causal_role="descendant",
        mechanism="post-T leak",
        recommended_remediation="drop",
        cited_pmids=(),
    )
    verdict = _compose_legacy_verdict(
        "leaky_feat",
        voter=voter,
        adversarial_input=adversarial_input,
        llm_verdict=llm_verdict,
    )
    # LLM path fired and is NOT capped (joint check did not fire).
    assert verdict["decided_by"] == "llm"
    assert verdict["layer"] == "4"
    # Final severity reflects the LLM's leak-role assessment (high),
    # NOT the moderate adversarial input. This is normal Phase 2.9
    # Stage 3 behaviour for the unclamped path.
    assert verdict["severity"] == "high", (
        f"When joint check did NOT fire, the LLM leak-role verdict MUST "
        f"propagate to final severity. Got severity="
        f"{verdict.get('severity')!r}. Full verdict: {verdict}"
    )
    assert verdict["remediation"] == "drop"
    # No #212 cap annotation in evidence (cap did not fire).
    assert "212 cap" not in verdict["evidence"]


def test_compose_legacy_verdict_does_not_cap_when_ablation_corroborated(
    reset_dspy_lm,
) -> None:
    """Issue #212 codex pass-2 MED-1 follow-on: the cap MUST NOT fire
    when issue #196's ablation pass independently corroborated the
    signal (``ablation_severity`` in {moderate, high}).

    Scenario: permutation z=4σ + |delta_AUC|=0.05 (below floor 0.10)
    → joint check clamps severity to info, delta_auc_below_floor=True,
    severity_pre_joint_check=moderate. But ablation z=4σ +
    ablation_delta_AUC=0.15 (above floor) → ablation classifier
    returns 'moderate' → _combine_ablation_with_permutation escalates
    severity (info → moderate) AND severity_pre_joint_check (moderate
    preserved). Ablation_severity='moderate' set as audit field.

    LLM verdict 'descendant' (leak) → voter assigns severity=high/drop.

    PRE-CORROBORATION-GUARD FIX: the cap would fire because
    delta_auc_below_floor=True AND decided_by='llm', capping severity
    back to permutation's joint-clamped 'info' — SILENTLY relaxing the
    ablation contract. The corroboration guard skips the cap when
    ablation independently passed its own joint check.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import LLMVerdict

    voter = EnsembleVoter()
    # Adversarial input: permutation joint-clamped, ablation escalated.
    adversarial_input = {
        "layer": "3",
        "severity": "moderate",  # ablation escalated this
        "severity_pre_joint_check": "moderate",
        "remediation": "ambiguous",
        "evidence": "permutation joint-clamped but ablation escalated",
        "z_score": 4.0,
        "actual_auc": 0.55,
        "null_mean": 0.50,
        "null_std": 0.0125,
        "p_value": 0.001,
        "n_permutations": 200,
        "delta_auc": 0.05,  # permutation below floor
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": True,  # permutation joint check fired
        # Ablation independently corroborated the signal — its
        # classifier ALREADY applied the joint check on its OWN
        # delta_auc and returned 'moderate'.
        "ablation_z_score": 4.0,
        "ablation_delta_auc": 0.15,  # ABOVE floor (independent corroboration)
        "ablation_null_mean": 0.0,
        "ablation_null_std": 0.03,
        "ablation_severity": "moderate",  # corroborates Layer 3
        "_hblp_classified": True,
    }
    llm_verdict = LLMVerdict(
        causal_role="descendant",  # leak role
        mechanism="ablation-corroborated leak path",
        recommended_remediation="drop",
        cited_pmids=(),
    )
    verdict = _compose_legacy_verdict(
        "corroborated_feat",
        voter=voter,
        adversarial_input=adversarial_input,
        llm_verdict=llm_verdict,
    )
    # Audit fields preserved.
    assert verdict["decided_by"] == "llm"
    assert verdict["layer"] == "4"
    # Final severity is NOT capped because ablation independently
    # corroborated. The voter's LLM-derived severity propagates.
    assert verdict["severity"] == "high", (
        f"Issue #212 pass-2 MED-1 follow-on: cap MUST NOT fire when "
        f"ablation_severity in {{moderate, high}} — that means #196's "
        f"ablation joint check independently corroborated the signal. "
        f"Got severity={verdict['severity']!r}, expected 'high' "
        f"(LLM-driven). Full verdict: {verdict}"
    )
    assert verdict["remediation"] == "drop"
    # No #212 cap annotation in evidence (cap did not fire).
    assert "212 cap" not in verdict["evidence"], (
        f"Cap annotation must NOT appear when ablation corroborated; "
        f"got evidence={verdict['evidence']!r}"
    )


def test_combine_then_compose_skips_cap_when_ablation_independently_corroborates(
    reset_dspy_lm,
) -> None:
    """Issue #212 codex pass-3 LOW-1: end-to-end production-wiring test
    for the ablation corroboration guard.

    The sibling ``test_compose_legacy_verdict_does_not_cap_when_ablation_corroborated``
    isolates the cap predicate by hand-building ``adversarial_input``
    with ``ablation_severity='moderate'``. That verifies the guard's
    LOGIC but does NOT verify that ``_combine_ablation_with_permutation``
    is the producer that populates the field the guard reads. If the
    producer drifts (e.g. a future refactor renames the audit key or
    only sets it on the escalation branch), the guard becomes inert and
    the cap silently relaxes #196's ablation contract.

    This test exercises the production wiring: run the combiner FIRST,
    assert ``ablation_severity`` is populated, then feed the resulting
    dict through ``_compose_legacy_verdict`` and verify the cap is
    skipped.

    Scenario:
      * Permutation z=4σ + |delta_AUC|=0.05 (below floor 0.10) →
        hblp_classify joint-clamps severity to 'info';
        ``severity_pre_joint_check='moderate'`` (raw z lands in moderate
        band); ``delta_auc_below_floor=True``.
      * Ablation z=4σ + |delta_AUC|=0.15 (above floor) →
        ``_classify_ablation_severity`` returns 'moderate'.
      * ``_combine_ablation_with_permutation`` MAX-escalates severity
        info → moderate AND publishes ``ablation_severity='moderate'``.
      * LLM 'descendant' leak verdict triggers Layer 4 via the voter.
      * Cap predicate sees ``delta_auc_below_floor=True`` AND
        ``decided_by='llm'`` AND ``ablation_severity='moderate'`` →
        corroboration guard fires → cap SKIPPED → LLM severity
        ('high') propagates.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _combine_ablation_with_permutation,
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import LLMVerdict

    # Joint-clamped permutation-side input mirroring what hblp_classify
    # emits after #194's joint check fires: raw z=4σ in moderate band,
    # |delta_auc|=0.05 below 0.10 floor → severity clamped to 'info' but
    # ``severity_pre_joint_check`` retains 'moderate'.
    perm_input: dict[str, Any] = {
        "layer": "3",
        "severity": "info",
        "severity_pre_joint_check": "moderate",
        "remediation": "keep",
        "evidence": "permutation joint-clamped",
        "z_score": 4.0,
        "actual_auc": 0.55,
        "null_mean": 0.50,
        "null_std": 0.0125,
        "p_value": 0.001,
        "n_permutations": 200,
        "delta_auc": 0.05,
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": True,
        "_hblp_classified": True,
    }
    # Ablation independently passes its own joint check:
    #   z=4σ > MODERATE_Z=3.0 AND |delta_auc|=0.15 > floor 0.10 →
    #   _classify_ablation_severity → 'moderate'.
    ablation_row: dict[str, Any] = {
        "z_score": 4.0,
        "delta_auc": 0.15,
        "null_mean": 0.0,
        "null_std": 0.0375,
    }

    combined = _combine_ablation_with_permutation(perm_input, ablation_row)

    # Production combiner populated the audit field that the cap guard
    # reads. If this assertion fails the guard is structurally inert
    # regardless of how the cap predicate is written.
    assert combined["ablation_severity"] == "moderate", (
        f"Expected _combine_ablation_with_permutation to publish "
        f"ablation_severity='moderate'; got "
        f"{combined.get('ablation_severity')!r}. The corroboration "
        f"guard in _compose_legacy_verdict depends on this field — if "
        f"the producer drifts, the cap silently relaxes #196's "
        f"ablation contract."
    )
    # MAX-rule escalated permutation 'info' → 'moderate'.
    assert combined["severity"] == "moderate"
    assert combined["remediation"] == "ambiguous"
    # Permutation-side joint-clamp signal still intact — that is the
    # cap predicate's trigger condition.
    assert combined["delta_auc_below_floor"] is True

    # Now feed the combined dict through the voter + cap with a leak
    # LLM verdict. The cap predicate sees delta_auc_below_floor=True
    # AND decided_by='llm', but ALSO sees ablation_severity='moderate'
    # (corroboration) and must skip the cap.
    voter = EnsembleVoter()
    llm_verdict = LLMVerdict(
        causal_role="descendant",
        mechanism="end-to-end ablation-corroborated leak",
        recommended_remediation="drop",
        cited_pmids=(),
    )
    verdict = _compose_legacy_verdict(
        "ablation_corroborated_feat",
        voter=voter,
        adversarial_input=combined,
        llm_verdict=llm_verdict,
    )

    # LLM audit fields surfaced.
    assert verdict["decided_by"] == "llm"
    assert verdict["layer"] == "4"
    # Final severity follows the LLM (corroboration guard skipped cap).
    assert verdict["severity"] == "high", (
        f"Issue #212 pass-3 LOW-1: end-to-end corroboration guard must "
        f"skip the cap when ablation_severity in {{moderate, high}}, "
        f"as populated by _combine_ablation_with_permutation. Got "
        f"severity={verdict['severity']!r}; expected 'high'. "
        f"Full verdict: {verdict}"
    )
    assert verdict["remediation"] == "drop"
    # Cap annotation must NOT appear when ablation corroborated.
    assert "212 cap" not in verdict["evidence"], (
        f"Cap annotation must not appear when ablation independently "
        f"corroborates the Layer-3 signal. Got "
        f"evidence={verdict['evidence']!r}"
    )


def test_compose_legacy_verdict_cap_fires_on_remediation_only_difference(
    reset_dspy_lm,
) -> None:
    """Issue #212 codex pass-2 LOW-1: the cap MUST fire when the LLM
    verdict matches severity (both 'info') but DIFFERS on remediation
    (e.g. LLM 'ancestor' → 'keep_with_caveat' vs joint-clamped 'keep').

    Pre-pass-2 the cap guard only checked severity, so a non-leak LLM
    role on a joint-clamped feature would leave remediation as the
    voter's value, diverging from the joint-clamped contract and
    omitting the #212 audit annotation.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import LLMVerdict

    voter = EnsembleVoter()
    # Joint-check-clamped adversarial input (same shape as the prior
    # cap test). Joint-clamped severity='info', remediation='keep'.
    adversarial_input = {
        "layer": "3",
        "severity": "info",
        "severity_pre_joint_check": "moderate",
        "remediation": "keep",
        "evidence": "test joint-clamped moderate signal",
        "z_score": 4.0,
        "actual_auc": 0.55,
        "null_mean": 0.50,
        "null_std": 0.0125,
        "p_value": 0.001,
        "n_permutations": 200,
        "delta_auc": 0.05,
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": True,
        "_hblp_classified": True,
    }
    # LLM verdict: NON-leak role 'ancestor' with 'keep_with_caveat'
    # remediation. The voter's LLM path maps this to severity='info'
    # via ``_llm_severity`` (non-leak). But remediation comes from
    # ``_role_to_remediation`` which returns 'keep_with_caveat' for
    # accept-roles with the LLM's recommended remediation. Joint-
    # clamped remediation is 'keep'; the cap must reset it.
    llm_verdict = LLMVerdict(
        causal_role="ancestor",
        mechanism="pre-index demographic",
        recommended_remediation="keep_with_caveat",
        cited_pmids=(),
    )
    verdict = _compose_legacy_verdict(
        "weak_feat",
        voter=voter,
        adversarial_input=adversarial_input,
        llm_verdict=llm_verdict,
    )
    # Audit fields preserved.
    assert verdict["decided_by"] == "llm"
    assert verdict["layer"] == "4"
    assert verdict["llm_role"] == "ancestor"
    assert verdict["llm_remediation"] == "keep_with_caveat"
    # Severity matches both sides ('info') but remediation must be
    # capped from 'keep_with_caveat' (or whatever voter produced) to
    # the joint-clamped 'keep'.
    assert verdict["severity"] == "info"
    assert verdict["remediation"] == "keep", (
        f"Issue #212 pass-2 LOW-1: when joint check fired AND voter "
        f"selected LLM path AND remediation differs, the cap MUST "
        f"reset remediation to joint-clamped 'keep'. Got "
        f"remediation={verdict.get('remediation')!r}"
    )
    # Audit annotation must appear even though severity matched.
    assert "212 cap" in verdict["evidence"] or "issue #212" in verdict["evidence"].lower(), (
        f"Issue #212 pass-2 LOW-1: cap annotation must appear in "
        f"evidence whenever the cap condition is reached, even when "
        f"only remediation differs. Got evidence={verdict['evidence']!r}"
    )


def test_compose_legacy_verdict_without_llm_falls_back_to_adversarial(
    reset_dspy_lm,
) -> None:
    """When llm_verdict=None, _compose_legacy_verdict preserves the legacy
    adversarial-alone bypass — decided_by stays 'adversarial', not 'llm'.

    Codex pass-1 deliverable: ensure the LLM path is opt-in and the
    no-LLM path does not silently change behaviour.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter

    voter = EnsembleVoter()
    adversarial_input = {
        "layer": "3",
        "severity": "moderate",
        "remediation": "ambiguous",
        "evidence": "test moderate signal",
        "z_score": 3.5,
        "actual_auc": 0.62,
        "null_mean": 0.50,
        "null_std": 0.035,
        "p_value": 0.001,
        "n_permutations": 200,
        "_hblp_classified": True,
    }
    verdict = _compose_legacy_verdict(
        "age_continuous",
        voter=voter,
        adversarial_input=adversarial_input,
        llm_verdict=None,
    )
    assert verdict["decided_by"] == "adversarial"


# --------------------------------------------------------------------------
# Test 4 — End-to-end adaptive_verdicts flow (acceptance criterion #4)
# --------------------------------------------------------------------------


def _make_layer_3_moderate_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Synthesize a (feature, target) frame whose Layer 3 z-score lands in
    the ``moderate`` bucket (3σ < z ≤ 5σ).

    The exact z depends on the permutation null variance, which scales as
    ~1/sqrt(n_pos). At n=400 with prevalence 0.50 the null std is ~0.018;
    we tune signal strength to land in the 3σ-5σ window. Seed chosen so
    the realized z is stable across pytest-xdist parallelisation orderings.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # x = signal * y + noise — tune so AUC lands ~0.58 (z ~ 3.5σ).
    x = 0.18 * y + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"age_continuous": x, "treatment_initiated": y})


def test_adaptive_validity_check_emits_decided_by_llm_on_csu_feature(
    reset_dspy_lm,
) -> None:
    """End-to-end: the orchestrator emits decided_by='llm' for at least
    one CSU manifest feature when the LLM verdict path is active.

    Acceptance criterion #4 from issue #193:
        "adaptive_verdicts output for a CSU manifest feature contains
         decided_by='llm' when LM endpoint is configured"

    Uses ``age_at_index`` because it is in the CSU manifest as a
    legitimate pre-index demographic (knowable_at <= index_date), and the
    LLM stub classifies it as ``ancestor`` so the voter's LLM precedence
    rule fires.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    dspy.configure(lm=_stub_dspy_lm_with_role("ancestor", "keep_with_caveat"))

    df = _make_layer_3_moderate_df()
    state = {
        "experiment_id": "test-issue-193",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": "treatment_initiated",
            "required_features": ["age_continuous"],
            "excluded_features": [],
            "feature_manifest_source": "csu",
        },
        "leakage_findings": [],
        "leaked_features": [],
        # Phase 1: opt into the LLM auditor CALL (off by default). Combined with
        # the autouse ADAPTIVE_LAYER4_LLM_DECIDES=1 fixture, this exercises the
        # full legacy decided_by="llm" path end-to-end.
        "adaptive_layer4_enabled": True,
    }
    result = asyncio.run(adaptive_validity_check(state))
    verdicts = result["adaptive_verdicts"]
    decided_by_values = [v.get("decided_by") for v in verdicts]
    assert any(d == "llm" for d in decided_by_values), (
        f"Expected at least one verdict with decided_by='llm'; got "
        f"{decided_by_values}. Full verdicts: {verdicts}"
    )
    # The 'llm' verdict's layer must map to '4' per _DECIDED_BY_TO_LAYER.
    llm_verdicts = [v for v in verdicts if v.get("decided_by") == "llm"]
    assert all(v.get("layer") == "4" for v in llm_verdicts), (
        f"Expected layer='4' for all decided_by='llm' verdicts; got "
        f"{[(v['feature'], v.get('layer')) for v in llm_verdicts]}"
    )


def test_layer4_llm_not_called_by_default(reset_dspy_lm, monkeypatch) -> None:
    """Plan v4 Phase 1 — the FULL production default (BOTH gates OFF) end-to-end.
    With an LM configured but adaptive_layer4_enabled unset (call-gate OFF) AND
    ADAPTIVE_LAYER4_LLM_DECIDES unset (voter audit-only), the node does NOT
    invoke the LLM — no verdict is decided_by='llm' and no llm_role is surfaced;
    the FDR confident set + the deterministic voter decide. Deletes the file's
    autouse decides env so this regression-guards the true production path
    (codex iter-1 MEDIUM: no single test covered both defaults OFF together)."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    # Override the file-level autouse fixture: exercise BOTH production defaults
    # (call-gate OFF via the omitted state flag + voter audit-only via no env).
    monkeypatch.delenv("ADAPTIVE_LAYER4_LLM_DECIDES", raising=False)
    dspy.configure(lm=_stub_dspy_lm_with_role("ancestor", "keep_with_caveat"))
    df = _make_layer_3_moderate_df()
    state = {
        "experiment_id": "test-phase1-gate-off",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": "treatment_initiated",
            "required_features": ["age_continuous"],
            "excluded_features": [],
            "feature_manifest_source": "csu",
        },
        "leakage_findings": [],
        "leaked_features": [],
        # adaptive_layer4_enabled intentionally omitted → default OFF.
    }
    result = asyncio.run(adaptive_validity_check(state))
    verdicts = result["adaptive_verdicts"]
    assert all(v.get("decided_by") != "llm" for v in verdicts), (
        "LLM must not decide when the auditor is off by default; got "
        f"{[(v['feature'], v.get('decided_by')) for v in verdicts]}"
    )
    assert all(v.get("llm_role") is None for v in verdicts), (
        "LLM must not be called (no llm_role) when adaptive_layer4_enabled is off"
    )


def test_legacy_verdict_carries_llm_audit_fields_consistently(reset_dspy_lm) -> None:
    """Every emitted legacy verdict dict has ``llm_role`` + ``llm_remediation``
    keys (None when no LLM was supplied; populated when one was).

    Codex pass-3 LOW (issue #193): shape consistency across the
    voter-routed path AND every bypass path
    (``_legacy_adversarial_alone_verdict``, ``_legacy_info_verdict``,
    ``_legacy_short_circuit_verdict``). Without this pin, downstream
    audit consumers (``write_adaptive_verdicts_sidecar``) would
    KeyError on the bypass-emitted verdicts when querying
    ``llm_role``.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _legacy_adversarial_alone_verdict,
        _legacy_info_verdict,
        _legacy_short_circuit_verdict,
    )

    adv = {
        "z_score": 4.0,
        "actual_auc": 0.6,
        "null_mean": 0.5,
        "null_std": 0.025,
        "p_value": 0.001,
        "n_permutations": 200,
        "severity": "moderate",
        "remediation": "ambiguous",
        "evidence": "moderate signal",
        "_hblp_classified": True,
    }
    bypass_alone = _legacy_adversarial_alone_verdict("feat", adv)
    bypass_info = _legacy_info_verdict("feat", adversarial_input=adv, evidence="x")
    bypass_short = _legacy_short_circuit_verdict("feat", evidence="x")
    for d in (bypass_alone, bypass_info, bypass_short):
        assert "llm_role" in d
        assert "llm_remediation" in d
        assert d["llm_role"] is None
        assert d["llm_remediation"] is None
        # Issue #212 — schema uniformity for the pre-joint-check
        # severity audit field. Every bypass path emits it (always a
        # str), even when the bypass didn't invoke hblp_classify (it
        # falls back to ``"info"`` matching the bypass's final
        # severity for those cases).
        assert "severity_pre_joint_check" in d
        assert isinstance(d["severity_pre_joint_check"], str)


def test_ensure_dspy_lm_configured_typoed_provider_fails_closed(reset_dspy_lm, monkeypatch) -> None:
    """Typoed provider prefix (e.g. 'antropic/...') with only OPENAI_API_KEY
    must return False, NOT fall back to permissive any-key.

    Codex pass-3 MEDIUM (issue #193): the pass-2 fix closed the
    ANTHROPIC_API_KEY/OPENAI_API_KEY mismatch for KNOWN providers, but
    a typoed-but-non-empty provider prefix still fell through the
    unknown-provider permissive any-key fallback. An env with only
    OPENAI_API_KEY + model='antropic/claude-sonnet-4-6' (missing
    'h') would green-light configuration of an Anthropic LM, then
    classify_feature would silently fail every call. Fail closed for
    slash-shaped unknown prefixes.
    """
    from src.data.causal_role_classifier_loader import ensure_dspy_lm_configured

    dspy.settings.configure(lm=None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only-not-real")
    # Typoed provider — looks like a provider/path but the prefix is
    # not in _PROVIDER_TO_ENV_VARS.
    assert ensure_dspy_lm_configured(model="antropic/claude-sonnet-4-6") is False
    assert getattr(dspy.settings, "lm", None) is None


def test_ensure_dspy_lm_configured_provider_mismatch_skips(reset_dspy_lm, monkeypatch) -> None:
    """OPENAI_API_KEY set but model is anthropic/* → skip (return False).

    Codex pass-2 MEDIUM (issue #193): the credential gate must match
    the model's provider prefix. An env with only OPENAI_API_KEY set
    must NOT green-light configuration of the default Anthropic model
    (which would then fail every classify_feature call with no LLM auth).
    """
    from src.data.causal_role_classifier_loader import ensure_dspy_lm_configured

    dspy.settings.configure(lm=None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only-not-real")

    # Default model is anthropic/*; with no ANTHROPIC_API_KEY this must
    # short-circuit to False even though OPENAI_API_KEY is "set".
    assert ensure_dspy_lm_configured() is False
    assert getattr(dspy.settings, "lm", None) is None


def test_ensure_dspy_lm_configured_rejects_whitespace_only_key(reset_dspy_lm, monkeypatch) -> None:
    """A whitespace-only key value must be treated as no key.

    Codex pass-2 MEDIUM (issue #193): ``os.environ.get(v)`` returns the
    string verbatim, so a value of ``"   "`` is truthy by length. The
    helper now ``.strip()``s before checking; this test pins that.
    """
    from src.data.causal_role_classifier_loader import ensure_dspy_lm_configured

    dspy.settings.configure(lm=None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   \t\n  ")
    assert ensure_dspy_lm_configured() is False
    assert getattr(dspy.settings, "lm", None) is None


def test_ensure_dspy_lm_configured_provider_match_proceeds(reset_dspy_lm, monkeypatch) -> None:
    """OPENAI_API_KEY set AND model is openai/* → configure (return True).

    Verifies the provider-aware gate is symmetric: OpenAI key + OpenAI
    model proceeds; the helper does not require Anthropic specifically.
    """
    from src.data.causal_role_classifier_loader import ensure_dspy_lm_configured

    dspy.settings.configure(lm=None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only-not-real")
    # dspy.LM('openai/gpt-4o-mini') instantiation does NOT contact the
    # network — only at inference time. So the helper proceeds without
    # a real key as long as the env var is present and matches the provider.
    assert ensure_dspy_lm_configured(model="openai/gpt-4o-mini") is True
    assert getattr(dspy.settings, "lm", None) is not None


def test_ensure_dspy_lm_configured_skips_when_no_key(reset_dspy_lm, monkeypatch) -> None:
    """ensure_dspy_lm_configured returns False when no API key in env.

    Codex pass-1 HIGH-1 (issue #193): the helper bridges the gap between
    "LM not configured" and "production has a key in env but never called
    dspy.configure()". This test pins the no-key branch — without a key,
    the helper must NOT raise, must NOT configure any LM, must return
    False so the orchestrator's caller falls through to the Layer 4 skip.
    """
    from src.data.causal_role_classifier_loader import ensure_dspy_lm_configured

    dspy.settings.configure(lm=None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ensure_dspy_lm_configured() is False
    # And nothing got configured silently.
    assert getattr(dspy.settings, "lm", None) is None


def test_ensure_dspy_lm_configured_idempotent_when_already_configured(
    reset_dspy_lm,
) -> None:
    """A second call with an LM already configured is a no-op.

    Pins the once-flag-like behaviour without an actual global flag (so
    tests can freely reset ``dspy.settings.lm`` and re-trigger).
    """
    from src.data.causal_role_classifier_loader import ensure_dspy_lm_configured

    # Pre-configure with a Dummy so the helper sees lm != None.
    pre_lm = _stub_dspy_lm_with_role("ancestor")
    dspy.configure(lm=pre_lm)
    assert ensure_dspy_lm_configured() is True
    # And the pre-configured LM is still in place (helper did not replace it).
    assert dspy.settings.lm is pre_lm


def test_ensure_dspy_lm_configured_high_declared_safe_records_llm_disagreement(
    reset_dspy_lm, monkeypatch
) -> None:
    """High adversarial + Layer 1 declared safe + LLM=accept-role records
    the disagreement in the audit trail.

    Codex pass-1 MEDIUM-2 (issue #193): the Layer 4 trigger now also
    fires for ``severity=high AND layer_1_declared_safe`` so the
    voter's ``adversarial=high but llm=<accept-role>`` disagreement
    string lands in the audit trail. Without this, the operator has no
    Layer 4 signal to triage the Layer-1-vs-Layer-3 disagreement.

    Test setup: builds a high-signal feature (z > 5σ → severity=high)
    that is also in the CSU manifest as declared-safe; stubs the LLM
    with ``ancestor`` (accept-role); asserts the resulting verdict
    carries the disagreement.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    # Stub LM with accept-role so the voter records the disagreement.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-stub-key")
    dspy.configure(lm=_stub_dspy_lm_with_role("ancestor", "keep_with_caveat"))

    # Build a strong-signal frame so adversarial severity is high.
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 400)
    x = 0.8 * y + 0.3 * rng.standard_normal(400)
    df = pd.DataFrame({"age_continuous": x, "treatment_initiated": y})

    state = {
        "experiment_id": "test-issue-193-high-safe",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": "treatment_initiated",
            "required_features": ["age_continuous"],
            "excluded_features": [],
            "feature_manifest_source": "csu",
        },
        "leakage_findings": [],
        "leaked_features": [],
        # Phase 1: opt into the LLM auditor CALL so the disagreement is recorded.
        # The adversarial-high veto still wins on severity (decided_by stays
        # "adversarial"); the LLM is audited, not decisive — exactly Part B's
        # intent. (No decides flag needed: rule-2 precedes the LLM rule.)
        "adaptive_layer4_enabled": True,
    }
    result = asyncio.run(adaptive_validity_check(state))
    age_verdicts = [v for v in result["adaptive_verdicts"] if v["feature"] == "age_continuous"]
    assert len(age_verdicts) >= 1
    age_v = age_verdicts[0]
    # The adversarial veto wins on severity (deterministic high), so
    # decided_by stays "adversarial", but disagreements should include
    # the LLM accept-role.
    assert age_v["severity"] == "high"
    assert age_v["decided_by"] == "adversarial"
    disagreements = age_v.get("disagreements", [])
    assert any("llm=ancestor" in d for d in disagreements), (
        f"Expected disagreements to include 'llm=ancestor' for high+declared_safe "
        f"feature; got {disagreements}. Full verdict: {age_v}"
    )
    # Codex pass-3 LOW (issue #193): the LLM's role/remediation MUST be
    # surfaced in the legacy dict even when adversarial wins on severity.
    # Without this, the audit-cost of a Layer 4 LLM call is observable
    # only through the free-text disagreements field, not as a
    # structured query-friendly column.
    assert age_v.get("llm_role") == "ancestor", (
        f"Expected llm_role='ancestor' surfaced when adversarial wins; "
        f"got {age_v.get('llm_role')!r}. Full verdict: {age_v}"
    )
    assert age_v.get("llm_remediation") == "keep_with_caveat", (
        f"Expected llm_remediation='keep_with_caveat' surfaced when "
        f"adversarial wins; got {age_v.get('llm_remediation')!r}."
    )


def test_adaptive_validity_check_skips_layer_4_when_no_lm(reset_dspy_lm, monkeypatch) -> None:
    """When no LM is configured AND no API key is in env, Layer 4 silently
    skips — the moderate adversarial verdict goes through the legacy bypass.

    Defense against a future regression that silently fails the LLM path
    open. The non-LLM-configured-and-no-key run must NEVER emit
    ``decided_by='llm'``; if it does, something has cached a stale LM
    somewhere OR the env-vars-read path is broken (codex pass-1 HIGH-1
    introduced ``ensure_dspy_lm_configured`` which configures a default
    LM iff a recognised API key env var is non-empty — this test pins
    the no-key, no-prior-config path).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    # Clear LM + clear every API key env var so ensure_dspy_lm_configured
    # returns False (the documented CI / developer-laptop path).
    dspy.settings.configure(lm=None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    df = _make_layer_3_moderate_df()
    state = {
        "experiment_id": "test-issue-193-no-lm",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": "treatment_initiated",
            "required_features": ["age_continuous"],
            "excluded_features": [],
            "feature_manifest_source": "csu",
        },
        "leakage_findings": [],
        "leaked_features": [],
    }
    result = asyncio.run(adaptive_validity_check(state))
    for v in result["adaptive_verdicts"]:
        assert v.get("decided_by") != "llm", (
            f"Expected NO decided_by='llm' verdicts when LM is unset; got {v}"
        )
