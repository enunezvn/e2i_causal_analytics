"""Phase-4 S12 Option C §3.5 Stage 2: live-LM integration smoke.

Plan: ``.claude/plans/option_c_dspy_recompile_for_s12_FINAL.md`` §3.5 Stage 2
+ §4.2.

This test invokes ``classify_feature()`` against the live LM
(``anthropic/claude-sonnet-4-6`` by default per the loader's
``_DEFAULT_LM_MODEL``) for the 12 paired (T, Y)-explicit fixtures pinned by
the §3.5 Stage 1 unit-test gate, then asserts that the LM agrees with the
labeled role on ≥10 of 12 invocations.

The Stage 1 artifact-level gate (in ``tests/unit/test_data/test_causal_role_classifier.py``)
verifies the demos are PERSISTED with the expected roles; Stage 2 verifies the
classifier WIRING (signature -> few-shot context -> LM call -> output parsing)
end-to-end. The two gates protect different failure modes:

- Stage 1 fires if BootstrapFewShot drops a paired-fixture demo from the
  artifact OR assigns it the wrong role at compile time.
- Stage 2 fires if the LM, given the persisted demos as few-shot context,
  fails to reproduce the labeled role at inference time on the same inputs.

Run manually pre-PR with ``ANTHROPIC_API_KEY`` in env::

    pytest tests/integration/test_causal_role_classifier_stage2_smoke.py \\
        -v -m integration --no-header

Cost: ~$0.50 / ~2-3 minutes wall time at the default model. Marked
``@pytest.mark.integration`` so the default ``pytest`` invocation skips it
unless the marker is selected.
"""

from __future__ import annotations

import os

import pytest

# Re-use the canonical 12 quadruple list to enforce single-source-of-truth.
# Stage 1 (unit) and Stage 2 (this file, integration) both consume it; a
# drift between the two would let one gate go stale while the other passes.
from tests.unit.test_data.test_causal_role_classifier import (
    EXPECTED_TREATMENT_OUTCOME_QUADRUPLES,
)

# Threshold: plan §3.5 Stage 2 + §4.2 set the bar at ``>= 10/12 return
# expected role``. This is 83.3% — well above random (1/6 = 16.7%) for a
# 6-way categorical signature, while leaving slack for LM stochasticity.
STAGE_2_PASS_THRESHOLD = 10


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_stage2_classifier_reproduces_paired_fixture_roles() -> None:
    """Stage 2 live-LM smoke: ≥10 of 12 paired (T, Y) fixtures return the
    expected role via the production ``classify_feature()`` entry point.

    The test calls the same loader the production caller uses
    (``src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py:892-893``)
    so a regression in any layer of the wiring stack (loader, signature,
    DSPy LM dispatch, output parsing, role coercion) trips this gate.

    Failure-mode message lists every (feature, T, Y, expected, observed)
    so the operator can attribute the disagreement to a specific paired
    fixture and decide whether to (a) accept the LM's revised role,
    (b) swap the paired fixture per Option C plan §5 row 2 recovery, or
    (c) recompile with a stronger demo for that role.
    """
    # Skip cleanly when no usable provider key is present (developer
    # laptops without `.env` loaded; CI environments that set a
    # placeholder). The repo's CI workflow (`.github/workflows/backend-tests.yml`
    # `integration-tests` job) sets ``ANTHROPIC_API_KEY: 'test-key'`` as a
    # placeholder so other code paths don't crash on a missing var — but
    # that placeholder is rejected by the Anthropic API with 401. A
    # plain non-empty check would let the test fire against the bad key,
    # which is exactly what happened on PR #371's first CI run (0/12
    # PASS via 12x `litellm.AuthenticationError: invalid x-api-key`).
    #
    # We require the value to look like a real Anthropic key by prefix
    # (``sk-ant-``). This correctly skips:
    #   - empty / unset (no key at all)
    #   - ``test-key`` (CI placeholder per backend-tests.yml)
    #   - ``sk-test`` (unit-test placeholder per
    #     ``tests/integration/test_layer4_evaluator_audit.py:23``)
    # and correctly proceeds on real provider keys.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key.startswith("sk-ant-"):
        pytest.skip(
            "ANTHROPIC_API_KEY missing or not a real Anthropic key "
            "(expected `sk-ant-` prefix; got "
            f"{api_key[:7] + '…' if api_key else '<empty>'!r}). Stage 2 "
            "requires a live Anthropic key — run locally with `.env` loaded "
            "(`set -a && source .env && set +a`) before invoking pytest. "
            "CI environments that set `ANTHROPIC_API_KEY=test-key` will "
            "skip this test as designed (plan §3.5 Stage 2 is manual-only)."
        )

    # Local imports so import-time costs don't fire when the test is
    # skipped (the loader pulls dspy + the compiled artifact JSON).
    import dspy

    from src.data.causal_role_classifier import build_compile_set
    from src.data.causal_role_classifier_loader import (
        classify_feature,
        ensure_dspy_lm_configured,
        load_compiled_classifier,
    )

    # Build a lookup from the canonical compile set so the test consumes
    # the SAME inputs the artifact was compiled with. This deliberately
    # tests reproduction-on-same-inputs (the plan §3.5 Stage 2 contract),
    # NOT generalisation — generalisation is the S12 golden-set problem
    # (issue #358) which builds independent test fixtures.
    paired_inputs: dict[tuple[str, str, str], tuple[str, str]] = {}
    for example in build_compile_set():
        dataset_context = example.dataset_context
        if "treatment=" not in dataset_context:
            continue
        # Parse `treatment=X; outcome=Y` from the semicolon-delimited
        # context. This mirrors the loader's interpretation: the demos
        # are typed dspy.Example objects, so we can pull `treatment` and
        # `outcome` by string-split rather than re-implementing parsing.
        treatment = None
        outcome = None
        for field in dataset_context.split(";"):
            field = field.strip()
            if field.startswith("treatment="):
                treatment = field.split("=", 1)[1].strip()
            elif field.startswith("outcome="):
                outcome = field.split("=", 1)[1].strip()
        if treatment is None or outcome is None:
            continue
        paired_inputs[(example.feature_name, treatment, outcome)] = (
            example.derivation_pseudocode,
            dataset_context,
        )

    missing_in_compile_set: list[tuple[str, str, str, str]] = []
    for feature_name, treatment, outcome, _expected_role in EXPECTED_TREATMENT_OUTCOME_QUADRUPLES:
        if (feature_name, treatment, outcome) not in paired_inputs:
            missing_in_compile_set.append((feature_name, treatment, outcome, "?"))

    assert not missing_in_compile_set, (
        f"Stage 2 test fixture and build_compile_set() are out of sync. "
        f"Quadruples missing from compile set: {missing_in_compile_set}. "
        f"This is an upstream defect (Stage 1 unit test should also be "
        f"failing) — fix build_compile_set() to add the missing paired demo, "
        f"then recompile, then re-run Stage 2."
    )

    # Configure DSPy LM with caching DISABLED for this test (codex iter-0
    # MED finding). DSPy's `dspy.LM(model)` defaults to `cache=True` —
    # subsequent calls with the same prompt replay the in-memory cache and
    # bypass provider dispatch. For a manual pre-PR live-LM smoke we want
    # FRESH provider responses on every run so the gate covers:
    # (a) provider reachability, (b) API key validity, (c) model
    # availability, and (d) current LM behavior on the paired fixtures.
    # Cached-replay would let a deprecated model or revoked key sneak past.
    #
    # We bypass ``ensure_dspy_lm_configured`` (which uses cache=True) but
    # mirror its credential gating: explicit ANTHROPIC_API_KEY check above
    # already covers (b). Provider-prefix correctness is verified by the
    # `dspy.configure` call succeeding (LiteLLM raises on bad provider).
    model_string = "anthropic/claude-sonnet-4-6"
    dspy.configure(lm=dspy.LM(model_string, cache=False))
    # Sanity-confirm configuration took effect; ``ensure_dspy_lm_configured``
    # is idempotent so calling it now without re-configuring lets us reuse
    # its post-config validity check.
    configured = ensure_dspy_lm_configured(require_api_key=True)
    assert configured, (
        "ensure_dspy_lm_configured returned False — the provider-aware "
        "credential gate refused to confirm the DSPy LM configured above. "
        "Check the model string's provider prefix matches the env-var "
        "convention in "
        "`src/data/causal_role_classifier_loader.py::_PROVIDER_TO_ENV_VARS`."
    )

    # Load the compiled classifier once so all 12 invocations share the
    # same few-shot context. classify_feature can lazy-load per call but
    # that would multiply load time across invocations.
    classifier = load_compiled_classifier()
    assert classifier is not None, (
        "load_compiled_classifier returned None — the persisted artifact at "
        "`artifacts/dspy/causal_role_classifier.json` is missing or "
        "malformed. Recompile via "
        "`python scripts/compile_causal_role_classifier.py "
        "--lm-model anthropic/claude-sonnet-4-6 --force`."
    )

    results: list[tuple[str, str, str, str, str, bool]] = []
    try:
        for (
            feature_name,
            treatment,
            outcome,
            expected_role,
        ) in EXPECTED_TREATMENT_OUTCOME_QUADRUPLES:
            derivation_pseudocode, dataset_context = paired_inputs[
                (feature_name, treatment, outcome)
            ]
            # Codex iter-0 LOW: catch per-case exceptions so a single
            # network/parse failure does not abort the loop and obscure
            # the remaining cases. Record the exception type+message as
            # the observed value with match=False so the per-quadruple
            # table surfaces the exception inline with wrong-label
            # failures.
            observed_role: str
            try:
                verdict = classify_feature(
                    feature_name=feature_name,
                    derivation_pseudocode=derivation_pseudocode,
                    dataset_context=dataset_context,
                    classifier=classifier,
                )
                observed_role = verdict.causal_role if verdict is not None else "<None>"
            except Exception as exc:  # pragma: no cover — defensive; covered by failure path
                observed_role = f"<EXCEPTION: {type(exc).__name__}: {exc}>"
            match = observed_role == expected_role
            results.append((feature_name, treatment, outcome, expected_role, observed_role, match))
    finally:
        # Restore DSPy LM to None so subsequent tests in the same pytest
        # session don't accidentally inherit a configured LM. Mirrors the
        # cleanup pattern in `test_synthetic_borderline_genuine_hblp_contrast.py`.
        dspy.settings.configure(lm=None)

    pass_count = sum(1 for _, _, _, _, _, match in results if match)
    failures = [
        (feature_name, treatment, outcome, expected, observed)
        for feature_name, treatment, outcome, expected, observed, match in results
        if not match
    ]

    # Detailed per-quadruple table in the assertion message so failure
    # output gives the operator everything needed to attribute + decide.
    detail_rows = "\n".join(
        f"  {'PASS' if match else 'FAIL'} "
        f"{feature_name}, T={treatment}, Y={outcome}, "
        f"expected={expected!r}, observed={observed!r}"
        for feature_name, treatment, outcome, expected, observed, match in results
    )

    assert pass_count >= STAGE_2_PASS_THRESHOLD, (
        f"Stage 2 live-LM smoke: only {pass_count}/12 paired fixtures "
        f"returned the expected role; threshold is "
        f"{STAGE_2_PASS_THRESHOLD}/12 per Option C plan §3.5 Stage 2 + §4.2.\n"
        f"Failures ({len(failures)}/12):\n"
        + "\n".join(
            f"  - {fn} (T={t}, Y={y}): expected {e!r}, observed {o!r}"
            for fn, t, y, e, o in failures
        )
        + "\n\n"
        f"Full per-quadruple table:\n{detail_rows}\n\n"
        f"Recovery options (per Option C plan §5 risk register):\n"
        f"  (a) Accept the LM's revised role if domain-defensible "
        f"(update the quadruple in EXPECTED_TREATMENT_OUTCOME_QUADRUPLES + "
        f"build_compile_set() + recompile);\n"
        f"  (b) Swap the disputed pair for one with a less-contestable "
        f"role split per plan §5 row 2 recovery procedure;\n"
        f"  (c) Strengthen the disagreeing demo's mechanism rationale "
        f"and recompile (most useful if the LM is hovering near the "
        f"role boundary)."
    )
