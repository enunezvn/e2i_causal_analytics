"""S12a synthetic golden-set integration test (TDD red-first).

Plan: ``.claude/plans/s12_synthetic_golden_set_plan.md`` §5 + §6.2.

Replays the synthetic golden-set (``tests/fixtures/causal_role_golden_set_synthetic.json``)
through the compiled ``CausalRoleClassifier`` (artifact at
``artifacts/dspy/causal_role_classifier.json`` — last recompiled in PR #371,
``84c7adbc``) and asserts the Tier 1 unconditional sanity invariants on
Family A (cohort-only `dataset_context`) per the two-tier integration
gate design.

**Tier 1 (unconditional)** fires regardless of threshold-locked state:

- Family A non-empty (``>= 30`` entries).
- All Family A predictions parse to a known role value; ``None`` is a
  FAILED prediction and Tier 1 rejects it.
- ``cohort_only_macro_f1`` is finite and in ``(0.0, 1.0]``.

**Tier 2 (threshold-gated)** lands in a follow-up PR; the placeholder
``MACRO_F1_THRESHOLD = None`` skips the assertion until the
repeat-measurement protocol seals the value. Family B (T,Y)-explicit
re-emissions are recorded as informational only.

Run manually pre-PR with ``ANTHROPIC_API_KEY`` in env::

    pytest tests/integration/test_causal_role_classifier_golden_set.py \\
        -v -m integration --no-header

Cost: ~$2.96 / ~9.7 min wall (74 calls × $0.04 × 7.9s/call) anchored to
Option C Stage 2's measured run. Timeout 1200s accommodates 1.5× retry
multiplier + fixture/setup overhead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pytest

# Placeholder sentinel per plan §5 / §7 / §8: while ``None`` the test
# SKIPS the threshold assertion (Tier 2). Tier 1 still fires.
# Locked in a follow-up PR after the pre-registered repeat-measurement
# protocol (≥5 runs, σ-observed, threshold = floor(mean - 2·σ)).
MACRO_F1_THRESHOLD: Optional[float] = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "causal_role_golden_set_synthetic.json"


VALID_ROLES = frozenset(
    {"ancestor", "confounder", "mediator", "collider", "descendant", "instrument"}
)
GRAPH_READABLE_ROLES = frozenset({"ancestor", "confounder", "mediator", "collider", "descendant"})


def _macro_f1(
    results: list[tuple[str, str, str, str]],
    roles: frozenset[str],
) -> float:
    """Compute macro-averaged F1 over the given roles.

    Each result is ``(scenario, feature_name, expected_role, observed_role)``.
    Macro averaging weights each class equally (per plan §6.1 macro choice).
    """
    f1_per_role: list[float] = []
    for role in sorted(roles):
        tp = sum(1 for _, _, e, o in results if e == role and o == role)
        fp = sum(1 for _, _, e, o in results if e != role and o == role)
        fn = sum(1 for _, _, e, o in results if e == role and o != role)
        if tp == 0 and fp == 0 and fn == 0:
            # Class absent from both expected and observed — skip rather
            # than penalize (a fair macro-averaging convention).
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_per_role.append(f1)
    if not f1_per_role:
        return 0.0
    return sum(f1_per_role) / len(f1_per_role)


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_classifier_passes_tier_1_sanity_on_golden_set() -> None:
    """Replay the synthetic golden set through the compiled classifier and
    assert Tier 1 unconditional sanity invariants on Family A (cohort-only).

    Tier 2 (threshold assertion) is gated on ``MACRO_F1_THRESHOLD``; while
    that placeholder is ``None`` the threshold check skips. Family B
    measurements are logged as informational.
    """
    # Skip when no real Anthropic key is present (per
    # feedback-live-lm-skip-must-check-key-shape memory — `'test-key'`
    # placeholder in CI and `'sk-test'` placeholder in unit tests both
    # fail the prefix check).
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key.startswith("sk-ant-"):
        pytest.skip(
            "ANTHROPIC_API_KEY missing or not a real Anthropic key "
            f"(expected `sk-ant-` prefix; got "
            f"{api_key[:7] + '…' if api_key else '<empty>'!r}). "
            "S12a golden-set replay requires a live Anthropic key — run "
            "locally with `.env` loaded before invoking pytest. CI "
            "environments with `ANTHROPIC_API_KEY=test-key` will skip "
            "this test as designed."
        )

    # Load the fixture (must already exist; the unit test fixture-pin
    # gate is responsible for ensuring it matches the scenario builders).
    assert FIXTURE_PATH.exists(), (
        f"golden-set fixture missing at {FIXTURE_PATH}; "
        f"run `python scripts/build_causal_role_golden_set.py` first"
    )
    golden = json.loads(FIXTURE_PATH.read_text())

    # Local imports so import-time costs don't fire when the test is
    # skipped (DSPy + compiled artifact are heavyweight).
    import dspy

    from src.data.causal_role_classifier_loader import (
        classify_feature,
        ensure_dspy_lm_configured,
        load_compiled_classifier,
    )

    # Configure DSPy LM with caching DISABLED (Option C iter-0 MED
    # finding: cached replay defeats live-LM signal).
    model_string = "anthropic/claude-sonnet-4-6"
    dspy.configure(lm=dspy.LM(model_string, cache=False))
    configured = ensure_dspy_lm_configured(require_api_key=True)
    assert configured, (
        "ensure_dspy_lm_configured returned False — provider gating refused "
        "the configured LM. Check the model string's provider prefix."
    )

    # Load the compiled classifier once.
    classifier = load_compiled_classifier()
    assert classifier is not None, (
        "load_compiled_classifier returned None — recompile via "
        "`python scripts/compile_causal_role_classifier.py "
        "--lm-model anthropic/claude-sonnet-4-6 --force`."
    )

    family_a: list[tuple[str, str, str, str]] = []  # (scenario, feature, expected, observed)
    family_b: list[tuple[str, str, str, str]] = []
    try:
        for entry in golden["entries"]:
            try:
                verdict = classify_feature(
                    feature_name=entry["feature_name"],
                    derivation_pseudocode=entry["derivation_pseudocode"],
                    dataset_context=entry["dataset_context"],
                    classifier=classifier,
                )
                observed = verdict.causal_role if verdict is not None else "<None>"
            except Exception as exc:  # pragma: no cover — defensive
                observed = f"<EXCEPTION: {type(exc).__name__}: {exc}>"
            row = (
                entry["scenario"],
                entry["feature_name"],
                entry["ground_truth_role"],
                observed,
            )
            if entry["treatment_explicit"]:
                family_b.append(row)
            else:
                family_a.append(row)
    finally:
        dspy.settings.configure(lm=None)

    # ---------------- Tier 1 (unconditional sanity) ----------------
    assert len(family_a) >= 30, (
        f"Family A (cohort-only) result set too small: {len(family_a)} < 30. "
        f"Either the golden-set fixture is malformed or the classifier "
        f"refused most inputs."
    )

    # All Family A predictions must parse to a known role value.
    # `None` / `<None>` / exception-strings all fail Tier 1.
    bad_predictions = [
        (scenario, feat, exp, obs)
        for scenario, feat, exp, obs in family_a
        if obs not in VALID_ROLES
    ]
    assert not bad_predictions, (
        f"Family A has {len(bad_predictions)} predictions that failed to "
        f"parse to a known role (None/exception/unknown). Tier 1 rejects "
        f"these as failed classifications.\n"
        + "\n".join(
            f"  - {s} / {f}: expected={e!r}, observed={o!r}" for s, f, e, o in bad_predictions[:10]
        )
        + (f"\n  ... (+{len(bad_predictions) - 10} more)" if len(bad_predictions) > 10 else "")
    )

    cohort_only_macro_f1 = _macro_f1(family_a, GRAPH_READABLE_ROLES)
    assert 0.0 <= cohort_only_macro_f1 <= 1.0, cohort_only_macro_f1
    assert cohort_only_macro_f1 > 0.0, (
        "cohort_only_macro_f1 == 0.0 — complete zero-score collapse on "
        "Family A. The classifier produced parseable role labels but "
        "none of them matched the ground-truth role on any class. "
        "Investigate the compiled artifact + scenario authoring."
    )

    # ---------------- Tier 2 (threshold-gated) ----------------
    if MACRO_F1_THRESHOLD is not None:
        assert cohort_only_macro_f1 >= MACRO_F1_THRESHOLD, (
            f"cohort_only_macro_f1 = {cohort_only_macro_f1:.4f} below "
            f"threshold {MACRO_F1_THRESHOLD}. See plan §5 Tier 2."
        )

    # ---------------- Informational (Family B + instrument) ----------------
    # Family B is recorded but does NOT gate. Compute the same macro F1
    # for visibility in the test output.
    treatment_explicit_macro_f1 = _macro_f1(family_b, GRAPH_READABLE_ROLES) if family_b else None

    # Instrument precision/recall — informational only (synthetic
    # instruments don't test the domain-judgment half of the classifier
    # per feasibility doc §3).
    iv_tp = sum(1 for _, _, e, o in family_a if e == "instrument" and o == "instrument")
    iv_fp = sum(1 for _, _, e, o in family_a if e != "instrument" and o == "instrument")
    iv_fn = sum(1 for _, _, e, o in family_a if e == "instrument" and o != "instrument")
    iv_precision = iv_tp / (iv_tp + iv_fp) if (iv_tp + iv_fp) > 0 else 0.0
    iv_recall = iv_tp / (iv_tp + iv_fn) if (iv_tp + iv_fn) > 0 else 0.0

    print(
        "\nS12a golden-set replay results:\n"
        f"  Family A (cohort-only, GATED):       macro_f1 = {cohort_only_macro_f1:.4f} "
        f"(N={len(family_a)})\n"
        f"  Family B ((T,Y)-explicit, INFO):     macro_f1 = "
        f"{treatment_explicit_macro_f1:.4f} (N={len(family_b)})\n"
        if treatment_explicit_macro_f1 is not None
        else "  Family B ((T,Y)-explicit, INFO):     <no entries>\n"
    )
    print(
        f"  Instrument (Family A, INFO):          precision={iv_precision:.4f}, "
        f"recall={iv_recall:.4f}\n"
        f"  Tier 2 threshold:                     "
        f"{'SKIPPED (MACRO_F1_THRESHOLD=None)' if MACRO_F1_THRESHOLD is None else MACRO_F1_THRESHOLD}"
    )
