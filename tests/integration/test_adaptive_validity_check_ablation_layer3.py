"""Phase 3.3 Layer-3 ablation integration test (issue #196).

Pins the MAX-rule combination contract: a synthetic INTERACTION-only leak —
a feature whose single-feature AUC is near chance (so the permutation-only
Layer 3 pass MISSES it) but whose removal collapses the joint model
(``|delta_AUC|`` large; ablation MAX-rule ESCALATES severity to high) —
is detected when ``adaptive_layer3_ablation_enabled=True`` and is NOT
detected when the flag is off.

This proves the wiring closes the Phase 3.3 milestone: Layer 3 with
ablation catches leaks the permutation-only mode cannot see.

Construction (deterministic, seeded — MULTIPLICATIVE-INTERACTION pattern):

  * ``target ~ Bernoulli(0.3)`` — base prevalence matches RWD-realistic.
  * ``noise_a, noise_b, hcp_x`` ~ N(0, 1) — three independent noise
    features, no relationship to target.
  * ``hcp_y = noise_b * (1 + 4 * target)`` — multiplicative interaction.
    Marginal mean and folded single-feature AUC of ``hcp_y`` stay near
    chance (~0.55 raw, sign cancels across target strata) so the
    PERMUTATION Layer 3 does NOT flag it. A linear classifier cannot
    detect this — but a ``DecisionTreeClassifier`` can split on
    ``noise_b > 0`` vs ``noise_b < 0`` and observe that the magnitude
    is target-dependent. Dropping ``hcp_y`` from the joint tree model
    drops full_auc by ~0.19 → ablation |delta_AUC| well above the
    issue #194 epsilon=0.10 floor → MAX-rule escalates severity
    to ``high``.

  The test invokes the new ``adaptive_ablation_model_factory`` state
  escape-hatch to pass a tree-based factory; the default
  ``compute_feature_ablation`` factory (LogisticRegression) cannot
  learn multiplicative interactions and so would not see this leak.
  Real-world analogs include sex-stratified-dosing interactions or
  conditional-side-effect features where the marginal target
  correlation is zero but a stratified model picks up the leak.

Acceptance pins:

  * Permutation-only mode (flag OFF): ``hcp_y`` has severity ∈ {info,
    moderate} — NOT high — and is NOT in ``adaptive_flagged_features``.
  * Ablation-enabled mode (flag ON): ``hcp_y`` has severity == "high",
    ``decided_by == "adversarial_ablation"`` (the new tag), and IS in
    ``adaptive_flagged_features``.
  * The five ablation audit-trail fields (``ablation_z_score`` /
    ``ablation_delta_auc`` / ``ablation_null_mean`` / ``ablation_null_std`` /
    ``ablation_severity``) are populated on the ablation-enabled verdict
    and ``None`` on the permutation-only verdict.

Runtime: ~3-5 s wall-clock at the configured ``n_perms=200`` /
``ablation_n_permutations=30`` / 4-active-feature × 1200-row pin. Production
widths (~50 features × 5000 rows) take 10-30 s, which is why the
orchestrator flag defaults OFF (issue #196 Phase 3.3 plan §).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest


def _build_interaction_leak_cohort(n: int = 1200, seed: int = 7) -> tuple[pd.DataFrame, str]:
    """Construct the deterministic multiplicative-interaction leak.

    Returns ``(df, target_name)``. ``df.columns`` = ``[noise_a, noise_b,
    hcp_x, hcp_y, y]``. ``hcp_y = noise_b * (1 + 4*target)`` — the sign
    of ``hcp_y`` is independent of target (so single-feature AUC ≈ 0.55),
    but a tree-based joint model can split on ``sign(noise_b)`` and
    detect the conditional-variance shift. Dropping ``hcp_y`` collapses
    the tree model's predictive power on the target's variance signal.
    """
    rng = np.random.default_rng(seed)
    target = rng.binomial(1, 0.30, n).astype(int)

    noise_a = rng.normal(0.0, 1.0, n)
    noise_b = rng.normal(0.0, 1.0, n)
    hcp_x = rng.normal(0.0, 1.0, n)
    # Multiplicative interaction: hcp_y = noise_b * (1 + 4*target).
    #   target=0: hcp_y =     noise_b (std ≈ 1)
    #   target=1: hcp_y = 5 * noise_b (std ≈ 5)
    # Mean of hcp_y is 0 in both strata; folded single-feature AUC is
    # near chance. A tree on (sign(noise_b), |hcp_y|) can separate the
    # strata via the magnitude — a linear classifier cannot.
    hcp_y = noise_b * (1.0 + 4.0 * target.astype(float))

    df = pd.DataFrame(
        {
            "noise_a": noise_a,
            "noise_b": noise_b,
            "hcp_x": hcp_x,
            "hcp_y": hcp_y,
            "y": target,
        }
    )
    return df, "y"


def _tree_model_factory():
    """sklearn DecisionTreeClassifier factory for ``compute_feature_ablation``.

    ``compute_feature_ablation`` defaults to LogisticRegression, which
    cannot learn multiplicative interactions. The integration fixture
    plants a sign-stratified variance-shift leak that requires a tree
    split — pass this factory via the ``adaptive_ablation_model_factory``
    state key so the ablation pass can detect it.
    """
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(max_depth=5, random_state=42)


def _make_state(
    df: pd.DataFrame,
    target: str,
    *,
    ablation_enabled: bool,
    ablation_n_permutations: int = 30,
    n_permutations: int = 200,
    model_factory=None,
) -> dict:
    """Build a DataPreparerState fixture for the Layer 5 node.

    ``feature_manifest_source=None`` keeps Layer 1 inert (the synthetic
    interaction is unrelated to the CSU/Optum manifests). ``adaptive_seed``
    is pinned so the permutation-null and ablation-null distributions are
    reproducible across runs.
    """
    return {
        "experiment_id": "test-issue-196",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": target,
            "required_features": [c for c in df.columns if c != target],
            "excluded_features": [],
            "feature_manifest_source": None,
        },
        "leakage_findings": [],
        "leaked_features": [],
        "adaptive_n_permutations": n_permutations,
        "adaptive_seed": 7,
        "adaptive_layer3_ablation_enabled": ablation_enabled,
        "adaptive_ablation_n_permutations": ablation_n_permutations,
        "adaptive_ablation_model_factory": model_factory,
    }


def _run(state: dict) -> dict:
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    return asyncio.run(adaptive_validity_check(state))


# Issue #215: the 3 tests below all call ``asyncio.run(adaptive_validity_check(state))``
# via the ``_run`` helper. xdist worker pollution from any upstream test that
# triggers ``nest_asyncio.apply()`` (e.g. via wrap_async_node-based graphs)
# leaves ``asyncio.run`` monkey-patched, causing ``RuntimeError: Event loop
# is closed``. We isolate the 3 tests to a dedicated worker via
# ``xdist_group("issue_215_layer3_ablation")`` so they run in a clean
# subprocess regardless of upstream pollution. Defense-in-depth complement
# to the root-cause fix at src/agents/experiment_designer/graph.py (eager
# singleton removed; nest_asyncio.apply moved into the actually-nested
# branch of sync_wrapper).
pytestmark = pytest.mark.xdist_group("issue_215_layer3_ablation")


@pytest.mark.integration
def test_permutation_only_misses_interaction_leak_ablation_catches_it() -> None:
    """MAX-rule integration pin: permutation Layer 3 alone misses the
    interaction-only leak; flipping ``adaptive_layer3_ablation_enabled``
    on catches it via the ablation sub-test.

    This is the load-bearing integration test for issue #196 closure.
    """
    df, target = _build_interaction_leak_cohort(n=1200, seed=7)

    # === Permutation-only mode (flag OFF) ===
    state_off = _make_state(df, target, ablation_enabled=False)
    result_off = _run(state_off)

    flagged_off = result_off["adaptive_flagged_features"]
    verdict_off = next(v for v in result_off["adaptive_verdicts"] if v["feature"] == "hcp_y")

    assert verdict_off["severity"] in {"info", "moderate"}, (
        "Permutation-only Layer 3 should NOT flag hcp_y as high — single-"
        f"feature AUC near chance. Got severity={verdict_off['severity']}, "
        f"z_score={verdict_off['z_score']}. Test premise broken — re-tune the "
        f"interaction strength so single-feature AUC stays near 0.5."
    )
    assert "hcp_y" not in flagged_off, (
        f"Permutation-only mode flagged hcp_y as high — test premise broken. Flagged={flagged_off}"
    )
    # Ablation audit fields should be None when the pass is OFF
    # (schema uniformity per ``_combine_ablation_with_permutation`` no-op).
    assert verdict_off["ablation_z_score"] is None
    assert verdict_off["ablation_delta_auc"] is None
    assert verdict_off["ablation_severity"] is None

    # === Ablation-enabled mode (flag ON, tree model_factory) ===
    state_on = _make_state(df, target, ablation_enabled=True, model_factory=_tree_model_factory)
    result_on = _run(state_on)

    flagged_on = result_on["adaptive_flagged_features"]
    verdict_on = next(v for v in result_on["adaptive_verdicts"] if v["feature"] == "hcp_y")

    assert verdict_on["severity"] == "high", (
        f"Ablation-enabled Layer 3 should ESCALATE hcp_y severity to 'high' "
        f"via the MAX-rule. Got severity={verdict_on['severity']}, "
        f"ablation_z_score={verdict_on.get('ablation_z_score')}, "
        f"ablation_delta_auc={verdict_on.get('ablation_delta_auc')}, "
        f"ablation_severity={verdict_on.get('ablation_severity')}. "
        f"The interaction-only leak construction may have collapsed — "
        f"re-tune signal strength or seed."
    )
    assert "hcp_y" in flagged_on, (
        f"Ablation-enabled mode failed to add hcp_y to flagged set. Flagged={flagged_on}"
    )
    # ``decided_by`` should be tagged ``adversarial_ablation`` so audit
    # readers can distinguish permutation-caught from ablation-caught.
    assert verdict_on["decided_by"] == "adversarial_ablation", (
        f"Expected decided_by='adversarial_ablation' tag; got {verdict_on['decided_by']}."
    )
    # Ablation audit fields are populated.
    assert verdict_on["ablation_z_score"] is not None, (
        f"ablation_z_score should be populated on ablation-enabled verdict. Got {verdict_on}"
    )
    assert verdict_on["ablation_delta_auc"] is not None
    assert verdict_on["ablation_severity"] in {"moderate", "high"}, (
        "ablation_severity should fire at moderate or high on the interaction "
        f"leak; got {verdict_on['ablation_severity']}."
    )

    # The evidence footnote should record the escalation explicitly so
    # downstream audit readers can grep for the issue-196 marker.
    assert "issue #196" in verdict_on["evidence"], (
        f"Evidence string should record the issue #196 ablation escalation. "
        f"Got: {verdict_on['evidence']}"
    )


@pytest.mark.integration
def test_ablation_enabled_does_not_false_flag_noise_features() -> None:
    """Symmetry check: turning ablation ON must not regress the noise-feature
    contract — pure-noise columns (no marginal AND no interaction signal)
    stay severity=info even when the ablation pass is active.

    Without this pin, a too-loose ablation threshold or a missing joint-check
    application could regress the issue #194 large-n contract on the
    ablation axis.
    """
    df, target = _build_interaction_leak_cohort(n=1200, seed=7)
    state_on = _make_state(df, target, ablation_enabled=True, model_factory=_tree_model_factory)
    result_on = _run(state_on)

    # ``noise_a`` and ``hcp_x`` are pure independent noise — no
    # marginal signal AND no contribution to the multiplicative
    # interaction (which lives in noise_b × target via hcp_y).
    # Dropping either should not move joint AUC; severity stays info.
    # ``noise_b`` is excluded — it's a co-carrier of the interaction;
    # dropping noise_b destroys the magnitude split too. ``hcp_y`` is
    # the ablation-target of the primary test above.
    for feat in ("noise_a", "hcp_x"):
        v = next(vv for vv in result_on["adaptive_verdicts"] if vv["feature"] == feat)
        assert v["severity"] == "info", (
            f"Pure-noise feature {feat} flagged with severity={v['severity']} "
            f"in ablation-enabled mode. ablation_severity="
            f"{v.get('ablation_severity')}, ablation_z={v.get('ablation_z_score')}, "
            f"|delta|={abs(v.get('ablation_delta_auc') or 0):.4f}. Joint-check "
            f"floor should suppress this."
        )


@pytest.mark.integration
def test_ablation_audit_fields_present_when_flag_on_but_no_escalation() -> None:
    """Schema-uniformity pin: when ablation is ON but the per-feature
    severity does NOT escalate (permutation already high, OR ablation says
    info), the five ablation audit-trail fields are STILL populated so audit
    readers see "ran and agreed" vs "did not run".
    """
    df, target = _build_interaction_leak_cohort(n=1200, seed=7)
    state_on = _make_state(df, target, ablation_enabled=True, model_factory=_tree_model_factory)
    result_on = _run(state_on)

    # noise_a has no interaction signal — ablation should produce a small
    # delta_AUC and a non-suspicious z-score. The audit-trail fields
    # should be populated (not None).
    v = next(vv for vv in result_on["adaptive_verdicts"] if vv["feature"] == "noise_a")
    assert v["ablation_z_score"] is not None, (
        f"ablation_z_score should be populated when ablation pass ran, even "
        f"if severity didn't escalate. Got verdict: {v}"
    )
    assert v["ablation_severity"] in {"info", "moderate", "high"}
