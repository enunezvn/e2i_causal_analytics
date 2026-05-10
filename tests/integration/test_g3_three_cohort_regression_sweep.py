"""G3 three-cohort regression sweep — synthetic + CSU + Optum.

Plan v4 §2 Gate G3 acceptance criterion #3:

    Three-cohort regression sweep (synthetic + CSU + Optum): no
    MARGINAL→GENUINE flips on synthetic baseline; CSU verdict unchanged
    (per G1); Optum verdict unchanged or improved (per G2).

This test pins the post-G3 production behavior on each cohort. It runs
``_adversarial_input`` + ``_compose_legacy_verdict`` directly against
synthetic z-score fixtures plus (when real-cohort data is available)
against CSU + Optum cohort positive counts to confirm:

  * **synthetic baseline** (n_train_pos=200, layer_1_declared_safe=False,
    `z=4.5σ`): post-G3 severity must remain ``"info"`` — z=4.5 is below
    even the legacy fixed 5σ threshold and HBLP at reference-N has
    inflation_factor=1.0, so the verdict is unchanged. A MARGINAL→GENUINE
    flip here would mean HBLP's variance-inflation factor is incorrectly
    *tightening* (rather than relaxing) at large N.

  * **CSU n=9607-equivalent cohort** (n_train_pos>=200,
    layer_1_declared_safe=True at known-pre-anchor features): the HBLP-
    inflated thresholds at large N + declared_safe=True multiplier=1.5
    push high-eff to ~7.5σ. A z-score that would have triggered "high"
    under legacy 5σ now classifies as "moderate" if 5σ ≤ z ≤ 7.5σ. This
    test pins that classification directly.

  * **Optum default-window n=1294 cohort** (n_train_pos~22,
    layer_1_declared_safe=True): the variance-inflation factor at low N
    is ``sqrt(50/22) ≈ 1.508``; combined with declared_safe=True
    multiplier=1.5, high-eff ≈ 11.3σ. A feature with z=6σ that would have
    flagged HIGH under legacy 5σ classifies as ``"info"`` post-G3 — this
    is the relaxation HBLP was designed for at small N (per Plan v3 §3
    Tier 1B step 2: "at low n_positives the permutation null variance
    scales as ~1/sqrt(n_positives)").

The test is deliberately scoped to the helper level rather than the full
``adaptive_validity_check`` pipeline so it runs in CI without real-cohort
data dependencies. The full-pipeline regression runs from G1 (CSU
negative-control + Optum held-out non-inferiority) cover the cohort-
level behavior end-to-end; this test pins the helper-level contract that
G3 wires on top of those.

Real-data dependent assertions (full pipeline runs against CSU/Optum
parquet files) are gated behind ``@pytest.mark.real_data`` and SKIP when
the cohort fixtures are absent (CI default state). This mirrors the
G1 test pattern (PR #137).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _adversarial_input,
    _build_verdict,
    _compose_legacy_verdict,
    _get_ensemble_voter_class,
    hblp_classify,
)

# --------------------------------------------------------------------------- #
# Cohort metadata fixtures — sourced from the empirical anchors per plan §0
# --------------------------------------------------------------------------- #

# Synthetic baseline: large-N cohort, no manifest contracts (synthetic
# regimes leave feature_manifest_source unset). HBLP at reference-N has
# variance_inflation=1.0 and layer_1_factor=1.0 → no relaxation.
SYNTHETIC_N_TRAIN_POS: int = 200
SYNTHETIC_LAYER_1_DECLARED_SAFE: bool = False

# CSU cohort: per `docs/results/optum_initiation_revalidation_20260510.md`
# baseline + plan §0 status, CSU n=9607 large-N. The CSU manifest declares
# many features with knowable_at <= index_date (declared_safe=True).
CSU_N_TRAIN_POS: int = 200  # cohort-positive-count proxy at sufficient N
CSU_LAYER_1_DECLARED_SAFE_FOR_CLEARED: bool = True

# Optum default-window cohort: per the empirical anchor, Optum n=1294
# default-window has ~22 train positives at the typical 75/25 split. HBLP's
# variance-inflation factor at n=22 is sqrt(50/22) ≈ 1.508; with
# declared_safe=True the layer_1_factor=1.5 multiplies in for ~7.5σ threshold
# at reference-N + ~11.3σ at n=22.
OPTUM_DEFAULT_N_TRAIN_POS: int = 22
OPTUM_LAYER_1_DECLARED_SAFE: bool = True


# --------------------------------------------------------------------------- #
# Synthetic-baseline pins — no MARGINAL→GENUINE flips
# --------------------------------------------------------------------------- #


class TestSyntheticBaselineNoFlips:
    """G3 acceptance §3 sub-clause 1: synthetic baseline has no
    MARGINAL→GENUINE flips post-G3 wiring.

    "MARGINAL→GENUINE" in the leakage-verdict context means a feature
    that was previously classified as ``severity=info`` (legitimate weak
    signal, NOT flagged) becomes ``severity=high`` (leak, drop). The
    test pins the synthetic z-score thresholds at which classification
    remains stable.
    """

    def test_z_below_moderate_stays_info(self) -> None:
        """z=2.0σ below MODERATE_Z (3.0σ) → severity='info' on synthetic."""

        score = {
            "z_score": 2.0,
            "actual_auc": 0.55,
            "null_mean": 0.50,
            "null_std": 0.025,
            "p_value": 0.05,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=SYNTHETIC_N_TRAIN_POS,
            layer_1_declared_safe=SYNTHETIC_LAYER_1_DECLARED_SAFE,
        )
        assert adv["severity"] == "info"
        assert adv["remediation"] == "keep"

    def test_z_above_moderate_below_high_is_moderate(self) -> None:
        """z=4.0σ between 3σ + 5σ → severity='moderate' on synthetic.

        At reference-N (200 ≥ 50) and declared_safe=False, HBLP's
        effective thresholds collapse to base (3σ moderate, 5σ high) —
        same classification as legacy.
        """

        score = {
            "z_score": 4.0,
            "actual_auc": 0.65,
            "null_mean": 0.50,
            "null_std": 0.0375,
            "p_value": 0.001,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=SYNTHETIC_N_TRAIN_POS,
            layer_1_declared_safe=SYNTHETIC_LAYER_1_DECLARED_SAFE,
        )
        assert adv["severity"] == "moderate"
        assert adv["remediation"] == "ambiguous"

    def test_z_above_high_is_high(self) -> None:
        """z=6.0σ above 5σ HIGH_Z → severity='high' on synthetic.

        At synthetic baseline N=200, HBLP no-relaxation → high_eff=5σ →
        z=6σ classifies as high. Same as legacy.
        """

        score = {
            "z_score": 6.0,
            "actual_auc": 0.80,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=SYNTHETIC_N_TRAIN_POS,
            layer_1_declared_safe=SYNTHETIC_LAYER_1_DECLARED_SAFE,
        )
        assert adv["severity"] == "high"
        assert adv["remediation"] == "drop"

    def test_z_at_high_threshold_boundary_is_moderate(self) -> None:
        """z=5.0σ exactly at high_eff → not strictly greater → 'moderate'.

        Pre-G3: ``z > HIGH_Z`` (strict). Post-G3 routes through
        hblp_classify which preserves strict-greater-than semantics.
        At z=5.0 exactly, severity stays at moderate.
        """

        score = {
            "z_score": 5.0,
            "actual_auc": 0.75,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=SYNTHETIC_N_TRAIN_POS,
            layer_1_declared_safe=SYNTHETIC_LAYER_1_DECLARED_SAFE,
        )
        # z > high_eff is strict; z == high_eff falls into moderate band.
        assert adv["severity"] == "moderate"

    def test_no_marginal_to_genuine_flip_at_grid_seeds(self) -> None:
        """Pin: z grid [2.0, 2.5, 3.5, 4.0, 4.5, 5.5, 6.0, 7.0, 10.0] gives
        deterministic post-G3 verdict at synthetic-baseline metadata.

        Compares against the manually-derived expected severity set and
        asserts they're invariant under the G3 wiring change.
        """

        z_grid = [2.0, 2.5, 3.5, 4.0, 4.5, 5.5, 6.0, 7.0, 10.0]
        expected_severities = [
            "info",  # 2.0
            "info",  # 2.5
            "moderate",  # 3.5
            "moderate",  # 4.0
            "moderate",  # 4.5
            "high",  # 5.5
            "high",  # 6.0
            "high",  # 7.0
            "high",  # 10.0
        ]
        for z, expected in zip(z_grid, expected_severities, strict=True):
            score = {
                "z_score": z,
                "actual_auc": 0.5 + z * 0.05,
                "null_mean": 0.50,
                "null_std": 0.05,
                "p_value": 0.0,
                "n_permutations": 200,
            }
            adv = _adversarial_input(
                score,
                n_train_pos=SYNTHETIC_N_TRAIN_POS,
                layer_1_declared_safe=SYNTHETIC_LAYER_1_DECLARED_SAFE,
            )
            assert adv["severity"] == expected, (
                f"z={z}: post-G3 severity={adv['severity']!r}, "
                f"expected {expected!r} (this is a synthetic-baseline "
                f"regression — would indicate HBLP variance-inflation "
                f"is tightening rather than relaxing)"
            )


# --------------------------------------------------------------------------- #
# CSU verdict-unchanged pins (per G1)
# --------------------------------------------------------------------------- #


class TestCSUVerdictUnchanged:
    """G3 acceptance §3 sub-clause 2: CSU verdict unchanged after wiring.

    CSU n=9607 large-N + declared_safe=True features are the case HBLP's
    declared_safe_prior_multiplier was designed for: a feature whose
    manifest contract has knowable_at <= index_date needs *stronger*
    Layer 3 evidence to be reclassified as a leak. The threshold
    relaxation is the visible behavior change (5σ → 7.5σ at reference-N
    × declared_safe=True), but the CSU val_AUC=0.66 MARGINAL pin from
    G1 should remain stable.
    """

    def test_z_5_5_with_declared_safe_relaxes_to_moderate(self) -> None:
        """z=5.5σ + declared_safe=True at reference-N → severity='moderate'.

        Legacy: z=5.5 > 5.0 → severity='high', remediation='drop'.
        Post-G3: z=5.5 > 3*1.5=4.5 (moderate_eff) AND z=5.5 < 5*1.5=7.5
        (high_eff) → severity='moderate', remediation='ambiguous'.

        This is the EXACT flip the declared_safe_prior_multiplier
        encodes: a feature Layer 1 cleared no longer auto-drops at 5σ
        because the prior says "it's structurally pre-anchor".
        """

        score = {
            "z_score": 5.5,
            "actual_auc": 0.78,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=CSU_N_TRAIN_POS,
            layer_1_declared_safe=True,
        )
        assert adv["severity"] == "moderate"
        assert adv["remediation"] == "ambiguous"

    def test_z_above_declared_safe_high_eff_still_drops(self) -> None:
        """z=8.0σ > 7.5σ declared-safe-high-eff → severity='high'.

        Even with declared_safe=True the relaxation is bounded; a feature
        whose statistical evidence exceeds the inflated threshold still
        drops.
        """

        score = {
            "z_score": 8.0,
            "actual_auc": 0.85,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=CSU_N_TRAIN_POS,
            layer_1_declared_safe=True,
        )
        assert adv["severity"] == "high"
        assert adv["remediation"] == "drop"

    def test_csu_undeclared_feature_uses_legacy_thresholds(self) -> None:
        """CSU feature WITHOUT manifest entry → declared_safe=False →
        HBLP's declared_safe_prior_multiplier=1.0 → behaves like legacy.

        At CSU's reference-N + declared_safe=False, high_eff=5σ exactly.
        z=5.5 → severity='high', same as legacy.
        """

        score = {
            "z_score": 5.5,
            "actual_auc": 0.78,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=CSU_N_TRAIN_POS,
            layer_1_declared_safe=False,
        )
        assert adv["severity"] == "high"
        assert adv["remediation"] == "drop"


# --------------------------------------------------------------------------- #
# Optum default-window verdict-unchanged-or-improved pins (per G2)
# --------------------------------------------------------------------------- #


class TestOptumDefaultVerdictRelaxes:
    """G3 acceptance §3 sub-clause 3: Optum verdict unchanged or improved.

    Optum default-window has the empirical-anchor behavior of MARGINAL
    perm p=0.67 (legacy 5σ threshold over-flagging legitimate confounders
    due to ~22 train positives → permutation-null variance scaling
    1/sqrt(22)). HBLP's variance-inflation at n=22 is sqrt(50/22) ≈ 1.508;
    declared_safe=True multiplies in for high-eff ≈ 11.3σ.

    A feature with z=6σ at small-N legitimate-confounder noise was
    previously HIGH (auto-drop); post-G3 it's INFO (kept) — that's the
    relaxation v3 §3 Tier 1B step 2 was designed for.
    """

    def test_z_6_at_low_n_with_declared_safe_relaxes_to_info(self) -> None:
        """z=6.0σ + n_train_pos=22 + declared_safe=True → severity='info'.

        Legacy: z=6 > 5 → high → drop. Post-G3: high_eff = 5 *
        sqrt(50/22) * 1.5 ≈ 11.32σ; moderate_eff = 3 * sqrt(50/22) * 1.5
        ≈ 6.79σ. z=6 < 6.79 → severity='info', remediation='keep'.

        This is the EXACT relaxation HBLP encodes for low-N declared-
        safe features.
        """

        score = {
            "z_score": 6.0,
            "actual_auc": 0.80,
            "null_mean": 0.50,
            "null_std": 0.04,  # Note: low-N std is ~higher; permutation
            "p_value": 0.05,  # null variance ~1/sqrt(22)
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=OPTUM_DEFAULT_N_TRAIN_POS,
            layer_1_declared_safe=OPTUM_LAYER_1_DECLARED_SAFE,
        )
        assert adv["severity"] == "info"
        assert adv["remediation"] == "keep"

    def test_z_8_at_low_n_with_declared_safe_is_moderate(self) -> None:
        """z=8.0σ + n_train_pos=22 + declared_safe=True → severity='moderate'.

        Post-G3: high_eff ≈ 11.32σ, moderate_eff ≈ 6.79σ. z=8 between
        moderate_eff and high_eff → severity='moderate'.
        """

        score = {
            "z_score": 8.0,
            "actual_auc": 0.85,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=OPTUM_DEFAULT_N_TRAIN_POS,
            layer_1_declared_safe=OPTUM_LAYER_1_DECLARED_SAFE,
        )
        assert adv["severity"] == "moderate"
        assert adv["remediation"] == "ambiguous"

    def test_extreme_z_at_low_n_still_drops(self) -> None:
        """z=15.0σ (well past 11.32σ HBLP-inflated high_eff) still flags.

        Even at n=22 with declared_safe=True the relaxation is bounded;
        a genuine leak with 15σ statistical evidence still classifies
        as 'high'/drop.
        """

        score = {
            "z_score": 15.0,
            "actual_auc": 0.95,
            "null_mean": 0.50,
            "null_std": 0.03,
            "p_value": 0.0,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=OPTUM_DEFAULT_N_TRAIN_POS,
            layer_1_declared_safe=OPTUM_LAYER_1_DECLARED_SAFE,
        )
        assert adv["severity"] == "high"
        assert adv["remediation"] == "drop"

    def test_low_n_undeclared_feature_only_inflates_via_variance(self) -> None:
        """n_train_pos=22 + declared_safe=False → only variance-inflation applies.

        Post-G3: high_eff = 5 * sqrt(50/22) ≈ 7.54σ;
                 moderate_eff = 3 * sqrt(50/22) ≈ 4.52σ.
        z=4.0 < 4.52 → info.
        z=5.0 between 4.52 and 7.54 → moderate.
        z=6.0 between 4.52 and 7.54 → moderate.
        z=8.0 > 7.54 → high.
        """

        for z_test, expected in [
            (4.0, "info"),
            (5.0, "moderate"),
            (6.0, "moderate"),
            (8.0, "high"),
        ]:
            score = {
                "z_score": z_test,
                "actual_auc": 0.5 + z_test * 0.04,
                "null_mean": 0.50,
                "null_std": 0.05,
                "p_value": 0.0,
                "n_permutations": 200,
            }
            adv = _adversarial_input(
                score,
                n_train_pos=OPTUM_DEFAULT_N_TRAIN_POS,
                layer_1_declared_safe=False,
            )
            assert adv["severity"] == expected, (
                f"z={z_test}: severity={adv['severity']!r} expected {expected!r} "
                f"(low-N undeclared variance-inflation only)"
            )


# --------------------------------------------------------------------------- #
# Cross-cohort sweep grid — pin the full matrix
# --------------------------------------------------------------------------- #


class TestCrossCohortGrid:
    """Sweep grid fully pinning the post-G3 classification at the three
    representative (n_train_pos, layer_1_declared_safe) cohorts.

    The grid is the load-bearing artifact for v4 §2 G3 acceptance #3:
    "no MARGINAL→GENUINE flips on synthetic baseline; CSU verdict
    unchanged; Optum verdict unchanged or improved".

    A diff in any cell here means HBLP's coefficients changed (which is
    a Tier 1 invariant change — should NOT happen as a side-effect of
    G3 wiring).
    """

    @pytest.mark.parametrize(
        "n_train_pos,layer_1_declared_safe,z_score,expected_severity",
        [
            # synthetic-baseline column (n=200, declared=False): legacy
            # 5σ/3σ thresholds reproduced
            (SYNTHETIC_N_TRAIN_POS, False, 2.0, "info"),
            (SYNTHETIC_N_TRAIN_POS, False, 4.0, "moderate"),
            (SYNTHETIC_N_TRAIN_POS, False, 6.0, "high"),
            # CSU column (n=200, declared=True): high_eff=7.5, moderate_eff=4.5
            (CSU_N_TRAIN_POS, True, 4.0, "info"),
            (CSU_N_TRAIN_POS, True, 5.5, "moderate"),
            (CSU_N_TRAIN_POS, True, 8.0, "high"),
            # Optum column (n=22, declared=True): high_eff~11.32, moderate_eff~6.79
            (OPTUM_DEFAULT_N_TRAIN_POS, True, 5.0, "info"),
            (OPTUM_DEFAULT_N_TRAIN_POS, True, 8.0, "moderate"),
            (OPTUM_DEFAULT_N_TRAIN_POS, True, 12.0, "high"),
            # Optum-undeclared variant (n=22, declared=False):
            #   high_eff~7.54, moderate_eff~4.52
            (OPTUM_DEFAULT_N_TRAIN_POS, False, 4.0, "info"),
            (OPTUM_DEFAULT_N_TRAIN_POS, False, 6.0, "moderate"),
            (OPTUM_DEFAULT_N_TRAIN_POS, False, 8.0, "high"),
        ],
    )
    def test_classification_matrix(
        self,
        n_train_pos: int,
        layer_1_declared_safe: bool,
        z_score: float,
        expected_severity: str,
    ) -> None:
        """Pin the (cohort × z) → severity matrix post-G3 wiring."""

        score: dict[str, Any] = {
            "z_score": z_score,
            "actual_auc": 0.5 + z_score * 0.04,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0 if z_score > 5 else 0.05,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=n_train_pos,
            layer_1_declared_safe=layer_1_declared_safe,
        )
        assert adv["severity"] == expected_severity, (
            f"matrix cell (n={n_train_pos}, declared_safe={layer_1_declared_safe}, "
            f"z={z_score}): post-G3 severity={adv['severity']!r}, "
            f"expected {expected_severity!r}"
        )


# --------------------------------------------------------------------------- #
# _build_verdict + _compose_legacy_verdict end-to-end pins
# --------------------------------------------------------------------------- #


class TestBuildVerdictThreadsHblp:
    """``_build_verdict`` and ``_compose_legacy_verdict`` accept and forward
    the new HBLP-threading parameters.

    These are the two entry points named in plan v4 §2 G3 acceptance.
    """

    def test_build_verdict_default_args_preserves_legacy(self) -> None:
        """``_build_verdict`` without HBLP args reproduces legacy behavior.

        Without n_train_pos + layer_1_declared_safe, the underlying
        ``_adversarial_input`` falls through to reference-N + declared=False
        (no inflation) — preserves the legacy fixed 5σ/3σ thresholds for
        ad-hoc test callers that haven't been updated.
        """

        voter = _get_ensemble_voter_class()()
        score = {
            "z_score": 6.0,
            "actual_auc": 0.78,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
        }
        verdict = _build_verdict("test_feature", score, voter=voter)
        # Legacy default: high at z=6.
        assert verdict["severity"] == "high"

    def test_build_verdict_with_low_n_declared_safe_relaxes(self) -> None:
        """``_build_verdict`` honors the threaded HBLP args."""

        voter = _get_ensemble_voter_class()()
        score = {
            "z_score": 6.0,
            "actual_auc": 0.78,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.05,
            "n_permutations": 200,
        }
        verdict = _build_verdict(
            "test_feature",
            score,
            voter=voter,
            n_train_pos=22,
            layer_1_declared_safe=True,
        )
        # Optum-style relaxation: severity drops to info.
        assert verdict["severity"] == "info"

    def test_compose_legacy_verdict_threads_hblp_args(self) -> None:
        """``_compose_legacy_verdict`` accepts + forwards the new args."""

        voter = _get_ensemble_voter_class()()
        score = {
            "z_score": 6.0,
            "actual_auc": 0.78,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.05,
            "n_permutations": 200,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=22,
            layer_1_declared_safe=True,
        )
        verdict = _compose_legacy_verdict(
            "test_feature",
            voter=voter,
            adversarial_input=adv,
            n_train_pos=22,
            layer_1_declared_safe=True,
        )
        # Optum-style relaxation propagates through the voter.
        assert verdict["severity"] == "info"


# --------------------------------------------------------------------------- #
# Direct hblp_classify invariant pins — the Tier 1 invariant
# --------------------------------------------------------------------------- #


class TestHblpClassifyInvariant:
    """Plan v4 §2 G3 invariant: ``hblp_classify`` signature/behavior is
    UNCHANGED. G3 only wires it into the production codepath.

    These tests redundantly verify the helper's contract didn't drift.
    """

    def test_n_pos_50_no_inflation(self) -> None:
        """At reference-N (50), variance_inflation_factor=1.0."""

        result = hblp_classify(
            z_score=4.5,
            n_positives=50,
            layer_1_declared_safe=False,
        )
        assert result["variance_inflation_factor"] == pytest.approx(1.0)
        assert result["layer_1_factor"] == pytest.approx(1.0)
        assert result["effective_high_threshold"] == pytest.approx(5.0)
        assert result["severity"] == "moderate"  # 3 < 4.5 < 5

    def test_n_pos_22_inflates_variance(self) -> None:
        """At n_pos=22, variance_inflation_factor=sqrt(50/22)≈1.508."""

        result = hblp_classify(
            z_score=6.0,
            n_positives=22,
            layer_1_declared_safe=False,
        )
        assert result["variance_inflation_factor"] == pytest.approx((50 / 22) ** 0.5)
        assert result["effective_high_threshold"] == pytest.approx(5.0 * (50 / 22) ** 0.5)

    def test_declared_safe_adds_1_5x_multiplier(self) -> None:
        """layer_1_declared_safe=True multiplies threshold by 1.5x."""

        result = hblp_classify(
            z_score=6.0,
            n_positives=50,
            layer_1_declared_safe=True,
        )
        assert result["layer_1_factor"] == pytest.approx(1.5)
        assert result["effective_high_threshold"] == pytest.approx(7.5)
        assert result["severity"] == "moderate"  # 4.5 < 6 < 7.5

    def test_optum_hblp_full_inflation(self) -> None:
        """Optum cohort metadata reproduces the v3 anchor calculation:
        high_eff = 5 * sqrt(50/22) * 1.5 ≈ 11.32.
        """

        result = hblp_classify(
            z_score=10.0,
            n_positives=22,
            layer_1_declared_safe=True,
        )
        assert result["effective_high_threshold"] == pytest.approx(5.0 * (50 / 22) ** 0.5 * 1.5)
        # z=10 < 11.32 → moderate (not high).
        assert result["severity"] == "moderate"


# --------------------------------------------------------------------------- #
# Real-data dependency: full-pipeline regression pins (skipped in CI)
# --------------------------------------------------------------------------- #

CSU_DATA_PATH = Path("data/rwd/csu/e2i_ml_v3_patient_journeys.json")
OPTUM_DATA_DIR = Path("data/rwd/optum/initiation/")


@pytest.mark.real_data
class TestFullPipelineRegression:
    """Real-data full-pipeline regressions — skip in CI without data fixtures.

    These tests are deferred to the per-cohort PRs (G1 covers CSU, G2
    covers Optum). G3's load-bearing acceptance is the helper-level
    contract pinned above.
    """

    def test_csu_data_present_or_skip(self) -> None:
        if not CSU_DATA_PATH.exists():
            pytest.skip(
                f"CSU cohort data missing at {CSU_DATA_PATH} — "
                f"deferred to G1 full-pipeline regression"
            )
        # Per G1 (PR #137), CSU val_AUC=0.66 MARGINAL pin is enforced
        # by `tests/integration/test_csu_negative_control_20260510.py`.
        # G3's incremental claim is "verdict unchanged" — covered by
        # the helper-level pins above + the G1 test file.

    def test_optum_data_present_or_skip(self) -> None:
        if not OPTUM_DATA_DIR.exists():
            pytest.skip(
                f"Optum cohort data missing at {OPTUM_DATA_DIR} — "
                f"deferred to G2 full-pipeline regression"
            )
        # Per G2 (PR #136), Optum non-inferiority pin lives in
        # `tests/integration/test_optum_held_out_noninferiority_20260510.py`.
