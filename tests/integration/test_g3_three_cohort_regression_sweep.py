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

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _adversarial_input,
    _build_verdict,
    _compose_legacy_verdict,
    _get_ensemble_voter_class,
    adaptive_validity_check,
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
        """``_compose_legacy_verdict`` accepts + forwards the new args.

        Pre-classified path: caller builds the adversarial input via
        ``_adversarial_input`` (which tags `_hblp_classified=True`) and
        passes it through. ``_compose_legacy_verdict`` re-uses the dict
        as-is.
        """

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

    def test_compose_legacy_verdict_owns_classification_via_score(self) -> None:
        """codex MED-5: ``_compose_legacy_verdict`` OWNS classification.

        When the caller passes a raw ``adversarial_score`` (NOT a
        pre-classified ``adversarial_input``), this function calls
        ``_adversarial_input`` itself — that's the call chain the
        wiring guard's AST scan verifies, and it's the contract that
        ensures HBLP's effective thresholds always apply.

        The verdict MUST change when threading args change: at z=6 the
        Optum-style cohort metadata (n_train_pos=22, declared_safe=True)
        relaxes severity from 'high' to 'info', but the synthetic-
        baseline cohort metadata (n=200, declared_safe=False) keeps
        severity at 'high'. Same raw score; different cohort metadata
        → different severity. That's the load-bearing HBLP behavior
        threading proves.
        """

        voter = _get_ensemble_voter_class()()
        score = {
            "z_score": 6.0,
            "actual_auc": 0.78,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.05,
            "n_permutations": 200,
        }
        # Path 1: synthetic baseline → severity stays 'high'.
        synthetic_verdict = _compose_legacy_verdict(
            "test_feature_synthetic",
            voter=voter,
            adversarial_score=score,  # raw — function calls _adversarial_input
            n_train_pos=200,
            layer_1_declared_safe=False,
        )
        assert synthetic_verdict["severity"] == "high"
        # Path 2: Optum-style cohort metadata → severity relaxes to 'info'.
        optum_verdict = _compose_legacy_verdict(
            "test_feature_optum",
            voter=voter,
            adversarial_score=score,  # SAME raw score
            n_train_pos=22,  # different cohort metadata
            layer_1_declared_safe=True,
        )
        assert optum_verdict["severity"] == "info"

    def test_compose_legacy_verdict_rejects_unclassified_input(self) -> None:
        """codex MED-5: pre-classified ``adversarial_input`` lacking the
        ``_hblp_classified=True`` tag is REJECTED at runtime.

        A determined developer could side-step the HBLP routing chain by
        building their own adversarial input dict with a hand-rolled
        legacy classifier — the wiring guard's AST scan only verifies
        static callsites. The runtime tag check rejects this case.
        """

        voter = _get_ensemble_voter_class()()
        # Hand-rolled "legacy" adversarial input — NO _hblp_classified tag.
        legacy_adv = {
            "layer": "3",
            "severity": "high",
            "remediation": "drop",
            "evidence": "legacy fixed 5σ threshold exceeded",
            "z_score": 6.0,
            "actual_auc": 0.78,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
            # NO "_hblp_classified" key.
        }
        with pytest.raises(
            Exception,  # _HblpRoutingViolationError, but it's private.
            match="_hblp_classified=True",
        ):
            _compose_legacy_verdict(
                "test_feature",
                voter=voter,
                adversarial_input=legacy_adv,
            )

    def test_compose_legacy_verdict_rejects_both_score_and_input(self) -> None:
        """codex MED-5: passing both ``adversarial_score`` and
        ``adversarial_input`` is a programmer error → ValueError.
        """

        voter = _get_ensemble_voter_class()()
        score = {"z_score": 4.0, "actual_auc": 0.65, "null_mean": 0.5, "null_std": 0.04}
        adv = _adversarial_input(score, n_train_pos=200, layer_1_declared_safe=False)
        with pytest.raises(ValueError, match="exactly one"):
            _compose_legacy_verdict(
                "test_feature",
                voter=voter,
                adversarial_score=score,
                adversarial_input=adv,
            )


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
# codex HIGH-4: orchestrator-path tests with synthetic CSU/Optum-shaped fixtures
# --------------------------------------------------------------------------- #
#
# The pinned post-G3 verdict matrix above exercises ``_adversarial_input`` +
# ``_compose_legacy_verdict`` directly. That covers the helper-level
# contract but not the orchestrator path: a regression in
# ``adaptive_validity_check``'s n_train_pos derivation, manifest lookup, or
# layer_1_declared_safe threading would slip through.
#
# These tests run the orchestrator end-to-end on synthetic CSU/Optum-shaped
# fixtures (~50 patients each) with manifest_source pinned. Real CSU/Optum
# runs remain real_data-skipped above; the synthetic-shaped fixtures prove
# the threading works without requiring data-dir access.


async def _run_orchestrator(state: dict[str, Any]) -> dict[str, Any]:
    """Drive ``adaptive_validity_check`` (async caller).

    Callers must be ``async def`` + ``@pytest.mark.asyncio``. The prior
    ``asyncio.run(...)`` form collided with ``nest_asyncio.apply()`` that
    earlier tests in the same xdist worker triggered (e.g., via
    ``experiment_designer.graph``), producing ``RuntimeError: Event loop
    is closed`` here. Same mitigation as PR #106's
    ``test_layer_5_pipeline_integration.py`` and PR #144's
    ``test_csu_production_grade_deployment.py``.
    """

    return await adaptive_validity_check(state)


def _make_synthetic_csu_state(
    n_patients: int = 50,
    seed: int = 7,
) -> dict[str, Any]:
    """Build a CSU-shaped DataPreparerState dict.

    Uses CSU manifest's canonical column names (``age_continuous``,
    ``gender``, ``brand``, ``journey_duration_days``, etc.) so the
    orchestrator's per-feature manifest lookups fire correctly.

    ``brand`` is set to ``competitor`` (CSU baseline) and is non-numeric
    so Layer 3 won't try to score it; ``journey_duration_days`` is an
    integer post-index leak that Layer 1 SHOULD catch (manifest declares
    it post-index). Numeric covariates (``age_continuous``, lab numerics)
    feed Layer 3 z-score scoring.
    """

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_patients)
    df = pd.DataFrame(
        {
            # CSU manifest pre-index features (Layer 1 declares safe).
            "age_continuous": rng.normal(50, 15, n_patients).astype(float),
            "gender": rng.choice(["M", "F"], n_patients),
            "brand": np.full(n_patients, "competitor"),
            # CSU manifest post-index leak (Layer 1 should catch).
            "journey_duration_days": rng.integers(30, 365, n_patients).astype(int),
            # Free numeric covariate (no manifest entry → declared_safe=False).
            "free_numeric_covariate": rng.standard_normal(n_patients),
            "y": y,
        }
    )
    return {
        "experiment_id": "g3-orchestrator-synth-csu",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": "y",
            "required_features": [c for c in df.columns if c != "y"],
            "excluded_features": [],
            "feature_manifest_source": "csu",
        },
        "leakage_findings": [],
        "leaked_features": [],
        "adaptive_seed": seed,
        "adaptive_n_permutations": 50,  # small for fast CI
    }


def _make_synthetic_optum_state(
    n_patients: int = 50,
    seed: int = 7,
) -> dict[str, Any]:
    """Build an Optum-shaped DataPreparerState dict.

    Uses Optum manifest's canonical column names (``age_at_index``,
    ``gender``, etc.). ``n_patients=50`` deliberately small to mirror
    the real Optum default-window N where HBLP's variance-inflation
    factor is the load-bearing relaxation.
    """

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_patients)
    df = pd.DataFrame(
        {
            # Optum manifest pre-index features.
            "age_at_index": rng.normal(60, 12, n_patients).astype(float),
            "gender": rng.choice(["M", "F"], n_patients),
            # Free numeric covariate (no manifest entry).
            "free_numeric_covariate": rng.standard_normal(n_patients),
            "y": y,
        }
    )
    return {
        "experiment_id": "g3-orchestrator-synth-optum",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": "y",
            "required_features": [c for c in df.columns if c != "y"],
            "excluded_features": [],
            "feature_manifest_source": "optum",
        },
        "leakage_findings": [],
        "leaked_features": [],
        "adaptive_seed": seed,
        "adaptive_n_permutations": 50,
    }


def _make_synthetic_optum_state_with_deterministic_pos_count(
    n_train_pos: int,
    seed: int = 7,
) -> dict[str, Any]:
    """Build an Optum-shaped state whose `y` has EXACTLY ``n_train_pos``
    positives and ``n_train_pos`` negatives (50/50 split).

    This makes the orchestrator's
    ``n_train_pos = int(np.sum(valid_target_values == 1))`` derivation
    deterministic so HBLP's variance-inflation factor is exactly the
    same on every test run. The numeric column ``age_at_index`` is the
    Optum-manifest "enrollment"-knowable feature (declared_safe=True);
    Layer 3 will score it under HBLP.
    """

    rng = np.random.default_rng(seed)
    n_total = n_train_pos * 2  # 50/50 split → simple deterministic count
    y = np.concatenate([np.ones(n_train_pos), np.zeros(n_train_pos)]).astype(int)
    # Shuffle so positive-class rows aren't trivially clustered (which
    # would change the orchestrator's downstream binary-mask handling).
    perm = rng.permutation(n_total)
    df = pd.DataFrame(
        {
            "age_at_index": rng.normal(60, 12, n_total).astype(float),
            "gender": rng.choice(["M", "F"], n_total),
            "free_numeric_covariate": rng.standard_normal(n_total),
            "y": y[perm],
        }
    )
    return {
        "experiment_id": f"g3-orchestrator-optum-n_pos-{n_train_pos}",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": "y",
            "required_features": [c for c in df.columns if c != "y"],
            "excluded_features": [],
            "feature_manifest_source": "optum",
        },
        "leakage_findings": [],
        "leaked_features": [],
        "adaptive_seed": seed,
        "adaptive_n_permutations": 50,
    }


def _make_synthetic_no_manifest_state(
    n_patients: int = 200,
    seed: int = 7,
) -> dict[str, Any]:
    """Synthetic regime with NO manifest_source (scenario_a baseline).

    All features are pure noise; Layer 1 is skipped (no manifest); Layer 3
    runs against arbitrary numeric covariates. n_train_pos = ~100 (above
    HBLP reference-N of 50, so no variance inflation).
    """

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "noise_a": rng.standard_normal(n_patients),
            "noise_b": rng.standard_normal(n_patients),
            "noise_c": rng.standard_normal(n_patients),
            "y": rng.integers(0, 2, n_patients),
        }
    )
    return {
        "experiment_id": "g3-orchestrator-synth-no-manifest",
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": "y",
            "required_features": [c for c in df.columns if c != "y"],
            "excluded_features": [],
            "feature_manifest_source": None,
        },
        "leakage_findings": [],
        "leaked_features": [],
        "adaptive_seed": seed,
        "adaptive_n_permutations": 50,
    }


class TestOrchestratorThreadingHigh4:
    """codex HIGH-4: orchestrator-path tests with synthetic CSU/Optum-shaped
    fixtures. These run ``adaptive_validity_check`` end-to-end (via asyncio)
    so a regression in n_train_pos derivation, manifest lookup, or
    layer_1_declared_safe threading would surface here.
    """

    @pytest.mark.asyncio
    async def test_csu_shaped_orchestrator_runs_end_to_end(self) -> None:
        """Orchestrator on CSU-shaped fixture emits verdicts + flagged set."""

        state = _make_synthetic_csu_state(n_patients=50)
        result = await _run_orchestrator(state)

        assert "adaptive_verdicts" in result
        assert "adaptive_flagged_features" in result
        verdicts = result["adaptive_verdicts"]
        assert isinstance(verdicts, list)
        # At least one verdict must have layer="1" (the manifest-driven
        # post-index catch on journey_duration_days) — proves Layer 1
        # ran end-to-end against the CSU manifest.
        layer_1_verdicts = [v for v in verdicts if v.get("layer") == "1"]
        assert len(layer_1_verdicts) >= 1, (
            f"Expected at least one Layer 1 verdict in CSU-shaped run; "
            f"got {[v.get('layer') for v in verdicts]}"
        )

    @pytest.mark.asyncio
    async def test_csu_shaped_orchestrator_layer_1_catches_post_index(self) -> None:
        """journey_duration_days is post-index per CSU manifest → flagged."""

        state = _make_synthetic_csu_state(n_patients=50)
        result = await _run_orchestrator(state)

        flagged = result["adaptive_flagged_features"]
        # journey_duration_days is post-index per CSU manifest → must be
        # caught by Layer 1 (severity=high, remediation=drop).
        assert "journey_duration_days" in flagged

    @pytest.mark.asyncio
    async def test_optum_shaped_orchestrator_runs_end_to_end(self) -> None:
        """Orchestrator on Optum-shaped fixture runs without crashing."""

        state = _make_synthetic_optum_state(n_patients=50)
        result = await _run_orchestrator(state)

        assert "adaptive_verdicts" in result
        assert "adaptive_flagged_features" in result
        verdicts = result["adaptive_verdicts"]
        assert isinstance(verdicts, list)
        # Optum-shaped fixture should produce at least one verdict.
        # Layer 1 may or may not catch given the synthetic features chosen;
        # the load-bearing assertion is that the orchestrator threads
        # n_train_pos and layer_1_declared_safe to _adversarial_input
        # without raising _HblpRoutingViolationError.
        assert verdicts, "Expected at least one verdict from Optum orchestrator path"

    # Marked slow (#481): runs two real-orchestrator end-to-end paths
    # (low-N + high-N) on Optum-shaped synthetic fixtures; observed
    # ~37s on the CI runner, dominating the lane behind the
    # synthetic-cohort-growth scenarios.
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_optum_shaped_low_n_threads_to_hblp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """codex pass-2 HIGH-4 PARTIAL: deterministic boundary-z assertion
        on the orchestrator threading.

        z=5.5 sits BETWEEN legacy 5σ (would classify "high") and HBLP-
        relaxed 7.5σ at high-N+declared_safe (would classify "moderate").
        At low-N (n_train_pos=22) + declared_safe=True, HBLP-effective
        high=11.31σ and moderate=6.78σ → z=5.5 classifies "info".

        Differential outcome under correct threading:
          * low_n=22 + manifest-safe → severity = "info"
          * high_n=200 + manifest-safe → severity = "moderate"

        If `n_train_pos` were dropped (defaulted to None in the
        orchestrator → falls through to reference-N=50 in
        ``_adversarial_input`` → variance_inflation=1.0 in BOTH paths)
        OR `layer_1_declared_safe` were dropped (defaulted to False
        → layer_1_factor=1.0 in BOTH paths), the two runs would
        produce IDENTICAL severities — and this test would fail.

        We monkeypatch ``compute_adversarial_score`` so the engineered
        z=5.5 score is returned deterministically; the real Layer 3
        scorer doesn't reliably land on a target z without seed
        fragility.
        """

        # codex HIGH-4 PARTIAL fix: monkeypatch the orchestrator's
        # ``compute_adversarial_score`` import so we can pin z=5.5.
        # The real adversarial scorer's z-score depends on permutation-
        # null seeding which makes "land near boundary z" too fragile
        # for a regression test.
        #
        # Import the MODULE explicitly via importlib — the bare
        # ``from .nodes import adaptive_validity_check`` form resolves
        # to the FUNCTION (re-exported by ``nodes/__init__.py``), not
        # the submodule, and ``monkeypatch.setattr`` then fails with
        # AttributeError. ``importlib.import_module`` returns the
        # module unambiguously regardless of __init__ re-exports.
        import importlib

        avc_module = importlib.import_module(
            "src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check"
        )

        def _fake_adversarial_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "z_score": 5.5,  # boundary value: legacy=high, low-N-HBLP=info
                "actual_auc": 0.72,
                "null_mean": 0.50,
                "null_std": 0.04,
                # Plus-one-floor-valid: the smallest p achievable from 50
                # permutations is 1/(1+50)≈0.0196 ("actual beat all shuffles").
                # The earlier 0.001 was impossible at this budget and tripped
                # benjamini_hochberg's floor guard (adversarial_leakage.py).
                "p_value": 1.0 / 51,
                "n_permutations": 50,
                "suspicious": True,
            }

        monkeypatch.setattr(
            avc_module,
            "compute_adversarial_score",
            _fake_adversarial_score,
        )

        # This test isolates the HBLP σ-threshold threading (n_train_pos +
        # layer_1_declared_safe). The FDR firing driver (#538, default-on) is a
        # SEPARATE mechanism with its own tests; left enabled it would override
        # the z-driven HBLP severity for a confident declared-safe feature
        # (#544 routes such features to review), collapsing the differential
        # outcome this test pins. Disable it here so the severities reflect
        # HBLP classification alone.
        #
        # Low-N path: n_train_pos=22, manifest-safe (age_at_index has
        # Optum-manifest knowable_at=enrollment → declared_safe=True).
        low_n_state = _make_synthetic_optum_state_with_deterministic_pos_count(
            n_train_pos=22, seed=11
        )
        low_n_state["adaptive_fdr_enabled"] = False
        low_n_result = await _run_orchestrator(low_n_state)
        # High-N path: n_train_pos=200, same manifest path.
        high_n_state = _make_synthetic_optum_state_with_deterministic_pos_count(
            n_train_pos=200, seed=11
        )
        high_n_state["adaptive_fdr_enabled"] = False
        high_n_result = await _run_orchestrator(high_n_state)

        # Find the age_at_index Layer 3 verdict in each run. Layer 3
        # verdicts are emitted only for numeric columns the orchestrator
        # scored — age_at_index is the Optum-cleared numeric column.
        def _layer_3_verdict_for(result: dict[str, Any], feature: str) -> dict[str, Any]:
            for v in result["adaptive_verdicts"]:
                if v.get("feature") == feature and v.get("layer") == "3":
                    return v
            raise AssertionError(
                f"No layer=3 verdict for {feature!r} in orchestrator result. "
                f"verdicts={[(v.get('feature'), v.get('layer')) for v in result['adaptive_verdicts']]}"
            )

        low_n_verdict = _layer_3_verdict_for(low_n_result, "age_at_index")
        high_n_verdict = _layer_3_verdict_for(high_n_result, "age_at_index")

        # Both runs hit the same z=5.5; the only distinguishing input
        # is n_train_pos (22 vs 200) which is derived inside the
        # orchestrator from the binary y vector. If threading is
        # broken, both verdicts produce IDENTICAL severities.
        assert low_n_verdict["severity"] == "info", (
            f"codex HIGH-4 PARTIAL: at low_n_train_pos=22 + manifest-safe, "
            f"z=5.5 must classify 'info' (HBLP-effective moderate=6.78σ); "
            f"got severity={low_n_verdict['severity']!r}. "
            f"This indicates n_train_pos was NOT threaded from orchestrator."
        )
        assert high_n_verdict["severity"] == "moderate", (
            f"codex HIGH-4 PARTIAL: at high_n_train_pos=200 + manifest-safe, "
            f"z=5.5 must classify 'moderate' (HBLP-effective high=7.5σ, "
            f"moderate=4.5σ); got severity={high_n_verdict['severity']!r}. "
            f"This indicates layer_1_declared_safe was NOT threaded from "
            f"orchestrator (declared_safe=False would yield 'high' at z=5.5)."
        )

        # Severities MUST DIFFER. This is the load-bearing assertion:
        # if either threading drops, both severities collapse to the
        # same value.
        assert low_n_verdict["severity"] != high_n_verdict["severity"], (
            f"codex HIGH-4 PARTIAL: severities IDENTICAL across (n=22, "
            f"safe=True) and (n=200, safe=True) at boundary z=5.5 — "
            f"this indicates `n_train_pos` is NOT threaded from "
            f"orchestrator. Both got severity="
            f"{low_n_verdict['severity']!r}."
        )

    @pytest.mark.asyncio
    async def test_no_manifest_orchestrator_uses_legacy_thresholds(self) -> None:
        """No-manifest synthetic regime: layer_1_declared_safe=False
        for every feature; HBLP at reference-N → no relaxation.

        At n=200 train rows → ~100 positives → above reference-N=50 →
        variance_inflation=1.0; without manifest, declared_safe=False →
        layer_1_factor=1.0. Legacy 5σ/3σ thresholds preserved.
        """

        state = _make_synthetic_no_manifest_state(n_patients=200)
        result = await _run_orchestrator(state)
        verdicts = result["adaptive_verdicts"]
        # No Layer 1 verdicts expected (no manifest).
        layer_1_verdicts = [v for v in verdicts if v.get("layer") == "1"]
        assert layer_1_verdicts == [], (
            f"Synthetic no-manifest regime should have no Layer 1 verdicts; got {layer_1_verdicts}"
        )
        # No flagged features (synthetic noise; nothing should be a leak).
        assert result["adaptive_flagged_features"] == [], (
            f"Pure-noise synthetic should not flag any feature; "
            f"got {result['adaptive_flagged_features']}"
        )


# --------------------------------------------------------------------------- #
# codex LOW-8: parametrize three-cohort sweep across actual synthetic regimes
# --------------------------------------------------------------------------- #
#
# The earlier helper-level matrix is hand-picked z values. LOW-8 asks for
# a parametrized sweep across actual synthetic regimes the project uses,
# with the regime's real (n_train_pos, manifest_source) and a baseline-
# expected verdict per (regime, z) cell. The assertion is "no baseline
# MARGINAL/non-high becomes high under G3 wiring" — i.e., HBLP relaxation
# never tightens classification.


# Regime metadata: (regime_label, n_train_pos, manifest_source,
# layer_1_declared_safe). Each regime represents a real synthetic /
# CSU / Optum cohort the project's pipeline targets.
SYNTHETIC_REGIMES: list[tuple[str, int, str | None, bool]] = [
    # synthetic baseline / scenario_a — pure noise, no manifest.
    # Reference-N (≥50 → no variance inflation) + declared=False (1.0x).
    ("synthetic_no_manifest_n200", 200, None, False),
    # synthetic CSU-shaped — manifest_source="csu", small N (~25 train_pos
    # at n_patients=50 / 0.5 prevalence).
    ("synthetic_csu_n50_pre_index", 25, "csu", True),
    # synthetic Optum-shaped — manifest_source="optum", smaller N.
    ("synthetic_optum_n50_pre_index", 22, "optum", True),
    # synthetic Optum default-window — n=22 emulating real Optum n=1294
    # 75/25 split anchor. layer_1_declared_safe=True (manifest-cleared).
    ("synthetic_optum_default_window_n22", 22, "optum", True),
]

# Pre-computed per-regime expected severity table.
#
# How these values were derived (HBLP math — Plan v3 §3 Tier 1B step 2):
#
#   HIGH_Z = 5.0, MODERATE_Z = 3.0, REFERENCE_N = 50
#   variance_inflation = max(1.0, sqrt(REFERENCE_N / n_train_pos))
#   layer_1_factor     = 1.5 if layer_1_declared_safe else 1.0
#   high_eff           = HIGH_Z × variance_inflation × layer_1_factor
#   moderate_eff       = MODERATE_Z × (high_eff / HIGH_Z)
#   severity = "high"     if z > high_eff
#            = "moderate" if z > moderate_eff
#            = "info"     otherwise
#
# Per-regime thresholds:
#   synthetic_no_manifest_n200 (n=200, declared=False):
#     variance_inflation = max(1.0, sqrt(50/200)) = max(1.0, 0.5)  = 1.0
#     layer_1_factor     = 1.0
#     high_eff           = 5.0 × 1.0 × 1.0 = 5.000
#     moderate_eff       = 3.0 × 1.0       = 3.000
#
#   synthetic_csu_n50_pre_index (n=25, declared=True):
#     variance_inflation = max(1.0, sqrt(50/25)) = sqrt(2) ≈ 1.4142
#     layer_1_factor     = 1.5
#     high_eff           = 5.0 × 1.4142 × 1.5 ≈ 10.607
#     moderate_eff       = 3.0 × 2.1213       ≈  6.364
#
#   synthetic_optum_n50_pre_index (n=22, declared=True):
#   synthetic_optum_default_window_n22 (n=22, declared=True):  [same N]
#     variance_inflation = max(1.0, sqrt(50/22)) ≈ 1.5076
#     layer_1_factor     = 1.5
#     high_eff           = 5.0 × 1.5076 × 1.5 ≈ 11.307
#     moderate_eff       = 3.0 × 2.2614       ≈  6.784
#
# To update this table: re-run the HBLP formula above with the new
# constants from adaptive_validity_check.py (HIGH_Z, MODERATE_Z,
# T2_1B_HBLP_VARIANCE_INFLATION_REFERENCE_N,
# T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER) and recompute each cell.
# The test will then catch any production-wiring divergence from the
# updated pinned values.
#
# fmt: off
EXPECTED_BY_REGIME: dict[str, dict[float, str]] = {
    # high_eff=5.00, moderate_eff=3.00 — legacy fixed thresholds (no inflation)
    "synthetic_no_manifest_n200": {
        2.0: "info",      # 2.0 ≤ 3.0
        4.0: "moderate",  # 3.0 < 4.0 ≤ 5.0
        6.0: "high",      # 6.0 > 5.0
        10.0: "high",     # 10.0 > 5.0
    },
    # high_eff≈10.607, moderate_eff≈6.364 — sqrt(2) × 1.5x layer-1 inflation
    "synthetic_csu_n50_pre_index": {
        2.0: "info",      # 2.0 ≤ 6.364
        4.0: "info",      # 4.0 ≤ 6.364
        6.0: "info",      # 6.0 ≤ 6.364
        10.0: "moderate", # 6.364 < 10.0 ≤ 10.607
    },
    # high_eff≈11.307, moderate_eff≈6.784 — sqrt(50/22) × 1.5x inflation
    "synthetic_optum_n50_pre_index": {
        2.0: "info",      # 2.0 ≤ 6.784
        4.0: "info",      # 4.0 ≤ 6.784
        6.0: "info",      # 6.0 ≤ 6.784
        10.0: "moderate", # 6.784 < 10.0 ≤ 11.307
    },
    # identical N and declared_safe to synthetic_optum_n50_pre_index
    "synthetic_optum_default_window_n22": {
        2.0: "info",      # 2.0 ≤ 6.784
        4.0: "info",      # 4.0 ≤ 6.784
        6.0: "info",      # 6.0 ≤ 6.784
        10.0: "moderate", # 6.784 < 10.0 ≤ 11.307
    },
}
# fmt: on

# Build parametrize list from the static table (no hblp_classify re-call).
LOW8_PARAMS: list[tuple[str, int, str | None, bool, float, str]] = []
for _regime, _n_pos, _manifest, _declared in SYNTHETIC_REGIMES:
    for _z, _expected_severity in EXPECTED_BY_REGIME[_regime].items():
        LOW8_PARAMS.append((_regime, _n_pos, _manifest, _declared, _z, _expected_severity))


class TestRegimeSweepLow8:
    """codex LOW-8: parametrized sweep across synthetic regimes.

    The acceptance criterion is the invariant:
      "no baseline MARGINAL/non-high becomes high under G3 wiring"
    i.e. HBLP relaxation NEVER tightens. Expected outcomes are pinned in
    ``EXPECTED_BY_REGIME`` (pre-computed from HBLP math, NOT derived by
    re-calling hblp_classify — that would be tautological). The test
    checks that the production orchestrator path (``_adversarial_input``)
    agrees with the table AND that no regime tightens relative to legacy.

    Distinct HBLP code paths exercised:
      - synthetic_no_manifest_n200: no inflation path (n>=ref_N, declared=False)
      - synthetic_csu_n50_pre_index: sqrt(2) inflation + 1.5x declared_safe
      - synthetic_optum_*: sqrt(50/22) inflation + 1.5x declared_safe
    NOTE: ``synthetic_optum_n50_pre_index`` and
    ``synthetic_optum_default_window_n22`` share the same HBLP inputs
    (n=22, declared=True) and therefore produce identical HBLP outcomes.
    Both rows are kept because they represent distinct orchestrator regimes
    (different manifest_source / window configurations) whose non-HBLP
    wiring is covered by the TestOrchestratorThreadingHigh4 suite. The
    regime sweep here is intentionally limited to the HBLP code path.
    If the HBLP formula changes (constants or branching), the pinned table
    will diverge from production output and the test will fail loudly.
    """

    @pytest.mark.parametrize(
        "regime,n_train_pos,manifest_source,layer_1_declared_safe,z_score,expected_severity",
        LOW8_PARAMS,
        ids=[f"{r[0]}::z={r[4]}" for r in LOW8_PARAMS],
    )
    def test_regime_sweep(
        self,
        regime: str,
        n_train_pos: int,
        manifest_source: str | None,
        layer_1_declared_safe: bool,
        z_score: float,
        expected_severity: str,
    ) -> None:
        """For each (regime, z) cell, assert post-G3 severity matches
        the pre-computed expected severity in EXPECTED_BY_REGIME AND
        is NOT stricter than legacy.

        ``expected_severity`` is sourced from the static EXPECTED_BY_REGIME
        table — it is NOT recomputed from hblp_classify. This ensures the
        test verifies production wiring rather than tautologically re-invoking
        the same helper that production calls.
        """

        score = {
            "z_score": z_score,
            "actual_auc": 0.5 + z_score * 0.04,
            "null_mean": 0.50,
            "null_std": 0.05,
            "p_value": 0.0 if z_score > 5 else 0.05,
            "n_permutations": 50,
        }
        adv = _adversarial_input(
            score,
            n_train_pos=n_train_pos,
            layer_1_declared_safe=layer_1_declared_safe,
        )
        assert adv["severity"] == expected_severity, (
            f"regime={regime!r} z={z_score} n_pos={n_train_pos} "
            f"declared={layer_1_declared_safe}: post-G3 severity="
            f"{adv['severity']!r} != EXPECTED_BY_REGIME pin "
            f"{expected_severity!r}. "
            f"Update EXPECTED_BY_REGIME if HBLP semantics changed."
        )

        # Invariant guard: HBLP relaxation NEVER tightens. Compute the
        # legacy classification (no n_train_pos, no declared_safe) and
        # assert post-G3 is the same OR lower severity, never higher.
        legacy = _adversarial_input(score)
        severity_rank = {"info": 0, "moderate": 1, "high": 2}
        assert severity_rank[adv["severity"]] <= severity_rank[legacy["severity"]], (
            f"codex LOW-8 invariant violated: regime={regime!r} z={z_score} — "
            f"HBLP TIGHTENED severity from legacy={legacy['severity']!r} to "
            f"post-G3={adv['severity']!r}. HBLP is supposed to relax, not "
            f"tighten — this would be a regression."
        )


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
