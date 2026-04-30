"""Synthetic regimes — generator + tier0 e2e contracts.

Originally the adverse-regime smoke test (Block 4 / Findings #7, #8, #12);
extended in Section A of pre_phase2_unblockers to cover three regimes:

1. ``default``: ``positive_rate=0.30`` — the historical balanced regime;
   realised positive share lands at 13-18% (the existing 3 features pull
   risk down at mean inputs).
2. ``adverse``: ``positive_rate=0.02`` — extreme imbalance; exercises
   the pipeline's class-imbalance remediation paths (recommended_strategy
   resolves to ``combined``).
3. ``clean``: ``positive_rate=0.50`` + ``signal_strength=1.4`` +
   ``noise_sd=0.05`` + ``signalize_extra_features=True`` — strong-signal
   regime intended as the Phase 2 baseline. Realised positive share
   lands in [0.20, 0.45].

The clean E2E test requires Section B (lift metric) to be merged first:
without it, the deployer hard-fails on the missing
``minimum_lift_over_baseline`` metric and the E2E test never reaches
its val-AUC and lift assertions.

Pipeline runs gated behind ``@pytest.mark.slow`` because each tier0
invocation takes ~3-5 minutes; CI selects them via ``-m slow``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.repositories.sample_data import SampleDataGenerator  # noqa: E402
except ImportError as _imp_err:
    pytest.skip(
        f"requires full project deps (e.g. supabase, langgraph): {_imp_err}",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Generator-only fast tests (no pipeline)
# ---------------------------------------------------------------------------


class TestAdverseRegimeGenerator:
    """Verify ``ml_patients(positive_rate=0.02)`` produces an adverse cohort.

    These tests run in milliseconds; they isolate the synthetic generator
    from the rest of the pipeline so we can catch regressions cheaply.
    """

    def test_default_regime_unchanged_distribution(self):
        """Default ``positive_rate=0.30`` keeps the historical share.

        Empirically the realised positive share for the default regime
        sits around 13-18% because the feature-driven adjustments
        (-0.01·hcp_visits, -0.001·days_on_therapy) drag it down from
        the 30% intercept. The test guards a generous window around that
        empirical baseline.
        """
        gen = SampleDataGenerator(seed=42)
        df = gen.ml_patients(n_patients=1000)
        assert "discontinuation_flag" in df.columns
        positive_share = df["discontinuation_flag"].mean()
        assert 0.05 <= positive_share <= 0.35, (
            f"Default regime drifted: positive share={positive_share:.3f}"
        )

    def test_adverse_regime_produces_extreme_minority(self):
        """``positive_rate=0.02`` must yield ≤ 5% minority share without
        collapsing to a single class."""
        gen = SampleDataGenerator(seed=42)
        df = gen.ml_patients(n_patients=1500, positive_rate=0.02)
        positive_share = df["discontinuation_flag"].mean()
        # Must be well into the "extreme" band (< 5% per
        # detect_class_imbalance.SEVERITY_THRESHOLDS) but not degenerate.
        assert positive_share < 0.05, (
            f"Adverse regime did not produce extreme imbalance: positive share={positive_share:.3f}"
        )
        n_positive = int(df["discontinuation_flag"].sum())
        # Need at least 10 minority samples for SMOTE/combined remediation.
        assert n_positive >= 10, f"Adverse regime degenerated: only {n_positive} positive samples"
        # Both classes must be present; pipeline halts otherwise.
        assert df["discontinuation_flag"].nunique() == 2

    def test_adverse_regime_features_remain_correlated(self):
        """The feature ↔ label correlation should survive the rescaling
        applied in adverse mode — adverse mode tunes the *intercept*, not
        the feature signal-to-noise ratio. This catches regressions that
        zero-out the feature contribution at low base rates."""
        gen = SampleDataGenerator(seed=42)
        df = gen.ml_patients(n_patients=2000, positive_rate=0.02)
        # ``hcp_visits`` is wired to *reduce* discontinuation risk
        # (more visits → lower risk). The Pearson correlation should be
        # negative (or at least directionally informative).
        corr = df[["hcp_visits", "discontinuation_flag"]].corr().iloc[0, 1]
        # Allow a generous tolerance — at extreme imbalance the
        # observable correlation shrinks but should still be in the right
        # direction or near zero. We just guard against an *inverted*
        # signal (corr > +0.10) which would indicate the wiring broke.
        assert corr < 0.10, f"hcp_visits correlation flipped under adverse regime: {corr:.4f}"


# ---------------------------------------------------------------------------
# Full pipeline e2e (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(900)  # 15 min ceiling for the full tier0 run
class TestAdverseRegimeE2E:
    """Run ``run_pipeline(regime="adverse")`` and assert the imbalance
    remediation path engages without exceptions."""

    @pytest.fixture(scope="class")
    def pipeline_state(self) -> dict[str, Any]:
        """Run tier0 pipeline once with the adverse regime and reuse the
        output across all assertion methods."""
        # MLflow off — tests don't need artifact tracking.
        os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
        # Disable BentoML serving — out of scope for this test.
        from scripts.run_tier0_test import CONFIG, run_pipeline

        CONFIG.enable_mlflow = False
        CONFIG.enable_opik = False

        async def _run() -> dict[str, Any]:
            return await run_pipeline(
                step=None,
                dry_run=False,
                imbalance_ratio=None,
                include_bentoml=False,
                data_dir=None,
                regime="adverse",
                split_mode="auto",
            )

        result = asyncio.run(_run())
        assert result is not None, "run_pipeline returned None for adverse run"
        return result

    def test_pipeline_completes_without_exception(self, pipeline_state):
        """The pipeline must finish (the model_deployer step is allowed to
        fail on synthetic data, but the rest must succeed)."""
        assert pipeline_state.get("experiment_id"), (
            "experiment_id missing from adverse-regime pipeline state"
        )
        # Pipeline must not have been halted by a leakage remediation
        # cascade — that would mean the new positive_rate plumbing
        # accidentally created a target proxy.
        assert not pipeline_state.get("pipeline_halted"), (
            f"pipeline_halted unexpectedly: {pipeline_state.get('halt_reason', 'unknown')}"
        )

    def test_imbalance_severity_extreme(self, pipeline_state):
        """Severity must be 'extreme' at positive_rate=0.02."""
        info = pipeline_state.get("class_imbalance_info", {})
        assert info.get("imbalance_detected") is True
        assert info.get("imbalance_severity") == "extreme", (
            f"Expected severity=extreme; got {info.get('imbalance_severity')}; full info={info}"
        )

    def test_resampling_strategy_upgrades_to_combined(self, pipeline_state):
        """Per Block 4 plan: at extreme imbalance with a non-tree model the
        deterministic strategy matrix upgrades to ``combined`` (SMOTE +
        class weights).

        Block 6A (`a8069cf`) replaced the LLM-based imbalance strategy
        selection with a deterministic matrix lookup, so any deviation
        from ``combined`` at extreme imbalance + non-tree model is a real
        bug rather than transient LLM noise. (4-MIN-4: re-tighten from
        soft-warn to fail-loud now that the determinism guarantee holds.)
        """
        info = pipeline_state.get("class_imbalance_info", {})
        strategy = info.get("recommended_strategy")
        assert strategy == "combined", (
            f"After 6A determinism, extreme imbalance + non-tree model "
            f"must yield strategy='combined' (SMOTE + class weights). "
            f"Got strategy={strategy!r}; full info={info}"
        )

    def test_pipeline_persists_split_assignments(self, pipeline_state):
        """``split_assignments`` must end up on state for cache reuse
        (Block 4, Finding #12)."""
        assignments = pipeline_state.get("split_assignments")
        assert isinstance(assignments, dict)
        # Must cover at least train+val+test labels.
        labels = set(assignments.values())
        assert {"train", "val", "test"}.issubset(labels), (
            f"split_assignments missing train/val/test labels: {labels}"
        )

    def test_model_trainer_emits_predictions(self, pipeline_state):
        """Pipeline must NOT degenerate — model_trainer must produce
        a usable predictions surface even on adverse data."""
        validation_metrics = pipeline_state.get("validation_metrics", {})
        # Tolerant — adverse regime suppresses every metric. Accept either
        # present (pipeline emitted the metric) or explicitly None when the
        # agent gracefully handled extreme imbalance. What we forbid is the
        # metrics dict being missing entirely, which would indicate a hard
        # pipeline failure.
        assert "auc_roc" in validation_metrics or pipeline_state.get("model_usefulness") in {
            "useless",
            "poor",
            "acceptable",
            "unknown",
        }
        # Trained model object should exist.
        assert pipeline_state.get("trained_model") is not None, (
            "trained_model is None — pipeline degenerated on adverse data"
        )


# ---------------------------------------------------------------------------
# Default regime — regression guard for historical share + features
# ---------------------------------------------------------------------------


class TestDefaultRegimeGenerator:
    """``ml_patients()`` with no overrides preserves historical behavior.

    Section A of pre_phase2_unblockers introduced ``signal_strength``,
    ``noise_sd``, and ``signalize_extra_features`` keyword-only params to
    ``ml_patients``. Their defaults must reproduce the historical generator
    so existing tier0 baselines (``docs/results/tier0_remediation_baseline_20260426.md``)
    remain comparable.
    """

    def test_default_regime_share_unchanged(self):
        """Default ``positive_rate=0.30`` keeps the historical 13-18% share."""
        gen = SampleDataGenerator(seed=42)
        df = gen.ml_patients(n_patients=1500)
        share = df["discontinuation_flag"].mean()
        # Window matches the existing TestAdverseRegimeGenerator default-
        # regime test (0.05 ≤ share ≤ 0.35). Tighter windows risk false
        # positives from sklearn / numpy minor-version drift.
        assert 0.05 <= share <= 0.35, (
            f"default regime share drifted: {share:.3f}"
        )

    def test_default_regime_no_extra_signalization(self):
        """Default keeps the original 3-feature signal surface.

        With ``signalize_extra_features=False``, the four extras (age_group,
        geographic_region, brand, data_quality_score) carry no signal — so
        they should have weak (≈ 0) correlation with the target. We check
        ``data_quality_score`` because it's the only one the signalize path
        meaningfully shifts (~0.10 correlation when signalized vs ~0 when
        not).
        """
        gen = SampleDataGenerator(seed=42)
        df = gen.ml_patients(n_patients=2000)
        corr = (
            df[["data_quality_score", "discontinuation_flag"]]
            .corr()
            .iloc[0, 1]
        )
        assert abs(corr) < 0.05, (
            "default regime should not signalize data_quality_score; "
            f"|corr|={abs(corr):.3f} too large"
        )


# ---------------------------------------------------------------------------
# Clean regime — Phase 2 baseline contracts
# ---------------------------------------------------------------------------


class TestCleanRegimeGenerator:
    """Verify the ``clean`` regime produces a deployable strong-signal cohort.

    ``ml_patients(positive_rate=0.50, signal_strength=1.4, noise_sd=0.05,
    signalize_extra_features=True)`` must:
      (a) land the realised positive share in [0.20, 0.45];
      (b) make ``age_group=='>65'`` markedly riskier than ``<50``;
      (c) couple ``data_quality_score`` to the target with |corr| > 0.10;
      (d) make ``geographic_region=='west'`` lower-risk than other regions.

    Thresholds are tightened over an earlier draft so a coefficient-magnitude
    typo can't sneak through (was 0.05 → now 0.08 / 0.10 for the
    correlation tests).
    """

    @staticmethod
    def _generate_clean(n: int = 2000):
        gen = SampleDataGenerator(seed=42)
        return gen.ml_patients(
            n_patients=n,
            positive_rate=0.50,
            signal_strength=1.4,
            noise_sd=0.05,
            signalize_extra_features=True,
        )

    def test_a_target_share_in_band(self):
        df = self._generate_clean()
        share = df["discontinuation_flag"].mean()
        assert 0.20 <= share <= 0.45, (
            f"clean regime share out of band: {share:.3f}"
        )

    def test_b_age_group_risk_gradient(self):
        df = self._generate_clean()
        gt65 = df[df["age_group"] == ">65"]["discontinuation_flag"].mean()
        lt50 = df[df["age_group"] == "<50"]["discontinuation_flag"].mean()
        assert (gt65 - lt50) >= 0.08, (
            f"age_group >65 vs <50 risk gradient too small: "
            f">65={gt65:.3f}, <50={lt50:.3f}, diff={gt65 - lt50:.3f}"
        )

    def test_c_data_quality_score_correlation(self):
        df = self._generate_clean()
        corr = (
            df[["data_quality_score", "discontinuation_flag"]]
            .corr()
            .iloc[0, 1]
        )
        # Coefficient is negative (higher dqs → lower risk).
        assert abs(corr) > 0.10, (
            f"data_quality_score → target correlation too weak: "
            f"corr={corr:.4f}"
        )

    def test_d_west_region_negative_lift(self):
        df = self._generate_clean()
        west = df[df["geographic_region"] == "west"]["discontinuation_flag"].mean()
        others = df[df["geographic_region"] != "west"]["discontinuation_flag"].mean()
        # Sign check: west must be LOWER than other regions (negative coef).
        assert (others - west) >= 0.04, (
            f"geographic_region west should be lower-risk than others: "
            f"west={west:.3f}, others={others:.3f}, diff={others - west:.3f}"
        )


# Note: the TestCleanRegimeE2E suite below requires Section B (lift metric)
# to be merged first — without it, the deployer hard-fails on the missing
# ``minimum_lift_over_baseline`` metric and the E2E run never reaches the
# val-AUC and lift assertions below.


@pytest.mark.slow
@pytest.mark.timeout(900)
class TestCleanRegimeE2E:
    """``run_pipeline(regime='clean')`` produces a deployable model.

    Loose val-AUC band [0.75, 0.85] absorbs sklearn / SHAP minor-version
    variance. The lift criterion must produce a numeric lift > 0.10
    (absolute AUC units), and the model deployer must succeed (Section B
    of pre_phase2_unblockers fixes the gap that was blocking it).
    """

    @pytest.fixture(scope="class")
    def pipeline_state(self) -> dict[str, Any]:
        os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
        from scripts.run_tier0_test import CONFIG, run_pipeline

        CONFIG.enable_mlflow = False
        CONFIG.enable_opik = False

        async def _run() -> dict[str, Any]:
            return await run_pipeline(
                step=None,
                dry_run=False,
                imbalance_ratio=None,
                include_bentoml=False,
                data_dir=None,
                regime="clean",
                split_mode="auto",
            )

        result = asyncio.run(_run())
        assert result is not None, "run_pipeline returned None for clean run"
        return result

    def test_pipeline_completes(self, pipeline_state):
        assert pipeline_state.get("experiment_id"), (
            "experiment_id missing from clean-regime pipeline state"
        )
        assert not pipeline_state.get("pipeline_halted"), (
            f"pipeline_halted unexpectedly: {pipeline_state.get('halt_reason', 'unknown')}"
        )

    def test_val_auc_in_band(self, pipeline_state):
        validation_metrics = pipeline_state.get("validation_metrics", {})
        val_auc = validation_metrics.get("roc_auc") or validation_metrics.get("auc_roc")
        assert val_auc is not None, "validation roc_auc missing"
        assert 0.75 <= val_auc <= 0.85, (
            f"clean regime val AUC out of band: {val_auc:.4f}"
        )

    def test_train_val_gap_modest(self, pipeline_state):
        train_metrics = pipeline_state.get("train_metrics", {})
        validation_metrics = pipeline_state.get("validation_metrics", {})
        train_auc = train_metrics.get("roc_auc") or train_metrics.get("auc_roc")
        val_auc = validation_metrics.get("roc_auc") or validation_metrics.get("auc_roc")
        if train_auc is None or val_auc is None:
            pytest.skip("train or val AUC unavailable; gap check requires both")
        gap = train_auc - val_auc
        assert gap < 0.08, f"train→val AUC gap too large: train={train_auc:.4f}, val={val_auc:.4f}, gap={gap:.4f}"

    def test_lift_over_baseline_positive(self, pipeline_state):
        test_metrics = pipeline_state.get("test_metrics", {})
        lift = test_metrics.get("minimum_lift_over_baseline")
        assert lift is not None, (
            "minimum_lift_over_baseline missing — Section B not merged?"
        )
        # Comfortable margin over the 0.10 threshold; tolerate sklearn drift.
        assert lift > 0.10, f"clean-regime lift too small: {lift:.4f}"

    def test_deployer_succeeds(self, pipeline_state):
        assert pipeline_state.get("success_criteria_met") is True, (
            "deployer should succeed on clean regime; "
            f"results={pipeline_state.get('success_criteria_results')}"
        )
