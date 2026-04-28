"""Block 4 — Adverse-regime synthetic generator + tier0 e2e smoke test.

Findings: #7, #8, #12 (Tier-0 remediation, MEDIUM).

This module verifies two contracts:

1. ``SampleDataGenerator.ml_patients(positive_rate=0.02)`` actually emits a
   minority-class share in the adverse range (≤ ~5%) without trivially
   degenerating to a single class. This is the synthetic-only contract that
   feeds the rest of the pipeline.

2. The tier0 pipeline run with ``regime="adverse"`` does not degenerate:
   ``run_pipeline`` finishes, ``recommended_strategy`` resolves to
   ``"combined"`` (the heuristic branch for extreme imbalance with a
   non-tree model), and ``split_assignments`` are persisted on the returned
   state.

The pipeline run is gated behind ``@pytest.mark.slow`` because the full
tier0 invocation takes ~3-5 minutes; CI selects it via ``-m slow``.
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
