"""P9 / H8 + H9 — consensus agreement label + inverse-variance weighting.

- H8: ``consensus_confidence`` (mean of per-library confidences) was surfaced to
  users under the name ``library_agreement_score`` — two libraries reporting
  +0.5 and −0.5 cancel to ~0 effect yet keep a high consensus_confidence, so the
  reported "agreement" was misleading. Surface a REAL agreement metric instead.
- H9: DoWhy hardcodes confidence=1.0 while EconML/CausalML report <1 on
  incommensurable scales, so the confidence-weighted consensus_effect was
  DoWhy-dominated. Weight by inverse variance (precision) when SEs are available.
"""

from __future__ import annotations

from src.causal_engine.pipeline.sequential import (
    _apply_agreement_score,
    _apply_consensus,
    _apply_pairwise_agreement,
)


class TestAgreementScore:
    def test_sign_disagreement_yields_low_agreement(self):
        # +0.5 vs −0.5: total disagreement → agreement ≈ 0.
        effects = [("dowhy", 0.5, 0.9), ("econml", -0.5, 0.9)]
        state: dict = {}
        _apply_pairwise_agreement(state, effects)
        _apply_agreement_score(state)
        assert state["library_agreement_score"] is not None
        assert state["library_agreement_score"] < 0.3, (
            f"sign-disagreeing libraries must report LOW agreement, got "
            f"{state['library_agreement_score']}"
        )

    def test_close_agreement_yields_high_score(self):
        effects = [("dowhy", 0.50, 0.9), ("econml", 0.52, 0.9)]
        state: dict = {}
        _apply_pairwise_agreement(state, effects)
        _apply_agreement_score(state)
        assert state["library_agreement_score"] > 0.8


class TestInverseVarianceWeighting:
    def _state_with_ses(self):
        # dowhy: imprecise (SE=5.0, effect 10); econml/causalml precise (SE≈0.1).
        half = 1.959963984540054 * 0.1  # half-width for SE=0.1
        return {
            "dowhy_result": {"result": {"standard_error": 5.0}},
            "econml_result": {"result": {"ate_ci_lower": 2.0 - half, "ate_ci_upper": 2.0 + half}},
            "uplift_summary": {"ate_ci_lower": 2.1 - half, "ate_ci_upper": 2.1 + half},
        }

    def test_consensus_is_precision_weighted_not_dowhy_dominated(self):
        state = self._state_with_ses()
        effects = [("dowhy", 10.0, 1.0), ("econml", 2.0, 0.7), ("causalml", 2.1, 0.7)]
        _apply_consensus(state, effects)
        assert state["consensus_weighting"] == "inverse_variance"
        # Confidence-weighting (DoWhy conf=1.0) would give ≈5.36 (DoWhy-dominated);
        # inverse-variance gives ≈2.05 (the precise estimators dominate).
        assert state["consensus_effect"] < 3.0, (
            f"consensus must be precision-weighted (~2.05), not DoWhy-dominated, "
            f"got {state['consensus_effect']}"
        )

    def test_falls_back_to_confidence_when_se_unavailable(self):
        # No SE/CI for any library → confidence-weighting fallback.
        state: dict = {}
        effects = [("dowhy", 10.0, 1.0), ("econml", 2.0, 0.7)]
        _apply_consensus(state, effects)
        assert state["consensus_weighting"] == "confidence"
        # (10·1 + 2·0.7) / 1.7 ≈ 6.71 — the documented fallback.
        assert state["consensus_effect"] > 5.0
