"""Tests for Instrument-Availability Bonus in Gap Analyzer (#357, Option-3).

Tests the instrument-availability bonus mechanism that boosts opportunity ROI
when a STRONG instrument (first-stage F >= 10, Staiger-Stock) is available for
the opportunity's feature, sourced from REAL IV first-stage F-tests
(src/causal_engine/iv).

Covers:
- PrioritizerNode._has_instrument_evidence
- PrioritizerNode._apply_instrument_availability_bonus
- Interaction with the V4.4 causal-evidence boost (compounding, D-3)
- InstrumentAnalyzerNode (the P-2 live producer) running a REAL TwoStageLSEstimator
  first stage over tier0_data (anti-mocking AC2)
- GapAnalyzerAgent._initialize_state copying the new fields
- End-to-end routing through the real gap_analyzer graph (the producer wire, AC4)

Mirrors the structure of test_causal_prioritization.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.gap_analyzer.agent import GapAnalyzerAgent
from src.agents.gap_analyzer.graph import create_gap_analyzer_graph
from src.agents.gap_analyzer.nodes.instrument_analyzer import InstrumentAnalyzerNode
from src.agents.gap_analyzer.nodes.prioritizer import (
    DIRECT_CAUSE_BOOST,
    NO_CAUSAL_EVIDENCE_PENALTY,
    STRONG_INSTRUMENT_BONUS,
    STRONG_INSTRUMENT_F_FLOOR,
    PrioritizerNode,
)
from src.agents.gap_analyzer.state import (
    GapAnalyzerState,
    PerformanceGap,
    ROIEstimate,
)

# =============================================================================
# Shared fixtures / builders
# =============================================================================


def _make_gap(metric: str = "trx", gap_id: str | None = None) -> PerformanceGap:
    return {
        "gap_id": gap_id or f"region_Northeast_{metric}_vs_target",
        "metric": metric,
        "segment": "region",
        "segment_value": "Northeast",
        "current_value": 400.0,
        "target_value": 500.0,
        "gap_size": 100.0,
        "gap_percentage": 20.0,
        "gap_type": "vs_target",
    }


def _make_roi(gap_id: str, roi: float = 3.0) -> ROIEstimate:
    return {
        "gap_id": gap_id,
        "estimated_revenue_impact": 40000.0,
        "estimated_cost_to_close": 10000.0,
        "expected_roi": roi,
        "payback_period_months": 6,
        "confidence": 0.8,
        "assumptions": ["Test assumption"],
    }


def _make_opportunity(metric: str = "trx", roi: float = 3.0):
    gap = _make_gap(metric=metric)
    roi_estimate = _make_roi(gap["gap_id"], roi=roi)
    return {
        "rank": 1,
        "gap": gap,
        "roi_estimate": roi_estimate,
        "recommended_action": "Test action",
        "implementation_difficulty": "medium",
        "time_to_impact": "3-6 months",
    }


def _strong_diag(f_stat: float = 24.0) -> dict:
    """An IVDiagnostics.to_dict()-shaped dict for a strong instrument."""
    return {
        "instrument_strength": "strong",
        "is_weak_instrument": False,
        "first_stage_f_stat": f_stat,
    }


# =============================================================================
# Consumer (boost) tests
# =============================================================================


class TestInstrumentAvailabilityBonus:
    """Test _apply_instrument_availability_bonus method."""

    def test_strong_instrument_boosts_roi(self):
        """Feature with strong instrument -> expected_roi multiplied by bonus."""
        node = PrioritizerNode()
        opportunities = [_make_opportunity(metric="trx", roi=3.0)]
        instrument_lookup = {"trx": _strong_diag(24.0)}

        adjusted, warnings = node._apply_instrument_availability_bonus(
            opportunities, instrument_lookup
        )

        assert len(adjusted) == 1
        roi = adjusted[0]["roi_estimate"]
        assert roi["expected_roi"] == pytest.approx(3.0 * STRONG_INSTRUMENT_BONUS)
        assert roi["instrument_adjustment_factor"] == STRONG_INSTRUMENT_BONUS
        assert roi["instrument_adjustment_reason"] == "strong_instrument_bonus"

    def test_moderate_instrument_unchanged(self):
        """Moderate instrument (F in 5-10) -> factor 1.0, no boost key added."""
        node = PrioritizerNode()
        opportunities = [_make_opportunity(metric="trx", roi=3.0)]
        instrument_lookup = {
            "trx": {
                "instrument_strength": "moderate",
                "is_weak_instrument": True,
                "first_stage_f_stat": 7.0,
            }
        }

        adjusted, _ = node._apply_instrument_availability_bonus(opportunities, instrument_lookup)

        roi = adjusted[0]["roi_estimate"]
        assert roi["expected_roi"] == 3.0
        assert "instrument_adjustment_factor" not in roi

    def test_weak_instrument_unchanged(self):
        """Weak and very_weak instruments -> unchanged (no penalty, D-4)."""
        node = PrioritizerNode()
        for strength, f in (("weak", 3.0), ("very_weak", 1.0)):
            opportunities = [_make_opportunity(metric="trx", roi=3.0)]
            instrument_lookup = {
                "trx": {
                    "instrument_strength": strength,
                    "is_weak_instrument": True,
                    "first_stage_f_stat": f,
                }
            }
            adjusted, _ = node._apply_instrument_availability_bonus(
                opportunities, instrument_lookup
            )
            roi = adjusted[0]["roi_estimate"]
            assert roi["expected_roi"] == 3.0, f"{strength} should not change ROI"
            assert "instrument_adjustment_factor" not in roi

    def test_strong_enum_but_low_fstat_does_not_boost(self):
        """Strong enum but real F below the floor -> NO boost (belt-and-suspenders)."""
        node = PrioritizerNode()
        opportunities = [_make_opportunity(metric="trx", roi=3.0)]
        instrument_lookup = {
            "trx": {
                "instrument_strength": "strong",
                "is_weak_instrument": True,
                "first_stage_f_stat": 4.0,  # below STRONG_INSTRUMENT_F_FLOOR
            }
        }

        adjusted, _ = node._apply_instrument_availability_bonus(opportunities, instrument_lookup)

        roi = adjusted[0]["roi_estimate"]
        assert STRONG_INSTRUMENT_F_FLOOR == 10.0
        assert roi["expected_roi"] == 3.0
        assert "instrument_adjustment_factor" not in roi

    def test_feature_not_in_lookup_no_change(self):
        """Lookup present but this feature missing -> factor 1.0."""
        node = PrioritizerNode()
        opportunities = [_make_opportunity(metric="nrx", roi=2.5)]
        instrument_lookup = {"trx": _strong_diag(24.0)}

        adjusted, _ = node._apply_instrument_availability_bonus(opportunities, instrument_lookup)

        roi = adjusted[0]["roi_estimate"]
        assert roi["expected_roi"] == 2.5
        assert "instrument_adjustment_factor" not in roi


class TestHasInstrumentEvidence:
    """Test _has_instrument_evidence gate."""

    def _base_state(self) -> GapAnalyzerState:
        return {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            "gaps_detected": [],
            "gaps_by_segment": None,
            "total_gap_value": 0.0,
            "roi_estimates": [],
            "total_addressable_value": 0.0,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 0,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

    def test_has_instrument_evidence_present(self):
        node = PrioritizerNode()
        state = self._base_state()
        state["instrument_strength_by_feature"] = {"trx": _strong_diag(24.0)}
        assert node._has_instrument_evidence(state) is True

    def test_no_instrument_evidence_absent(self):
        node = PrioritizerNode()
        state = self._base_state()
        assert node._has_instrument_evidence(state) is False

    def test_no_instrument_evidence_empty(self):
        node = PrioritizerNode()
        state = self._base_state()
        state["instrument_strength_by_feature"] = {}
        assert node._has_instrument_evidence(state) is False


# =============================================================================
# Interaction with V4.4 (order/compounding well-defined — D-3 = COMPOUND)
# =============================================================================


class TestInstrumentBonusExecuteIntegration:
    """Integration tests in PrioritizerNode.execute combining both mechanisms."""

    def _state(
        self,
        gaps,
        roi_estimates,
        *,
        causal_rankings=None,
        discovery_gate_decision=None,
        direct_cause_features=None,
        predictive_only_features=None,
        instrument_strength_by_feature=None,
    ) -> GapAnalyzerState:
        return {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            "gaps_detected": gaps,
            "gaps_by_segment": None,
            "total_gap_value": 1000.0,
            "roi_estimates": roi_estimates,
            "total_addressable_value": 100000.0,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 50,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "prioritizing",
            "causal_rankings": causal_rankings,
            "discovery_gate_decision": discovery_gate_decision,
            "direct_cause_features": direct_cause_features or [],
            "predictive_only_features": predictive_only_features or [],
            "instrument_strength_by_feature": instrument_strength_by_feature,
        }

    @pytest.mark.asyncio
    async def test_absent_instrument_field_no_change(self):
        """No instrument field -> ROI byte-identical to V4.4-only baseline (additivity)."""
        node = PrioritizerNode()
        gap = _make_gap(metric="trx")
        roi = _make_roi(gap["gap_id"], roi=3.0)

        state = self._state(
            [gap],
            [roi],
            instrument_strength_by_feature=None,
        )
        result = await node.execute(state)

        opp = result["prioritized_opportunities"][0]
        assert opp["roi_estimate"]["expected_roi"] == 3.0
        assert "instrument_adjustment_factor" not in opp["roi_estimate"]

    @pytest.mark.asyncio
    async def test_instrument_bonus_without_causal_evidence(self):
        """Bonus applies even when causal_rankings is absent (gates are independent)."""
        node = PrioritizerNode()
        gap = _make_gap(metric="trx")
        roi = _make_roi(gap["gap_id"], roi=3.0)

        state = self._state(
            [gap],
            [roi],
            causal_rankings=None,
            discovery_gate_decision=None,
            instrument_strength_by_feature={"trx": _strong_diag(24.0)},
        )
        result = await node.execute(state)

        opp = result["prioritized_opportunities"][0]
        assert opp["roi_estimate"]["expected_roi"] == pytest.approx(3.0 * STRONG_INSTRUMENT_BONUS)
        assert opp["roi_estimate"]["instrument_adjustment_reason"] == "strong_instrument_bonus"
        # V4.4 key must NOT be present (no causal evidence)
        assert "causal_adjustment_factor" not in opp["roi_estimate"]

    @pytest.mark.asyncio
    async def test_direct_cause_and_strong_instrument_compound(self):
        """Feature both direct-cause AND strong-instrument -> compounded (D-3)."""
        node = PrioritizerNode()
        gap = _make_gap(metric="trx")
        roi = _make_roi(gap["gap_id"], roi=3.0)

        causal_rankings = [{"feature_name": "trx", "causal_score": 0.9, "is_direct_cause": True}]
        state = self._state(
            [gap],
            [roi],
            causal_rankings=causal_rankings,
            discovery_gate_decision="accept",
            direct_cause_features=["trx"],
            instrument_strength_by_feature={"trx": _strong_diag(24.0)},
        )
        result = await node.execute(state)

        opp = result["prioritized_opportunities"][0]
        roi_est = opp["roi_estimate"]
        # Compound: original * DIRECT_CAUSE_BOOST * STRONG_INSTRUMENT_BONUS
        assert roi_est["expected_roi"] == pytest.approx(
            3.0 * DIRECT_CAUSE_BOOST * STRONG_INSTRUMENT_BONUS
        )
        # Both records present and not clobbered
        assert roi_est["causal_adjustment_factor"] == DIRECT_CAUSE_BOOST
        assert roi_est["causal_adjustment_reason"] == "direct_cause_boost"
        assert roi_est["instrument_adjustment_factor"] == STRONG_INSTRUMENT_BONUS
        assert roi_est["instrument_adjustment_reason"] == "strong_instrument_bonus"

    @pytest.mark.asyncio
    async def test_predictive_only_penalty_with_strong_instrument(self):
        """Predictive-only (penalty) AND strong instrument -> compound, bonus doesn't cancel penalty."""
        node = PrioritizerNode()
        gap = _make_gap(metric="market_share")
        roi = _make_roi(gap["gap_id"], roi=4.0)

        causal_rankings = [
            {"feature_name": "market_share", "causal_score": 0.3, "is_direct_cause": False}
        ]
        state = self._state(
            [gap],
            [roi],
            causal_rankings=causal_rankings,
            discovery_gate_decision="accept",
            predictive_only_features=["market_share"],
            instrument_strength_by_feature={"market_share": _strong_diag(30.0)},
        )
        result = await node.execute(state)

        opp = result["prioritized_opportunities"][0]
        roi_est = opp["roi_estimate"]
        # Compound: original * NO_CAUSAL_EVIDENCE_PENALTY * STRONG_INSTRUMENT_BONUS
        assert roi_est["expected_roi"] == pytest.approx(
            4.0 * NO_CAUSAL_EVIDENCE_PENALTY * STRONG_INSTRUMENT_BONUS
        )
        assert roi_est["causal_adjustment_factor"] == NO_CAUSAL_EVIDENCE_PENALTY
        assert roi_est["instrument_adjustment_factor"] == STRONG_INSTRUMENT_BONUS


# =============================================================================
# Producer (P-2) — REAL TwoStageLSEstimator first stage over tier0_data (AC2)
# =============================================================================


class TestInstrumentAnalyzerProducer:
    """Test InstrumentAnalyzerNode populating strength from a REAL first stage."""

    def _strong_dgp_df(self, n: int = 200, seed: int = 42) -> pd.DataFrame:
        """A DGP where Z strongly predicts D (treatment), D causes Y (outcome)."""
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=n)
        U = rng.normal(size=n)
        D = 2.0 * Z + 1.5 * U + rng.normal(scale=0.3, size=n)
        Y = 1.0 * D + 1.0 * U + rng.normal(scale=0.5, size=n)
        return pd.DataFrame({"trx": Y, "rep_calls": D, "weather_shock": Z})

    def _weak_dgp_df(self, n: int = 200, seed: int = 7) -> pd.DataFrame:
        """A DGP where Z barely predicts D (weak instrument)."""
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=n)
        U = rng.normal(size=n)
        D = 0.03 * Z + 1.5 * U + rng.normal(scale=1.0, size=n)
        Y = 1.0 * D + 1.0 * U + rng.normal(scale=0.5, size=n)
        return pd.DataFrame({"nrx": Y, "rep_calls": D, "weather_shock": Z})

    @pytest.mark.asyncio
    async def test_iv_step_populates_instrument_strength_from_real_first_stage(self):
        """Strong DGP -> 'strong' with real F >= 10, computed by the REAL estimator (no mock)."""
        node = InstrumentAnalyzerNode()
        df = self._strong_dgp_df()
        state = {
            "tier0_data": df,
            "instrument_specs": {
                "trx": {
                    "treatment_col": "rep_calls",
                    "instrument_cols": ["weather_shock"],
                    "outcome_col": "trx",
                }
            },
        }

        result = await node.execute(state)  # type: ignore[arg-type]

        by_feature = result["instrument_strength_by_feature"]
        assert "trx" in by_feature
        diag = by_feature["trx"]
        assert diag["instrument_strength"] == "strong"
        assert diag["first_stage_f_stat"] >= 10.0
        assert diag["is_weak_instrument"] is False

    @pytest.mark.asyncio
    async def test_iv_step_weak_dgp_classifies_weak(self):
        """Weak DGP -> not 'strong' (real F below 10)."""
        node = InstrumentAnalyzerNode()
        df = self._weak_dgp_df()
        state = {
            "tier0_data": df,
            "instrument_specs": {
                "nrx": {
                    "treatment_col": "rep_calls",
                    "instrument_cols": ["weather_shock"],
                    "outcome_col": "nrx",
                }
            },
        }

        result = await node.execute(state)  # type: ignore[arg-type]

        diag = result["instrument_strength_by_feature"]["nrx"]
        assert diag["instrument_strength"] != "strong"
        assert diag["first_stage_f_stat"] < 10.0

    @pytest.mark.asyncio
    async def test_iv_step_missing_columns_absent_from_map(self):
        """Spec referencing missing columns -> feature ABSENT from the map (no stub)."""
        node = InstrumentAnalyzerNode()
        df = self._strong_dgp_df()
        state = {
            "tier0_data": df,
            "instrument_specs": {
                "trx": {
                    "treatment_col": "does_not_exist",
                    "instrument_cols": ["weather_shock"],
                    "outcome_col": "trx",
                }
            },
        }

        result = await node.execute(state)  # type: ignore[arg-type]

        assert "trx" not in result.get("instrument_strength_by_feature", {})

    @pytest.mark.asyncio
    async def test_iv_step_below_n_floor_absent_from_map(self):
        """Below the n-floor -> feature ABSENT (no signal, not a stub)."""
        node = InstrumentAnalyzerNode()
        df = self._strong_dgp_df(n=10)  # below n>=30 floor
        state = {
            "tier0_data": df,
            "instrument_specs": {
                "trx": {
                    "treatment_col": "rep_calls",
                    "instrument_cols": ["weather_shock"],
                    "outcome_col": "trx",
                }
            },
        }

        result = await node.execute(state)  # type: ignore[arg-type]

        assert "trx" not in result.get("instrument_strength_by_feature", {})

    @pytest.mark.asyncio
    async def test_iv_step_noop_when_no_specs(self):
        """No instrument_specs -> no-op, empty/absent map (AC4b fail-closed)."""
        node = InstrumentAnalyzerNode()
        df = self._strong_dgp_df()
        state = {"tier0_data": df, "instrument_specs": None}

        result = await node.execute(state)  # type: ignore[arg-type]

        assert not result.get("instrument_strength_by_feature")

    @pytest.mark.asyncio
    async def test_noop_preserves_precomputed_strength(self):
        """No specs/tier0_data but a precomputed strength on state -> PRESERVED, not clobbered.

        Regression for the codex-flagged routing bug: the IV node must not erase a
        precomputed instrument_strength_by_feature (the P-1 passthrough path that
        _initialize_state already copies) just because it has nothing of its own to run.
        """
        node = InstrumentAnalyzerNode()
        state = {
            "instrument_specs": None,
            "tier0_data": None,
            "instrument_strength_by_feature": {"trx": _strong_diag(24.0)},
        }

        result = await node.execute(state)  # type: ignore[arg-type]

        by_feature = result["instrument_strength_by_feature"]
        assert by_feature["trx"]["instrument_strength"] == "strong"
        assert by_feature["trx"]["first_stage_f_stat"] == 24.0

    @pytest.mark.asyncio
    async def test_real_estimate_merges_onto_precomputed(self):
        """A real first-stage result merges on top of a precomputed map (recompute wins; others kept)."""
        node = InstrumentAnalyzerNode()
        df = self._strong_dgp_df()
        state = {
            "tier0_data": df,
            "instrument_specs": {
                "trx": {
                    "treatment_col": "rep_calls",
                    "instrument_cols": ["weather_shock"],
                    "outcome_col": "trx",
                }
            },
            # A different feature precomputed via passthrough must be preserved.
            "instrument_strength_by_feature": {"nrx": _strong_diag(18.0)},
        }

        result = await node.execute(state)  # type: ignore[arg-type]

        by_feature = result["instrument_strength_by_feature"]
        # Precomputed nrx preserved...
        assert by_feature["nrx"]["first_stage_f_stat"] == 18.0
        # ...and trx computed from the REAL first stage.
        assert by_feature["trx"]["instrument_strength"] == "strong"
        assert by_feature["trx"]["first_stage_f_stat"] >= 10.0


# =============================================================================
# Routing — _initialize_state copy + end-to-end graph wire (AC4)
# =============================================================================


class TestInstrumentRoutingWire:
    """Test the producer wire reaches the prioritizer through real plumbing."""

    def test_initialize_state_copies_instrument_fields(self):
        """_initialize_state lands instrument_specs + instrument_strength_by_feature."""
        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
        specs = {
            "trx": {
                "treatment_col": "rep_calls",
                "instrument_cols": ["weather_shock"],
                "outcome_col": "trx",
            }
        }
        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "instrument_specs": specs,
            "instrument_strength_by_feature": {"trx": _strong_diag(24.0)},
        }
        state = agent._initialize_state(input_data)

        assert state["instrument_specs"] == specs
        assert state["instrument_strength_by_feature"]["trx"]["instrument_strength"] == "strong"

    @pytest.mark.asyncio
    async def test_full_graph_strong_instrument_reorders_ranking(self):
        """End-to-end: a strong-instrument feature outranks an equal-ROI non-instrumented one.

        Proves the new state field is populated by the IV producer and reaches the
        prioritizer through the REAL compiled graph (the wire, not just arithmetic).
        """
        rng = np.random.default_rng(11)
        n = 200
        Z = rng.normal(size=n)
        U = rng.normal(size=n)
        D = 2.0 * Z + 1.5 * U + rng.normal(scale=0.3, size=n)
        trx = 1.0 * D + 1.0 * U + rng.normal(scale=0.5, size=n)
        # tier0_data with both metrics; only trx has a strong instrument spec
        df = pd.DataFrame(
            {
                "region": ["Northeast"] * n,
                "trx": trx,
                "market_share": rng.normal(loc=10.0, scale=2.0, size=n),
                "rep_calls": D,
                "weather_shock": Z,
            }
        )

        # Two gaps with EQUAL base ROI; only trx is strong-instrumented.
        gap_trx = _make_gap(metric="trx", gap_id="region_Northeast_trx_vs_target")
        gap_ms = _make_gap(metric="market_share", gap_id="region_Northeast_market_share_vs_target")
        roi_trx = _make_roi(gap_trx["gap_id"], roi=3.0)
        roi_ms = _make_roi(gap_ms["gap_id"], roi=3.0)

        state = {
            "query": "test",
            "metrics": ["trx", "market_share"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "tier0_data": df,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            # Pre-populate detection + ROI so the graph nodes flow into prioritizer.
            "gaps_detected": [gap_trx, gap_ms],
            "roi_estimates": [roi_trx, roi_ms],
            "instrument_specs": {
                "trx": {
                    "treatment_col": "rep_calls",
                    "instrument_cols": ["weather_shock"],
                    "outcome_col": "trx",
                }
            },
            "detection_latency_ms": 0,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        graph = create_gap_analyzer_graph()
        final = await graph.ainvoke(state)

        by_feature = final.get("instrument_strength_by_feature") or {}
        assert by_feature.get("trx", {}).get("instrument_strength") == "strong"

        opps = final["prioritized_opportunities"]
        trx_opp = next(o for o in opps if o["gap"]["metric"] == "trx")
        ms_opp = next(o for o in opps if o["gap"]["metric"] == "market_share")
        # trx boosted above the equal-ROI market_share opportunity.
        assert trx_opp["roi_estimate"]["expected_roi"] > ms_opp["roi_estimate"]["expected_roi"]
        assert trx_opp["rank"] < ms_opp["rank"]
