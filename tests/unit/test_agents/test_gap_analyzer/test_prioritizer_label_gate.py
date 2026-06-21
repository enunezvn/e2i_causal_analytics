"""prioritizer label-gating: an off-label opportunity that is the BEST raw bet
(highest expected_roi) must be flagged off_label and DEMOTED below on-label
opportunities — the user's off-label-uplift example. Gating is opt-in
(label_segmentation); off => unchanged.

The provider is injected with a real IndicatedPopulation (real criteria + the real
gate evaluator run); only the network fetch is bypassed — not behaviour mocking.
Mirrors test_policy_learner_label_gate.py for the heterogeneous_optimizer.
"""

import pytest

from src.agents.cohort_constructor.types import Criterion, CriterionType, Operator
from src.agents.gap_analyzer.nodes.prioritizer import PrioritizerNode
from src.agents.gap_analyzer.state import (
    GapAnalyzerState,
    PerformanceGap,
    ROIEstimate,
)
from src.services.clinical_context.label_gate import GateCriterion, IndicatedPopulation


class _FixedProvider:
    """Returns a fixed IndicatedPopulation (hr_status==positive is the label-evidenced
    Kisqali HR+/HER2- breast-cancer inclusion criterion)."""

    def __init__(self, population):
        self._pop = population

    def derive(self, brand, indication=None):
        return self._pop


def _kisqali_population() -> IndicatedPopulation:
    gc = GateCriterion(
        criterion=Criterion(
            field="hr_status",
            operator=Operator.EQUAL,
            value="positive",
            criterion_type=CriterionType.INCLUSION,
            clinical_rationale="indicated for hormone-receptor-positive disease",
        ),
        label_evidenced=True,
        label_evidence="…HR-positive, HER2-negative advanced or metastatic breast cancer…",
    )
    return IndicatedPopulation(
        brand="Kisqali", indication="hr_her2_bc", criteria=[gc], source="openfda_evidenced"
    )


def _gap(segment_value: str) -> PerformanceGap:
    return {
        "gap_id": f"hr_status_{segment_value}_trx_vs_target",
        "metric": "trx",
        "segment": "hr_status",
        "segment_value": segment_value,
        "current_value": 400.0,
        "target_value": 500.0,
        "gap_size": 100.0,
        "gap_percentage": 20.0,
        "gap_type": "vs_target",
    }


def _roi(gap_id: str, roi: float) -> ROIEstimate:
    cost = 10000.0
    revenue = cost * (roi + 1)
    return {
        "gap_id": gap_id,
        "estimated_revenue_impact": revenue,
        "estimated_cost_to_close": cost,
        "expected_roi": roi,
        "payback_period_months": 6,
        "confidence": 0.8,
        "assumptions": ["Test assumption"],
    }


def _state(label_segmentation: bool) -> GapAnalyzerState:
    # "negative" (off-label) is the BEST raw responder (HIGHER expected_roi).
    gap_pos = _gap("positive")
    gap_neg = _gap("negative")
    state: dict = {
        "query": "hr_status segment uplift",
        "metrics": ["trx"],
        "segments": ["hr_status"],
        "brand": "Kisqali",
        "time_period": "current_quarter",
        "filters": None,
        "gap_type": "vs_target",
        "min_gap_threshold": 5.0,
        "max_opportunities": 10,
        "gaps_detected": [gap_pos, gap_neg],
        "gaps_by_segment": None,
        "total_gap_value": 1000.0,
        "roi_estimates": [
            _roi(gap_pos["gap_id"], 2.0),  # on-label, LOWER roi
            _roi(gap_neg["gap_id"], 5.0),  # off-label, HIGHER roi (better-looking bet)
        ],
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
        "indication": "hr_her2_bc",
        "label_segmentation": label_segmentation,
    }
    return state  # type: ignore[return-value]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_off_label_segment_flagged_and_demoted():
    node = PrioritizerNode(label_criteria_provider=_FixedProvider(_kisqali_population()))
    result = await node.execute(_state(label_segmentation=True))
    opps = result["prioritized_opportunities"]
    by_seg = {o["gap"]["segment_value"]: o for o in opps}

    # The off-label ("negative") opportunity is flagged.
    assert by_seg["negative"]["roi_estimate"]["off_label"] is True
    assert by_seg["negative"]["roi_estimate"]["label_verdict"] == "off_label"
    assert by_seg["negative"]["roi_estimate"].get("off_label_reason")
    assert by_seg["negative"]["roi_estimate"]["label_evidence_confirmed"] is True
    # The on-label ("positive") opportunity is on_label, not flagged.
    assert by_seg["positive"]["roi_estimate"]["off_label"] is False
    assert by_seg["positive"]["roi_estimate"]["label_verdict"] == "on_label"

    # Despite the off-label opp having the HIGHER raw expected_roi, it ranks BELOW
    # the on-label opp (de-prioritization, rank demotion only).
    order = [o["gap"]["segment_value"] for o in opps]
    assert order.index("positive") < order.index("negative")
    assert by_seg["positive"]["rank"] < by_seg["negative"]["rank"]

    # The ROI values themselves are NOT tampered with (off-label still higher).
    assert (
        by_seg["negative"]["roi_estimate"]["expected_roi"]
        > by_seg["positive"]["roi_estimate"]["expected_roi"]
    )
    assert by_seg["negative"]["roi_estimate"]["expected_roi"] == 5.0
    assert by_seg["positive"]["roi_estimate"]["expected_roi"] == 2.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_label_segmentation_off_is_unchanged():
    node = PrioritizerNode(label_criteria_provider=_FixedProvider(_kisqali_population()))
    result = await node.execute(_state(label_segmentation=False))
    opps = result["prioritized_opportunities"]

    # No gating: ranked purely by expected_roi (off-label "negative" first), no flag.
    assert opps[0]["gap"]["segment_value"] == "negative"
    assert opps[0]["roi_estimate"]["expected_roi"] == 5.0
    assert all("off_label" not in o["roi_estimate"] for o in opps)
