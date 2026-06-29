"""policy_learner label-gating: an off-label (treatment-naive) segment that is the
BEST raw responder must be flagged off_label and DEMOTED below on-label segments —
the user's CSU example. Gating is opt-in (label_segmentation); off => unchanged.

The provider is injected with a real IndicatedPopulation (real criteria + the real
gate evaluator run); only the network fetch is bypassed — not behaviour mocking.
"""

import pytest

from src.agents.cohort_constructor.types import Criterion, CriterionType, Operator
from src.agents.heterogeneous_optimizer.nodes.policy_learner import PolicyLearnerNode
from src.services.clinical_context.label_gate import GateCriterion, IndicatedPopulation


class _FixedProvider:
    """Returns a fixed IndicatedPopulation (prior_antihistamine_therapy==True is the
    label-evidenced CSU inclusion criterion)."""

    def __init__(self, population):
        self._pop = population

    def derive(self, brand, indication=None):
        return self._pop


def _csu_population():
    gc = GateCriterion(
        criterion=Criterion(
            field="prior_antihistamine_therapy",
            operator=Operator.EQUAL,
            value=True,
            criterion_type=CriterionType.INCLUSION,
            clinical_rationale="indicated after H1-antihistamine failure",
        ),
        label_evidenced=True,
        label_evidence="…symptomatic despite H1 antihistamine treatment…",
    )
    return IndicatedPopulation(
        brand="Remibrutinib", indication="csu", criteria=[gc], source="openfda_evidenced"
    )


def _cate(value, cate, n=500):
    return {
        "segment_name": "prior_antihistamine_therapy",
        "segment_value": value,
        "cate_estimate": cate,
        "cate_ci_lower": cate - 0.05,
        "cate_ci_upper": cate + 0.05,
        "sample_size": n,
        "statistical_significance": True,
    }


def _responder(value, cate, n=500):
    """High-responder tier profile as segment_analyzer would emit it (both CATEs
    here, 0.30 and 0.50, clear the strict 1.5x|ATE|=0.3 bar). The policy direction
    follows this tier; without it every segment would maintain at 0.5 (0 lift)."""
    return {
        "segment_id": f"prior_antihistamine_therapy_{value}",
        "responder_type": "high",
        "cate_estimate": cate,
        "defining_features": [{"variable": "prior_antihistamine_therapy", "value": value}],
        "size": n,
        "size_percentage": 50.0,
        "recommendation": "increase",
    }


def _state(label_segmentation: bool):
    # "False" (treatment-naive) is the BEST raw responder (highest CATE -> highest lift).
    return {
        "query": "csu segment uplift",
        "treatment_var": "treatment",
        "outcome_var": "persistent_180d",
        "segment_vars": ["prior_antihistamine_therapy"],
        "effect_modifiers": [],
        "data_source": "patient_journeys",
        "overall_ate": 0.2,
        "cate_by_segment": {
            "prior_antihistamine_therapy": [_cate("True", 0.30), _cate("False", 0.50)],
        },
        "high_responders": [_responder("True", 0.30), _responder("False", 0.50)],
        "low_responders": [],
        "estimation_latency_ms": 0,
        "analysis_latency_ms": 0,
        "warnings": [],
        "errors": [],
        "status": "optimizing",
        "brand": "Remibrutinib",
        "indication": "csu",
        "label_segmentation": label_segmentation,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_off_label_naive_segment_flagged_and_demoted():
    node = PolicyLearnerNode(label_criteria_provider=_FixedProvider(_csu_population()))
    out = await node.execute(_state(label_segmentation=True))  # type: ignore[arg-type]
    recs = out["policy_recommendations"]
    by_seg = {r["segment"]: r for r in recs}

    assert by_seg["prior_antihistamine_therapy=False"]["off_label"] is True
    assert by_seg["prior_antihistamine_therapy=False"]["label_verdict"] == "off_label"
    assert by_seg["prior_antihistamine_therapy=False"].get("off_label_reason")
    assert by_seg["prior_antihistamine_therapy=True"]["off_label"] is False
    assert by_seg["prior_antihistamine_therapy=True"]["label_verdict"] == "on_label"

    # Despite the off-label segment having the HIGHER raw expected outcome, it ranks
    # BELOW the on-label segment (de-prioritization).
    order = [r["segment"] for r in recs]
    assert order.index("prior_antihistamine_therapy=True") < order.index(
        "prior_antihistamine_therapy=False"
    )
    # Outcome value itself is NOT tampered with (off-label still has the higher value).
    assert (
        by_seg["prior_antihistamine_therapy=False"]["expected_incremental_outcome"]
        > by_seg["prior_antihistamine_therapy=True"]["expected_incremental_outcome"]
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_label_segmentation_off_is_unchanged():
    node = PolicyLearnerNode(label_criteria_provider=_FixedProvider(_csu_population()))
    out = await node.execute(_state(label_segmentation=False))  # type: ignore[arg-type]
    recs = out["policy_recommendations"]
    # No gating: ranked purely by outcome (off-label naive first), no off_label flag.
    assert recs[0]["segment"] == "prior_antihistamine_therapy=False"
    assert all("off_label" not in r for r in recs)
