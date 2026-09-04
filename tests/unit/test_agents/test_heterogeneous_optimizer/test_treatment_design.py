"""Shared treatment / design-matrix invariants across the three HTE estimation nodes.

2026-09-03 (wave 53), live seg_05f29d1b3295 (Remibrutinib, urticaria_severity_uas7 ->
persistent_180d): the treatment column sat INSIDE the effect-modifier matrix X, so
the median-split treatment was a deterministic function of an X column and
CausalForestDML returned ATE -0.514 on a 0/1 outcome. Separately, the uplift node
handed CausalML the RAW continuous score (27 "treatment groups", control "16.0")
while the CATE and hierarchical nodes binarized at the median — the cross-library
validator then compared two different estimands and reported 9% agreement.

Two invariants, enforced in ONE place (``design.py``) and consumed by all three
nodes:

* ``sanitize_effect_modifiers``: X never contains the treatment, the outcome, or a
  provenance column — whatever the caller passed.
* ``binarize_treatment``: a continuous treatment is split at the median by the same
  rule everywhere, so EconML and CausalML estimate the same contrast.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.agents.heterogeneous_optimizer.design import (
    binarize_treatment,
    sanitize_effect_modifiers,
)
from src.agents.heterogeneous_optimizer.nodes.cate_estimator import CATEEstimatorNode
from src.agents.heterogeneous_optimizer.nodes.hierarchical_analyzer import (
    HierarchicalAnalyzerNode,
)
from src.agents.heterogeneous_optimizer.nodes.uplift_analyzer import UpliftAnalyzerNode

TREATMENT = "urticaria_severity_uas7"
OUTCOME = "persistent_180d"


def _frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """A Remibrutinib-shaped cohort: a continuous 0-42 UAS7 score, a 0/1 outcome
    with a planted +0.15 lift above the >=28 axis, and two numeric modifiers."""
    rng = np.random.default_rng(seed)
    uas7 = rng.integers(0, 43, n).astype(float)
    y = (rng.random(n) < 0.45 + 0.15 * (uas7 >= 28)).astype(float)
    return pd.DataFrame(
        {
            TREATMENT: uas7,
            OUTCOME: y,
            "disease_severity": rng.normal(5, 2, n),
            "age_at_diagnosis": rng.normal(50, 10, n),
            "engagement_score": rng.normal(size=n),
            "disease_severity_band": rng.choice(["low", "medium", "high"], n),
        }
    )


def _state(df: pd.DataFrame, modifiers: list[str]) -> dict:
    return {
        "treatment_var": TREATMENT,
        "outcome_var": OUTCOME,
        "effect_modifiers": modifiers,
        "segment_vars": ["disease_severity_band"],
        "confounders": ["engagement_score"],
        "tier0_data": df,
        "data_source": "patient_journeys",
        "filters": None,
        "n_estimators": 8,
    }


# -----------------------------------------------------------------------------
# The shared helpers
# -----------------------------------------------------------------------------


def test_binarize_treatment_splits_above_the_median():
    """The rule cate_estimator has always used: 1 when strictly above the median."""
    out, info = binarize_treatment(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert out.tolist() == [0, 0, 0, 1, 1]
    assert info is not None
    assert info["median_threshold"] == 3.0
    assert info["original_unique_values"] == 5
    assert info["treated_count"] == 2 and info["control_count"] == 3


def test_binarize_treatment_leaves_a_binary_treatment_untouched():
    raw = np.array([0.0, 1.0, 1.0, 0.0])
    out, info = binarize_treatment(raw)
    assert np.array_equal(out, raw)
    assert info is None


def test_sanitize_effect_modifiers_drops_question_slots_and_provenance():
    state = {
        "treatment_var": TREATMENT,
        "outcome_var": OUTCOME,
        "effect_modifiers": [
            "disease_severity",
            TREATMENT,
            OUTCOME,
            "is_synthetic",
            "age_at_diagnosis",
        ],
    }
    kept, dropped = sanitize_effect_modifiers(state)
    assert kept == ["disease_severity", "age_at_diagnosis"]
    assert dropped == [TREATMENT, OUTCOME, "is_synthetic"]


# -----------------------------------------------------------------------------
# The nodes consume them
# -----------------------------------------------------------------------------


def test_uplift_prepare_data_binarizes_a_continuous_treatment_like_the_cate_node():
    """Live: CausalML received 27 raw score groups (control '16.0') and the
    retained column was the uplift of score 42 vs 16 — not the estimand EconML
    reported. The uplift node must split at the median exactly as cate_estimator does."""
    df = _frame()
    node = UpliftAnalyzerNode(model_type="random_forest", n_estimators=4, max_depth=3)
    _x, treatment, _y = node._prepare_data(df, _state(df, ["disease_severity", "age_at_diagnosis"]))
    raw = df[TREATMENT].to_numpy()
    assert set(np.unique(treatment).tolist()) == {0, 1}
    assert np.array_equal(treatment, (raw > np.median(raw)).astype(int))


def test_hierarchical_prepare_data_uses_the_same_rule():
    """Refactor guard: the hierarchical node already binarized at the median; it
    must keep doing so through the shared helper."""
    df = _frame()
    node = HierarchicalAnalyzerNode(data_connector=MagicMock())
    _x, treatment, _y = node._prepare_data(df, _state(df, ["disease_severity", "age_at_diagnosis"]))
    raw = df[TREATMENT].to_numpy()
    assert np.array_equal(treatment, (raw > np.median(raw)).astype(int))


@pytest.mark.parametrize(
    "make_node",
    [
        lambda: HierarchicalAnalyzerNode(data_connector=MagicMock()),
        lambda: UpliftAnalyzerNode(model_type="random_forest", n_estimators=4, max_depth=3),
    ],
    ids=["hierarchical", "uplift"],
)
def test_prepare_data_drops_the_question_slots_from_x(make_node):
    df = _frame()
    node = make_node()
    x, _t, _y = node._prepare_data(
        df, _state(df, ["disease_severity", TREATMENT, OUTCOME, "age_at_diagnosis"])
    )
    assert list(x.columns) == ["disease_severity", "age_at_diagnosis"]


@pytest.mark.asyncio
async def test_cate_estimator_drops_the_treatment_from_x_before_fitting():
    """End-to-end on the real forest: with the treatment inside X the propensity
    model is perfect and the DML residual is zero (live ATE -0.514 on a 0/1
    outcome). The node must fit on X minus the question slots, report feature
    importance for those columns only, and return an ATE a 0/1 outcome can carry."""
    df = _frame()
    node = CATEEstimatorNode(data_connector=MagicMock())
    result = await node.execute(_state(df, ["disease_severity", "age_at_diagnosis", TREATMENT]))

    assert result.get("status") != "failed", result.get("errors")
    assert set(result["feature_importance"]) == {"disease_severity", "age_at_diagnosis"}
    assert abs(result["overall_ate"]) <= 1.0
