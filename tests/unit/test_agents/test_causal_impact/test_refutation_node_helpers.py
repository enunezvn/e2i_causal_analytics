"""Fast (non-slow) unit tests for refutation-node helpers.

``test_refutation.py`` is module-marked ``slow`` (it drives the real DoWhy
refutation suite, minutes per test, off-PR). These pure-logic helper tests have
no DoWhy dependency, so they live here to run in the on-PR backend lane.
"""

import asyncio
from types import SimpleNamespace

from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.agents.causal_impact.nodes.refutation import (
    RefutationNode,
    _scaled_nuisance_init_params,
)
from src.causal_engine.refutation_runner import GateDecision


class TestScaledNuisanceInitParams:
    """Perf-fix: the reconstructed estimator's nuisance models are scaled
    (StandardScaler) so the logistic propensity converges fast instead of
    grinding lbfgs to max_iter (~40s/refit) on mixed-scale covariates. Applied
    ONLY to the lbfgs-prone estimators (LinearDML / DRLearner); forest-based
    (CausalForestDML) and plain linear methods are left untouched."""

    def test_lineardml_discrete_scaled_classifier_propensity(self):
        p = _scaled_nuisance_init_params("backdoor.econml.dml.LinearDML", discrete_treatment=True)
        assert set(p) == {"model_y", "model_t"}
        for m in p.values():
            assert isinstance(m, Pipeline)
            assert isinstance(m.steps[0][1], StandardScaler)
        # discrete treatment -> propensity model_t is a classifier; model_y a regressor
        assert is_classifier(p["model_t"])
        assert not is_classifier(p["model_y"])

    def test_lineardml_continuous_treatment_uses_regressor(self):
        p = _scaled_nuisance_init_params("backdoor.econml.dml.LinearDML", discrete_treatment=False)
        assert not is_classifier(p["model_t"])

    def test_drlearner_left_at_defaults(self):
        # DRLearner is intentionally NOT scaled here: its scaled-linear
        # reconstruction is unvalidated against the selector's GradientBoosting
        # nuisance, so we don't ship that numeric change. Left at econml defaults.
        assert (
            _scaled_nuisance_init_params("backdoor.econml.dr.DRLearner", discrete_treatment=True)
            == {}
        )

    def test_forest_and_linear_methods_get_no_override(self):
        # CausalForestDML uses scale-invariant forest nuisance -> no scaling needed.
        assert (
            _scaled_nuisance_init_params(
                "backdoor.econml.dml.CausalForestDML", discrete_treatment=True
            )
            == {}
        )
        # Plain linear regression has no iterative nuisance to converge.
        assert (
            _scaled_nuisance_init_params("backdoor.linear_regression", discrete_treatment=True)
            == {}
        )


class _FakeReviewGate:
    """Records the check_approval call and returns a canned review result."""

    def __init__(self, review_id="rev-abc123"):
        self._review_id = review_id
        self.calls: list[dict] = []

    async def check_approval(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            decision=SimpleNamespace(value="pending_review"),
            review_id=self._review_id,
        )


def _suite(decision, confidence=0.6):
    return SimpleNamespace(gate_decision=decision, confidence_score=confidence)


class TestConsultReviewGateRouting:
    """R6-F2 fix: the gate must be keyed on the REAL DAG hash (``dag_version_hash``,
    not the never-written ``dag_hash``), must route BOTH the REVIEW and BLOCK
    bands to the expert-review queue, and must propagate the created ``review_id``.
    """

    def test_uses_dag_version_hash_not_the_never_written_dag_hash(self):
        # Root-cause regression: the node previously read ``dag_hash`` (never
        # populated) -> every row was keyed on "". It must read ``dag_version_hash``.
        gate = _FakeReviewGate()
        node = RefutationNode(expert_review_gate=gate)
        state = {
            "dag_version_hash": "real-hash-9f",
            "dag_hash": "",  # the old, never-written key — must be ignored
            "treatment_var": "t",
            "outcome_var": "y",
            "brand": "Kisqali",
            "query_id": "q-1",
        }
        fields = asyncio.run(node._consult_review_gate(state, _suite(GateDecision.REVIEW)))
        assert gate.calls, "ExpertReviewGate.check_approval was not called"
        assert gate.calls[0]["dag_hash"] == "real-hash-9f"
        assert fields["expert_review_id"] == "rev-abc123"

    def test_review_band_caveat_and_no_explicit_needs_review(self):
        gate = _FakeReviewGate()
        node = RefutationNode(expert_review_gate=gate)
        state = {"dag_version_hash": "h", "treatment_var": "t", "outcome_var": "y", "query_id": "q"}
        fields = asyncio.run(node._consult_review_gate(state, _suite(GateDecision.REVIEW)))
        assert "REVIEW" in fields["review_caveat"]
        # needs_review is set by the caller from suite.needs_review, not here.
        assert "needs_review" not in fields

    def test_block_band_is_also_routed_with_block_caveat(self):
        gate = _FakeReviewGate(review_id="rev-block-1")
        node = RefutationNode(expert_review_gate=gate)
        state = {
            "dag_version_hash": "h2",
            "treatment_var": "t",
            "outcome_var": "y",
            "query_id": "q",
        }
        fields = asyncio.run(
            node._consult_review_gate(state, _suite(GateDecision.BLOCK, confidence=0.3))
        )
        assert gate.calls[0]["dag_hash"] == "h2"
        assert "BLOCK" in fields["review_caveat"]
        assert fields["expert_review_id"] == "rev-block-1"

    def test_gate_error_degrades_without_breaking_the_node(self):
        class _BoomGate:
            async def check_approval(self, **kwargs):
                raise RuntimeError("repo down")

        node = RefutationNode(expert_review_gate=_BoomGate())
        state = {"dag_version_hash": "h", "treatment_var": "t", "outcome_var": "y", "query_id": "q"}
        fields = asyncio.run(node._consult_review_gate(state, _suite(GateDecision.REVIEW)))
        assert fields["expert_review_id"] is None
        assert "REVIEW" in fields["review_caveat"]
