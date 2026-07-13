"""Fast (non-slow) unit tests for refutation-node helpers.

``test_refutation.py`` is module-marked ``slow`` (it drives the real DoWhy
refutation suite, minutes per test, off-PR). These pure-logic helper tests have
no DoWhy dependency, so they live here to run in the on-PR backend lane.
"""

import asyncio
from types import SimpleNamespace

from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.agents.causal_impact.nodes.refutation import (
    RefutationNode,
    _reconstruction_nuisance_init_params,
)
from src.causal_engine.refutation_runner import GateDecision


class TestReconstructionNuisanceInitParams:
    """The reconstructed LinearDML must use the SAME nuisance models production's
    LinearDMLWrapper uses (RandomForest outcome + treatment), so the reconstructed
    ATE reproduces the reported one and the tolerance guard validates the ACTUAL
    estimate. A prior scaled-LINEAR substitution refit a different model and
    diverged on nonlinear data (hcp_adoption: 0.2033 vs 0.0248), fail-closing
    refutation. Forest / plain-linear methods take no override."""

    def test_lineardml_discrete_uses_randomforest_nuisance(self):
        p = _reconstruction_nuisance_init_params(
            "backdoor.econml.dml.LinearDML", discrete_treatment=True
        )
        assert set(p) == {"model_y", "model_t"}
        # Mirror production exactly: RF regressor outcome, RF classifier propensity
        # (discrete treatment). Same params as LinearDMLWrapper (n_estimators=50, ...).
        assert isinstance(p["model_y"], RandomForestRegressor)
        assert isinstance(p["model_t"], RandomForestClassifier)
        assert p["model_y"].n_estimators == 50
        assert p["model_y"].min_samples_leaf == 5
        assert is_classifier(p["model_t"])
        assert not is_classifier(p["model_y"])

    def test_lineardml_continuous_treatment_uses_regressor(self):
        p = _reconstruction_nuisance_init_params(
            "backdoor.econml.dml.LinearDML", discrete_treatment=False
        )
        assert isinstance(p["model_t"], RandomForestRegressor)
        assert not is_classifier(p["model_t"])

    def test_drlearner_mirrors_selector_models(self):
        # #1188: the deferred DRLearner mirror shipped — the selector's DR
        # wrapper now uses GB nuisances + a StatsModelsLinearRegression final
        # stage (honest ATE inference), and the reconstruction mirrors those
        # EXACT models so the tolerance guard validates the actual estimate.
        # (Supersedes the old left-at-defaults deferral; full assertions in
        # test_refutation_efficiency.py::TestDrLearnerNuisanceMirror.)
        params = _reconstruction_nuisance_init_params(
            "backdoor.econml.dr.DRLearner", discrete_treatment=True
        )
        assert set(params) == {"model_regression", "model_propensity", "model_final"}

    def test_forest_and_linear_methods_get_no_override(self):
        # CausalForestDML uses scale-invariant forest nuisance -> no override needed.
        assert (
            _reconstruction_nuisance_init_params(
                "backdoor.econml.dml.CausalForestDML", discrete_treatment=True
            )
            == {}
        )
        # Plain linear regression has no iterative nuisance to converge.
        assert (
            _reconstruction_nuisance_init_params(
                "backdoor.linear_regression", discrete_treatment=True
            )
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

    def test_passes_causal_graph_and_validation_ids_to_gate(self):
        """Mig 097: the consult must forward the in-state graph (so the queued
        row is renderable) and the persisted validation-row ids (evidence link)."""
        gate = _FakeReviewGate()
        node = RefutationNode(expert_review_gate=gate)
        graph = {
            "nodes": ["t", "y", "c"],
            "edges": [("t", "y"), ("c", "t"), ("c", "y")],
            "treatment_nodes": ["t"],
            "outcome_nodes": ["y"],
        }
        state = {
            "dag_version_hash": "h3",
            "causal_graph": graph,
            "treatment_var": "t",
            "outcome_var": "y",
            "query_id": "q",
        }
        asyncio.run(
            node._consult_review_gate(
                state, _suite(GateDecision.REVIEW), validation_ids=["val-9", "val-10"]
            )
        )
        assert gate.calls[0]["dag_structure"] == graph
        assert gate.calls[0]["related_validation_ids"] == ["val-9", "val-10"]

    def test_absent_graph_and_ids_forward_none(self):
        """Defensive: a state without causal_graph must not break the consult."""
        gate = _FakeReviewGate()
        node = RefutationNode(expert_review_gate=gate)
        state = {"dag_version_hash": "h", "treatment_var": "t", "outcome_var": "y", "query_id": "q"}
        asyncio.run(node._consult_review_gate(state, _suite(GateDecision.REVIEW)))
        assert gate.calls[0]["dag_structure"] is None
        assert gate.calls[0]["related_validation_ids"] is None


class _ApprovedReviewGate:
    """Gate whose check_approval finds an ACTIVE approval for this DAG hash."""

    def __init__(self):
        self.calls: list[dict] = []

    async def check_approval(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            decision=SimpleNamespace(value="proceed"),
            review_id="rev-approved-7",
            is_approved=True,
            reviewer_name="Dr. Jane Roe",
            valid_until="2026-09-30",
        )


class TestApprovalAwareCaveat:
    """When the DAG already holds an ACTIVE expert approval, the surfaced caveat
    must acknowledge it (reviewer + validity) instead of reading as if the HITL
    loop never happened — while still stating that the approval covers the DAG
    STRUCTURE, not this estimate's statistical robustness (never overclaims)."""

    def test_review_band_caveat_acknowledges_active_approval(self):
        gate = _ApprovedReviewGate()
        node = RefutationNode(expert_review_gate=gate)
        state = {"dag_version_hash": "h", "treatment_var": "t", "outcome_var": "y", "query_id": "q"}
        fields = asyncio.run(node._consult_review_gate(state, _suite(GateDecision.REVIEW)))
        assert fields["expert_review_decision"] == "proceed"
        caveat = fields["review_caveat"]
        assert "expert-approved" in caveat
        assert "Dr. Jane Roe" in caveat
        assert "2026-09-30" in caveat
        # The approval must NOT be presented as validating the estimate itself.
        assert "not" in caveat and "robustness" in caveat
        # The band wording is still present (borderline stays borderline).
        assert "REVIEW" in caveat

    def test_block_band_caveat_also_acknowledges_but_stays_blocked(self):
        gate = _ApprovedReviewGate()
        node = RefutationNode(expert_review_gate=gate)
        state = {"dag_version_hash": "h", "treatment_var": "t", "outcome_var": "y", "query_id": "q"}
        fields = asyncio.run(
            node._consult_review_gate(state, _suite(GateDecision.BLOCK, confidence=0.3))
        )
        caveat = fields["review_caveat"]
        assert "BLOCK" in caveat
        assert "expert-approved" in caveat

    def test_pending_review_caveat_unchanged(self):
        """No active approval -> the existing band caveat, no approval note."""
        gate = _FakeReviewGate()
        node = RefutationNode(expert_review_gate=gate)
        state = {"dag_version_hash": "h", "treatment_var": "t", "outcome_var": "y", "query_id": "q"}
        fields = asyncio.run(node._consult_review_gate(state, _suite(GateDecision.REVIEW)))
        assert "expert-approved" not in fields["review_caveat"]
