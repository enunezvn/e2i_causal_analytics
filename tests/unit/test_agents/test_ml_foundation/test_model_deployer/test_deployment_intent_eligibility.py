"""Deployment-intent-aware regulatory eligibility (clinical | commercial).

The literature-anchor registry is keyed on (gate, value, intent). A commercial
targeting model resolves the AUC 0.60 anchor; a clinical model keeps 0.75. The
intent is part of the key so a clinical run can NEVER borrow the commercial
anchor (anti-laundering, codex N1-H2). Default intent is "clinical".
"""

from __future__ import annotations

from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    _evaluate_absolute_threshold_gates,
)
from src.agents.ml_foundation.model_deployer.regulatory_audit import (
    RegulatoryEligibilityAudit,
    classify_threshold_provenance,
    get_literature_anchor_doc_ref,
)

_TS = "2026-06-07T00:00:00Z"


class TestIntentAwareProvenance:
    def test_clinical_075_anchored_by_default(self) -> None:
        assert (
            classify_threshold_provenance(gate_name="minimum_auc", threshold=0.75)
            == "literature_anchored"
        )

    def test_commercial_060_anchored_under_commercial_intent(self) -> None:
        assert (
            classify_threshold_provenance(
                gate_name="minimum_auc", threshold=0.60, deployment_intent="commercial"
            )
            == "literature_anchored"
        )

    def test_clinical_060_NOT_anchored_antilaundering(self) -> None:
        """A clinical run cannot deploy at the commercial 0.60 floor."""
        assert (
            classify_threshold_provenance(
                gate_name="minimum_auc", threshold=0.60, deployment_intent="clinical"
            )
            == "unknown"
        )

    def test_commercial_075_NOT_anchored(self) -> None:
        """Each intent resolves only its own anchor."""
        assert (
            classify_threshold_provenance(
                gate_name="minimum_auc", threshold=0.75, deployment_intent="commercial"
            )
            == "unknown"
        )

    def test_doc_ref_commercial(self) -> None:
        ref = get_literature_anchor_doc_ref("minimum_auc", 0.60, "commercial")
        assert ref is not None and ("Hosmer" in ref or "propensity" in ref)


class TestIntentAwareGateEvaluation:
    def _state(self, intent: str, threshold: float, auc: float) -> dict:
        return {
            "deployment_intent": intent,
            "success_criteria": {"minimum_auc": threshold, "deployment_intent": intent},
            "validation_metrics": {"roc_auc": auc},
        }

    def test_commercial_model_clears_gate_at_060(self) -> None:
        res = _evaluate_absolute_threshold_gates(
            self._state("commercial", 0.60, 0.62), RegulatoryEligibilityAudit(), _TS
        )
        assert res["all_thresholds_cleared"] is True
        assert res["failures"] == []

    def test_commercial_model_below_060_fails(self) -> None:
        res = _evaluate_absolute_threshold_gates(
            self._state("commercial", 0.60, 0.58), RegulatoryEligibilityAudit(), _TS
        )
        assert res["all_thresholds_cleared"] is False

    def test_clinical_model_at_060_not_anchored(self) -> None:
        """Same 0.60 threshold + AUC 0.62 but CLINICAL intent → not eligible
        (threshold not literature-anchored for clinical)."""
        res = _evaluate_absolute_threshold_gates(
            self._state("clinical", 0.60, 0.62), RegulatoryEligibilityAudit(), _TS
        )
        assert res["all_thresholds_cleared"] is False

    def test_clinical_model_clears_at_075(self) -> None:
        res = _evaluate_absolute_threshold_gates(
            self._state("clinical", 0.75, 0.80), RegulatoryEligibilityAudit(), _TS
        )
        assert res["all_thresholds_cleared"] is True

    def test_intent_falls_back_to_success_criteria_stamp(self) -> None:
        """When top-level deployment_intent is absent, the success_criteria
        stamp drives the anchor resolution."""
        state = {
            "success_criteria": {"minimum_auc": 0.60, "deployment_intent": "commercial"},
            "validation_metrics": {"roc_auc": 0.62},
        }
        res = _evaluate_absolute_threshold_gates(state, RegulatoryEligibilityAudit(), _TS)
        assert res["all_thresholds_cleared"] is True
