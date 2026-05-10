"""Gate N1 (plan v4 §2) — validate_promotion regulatory-eligibility checks.

Pins the three preconditions ``validate_promotion`` evaluates before
granting ``regulatory_eligible=True`` per codex-rescue HIGH-3:

  1. All literature-anchored absolute thresholds clear (e.g.
     ``minimum_auc``).
  2. ``adaptation_history == []`` — no adaptive relaxation during the
     model's lifecycle.
  3. ``gate_history`` shows EVERY required gate evaluated against its
     literature-anchored threshold (no advisory bypasses).

Also pins the complementary flag ``adapted_regulatory_candidate=True``
which signals "would be eligible if external validation cohort
confirms" — set when (1) holds but (2) does not.

Test coverage maps to the spec's required tests 3, 4, 5, 6:

  * TEST 3: regulatory_eligible=False when adaptation_history non-empty
    even if absolute thresholds met.
  * TEST 4: regulatory_eligible=True when all 3 preconditions met.
  * TEST 5: adapted_regulatory_candidate=True when (1) met but (2) not.
  * TEST 6: gate_history missing a gate → precondition fails.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    N1_REQUIRED_REGULATORY_GATES,
    _evaluate_regulatory_eligibility,
    validate_promotion,
)

# --------------------------------------------------------------------------- #
# Module-level fixtures                                                       #
# --------------------------------------------------------------------------- #


def _state_with_passing_minimum_auc(
    *,
    audit: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a state dict where the literature-anchored minimum_auc
    threshold (0.75) is cleared by validation_metrics roc_auc=0.80."""
    return {
        "current_stage": "None",
        "target_environment": "staging",
        "success_criteria": {"minimum_auc": 0.75},
        "validation_metrics": {
            "roc_auc": 0.80,
            "regulatory_eligibility_audit": audit or {},
        },
    }


def _state_with_failing_minimum_auc() -> Dict[str, Any]:
    return {
        "current_stage": "None",
        "target_environment": "staging",
        "success_criteria": {"minimum_auc": 0.75},
        "validation_metrics": {
            "roc_auc": 0.50,  # below 0.75
            "regulatory_eligibility_audit": {},
        },
    }


# --------------------------------------------------------------------------- #
# Module constants — pin the required-gate list.                              #
# --------------------------------------------------------------------------- #


class TestN1RequiredGates:
    def test_required_gates_includes_minimum_auc(self) -> None:
        assert "minimum_auc" in N1_REQUIRED_REGULATORY_GATES

    def test_required_gates_is_list(self) -> None:
        assert isinstance(N1_REQUIRED_REGULATORY_GATES, list)


# --------------------------------------------------------------------------- #
# TEST 4 (spec): regulatory_eligible=True when all 3 preconditions met.       #
# --------------------------------------------------------------------------- #


class TestRegulatoryEligibleHappyPath:
    @pytest.mark.asyncio
    async def test_regulatory_eligible_true_when_all_preconditions_met(self) -> None:
        state = _state_with_passing_minimum_auc()
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is True
        assert result["adapted_regulatory_candidate"] is False
        # The gate_history now records the minimum_auc evaluation as "pass".
        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        gate_names = [e["gate_name"] for e in gate_history]
        assert "minimum_auc" in gate_names
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "pass"
        assert minimum_auc_entry["threshold"] == 0.75
        assert minimum_auc_entry["value"] == 0.80

    @pytest.mark.asyncio
    async def test_eligibility_evaluation_does_not_block_promotion(self) -> None:
        """The eligibility flag is signal-only — promotion_allowed unchanged."""
        state = _state_with_passing_minimum_auc()
        result = await validate_promotion(state)

        assert result["promotion_allowed"] is True
        # And eligibility is granted.
        assert result["regulatory_eligible"] is True


# --------------------------------------------------------------------------- #
# TEST 3 (spec): regulatory_eligible=False when adaptation_history non-empty. #
# --------------------------------------------------------------------------- #


class TestRegulatoryEligibleWithAdaptation:
    @pytest.mark.asyncio
    async def test_eligible_false_when_adaptation_history_non_empty(self) -> None:
        """ANY adaptation entry disqualifies eligibility — even if all
        absolute thresholds clear at promotion time."""
        adaptation_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [adaptation_entry],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        result = await validate_promotion(state)

        # Absolute threshold STILL passes (0.80 >= 0.75).
        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "pass"

        # But eligibility is denied because adaptation_history is non-empty.
        assert result["regulatory_eligible"] is False

        # Failures list explains why.
        assert "regulatory_eligibility_failures" in result
        failures = result["regulatory_eligibility_failures"]
        assert any("adaptation_history" in f for f in failures)


# --------------------------------------------------------------------------- #
# TEST 5 (spec): adapted_regulatory_candidate=True when (1) but not (2).      #
# --------------------------------------------------------------------------- #


class TestAdaptedRegulatoryCandidate:
    @pytest.mark.asyncio
    async def test_candidate_true_when_thresholds_met_but_adaptation_non_empty(
        self,
    ) -> None:
        adaptation_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [adaptation_entry],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        result = await validate_promotion(state)

        # Absolute thresholds met → candidate=True.
        assert result["adapted_regulatory_candidate"] is True
        # But not eligible (adaptation_history non-empty).
        assert result["regulatory_eligible"] is False

    @pytest.mark.asyncio
    async def test_not_candidate_when_thresholds_failing(self) -> None:
        """If absolute thresholds fail, candidate also False — neither
        flag flips True."""
        adaptation_entry = {
            "commit_sha": "abc",
            "justification_doc": "doc",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "t",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [adaptation_entry],
        }
        state = _state_with_failing_minimum_auc()
        state["validation_metrics"]["regulatory_eligibility_audit"] = prior_audit
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        assert result["adapted_regulatory_candidate"] is False


# --------------------------------------------------------------------------- #
# TEST 6 (spec): gate_history missing a gate → precondition fails.            #
# --------------------------------------------------------------------------- #


class TestGateHistoryMissingGate:
    @pytest.mark.asyncio
    async def test_precondition_fails_when_minimum_auc_missing_from_metrics(
        self,
    ) -> None:
        """When validation_metrics lacks roc_auc, the gate evaluator
        records ``outcome="skipped"`` — and eligibility CANNOT be
        granted (skipped != pass)."""
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            "success_criteria": {"minimum_auc": 0.75},
            # validation_metrics missing roc_auc entirely.
            "validation_metrics": {"regulatory_eligibility_audit": {}},
        }
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "skipped"

        failures = result["regulatory_eligibility_failures"]
        assert any("'minimum_auc' skipped" in f for f in failures)

    @pytest.mark.asyncio
    async def test_precondition_fails_when_threshold_missing_from_criteria(
        self,
    ) -> None:
        """When success_criteria lacks ``minimum_auc``, the gate is
        ``"skipped"`` and eligibility is denied."""
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            "success_criteria": {},  # no minimum_auc
            "validation_metrics": {
                "roc_auc": 0.80,
                "regulatory_eligibility_audit": {},
            },
        }
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False


# --------------------------------------------------------------------------- #
# Auc_roc / roc_auc dual-key handling — backward-compat with legacy schema.   #
# --------------------------------------------------------------------------- #


class TestRocAucKeyAliasing:
    @pytest.mark.asyncio
    async def test_auc_roc_canonical_key_is_read_when_roc_auc_missing(self) -> None:
        """``MetricsSchema`` declares ``auc_roc`` as the canonical name;
        ``roc_auc`` is the modern producer key. The eligibility evaluator
        reads BOTH so legacy dict inputs (without the alias) still
        resolve."""
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            "success_criteria": {"minimum_auc": 0.75},
            "validation_metrics": {
                "auc_roc": 0.80,  # legacy key only
                "regulatory_eligibility_audit": {},
            },
        }
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is True
        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "pass"


# --------------------------------------------------------------------------- #
# _evaluate_regulatory_eligibility direct unit tests.                          #
# --------------------------------------------------------------------------- #


class TestEvaluateRegulatoryEligibilityDirect:
    """Unit-test the helper directly so its contract is pinned independent
    of validate_promotion's wiring."""

    def test_helper_returns_audit_dict(self) -> None:
        state = _state_with_passing_minimum_auc()
        result = _evaluate_regulatory_eligibility(state)

        assert "regulatory_eligible" in result
        assert "adapted_regulatory_candidate" in result
        assert "regulatory_eligibility_audit" in result
        assert isinstance(result["regulatory_eligibility_audit"], dict)
        assert "gate_history" in result["regulatory_eligibility_audit"]
        assert "adaptation_history" in result["regulatory_eligibility_audit"]

    def test_helper_appends_to_existing_audit(self) -> None:
        prior_audit = {
            "gate_history": [
                {
                    "timestamp": "earlier",
                    "gate_name": "earlier_gate",
                    "threshold": 0.5,
                    "value": 0.6,
                    "outcome": "pass",
                }
            ],
            "adaptation_history": [],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        result = _evaluate_regulatory_eligibility(state)

        # Both the old + new entries are present.
        gate_names = [
            e["gate_name"] for e in result["regulatory_eligibility_audit"]["gate_history"]
        ]
        assert "earlier_gate" in gate_names
        assert "minimum_auc" in gate_names

    def test_helper_does_not_mutate_input_state(self) -> None:
        """The eligibility evaluator must not mutate the caller's audit
        in place — the runtime guard's ``__setitem__`` would catch in-
        place rewrites, but the helper writes via append + returns a
        deep-copy."""
        prior_audit: Dict[str, Any] = {
            "gate_history": [],
            "adaptation_history": [],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        _ = _evaluate_regulatory_eligibility(state)

        # The original prior_audit dict in state is unchanged at the
        # source-of-truth level. Note: validation_metrics is a dict ref;
        # the helper reads from it without writing back.
        original_audit = state["validation_metrics"]["regulatory_eligibility_audit"]
        assert original_audit["gate_history"] == []
        assert original_audit["adaptation_history"] == []
