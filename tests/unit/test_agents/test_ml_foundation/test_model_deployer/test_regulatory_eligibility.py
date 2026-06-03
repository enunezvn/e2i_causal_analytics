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
from uuid import uuid4

import pytest

from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    N1_REQUIRED_REGULATORY_GATES,
    _evaluate_regulatory_eligibility,
    validate_promotion,
)
from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

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


# --------------------------------------------------------------------------- #
# Codex N1-H2: validate_promotion blocks eligibility for non-literature-      #
# anchored thresholds. The ``_evaluate_absolute_threshold_gates`` helper      #
# classifies the (gate, threshold) pair against the literature registry      #
# and SKIPS gates that don't match.                                          #
# --------------------------------------------------------------------------- #


class TestN1H2ValidatePromotionRejectsNonLiteratureThresholds:
    """Pre-fix: a relaxed ``minimum_auc=0.50`` could pass because the
    evaluator only checked value >= threshold.
    Post-fix: the gate is SKIPPED because 0.50 is not the literature-
    anchored anchor (0.75)."""

    @pytest.mark.asyncio
    async def test_relaxed_minimum_auc_skipped_and_eligibility_denied(self) -> None:
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            # Relaxed threshold below the literature anchor.
            "success_criteria": {"minimum_auc": 0.50},
            "validation_metrics": {
                "roc_auc": 0.55,  # technically clears 0.50
                "regulatory_eligibility_audit": {},
            },
        }
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "skipped"
        assert minimum_auc_entry.get("reason") == "non_literature_threshold"
        failures = result["regulatory_eligibility_failures"]
        assert any("not in literature-anchored registry" in f for f in failures)

    @pytest.mark.asyncio
    async def test_anchor_match_keeps_eligibility(self) -> None:
        """Sanity check: the anchor value (0.75) still passes."""
        state = _state_with_passing_minimum_auc()
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is True
        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "pass"
        assert minimum_auc_entry.get("threshold_provenance") == "literature_anchored"

    @pytest.mark.asyncio
    async def test_caller_declared_literature_anchored_does_not_launder(self) -> None:
        """Caller cannot pass ``threshold_provenance.minimum_auc =
        "literature_anchored"`` to relax the gate — the classifier
        cross-checks against the registry."""
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            "success_criteria": {"minimum_auc": 0.50},
            "threshold_provenance": {"minimum_auc": "literature_anchored"},
            "validation_metrics": {
                "roc_auc": 0.55,
                "regulatory_eligibility_audit": {},
            },
        }
        result = await validate_promotion(state)
        assert result["regulatory_eligible"] is False


# --------------------------------------------------------------------------- #
# Codex N1-H3: leftover regulatory_adaptation_entry — fail closed.            #
# When state has a regulatory_adaptation_entry payload that hasn't been       #
# aggregated into adaptation_history, the deployer must refuse to grant       #
# regulatory_eligible=True.                                                   #
# --------------------------------------------------------------------------- #


class TestN1H3LeftoverAdaptationEntry:
    @pytest.mark.asyncio
    async def test_leftover_entry_blocks_eligibility(self) -> None:
        """A regulatory_adaptation_entry in state without a matching
        entry in adaptation_history fails closed."""
        leftover_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        state = _state_with_passing_minimum_auc()
        # Simulate orchestrator failing to aggregate the entry into
        # validation_metrics["regulatory_eligibility_audit"][
        # "adaptation_history"].
        state["regulatory_adaptation_entry"] = leftover_entry
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        assert "regulatory_leftover_adaptation_entries" in result
        leftovers = result["regulatory_leftover_adaptation_entries"]
        assert len(leftovers) == 1
        assert leftovers[0] == leftover_entry

        failures = result["regulatory_eligibility_failures"]
        assert any("leftover regulatory_adaptation_entry" in f for f in failures)

    @pytest.mark.asyncio
    async def test_leftover_entry_with_passing_thresholds_becomes_candidate(self) -> None:
        """When thresholds clear AND there's a leftover entry, the model
        is flagged as adapted_regulatory_candidate (would be eligible if
        cohort confirms)."""
        leftover_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        state = _state_with_passing_minimum_auc()
        state["regulatory_adaptation_entry"] = leftover_entry
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        assert result["adapted_regulatory_candidate"] is True

    @pytest.mark.asyncio
    async def test_ingested_entry_does_not_count_as_leftover(self) -> None:
        """If state's entry IS already in adaptation_history (matched on
        commit_sha + gate_name + timestamp), the deployer recognizes the
        ingestion and only the standard adaptation_history denial logic
        applies."""
        entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [entry],  # ingested
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        state["regulatory_adaptation_entry"] = entry  # same key
        result = await validate_promotion(state)

        # Not eligible because adaptation_history non-empty.
        assert result["regulatory_eligible"] is False
        # But the leftover key should NOT be present (entry is ingested).
        assert "regulatory_leftover_adaptation_entries" not in result
        # And the failure should be the standard adaptation_history one.
        failures = result["regulatory_eligibility_failures"]
        assert any("adaptation_history" in f for f in failures)
        # NOT the leftover failure.
        assert not any("leftover" in f for f in failures)

    @pytest.mark.asyncio
    async def test_no_state_entry_no_leftover(self) -> None:
        """The standard happy path — no state entry, no leftover."""
        state = _state_with_passing_minimum_auc()
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is True
        assert "regulatory_leftover_adaptation_entries" not in result

    @pytest.mark.asyncio
    async def test_list_of_leftover_entries(self) -> None:
        """Future-proof: accept list-shape regulatory_adaptation_entry."""
        entries = [
            {
                "commit_sha": "abc123",
                "justification_doc": "doc",
                "gate_name": "minimum_auc",
                "before_threshold": 0.85,
                "after_threshold": 0.75,
                "timestamp": "t1",
            },
            {
                "commit_sha": "def456",
                "justification_doc": "doc",
                "gate_name": "minimum_auc",
                "before_threshold": 0.85,
                "after_threshold": 0.75,
                "timestamp": "t2",
            },
        ]
        state = _state_with_passing_minimum_auc()
        state["regulatory_adaptation_entry"] = entries
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        assert len(result["regulatory_leftover_adaptation_entries"]) == 2

    @pytest.mark.asyncio
    async def test_partial_ingestion_surfaces_remaining_leftover(self) -> None:
        """If list-shape entries are partially ingested, only the
        un-ingested ones surface as leftovers."""
        ingested = {
            "commit_sha": "abc123",
            "justification_doc": "doc",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "t1",
        }
        unmingested = {
            "commit_sha": "def456",
            "justification_doc": "doc",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "t2",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [ingested],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        state["regulatory_adaptation_entry"] = [ingested, unmingested]
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        leftovers = result["regulatory_leftover_adaptation_entries"]
        assert len(leftovers) == 1
        assert leftovers[0] == unmingested


# --------------------------------------------------------------------------- #
# Codex N1-H3 pass-2 + new MED: canonical-hash matching catches tampered      #
# payloads. The pre-fix 3-tuple (commit_sha, gate_name, timestamp) match      #
# treated a payload with the same identity fields but altered                  #
# before_threshold / after_threshold / justification_doc as ingested.         #
# Post-fix: sha256 over the full canonical entry → any field tampering         #
# invalidates the match and the entry surfaces as a leftover.                  #
# --------------------------------------------------------------------------- #


class TestN1H3CanonicalHashMatching:
    @pytest.mark.asyncio
    async def test_tampered_before_threshold_detected_as_unmatched(self) -> None:
        """A state payload with the same key fields (commit_sha,
        gate_name, timestamp) but altered ``before_threshold`` MUST be
        detected as a leftover — pre-fix the 3-tuple match would treat
        it as ingested."""
        ingested_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,  # original
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        # Same identity fields (commit_sha, gate_name, timestamp) — but
        # the threshold delta is tampered. Pre-fix this would silently
        # match. Post-fix the canonical hash differs.
        tampered_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.99,  # TAMPERED
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [ingested_entry],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        state["regulatory_adaptation_entry"] = tampered_entry
        result = await validate_promotion(state)

        # Eligibility denied — the tampered entry is a leftover.
        assert result["regulatory_eligible"] is False
        # And the leftover surfaces explicitly so the operator sees it.
        assert "regulatory_leftover_adaptation_entries" in result
        leftovers = result["regulatory_leftover_adaptation_entries"]
        assert len(leftovers) == 1
        assert leftovers[0] == tampered_entry

    @pytest.mark.asyncio
    async def test_tampered_after_threshold_detected_as_unmatched(self) -> None:
        """A tampered ``after_threshold`` is also caught by canonical-
        hash matching."""
        ingested_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,  # original
            "timestamp": "2026-05-10T00:00:00",
        }
        tampered_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.50,  # TAMPERED — looser threshold
            "timestamp": "2026-05-10T00:00:00",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [ingested_entry],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        state["regulatory_adaptation_entry"] = tampered_entry
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        leftovers = result["regulatory_leftover_adaptation_entries"]
        assert len(leftovers) == 1
        assert leftovers[0] == tampered_entry

    @pytest.mark.asyncio
    async def test_tampered_justification_doc_detected_as_unmatched(self) -> None:
        """A tampered ``justification_doc`` invalidates the match — the
        whole canonical form is hashed."""
        ingested_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        tampered_entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/forged_signoff.md",  # TAMPERED
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [ingested_entry],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        state["regulatory_adaptation_entry"] = tampered_entry
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        leftovers = result["regulatory_leftover_adaptation_entries"]
        assert len(leftovers) == 1
        assert leftovers[0] == tampered_entry

    @pytest.mark.asyncio
    async def test_byte_for_byte_match_recognized_as_ingested(self) -> None:
        """Sanity: a byte-for-byte identical payload IS recognized as
        ingested under canonical-hash matching."""
        entry = {
            "commit_sha": "abc123",
            "justification_doc": "docs/relaxation_signoff.md",
            "gate_name": "minimum_auc",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "2026-05-10T00:00:00",
        }
        prior_audit = {
            "gate_history": [],
            "adaptation_history": [entry],
        }
        state = _state_with_passing_minimum_auc(audit=prior_audit)
        state["regulatory_adaptation_entry"] = entry  # exact same dict
        result = await validate_promotion(state)

        # Not eligible (adaptation_history non-empty) — but no leftover.
        assert result["regulatory_eligible"] is False
        assert "regulatory_leftover_adaptation_entries" not in result

    def test_compute_canonical_entry_hash_deterministic(self) -> None:
        """The hash is a deterministic function of the canonical
        projection — two equal entries produce the same hash regardless
        of dict-key order in the source."""
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            compute_canonical_entry_hash,
        )

        entry_a = {
            "commit_sha": "abc",
            "justification_doc": "d",
            "gate_name": "g",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "t",
        }
        # Same fields, different key insertion order.
        entry_b = {
            "timestamp": "t",
            "after_threshold": 0.75,
            "before_threshold": 0.85,
            "gate_name": "g",
            "justification_doc": "d",
            "commit_sha": "abc",
        }
        assert compute_canonical_entry_hash(entry_a) == compute_canonical_entry_hash(entry_b)

    def test_compute_canonical_entry_hash_differs_on_tamper(self) -> None:
        """Any field-level mutation produces a different hash — covers
        the canonical fields exhaustively."""
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            ADAPTATION_ENTRY_CANONICAL_FIELDS,
            compute_canonical_entry_hash,
        )

        base = {
            "commit_sha": "abc",
            "justification_doc": "d",
            "gate_name": "g",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "t",
        }
        base_hash = compute_canonical_entry_hash(base)
        # Mutate each canonical field in turn — every mutation changes
        # the hash. This is the property that closes the new MED.
        for field_name in ADAPTATION_ENTRY_CANONICAL_FIELDS:
            tampered = dict(base)
            tampered[field_name] = "TAMPERED"
            assert compute_canonical_entry_hash(tampered) != base_hash, (
                f"Tampering field '{field_name}' did not change canonical hash"
            )

    def test_compute_canonical_entry_hash_ignores_extra_fields(self) -> None:
        """Extra (non-canonical) fields do NOT affect the hash — only
        the canonical projection is hashed. This lets the audit roundtrip
        with auxiliary metadata that isn't part of the identity."""
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            compute_canonical_entry_hash,
        )

        base = {
            "commit_sha": "abc",
            "justification_doc": "d",
            "gate_name": "g",
            "before_threshold": 0.85,
            "after_threshold": 0.75,
            "timestamp": "t",
        }
        with_extra = dict(base)
        with_extra["extra_audit_metadata"] = "ignored"
        assert compute_canonical_entry_hash(base) == compute_canonical_entry_hash(with_extra)


# --------------------------------------------------------------------------- #
# Codex N1-M2: malformed metric (non-numeric value/threshold) falls into a    #
# SKIPPED gate evaluation with reason="malformed_metric" — does NOT collapse  #
# into validate_promotion's broad except path.                                 #
# --------------------------------------------------------------------------- #


class TestN1M2MalformedMetricHandling:
    @pytest.mark.asyncio
    async def test_string_value_skipped_with_malformed_metric_reason(self) -> None:
        """A non-numeric string value triggers SKIPPED + malformed_metric
        reason — must not collapse into the broad validate_promotion
        exception path."""
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            "success_criteria": {"minimum_auc": 0.75},
            "validation_metrics": {
                "roc_auc": "not-a-number",
                "regulatory_eligibility_audit": {},
            },
        }
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        # Promotion validation itself should NOT error.
        assert (
            "error_type" not in result or result.get("error_type") != "promotion_validation_error"
        )

        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "skipped"
        assert minimum_auc_entry.get("reason") == "malformed_metric"

        failures = result["regulatory_eligibility_failures"]
        assert any("malformed metric" in f for f in failures)

    @pytest.mark.asyncio
    async def test_dict_value_skipped_with_malformed_metric_reason(self) -> None:
        """A dict value (non-numeric, non-string) triggers SKIPPED +
        malformed_metric reason."""
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            "success_criteria": {"minimum_auc": 0.75},
            "validation_metrics": {
                "roc_auc": {"oops": "dict"},
                "regulatory_eligibility_audit": {},
            },
        }
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        assert (
            "error_type" not in result or result.get("error_type") != "promotion_validation_error"
        )

        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "skipped"
        assert minimum_auc_entry.get("reason") == "malformed_metric"

    @pytest.mark.asyncio
    async def test_list_value_skipped_with_malformed_metric_reason(self) -> None:
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            "success_criteria": {"minimum_auc": 0.75},
            "validation_metrics": {
                "roc_auc": [0.80, 0.85],  # list, not numeric
                "regulatory_eligibility_audit": {},
            },
        }
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        gate_history = result["regulatory_eligibility_audit"]["gate_history"]
        minimum_auc_entry = next(e for e in gate_history if e["gate_name"] == "minimum_auc")
        assert minimum_auc_entry["outcome"] == "skipped"
        assert minimum_auc_entry.get("reason") == "malformed_metric"

    @pytest.mark.asyncio
    async def test_promotion_allowed_unaffected_by_malformed_metric(self) -> None:
        """Eligibility is signal-only — even on malformed metric, the
        promotion validation path itself returns promotion_allowed=True
        for staging promotions."""
        state: Dict[str, Any] = {
            "current_stage": "None",
            "target_environment": "staging",
            "success_criteria": {"minimum_auc": 0.75},
            "validation_metrics": {
                "roc_auc": "not-a-number",
                "regulatory_eligibility_audit": {},
            },
        }
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        # promotion_allowed for staging is signal-only — eligibility
        # denial doesn't block.
        assert result["promotion_allowed"] is True


# --------------------------------------------------------------------------- #
# Gate N1 regression (state-contract / regulatory-integrity):                 #
# the backstop must be ABLE TO FIRE through the real LangGraph channel        #
# boundary. The other N1-H3 tests above pass a raw dict straight to           #
# ``validate_promotion`` — they bypass ``ModelDeployerState`` validation, so  #
# they passed even while ``regulatory_adaptation_entry`` was UNDECLARED on    #
# the state and ``extra="ignore"`` silently dropped it on every real run.     #
# These tests exercise the field through ``ModelDeployerState(...).model_dump #
# ()`` (the faithful channel round-trip) so a regression that re-drops the    #
# field is caught.                                                            #
# --------------------------------------------------------------------------- #


def _roundtrip_state(
    *,
    regulatory_adaptation_entry: Any = None,
    scope_spec: Dict[str, Any] | None = None,
    audit: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build deployer state the way LangGraph does: construct the typed
    ``ModelDeployerState`` (which applies ``extra="ignore"`` at the channel
    boundary) and ``model_dump()`` it back to the dict the nodes consume.

    ``success_criteria`` is not a declared ModelDeployerState field (the nodes
    read it off raw state), so it is added after the dump — mirroring how the
    runtime threads success criteria alongside the validated channel."""
    kwargs: Dict[str, Any] = {
        "audit_workflow_id": uuid4(),
        "current_stage": "None",
        "target_environment": "staging",
        "validation_metrics": {
            "roc_auc": 0.80,
            "regulatory_eligibility_audit": audit or {},
        },
    }
    if regulatory_adaptation_entry is not None:
        kwargs["regulatory_adaptation_entry"] = regulatory_adaptation_entry
    if scope_spec is not None:
        kwargs["scope_spec"] = scope_spec
    state = ModelDeployerState(**kwargs).model_dump()
    state["success_criteria"] = {"minimum_auc": 0.75}
    return state


_LEFTOVER_ENTRY: Dict[str, Any] = {
    "commit_sha": "abc123",
    "justification_doc": "data_preparer/leakage_remediation",
    "gate_name": "leakage_remediation_feature_drop",
    "before_threshold": {"leaked_features_count": 2, "dropped_features": ["f1", "f2"]},
    "after_threshold": {"remediated_features_count": 3, "added_features": ["g1"]},
    "timestamp": "2026-06-03T00:00:00",
}


class TestN1H3StateContractRoundTrip:
    """The backstop must survive the ModelDeployerState channel boundary."""

    def test_regulatory_adaptation_entry_survives_state_validation(self) -> None:
        """Declared field round-trips through ModelDeployerState. Pre-fix the
        field was undeclared and ``extra="ignore"`` dropped it (the
        deployer backstop then always read ``None``)."""
        state = ModelDeployerState(
            audit_workflow_id=uuid4(),
            regulatory_adaptation_entry=_LEFTOVER_ENTRY,
        )
        # Guard against vacuous test: the field must really be carried.
        assert state.regulatory_adaptation_entry == _LEFTOVER_ENTRY
        dumped = state.model_dump()
        assert "regulatory_adaptation_entry" in dumped
        assert dumped["regulatory_adaptation_entry"] == _LEFTOVER_ENTRY

    @pytest.mark.asyncio
    async def test_leftover_entry_blocks_eligibility_through_roundtrip(self) -> None:
        """A leakage-remediated model whose adaptation entry was NOT ingested
        into adaptation_history must NOT be granted regulatory_eligible=True
        when state is built via the faithful channel round-trip. Pre-fix the
        round-trip dropped the entry and eligibility was wrongly granted."""
        state = _roundtrip_state(regulatory_adaptation_entry=_LEFTOVER_ENTRY)

        # Faithfulness guard: the round-trip really preserved the entry.
        assert state.get("regulatory_adaptation_entry") == _LEFTOVER_ENTRY

        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        assert result["adapted_regulatory_candidate"] is True
        assert "regulatory_leftover_adaptation_entries" in result
        assert result["regulatory_leftover_adaptation_entries"] == [_LEFTOVER_ENTRY]
        failures = result["regulatory_eligibility_failures"]
        assert any("leftover regulatory_adaptation_entry" in f for f in failures)

    @pytest.mark.asyncio
    async def test_leftover_entry_in_scope_spec_blocks_eligibility(self) -> None:
        """The deployer agent threads scope_spec (not arbitrary top-level keys)
        onto its initial state. The pipeline therefore nests the adaptation
        entry under scope_spec — the backstop must read it from that carrier
        and fail closed."""
        scope_spec = {
            "feature_manifest_source": "optum",
            "regulatory_adaptation_entry": _LEFTOVER_ENTRY,
        }
        state = _roundtrip_state(scope_spec=scope_spec)

        # Faithfulness guard: scope_spec carrier survived the round-trip.
        assert (
            state["scope_spec"]["regulatory_adaptation_entry"] == _LEFTOVER_ENTRY
        )
        # And the top-level channel is empty — proving the scope_spec carrier
        # alone drives the backstop.
        assert state.get("regulatory_adaptation_entry") is None

        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        assert result["adapted_regulatory_candidate"] is True
        assert result["regulatory_leftover_adaptation_entries"] == [_LEFTOVER_ENTRY]

    @pytest.mark.asyncio
    async def test_same_entry_on_both_carriers_reported_once(self) -> None:
        """The same entry threaded via BOTH the top-level channel and
        scope_spec must be de-duplicated (reported as a single leftover)."""
        scope_spec = {"regulatory_adaptation_entry": _LEFTOVER_ENTRY}
        state = _roundtrip_state(
            regulatory_adaptation_entry=_LEFTOVER_ENTRY,
            scope_spec=scope_spec,
        )

        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is False
        assert len(result["regulatory_leftover_adaptation_entries"]) == 1

    @pytest.mark.asyncio
    async def test_clean_model_still_eligible_through_roundtrip(self) -> None:
        """No leakage remediation → no adaptation entry on either carrier →
        eligibility is granted (no false-positive block)."""
        state = _roundtrip_state(scope_spec={"feature_manifest_source": "optum"})
        result = await validate_promotion(state)

        assert result["regulatory_eligible"] is True
        assert "regulatory_leftover_adaptation_entries" not in result
