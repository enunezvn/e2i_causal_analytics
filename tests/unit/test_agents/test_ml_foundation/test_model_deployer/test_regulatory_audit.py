"""Gate N1 (plan v4 §2) — RegulatoryEligibilityAudit primitives.

Pins the append-only invariant + the eligibility-helper logic that
``validate_promotion`` reads. The runtime guard is the load-bearing
mitigation per codex-rescue HIGH-3: without these tests pinning
``__setitem__ → raise``, a future refactor could silently re-enable
in-place mutation and let a caller blank an adaptation_history right
before the eligibility check.

Test coverage:

  * ``__setitem__`` → ``RegulatoryAuditMutationError`` (TEST 1 in spec).
  * ``append_gate_evaluation`` / ``append_adaptation`` succeed (TEST 2).
  * ``__getitem__`` returns deep-copy snapshot (mutation isolation).
  * ``to_dict`` / ``from_dict`` round-trip preserves contents.
  * ``is_regulatory_eligible`` precondition logic.
  * ``is_adapted_regulatory_candidate`` complementary logic.
"""

from __future__ import annotations

import pytest

from src.agents.ml_foundation.model_deployer.regulatory_audit import (
    RegulatoryAuditMutationError,
    RegulatoryEligibilityAudit,
    is_adapted_regulatory_candidate,
    is_regulatory_eligible,
)

# --------------------------------------------------------------------------- #
# TEST 1 (spec): __setitem__ raises on existing entry.                        #
# --------------------------------------------------------------------------- #


class TestRegulatoryAuditMutationGuard:
    """The runtime guard's ``__setitem__`` must raise unconditionally."""

    def test_setitem_on_gate_history_raises(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="2026-05-10T00:00:00",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
        )
        with pytest.raises(RegulatoryAuditMutationError):
            audit["gate_history"] = []  # type: ignore[index]

    def test_setitem_on_adaptation_history_raises(self) -> None:
        audit = RegulatoryEligibilityAudit()
        with pytest.raises(RegulatoryAuditMutationError):
            audit["adaptation_history"] = [{"foo": "bar"}]  # type: ignore[index]

    def test_setitem_on_unknown_field_raises(self) -> None:
        audit = RegulatoryEligibilityAudit()
        with pytest.raises(RegulatoryAuditMutationError):
            audit["nonexistent_field"] = []  # type: ignore[index]

    def test_setitem_error_message_directs_to_append_api(self) -> None:
        audit = RegulatoryEligibilityAudit()
        with pytest.raises(RegulatoryAuditMutationError) as exc_info:
            audit["gate_history"] = []  # type: ignore[index]
        msg = str(exc_info.value)
        assert "append_gate_evaluation" in msg
        assert "append_adaptation" in msg
        assert "append-only" in msg


# --------------------------------------------------------------------------- #
# TEST 2 (spec): append on both list fields succeeds.                         #
# --------------------------------------------------------------------------- #


class TestRegulatoryAuditAppend:
    """``append_gate_evaluation`` and ``append_adaptation`` are the only
    sanctioned mutation surface."""

    def test_append_gate_evaluation_grows_gate_history(self) -> None:
        audit = RegulatoryEligibilityAudit()
        # Codex-rescue N1-H1: gate_history is a read-only tuple snapshot.
        assert audit.gate_history == ()

        audit.append_gate_evaluation(
            timestamp="2026-05-10T00:00:00",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
        )
        assert len(audit.gate_history) == 1
        entry = audit.gate_history[0]
        assert entry["timestamp"] == "2026-05-10T00:00:00"
        assert entry["gate_name"] == "minimum_auc"
        assert entry["threshold"] == 0.75
        assert entry["value"] == 0.80
        assert entry["outcome"] == "pass"

    def test_append_gate_evaluation_multiple_times(self) -> None:
        audit = RegulatoryEligibilityAudit()
        for i in range(5):
            audit.append_gate_evaluation(
                timestamp=f"2026-05-10T00:00:0{i}",
                gate_name=f"gate_{i}",
                threshold=0.5 + i * 0.05,
                value=0.6 + i * 0.05,
                outcome="pass",
            )
        assert len(audit.gate_history) == 5
        # Entries are appended in order; the latest is the highest index.
        assert audit.gate_history[-1]["gate_name"] == "gate_4"

    def test_append_adaptation_grows_adaptation_history(self) -> None:
        audit = RegulatoryEligibilityAudit()
        # Codex-rescue N1-H1: adaptation_history is a read-only tuple snapshot.
        assert audit.adaptation_history == ()

        audit.append_adaptation(
            commit_sha="abc1234",
            justification_doc="docs/relaxation_signoff.md",
            gate_name="minimum_auc",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="2026-05-10T00:00:00",
        )
        assert len(audit.adaptation_history) == 1
        entry = audit.adaptation_history[0]
        assert entry["commit_sha"] == "abc1234"
        assert entry["justification_doc"] == "docs/relaxation_signoff.md"
        assert entry["gate_name"] == "minimum_auc"
        assert entry["before_threshold"] == 0.85
        assert entry["after_threshold"] == 0.75
        assert entry["timestamp"] == "2026-05-10T00:00:00"

    def test_append_deep_copies_threshold_and_value(self) -> None:
        """Caller-supplied mutable threshold/value must not leak references."""
        audit = RegulatoryEligibilityAudit()
        threshold_dict = {"min": 0.75, "max": 1.0}
        value_dict = {"observed": 0.80}

        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="g",
            threshold=threshold_dict,
            value=value_dict,
            outcome="pass",
        )

        # Mutate the caller-side dicts post-append.
        threshold_dict["min"] = 0.50
        value_dict["observed"] = 0.99

        # The audit copy is unaffected.
        entry = audit.gate_history[0]
        assert entry["threshold"]["min"] == 0.75
        assert entry["value"]["observed"] == 0.80


# --------------------------------------------------------------------------- #
# Mapping-like read access — deep-copy snapshot semantics.                    #
# --------------------------------------------------------------------------- #


class TestRegulatoryAuditRead:
    def test_getitem_returns_gate_history(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        snapshot = audit["gate_history"]
        assert len(snapshot) == 1
        assert snapshot[0]["gate_name"] == "g"

    def test_getitem_returns_deep_copy(self) -> None:
        """Mutating the snapshot must not leak back into the guarded state."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        snapshot = audit["gate_history"]
        # Codex-rescue N1-H1: snapshot is a tuple; mutating individual
        # entry dicts must not leak back. Tuples themselves are
        # immutable so the .clear() / slice-assign attacks are
        # syntactically blocked.
        snapshot[0]["outcome"] = "fail"  # mutate the dict inside the tuple

        # The audit's actual list is unchanged — outcome is still pass.
        assert len(audit.gate_history) == 1
        assert audit.gate_history[0]["outcome"] == "pass"

    def test_getitem_unknown_key_raises_keyerror(self) -> None:
        audit = RegulatoryEligibilityAudit()
        with pytest.raises(KeyError):
            _ = audit["nonexistent"]

    def test_contains_for_known_fields(self) -> None:
        audit = RegulatoryEligibilityAudit()
        assert "gate_history" in audit
        assert "adaptation_history" in audit
        assert "nonexistent" not in audit


# --------------------------------------------------------------------------- #
# Codex-rescue N1-H1: read-only-tuple property invariants.                     #
# Pins the public-list-bypass mitigation: callers cannot ``.append()``,        #
# ``[i] = ...``, ``.clear()`` or slice-assign on the gate_history /            #
# adaptation_history attributes — they return a new tuple snapshot each        #
# read, so any in-place mutation would either raise (tuple immutability)       #
# or write to a throwaway copy.                                                #
# --------------------------------------------------------------------------- #


class TestRegulatoryAuditN1H1ReadOnlyProperties:
    """Codex N1-H1: gate_history / adaptation_history must be read-only.

    The pre-fix exposure was: ``RegulatoryEligibilityAudit`` declared
    public list fields, so a caller could call
    ``audit.gate_history.append({"outcome": "pass"})`` or
    ``audit.gate_history[0]["outcome"] = "pass"`` and bypass the
    ``__setitem__`` guard entirely. The fix: store via private fields,
    expose tuple-returning properties.
    """

    def test_gate_history_is_tuple(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        assert isinstance(audit.gate_history, tuple)

    def test_adaptation_history_is_tuple(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="g",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )
        assert isinstance(audit.adaptation_history, tuple)

    def test_gate_history_append_raises_attribute_error(self) -> None:
        """Tuples have no ``.append()`` — the in-place attack is blocked."""
        audit = RegulatoryEligibilityAudit()
        with pytest.raises(AttributeError):
            audit.gate_history.append({"foo": "bar"})  # type: ignore[attr-defined]

    def test_adaptation_history_append_raises_attribute_error(self) -> None:
        audit = RegulatoryEligibilityAudit()
        with pytest.raises(AttributeError):
            audit.adaptation_history.append({"foo": "bar"})  # type: ignore[attr-defined]

    def test_gate_history_clear_raises_attribute_error(self) -> None:
        """Tuples have no ``.clear()`` — the wipe attack is blocked."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        with pytest.raises(AttributeError):
            audit.gate_history.clear()  # type: ignore[attr-defined]

    def test_adaptation_history_clear_raises_attribute_error(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="g",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )
        with pytest.raises(AttributeError):
            audit.adaptation_history.clear()  # type: ignore[attr-defined]

    def test_gate_history_slice_assign_raises_typeerror(self) -> None:
        """Tuples reject ``__setitem__`` — slice-assign is blocked."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        with pytest.raises(TypeError):
            audit.gate_history[0] = {"foo": "bar"}  # type: ignore[index]

    def test_inner_dict_mutation_does_not_leak_back(self) -> None:
        """A dict reachable through the tuple snapshot is a deep copy —
        mutating it must not mutate the audit's stored entry."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        snapshot = audit.gate_history
        snapshot[0]["outcome"] = "fail"  # mutate the deep-copied dict
        # Re-read; the source entry is unchanged.
        assert audit.gate_history[0]["outcome"] == "pass"

    def test_subsequent_reads_independent(self) -> None:
        """Two consecutive ``audit.gate_history`` reads return distinct
        tuples whose dicts are independent deep copies."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        snap_a = audit.gate_history
        snap_b = audit.gate_history
        # Same content, different dict identities.
        assert snap_a[0] == snap_b[0]
        assert snap_a[0] is not snap_b[0]

    def test_assigning_to_property_raises_attribute_error(self) -> None:
        """Replacing the property attribute itself must fail.

        Without a setter, ``audit.gate_history = []`` raises
        ``AttributeError`` — the public list reassignment attack is
        blocked.
        """
        audit = RegulatoryEligibilityAudit()
        with pytest.raises(AttributeError):
            audit.gate_history = []  # type: ignore[misc]
        with pytest.raises(AttributeError):
            audit.adaptation_history = []  # type: ignore[misc]

    def test_to_dict_unaffected_by_snapshot_mutation(self) -> None:
        """Even if a caller mutates a tuple snapshot's dict, ``to_dict``
        returns the source's pristine state."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        snap = audit.gate_history
        snap[0]["outcome"] = "fail"  # corrupt snapshot
        # Source is intact.
        assert audit.to_dict()["gate_history"][0]["outcome"] == "pass"


class TestRegulatoryAuditN1H1FrozenEntries:
    """Codex N1-H1: individual entries are frozen dataclass-derived dicts.

    The dicts come out of frozen ``GateEvaluationEntry`` /
    ``AdaptationEntry`` instances via ``to_dict()`` so the stored
    entries are append-once. Re-imports preserve frozen status.
    """

    def test_gate_evaluation_entry_is_frozen(self) -> None:
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            GateEvaluationEntry,
        )

        entry = GateEvaluationEntry(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.outcome = "fail"  # type: ignore[misc]

    def test_adaptation_entry_is_frozen(self) -> None:
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            AdaptationEntry,
        )

        entry = AdaptationEntry(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="g",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.gate_name = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Serialization round-trip.                                                    #
# --------------------------------------------------------------------------- #


class TestRegulatoryAuditSerialization:
    def test_to_dict_returns_both_lists(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t1", gate_name="g1", threshold=0.5, value=0.6, outcome="pass"
        )
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="g1",
            before_threshold=0.7,
            after_threshold=0.5,
            timestamp="t2",
        )

        snapshot = audit.to_dict()
        assert "gate_history" in snapshot
        assert "adaptation_history" in snapshot
        assert len(snapshot["gate_history"]) == 1
        assert len(snapshot["adaptation_history"]) == 1

    def test_to_dict_returns_deep_copy(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t", gate_name="g", threshold=0.5, value=0.6, outcome="pass"
        )
        snapshot = audit.to_dict()
        snapshot["gate_history"].clear()
        # Re-read; the audit's source list is unchanged.
        assert len(audit.to_dict()["gate_history"]) == 1

    def test_from_dict_round_trip(self) -> None:
        audit_a = RegulatoryEligibilityAudit()
        audit_a.append_gate_evaluation(
            timestamp="t1", gate_name="g1", threshold=0.5, value=0.6, outcome="pass"
        )
        audit_a.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="g1",
            before_threshold=0.7,
            after_threshold=0.5,
            timestamp="t2",
        )

        payload = audit_a.to_dict()
        audit_b = RegulatoryEligibilityAudit.from_dict(payload)

        assert audit_b.to_dict() == payload

    def test_from_dict_empty_payload_yields_empty_audit(self) -> None:
        audit = RegulatoryEligibilityAudit.from_dict({})
        # Codex-rescue N1-H1: read-only tuple properties.
        assert audit.gate_history == ()
        assert audit.adaptation_history == ()

    def test_from_dict_decouples_from_source(self) -> None:
        """Mutating the source dict post-from_dict must not affect the audit."""
        source = {
            "gate_history": [
                {
                    "timestamp": "t",
                    "gate_name": "g",
                    "threshold": 0.5,
                    "value": 0.6,
                    "outcome": "pass",
                }
            ],
            "adaptation_history": [],
        }
        audit = RegulatoryEligibilityAudit.from_dict(source)

        # Corrupt the source.
        source["gate_history"].clear()
        source["adaptation_history"].append({"corrupting": True})

        # The audit's lists are unchanged.
        assert len(audit.gate_history) == 1
        # Codex-rescue N1-H1: read-only tuple property.
        assert audit.adaptation_history == ()

    def test_from_dict_rejects_non_list_gate_history(self) -> None:
        with pytest.raises(TypeError):
            RegulatoryEligibilityAudit.from_dict({"gate_history": "not-a-list"})

    def test_from_dict_rejects_non_list_adaptation_history(self) -> None:
        with pytest.raises(TypeError):
            RegulatoryEligibilityAudit.from_dict({"adaptation_history": {"oops": "dict"}})

    def test_from_dict_handles_explicit_none_for_lists(self) -> None:
        """Pydantic / JSON may serialize empty list as None — accept that
        path so checkpoint-restart doesn't blow up on a benign schema
        difference."""
        audit = RegulatoryEligibilityAudit.from_dict(
            {"gate_history": None, "adaptation_history": None}
        )
        # Codex-rescue N1-H1: read-only tuple properties.
        assert audit.gate_history == ()
        assert audit.adaptation_history == ()


# --------------------------------------------------------------------------- #
# Eligibility helper logic — feeds validate_promotion.                         #
# --------------------------------------------------------------------------- #


class TestIsRegulatoryEligible:
    """``is_regulatory_eligible`` returns True iff all 3 preconditions hold."""

    def _gate_pass(self, audit: RegulatoryEligibilityAudit, gate_name: str) -> None:
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name=gate_name,
            threshold=0.75,
            value=0.80,
            outcome="pass",
        )

    def _gate_fail(self, audit: RegulatoryEligibilityAudit, gate_name: str) -> None:
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name=gate_name,
            threshold=0.75,
            value=0.50,
            outcome="fail",
        )

    def test_eligible_when_all_three_preconditions_hold(self) -> None:
        audit = RegulatoryEligibilityAudit()
        self._gate_pass(audit, "minimum_auc")
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is True

    def test_not_eligible_when_adaptation_history_non_empty(self) -> None:
        audit = RegulatoryEligibilityAudit()
        self._gate_pass(audit, "minimum_auc")
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="minimum_auc",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is False

    def test_not_eligible_when_required_gate_missing(self) -> None:
        audit = RegulatoryEligibilityAudit()
        # Only "some_other_gate" passes; "minimum_auc" never evaluated.
        self._gate_pass(audit, "some_other_gate")
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is False

    def test_not_eligible_when_required_gate_failed(self) -> None:
        audit = RegulatoryEligibilityAudit()
        self._gate_fail(audit, "minimum_auc")
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is False

    def test_not_eligible_when_any_other_gate_failed(self) -> None:
        """Precondition 3: every gate evaluation in history must pass."""
        audit = RegulatoryEligibilityAudit()
        self._gate_pass(audit, "minimum_auc")  # required + passing
        self._gate_fail(audit, "some_other_gate")  # not required, but failing

        # is_regulatory_eligible re-checks the WHOLE history (not just the
        # required gates) — a failed out-of-band gate disqualifies.
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is False

    def test_eligible_with_advisory_outcome_disqualifies(self) -> None:
        """Outcomes other than ``"pass"`` (e.g. ``"advisory"``) fail
        precondition 1."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="advisory",
        )
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is False

    def test_eligible_when_required_gates_empty(self) -> None:
        """No required gates + clean adaptation_history → eligible. Edge
        case for callers that run with literature gates disabled."""
        audit = RegulatoryEligibilityAudit()
        assert is_regulatory_eligible(audit, []) is True


class TestIsAdaptedRegulatoryCandidate:
    """``is_adapted_regulatory_candidate`` returns True iff (1) holds but (2)
    does not — model would be eligible if external validation confirms."""

    def test_candidate_when_all_gates_pass_but_adaptation_history_non_empty(
        self,
    ) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
        )
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="minimum_auc",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )
        assert is_adapted_regulatory_candidate(audit, ["minimum_auc"]) is True

    def test_not_candidate_when_adaptation_history_empty(self) -> None:
        """Clean lifecycle → eligible, NOT candidate. Mutually exclusive."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
        )
        assert is_adapted_regulatory_candidate(audit, ["minimum_auc"]) is False

    def test_not_candidate_when_required_gate_missing(self) -> None:
        """An adaptation history WITHOUT a passing required gate is still
        not a candidate — the model didn't even meet the absolute floor."""
        audit = RegulatoryEligibilityAudit()
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="minimum_auc",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )
        # No gate_history entries at all.
        assert is_adapted_regulatory_candidate(audit, ["minimum_auc"]) is False

    def test_eligible_and_candidate_are_mutually_exclusive(self) -> None:
        """The two flags must NEVER both be True. Sweep representative cases."""
        # Case A: clean lifecycle + passing gate → eligible only.
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
        )
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is True
        assert is_adapted_regulatory_candidate(audit, ["minimum_auc"]) is False

        # Case B: adapted lifecycle + passing gate → candidate only.
        audit_b = RegulatoryEligibilityAudit.from_dict(audit.to_dict())
        audit_b.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="minimum_auc",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )
        assert is_regulatory_eligible(audit_b, ["minimum_auc"]) is False
        assert is_adapted_regulatory_candidate(audit_b, ["minimum_auc"]) is True

        # Case C: failing gate + clean lifecycle → neither.
        audit_c = RegulatoryEligibilityAudit()
        audit_c.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.50,
            outcome="fail",
        )
        assert is_regulatory_eligible(audit_c, ["minimum_auc"]) is False
        assert is_adapted_regulatory_candidate(audit_c, ["minimum_auc"]) is False
