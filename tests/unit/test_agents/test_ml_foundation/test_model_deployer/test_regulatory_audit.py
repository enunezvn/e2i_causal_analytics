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
    LITERATURE_ANCHORED_THRESHOLDS,
    RegulatoryAuditMutationError,
    RegulatoryEligibilityAudit,
    classify_threshold_provenance,
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
        # Codex N1-H2: passing gates require literature_anchored provenance.
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name=gate_name,
            threshold=0.75,
            value=0.80,
            outcome="pass",
            threshold_provenance="literature_anchored",
        )

    def _gate_fail(self, audit: RegulatoryEligibilityAudit, gate_name: str) -> None:
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name=gate_name,
            threshold=0.75,
            value=0.50,
            outcome="fail",
            threshold_provenance="literature_anchored",
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
            threshold_provenance="literature_anchored",
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
            threshold_provenance="literature_anchored",
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
            threshold_provenance="literature_anchored",
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
            threshold_provenance="literature_anchored",
        )
        assert is_regulatory_eligible(audit_c, ["minimum_auc"]) is False
        assert is_adapted_regulatory_candidate(audit_c, ["minimum_auc"]) is False


# --------------------------------------------------------------------------- #
# Codex N1-H2: threshold-provenance enforcement.                               #
# --------------------------------------------------------------------------- #


class TestN1H2ThresholdProvenanceRegistry:
    """Codex N1-H2: ``LITERATURE_ANCHORED_THRESHOLDS`` registry pins
    the canonical (gate, threshold) → doc-ref mapping.

    Pass-2 sharpening: registry is now ``Dict[Tuple[str, float], str]`` —
    the key is the EXACT (gate, value) pair so a relaxed threshold can
    never auto-classify as ``literature_anchored``.
    """

    def test_minimum_auc_zero_seven_five_pair_registered(self) -> None:
        # Anchored to scope_definer/criteria_validator.py (clinical intent).
        # Registry is now keyed on (gate, value, deployment_intent).
        assert ("minimum_auc", 0.75, "clinical") in LITERATURE_ANCHORED_THRESHOLDS
        # Doc-ref must be a non-empty citable string.
        doc_ref = LITERATURE_ANCHORED_THRESHOLDS[("minimum_auc", 0.75, "clinical")]
        assert isinstance(doc_ref, str)
        assert doc_ref  # non-empty

    def test_commercial_minimum_auc_pair_registered(self) -> None:
        # The commercial use case has its own separately-cited anchor (0.65).
        assert ("minimum_auc", 0.60, "commercial") in LITERATURE_ANCHORED_THRESHOLDS
        doc_ref = LITERATURE_ANCHORED_THRESHOLDS[("minimum_auc", 0.60, "commercial")]
        assert isinstance(doc_ref, str) and doc_ref

    def test_relaxed_minimum_auc_pair_NOT_in_registry(self) -> None:
        """A relaxed (or arbitrarily tightened) minimum_auc threshold
        is NOT in the registry — pass-2 sharpening pins the exact value.
        Cross-intent borrowing is also absent (anti-laundering guard)."""
        assert ("minimum_auc", 0.50, "clinical") not in LITERATURE_ANCHORED_THRESHOLDS
        assert ("minimum_auc", 0.80, "clinical") not in LITERATURE_ANCHORED_THRESHOLDS
        # A clinical run cannot borrow the commercial 0.60 anchor and vice versa.
        assert ("minimum_auc", 0.60, "clinical") not in LITERATURE_ANCHORED_THRESHOLDS
        assert ("minimum_auc", 0.75, "commercial") not in LITERATURE_ANCHORED_THRESHOLDS


class TestN1H2GetLiteratureAnchorDocRef:
    """Pass-2 sharpening: ``get_literature_anchor_doc_ref`` returns the
    canonical doc-ref for a registered (gate, threshold) pair."""

    def test_registered_pair_returns_doc_ref(self) -> None:
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            get_literature_anchor_doc_ref,
        )

        doc_ref = get_literature_anchor_doc_ref("minimum_auc", 0.75)
        assert doc_ref is not None
        assert "criteria_validator" in doc_ref or "Vickers" in doc_ref

    def test_unregistered_pair_returns_none(self) -> None:
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            get_literature_anchor_doc_ref,
        )

        assert get_literature_anchor_doc_ref("minimum_auc", 0.50) is None

    def test_non_numeric_threshold_returns_none(self) -> None:
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            get_literature_anchor_doc_ref,
        )

        assert get_literature_anchor_doc_ref("minimum_auc", "not-a-number") is None  # type: ignore[arg-type]


class TestN1H2ClassifyThresholdProvenance:
    """Codex N1-H2: ``classify_threshold_provenance`` returns the
    ``"literature_anchored"`` literal ONLY when (gate, threshold) is
    EXACTLY in the registry. Pass-2 sharpening: returns ``"unknown"``
    by default (no more silent ``None``); ``"cohort_fitted"`` /
    ``"operator_override"`` require explicit caller opt-in.
    """

    def test_minimum_auc_at_anchor_classifies_literature_anchored(self) -> None:
        assert (
            classify_threshold_provenance(gate_name="minimum_auc", threshold=0.75)
            == "literature_anchored"
        )

    def test_minimum_auc_relaxed_classifies_unknown(self) -> None:
        """Pass-2 sharpening: a relaxed threshold with no caller
        declaration is ``"unknown"`` (NOT ``None`` and NOT
        ``"cohort_fitted"`` — the caller MUST explicitly opt into
        ``cohort_fitted`` if they intend that semantics)."""
        assert classify_threshold_provenance(gate_name="minimum_auc", threshold=0.50) == "unknown"

    def test_minimum_auc_arbitrarily_tightened_classifies_unknown(self) -> None:
        """Even a tighter-than-anchor threshold (0.85 vs 0.75) is NOT
        ``literature_anchored`` — the registry pins the exact signed-off
        value, not a floor."""
        assert classify_threshold_provenance(gate_name="minimum_auc", threshold=0.85) == "unknown"

    def test_unregistered_gate_returns_unknown(self) -> None:
        assert (
            classify_threshold_provenance(gate_name="some_other_gate", threshold=0.75) == "unknown"
        )

    def test_relaxed_threshold_with_declared_literature_does_not_launder(self) -> None:
        """A caller declaring ``threshold_provenance="literature_anchored"``
        on a non-registered (gate, value) pair is NOT trusted — the
        classifier ignores the declaration and returns ``"unknown"``."""
        assert (
            classify_threshold_provenance(
                gate_name="minimum_auc",
                threshold=0.50,
                declared_provenance="literature_anchored",
            )
            == "unknown"
        )

    def test_cohort_fitted_provenance_requires_explicit_opt_in(self) -> None:
        """Pass-2 sharpening: ``cohort_fitted`` is recorded for audit ONLY
        when the caller explicitly declares it. The registry-mismatch path
        never returns ``cohort_fitted`` silently."""
        result = classify_threshold_provenance(
            gate_name="minimum_auc",
            threshold=0.50,
            declared_provenance="cohort_fitted",
        )
        assert result == "cohort_fitted"

    def test_operator_override_provenance_requires_explicit_opt_in(self) -> None:
        result = classify_threshold_provenance(
            gate_name="minimum_auc",
            threshold=0.50,
            declared_provenance="operator_override",
        )
        assert result == "operator_override"

    def test_invalid_declared_provenance_returns_unknown(self) -> None:
        """Pass-2 sharpening: garbage provenance falls back to
        ``"unknown"`` (NOT ``None``)."""
        assert (
            classify_threshold_provenance(
                gate_name="minimum_auc",
                threshold=0.50,
                declared_provenance="garbage_value",
            )
            == "unknown"
        )

    def test_non_numeric_threshold_returns_unknown(self) -> None:
        assert (
            classify_threshold_provenance(gate_name="minimum_auc", threshold="not-a-number")
            == "unknown"
        )

    def test_classifier_never_returns_none(self) -> None:
        """Pass-2 sharpening contract: every code path returns a string
        from ALLOWED_THRESHOLD_PROVENANCE — None is no longer reachable."""
        from src.agents.ml_foundation.model_deployer.regulatory_audit import (
            ALLOWED_THRESHOLD_PROVENANCE,
        )

        # Sweep representative cases — no None results.
        cases = [
            ("minimum_auc", 0.75, None),
            ("minimum_auc", 0.50, None),
            ("minimum_auc", 0.50, "cohort_fitted"),
            ("minimum_auc", 0.50, "operator_override"),
            ("minimum_auc", 0.50, "literature_anchored"),
            ("minimum_auc", 0.50, "garbage"),
            ("minimum_auc", "not-a-number", None),
            ("some_other_gate", 0.75, None),
        ]
        for gate_name, threshold, declared in cases:
            result = classify_threshold_provenance(
                gate_name=gate_name, threshold=threshold, declared_provenance=declared
            )
            assert result is not None, (
                f"classify_threshold_provenance returned None for "
                f"({gate_name=}, {threshold=}, {declared=})"
            )
            assert result in ALLOWED_THRESHOLD_PROVENANCE


class TestN1H2EligibilityRequiresLiteratureAnchored:
    """Codex N1-H2: ``is_regulatory_eligible`` denies eligibility when the
    required gate's ``threshold_provenance`` is not ``"literature_anchored"``."""

    def test_passing_gate_without_provenance_is_NOT_eligible(self) -> None:
        """A gate appended without any provenance defaults to None — must
        deny eligibility."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
            # threshold_provenance not provided
        )
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is False

    def test_passing_gate_with_cohort_fitted_provenance_NOT_eligible(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.50,  # relaxed
            value=0.55,
            outcome="pass",
            threshold_provenance="cohort_fitted",
        )
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is False

    def test_passing_gate_with_operator_override_NOT_eligible(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.50,
            value=0.55,
            outcome="pass",
            threshold_provenance="operator_override",
        )
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is False

    def test_passing_gate_with_literature_anchored_IS_eligible(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
            threshold_provenance="literature_anchored",
        )
        assert is_regulatory_eligible(audit, ["minimum_auc"]) is True


class TestN1H2CandidateRequiresLiteratureAnchored:
    """Codex N1-H2: ``is_adapted_regulatory_candidate`` likewise requires
    literature_anchored provenance for required gates."""

    def test_candidate_denied_without_literature_anchored(self) -> None:
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.50,
            value=0.55,
            outcome="pass",
            threshold_provenance="cohort_fitted",
        )
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="minimum_auc",
            before_threshold=0.85,
            after_threshold=0.50,
            timestamp="t",
        )
        # Even with adaptation history, the candidate flag is denied
        # because the required gate's threshold isn't literature-anchored.
        assert is_adapted_regulatory_candidate(audit, ["minimum_auc"]) is False


# --------------------------------------------------------------------------- #
# Codex N1-M1: candidate logic must mirror eligibility's failed-gate check.   #
# Pre-fix: a model with minimum_auc=pass + some_other_gate=fail + adaptation  #
# became a candidate. Post-fix: any failed gate (required or not) denies.    #
# --------------------------------------------------------------------------- #


class TestN1M1CandidateMirrorsFailedNonRequiredGate:
    def test_candidate_denied_when_non_required_gate_failed(self) -> None:
        """A non-required gate failing in history must deny candidate
        status — mirrors precondition 3 from is_regulatory_eligible."""
        audit = RegulatoryEligibilityAudit()
        # Required gate passes with literature_anchored provenance.
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
            threshold_provenance="literature_anchored",
        )
        # Non-required gate FAILS.
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="some_other_gate",
            threshold=0.5,
            value=0.3,
            outcome="fail",
        )
        # Adaptation entry present → would otherwise be a candidate.
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="minimum_auc",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )

        # Pre-fix: this returned True. Post-fix: False because the
        # non-required gate failed.
        assert is_adapted_regulatory_candidate(audit, ["minimum_auc"]) is False

    def test_candidate_granted_when_all_evaluations_pass(self) -> None:
        """Sanity: the existing happy path still works."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
            threshold_provenance="literature_anchored",
        )
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="some_other_gate",
            threshold=0.5,
            value=0.7,
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

    def test_candidate_denied_when_advisory_outcome_in_history(self) -> None:
        """An ``"advisory"`` outcome (non-pass) likewise denies candidate."""
        audit = RegulatoryEligibilityAudit()
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="minimum_auc",
            threshold=0.75,
            value=0.80,
            outcome="pass",
            threshold_provenance="literature_anchored",
        )
        audit.append_gate_evaluation(
            timestamp="t",
            gate_name="advisory_gate",
            threshold=0.5,
            value=0.4,
            outcome="advisory",
        )
        audit.append_adaptation(
            commit_sha="sha",
            justification_doc="doc",
            gate_name="minimum_auc",
            before_threshold=0.85,
            after_threshold=0.75,
            timestamp="t",
        )
        assert is_adapted_regulatory_candidate(audit, ["minimum_auc"]) is False
