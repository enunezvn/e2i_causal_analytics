"""Regulatory-eligibility audit primitives — Gate N1 (plan v4 §2).

Implements the append-only audit trail that ``validate_promotion`` reads
when deciding whether to grant ``regulatory_eligible=True`` on a model.

The two append-only sub-fields enforce the codex-rescue HIGH-3 invariant:
without an immutable record of every gate evaluation + every threshold
relaxation, a model could pass absolute thresholds at promotion time
even though it adaptively relaxed them during development. The audit
trail is load-bearing for the eligibility flag — see
``validate_promotion`` for the precondition logic.

Design choices:

* The runtime guard stores history in PRIVATE fields ``_gate_history``
  and ``_adaptation_history``. Public read-only access is via
  ``audit.gate_history`` / ``audit.adaptation_history`` properties that
  return ``tuple(...)`` snapshots, so callers cannot mutate the lists in
  place (``.append()``, ``[i] = ...``, ``.clear()``, slice assignment,
  list reassignment). Codex-rescue N1-H1: a public list bypassed the
  ``__setitem__`` guard because callers could call
  ``audit.gate_history.append({...})`` or ``audit.gate_history[0]
  ["outcome"] = "pass"`` and corrupt the audit trail.
* Mutation goes through ``append_gate_evaluation`` /
  ``append_adaptation`` only. Both methods build a frozen dataclass
  entry (``GateEvaluationEntry`` / ``AdaptationEntry``) so individual
  entries are also immutable post-append.
* ``__setitem__`` (i.e., ``audit["gate_history"] = [...]``) raises
  ``RegulatoryAuditMutationError`` for backward-compat with callers
  that may not have migrated to the property interface yet.
* ``to_dict()`` returns a fresh deep-copy snapshot for serialization —
  the on-disk representation is plain JSON, but reads via the guard
  expose only the append-only API.
* ``RegulatoryEligibilityAudit.from_dict`` accepts a previously-
  serialized payload (e.g. checkpoint restart) and reconstructs the
  guard without mutating the source.

Plan reference:
.claude/plans/disease_agnostic_quality_uplift_v4.md §2 Gate N1.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple


class RegulatoryAuditMutationError(RuntimeError):
    """Raised when caller attempts to mutate a sealed audit entry.

    Triggered by:

    * ``audit[<key>] = ...`` — direct ``__setitem__`` is blocked for ALL
      keys, even the append-only list fields. Use the explicit
      ``append_gate_evaluation`` / ``append_adaptation`` methods.
    * Replacing existing list entries (slice assignment, popping, or
      reassigning a list reference) — the ``to_dict()`` snapshot returns
      copies so external mutation cannot leak back into the guarded
      state.

    The error class is a ``RuntimeError`` subclass (not ``TypeError`` or
    ``ValueError``) so callers can distinguish mutation attempts from
    type / value bugs without having to inspect the message string.
    """


# --------------------------------------------------------------------------- #
# Frozen entry dataclasses — individual audit entries are immutable.          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateEvaluationEntry:
    """One gate-evaluation entry in ``gate_history`` — frozen.

    The frozen dataclass enforces per-entry immutability: even with
    direct access to the underlying list, callers cannot mutate
    individual entries in place. This is the per-entry analog to the
    list-level read-only-tuple-property guard.
    """

    timestamp: str
    gate_name: str
    threshold: Any
    value: Any
    outcome: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gate_name": self.gate_name,
            "threshold": copy.deepcopy(self.threshold),
            "value": copy.deepcopy(self.value),
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class AdaptationEntry:
    """One adaptation entry in ``adaptation_history`` — frozen."""

    commit_sha: str
    justification_doc: str
    gate_name: str
    before_threshold: Any
    after_threshold: Any
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_sha": self.commit_sha,
            "justification_doc": self.justification_doc,
            "gate_name": self.gate_name,
            "before_threshold": copy.deepcopy(self.before_threshold),
            "after_threshold": copy.deepcopy(self.after_threshold),
            "timestamp": self.timestamp,
        }


@dataclass
class RegulatoryEligibilityAudit:
    """Append-only audit trail for regulatory-eligibility evaluation.

    Two sub-fields:

    * ``gate_history`` — tuple of dicts, each documenting one gate
      evaluation with keys ``timestamp``, ``gate_name``, ``threshold``,
      ``value``, ``outcome``. Outcome is typically ``"pass" | "fail" |
      "advisory"`` but is not enforced by this layer (the
      ``validate_promotion`` consumer enforces the schema; this layer
      enforces append-only semantics).
    * ``adaptation_history`` — tuple of dicts, each documenting one
      threshold adaptation that happened during the model's lifecycle.
      Keys: ``commit_sha``, ``justification_doc``, ``gate_name``,
      ``before_threshold``, ``after_threshold``, ``timestamp``.

    The eligibility precondition (codex-rescue HIGH-3) is:
    ``regulatory_eligible=True`` requires ``adaptation_history == []`` AND
    every gate in ``gate_history`` was evaluated against its
    literature-anchored absolute threshold (no advisory bypasses) AND all
    absolute thresholds were cleared. See ``validate_promotion`` for the
    full check sequence.

    Codex-rescue N1-H1: storage is via private fields (``_gate_history`` /
    ``_adaptation_history``) — public read access is the
    ``gate_history`` / ``adaptation_history`` properties returning
    ``tuple(...)`` snapshots so callers cannot mutate the lists in place.
    """

    _gate_history: List[Dict[str, Any]] = field(default_factory=list)
    _adaptation_history: List[Dict[str, Any]] = field(default_factory=list)

    # ----------------------------------------------------------------- #
    # Read-only properties returning tuple(...) snapshots.              #
    # ----------------------------------------------------------------- #

    @property
    def gate_history(self) -> Tuple[Dict[str, Any], ...]:
        """Read-only snapshot of the gate-evaluation history.

        Returns a ``tuple`` of deep-copied dicts so callers cannot mutate
        the audit's source list (``.append()``, ``[i] = ...``,
        ``.clear()``, slice assignment).
        """
        return tuple(copy.deepcopy(entry) for entry in self._gate_history)

    @property
    def adaptation_history(self) -> Tuple[Dict[str, Any], ...]:
        """Read-only snapshot of the adaptation history.

        Returns a ``tuple`` of deep-copied dicts so callers cannot mutate
        the audit's source list (``.append()``, ``[i] = ...``,
        ``.clear()``, slice assignment).
        """
        return tuple(copy.deepcopy(entry) for entry in self._adaptation_history)

    # ----------------------------------------------------------------- #
    # Append API — the only sanctioned mutation surface.                #
    # ----------------------------------------------------------------- #

    def append_gate_evaluation(
        self,
        timestamp: str,
        gate_name: str,
        threshold: Any,
        value: Any,
        outcome: str,
    ) -> None:
        """Append one gate-evaluation entry to ``gate_history``.

        The entry is constructed as a frozen ``GateEvaluationEntry``
        dataclass so its fields are immutable post-append. The dict
        snapshot stored in ``_gate_history`` is also deep-copied so
        callers can't retain a mutable reference into the audit list.
        """
        entry_obj = GateEvaluationEntry(
            timestamp=timestamp,
            gate_name=gate_name,
            threshold=copy.deepcopy(threshold),
            value=copy.deepcopy(value),
            outcome=outcome,
        )
        self._gate_history.append(entry_obj.to_dict())

    def append_adaptation(
        self,
        commit_sha: str,
        justification_doc: str,
        gate_name: str,
        before_threshold: Any,
        after_threshold: Any,
        timestamp: str,
    ) -> None:
        """Append one adaptation entry to ``adaptation_history``.

        ANY entry here disqualifies the model from
        ``regulatory_eligible=True`` per codex-rescue HIGH-3 (plan v4 §2
        Gate N1). The model can still be flagged
        ``adapted_regulatory_candidate=True`` if absolute thresholds
        clear at promotion time — i.e. "would be eligible if external
        validation cohort confirms".
        """
        entry_obj = AdaptationEntry(
            commit_sha=commit_sha,
            justification_doc=justification_doc,
            gate_name=gate_name,
            before_threshold=copy.deepcopy(before_threshold),
            after_threshold=copy.deepcopy(after_threshold),
            timestamp=timestamp,
        )
        self._adaptation_history.append(entry_obj.to_dict())

    # ----------------------------------------------------------------- #
    # Mapping-like access — read-only.                                  #
    # ----------------------------------------------------------------- #

    def __getitem__(self, key: str) -> Tuple[Dict[str, Any], ...]:
        """Read-only ``audit[key]`` access. Returns a tuple snapshot.

        The tuple-of-deep-copies semantics ensure mutation of the
        returned sequence does not leak back into the guarded state.
        Callers that want live append must use ``append_gate_evaluation``
        / ``append_adaptation``.
        """
        if key == "gate_history":
            return self.gate_history
        if key == "adaptation_history":
            return self.adaptation_history
        raise KeyError(
            f"RegulatoryEligibilityAudit has no field '{key}'. "
            "Allowed: 'gate_history', 'adaptation_history'."
        )

    def __setitem__(self, key: str, value: Any) -> None:
        """ALL ``__setitem__`` raises — append-only invariant.

        This guard catches both:

        * Replacing the entire list (``audit["gate_history"] = []``) —
          would erase the audit trail.
        * Setting a non-existent key — would silently shadow the
          field name.

        The eligibility precondition reads ``adaptation_history`` as
        load-bearing evidence; permitting ``__setitem__`` would let
        a caller blank-out a non-empty adaptation history at the last
        minute and pass the eligibility check. The guard makes that
        impossible at the python level.
        """
        raise RegulatoryAuditMutationError(
            f"Cannot mutate RegulatoryEligibilityAudit field '{key}' via "
            "__setitem__ — the audit trail is append-only. Use "
            "append_gate_evaluation() or append_adaptation() instead."
        )

    def __contains__(self, key: object) -> bool:
        """``key in audit`` truthy for both sub-fields."""
        return key in ("gate_history", "adaptation_history")

    # ----------------------------------------------------------------- #
    # Serialization helpers.                                            #
    # ----------------------------------------------------------------- #

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return a deep-copy snapshot of the audit trail.

        Used by the deployer when persisting state to MLflow / disk.
        Deep-copy is mandatory: a shallow snapshot would let the caller
        mutate the lists post-snapshot and corrupt the audit.
        """
        return {
            "gate_history": [copy.deepcopy(entry) for entry in self._gate_history],
            "adaptation_history": [copy.deepcopy(entry) for entry in self._adaptation_history],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegulatoryEligibilityAudit":
        """Reconstruct from a previously-serialized payload.

        The payload is deep-copied so the new instance is decoupled
        from the source dict. Missing sub-fields default to empty lists
        (i.e., a fresh audit trail) — consistent with checkpoint-
        restart behavior elsewhere in the codebase.

        Raises:
            TypeError: if either sub-field is not a list.
        """
        gate_history_raw = payload.get("gate_history", []) or []
        adaptation_history_raw = payload.get("adaptation_history", []) or []
        if not isinstance(gate_history_raw, list):
            raise TypeError(f"gate_history must be a list, got {type(gate_history_raw).__name__}")
        if not isinstance(adaptation_history_raw, list):
            raise TypeError(
                f"adaptation_history must be a list, got {type(adaptation_history_raw).__name__}"
            )
        return cls(
            _gate_history=copy.deepcopy(gate_history_raw),
            _adaptation_history=copy.deepcopy(adaptation_history_raw),
        )


# --------------------------------------------------------------------------- #
# Eligibility precondition logic — exposed for the deployer.                  #
# --------------------------------------------------------------------------- #


def is_regulatory_eligible(
    audit: RegulatoryEligibilityAudit,
    required_gates: List[str],
) -> bool:
    """Return True iff ALL three preconditions hold.

    The three preconditions per plan v4 §2 Gate N1:

    1. Every gate in ``required_gates`` appears in ``audit.gate_history``
       with outcome ``"pass"`` (no advisory bypasses).
    2. ``audit.adaptation_history == []`` — no adaptive relaxation
       happened during this model's lifecycle.
    3. Every gate evaluation's ``outcome == "pass"`` — no failed gates
       (precondition 1 already implies this for required_gates, but we
       re-check the whole history to catch any out-of-band gates that
       still fired and failed).

    NOTE: this function does NOT check that the literature-anchored
    thresholds themselves are correct — that is the caller's
    responsibility. It only checks that the gates were evaluated and
    passed.

    Args:
        audit: the audit trail
        required_gates: list of gate names that MUST be present in
            gate_history with outcome="pass" (e.g. ["minimum_auc"]).

    Returns:
        True iff all three preconditions hold; False otherwise.
    """
    # Precondition 2: no adaptations.
    if audit.adaptation_history:
        return False

    # Precondition 1: every required gate evaluated AND passed.
    gate_outcomes: Dict[str, str] = {}
    for entry in audit.gate_history:
        gate = entry.get("gate_name")
        outcome = entry.get("outcome")
        if isinstance(gate, str) and isinstance(outcome, str):
            # Latest entry wins if a gate appears multiple times.
            gate_outcomes[gate] = outcome

    for gate in required_gates:
        if gate_outcomes.get(gate) != "pass":
            return False

    # Precondition 3: every gate evaluation in the history passed.
    for entry in audit.gate_history:
        if entry.get("outcome") != "pass":
            return False

    return True


def is_adapted_regulatory_candidate(
    audit: RegulatoryEligibilityAudit,
    required_gates: List[str],
) -> bool:
    """Return True iff (1) holds but (2) does not.

    Plan v4 §2: when absolute thresholds clear (every required gate
    passed in ``gate_history``) but the audit recorded an adaptive
    relaxation, the model is flagged ``adapted_regulatory_candidate``
    — "would be eligible if external validation cohort confirms".

    This is NOT a regulatory-deployment authorization. It signals
    that the model met its absolute thresholds at promotion time but
    its threshold history is not clean.

    Args:
        audit: the audit trail
        required_gates: same definition as ``is_regulatory_eligible``.

    Returns:
        True iff every required gate passed AND adaptation_history is
        non-empty.
    """
    if not audit.adaptation_history:
        return False

    gate_outcomes: Dict[str, str] = {}
    for entry in audit.gate_history:
        gate = entry.get("gate_name")
        outcome = entry.get("outcome")
        if isinstance(gate, str) and isinstance(outcome, str):
            gate_outcomes[gate] = outcome

    for gate in required_gates:
        if gate_outcomes.get(gate) != "pass":
            return False

    return True
