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
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


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

    Codex N1-H2: ``threshold_provenance`` is the load-bearing field for
    the (1) "absolute thresholds cleared" precondition. Only
    ``"literature_anchored"`` thresholds count toward eligibility;
    ``"cohort_fitted"`` / ``"operator_override"`` / ``None`` cause the
    gate to be SKIPPED for the eligibility check even if the value
    happens to clear the threshold.
    Codex N1-M2: ``reason`` is an optional human-readable string
    explaining a ``"skipped"`` outcome (e.g. ``"malformed_metric"``).
    """

    timestamp: str
    gate_name: str
    threshold: Any
    value: Any
    outcome: str
    threshold_provenance: Any = None
    reason: Any = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "gate_name": self.gate_name,
            "threshold": copy.deepcopy(self.threshold),
            "value": copy.deepcopy(self.value),
            "outcome": self.outcome,
        }
        # Include optional fields only when set so existing on-disk
        # snapshots (without provenance / reason) round-trip identically.
        if self.threshold_provenance is not None:
            out["threshold_provenance"] = self.threshold_provenance
        if self.reason is not None:
            out["reason"] = self.reason
        return out


# --------------------------------------------------------------------------- #
# Threshold-provenance registry — codex-rescue N1-H2.                         #
# --------------------------------------------------------------------------- #

# Allowed values for ``GateEvaluationEntry.threshold_provenance``. Only
# ``"literature_anchored"`` lets a gate count toward the (1) precondition
# in ``is_regulatory_eligible``. Other values are recorded for audit
# purposes but cause the gate to be SKIPPED for eligibility decisions.
#
# Codex-rescue N1-H2 pass-2 sharpening: ``"unknown"`` is the default
# verdict when the (gate, value) pair has no canonical doc reference —
# replaces silent ``None``/``"cohort_fitted"`` defaults that the prior
# classifier returned. The eligibility check denies for any provenance
# other than ``"literature_anchored"``, and ``"unknown"`` makes the
# denial reason explicit on the gate-history entry.
THRESHOLD_PROVENANCE_LITERATURE_ANCHORED: str = "literature_anchored"
THRESHOLD_PROVENANCE_COHORT_FITTED: str = "cohort_fitted"
THRESHOLD_PROVENANCE_OPERATOR_OVERRIDE: str = "operator_override"
THRESHOLD_PROVENANCE_UNKNOWN: str = "unknown"

ALLOWED_THRESHOLD_PROVENANCE: frozenset[str] = frozenset(
    {
        THRESHOLD_PROVENANCE_LITERATURE_ANCHORED,
        THRESHOLD_PROVENANCE_COHORT_FITTED,
        THRESHOLD_PROVENANCE_OPERATOR_OVERRIDE,
        THRESHOLD_PROVENANCE_UNKNOWN,
    }
)

# Registry of (gate, exact-threshold-value) → canonical doc-reference.
# Codex-rescue N1-H2 pass-2 sharpening: the registry is no longer
# ``Dict[str, float]`` (gate → anchor) because that conflates "value
# matches a literature-anchored value" with "operator certified the
# value as such". A registered gate's value is now keyed on the EXACT
# (gate, value) pair so registered ``minimum_auc=0.75`` is the only
# pair that auto-classifies as ``literature_anchored``; any other
# value (relaxed OR coincidentally-tighter) classifies as ``unknown``
# and the caller must explicitly opt into a non-literature provenance
# via the ``register_cohort_fitted_threshold`` API or pass
# ``declared_provenance`` explicitly to ``classify_threshold_provenance``.
#
# Doc-reference values point at the canonical sign-off doc whose
# threshold is being mirrored — operators must be able to trace any
# literature_anchored verdict to a citable artifact.
#
# Sources:
#   - minimum_auc=0.75 → scope_definer/nodes/criteria_validator.py:
#     118-120, anchored to the binary-classification floor for
#     clinical-decision models (Vickers 2019; Cook 2007). Specifically
#     the floor below which the literature treats the model as
#     ineligible regardless of other quality signals.
# Registry is keyed on (gate, exact-value, deployment_intent). The
# deployment-intent dimension recalibrates the anchor to the USE CASE: a
# clinical-decision model keeps the AUC 0.75 floor (Vickers 2019; Cook 2007);
# a COMMERCIAL targeting/propensity model (never used at site of care) uses the
# separately-cited AUC 0.65 floor (Hosmer & Lemeshow 2013; marketing/propensity
# convention). Keying the intent INTO the registry is the anti-laundering guard:
# a clinical run can never borrow the commercial anchor (and vice versa) because
# the (gate, 0.65, "clinical") pair is simply absent.
_VALID_DEPLOYMENT_INTENTS: frozenset = frozenset({"clinical", "commercial"})
_DEFAULT_DEPLOYMENT_INTENT: str = "clinical"

LITERATURE_ANCHORED_THRESHOLDS: Dict[Tuple[str, float, str], str] = {
    (
        "minimum_auc",
        0.75,
        "clinical",
    ): "scope_definer/nodes/criteria_validator.py (Vickers 2019; Cook 2007 — clinical-decision floor)",
    (
        "minimum_auc",
        0.60,
        "commercial",
    ): "scope_definer/nodes/criteria_validator.py (Hosmer & Lemeshow 2013; marketing/propensity minimum-useful-discrimination floor AUC>=0.60 — targeting models are used by ranking; owner-ratified 2026-06-07)",
}


def _normalize_intent(deployment_intent: Any) -> str:
    """Return a valid deployment-intent literal (defaults to ``"clinical"``)."""
    return (
        deployment_intent
        if deployment_intent in _VALID_DEPLOYMENT_INTENTS
        else _DEFAULT_DEPLOYMENT_INTENT
    )


def get_literature_anchor_doc_ref(
    gate_name: str, threshold: float, deployment_intent: Any = _DEFAULT_DEPLOYMENT_INTENT
) -> Optional[str]:
    """Return the canonical doc-reference for a (gate, threshold, intent) triple.

    Codex-rescue N1-H2 pass-2 sharpening: callers can recover the
    sign-off doc-ref for any registered (gate, value) pair. Returns
    None if the triple is not registered — a missing registration is the
    expected signal that the caller is trying to relax / tighten the
    gate without a literature anchor (or is using the wrong intent's anchor).
    """
    try:
        threshold_f = float(threshold)
    except (TypeError, ValueError):
        return None
    return LITERATURE_ANCHORED_THRESHOLDS.get(
        (gate_name, threshold_f, _normalize_intent(deployment_intent))
    )


def classify_threshold_provenance(
    gate_name: str,
    threshold: Any,
    declared_provenance: Any = None,
    *,
    deployment_intent: Any = _DEFAULT_DEPLOYMENT_INTENT,
) -> str:
    """Return the provenance literal for a (gate, threshold, intent) triple.

    Codex-rescue N1-H2 pass-2 sharpening: the registry is now
    ``Dict[Tuple[str, float, str], str]`` mapping ``(gate, exact_value,
    deployment_intent)`` to canonical doc-ref. The classifier returns
    ``"literature_anchored"`` ONLY when the (gate, threshold, intent)
    triple is EXACTLY in the registry. Any other case returns ``"unknown"``
    unless the caller explicitly declares a non-literature provenance
    (``"cohort_fitted"`` or ``"operator_override"``).

    ``deployment_intent`` is keyword-only and defaults to ``"clinical"`` so
    every existing 2-/3-arg call is unchanged — clinical runs still resolve
    the 0.75 anchor. A commercial run must pass ``deployment_intent="commercial"``
    to resolve the 0.65 anchor; passing a clinical threshold (0.75) under a
    commercial intent (or vice versa) is NOT anchored — the intent is part of
    the registry key, which is the anti-laundering guard.

    Behavior matrix:

    * Exact (gate, value, intent) match in registry → ``"literature_anchored"``,
      regardless of ``declared_provenance``.
    * Caller declared ``"literature_anchored"`` but triple NOT in registry
      → ``"unknown"`` (drops the laundering attack).
    * Caller declared ``"cohort_fitted"`` or ``"operator_override"``
      and triple NOT in registry → that declared value (passes through;
      counts as non-literature for eligibility).
    * No declaration AND triple NOT in registry → ``"unknown"``.

    The classifier no longer returns ``None`` — every code path returns
    a string from the ``ALLOWED_THRESHOLD_PROVENANCE`` set. ``"unknown"``
    is the new default when nothing matches.
    """
    intent = _normalize_intent(deployment_intent)
    # Try the exact (gate, value, intent) lookup first — this is the only path
    # that can return ``"literature_anchored"``.
    if threshold is not None:
        try:
            threshold_f = float(threshold)
        except (TypeError, ValueError):
            # Non-numeric threshold cannot be in the registry.
            threshold_f = None
        if (
            threshold_f is not None
            and (gate_name, threshold_f, intent) in LITERATURE_ANCHORED_THRESHOLDS
        ):
            return THRESHOLD_PROVENANCE_LITERATURE_ANCHORED

    # If caller declared "literature_anchored" but the pair is not in
    # the registry, drop it on the floor — do not launder a relaxed or
    # arbitrarily-tightened threshold by tagging it.
    if declared_provenance == THRESHOLD_PROVENANCE_LITERATURE_ANCHORED:
        return THRESHOLD_PROVENANCE_UNKNOWN

    # Pass-through caller-declared non-literature provenance values.
    # The eligibility check only counts ``literature_anchored``, so
    # this still SKIPs the gate for eligibility — but the explicit
    # tag is preserved for audit/dashboards.
    if declared_provenance == THRESHOLD_PROVENANCE_COHORT_FITTED:
        return THRESHOLD_PROVENANCE_COHORT_FITTED
    if declared_provenance == THRESHOLD_PROVENANCE_OPERATOR_OVERRIDE:
        return THRESHOLD_PROVENANCE_OPERATOR_OVERRIDE

    # Default: unknown — the gate-history entry will surface this so
    # the operator can see exactly why the gate was skipped.
    return THRESHOLD_PROVENANCE_UNKNOWN


# --------------------------------------------------------------------------- #
# Canonical-hash helpers — codex-rescue N1-H3 pass-2 + new MED.               #
# --------------------------------------------------------------------------- #

# Fields that constitute the canonical identity of an adaptation entry.
# Codex-rescue N1-H3 pass-2 + new MED: the prior 3-tuple
# ``(commit_sha, gate_name, timestamp)`` was insufficient — a tampered
# payload with the same key fields but altered ``before_threshold`` /
# ``after_threshold`` / ``justification_doc`` was treated as ingested.
# The hash now covers EVERY canonical field so the load-bearing
# threshold deltas are part of the identity.
ADAPTATION_ENTRY_CANONICAL_FIELDS: Tuple[str, ...] = (
    "commit_sha",
    "justification_doc",
    "gate_name",
    "before_threshold",
    "after_threshold",
    "timestamp",
)


def compute_canonical_entry_hash(entry: Mapping[str, Any]) -> str:
    """Return the sha256 hex-digest of the entry's canonical JSON form.

    Codex-rescue N1-H3 pass-2 + new MED: ingestion-detection in the
    deployer must catch tampered payloads. The 3-tuple key
    ``(commit_sha, gate_name, timestamp)`` matched a tampered entry as
    "ingested" if those three fields were preserved — even though the
    threshold deltas / justification doc were swapped. Replacing the
    tuple lookup with a sha256 over the full canonical entry catches
    any field-level tampering.

    The canonical form is the JSON serialization of the entry's
    ``ADAPTATION_ENTRY_CANONICAL_FIELDS`` projected with
    ``sort_keys=True`` (deterministic key order) and ``default=str``
    (handles datetime / non-JSON-native types). Missing fields are
    represented as ``null`` in the canonical form so two entries
    differing in the presence of a key produce different hashes.

    Args:
        entry: a mapping containing the canonical adaptation-entry
            fields. Extra fields are ignored — only the canonical
            projection is hashed.

    Returns:
        Hex-encoded sha256 digest of the canonical JSON.
    """
    canonical = {
        field_name: entry.get(field_name) for field_name in ADAPTATION_ENTRY_CANONICAL_FIELDS
    }
    payload = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        threshold_provenance: Any = None,
        reason: Any = None,
    ) -> None:
        """Append one gate-evaluation entry to ``gate_history``.

        The entry is constructed as a frozen ``GateEvaluationEntry``
        dataclass so its fields are immutable post-append. The dict
        snapshot stored in ``_gate_history`` is also deep-copied so
        callers can't retain a mutable reference into the audit list.

        ``threshold_provenance`` (codex-rescue N1-H2): one of
        ``"literature_anchored" | "cohort_fitted" | "operator_override" |
        None``. Only ``"literature_anchored"`` lets the gate count
        toward the (1) "absolute thresholds cleared" precondition in
        ``is_regulatory_eligible``. Other values cause the gate to be
        SKIPPED for the eligibility check.
        ``reason`` (codex-rescue N1-M2): optional human-readable string
        explaining a ``"skipped"`` outcome (e.g. ``"malformed_metric"``).
        """
        entry_obj = GateEvaluationEntry(
            timestamp=timestamp,
            gate_name=gate_name,
            threshold=copy.deepcopy(threshold),
            value=copy.deepcopy(value),
            outcome=outcome,
            threshold_provenance=threshold_provenance,
            reason=reason,
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
       with outcome ``"pass"`` AND ``threshold_provenance ==
       "literature_anchored"``. Codex-rescue N1-H2: cohort-fitted /
       operator-override / unknown-provenance thresholds do NOT count
       toward eligibility — eligibility requires the threshold be
       signed-off literature-anchored.
    2. ``audit.adaptation_history == []`` — no adaptive relaxation
       happened during this model's lifecycle.
    3. Every gate evaluation's ``outcome == "pass"`` — no failed gates
       (precondition 1 already implies this for required_gates, but we
       re-check the whole history to catch any out-of-band gates that
       still fired and failed).

    Args:
        audit: the audit trail
        required_gates: list of gate names that MUST be present in
            gate_history with outcome="pass" AND
            threshold_provenance="literature_anchored"
            (e.g. ["minimum_auc"]).

    Returns:
        True iff all three preconditions hold; False otherwise.
    """
    # Precondition 2: no adaptations.
    if audit.adaptation_history:
        return False

    # Precondition 1: every required gate evaluated AND passed AND its
    # threshold is literature-anchored (codex-rescue N1-H2). Build a
    # "latest entry per gate" view then check the latest outcome +
    # provenance.
    gate_latest: Dict[str, Dict[str, Any]] = {}
    for entry in audit.gate_history:
        gate = entry.get("gate_name")
        if isinstance(gate, str):
            # Latest entry wins if a gate appears multiple times.
            gate_latest[gate] = entry

    for gate in required_gates:
        latest = gate_latest.get(gate)
        if latest is None:
            return False
        if latest.get("outcome") != "pass":
            return False
        if latest.get("threshold_provenance") != THRESHOLD_PROVENANCE_LITERATURE_ANCHORED:
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
    passed in ``gate_history`` against a literature-anchored threshold)
    AND every gate evaluation passed (so the model is otherwise clean),
    but the audit recorded an adaptive relaxation, the model is flagged
    ``adapted_regulatory_candidate`` — "would be eligible if external
    validation cohort confirms".

    This is NOT a regulatory-deployment authorization. It signals
    that the model met its absolute thresholds at promotion time but
    its threshold history is not clean.

    Codex-rescue N1-M1: the candidate flag must mirror
    ``is_regulatory_eligible`` precondition 3 — if any gate evaluation
    failed (even a non-required gate), the candidate flag is False.
    Codex-rescue N1-H2: required gates must clear with
    ``threshold_provenance="literature_anchored"`` to count toward
    candidate status.

    Args:
        audit: the audit trail
        required_gates: same definition as ``is_regulatory_eligible``.

    Returns:
        True iff every required gate passed (with literature-anchored
        provenance) AND every other gate evaluation passed AND
        adaptation_history is non-empty.
    """
    if not audit.adaptation_history:
        return False

    gate_latest: Dict[str, Dict[str, Any]] = {}
    for entry in audit.gate_history:
        gate = entry.get("gate_name")
        if isinstance(gate, str):
            gate_latest[gate] = entry

    for gate in required_gates:
        latest = gate_latest.get(gate)
        if latest is None:
            return False
        if latest.get("outcome") != "pass":
            return False
        if latest.get("threshold_provenance") != THRESHOLD_PROVENANCE_LITERATURE_ANCHORED:
            return False

    # Codex-rescue N1-M1: every gate evaluation in the history must
    # pass. Mirrors precondition 3 from is_regulatory_eligible.
    for entry in audit.gate_history:
        if entry.get("outcome") != "pass":
            return False

    return True
