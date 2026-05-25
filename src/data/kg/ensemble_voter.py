"""Phase 2.7 — EnsembleVoter.

Combines up to four upstream verdicts into one structured ``EnsembleVerdict``
per feature. The four upstream verdicts are:

1. **Layer 1** — declarative `FeatureContract` from the manifest registry.
   When a contract declares ``knowable_at = post_index`` the feature is a
   deterministic leak (no statistical or KG check needed).
2. **Layer 3 (adversarial)** — permutation-baseline z-score from
   ``compute_adversarial_score``. ``z > 5σ`` is a deterministic statistical
   leak; ``3σ < z ≤ 5σ`` is moderate evidence to weigh against the LLM
   verdict.
3. **Layer 2 (KG)** — list of ``KGEdge`` from ``KnowledgeGraphQuerier``.
   Coarse-classified into a ``KGSignal`` by ``classify_kg_signal``.
4. **Layer 4 (LLM)** — ``CausalRoleClassifier`` prediction wrapped in a
   ``LLMVerdict`` (causal_role, mechanism, recommended_remediation,
   cited_pmids). Citation verdicts from ``CitationResolver`` modulate
   confidence.

Precedence rules (first match wins; later matches still recorded in
``disagreements`` when they conflict with the winning verdict):

    1. Layer 1 ``severity == "high"`` → severity=high, remediation=drop,
       decided_by=layer_1, confidence=1.0, final_role="descendant".
       Reason: a manifest contract is a hard, deterministic veto.
    2. Adversarial ``severity == "high"`` (z > 5σ) → severity=high,
       remediation=drop, decided_by=adversarial, confidence=0.95,
       final_role="descendant". Reason: a 5σ permutation deviation is a
       data-derived hard veto, but very-rarely a legitimate extreme
       signal could survive Layer 1 — confidence < 1.0 reflects that.
    3. LLM verdict present:
       a. If KG signal contradicts the LLM role
          (``leak_drug_treats_disease``/``taxonomic_descendant`` vs
          ``ancestor``/``confounder``/``instrument``) → ABSTAIN. Reason:
          the two strongest non-deterministic signals disagree; we want
          a human in the loop, not an averaged guess.
       b. Otherwise the LLM role drives the verdict; KG agreement +
          citation verification + adversarial ``moderate`` modulate the
          confidence (see ``_score_llm_verdict``).
    4. No LLM, KG signal in {``leak_drug_treats_disease``,
       ``taxonomic_descendant``} → severity=high, remediation=drop,
       decided_by=kg, confidence=0.7, final_role="descendant". Reason:
       deterministic KG signals are reliable enough to drop a feature
       even without an LLM cross-check; lower confidence than Layer 3
       because KG provenance can be uncurated.
    5. Adversarial ``severity == "moderate"`` and no other signal →
       severity=moderate, remediation=ambiguous, decided_by=adversarial,
       confidence=0.6.
    6. None of the above → ABSTAIN. severity=abstain, remediation=review,
       decided_by=abstain, confidence=0.0, final_role=None. Reason: no
       layer produced enough evidence; queue for human-in-the-loop.

Confidence weights (LLM-driven verdicts):

    - Base 0.85 when KG signal is ``no_signal`` AND citations were
      checked AND all came back verified.
    - +0.05 when KG signal corroborates the LLM role (``no_signal``
      counts as neutral, not corroboration).
    - -0.30 when LLM cited ≥ 1 PMIDs but ALL citation verdicts failed
      verification. (Per Phase 2.6 docstring: cited_pmids represent the
      LLM's evidence claim; if none verify, the LLM is hallucinating.)
    - -0.15 when LLM cited 0 PMIDs (no evidence offered, only training
      prior).
    - -0.15 when adversarial verdict was ``moderate`` AND the LLM role
      is in {``ancestor``, ``confounder``, ``instrument``} (the LLM says
      no leak; permutation says some signal — discount).

Reference: ``.claude/plans/adaptive_temporal_validity_redesign.md``
Phase 2.7. Architectural diagram §Layer 2 (lines 110-138).
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, TypedDict

from src.data.kg.types import (
    CausalRole,
    CitationVerdict,
    EnsembleSeverity,
    EnsembleVerdict,
    KGEdge,
    KGSignal,
    LLMVerdict,
    Remediation,
)

logger = logging.getLogger(__name__)


# Per-source confidence anchors. Keep these as named constants so
# Phase 4 active-learning calibration can adjust them centrally.
LAYER_1_CONFIDENCE = 1.0
ADVERSARIAL_HIGH_CONFIDENCE = 0.95
KG_ONLY_CONFIDENCE = 0.7
ADVERSARIAL_MODERATE_CONFIDENCE = 0.6
LLM_BASE_CONFIDENCE = 0.85
LLM_KG_CORROBORATION_BONUS = 0.05
LLM_CITATION_FAIL_PENALTY = 0.30
LLM_NO_CITATION_PENALTY = 0.15
LLM_ADVERSARIAL_MODERATE_PENALTY = 0.15

# Adversarial verdict severity strings (mirror Layer 5
# `adaptive_validity_check._build_verdict`). Centralised so a future
# rename in that file fails this module's import-time check too.
ADV_SEVERITY_HIGH = "high"
ADV_SEVERITY_MODERATE = "moderate"
ADV_SEVERITY_INFO = "info"

# Layer 1 verdict severity. Layer 5 only emits `_layer_1_verdict` with
# severity "high"; we tolerate the future addition of "moderate" by
# treating it as a non-veto (LLM/KG decide).
LAYER_1_SEVERITY_HIGH = "high"

# Roles the pipeline rule of `adaptive_temporal_validity_redesign.md`
# treats as accept / reject. "instrument" is "accept with IV-validity
# flag"; the voter doesn't enforce IV validity here — it just records
# the role and a remediation of `keep_with_caveat`.
LEAK_ROLES: frozenset[str] = frozenset({"mediator", "collider", "descendant"})
ACCEPT_ROLES: frozenset[str] = frozenset({"ancestor", "confounder", "instrument"})
VALID_LLM_ROLES: frozenset[str] = LEAK_ROLES | ACCEPT_ROLES

# Issue #501 §4.1 — role→remediation maps hoisted to module-level constants so
# both ``_role_to_remediation`` AND the structural-remediation gate read ONE
# source of truth (no duplicated map → no drift trap; cf. the #491/#496 threshold
# drift-trap lesson). Behaviour-identical to the prior function-local dicts
# (pinned by ``test_role_to_remediation_map_hoist_is_behavior_identical``).
ROLE_DEFAULT_REMEDIATION: dict[str, Remediation] = {
    "mediator": "window",
    "descendant": "drop",
    "collider": "drop",
    "ancestor": "keep_with_caveat",
    "confounder": "keep_with_caveat",
    "instrument": "keep_with_caveat",
}
# The set of remediations consistent with each role. A ``collider`` FORBIDS
# transform/window (you cannot re-derive a collider into safety; conditioning on
# the common effect IS the harm) → ``{drop}`` only. This is the load-bearing
# asymmetry the structural gate exploits (#501 §4.1).
ROLE_VALID_REMEDIATIONS: dict[str, frozenset[Remediation]] = {
    "mediator": frozenset({"window", "transform", "drop"}),
    "descendant": frozenset({"drop", "transform"}),
    "collider": frozenset({"drop"}),
    "ancestor": frozenset({"keep_with_caveat", "keep"}),
    "confounder": frozenset({"keep_with_caveat", "keep"}),
    "instrument": frozenset({"keep_with_caveat", "keep"}),
}

# Issue #240 Stage 3 — env kill-switch for the audit-evaluator soft-gate.
# DEFAULT OFF: when the env var is unset or any value other than the
# literal "1", the gate NEVER fires and ``vote`` is byte-identical to its
# pre-Stage-3 behaviour. Read at call time (not import time) so the
# operator runbook's "unset → next process invocation reverts" rollback
# story holds without a code change. Design:
# ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 3.
EVALUATOR_GATE_ENABLED_ENV = "ADAPTIVE_VALIDITY_EVALUATOR_GATE_ENABLED"

# The single severity transition the Stage-3 gate is allowed to perform
# (design §3 Stage 3 + §4 R1, reframed 2026-05-25): info → moderate.
# Remediation follows deterministically — the escalated "moderate" disposition
# routes the feature to review (matching the voter's adversarial-moderate
# branch, which uses remediation="review"). The original moderate→high
# transition was proved unreachable in production (the evaluator audit only
# rides valid-role verdicts → high/info, never moderate); see
# docs/plans/240-r1-reachability-investigation.md.
_GATE_PRECONDITION_SEVERITY: EnsembleSeverity = "info"
_GATE_ESCALATED_SEVERITY: EnsembleSeverity = "moderate"
_GATE_ESCALATED_REMEDIATION: Remediation = "review"
_GATE_EVIDENCE_TAG = "evaluator_gate:R1:info→moderate"


def _evaluator_gate_enabled() -> bool:
    """True iff the Stage-3 soft-gate kill-switch is explicitly ``"1"``.

    Default OFF. Any value other than the exact string ``"1"`` (including
    unset, ``"0"``, ``""``, ``"true"``) leaves the gate disabled — the
    conservative fail-closed reading for a behaviour-mutating flag.
    """
    return os.environ.get(EVALUATOR_GATE_ENABLED_ENV) == "1"


# KG predicates we recognise as drug→disease "treats" evidence
# (Open Targets) and as taxonomic isa (UMLS relations). Stored as
# lowercase for case-insensitive matching at classification time.
#
# These predicate sets are INDICATION-NEUTRAL (verified 2026-05-08
# disease-domain audit; see docs/superpowers/specs/2026-05-08-kg-
# predicate-reconciliation-design.md §"Disease-domain coupling
# assessment"). A CDK4/6 inhibitor "treats" breast cancer with the
# same vocabulary as a biologic "treats" CSU. The Open Targets
# ``datatypeId="known_drug"`` taxonomy applies across diseases (PR-0
# maps that datatypeId → predicate="treats" at the querier boundary,
# regardless of indication); UMLS taxonomic relations
# (isa/par/chd/etc.) are universal medical-ontology vocabulary.
#
# When a future cohort introduces a non-immunology indication, expand
# these sets ONLY if Open Targets adds new datatypes that carry
# therapeutic semantics (currently ``known_drug`` is the sole one) or
# UMLS adds taxonomic relations the project relies on. Externalising
# to per-domain config is YAGNI until that pressure arrives.
TREATS_PREDICATES: frozenset[str] = frozenset({"treats", "indicated_for", "treats_indicates"})
TAXONOMIC_PREDICATES: frozenset[str] = frozenset({"isa", "inverse_isa", "par", "chd", "rb", "rn"})

# Per Phase 2.6 `CitationResolver`: a citation is verified when its
# abstract resolved AND both entities were found AND a causal cue was
# found. Layer 2.7 reuses the verdict object's own fields rather than
# re-imposing a numeric threshold.


def is_citation_verified(verdict: CitationVerdict) -> bool:
    """Return True iff a `CitationVerdict` passes the Phase 2.6 verification
    bar: abstract resolved AND ≥ 2 entities found AND a causal cue found.

    The voter uses this rather than ``overall_confidence`` thresholding
    so that confidence-weight tuning in Phase 4 doesn't accidentally
    flip individual verdicts between verified/unverified.
    """
    return bool(
        verdict.abstract_resolved and len(verdict.entities_found) >= 2 and verdict.causal_cue_found
    )


def _is_finite_number(value: Any) -> bool:
    """True when ``value`` is a real number (not bool) with a finite value.

    Used by the M3/M4 audit-integrity guards to verify that
    deterministic vetoes carry the numeric evidence they claim.
    Booleans are explicitly rejected even though `isinstance(True, int)`
    is True in Python — a bool in a numeric field is a bug, not data.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _split_citations(
    citation_verdicts: Iterable[CitationVerdict],
) -> tuple[tuple[CitationVerdict, ...], tuple[CitationVerdict, ...]]:
    """Partition citation verdicts into (verified, unverified) tuples."""
    verified: list[CitationVerdict] = []
    unverified: list[CitationVerdict] = []
    for v in citation_verdicts:
        if is_citation_verified(v):
            verified.append(v)
        else:
            unverified.append(v)
    return tuple(verified), tuple(unverified)


def classify_kg_signal(
    kg_edges: Iterable[KGEdge],
    *,
    feature_entity_ids: Iterable[str],
    target_entity_ids: Iterable[str],
) -> tuple[KGSignal, tuple[KGEdge, ...]]:
    """Classify KG edges relating a feature concept to a target concept.

    Args:
        kg_edges: Edges produced by ``KnowledgeGraphQuerier`` for the
            feature/target pair.
        feature_entity_ids: Set of UMLS CUIs (or external IDs) that
            represent the feature. Multiple IDs cover the case where
            the feature is a code that cross-walks to several CUIs.
        target_entity_ids: Same shape, for the prediction target.

    Returns:
        Tuple of (signal, considered_edges). ``considered_edges`` is the
        subset of ``kg_edges`` that drove the classification.

    Classification rules:
        - At least one Open Targets ``treats``-style edge between
          feature and target entities → ``"leak_drug_treats_disease"``.
        - At least one taxonomic ``is_a``-style edge between feature
          and target entities → ``"taxonomic_descendant"``.
        - Both kinds present → ``"contradictory"``.
        - Neither present (or no edges at all) → ``"no_signal"``.

    Edges that don't connect the feature/target pair (e.g., concept
    relations involving siblings) are ignored. The voter's contract is
    "what does the KG say about THIS feature's relation to the target",
    not "what does the KG say in general".
    """
    feature_set = {fid for fid in feature_entity_ids if fid}
    target_set = {tid for tid in target_entity_ids if tid}
    if not feature_set or not target_set:
        return "no_signal", ()

    treats_edges: list[KGEdge] = []
    taxonomic_edges: list[KGEdge] = []
    for edge in kg_edges:
        if not _connects(edge, feature_set, target_set):
            continue
        predicate = edge.predicate.lower() if edge.predicate else ""
        if edge.evidence_source == "open_targets" and predicate in TREATS_PREDICATES:
            treats_edges.append(edge)
        elif predicate in TAXONOMIC_PREDICATES:
            taxonomic_edges.append(edge)

    if treats_edges and taxonomic_edges:
        return "contradictory", tuple(treats_edges + taxonomic_edges)
    if treats_edges:
        return "leak_drug_treats_disease", tuple(treats_edges)
    if taxonomic_edges:
        return "taxonomic_descendant", tuple(taxonomic_edges)
    return "no_signal", ()


def _connects(
    edge: KGEdge,
    feature_set: set[str],
    target_set: set[str],
) -> bool:
    """True iff ``edge``'s endpoints are one feature CUI + one target CUI.

    Direction-agnostic: ``feature → target`` and ``target → feature``
    both count. Direction is captured separately via ``edge.predicate``
    (``isa`` vs ``inverse_isa``); the voter doesn't need direction at
    edge-relevance time.
    """
    # Codex review MEDIUM (M2, 2026-05-08): self-loop edges
    # (subject_id == object_id) trivially pass any membership check
    # whenever the same CUI is registered as both feature and target.
    # That can happen when a feature concept is the disease itself
    # (e.g., a `target_disease_count` feature where the cohort's
    # target CUI cross-walks to the feature concept). Treat self-loops
    # as no information about the feature/target relation — they
    # encode "X is_a X", not "feature is descendant of target".
    if edge.subject_id == edge.object_id:
        return False
    return (edge.subject_id in feature_set and edge.object_id in target_set) or (
        edge.subject_id in target_set and edge.object_id in feature_set
    )


def _kg_signal_implies_leak(signal: KGSignal) -> bool:
    """True for KG signals that recommend dropping the feature."""
    return signal in ("leak_drug_treats_disease", "taxonomic_descendant")


def _llm_role_is_leak(role: CausalRole) -> bool:
    """True when an LLM-assigned role recommends dropping the feature."""
    return role in LEAK_ROLES


def _kg_contradicts_llm(signal: KGSignal, role: CausalRole) -> bool:
    """True when the KG signal and the LLM role point in opposite directions.

    KG-leak with LLM-accept (or vice-versa) is a hard contradiction —
    the two strongest non-deterministic signals disagree. The voter
    abstains rather than picking a winner. ``no_signal`` never
    contradicts the LLM (no evidence either way).

    Codex review MEDIUM (M1, 2026-05-08): ``"contradictory"`` was
    previously coded as "never contradicts" so the LLM could
    arbitrate. That silently ignored the leak side of the
    contradictory edge set whenever the LLM said "accept role"
    (ancestor / confounder / instrument). The fix: a contradictory
    KG only "doesn't contradict" the LLM when the LLM agrees there
    IS a leak (the LLM accepts the leak side of the contradictory
    pair). When the LLM says accept-role, the leak edges in the
    contradictory pair stand — that IS a contradiction, so abstain.
    """
    if signal == "no_signal":
        return False
    if signal == "contradictory":
        # Contradictory KG includes leak edges. Trust LLM only when it
        # ALSO says leak; otherwise the leak edges contradict the
        # accept-role LLM verdict.
        return not _llm_role_is_leak(role)
    kg_implies_leak = _kg_signal_implies_leak(signal)
    llm_implies_leak = _llm_role_is_leak(role)
    return kg_implies_leak != llm_implies_leak


def _role_to_remediation(
    role: CausalRole,
    llm_remediation: Optional[Remediation],
) -> Remediation:
    """Map an LLM `causal_role` to a `remediation`.

    Prefers the LLM's own ``recommended_remediation`` when it's
    consistent with the role (e.g., LLM says ``mediator`` and
    ``window`` — keep ``window``). Falls back to a deterministic
    default per role when the two disagree.

    Default per role:
        mediator → ``window``
        descendant → ``drop``
        collider → ``drop``
        ancestor → ``keep_with_caveat``
        confounder → ``keep_with_caveat``
        instrument → ``keep_with_caveat`` (IV-validity check upstream)
    """
    # Issue #501 §4.1 — read the hoisted module-level constants (single source
    # of truth, shared with the structural-remediation gate). Behaviour is
    # identical to the prior function-local dicts.
    default = ROLE_DEFAULT_REMEDIATION[role]
    if llm_remediation is None:
        return default
    # If LLM's remediation is sane for the role, prefer it.
    if llm_remediation in ROLE_VALID_REMEDIATIONS[role]:
        return llm_remediation
    return default


# Issue #501 §4.3 — sibling env kill-switch for the structural-remediation gate.
# DEFAULT OFF (parallel to EVALUATOR_GATE_ENABLED_ENV): when unset or any value
# other than the literal "1", the structural gate NEVER overrides remediation and
# the per-feature loop is byte-identical (modulo the additive None-valued audit
# keys). A SIBLING var (not shared with the R1 evaluator gate) so the two toggle
# independently. Read at call time so unset → next invocation reverts.
STRUCTURAL_GATE_ENABLED_ENV = "ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED"


def structural_gate_enabled() -> bool:
    """True iff the #501 structural-remediation gate is enabled (env == "1")."""
    import os

    return os.environ.get(STRUCTURAL_GATE_ENABLED_ENV) == "1"


def apply_structural_remediation_gate(
    *,
    structural_role: Optional[str],
    llm_role: Optional[str],
    current_remediation: Optional[str],
    llm_remediation: Optional[str],
) -> Optional[str]:
    """Compute the structure-constrained remediation override (Issue #501 §4.1).

    The REACHABLE functional seam for intra-LEAK-role disagreement is REMEDIATION,
    not severity (all LEAK_ROLES map to ``high`` severity, so an ``info→moderate``
    escalation is unreachable for these cases — codex iter-0 HIGH-1). When a
    feature carries an authored structural attestation whose derived role
    ``structural_role`` DISAGREES with the LLM's ``llm_role``, the structural role
    is treated as authoritative for remediation:
    ``_role_to_remediation(structural_role, llm_remediation)``.

    Because ``_role_to_remediation`` already prefers the LLM's own proposal IFF it
    lies in ``ROLE_VALID_REMEDIATIONS[structural_role]`` and falls back to
    ``ROLE_DEFAULT_REMEDIATION[structural_role]`` otherwise, this single call
    expresses BOTH halves:
      * ``structural_role == "collider"`` → the LLM's ``window``/``transform`` is
        rejected (not in ``{drop}``) → forced to ``drop``.
      * ``structural_role == "descendant"`` → a raw LLM ``transform`` is permitted
        (it is in ``{drop, transform}``) → a transformable descendant is NOT
        over-restricted.

    Returns the override remediation when the gate fires, else ``None`` (no
    override — the caller keeps ``current_remediation``). The gate fires ONLY when
    ALL hold: the env switch is on, both roles are present, and they disagree.
    Severity is NEVER mutated here (so R1's path and the byte-identity invariant
    are undisturbed). This function is PURE: no I/O beyond the env read in
    ``structural_gate_enabled`` (which the caller invokes), no mutation.
    """
    if structural_role is None or llm_role is None:
        return None
    if structural_role == llm_role:
        return None
    if structural_role not in ROLE_DEFAULT_REMEDIATION:
        # Defensive: an attestation-derived role outside the known taxonomy
        # (should not happen — extract_role only returns the six roles) does
        # not override.
        return None
    override = _role_to_remediation(
        structural_role,  # type: ignore[arg-type]
        llm_remediation,  # type: ignore[arg-type]
    )
    if override == current_remediation:
        # No actual change — treat as a no-op (do not tag a gate firing).
        return None
    return override


def _score_llm_verdict(
    *,
    role: CausalRole,
    kg_signal: KGSignal,
    cited_pmid_count: int,
    verified_count: int,
    unverified_count: int,
    adversarial_severity: Optional[str],
) -> tuple[float, list[str]]:
    """Compute the LLM-driven confidence and human-readable evidence lines.

    Returns ``(confidence in [0, 1], evidence_lines)``.
    """
    confidence = LLM_BASE_CONFIDENCE
    evidence: list[str] = [
        f"LLM verdict: causal_role={role}; base confidence={LLM_BASE_CONFIDENCE:.2f}"
    ]

    # KG corroboration
    llm_implies_leak = _llm_role_is_leak(role)
    if kg_signal == "no_signal":
        evidence.append("KG signal: no_signal (no corroboration, no contradiction)")
    elif kg_signal == "contradictory":
        # Self-contradictory KG (treats + taxonomic) is a soft warning
        # — record but don't flip the verdict.
        evidence.append(
            "KG signal: contradictory (mixed treats + taxonomic edges); no confidence change"
        )
    else:
        kg_implies_leak = _kg_signal_implies_leak(kg_signal)
        if kg_implies_leak == llm_implies_leak:
            confidence += LLM_KG_CORROBORATION_BONUS
            evidence.append(
                f"KG signal: {kg_signal} corroborates LLM role; +{LLM_KG_CORROBORATION_BONUS:.2f}"
            )
        else:  # pragma: no cover - caller short-circuits to abstain
            evidence.append(f"KG signal: {kg_signal} contradicts LLM role; should have abstained")

    # Citation verification
    if cited_pmid_count == 0:
        confidence -= LLM_NO_CITATION_PENALTY
        evidence.append(f"LLM cited 0 PMIDs (no evidence offered); -{LLM_NO_CITATION_PENALTY:.2f}")
    else:
        total = verified_count + unverified_count
        if total == 0:
            evidence.append(
                f"LLM cited {cited_pmid_count} PMIDs but no CitationVerdicts "
                "supplied; treating as if citations were not checked"
            )
        elif verified_count == 0:
            confidence -= LLM_CITATION_FAIL_PENALTY
            evidence.append(
                f"All {total} citation(s) failed verification; -{LLM_CITATION_FAIL_PENALTY:.2f}"
            )
        else:
            evidence.append(f"{verified_count} of {total} citation(s) verified")

    # Adversarial moderate vs accept-role
    if adversarial_severity == ADV_SEVERITY_MODERATE and not llm_implies_leak:
        confidence -= LLM_ADVERSARIAL_MODERATE_PENALTY
        evidence.append(
            f"Adversarial probe: moderate signal under non-leak role; "
            f"-{LLM_ADVERSARIAL_MODERATE_PENALTY:.2f}"
        )

    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    return confidence, evidence


def _llm_severity(role: CausalRole) -> str:
    """Map an LLM role to the EnsembleVerdict severity bucket."""
    if _llm_role_is_leak(role):
        return "high"
    return "info"


class EnsembleVoter:
    """Phase 2.7 voter.

    Stateless. Construct once and call ``vote`` per feature; the
    instance method form mirrors the other Layer 2 clients
    (``EntityLinker``, ``CitationResolver``) for consistency, even
    though the voter itself holds no resources.
    """

    def vote(
        self,
        feature_name: str,
        *,
        layer_1_verdict: Optional[dict] = None,
        adversarial_verdict: Optional[dict] = None,
        kg_edges: Iterable[KGEdge] = (),
        feature_entity_ids: Iterable[str] = (),
        target_entity_ids: Iterable[str] = (),
        llm_verdict: Optional[LLMVerdict] = None,
        citation_verdicts: Iterable[CitationVerdict] = (),
    ) -> EnsembleVerdict:
        """Combine upstream verdicts into one ensemble verdict.

        Computes the *candidate* verdict via :meth:`_vote_candidate` (the
        full precedence logic), then applies the Issue #240 Stage 3
        env-gated soft-gate via :meth:`_apply_evaluator_gate`. The gate is
        a NO-OP by default (the kill-switch env var
        ``ADAPTIVE_VALIDITY_EVALUATOR_GATE_ENABLED`` defaults OFF), so this
        wrapper returns the candidate verdict byte-identically unless an
        operator has explicitly enabled the gate AND rule R1 fires on a
        moderate candidate (design §3 Stage 3). The R1 helper is called at
        most once per ``vote`` (this single post-candidate call site).

        Args:
            feature_name: Name of the feature being adjudicated.
            layer_1_verdict: Layer 1 (manifest contract) verdict dict
                from ``adaptive_validity_check._layer_1_verdict``. None
                when no contract was registered for the feature.
            adversarial_verdict: Layer 3 verdict dict from
                ``adaptive_validity_check._build_verdict``. None when
                Layer 3 wasn't run (e.g., non-numeric column).
            kg_edges: KG edges produced by ``KnowledgeGraphQuerier``
                for the feature's entities.
            feature_entity_ids: UMLS CUIs / external IDs for the
                feature. Used to filter which kg_edges are relevant.
            target_entity_ids: Same, for the prediction target.
            llm_verdict: ``LLMVerdict`` from
                ``CausalRoleClassifier``. None when Layer 4 isn't yet
                running (gated on LM endpoint configuration).
            citation_verdicts: ``CitationVerdict`` records from
                ``CitationResolver`` for the LLM's cited PMIDs/DOIs.

        Returns:
            One ``EnsembleVerdict`` carrying the decision, the
            rationale, and the audit-trail-relevant inputs. Always
            returns a verdict — abstaining is itself a verdict
            (``severity="abstain"``).
        """
        candidate = self._vote_candidate(
            feature_name,
            layer_1_verdict=layer_1_verdict,
            adversarial_verdict=adversarial_verdict,
            kg_edges=kg_edges,
            feature_entity_ids=feature_entity_ids,
            target_entity_ids=target_entity_ids,
            llm_verdict=llm_verdict,
            citation_verdicts=citation_verdicts,
        )
        return self._apply_evaluator_gate(candidate, llm_verdict)

    def _vote_candidate(
        self,
        feature_name: str,
        *,
        layer_1_verdict: Optional[dict] = None,
        adversarial_verdict: Optional[dict] = None,
        kg_edges: Iterable[KGEdge] = (),
        feature_entity_ids: Iterable[str] = (),
        target_entity_ids: Iterable[str] = (),
        llm_verdict: Optional[LLMVerdict] = None,
        citation_verdicts: Iterable[CitationVerdict] = (),
    ) -> EnsembleVerdict:
        """Pre-gate precedence logic for :meth:`vote`.

        Returns the voter's *candidate* verdict from the four upstream
        signals. This is the full pre-Stage-3 ``vote`` body; the Stage-3
        gate is applied by the public :meth:`vote` wrapper so the gate sees
        a finished candidate and fires R1 at most once. Tests that need to
        assert pre-gate behaviour can call this directly.
        """
        kg_signal, considered_edges = classify_kg_signal(
            kg_edges,
            feature_entity_ids=feature_entity_ids,
            target_entity_ids=target_entity_ids,
        )
        verified, unverified = _split_citations(citation_verdicts)
        evidence: list[str] = []
        disagreements: list[str] = []
        # Codex review MEDIUM (M5, 2026-05-08): EnsembleVerdict is
        # frozen, but its `layer_1_input` and `adversarial_input` fields
        # held caller-owned dicts by reference. A caller mutating the
        # dict post-vote would change `v.layer_1_input["severity"]`
        # while `v.severity` stayed pinned — contradictory audit
        # evidence inside a frozen verdict. Shallow `dict(...)` copies
        # suffice because Layer 5's producers (`_layer_1_verdict`,
        # `_build_verdict`) only emit primitive scalar values; if a
        # nested mutable ever lands in those producers this snapshot
        # approach must change to a deepcopy.
        layer_1_snapshot = dict(layer_1_verdict) if layer_1_verdict is not None else None
        adversarial_snapshot = (
            dict(adversarial_verdict) if adversarial_verdict is not None else None
        )
        adv_severity: Optional[str] = None
        if adversarial_verdict is not None:
            raw_adv_severity = adversarial_verdict.get("severity")
            adv_z_score = adversarial_verdict.get("z_score")
            # Codex review MEDIUM (M3, 2026-05-08): a `severity=high`
            # adversarial verdict drives a confidence=0.95 deterministic
            # veto. Without a numeric `z_score` the audit trail records
            # a high-confidence drop with no underlying evidence —
            # that's an audit-integrity failure. Downgrade malformed
            # high verdicts so the voter falls through to the
            # LLM/KG/abstain path; the malformed input is logged so
            # operators see the misconfiguration.
            #
            # Issue #194 codex pass-2 MED-1 (2026-05-14): the M3 guard
            # ALSO rejected the legitimate ``z=+inf`` strong-effect
            # path the issue #194 fix added in ``hblp_classify``. When
            # ``compute_adversarial_score`` returns ``z=+inf`` because
            # the permutation null has zero variance AND the actual
            # AUC is far above null_mean (a deterministic high-effect
            # signal with degenerate null), the classifier correctly
            # emits ``severity=high``. The non-finite-z guard then
            # downgraded it before KG/LLM/abstain routing, which in
            # ``kg_mode="shadow"`` capped to info — re-opening the
            # exact false-negative the MED-1 fix was meant to close.
            #
            # The escape is principled: accept ``severity=high`` when
            # z is non-finite IF the adversarial verdict ALSO carries
            # ``delta_auc_below_floor=False`` (i.e., the joint check
            # confirmed the absolute effect is above the floor). The
            # audit trail then records the inf z AND the delta_AUC
            # corroboration so the deterministic veto has underlying
            # evidence.
            # Issue #194 codex pass-3 MED-1: tighten the joint-check
            # corroboration predicate so a stale/contradictory producer
            # can't slip a malformed high through. Three conditions ALL
            # required:
            #   (a) z_score is specifically ``+inf`` (the production
            #       MED-1 escape path; NOT just any non-finite — z=-inf
            #       or NaN are still rejected).
            #   (b) ``delta_auc`` is finite (numeric corroboration
            #       available).
            #   (c) ``abs(delta_auc) > delta_auc_floor`` directly checked
            #       (not just ``delta_auc_below_floor is False`` — that
            #       field could be set by a stale producer who computed
            #       it against a different floor).
            # Plus the producer-tag ``_hblp_classified=True`` to confirm
            # the dict came from production ``_adversarial_input``.
            adv_delta_auc = adversarial_verdict.get("delta_auc")
            adv_delta_auc_floor = adversarial_verdict.get("delta_auc_floor")
            adv_hblp_classified = adversarial_verdict.get("_hblp_classified", False)
            z_is_positive_inf = (
                isinstance(adv_z_score, (int, float))
                and not isinstance(adv_z_score, bool)
                and not (isinstance(adv_z_score, float) and math.isnan(adv_z_score))
                and not math.isfinite(float(adv_z_score))
                and float(adv_z_score) > 0
            )
            # Narrow types so mypy sees the runtime guards.
            joint_check_corroborated = bool(
                z_is_positive_inf
                and adv_delta_auc is not None
                and adv_delta_auc_floor is not None
                and _is_finite_number(adv_delta_auc)
                and _is_finite_number(adv_delta_auc_floor)
                and abs(float(adv_delta_auc)) > float(adv_delta_auc_floor)
                and bool(adv_hblp_classified)
            )
            if (
                raw_adv_severity == ADV_SEVERITY_HIGH
                and not _is_finite_number(adv_z_score)
                and not joint_check_corroborated
            ):
                evidence.append(
                    f"Adversarial verdict claims severity=high but z_score is "
                    f"{adv_z_score!r} (missing or non-finite) AND joint-check "
                    f"corroboration unavailable (delta_auc={adv_delta_auc!r}, "
                    f"floor={adv_delta_auc_floor!r}, _hblp_classified="
                    f"{adv_hblp_classified!r}); cannot honour as deterministic veto"
                )
                logger.warning(
                    "EnsembleVoter: malformed adversarial high verdict for %s; "
                    "z_score=%r, delta_auc=%r, delta_auc_floor=%r, "
                    "_hblp_classified=%r — downgrading to no signal",
                    feature_name,
                    adv_z_score,
                    adv_delta_auc,
                    adv_delta_auc_floor,
                    adv_hblp_classified,
                )
                adv_severity = None
            else:
                adv_severity = raw_adv_severity

        # Codex review HIGH (H1, 2026-05-08): LLMVerdict.causal_role is
        # a `Literal` annotation only, not validated at construction.
        # An untrusted upstream classifier can pass through a value
        # outside the 6-role vocabulary; without this guard the LLM
        # branch eventually crashes inside `_role_to_remediation` with
        # a KeyError. Treat it as "no LLM input" — the verdict is
        # garbage and should be discarded, not crashed on. Recorded in
        # `evidence` so the audit trail names the offending role.
        sanitised_llm = llm_verdict
        if llm_verdict is not None and llm_verdict.causal_role not in VALID_LLM_ROLES:
            evidence.append(
                f"LLM verdict ignored: causal_role={llm_verdict.causal_role!r} "
                f"is outside the supported vocabulary {sorted(VALID_LLM_ROLES)}"
            )
            sanitised_llm = None

        # 1. Layer 1 deterministic veto (manifest contract).
        # Codex review MEDIUM (M4, 2026-05-08): a Layer 1 verdict
        # with severity=high but `contract_source=None` used to drive
        # a confidence=1.0 deterministic veto with no manifest
        # provenance recorded. That makes a malformed verdict
        # indistinguishable from a verified contract veto in the audit
        # trail. Require `contract_source` to honour the high veto;
        # malformed verdicts fall through to LLM/KG/abstain.
        layer_1_high = (
            layer_1_verdict is not None and layer_1_verdict.get("severity") == LAYER_1_SEVERITY_HIGH
        )
        layer_1_has_source = bool(layer_1_verdict and layer_1_verdict.get("contract_source"))
        if layer_1_high and not layer_1_has_source:
            evidence.append(
                f"Layer 1 verdict claims severity=high but contract_source is "
                f"{layer_1_verdict.get('contract_source') if layer_1_verdict else None!r} "
                f"(missing or empty); cannot honour as deterministic veto"
            )
            logger.warning(
                "EnsembleVoter: malformed Layer 1 high verdict for %s; "
                "contract_source is missing — downgrading to no signal",
                feature_name,
            )
            layer_1_high = False
        if layer_1_high:
            assert layer_1_verdict is not None  # narrows for mypy; layer_1_high implies non-None
            evidence.append(
                f"Layer 1 manifest contract veto: severity=high, "
                f"contract_source={layer_1_verdict.get('contract_source')}"
            )
            self._record_disagreements_with_winner(
                winner="layer_1",
                kg_signal=kg_signal,
                adversarial_verdict=adversarial_verdict,
                llm_verdict=sanitised_llm,
                disagreements=disagreements,
            )
            return EnsembleVerdict(
                feature_name=feature_name,
                severity="high",
                remediation="drop",
                decided_by="layer_1",
                final_role="descendant",
                confidence=LAYER_1_CONFIDENCE,
                kg_signal=kg_signal,
                kg_edges_considered=considered_edges,
                verified_citations=verified,
                unverified_citations=unverified,
                disagreements=tuple(disagreements),
                evidence=tuple(evidence),
                layer_1_input=layer_1_snapshot,
                adversarial_input=adversarial_snapshot,
                llm_input=llm_verdict,
            )

        # 2. Adversarial deterministic veto (z > 5σ).
        if adv_severity == ADV_SEVERITY_HIGH:
            z = adversarial_verdict.get("z_score") if adversarial_verdict else None
            # Issue #194 codex pass-3 MED-2: when the high veto is
            # justified by the z=+inf + joint-check corroboration path
            # (rather than a normal finite-z deterministic veto),
            # record BOTH the inf z AND the delta_AUC corroboration in
            # the evidence text — otherwise downstream legacy-dict
            # readers (who only see joined EnsembleVerdict.evidence)
            # lose the human-readable rationale for the inf z.
            adv_delta_auc_for_evidence = (
                adversarial_verdict.get("delta_auc") if adversarial_verdict else None
            )
            adv_delta_auc_floor_for_evidence = (
                adversarial_verdict.get("delta_auc_floor") if adversarial_verdict else None
            )
            if (
                isinstance(z, (int, float))
                and not isinstance(z, bool)
                and not (isinstance(z, float) and math.isnan(z))
                and not math.isfinite(float(z))
                and float(z) > 0
                and adv_delta_auc_for_evidence is not None
                and adv_delta_auc_floor_for_evidence is not None
                and _is_finite_number(adv_delta_auc_for_evidence)
                and _is_finite_number(adv_delta_auc_floor_for_evidence)
            ):
                # Narrow non-None type for mypy after runtime guards.
                _adv_delta = float(adv_delta_auc_for_evidence)
                _adv_floor = float(adv_delta_auc_floor_for_evidence)
                evidence.append(
                    f"Adversarial probe veto: severity=high, z_score={z} "
                    f"(degenerate null; null_std=0), |delta_AUC|="
                    f"{abs(_adv_delta):.4f} > floor "
                    f"{_adv_floor:.4f} "
                    f"(issue #194 joint-check corroboration)"
                )
            else:
                evidence.append(f"Adversarial probe veto: severity=high, z_score={z}")
            self._record_disagreements_with_winner(
                winner="adversarial",
                kg_signal=kg_signal,
                adversarial_verdict=adversarial_verdict,
                llm_verdict=sanitised_llm,
                disagreements=disagreements,
            )
            return EnsembleVerdict(
                feature_name=feature_name,
                severity="high",
                remediation="drop",
                decided_by="adversarial",
                final_role="descendant",
                confidence=ADVERSARIAL_HIGH_CONFIDENCE,
                kg_signal=kg_signal,
                kg_edges_considered=considered_edges,
                verified_citations=verified,
                unverified_citations=unverified,
                disagreements=tuple(disagreements),
                evidence=tuple(evidence),
                layer_1_input=layer_1_snapshot,
                adversarial_input=adversarial_snapshot,
                llm_input=llm_verdict,
            )

        # 3. KG self-contradiction is an automatic abstain only when no
        # LLM is available to break the tie. A self-contradictory KG +
        # confident LLM should still let the LLM decide (with the
        # contradiction recorded in evidence). An LLM with an invalid
        # role was sanitised to ``None`` above, so it counts as
        # "no LLM" here for the abstain trigger.
        if kg_signal == "contradictory" and sanitised_llm is None:
            evidence.append(
                "KG returned contradictory edges (treats + taxonomic) "
                "and no LLM verdict available to arbitrate"
            )
            return EnsembleVerdict(
                feature_name=feature_name,
                severity="abstain",
                remediation="review",
                decided_by="abstain",
                final_role=None,
                confidence=0.0,
                kg_signal=kg_signal,
                kg_edges_considered=considered_edges,
                verified_citations=verified,
                unverified_citations=unverified,
                disagreements=tuple(disagreements),
                evidence=tuple(evidence),
                layer_1_input=layer_1_snapshot,
                adversarial_input=adversarial_snapshot,
                llm_input=llm_verdict,
            )

        # 4. LLM verdict path (with KG cross-check). `sanitised_llm` is
        # `llm_verdict` for valid roles, or None if the role was outside
        # the supported vocabulary; the original `llm_verdict` is still
        # carried into ``llm_input`` so the audit trail records what
        # was actually passed in.
        if sanitised_llm is not None:
            if _kg_contradicts_llm(kg_signal, sanitised_llm.causal_role):
                evidence.append(
                    f"KG signal {kg_signal!r} contradicts LLM role "
                    f"{sanitised_llm.causal_role!r}: abstaining"
                )
                disagreements.append(
                    f"kg={kg_signal} disagrees with llm={sanitised_llm.causal_role}"
                )
                return EnsembleVerdict(
                    feature_name=feature_name,
                    severity="abstain",
                    remediation="review",
                    decided_by="abstain",
                    final_role=None,
                    confidence=0.0,
                    kg_signal=kg_signal,
                    kg_edges_considered=considered_edges,
                    verified_citations=verified,
                    unverified_citations=unverified,
                    disagreements=tuple(disagreements),
                    evidence=tuple(evidence),
                    layer_1_input=layer_1_snapshot,
                    adversarial_input=adversarial_snapshot,
                    llm_input=llm_verdict,
                )

            confidence, llm_evidence = _score_llm_verdict(
                role=sanitised_llm.causal_role,
                kg_signal=kg_signal,
                cited_pmid_count=len(sanitised_llm.cited_pmids),
                verified_count=len(verified),
                unverified_count=len(unverified),
                adversarial_severity=adv_severity,
            )
            evidence.extend(llm_evidence)

            severity = _llm_severity(sanitised_llm.causal_role)
            remediation = _role_to_remediation(
                sanitised_llm.causal_role,
                sanitised_llm.recommended_remediation,
            )
            # Layer 1 was None or info-severity — record any soft
            # disagreement (e.g., adversarial=moderate while LLM says
            # accept) for the audit trail.
            if adv_severity == ADV_SEVERITY_MODERATE and not _llm_role_is_leak(
                sanitised_llm.causal_role
            ):
                disagreements.append("adversarial=moderate but llm says accept-role")
            return EnsembleVerdict(
                feature_name=feature_name,
                severity=severity,  # type: ignore[arg-type]
                remediation=remediation,
                decided_by="llm",
                final_role=sanitised_llm.causal_role,
                confidence=confidence,
                kg_signal=kg_signal,
                kg_edges_considered=considered_edges,
                verified_citations=verified,
                unverified_citations=unverified,
                disagreements=tuple(disagreements),
                evidence=tuple(evidence),
                layer_1_input=layer_1_snapshot,
                adversarial_input=adversarial_snapshot,
                llm_input=llm_verdict,
            )

        # 5. KG-only (no LLM) path: confident KG leak signal alone is
        # enough to drop the feature. Confidence is below the
        # adversarial deterministic veto because KG provenance is
        # uncurated.
        if _kg_signal_implies_leak(kg_signal):
            evidence.append(
                f"KG-only verdict: signal={kg_signal} (no LLM available); deterministic drop"
            )
            return EnsembleVerdict(
                feature_name=feature_name,
                severity="high",
                remediation="drop",
                decided_by="kg",
                final_role="descendant",
                confidence=KG_ONLY_CONFIDENCE,
                kg_signal=kg_signal,
                kg_edges_considered=considered_edges,
                verified_citations=verified,
                unverified_citations=unverified,
                disagreements=tuple(disagreements),
                evidence=tuple(evidence),
                layer_1_input=layer_1_snapshot,
                adversarial_input=adversarial_snapshot,
                llm_input=llm_verdict,
            )

        # 6. Adversarial moderate alone: ambiguous, queue for review.
        if adv_severity == ADV_SEVERITY_MODERATE:
            z = adversarial_verdict.get("z_score") if adversarial_verdict else None
            evidence.append(f"Adversarial probe: severity=moderate (z_score={z}); ambiguous")
            return EnsembleVerdict(
                feature_name=feature_name,
                severity="moderate",
                remediation="review",
                decided_by="adversarial",
                final_role=None,
                confidence=ADVERSARIAL_MODERATE_CONFIDENCE,
                kg_signal=kg_signal,
                kg_edges_considered=considered_edges,
                verified_citations=verified,
                unverified_citations=unverified,
                disagreements=tuple(disagreements),
                evidence=tuple(evidence),
                layer_1_input=layer_1_snapshot,
                adversarial_input=adversarial_snapshot,
                llm_input=llm_verdict,
            )

        # 7. Abstain: nothing decisive.
        evidence.append("No decisive layer; abstaining for human review")
        return EnsembleVerdict(
            feature_name=feature_name,
            severity="abstain",
            remediation="review",
            decided_by="abstain",
            final_role=None,
            confidence=0.0,
            kg_signal=kg_signal,
            kg_edges_considered=considered_edges,
            verified_citations=verified,
            unverified_citations=unverified,
            disagreements=tuple(disagreements),
            evidence=tuple(evidence),
            layer_1_input=layer_1_snapshot,
            adversarial_input=adversarial_snapshot,
            llm_input=llm_verdict,
        )

    @staticmethod
    def _apply_evaluator_gate(
        verdict: EnsembleVerdict,
        llm_verdict: Optional[LLMVerdict],
    ) -> EnsembleVerdict:
        """Issue #240 Stage 3 — env-gated, fail-open soft-gate.

        Applied to the voter's *candidate* verdict at every ``vote`` exit
        path. Behaviour:

        - **Flag OFF (default).** When
          ``ADAPTIVE_VALIDITY_EVALUATOR_GATE_ENABLED`` is unset or not the
          literal ``"1"``, return ``verdict`` UNCHANGED (identity) — the
          voter is byte-identical to its pre-Stage-3 behaviour. This is
          the load-bearing default-OFF guarantee (proven by the
          byte-identity test).
        - **Fail-open.** When the worker carried no evaluator audit
          (``llm_verdict.evaluator_audit is None`` — evaluator disabled)
          OR the evaluator errored (``satisfied is None``), R1 cannot fire
          and the verdict passes through unchanged, even with the flag on.
        - **R1 fires.** When the flag is ``"1"`` AND the candidate
          ``severity == "info"`` AND ``evaluate_r1`` returns
          ``"moderate"``, substitute ``severity="moderate"``,
          ``remediation="review"`` (the deterministic escalation per design
          §4 R1, reframed info→moderate), set ``decided_by="evaluator_gate"``,
          append the structured evidence tag, and record
          ``gate_rule_fired="R1"``. A NEW frozen ``EnsembleVerdict`` is
          returned (the input is never mutated); the original worker severity
          is recoverable downstream via ``gate_rule_fired`` (always "info" for
          R1) and is persisted to ``worker_severity_pre_gate`` by
          ``_ensemble_to_legacy_dict``.

        The gate intentionally does NOT set ``decided_by="evaluator_gate"``
        when the voter independently reached ``severity="moderate"`` (the
        precondition is exactly ``"info"``), so ``decided_by`` records
        the gate ONLY when it actually flipped a decision (design §3).
        """
        if not _evaluator_gate_enabled():
            return verdict
        if verdict.severity != _GATE_PRECONDITION_SEVERITY:
            # Gate is only allowed to act on an info (accept-role) candidate.
            # High / moderate / abstain pass through (R1's precondition is info).
            return verdict
        evaluator_audit = llm_verdict.evaluator_audit if llm_verdict is not None else None
        # Fail-open: no audit (evaluator disabled) OR evaluator errored.
        # ``satisfied is None`` is the runner's signal for an evaluator
        # exception; ``evaluate_r1`` already treats ``satisfied is not
        # False`` as no-fire, so a None audit / None satisfied cannot
        # trigger. The explicit guard documents the fail-open contract.
        if evaluator_audit is None or evaluator_audit.satisfied is None:
            return verdict
        # Lazy import keeps the module's top-level surface free of the
        # promotion-rules dependency (matches the shadow-mode call site in
        # ``adaptive_validity_check._ensemble_to_legacy_dict``).
        from src.data.evaluator_promotion_rules import evaluate_r1

        proposed = evaluate_r1(verdict.severity, evaluator_audit)
        if proposed != _GATE_ESCALATED_SEVERITY:
            return verdict
        # R1 fired: escalate info → moderate. Build a new frozen verdict
        # (dataclasses.replace would re-run __init__; an explicit copy of
        # the mutated fields keeps the intent obvious and the input
        # untouched).
        logger.info(
            "evaluator_gate: R1 escalated severity info→moderate for feature %r "
            "(decided_by %s→evaluator_gate)",
            verdict.feature_name,
            verdict.decided_by,
        )
        return EnsembleVerdict(
            feature_name=verdict.feature_name,
            severity=_GATE_ESCALATED_SEVERITY,
            remediation=_GATE_ESCALATED_REMEDIATION,
            decided_by="evaluator_gate",
            confidence=verdict.confidence,
            final_role=verdict.final_role,
            kg_signal=verdict.kg_signal,
            kg_edges_considered=verdict.kg_edges_considered,
            verified_citations=verdict.verified_citations,
            unverified_citations=verdict.unverified_citations,
            disagreements=verdict.disagreements,
            evidence=verdict.evidence + (_GATE_EVIDENCE_TAG,),
            layer_1_input=verdict.layer_1_input,
            adversarial_input=verdict.adversarial_input,
            llm_input=verdict.llm_input,
            gate_rule_fired="R1",
        )

    @staticmethod
    def _record_disagreements_with_winner(
        *,
        winner: str,
        kg_signal: KGSignal,
        adversarial_verdict: Optional[dict],
        llm_verdict: Optional[LLMVerdict],
        disagreements: list[str],
    ) -> None:
        """Note any sub-verdict that disagrees with the winning veto.

        Layer 1 / adversarial vetoes always win, but the audit trail
        should record when a downstream sub-verdict suggested keeping
        the feature anyway — that's a useful signal for Phase 4 active
        learning (the LLM might be right and the manifest wrong).
        """
        # When Layer 1 wins, an adversarial=info or LLM=accept is a
        # disagreement worth noting. Same when adversarial wins and
        # the LLM said accept.
        if winner == "layer_1":
            adv_sev = (
                adversarial_verdict.get("severity") if adversarial_verdict is not None else None
            )
            if adv_sev in (ADV_SEVERITY_INFO, None) and adversarial_verdict is not None:
                disagreements.append(f"layer_1=high but adversarial={adv_sev}")
            if llm_verdict is not None and not _llm_role_is_leak(llm_verdict.causal_role):
                disagreements.append(f"layer_1=high but llm={llm_verdict.causal_role}")
        elif winner == "adversarial":
            if llm_verdict is not None and not _llm_role_is_leak(llm_verdict.causal_role):
                disagreements.append(f"adversarial=high but llm={llm_verdict.causal_role}")
        # KG signal disagreements: Layer 1 / Adversarial are
        # deterministic, so a disagreeing KG signal is informational
        # only; we do NOT record it as a disagreement (KG cannot
        # contradict a deterministic veto by design).
        _ = kg_signal


# ---------------------------------------------------------------------------
# Phase 6 — FalkorDB causal-role persistence (Issue #237)
#
# Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §6.1-§6.4.
#
# These helpers form the Phase-6 Layer-2 KG voter for causal-role
# attributions. Unlike the older ``classify_kg_signal``/``EnsembleVoter``
# above (which classifies feature→target relations into a KGSignal for
# Layer-3 audit purposes), ``layer_2_kg_signal`` consults a per-feature
# ``(:Feature {experiment_id, name})`` node whose ``causal_role`` was
# persisted by ``scripts/mirror_role_attributions_to_falkordb.py`` or
# (in tests) by ``upsert_feature_role_node`` directly. The output is a
# typed ``KGRoleSignal`` dict that the ``kg_role_enrichment`` data-
# preparer node reconciles with the existing ``role_attributions``
# list (Phase-1 output) to either promote ``source="llm"`` to
# ``source="kg"`` (KG corroborates) or downgrade
# ``evaluator_satisfied`` (KG contradicts).
#
# Schema decision (codex-2 §6.1):
#   (:Feature {name, experiment_id, causal_role, causal_role_source,
#              evaluator_model, written_at})-[:FOR_BRAND]->(:Brand)
#
# ``FOR_BRAND`` is used (NOT ``BELONGS_TO``) to avoid type-name overload
# with ``model_trainer/memory_hooks.py:367`` which already uses
# ``BELONGS_TO`` for ``(:Model)-[:BELONGS_TO]->(:Experiment)``.
# ---------------------------------------------------------------------------


class KGRoleSignal(TypedDict):
    """One Phase-6 Layer-2 KG role signal for a feature.

    Returned by ``layer_2_kg_signal`` when a ``(:Feature)`` node exists
    for ``(feature, experiment_id)`` in the FalkorDB graph; ``None``
    otherwise (KG-silent, the enrichment node leaves the attribution
    unchanged).
    """

    causal_role: str
    causal_role_source: str
    evaluator_model: str


# Cypher pinned at module level so a future schema change requires
# touching one site, and the Phase-6 forcing tests can substring-grep
# for the ``Feature`` label and ``FOR_BRAND`` edge type at audit time.
_LAYER_2_KG_QUERY = (
    "MATCH (f:Feature {name: $feature, experiment_id: $experiment_id}) "
    "RETURN f.causal_role, f.causal_role_source, f.evaluator_model"
)

_UPSERT_FEATURE_QUERY = (
    "MERGE (f:Feature {name: $feature, experiment_id: $experiment_id}) "
    "SET f.causal_role = $causal_role, "
    "    f.causal_role_source = $causal_role_source, "
    "    f.evaluator_model = $evaluator_model, "
    "    f.written_at = $written_at "
    "WITH f "
    "MERGE (b:Brand {name: $brand}) "
    "MERGE (f)-[:FOR_BRAND]->(b)"
)


def layer_2_kg_signal(
    graph: Any,
    *,
    feature: str,
    experiment_id: str,
) -> Optional[KGRoleSignal]:
    """Query FalkorDB for the per-feature causal role persisted in Phase 6.

    Plan §6.4. Returns a ``KGRoleSignal`` when a Feature node exists,
    ``None`` (KG-silent) when not. Robust to malformed graph rows: any
    non-string field in the result row is treated as a miss.

    Args:
        graph: A FalkorDB graph handle (``client.select_graph(...)`` or
            the test fakes in ``test_falkordb_role_persistence.py``).
            Anything with a ``.query(cypher, params) -> result``
            interface works.
        feature: The feature name (matches the ``RoleAttribution.feature``
            and ``adaptive_verdicts[i]["feature"]``).
        experiment_id: The experiment that wrote the role. Scoped reads
            avoid cross-experiment role leakage; the mirror script writes
            ``(feature, experiment_id)`` pairs as the natural key.

    Returns:
        ``KGRoleSignal`` if a Feature node exists with non-None
        ``causal_role`` and ``causal_role_source``; ``None`` otherwise.
    """
    try:
        result = graph.query(
            _LAYER_2_KG_QUERY,
            {"feature": feature, "experiment_id": experiment_id},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "layer_2_kg_signal: graph.query raised for feature=%r exp=%r — "
            "treating as KG-silent. Cause: %s",
            feature,
            experiment_id,
            exc,
        )
        return None
    result_set = getattr(result, "result_set", None) or []
    if not result_set:
        return None
    row = result_set[0]
    if len(row) < 2:
        return None
    causal_role = row[0]
    causal_role_source = row[1]
    evaluator_model = row[2] if len(row) >= 3 else None
    if not isinstance(causal_role, str) or not causal_role:
        return None
    if not isinstance(causal_role_source, str) or not causal_role_source:
        return None
    if not isinstance(evaluator_model, str) or not evaluator_model:
        # Sentinel — Phase-6 KG-corroborated attributions stamp this
        # provenance string (``role_attribution.py`` documents
        # ``"kg:falkordb"`` for kg sources).
        evaluator_model = "kg:falkordb"
    return KGRoleSignal(
        causal_role=causal_role,
        causal_role_source=causal_role_source,
        evaluator_model=evaluator_model,
    )


def upsert_feature_role_node(
    graph: Any,
    *,
    feature: str,
    experiment_id: str,
    causal_role: str,
    causal_role_source: str,
    evaluator_model: str,
    brand: str,
    written_at: Optional[datetime] = None,
) -> None:
    """MERGE a Feature node + FOR_BRAND edge.

    Plan §6.1 / §6.2. Idempotent: re-upserting overwrites the role,
    source, and evaluator_model but does not create duplicate nodes or
    edges (``MERGE`` semantics).

    Args:
        graph: FalkorDB graph handle (see ``layer_2_kg_signal``).
        feature: The feature name.
        experiment_id: Scoping experiment.
        causal_role: One of {ancestor, confounder, mediator, collider,
            descendant, instrument}. Caller is responsible for
            validation; FalkorDB does not enforce CHECK constraints.
        causal_role_source: One of {manifest, llm, kg}.
        evaluator_model: Provenance string (``"n/a"`` for manifest,
            ``"kg:falkordb"`` for kg, model id for llm).
        brand: Brand name for the ``FOR_BRAND`` edge.
        written_at: Timestamp (UTC). Defaults to now.
    """
    ts = (written_at or datetime.now(timezone.utc)).isoformat()
    graph.query(
        _UPSERT_FEATURE_QUERY,
        {
            "feature": feature,
            "experiment_id": experiment_id,
            "causal_role": causal_role,
            "causal_role_source": causal_role_source,
            "evaluator_model": evaluator_model,
            "written_at": ts,
            "brand": brand,
        },
    )
