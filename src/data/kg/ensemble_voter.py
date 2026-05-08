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
from typing import Any, Iterable, Optional

from src.data.kg.types import (
    CausalRole,
    CitationVerdict,
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

# KG predicates we recognise as drug→disease "treats" evidence
# (Open Targets) and as taxonomic isa (UMLS relations). Stored as
# lowercase for case-insensitive matching at classification time.
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
    role_default: dict[str, Remediation] = {
        "mediator": "window",
        "descendant": "drop",
        "collider": "drop",
        "ancestor": "keep_with_caveat",
        "confounder": "keep_with_caveat",
        "instrument": "keep_with_caveat",
    }
    default = role_default[role]
    if llm_remediation is None:
        return default
    # If LLM's remediation is sane for the role, prefer it.
    valid_per_role: dict[str, frozenset[Remediation]] = {
        "mediator": frozenset({"window", "transform", "drop"}),
        "descendant": frozenset({"drop", "transform"}),
        "collider": frozenset({"drop"}),
        "ancestor": frozenset({"keep_with_caveat", "keep"}),
        "confounder": frozenset({"keep_with_caveat", "keep"}),
        "instrument": frozenset({"keep_with_caveat", "keep"}),
    }
    if llm_remediation in valid_per_role[role]:
        return llm_remediation
    return default


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
        kg_signal, considered_edges = classify_kg_signal(
            kg_edges,
            feature_entity_ids=feature_entity_ids,
            target_entity_ids=target_entity_ids,
        )
        verified, unverified = _split_citations(citation_verdicts)
        evidence: list[str] = []
        disagreements: list[str] = []
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
            if raw_adv_severity == ADV_SEVERITY_HIGH and not _is_finite_number(
                adv_z_score
            ):
                evidence.append(
                    f"Adversarial verdict claims severity=high but z_score is "
                    f"{adv_z_score!r} (missing or non-finite); cannot honour "
                    f"as deterministic veto"
                )
                logger.warning(
                    "EnsembleVoter: malformed adversarial high verdict for %s; "
                    "z_score=%r — downgrading to no signal",
                    feature_name,
                    adv_z_score,
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
        if layer_1_verdict is not None and layer_1_verdict.get("severity") == LAYER_1_SEVERITY_HIGH:
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
                layer_1_input=layer_1_verdict,
                adversarial_input=adversarial_verdict,
                llm_input=llm_verdict,
            )

        # 2. Adversarial deterministic veto (z > 5σ).
        if adv_severity == ADV_SEVERITY_HIGH:
            z = adversarial_verdict.get("z_score") if adversarial_verdict else None
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
                layer_1_input=layer_1_verdict,
                adversarial_input=adversarial_verdict,
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
                layer_1_input=layer_1_verdict,
                adversarial_input=adversarial_verdict,
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
                    layer_1_input=layer_1_verdict,
                    adversarial_input=adversarial_verdict,
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
                layer_1_input=layer_1_verdict,
                adversarial_input=adversarial_verdict,
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
                layer_1_input=layer_1_verdict,
                adversarial_input=adversarial_verdict,
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
                layer_1_input=layer_1_verdict,
                adversarial_input=adversarial_verdict,
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
            layer_1_input=layer_1_verdict,
            adversarial_input=adversarial_verdict,
            llm_input=llm_verdict,
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
