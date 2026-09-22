"""Cohort-DAG assembler: unions authored fragments for one (T, Y).

Lane B of the real-data causal estimation program (spec
``docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md``
§3 Lane B item 2). Input: the structural author's per-feature records
(``src.data.kg.structural_author.AuthoredAttestation``, or their ``to_dict()``
shape read back from ``attestations.json``). Output: one cohort DAG with

* the fragments' ``T`` / ``Y`` anchors mapped to the cohort's treatment and
  outcome column names, latents (``U_*``) shared by name across fragments
  (``merge_latents=True``: an author who names ``U_disease_severity`` in two
  fragments means the same unmeasured cause; the alternative namespaces them);
* per-edge provenance (which features authored it, the best citation grade,
  the Lane E constraint that dropped it, if any);
* every feature's role re-derived on the UNION (``extract_role``) and compared
  with its fragment role — drift is a finding for the reviewer, never patched;
* the adjustment set found with the graph builder's backdoor criterion
  (``src.ml.causal_role_dgp.backdoor.satisfies_backdoor_criterion``, the same
  function ``GraphBuilderNode`` delegates to) over OBSERVED candidates only:
  latents are never adjustable, a feature carrying a Lane E leak verdict is
  excluded whatever its authored edges say (spec §3 Lane E 3(b)), and
  instruments / T-descendants are not backdoor variables. The full admissible
  observed set is reported next to a deterministic greedy-minimal subset; when
  even the full observed set fails the criterion the DAG has an unblockable
  backdoor (a latent into both T and Y with no observed child on the path) and
  the assembler says so instead of returning an empty set that reads as
  "nothing to adjust";
* the ``dag_structure_json`` shape ``DagPanel`` renders
  (``DagStructureSnapshot``: nodes, edges, treatment_nodes, outcome_nodes,
  adjustment_sets, plus ``latent_nodes`` and ``edge_provenance`` as extra keys)
  and a per-feature evidence table for ``agent_assessment_json``.

Everything here is deterministic and LLM-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import networkx as nx

from src.data.kg.structural_author import (
    OUTCOME_NODE,
    TREATMENT_NODE,
    AuthoredAttestation,
)
from src.ml.causal_role_dgp.backdoor import satisfies_backdoor_criterion
from src.ml.causal_role_dgp.extractor import extract_role

__all__ = [
    "ASSEMBLER_SCHEMA_VERSION",
    "ADJUSTABLE_ROLES",
    "CohortDag",
    "assemble_cohort_dag",
]

ASSEMBLER_SCHEMA_VERSION = "1"

#: Roles whose feature may sit in a backdoor adjustment set. Instruments are
#: excluded on purpose (adjusting on an instrument inflates variance and
#: amplifies residual bias); mediators / colliders / descendants are excluded by
#: the criterion itself (T-descendants) and by role.
ADJUSTABLE_ROLES: frozenset[str] = frozenset({"confounder", "ancestor"})

_GRADE_RANK = {"direct": 3, "family": 2, "estimand": 1, "unsupported": 0}


@dataclass
class AssembledEdge:
    from_node: str
    to_node: str
    authored_by: list[str]
    grade: str  # best grade over the authoring fragments
    grades_by_feature: dict[str, str]
    citations: list[dict[str, Any]]
    dropped: bool = False
    dropped_reason: Optional[str] = None


@dataclass
class FeatureAssembly:
    feature: str
    fragment_role: Optional[str]
    cohort_role: Optional[str]
    role_drift: bool
    ambiguous: bool
    review_required: bool
    review_reasons: list[str]
    leak_verdict: bool
    leak_source: Optional[str]
    panel_final_role: Optional[str]
    edge_grades: dict[str, str]
    in_adjustment_set: bool
    in_minimal_adjustment_set: bool
    adjustment_exclusion: Optional[str]
    provenance: str
    model_id: str
    n_edges: int


@dataclass
class CohortDag:
    treatment: str
    outcome: str
    nodes: list[str]
    edges: list[AssembledEdge]
    latent_nodes: list[str]
    features: list[FeatureAssembly]
    is_dag: bool
    adjustment_set: list[str]
    minimal_adjustment_set: list[str]
    adjustment_valid: bool
    warnings: list[str]
    merge_latents: bool
    schema_version: str = ASSEMBLER_SCHEMA_VERSION

    # -- shapes the rest of the platform consumes --------------------------------

    def active_edges(self) -> list[list[str]]:
        return [[e.from_node, e.to_node] for e in self.edges if not e.dropped]

    def dag_structure_json(self) -> dict[str, Any]:
        """The ``DagStructureSnapshot`` shape ``DagPanel`` renders (mig 097).

        ``adjustment_sets`` lists the minimal set first, then the full observed
        admissible set when it differs; empty when no admissible observed set
        exists (``adjustment_valid`` False — a finding, not "nothing to adjust").
        """
        sets: list[list[str]] = []
        if self.adjustment_valid:
            sets.append(list(self.minimal_adjustment_set))
            if self.adjustment_set != self.minimal_adjustment_set:
                sets.append(list(self.adjustment_set))
        return {
            "nodes": list(self.nodes),
            "edges": self.active_edges(),
            "treatment_nodes": [self.treatment],
            "outcome_nodes": [self.outcome],
            "adjustment_sets": sets,
            "latent_nodes": list(self.latent_nodes),
            "edge_provenance": [asdict(e) for e in self.edges],
            "adjustment_valid": self.adjustment_valid,
            "is_dag": self.is_dag,
        }

    def evidence_table(self) -> list[dict[str, Any]]:
        """Per-feature rows for ``agent_assessment_json`` (sorted by feature)."""
        return [asdict(f) for f in sorted(self.features, key=lambda f: f.feature)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "merge_latents": self.merge_latents,
            "is_dag": self.is_dag,
            "adjustment_valid": self.adjustment_valid,
            "adjustment_set": list(self.adjustment_set),
            "minimal_adjustment_set": list(self.minimal_adjustment_set),
            "warnings": list(self.warnings),
            "dag_structure_json": self.dag_structure_json(),
            "features": self.evidence_table(),
        }


# ---------------------------------------------------------------------------


def _coerce(att: Union[AuthoredAttestation, Mapping[str, Any]]) -> AuthoredAttestation:
    if isinstance(att, AuthoredAttestation):
        return att
    return AuthoredAttestation.from_dict(att)


def _map_node(
    node: str,
    *,
    feature: str,
    treatment: str,
    outcome: str,
    t_label: str,
    y_label: str,
    merge_latents: bool,
) -> str:
    if node == t_label:
        return treatment
    if node == y_label:
        return outcome
    if node == feature:
        return feature
    # A latent: shared by name, or namespaced to its authoring feature.
    return node if merge_latents else f"{node}@{feature}"


def _greedy_minimal(
    dag: nx.DiGraph, full: Sequence[str], treatment: str, outcome: str
) -> list[str]:
    """Deterministic backward elimination: drop a node when the rest still
    satisfies the criterion. Sorted order → the same answer every run."""
    current = list(full)
    for node in sorted(full):
        trial = [n for n in current if n != node]
        if satisfies_backdoor_criterion(dag, trial, treatment, outcome):
            current = trial
    return sorted(current)


def assemble_cohort_dag(
    attestations: Sequence[Union[AuthoredAttestation, Mapping[str, Any]]],
    *,
    treatment: str,
    outcome: str,
    merge_latents: bool = True,
) -> CohortDag:
    """Union the fragments for one (T, Y) into a cohort DAG (see module doc)."""
    records = [_coerce(a) for a in attestations]
    if treatment == outcome:
        raise ValueError("treatment and outcome must differ")
    seen: set[str] = set()
    for r in records:
        if r.feature_name in seen:
            raise ValueError(f"duplicate attestation for feature {r.feature_name!r}")
        if r.feature_name in (treatment, outcome):
            raise ValueError(f"feature {r.feature_name!r} collides with the treatment/outcome")
        seen.add(r.feature_name)

    warnings: list[str] = []
    edge_index: dict[tuple[str, str], AssembledEdge] = {}
    latents: list[str] = []
    fragment_roles: dict[str, Optional[str]] = {}

    for r in records:
        fragment_roles[r.feature_name] = r.derived_role
        if not r.edges:
            continue
        t_label = r.treatment_node or TREATMENT_NODE
        y_label = r.outcome_node or OUTCOME_NODE
        prov_by_edge = {(p.from_node, p.to_node): p for p in r.edge_provenance}
        for raw_from, raw_to in r.edges:
            key = (
                _map_node(
                    raw_from,
                    feature=r.feature_node,
                    treatment=treatment,
                    outcome=outcome,
                    t_label=t_label,
                    y_label=y_label,
                    merge_latents=merge_latents,
                ),
                _map_node(
                    raw_to,
                    feature=r.feature_node,
                    treatment=treatment,
                    outcome=outcome,
                    t_label=t_label,
                    y_label=y_label,
                    merge_latents=merge_latents,
                ),
            )
            for node, raw in zip(key, (raw_from, raw_to), strict=True):
                if raw.startswith("U_") and node not in latents:
                    latents.append(node)
            prov = prov_by_edge.get((raw_from, raw_to))
            grade = prov.grade if prov is not None else "unsupported"
            violation = prov.constraint_violation if prov is not None else None
            cits = list(prov.citations) if prov is not None else []
            existing = edge_index.get(key)
            if existing is None:
                existing = AssembledEdge(
                    from_node=key[0],
                    to_node=key[1],
                    authored_by=[],
                    grade=grade,
                    grades_by_feature={},
                    citations=[],
                )
                edge_index[key] = existing
            existing.authored_by.append(r.feature_name)
            existing.grades_by_feature[r.feature_name] = grade
            if _GRADE_RANK.get(grade, 0) > _GRADE_RANK.get(existing.grade, 0):
                existing.grade = grade
            existing.citations.extend(cits)
            if violation is not None:
                # Lane E 3(b): the constraint voids the edge in the assembled
                # structure; the author's claim stays visible in provenance.
                existing.dropped = True
                existing.dropped_reason = violation

    # The estimand edge is a causal assumption on every cohort (guide §1.6); an
    # empty attestation list still yields a T -> Y graph so the shape is stable.
    if (treatment, outcome) not in edge_index:
        edge_index[(treatment, outcome)] = AssembledEdge(
            from_node=treatment,
            to_node=outcome,
            authored_by=[],
            grade="estimand",
            grades_by_feature={},
            citations=[],
        )
        if records:
            warnings.append("no fragment carried the estimand edge T -> Y; added as the assumption")

    edges = [edge_index[k] for k in sorted(edge_index)]
    graph = nx.DiGraph()
    graph.add_nodes_from([treatment, outcome])
    for e in edges:
        if e.dropped:
            graph.add_nodes_from([e.from_node, e.to_node])
            continue
        graph.add_edge(e.from_node, e.to_node)
    for e in edges:
        if e.dropped:
            warnings.append(
                f"edge {e.from_node} -> {e.to_node} dropped: {e.dropped_reason} "
                f"(authored by {', '.join(e.authored_by)})"
            )

    is_dag = nx.is_directed_acyclic_graph(graph)
    cycle_nodes: set[str] = set()
    if not is_dag:
        cycle = nx.find_cycle(graph)
        cycle_nodes = {u for u, _ in cycle} | {v for _, v in cycle}
        warnings.append(
            "the union of fragments is not acyclic (shared latents close a cycle): "
            + " -> ".join(u for u, _ in cycle)
            + f" -> {cycle[0][0]}; every feature on it is routed to review"
        )

    latent_set = set(latents)
    observed_features = [r.feature_name for r in records if r.edges]
    # Roles on the union (only meaningful on a DAG).
    cohort_roles: dict[str, Optional[str]] = {}
    role_errors: dict[str, str] = {}
    for feat in observed_features:
        if not is_dag or feat not in graph:
            cohort_roles[feat] = None
            continue
        try:
            cohort_roles[feat] = extract_role(feat, treatment, outcome, graph)
        except ValueError as exc:
            cohort_roles[feat] = None
            role_errors[feat] = str(exc)

    # Adjustment candidates: observed, adjustable role, no leak verdict.
    exclusions: dict[str, str] = {}
    candidates: list[str] = []
    for r in records:
        feat = r.feature_name
        if not r.edges:
            exclusions[feat] = "no authored fragment (review only)"
            continue
        leak = bool(r.panel_summary.get("leak_verdict")) if r.panel_summary else False
        if leak:
            exclusions[feat] = f"panel leak verdict ({r.panel_summary.get('leak_source')})"
            continue
        if feat in cycle_nodes:
            exclusions[feat] = "on a cycle in the union"
            continue
        role = cohort_roles.get(feat)
        if role is None:
            exclusions[feat] = "unclassifiable on the union"
            continue
        if role not in ADJUSTABLE_ROLES:
            exclusions[feat] = f"cohort role {role} is not a backdoor variable"
            continue
        candidates.append(feat)
    candidates = sorted(candidates)

    adjustment_valid = False
    full_set: list[str] = []
    minimal: list[str] = []
    if is_dag:
        if satisfies_backdoor_criterion(graph, candidates, treatment, outcome):
            adjustment_valid = True
            full_set = list(candidates)
            minimal = _greedy_minimal(graph, candidates, treatment, outcome)
        else:
            warnings.append(
                "no admissible OBSERVED adjustment set: the full candidate set "
                f"{candidates} does not block every backdoor path (a latent "
                f"{sorted(latent_set)} reaches both T and Y with no observed child on "
                "the path, or a required feature was excluded) — a finding for the reviewer"
            )
    features: list[FeatureAssembly] = []
    for r in records:
        feat = r.feature_name
        frag_role = fragment_roles.get(feat)
        cohort_role = cohort_roles.get(feat)
        drift = bool(r.edges) and is_dag and frag_role != cohort_role
        reasons = list(r.review_reasons)
        if drift:
            reasons.append(
                f"role drift on the cohort DAG: fragment {frag_role!r} -> union {cohort_role!r}"
            )
        if feat in role_errors:
            reasons.append(f"unclassifiable on the union: {role_errors[feat]}")
        if feat in cycle_nodes:
            reasons.append("on a cycle in the union of fragments")
        panel = r.panel_summary or {}
        features.append(
            FeatureAssembly(
                feature=feat,
                fragment_role=frag_role,
                cohort_role=cohort_role,
                role_drift=drift,
                ambiguous=bool(r.ambiguous),
                review_required=bool(r.review_required) or drift or feat in cycle_nodes,
                review_reasons=reasons,
                leak_verdict=bool(panel.get("leak_verdict", False)),
                leak_source=panel.get("leak_source"),
                panel_final_role=panel.get("ensemble_final_role"),
                edge_grades={f"{p.from_node}->{p.to_node}": p.grade for p in r.edge_provenance},
                in_adjustment_set=feat in full_set,
                in_minimal_adjustment_set=feat in minimal,
                adjustment_exclusion=exclusions.get(feat),
                provenance=r.provenance,
                model_id=r.model_id,
                n_edges=len(r.edges),
            )
        )

    nodes = sorted(graph.nodes(), key=lambda n: (n != treatment, n != outcome, n))
    return CohortDag(
        treatment=treatment,
        outcome=outcome,
        nodes=nodes,
        edges=edges,
        latent_nodes=sorted(latent_set),
        features=features,
        is_dag=is_dag,
        adjustment_set=full_set,
        minimal_adjustment_set=minimal,
        adjustment_valid=adjustment_valid,
        warnings=warnings,
        merge_latents=merge_latents,
    )
