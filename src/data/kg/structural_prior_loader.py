"""Approved structural-author review → the causal agent's structural prior.

Lane B of the real-data causal estimation program (spec §3 Lane B item 5):
an APPROVED expert review of a structural-author cohort DAG becomes the run's
``anchored_confounders`` (the structural-prior channel that already exists in
``src/api/routes/causal/agent.py`` / ``graph_builder._resolve_anchored_confounders``)
and its ``approved_structure_roles`` (Lane E's channel, consumed by
``derive_confounder_channels`` once that branch is on the tree). Unapproved
machine attestations are never used as priors: a pending, rejected or expired
review yields no prior, and a review that carries no structural-author
evidence is not a structural prior at all.

What is read, and from where. ``scripts/author_cohort_dag.py --review`` writes
the assembled DAG to ``expert_reviews.dag_structure_json`` (nodes, edges,
treatment/outcome, adjustment sets) and the per-feature evidence table
(feature → leak verdict, review flags, provenance, the cohort role the
assembler derived) to ``agent_assessment_json["structural_author"]``. The
prior is DERIVED FROM THE APPROVED SNAPSHOT (codex r1 HIGH 2): the row's
``dag_version_hash`` must be the hash of the stored snapshot (recomputed with
``compute_dag_hash``), the row's ``adjustment_set_hash`` must be what the
snapshot proves (``adjustment_hash_from_snapshot``), the evidence must have
graded that same hash, and every role is re-derived from the snapshot's edges
with ``extract_role`` — an evidence row that claims a role the approved edges
do not derive fails closed. Anchored confounders are the derived confounders
inside the approved adjustment sets with no leak verdict. The docs tree (and
``docs/layer4/generated/<manifest>_<T>_<Y>/attestations.json``) is not in the
API image (``.dockerignore:76``); when the file IS reachable (the dev box, a
CLI) it is cross-checked against the row — feature set and fragment roles —
and a mismatch fails closed.

Failure policy: a store outage or a malformed row never fails a run — the run
proceeds WITHOUT a prior and says so in ``warnings`` (spec §5). Fail-closed
mismatches raise :class:`StructuralPriorError` from the pure functions and are
reported as warnings by the async resolver.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

import networkx as nx

from src.ml.causal_role_dgp.extractor import extract_role
from src.repositories.expert_review import estimand_key_for, is_active_approval

logger = logging.getLogger(__name__)

__all__ = [
    "ApprovedStructuralPrior",
    "StructuralPriorError",
    "apply_structural_prior_to_state",
    "find_approved_structural_prior_row",
    "load_approved_structural_prior",
    "resolve_structural_prior_for_run",
    "structural_prior_from_review_row",
]

EVIDENCE_KEY = "structural_author"
REVIEW_TYPE = "initial_dag"


class StructuralPriorError(ValueError):
    """The approved row and the authored evidence do not describe one structure."""


@dataclass(frozen=True)
class ApprovedStructuralPrior:
    review_id: str
    dag_version_hash: str
    adjustment_set_hash: str
    #: feature → role DERIVED from the approved snapshot's edges.
    roles: dict[str, str]
    #: derived confounders inside the approved adjustment sets, no leak verdict.
    anchored_confounders: list[str]
    #: derived instruments (not adjusted, not anchored: graph_builder would
    #: force ``inst -> outcome`` for an anchored node).
    instruments: list[str]
    #: feature → why it is neither anchored nor an instrument.
    excluded: dict[str, str]
    source: str
    manifest: Optional[str] = None
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    model_id: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_dict(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    return value if isinstance(value, dict) else None


def _snapshot_graph(snapshot: Mapping[str, Any], review_id: str) -> tuple[nx.DiGraph, str, str]:
    nodes = snapshot.get("nodes")
    edges = snapshot.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise StructuralPriorError(f"review {review_id}: the approved snapshot has no nodes/edges")
    t_nodes = snapshot.get("treatment_nodes") or []
    y_nodes = snapshot.get("outcome_nodes") or []
    if len(t_nodes) != 1 or len(y_nodes) != 1:
        raise StructuralPriorError(
            f"review {review_id}: the approved snapshot must name exactly one treatment and one "
            f"outcome node (got {t_nodes!r}, {y_nodes!r})"
        )
    graph = nx.DiGraph()
    graph.add_nodes_from(str(n) for n in nodes)
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            raise StructuralPriorError(f"review {review_id}: malformed edge {e!r} in the snapshot")
        graph.add_edge(str(e[0]), str(e[1]))
    if not nx.is_directed_acyclic_graph(graph):
        raise StructuralPriorError(f"review {review_id}: the approved snapshot is not acyclic")
    return graph, str(t_nodes[0]), str(y_nodes[0])


def structural_prior_from_review_row(
    row: Mapping[str, Any],
    *,
    attestations_path: Optional[Path | str] = None,
    today: Optional[date] = None,
    manifest: Optional[str] = None,
) -> Optional[ApprovedStructuralPrior]:
    """The prior an approved row yields, or ``None`` when the row is no prior.

    ``None``: not an active approval (pending / rejected / expired), not an
    ``initial_dag`` review, or no structural-author evidence on the row.
    Raises :class:`StructuralPriorError` on a fail-closed mismatch: the row's
    hashes are not the stored snapshot's (both halves of the version pair are
    required), the evidence graded another hash, a claimed role is not what
    the approved edges derive, the review was authored for another feature
    manifest than ``manifest`` (when given), or the attestations file (when
    reachable) disagrees with the row.
    """
    if not is_active_approval(row, today):
        return None
    if row.get("review_type") != REVIEW_TYPE:
        return None
    evidence = _as_dict(row.get("agent_assessment_json"))
    if not evidence:
        return None
    sa = _as_dict(evidence.get(EVIDENCE_KEY))
    if not sa:
        return None
    review_id = str(row.get("review_id") or "")
    if manifest is not None and sa.get("manifest") != manifest:
        raise StructuralPriorError(
            f"review {review_id}: authored for manifest {sa.get('manifest')!r}, the run's dataset "
            f"declares {manifest!r} — not the same feature contract"
        )

    # The approved structure IS the stored snapshot; the hashes must prove it.
    from src.causal_engine.dag_hash import adjustment_hash_from_snapshot, compute_dag_hash

    snapshot = _as_dict(row.get("dag_structure_json"))
    if not snapshot:
        raise StructuralPriorError(f"review {review_id}: approved row carries no DAG snapshot")
    row_hash = row.get("dag_version_hash")
    snap_hash = compute_dag_hash(causal_graph=snapshot)
    if not row_hash or row_hash != snap_hash:
        raise StructuralPriorError(
            f"review {review_id}: dag_version_hash {row_hash!r} is not the hash of the stored "
            f"snapshot ({snap_hash})"
        )
    if sa.get("dag_version_hash") != row_hash:
        raise StructuralPriorError(
            f"review {review_id}: the structural author graded {sa.get('dag_version_hash')!r}, "
            f"the approved structure is {row_hash!r} — the structure moved after authoring"
        )
    snap_adj = adjustment_hash_from_snapshot(snapshot)
    if snap_adj is None:
        raise StructuralPriorError(
            f"review {review_id}: the snapshot's adjustment sets are unprovable (malformed)"
        )
    # Both halves of the version pair are REQUIRED (codex r2 HIGH 1): the CLI
    # always writes them; a row or evidence without one is not bound to the
    # complete version the human approved.
    row_adj = row.get("adjustment_set_hash")
    if row_adj != snap_adj:
        raise StructuralPriorError(
            f"review {review_id}: adjustment_set_hash {row_adj!r} is not what the snapshot "
            f"proves ({snap_adj}) — the covariate set moved or was never bound"
        )
    if sa.get("adjustment_set_hash") != snap_adj:
        raise StructuralPriorError(
            f"review {review_id}: the structural author graded adjustment sets "
            f"{sa.get('adjustment_set_hash')!r}, the approved snapshot proves {snap_adj!r}"
        )
    graph, treatment, outcome = _snapshot_graph(snapshot, review_id)
    for col, node in (("treatment_variable", treatment), ("outcome_variable", outcome)):
        if row.get(col) not in (None, node):
            raise StructuralPriorError(
                f"review {review_id}: {col}={row.get(col)!r} but the snapshot's node is {node!r}"
            )
    approved_union: set[str] = set()
    for s in snapshot.get("adjustment_sets") or []:
        approved_union.update(str(v) for v in s)

    features = sa.get("features")
    if not isinstance(features, list) or not features:
        raise StructuralPriorError(
            f"review {review_id}: structural_author evidence has no features"
        )

    roles: dict[str, str] = {}
    anchored: list[str] = []
    instruments: list[str] = []
    excluded: dict[str, str] = {}
    for f in features:
        name = str(f.get("feature"))
        claimed = f.get("cohort_role")
        if f.get("leak_verdict"):
            excluded[name] = f"leak verdict ({f.get('leak_source')})"
            continue
        if claimed is None:
            excluded[name] = "no cohort role (review only)"
            continue
        if name not in graph:
            raise StructuralPriorError(
                f"review {review_id}: evidence names {name!r}, which is not in the approved DAG"
            )
        try:
            derived = extract_role(name, treatment, outcome, graph)
        except ValueError as exc:
            raise StructuralPriorError(
                f"review {review_id}: {name!r} is unclassifiable on the approved DAG: {exc}"
            ) from exc
        if derived != claimed:
            raise StructuralPriorError(
                f"review {review_id}: evidence claims {name!r} is a {claimed}, the approved edges "
                f"derive {derived}"
            )
        roles[name] = derived
        if derived == "confounder" and name in approved_union:
            anchored.append(name)
        elif derived == "instrument":
            instruments.append(name)
        else:
            excluded[name] = f"cohort role {derived}" + (
                " not in the approved adjustment sets" if derived == "confounder" else ""
            )

    warnings: list[str] = []
    unlisted = sorted(approved_union - set(roles) - set(excluded))
    if unlisted:
        warnings.append(
            "approved adjustment set names feature(s) the evidence table does not cover, "
            f"not anchored: {', '.join(unlisted)}"
        )
    source = "review_row"
    if attestations_path is not None:
        path = Path(attestations_path)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("records") or []
            file_feats = {str(r.get("feature_name")): r for r in records}
            row_feats = {str(f.get("feature")): f for f in features}
            if set(file_feats) != set(row_feats):
                raise StructuralPriorError(
                    f"review {review_id}: attestations file {path} covers "
                    f"{len(file_feats)} features, the approved row {len(row_feats)}"
                )
            for name, rec in file_feats.items():
                if rec.get("derived_role") != row_feats[name].get("fragment_role"):
                    raise StructuralPriorError(
                        f"review {review_id}: {name} fragment role differs between the file "
                        f"({rec.get('derived_role')!r}) and the approved row "
                        f"({row_feats[name].get('fragment_role')!r})"
                    )
                if rec.get("provenance") not in ("machine", "machine_reviewed", "human"):
                    raise StructuralPriorError(f"review {review_id}: {name} has no provenance")
            source = "review_row+attestations_file"
        else:
            warnings.append(
                f"attestations file {path} not reachable here; roles derived from the review row"
            )

    return ApprovedStructuralPrior(
        review_id=review_id,
        dag_version_hash=str(row_hash),
        adjustment_set_hash=snap_adj,
        roles=roles,
        anchored_confounders=sorted(anchored),
        instruments=sorted(instruments),
        excluded=excluded,
        source=source,
        manifest=sa.get("manifest"),
        treatment=treatment,
        outcome=outcome,
        model_id=sa.get("model_id"),
        warnings=warnings,
    )


async def find_approved_structural_prior_row(
    repo: Any,
    *,
    treatment: str,
    outcome: str,
    brand: Optional[str] = None,
    today: Optional[date] = None,
    manifest: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Newest ACTIVE ``initial_dag`` approval of this estimand that carries
    structural-author evidence for ``manifest`` (when given). The brand-scoped
    estimand key is tried first, then the brand-less key the CLI writes when
    no ``--brand`` was given."""
    keys = [estimand_key_for(brand, treatment, outcome)]
    if brand:
        keys.append(estimand_key_for(None, treatment, outcome))
    for key in keys:
        rows = await repo.get_reviews_for_estimand(key, include_expired=False)
        for row in rows or []:
            if not is_active_approval(row, today):
                continue
            if row.get("review_type") != REVIEW_TYPE:
                continue
            evidence = _as_dict(row.get("agent_assessment_json")) or {}
            sa = _as_dict(evidence.get(EVIDENCE_KEY))
            if not sa:
                continue
            if manifest is not None and sa.get("manifest") != manifest:
                continue
            return dict(row)
    return None


async def load_approved_structural_prior(
    review_id: str,
    repo: Any,
    *,
    attestations_path: Optional[Path | str] = None,
) -> Optional[ApprovedStructuralPrior]:
    """The prior for one review id (``repo.get_by_id``), or ``None``."""
    row = await repo.get_by_id(review_id)
    if not row:
        return None
    return structural_prior_from_review_row(row, attestations_path=attestations_path)


async def resolve_structural_prior_for_run(
    *,
    treatment: str,
    outcome: str,
    brand: Optional[str],
    manifest: Optional[str],
    repo_factory: Callable[[], Awaitable[Any]],
) -> tuple[Optional[ApprovedStructuralPrior], list[str]]:
    """Look the prior up for a run; never raises.

    ``manifest`` is the feature-manifest source the run's dataset declares
    (``feature_manifest_source`` on its registry spec). A structural review is
    authored FOR a manifest, and (treatment, outcome) names are not unique
    across datasets (codex r2 HIGH 3), so a dataset that declares no manifest
    gets no prior — reported, never guessed.

    Returns ``(prior, warnings)``: a store outage, a missing row or a
    fail-closed mismatch yields ``(None, [why])`` — the run proceeds without a
    prior and the reason reaches the response's ``warnings``.
    """
    if not manifest:
        return None, [
            "structural prior not applied: the dataset declares no feature_manifest_source, "
            "so no authored structure can be matched to it"
        ]
    try:
        repo = await repo_factory()
        row = await find_approved_structural_prior_row(
            repo, treatment=treatment, outcome=outcome, brand=brand, manifest=manifest
        )
        if row is None:
            return None, []
        prior = structural_prior_from_review_row(row, manifest=manifest)
        return prior, list(prior.warnings) if prior else []
    except StructuralPriorError as exc:
        logger.warning("structural prior refused: %s", exc)
        return None, [f"structural prior refused (fail closed): {exc}"]
    except Exception as exc:  # noqa: BLE001 — a store outage never fails the run
        logger.warning("structural prior lookup failed: %s", exc)
        return None, [f"structural prior not consulted: {type(exc).__name__}: {exc}"]


def apply_structural_prior_to_state(
    state: dict[str, Any],
    prior: ApprovedStructuralPrior,
    *,
    covariates: Sequence[str],
) -> list[str]:
    """Seed the agent state's structural-prior channels from ``prior``.

    ``anchored_confounders`` (graph_builder forces ``conf -> T`` and
    ``conf -> Y`` for each) is restricted to the run's declared covariates —
    an approved confounder the caller did not offer cannot be adjusted for and
    is named in the returned warnings; a confounder approved by its root name
    anchors each declared ``root=level`` dummy. ``approved_structure_roles``
    carries every approved role (Lane E's ``derive_confounder_channels`` maps
    dummies back to their root column, so roles are passed unfiltered).
    Returns the provenance lines appended to ``state["warnings"]``.
    """
    # The estimation loader one-hot encodes categoricals, so a run's covariate
    # may be ``root=level``; an approved confounder named by its root anchors
    # every declared dummy of that root (the root itself is not a frame column
    # and graph_builder would drop it silently — codex r3 MED 2).
    declared = [str(c) for c in covariates]
    anchored: list[str] = []
    missing: list[str] = []
    for conf in prior.anchored_confounders:
        matches = [c for c in declared if c == conf or c.split("=", 1)[0] == conf]
        if matches:
            anchored.extend(m for m in matches if m not in anchored)
        else:
            missing.append(conf)
    state["anchored_confounders"] = anchored
    state["approved_structure_roles"] = dict(prior.roles)
    lines = [
        f"structural prior: approved expert review {prior.review_id} "
        f"(dag {prior.dag_version_hash[:12]}, {prior.source}) anchors "
        f"{len(anchored)} confounder(s); {len(prior.instruments)} instrument(s), "
        f"{len(prior.excluded)} excluded"
    ]
    if missing:
        lines.append(
            "structural prior: approved confounder(s) not among this run's covariates, "
            f"not anchored: {', '.join(missing)}"
        )
    lines.extend(prior.warnings)
    state.setdefault("warnings", []).extend(lines)
    return lines
