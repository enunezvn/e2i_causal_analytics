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
the assembled DAG to ``expert_reviews.dag_structure_json`` and the per-feature
evidence table (feature → cohort role, adjustment membership, leak verdict,
review flags, provenance) to ``agent_assessment_json["structural_author"]``,
next to the DAG hash it graded. The loader reads the roles from that row, so
it works inside the API container, where the ``docs/`` tree (and with it
``docs/layer4/generated/<manifest>_<T>_<Y>/attestations.json``) is not shipped
(``.dockerignore:76``). When the attestations file IS reachable (the dev box,
a CLI) it is cross-checked against the row — feature set and fragment roles —
and a mismatch fails closed. The row's ``dag_version_hash`` must equal the hash
the evidence was graded against: a review whose structure advanced after the
author wrote its evidence is not an approval of the authored DAG.

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
    #: feature → cohort role on the approved DAG (every feature that has one).
    roles: dict[str, str]
    #: approved confounders in the approved adjustment set, no leak verdict.
    anchored_confounders: list[str]
    #: approved instruments (not adjusted, not anchored: graph_builder would
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


def structural_prior_from_review_row(
    row: Mapping[str, Any],
    *,
    attestations_path: Optional[Path | str] = None,
    today: Optional[date] = None,
) -> Optional[ApprovedStructuralPrior]:
    """The prior an approved row yields, or ``None`` when the row is no prior.

    ``None``: not an active approval (pending / rejected / expired), not an
    ``initial_dag`` review, or no structural-author evidence on the row.
    Raises :class:`StructuralPriorError` on a fail-closed mismatch (hash,
    feature set, fragment roles).
    """
    if not is_active_approval(row, today):
        return None
    if row.get("review_type") not in (None, REVIEW_TYPE):
        return None
    evidence = _as_dict(row.get("agent_assessment_json"))
    if not evidence:
        return None
    sa = _as_dict(evidence.get(EVIDENCE_KEY))
    if not sa:
        return None
    review_id = str(row.get("review_id") or "")
    row_hash = row.get("dag_version_hash")
    sa_hash = sa.get("dag_version_hash")
    if not row_hash or not sa_hash or row_hash != sa_hash:
        raise StructuralPriorError(
            f"review {review_id}: approved dag_version_hash {row_hash!r} is not the hash the "
            f"structural author graded {sa_hash!r} — the structure moved after authoring"
        )
    features = sa.get("features")
    if not isinstance(features, list) or not features:
        raise StructuralPriorError(
            f"review {review_id}: structural_author evidence has no features"
        )
    approved_set = {str(f) for f in (sa.get("adjustment_set") or [])}

    roles: dict[str, str] = {}
    anchored: list[str] = []
    instruments: list[str] = []
    excluded: dict[str, str] = {}
    for f in features:
        name = str(f.get("feature"))
        role = f.get("cohort_role")
        if f.get("leak_verdict"):
            excluded[name] = f"leak verdict ({f.get('leak_source')})"
            continue
        if role is None:
            excluded[name] = "no cohort role (review only)"
            continue
        roles[name] = str(role)
        if role == "confounder" and name in approved_set:
            anchored.append(name)
        elif role == "instrument":
            instruments.append(name)
        else:
            excluded[name] = f"cohort role {role}" + (
                "" if role != "confounder" else " not in the approved adjustment set"
            )

    warnings: list[str] = []
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
                f"attestations file {path} not reachable here; roles read from the review row"
            )

    return ApprovedStructuralPrior(
        review_id=review_id,
        dag_version_hash=str(row_hash),
        roles=roles,
        anchored_confounders=sorted(anchored),
        instruments=sorted(instruments),
        excluded=excluded,
        source=source,
        manifest=sa.get("manifest"),
        treatment=sa.get("treatment"),
        outcome=sa.get("outcome"),
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
) -> Optional[dict[str, Any]]:
    """Newest ACTIVE ``initial_dag`` approval of this estimand that carries
    structural-author evidence. The brand-scoped estimand key is tried first,
    then the brand-less key the CLI writes when no ``--brand`` was given."""
    keys = [estimand_key_for(brand, treatment, outcome)]
    if brand:
        keys.append(estimand_key_for(None, treatment, outcome))
    for key in keys:
        rows = await repo.get_reviews_for_estimand(key, include_expired=False)
        for row in rows or []:
            if not is_active_approval(row, today):
                continue
            if row.get("review_type") not in (None, REVIEW_TYPE):
                continue
            evidence = _as_dict(row.get("agent_assessment_json")) or {}
            if _as_dict(evidence.get(EVIDENCE_KEY)):
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
    repo_factory: Callable[[], Awaitable[Any]],
) -> tuple[Optional[ApprovedStructuralPrior], list[str]]:
    """Look the prior up for a run; never raises.

    Returns ``(prior, warnings)``: a store outage, a missing row or a
    fail-closed mismatch yields ``(None, [why])`` — the run proceeds without a
    prior and the reason reaches the response's ``warnings``.
    """
    try:
        repo = await repo_factory()
        row = await find_approved_structural_prior_row(
            repo, treatment=treatment, outcome=outcome, brand=brand
        )
        if row is None:
            return None, []
        prior = structural_prior_from_review_row(row)
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
    is named in the returned warnings. ``approved_structure_roles`` carries
    every approved role (Lane E's ``derive_confounder_channels`` maps dummies
    ``<col>=<level>`` back to their root column, so roles are passed unfiltered).
    Returns the provenance lines appended to ``state["warnings"]``.
    """
    declared = {str(c) for c in covariates}
    roots = {c.split("=", 1)[0] for c in declared}
    anchored = [c for c in prior.anchored_confounders if c in declared or c in roots]
    missing = [c for c in prior.anchored_confounders if c not in declared and c not in roots]
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
