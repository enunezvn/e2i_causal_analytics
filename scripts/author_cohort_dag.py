"""Author the cohort DAG for one (manifest, T, Y) and, on request, queue it for review.

Lane B of the real-data causal estimation program (spec §3 Lane B item 4)::

    python -m scripts.author_cohort_dag --manifest {optum_mart,optum} \\
        --treatment <T column> --outcome <Y column> \\
        [--treatment-label "..."] [--outcome-label "..."] \\
        [--panel <Lane E panel.json>] [--features a,b,c] \\
        [--lm fake|real] [--i-accept-cost] [--resolver offline|live] \\
        [--out-root docs/layer4/generated] [--review [--allow-fake-review]] \\
        [--diff-manifest-attestations]

Writes ``<out-root>/<manifest>_<T>_<Y>/{attestations.json, dag.json, review.md}``
(plus ``manifest_diff.json`` for run (b) and ``review.json`` after ``--review``).
With ``--review`` it opens an ``expert_reviews`` row (``review_type='initial_dag'``,
``dag_structure_json`` = the assembled DAG, ``agent_assessment_json`` = the
per-feature evidence table) through ``ExpertReviewRepository`` — a prod write,
so it is refused for a fake-LM dry run unless ``--allow-fake-review`` is given,
and it fails loudly (exit 3, files kept) when no review row was created.

The two runs the spec names, both OWNER decisions (paid LLM + prod write):

(a) ``--manifest optum_mart --treatment treatment_dupixent --outcome
    persistent_at_180d_g28 --treatment-label "remibrutinib vs competitor
    biologic (CSU escalation therapy)" --review`` — the human gate. The brief
    frames T as the CSU escalation-therapy choice; the stated assumption the
    reviewer confirms as a checklist item: remibrutinib enters at the same
    decision point as the biologics (second line after H1-antihistamine
    failure), so the confounders of WHICH escalation therapy are the same set,
    and Lane A's rehearsal uses this DAG with ``treatment_dupixent``.
(b) ``--manifest optum --treatment biologic_initiation --outcome
    initiated_biologic_180d --diff-manifest-attestations`` — diff against the
    110 existing (machine-labelled) attestations: agreement per edge and per
    role, disagreements listed with both rationales, no threshold.

``--lm fake`` runs the whole pipeline with a ``DummyLM`` that answers a plain
confounder fragment for every feature (a stand-in, stamped ``model_id=dummy``)
so the artefacts, the review shape and the diff can be rehearsed at zero cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logger = logging.getLogger("author_cohort_dag")

_PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "docs" / "layer4" / "generated"
MANIFESTS = ("optum_mart", "optum")
OPTUM_RESEARCH_DOC = "docs/layer4/optum_initiation_attestation_research.md"

#: Lane E FeatureRoleRecord contract (PR #2226): the fields the author's
#: constraints read, all REQUIRED on every panel record.
_PANEL_RECORD_REQUIRED = (
    "feature",
    "layer_1",
    "layer_2",
    "layer_3",
    "layer_4",
    "ensemble",
    "leak_verdict",
    "leak_source",
    "review_required",
)
_PANEL_L1_VERDICTS = ("pre_index", "post_index", "no_contract")
_PANEL_ROLES_OR_NONE = (
    None,
    "ancestor",
    "confounder",
    "instrument",
    "mediator",
    "collider",
    "descendant",
)
_PANEL_LEAK_SOURCES = (None, "layer_1_post_index", "layer_3_high")

#: The stated assumption of run (a), confirmed by the reviewer as a checklist item.
ESCALATION_ASSUMPTION = (
    "Remibrutinib enters at the same decision point as the biologics (second line "
    "after H1-antihistamine failure), so the confounders of WHICH escalation therapy "
    "a patient receives are the same set; Lane A's rehearsal therefore uses this DAG "
    "with treatment_dupixent (Dupixent vs Xolair) on the same confounders."
)


class OfflineResolver:
    """Records every citation as unverifiable: no network on a fake run."""

    def verify_citation(self, identifier, *, identifier_kind, subject_name, object_name):
        from src.data.kg.types import CitationVerdict

        return CitationVerdict(
            identifier=identifier,
            identifier_kind=identifier_kind,
            abstract_resolved=False,
            error="offline resolver (fake run): citation not checked",
        )


# ---------------------------------------------------------------------------
# Briefs
# ---------------------------------------------------------------------------


def _safe_features(manifest: str) -> list[str]:
    from src.data import manifests as m

    if manifest == "optum_mart":
        return list(m.MART_SAFE_FEATURES)
    if manifest == "optum":
        return list(m.OPTUM_SAFE_FEATURES)
    raise ValueError(f"unknown manifest {manifest!r} (known: {MANIFESTS})")


def _layer4_inputs(feature: str, contract: Any, target: str, manifest: str, treatment: str):
    """``(derivation_pseudocode, dataset_context)`` as the Layer-4 classifier gets
    them, with the causal treatment appended (Lane E's format). Prefers Lane E's
    light helper when it is on the tree; otherwise the node's own builder."""
    try:
        from src.data.layer4_inputs import build_layer_4_inputs  # Lane E (PR #2226)

        return build_layer_4_inputs(feature, contract, target, manifest, treatment=treatment)
    except ImportError:
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _build_layer_4_inputs,
        )

        derivation, context = _build_layer_4_inputs(feature, contract, target, manifest)
        context += f"; treatment={treatment}; causal_question=effect of {treatment} on {target}"
        return derivation, context


def build_briefs(
    *,
    manifest: str,
    treatment: str,
    outcome: str,
    treatment_label: str,
    outcome_label: str,
    features: Optional[list[str]] = None,
    panel: Optional[dict[str, Any]] = None,
) -> list[Any]:
    from src.data.kg.structural_author import build_brief
    from src.data.manifests import lookup_feature_contract

    names = features if features else _safe_features(manifest)
    records = (panel or {}).get("records") or {}
    if panel is not None:
        # A panel that does not cover a feature would silently drop the Lane E
        # constraints for it (post-index veto, leak exclusion): refuse.
        missing = [f for f in names if f not in records]
        if missing:
            raise ValueError(f"panel has no record for {len(missing)} feature(s): {missing}")
    briefs = []
    for feat in names:
        contract = lookup_feature_contract(feat, data_source=manifest)
        derivation, context = _layer4_inputs(feat, contract, outcome, manifest, treatment)
        briefs.append(
            build_brief(
                feat,
                derivation_pseudocode=derivation,
                dataset_context=context,
                treatment_label=treatment_label,
                outcome_label=outcome_label,
                panel_record=records.get(feat),
            )
        )
    return briefs


def _load_panel(path: Path, *, manifest: str, treatment: str, outcome: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, want in (
        ("manifest_source", manifest),
        ("treatment", treatment),
        ("outcome", outcome),
    ):
        got = payload.get(key)
        if got != want:
            raise ValueError(
                f"panel {path}: {key}={got!r} does not match --{key.replace('_source', '')} {want!r}"
            )
    records = payload.get("records")
    if not isinstance(records, dict) or not records:
        raise ValueError(f"panel {path}: no records")
    for key, rec in records.items():
        if not isinstance(rec, dict):
            raise ValueError(f"panel {path}: record {key!r} is not an object")
        if rec.get("feature") != key:
            raise ValueError(f"panel {path}: record {key!r} carries feature={rec.get('feature')!r}")
        # Every safety-critical field must be PRESENT with the right type: a
        # missing field would silently default the Layer-1 veto, the leak
        # exclusion or the cross-check away (codex r2 HIGH 2).
        for field_name in _PANEL_RECORD_REQUIRED:
            if field_name not in rec:
                raise ValueError(f"panel {path}: record {key!r} lacks {field_name!r}")
        for flag in ("leak_verdict", "review_required"):
            if not isinstance(rec[flag], bool):
                raise ValueError(f"panel {path}: record {key!r}.{flag} is not a bool")
        for layer in ("layer_1", "layer_2", "layer_3", "layer_4", "ensemble"):
            if not isinstance(rec[layer], dict):
                raise ValueError(f"panel {path}: record {key!r}.{layer} is not an object")
        if rec["layer_1"].get("verdict") not in _PANEL_L1_VERDICTS:
            raise ValueError(
                f"panel {path}: record {key!r}.layer_1.verdict={rec['layer_1'].get('verdict')!r} "
                f"is not one of {_PANEL_L1_VERDICTS}"
            )
        if not isinstance(rec["layer_3"].get("ran"), bool):
            raise ValueError(f"panel {path}: record {key!r}.layer_3.ran is not a bool")
        if "final_role" not in rec["ensemble"] or "decided_by" not in rec["ensemble"]:
            raise ValueError(f"panel {path}: record {key!r}.ensemble lacks final_role/decided_by")
        if rec["ensemble"]["final_role"] not in _PANEL_ROLES_OR_NONE:
            raise ValueError(
                f"panel {path}: record {key!r}.ensemble.final_role="
                f"{rec['ensemble']['final_role']!r} is not a role"
            )
        if rec["leak_source"] not in _PANEL_LEAK_SOURCES:
            raise ValueError(
                f"panel {path}: record {key!r}.leak_source={rec['leak_source']!r} is not one of "
                f"{_PANEL_LEAK_SOURCES}"
            )
    return payload


# ---------------------------------------------------------------------------
# Run (b): diff against the manifest's machine attestations
# ---------------------------------------------------------------------------


def manifest_grounding(
    feature: str, *, doc: Path = PROJECT_ROOT / OPTUM_RESEARCH_DOC
) -> dict[str, Any]:
    """The manifest side's rationale for ``feature``: the family bullet(s) of the
    Optum research record that name the feature (with their PMIDs and the
    doc line), or an explicit "no grounding found". The manifest code itself
    carries only the two edge patterns, not prose."""
    if not doc.exists():
        return {"found": False, "source": str(doc), "note": "research record not on this tree"}
    bullets: list[dict[str, Any]] = []
    needle = f"`{feature}`"
    for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("-") and needle in line:
            pmids = sorted(set(re.findall(r"PMID\s*(\d{1,9})", line)))
            bullets.append({"line": lineno, "text": line.strip(), "pmids": pmids})
    return {
        "found": bool(bullets),
        "source": OPTUM_RESEARCH_DOC,
        "bullets": bullets,
        "edge_pattern": "src/data/manifests/optum_feature_manifest.py::_optum_attestation",
        "note": None if bullets else f"no bullet in {OPTUM_RESEARCH_DOC} names `{feature}`",
    }


def diff_against_manifest(records: list[Any], *, manifest: str) -> dict[str, Any]:
    """Agreement per edge and per role against the manifest's existing
    ``CausalStructureAttestation`` for each feature (no threshold)."""
    import networkx as nx

    from src.data.manifests import lookup_feature_contract
    from src.ml.causal_role_dgp.extractor import extract_role

    rows: list[dict[str, Any]] = []
    n_edge_exact = n_role_agree = n_compared = 0
    for rec in records:
        contract = lookup_feature_contract(rec.feature_name, data_source=manifest)
        att = contract.causal_structure if contract is not None else None
        if att is None or not rec.edges:
            rows.append(
                {
                    "feature": rec.feature_name,
                    "compared": False,
                    "reason": "no manifest attestation" if att is None else "no authored fragment",
                }
            )
            continue
        n_compared += 1
        # Map the authored fragment onto the manifest's node labels.
        relabel = {rec.treatment_node: att.treatment_node, rec.outcome_node: att.outcome_node}
        authored = {(relabel.get(a, a), relabel.get(b, b)) for a, b in rec.edges}
        manifest_edges = {tuple(e) for e in att.edges}
        inter = authored & manifest_edges
        union = authored | manifest_edges
        exact = authored == manifest_edges
        try:
            manifest_role = extract_role(
                att.feature_node, att.treatment_node, att.outcome_node, nx.DiGraph(list(att.edges))
            )
        except ValueError as exc:
            manifest_role = None
            logger.warning("manifest attestation for %s unclassifiable: %s", rec.feature_name, exc)
        role_agree = rec.derived_role == manifest_role
        n_edge_exact += int(exact)
        n_role_agree += int(role_agree)
        row = {
            "feature": rec.feature_name,
            "compared": True,
            "edge_exact": exact,
            "edge_jaccard": (len(inter) / len(union)) if union else 1.0,
            "authored_only": sorted(f"{a}->{b}" for a, b in authored - manifest_edges),
            "manifest_only": sorted(f"{a}->{b}" for a, b in manifest_edges - authored),
            "authored_role": rec.derived_role,
            "manifest_role": manifest_role,
            "role_agree": role_agree,
            "manifest_provenance": att.provenance,
        }
        if not exact or not role_agree:
            row["authored_rationale"] = {
                f"{p.from_node}->{p.to_node}": {"grade": p.grade, "rationale": p.rationale}
                for p in rec.edge_provenance
            }
            row["authored_reasoning"] = rec.reasoning
            row["manifest_rationale"] = manifest_grounding(rec.feature_name)
        rows.append(row)
    return {
        "manifest": manifest,
        "n_features": len(records),
        "n_compared": n_compared,
        "edge_exact_agreement": n_edge_exact,
        "role_agreement": n_role_agree,
        "disagreements": [
            r["feature"]
            for r in rows
            if r.get("compared") and not (r["edge_exact"] and r["role_agree"])
        ],
        "rows": rows,
        "note": "no threshold: the owner reads the diff and decides whether to relabel (spec §3 Lane B item 1)",
    }


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


async def open_review(
    *,
    dag: Any,
    attestations_path: Path,
    manifest: str,
    treatment: str,
    outcome: str,
    treatment_label: str,
    outcome_label: str,
    brand: Optional[str],
    model_id: str,
    prompt_hash: str,
    guide_hash: str,
    assumption: Optional[str],
    repo: Any = None,
) -> dict[str, Any]:
    """Create the ``initial_dag`` review row and cache the evidence table.

    Raises ``RuntimeError`` when no row was created (no client, refused write)
    or when the evidence write matched no row, so the CLI exits non-zero
    instead of pretending a usable review exists. ``repo`` lets a test inject
    an in-memory repository that enforces the same version-pair guard.
    """
    from src.causal_engine.dag_hash import compute_adjustment_set_hash, compute_dag_hash
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.expert_review import ExpertReviewRepository

    snapshot = dag.dag_structure_json()
    dag_hash = compute_dag_hash(causal_graph=snapshot)
    snapshot["dag_version_hash"] = dag_hash
    adj_hash = compute_adjustment_set_hash(snapshot.get("adjustment_sets") or [])
    evidence = {
        "structural_author": {
            "schema_version": "1",
            "manifest": manifest,
            "treatment": treatment,
            "outcome": outcome,
            "treatment_label": treatment_label,
            "outcome_label": outcome_label,
            "model_id": model_id,
            "prompt_hash": prompt_hash,
            "guide_hash": guide_hash,
            "dag_version_hash": dag_hash,
            "adjustment_set_hash": adj_hash,
            "attestations_path": str(attestations_path),
            "adjustment_valid": dag.adjustment_valid,
            "adjustment_set": list(dag.adjustment_set),
            "minimal_adjustment_set": list(dag.minimal_adjustment_set),
            "warnings": list(dag.warnings),
            "features": dag.evidence_table(),
        }
    }
    checklist = {
        "reviewer_confirmations": [
            {"id": "escalation_decision_point", "statement": assumption, "confirmed": None}
        ]
        if assumption
        else []
    }
    if repo is None:
        client = await get_async_supabase_client()
        repo = ExpertReviewRepository(supabase_client=client)
    review_id = await repo.create_review(
        reviewer_id="structural_author",
        review_type="initial_dag",
        dag_version_hash=dag_hash,
        reviewer_role="data_science",
        brand=brand,
        treatment_variable=treatment,
        outcome_variable=outcome,
        analysis_context=(
            f"Structural author (Lane B) cohort DAG for manifest={manifest}: "
            f"T={treatment} ({treatment_label}), Y={outcome} ({outcome_label}); "
            f"model={model_id}; {len(dag.features)} features; machine-authored, "
            "provenance=machine until approved."
            + (f" Stated assumption to confirm: {assumption}" if assumption else "")
        ),
        checklist=checklist,
        dag_structure=snapshot,
        adjustment_set_hash=adj_hash,
    )
    if not review_id:
        raise RuntimeError("no expert_reviews row was created (no client or the write was refused)")
    # The repository guards the write on the version PAIR: with only the DAG
    # half given, the adjustment half is matched as IS NULL and the row we just
    # minted (non-null adjustment hash) matches zero rows (codex r1 HIGH 1).
    persisted = await repo.update_agent_assessment(
        review_id, evidence, for_dag_version_hash=dag_hash, for_adjustment_set_hash=adj_hash
    )
    if not persisted:
        raise RuntimeError(
            f"review {review_id} was created but its structural-author evidence was NOT "
            "persisted (version filter matched no row); the review cannot seed a prior"
        )
    return {
        "review_id": review_id,
        "dag_version_hash": dag_hash,
        "adjustment_set_hash": adj_hash,
        "assessment_persisted": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# review.md
# ---------------------------------------------------------------------------


def render_review_md(
    *,
    dag: Any,
    records: list[Any],
    manifest: str,
    treatment: str,
    outcome: str,
    treatment_label: str,
    outcome_label: str,
    lm_label: str,
    prompt_hash: str,
    guide_hash: str,
    assumption: Optional[str],
    diff: Optional[dict[str, Any]],
    tree: Optional[dict[str, Any]] = None,
) -> str:
    by_feat = {r.feature_name: r for r in records}
    lines = [
        f"# Cohort DAG review: {manifest} — {treatment} → {outcome}",
        "",
        f"- treatment `{treatment}` = {treatment_label}",
        f"- outcome `{outcome}` = {outcome_label}",
        f"- author LM: `{lm_label}`; prompt hash `{prompt_hash}`; guide hash `{guide_hash}`",
        f"- tree: commit {(tree or {}).get('commit')} "
        f"(dirty src/scripts/tests: {(tree or {}).get('dirty_src_scripts_tests')})",
        "- provenance: `machine` on every fragment (audit-only until this review is approved)",
        f"- features authored: {len(records)}; latents: {', '.join(dag.latent_nodes) or 'none'}",
        f"- is a DAG: {dag.is_dag}; admissible observed adjustment set: {dag.adjustment_valid}",
        f"- minimal adjustment set: {dag.minimal_adjustment_set}",
        f"- full admissible set: {dag.adjustment_set}",
        "",
    ]
    if assumption:
        lines += ["## Reviewer checklist", "", f"- [ ] escalation_decision_point: {assumption}", ""]
    if dag.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in dag.warnings] + [""]
    lines += [
        "## Features",
        "",
        "| feature | fragment role | cohort role | drift | ambiguous | review | panel role | leak | grades |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in dag.evidence_table():
        grades = ", ".join(f"{k}:{v}" for k, v in row["edge_grades"].items())
        lines.append(
            f"| {row['feature']} | {row['fragment_role']} | {row['cohort_role']} | "
            f"{'yes' if row['role_drift'] else ''} | {'yes' if row['ambiguous'] else ''} | "
            f"{'yes' if row['review_required'] else ''} | {row['panel_final_role']} | "
            f"{row['leak_source'] or ''} | {grades} |"
        )
    needs = [r for r in records if r.review_required or r.ambiguous]
    lines += ["", f"## Review items ({len(needs)})", ""]
    for r in needs:
        lines.append(
            f"- **{r.feature_name}** (derived {r.derived_role}, expected {r.expected_role}):"
        )
        for reason in r.review_reasons:
            lines.append(f"  - {reason}")
        if r.cross_check.get("agrees") is False:
            lines.append(
                f"  - cross-check: author {r.cross_check['derived_role']} vs panel "
                f"{r.cross_check['panel_final_role']}"
            )
    if not needs:
        lines.append("- none")
    lines += ["", "## Edge rationale (non-estimand edges)", ""]
    for r in records:
        for p in r.edge_provenance:
            if p.grade == "estimand":
                continue
            cits = ", ".join(c.get("raw", "") for c in p.citations) or "no citation"
            flag = f" [dropped: {p.constraint_violation}]" if p.constraint_violation else ""
            lines.append(
                f"- {p.from_node} → {p.to_node} ({p.grade}{flag}): {p.rationale or '—'} — {cits}"
            )
    if diff is not None:
        lines += [
            "",
            "## Diff against the manifest's machine attestations",
            "",
            f"- compared {diff['n_compared']}/{diff['n_features']}; edge-exact agreement "
            f"{diff['edge_exact_agreement']}; role agreement {diff['role_agreement']}; "
            f"disagreements {len(diff['disagreements'])} (no threshold)",
        ]
        for row in diff["rows"]:
            if not row.get("compared") or (row["edge_exact"] and row["role_agree"]):
                continue
            lines.append(
                f"- **{row['feature']}**: authored {row['authored_role']} vs manifest "
                f"{row['manifest_role']}; authored-only {row['authored_only']}; "
                f"manifest-only {row['manifest_only']}"
            )
            for b in (row.get("manifest_rationale") or {}).get("bullets") or []:
                lines.append(
                    f"  - manifest ({row['manifest_rationale']['source']}:{b['line']}): {b['text']}"
                )
            if row.get("authored_reasoning"):
                lines.append(f"  - author: {row['authored_reasoning']}")
    _ = by_feat
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", choices=MANIFESTS, required=True)
    p.add_argument("--treatment", required=True, help="Treatment column (the node T).")
    p.add_argument("--outcome", required=True, help="Outcome column (the node Y).")
    p.add_argument("--treatment-label", default=None, help="What T stands for in the brief.")
    p.add_argument("--outcome-label", default=None, help="What Y stands for in the brief.")
    p.add_argument("--brand", default=None)
    p.add_argument("--features", default=None, help="Comma-separated subset (default: all SAFE).")
    p.add_argument(
        "--panel", type=Path, default=None, help="Lane E panel.json for this (manifest, T, Y)."
    )
    p.add_argument("--lm", choices=("fake", "real"), default="fake")
    p.add_argument("--i-accept-cost", action="store_true")
    p.add_argument("--resolver", choices=("offline", "live"), default=None)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--review", action="store_true", help="Open an expert_reviews row (prod write).")
    p.add_argument(
        "--allow-fake-review", action="store_true", help="Let a fake-LM run open a review."
    )
    p.add_argument("--no-assumption", action="store_true", help="Omit the run (a) checklist item.")
    p.add_argument(
        "--allow-no-panel", action="store_true", help="Let a real run author without --panel."
    )
    p.add_argument("--diff-manifest-attestations", action="store_true")
    p.add_argument("--merge-latents", choices=("yes", "no"), default="yes")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    if args.review and args.lm == "fake" and not args.allow_fake_review:
        logger.error(
            "--review with --lm fake would queue a stand-in DAG for a human; pass --allow-fake-review to rehearse"
        )
        return 4
    treatment_label = args.treatment_label or args.treatment
    outcome_label = args.outcome_label or args.outcome
    features = [f.strip() for f in args.features.split(",") if f.strip()] if args.features else None

    from src.data.kg.structural_author import (
        GUIDE_HASH,
        author_feature,
        prompt_hash,
        tree_identity,
    )
    from src.ml.causal_role_dgp.assembler import assemble_cohort_dag

    panel = None
    if args.panel is not None:
        panel = _load_panel(
            args.panel, manifest=args.manifest, treatment=args.treatment, outcome=args.outcome
        )

    resolver_mode = args.resolver or ("live" if args.lm == "real" else "offline")
    if args.lm == "real":
        if not args.i_accept_cost:
            logger.error("--lm real is a paid run (owner decision); re-run with --i-accept-cost")
            return 3
        from src.optimization.dspy_lm import ensure_dspy_configured

        if args.panel is None and not args.allow_no_panel:
            logger.error(
                "--lm real without --panel would author without the Lane E voters (spec: the "
                "panel is part of every brief); pass --panel, or --allow-no-panel deliberately"
            )
            return 4
        if not ensure_dspy_configured():
            logger.error("no DSPy LM could be configured (missing provider key?)")
            return 3
        lm_label = os.environ.get("DSPY_LM_MODEL") or "configured"
    else:
        for var in _PROVIDER_KEYS:
            os.environ.pop(var, None)
        lm_label = "fake"
        if resolver_mode == "live":
            logger.error("--resolver live is only allowed with --lm real")
            return 4
    if resolver_mode == "live":
        from src.data.kg.citation_resolver import CitationResolver

        resolver: Any = CitationResolver()
    else:
        resolver = OfflineResolver()

    briefs = build_briefs(
        manifest=args.manifest,
        treatment=args.treatment,
        outcome=args.outcome,
        treatment_label=treatment_label,
        outcome_label=outcome_label,
        features=features,
        panel=panel,
    )
    logger.info(
        "%d briefs for %s (%s → %s)", len(briefs), args.manifest, args.treatment, args.outcome
    )

    records = []
    for brief in briefs:
        if args.lm == "fake":
            from dspy.utils.dummies import DummyLM

            lm = DummyLM(
                [
                    {
                        "reasoning": "fake LM (dry run): stand-in confounder fragment, not an authored claim",
                        "edges": json.dumps(
                            [[brief.feature_name, "T"], [brief.feature_name, "Y"], ["T", "Y"]]
                        ),
                        "edge_rationales": "[]",
                        "entity_names": "{}",
                        "expected_role": "confounder",
                        "ambiguous": "false",
                    }
                ]
            )
            records.append(author_feature(brief, resolver=resolver, lm=lm))
        else:
            records.append(author_feature(brief, resolver=resolver))

    dag = assemble_cohort_dag(
        records,
        treatment=args.treatment,
        outcome=args.outcome,
        merge_latents=(args.merge_latents == "yes"),
    )
    assumption = None if args.no_assumption else ESCALATION_ASSUMPTION
    diff = (
        diff_against_manifest(records, manifest=args.manifest)
        if args.diff_manifest_attestations
        else None
    )

    out_dir = args.out_root / f"{args.manifest}_{args.treatment}_{args.outcome}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "authored_at": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "treatment": args.treatment,
        "outcome": args.outcome,
        "treatment_label": treatment_label,
        "outcome_label": outcome_label,
        "lm": lm_label,
        "resolver": resolver_mode,
        "prompt_hash": prompt_hash(),
        "guide_hash": GUIDE_HASH,
        "n_features": len(records),
        "panel": str(args.panel) if args.panel else None,
        "provenance": "machine",
        "tree": tree_identity(PROJECT_ROOT),
    }
    attestations_path = out_dir / "attestations.json"
    attestations_path.write_text(
        json.dumps({"meta": meta, "records": [r.to_dict() for r in records]}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dag.json").write_text(
        json.dumps({"meta": meta, **dag.to_dict()}, indent=2), encoding="utf-8"
    )
    if diff is not None:
        (out_dir / "manifest_diff.json").write_text(json.dumps(diff, indent=2), encoding="utf-8")
        logger.info(
            "manifest diff: compared %d, edge-exact %d, role-agree %d, disagreements %d",
            diff["n_compared"],
            diff["edge_exact_agreement"],
            diff["role_agreement"],
            len(diff["disagreements"]),
        )
    (out_dir / "review.md").write_text(
        render_review_md(
            dag=dag,
            records=records,
            manifest=args.manifest,
            treatment=args.treatment,
            outcome=args.outcome,
            treatment_label=treatment_label,
            outcome_label=outcome_label,
            lm_label=lm_label,
            prompt_hash=meta["prompt_hash"],
            guide_hash=GUIDE_HASH,
            tree=meta["tree"],
            assumption=assumption,
            diff=diff,
        ),
        encoding="utf-8",
    )
    logger.info(
        "wrote %s (is_dag=%s, adjustment_valid=%s, minimal=%s, review items=%d)",
        out_dir,
        dag.is_dag,
        dag.adjustment_valid,
        dag.minimal_adjustment_set,
        sum(1 for r in records if r.review_required or r.ambiguous),
    )

    if args.review:
        try:
            result = asyncio.run(
                open_review(
                    dag=dag,
                    attestations_path=attestations_path,
                    manifest=args.manifest,
                    treatment=args.treatment,
                    outcome=args.outcome,
                    treatment_label=treatment_label,
                    outcome_label=outcome_label,
                    brand=args.brand,
                    model_id=records[0].model_id if records else lm_label,
                    prompt_hash=meta["prompt_hash"],
                    guide_hash=GUIDE_HASH,
                    assumption=assumption,
                )
            )
        except Exception as exc:  # noqa: BLE001 — the CLI must not pretend a review exists
            logger.error("expert review NOT opened: %s (files kept in %s)", exc, out_dir)
            return 3
        (out_dir / "review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(
            "expert review opened: %s (dag hash %s)",
            result["review_id"],
            result["dag_version_hash"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
