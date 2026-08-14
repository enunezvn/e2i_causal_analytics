"""Phase 2.9 Stage 2 KG cache builder.

Reads a manifest module (e.g. ``src.data.manifests.optum_feature_manifest``),
queries KG for every feature with ``kg_entity_codes`` set, and writes a
cache file at::

    {out_dir}/{manifest_sha8}__{target_sha8}.json

Plus a companion summary report at::

    {out_dir}/{manifest_sha8}__{target_sha8}.summary.md

The cache filename has no cohort name — two cohorts with identical
(manifest, target_entity_codes) legitimately share a cache file.

Usage::

    # Schema-and-IO smoke (no clients; emits queried_no_edges records):
    python scripts/build_kg_cache.py \
        --manifest-module src.data.manifests.optum_feature_manifest \
        --features-attr OPTUM_FEATURES \
        --target-entity-codes RXNORM:479158,RXNORM:1011295 \
        --out data/kg_cache

    # Live KG querying (instantiates UMLSClient + OpenTargetsClient via
    # EntityLinker + KnowledgeGraphQuerier; UMLS_UTS_API_KEY required):
    python scripts/build_kg_cache.py --live ... (other args)

Item B of the engineering-actionable arc (PR-C live KG querying loop):
the ``--live`` mode replaces the prior NotImplementedError. Per-entity
dispatch handles ``("UMLS", <CUI>)`` entries directly via
``UMLSClient.cui_lookup`` (validate) + ``KGQuerier.query_disease_hierarchy``
(query) — bypassing ``EntityLinker.resolve``, which only knows the
source-vocab cross-walk systems (ICD10CM/LOINC/CPT/HCPCS/RXNORM and
friends) per ``_UTS_SOURCE_BY_SYSTEM``. Without this dispatch the ~30
UMLS-CUI entries in the Optum manifest would silently degrade to no_signal.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

# Allow `python scripts/build_kg_cache.py` invocation without `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_contract import FeatureContract  # noqa: E402
from src.data.kg.cache import (  # noqa: E402
    CacheRecord,
    compose_cache_filename,
    compute_manifest_fingerprint,
    compute_target_codes_fingerprint,
    save_cache,
)
from src.data.kg.types import KGEdge  # noqa: E402

if TYPE_CHECKING:
    from src.data.kg.entity_linker import EntityLinker
    from src.data.kg.kg_querier import KnowledgeGraphQuerier

logger = logging.getLogger(__name__)

# Pagination ceiling for individual UMLS / Open Targets queries — beyond
# this row count the response is likely truncated by the upstream API
# (see PR #86 record on cui_relations returning first 50 rows). Live
# querying logs a warning at this threshold so build operators can spot
# silent truncation in the cache build.
PAGINATION_WARNING_THRESHOLD = 50


def _parse_target_codes(arg: str) -> list[tuple[str, str]]:
    """Parse ``RXNORM:479158,RXNORM:1011295`` into ``[(RXNORM, 479158), ...]``.

    Empty / whitespace-only input parses to an empty list. Each piece must
    contain exactly one colon; both system and code must be non-empty
    after stripping (target codes bypass FeatureContract's ``__post_init__``
    code-validation, so the parser is the only gate against malformed
    input entering the cache fingerprint).
    """
    if not arg or not arg.strip():
        return []
    out: list[tuple[str, str]] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.count(":") != 1:
            raise ValueError(
                f"Bad target code {piece!r}; expected exactly SYSTEM:code with one colon"
            )
        system, code = (part.strip() for part in piece.split(":", 1))
        if not system or not code:
            raise ValueError(f"Bad target code {piece!r}; expected non-empty SYSTEM:code")
        out.append((system, code))
    return out


def _resolve_entity_to_cui(
    system: str,
    code: str,
    entity_linker: "EntityLinker",
) -> tuple[Optional[str], Optional[str]]:
    """Resolve a ``(system, code)`` entity to a UMLS CUI.

    Per the Item B / PR-B handoff comment in
    ``optum_feature_manifest.py``: ``EntityLinker.resolve`` accepts only
    the source-vocab cross-walk systems (ICD10CM / LOINC / CPT / HCPCS /
    RXNORM / SNOMEDCT_US / MESH); ``"UMLS"`` is NOT in
    ``_UTS_SOURCE_BY_SYSTEM`` so passing a CUI through ``resolve`` would
    silently degrade. UMLS entries are dispatched directly to
    ``UMLSClient.cui_lookup`` for validation and the bare CUI is used
    for downstream KG queries.

    Returns ``(cui, error)``: exactly one is non-None.
    """
    from src.data.kg.umls_uts import UMLSError, UMLSNotFoundError

    if system == "UMLS":
        try:
            entity_linker.umls.cui_lookup(code)
        except UMLSNotFoundError as exc:
            return None, f"UMLS CUI {code!r} not found: {exc}"
        except UMLSError as exc:
            return None, f"UMLS lookup failed for {code!r}: {exc}"
        return code, None

    try:
        link = entity_linker.resolve(code, system)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 — boundary error → cache record
        return None, f"EntityLinker.resolve({system}, {code}) raised: {exc}"
    if link.concept is None:
        return None, f"({system}, {code}) failed to resolve: {link.error or 'no concept'}"
    return link.concept.cui, None


def _query_edges_for_cui(
    cui: str,
    kg_querier: "KnowledgeGraphQuerier",
) -> tuple[tuple[KGEdge, ...], list[str]]:
    """Query KG for taxonomic edges anchored on ``cui``.

    Returns ``(edges, errors)``. Errors are non-fatal at this layer —
    the caller may have multiple entities per feature, and a partial
    failure on one should not poison the whole record. Logs a warning
    when the response hits the pagination ceiling so build operators
    can spot silently truncated results (codex B4).

    Catches ``UMLSError`` (transport / request-level UMLS failures) but
    NOT ``UMLSAuthError`` — auth failures are fatal across the entire
    UMLS surface, so we let them propagate up to abort the build with
    an unambiguous error rather than masquerade as per-feature
    ``source_error``. This narrowing (vs the prior ``except Exception``)
    is part of the codex H1 follow-up from PR #102: typed-error
    propagation lets ``status=source_error`` mean "transport failed",
    and a programming bug elsewhere now bubbles up as the bug it is.

    ``OpenTargetsError`` is a defensive guard, not currently reachable
    via ``query_disease_hierarchy`` (which only delegates to UMLS's
    ``cui_relations``). It's listed in the catch tuple so a future
    change wiring ``query_drug_disease_edges`` (Open Targets) into this
    helper path won't silently regress to broad-Exception behavior; the
    guard keeps the same typed-transport-failure semantics. Codex
    LOW-1 review on PR #103 flagged the catch as currently unreachable;
    this comment documents why we keep it.
    """
    from src.data.kg.open_targets import OpenTargetsError
    from src.data.kg.umls_uts import UMLSAuthError, UMLSError

    errors: list[str] = []
    try:
        edges = tuple(kg_querier.query_disease_hierarchy(cui))
    except UMLSAuthError:
        # Fatal — auth failure on one CUI implies auth is broken for
        # every CUI in this build. Surface to abort the whole run.
        raise
    except (UMLSError, OpenTargetsError) as exc:
        return (), [f"query_disease_hierarchy({cui!r}) raised: {exc}"]
    if len(edges) >= PAGINATION_WARNING_THRESHOLD:
        logger.warning(
            "KG query for CUI %s returned %d edges (>= pagination ceiling %d); "
            "result may be truncated",
            cui,
            len(edges),
            PAGINATION_WARNING_THRESHOLD,
        )
    return edges, errors


def _resolve_target_drug(
    target_entity_codes: list[tuple[str, str]],
    entity_linker: "EntityLinker",
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Resolve the prediction target to a ChEMBL drug id, ONCE per build.

    Returns ``(chembl_id, target_code_used, errors)``.

    This is the step that makes RxNav load-bearing at build time: a target
    expressed as an RXNORM code is turned into a drug NAME via RxNav, and the
    name is what Open Targets' ``search_drug`` can resolve to a ChEMBL id. A
    target already given as ``("CHEMBL", "CHEMBL1201589")`` is used directly.

    Returning ``(None, None, errors)`` is a normal outcome — the target may not
    be a drug at all — and simply means no drug-disease edges are attempted.
    """
    errors: list[str] = []
    for system, code in target_entity_codes:
        system_upper = system.upper()
        name: Optional[str] = None
        if system_upper == "CHEMBL":
            return code, code, errors
        if system_upper == "RXNORM":
            try:
                props = entity_linker.rxnav.properties(code)
            except Exception as exc:  # noqa: BLE001 - best-effort resolution
                errors.append(f"rxnav properties({code}) failed: {exc}")
                continue
            if not props:
                errors.append(f"rxnav: RXNORM:{code} is not a known RxCUI")
                continue
            name = str(props.get("name") or "")
        elif system_upper in ("UMLS", "MESH"):
            # A target expressed as a concept: use its preferred name.
            try:
                name = entity_linker.umls.cui_lookup(code).preferred_name
            except Exception as exc:  # noqa: BLE001
                errors.append(f"umls cui_lookup({code}) failed: {exc}")
                continue
        if not name:
            continue
        try:
            chembl_id = entity_linker.open_targets.search_drug(name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"open_targets search_drug({name!r}) failed: {exc}")
            continue
        if chembl_id:
            return chembl_id, code, errors
        errors.append(f"open_targets: no ChEMBL id for target drug {name!r}")
    return None, None, errors


def _drug_disease_edges_for_cui(
    cui: str,
    *,
    entity_linker: "EntityLinker",
    kg_querier: "KnowledgeGraphQuerier",
    target_chembl_id: str,
    target_code: str,
) -> tuple[tuple[KGEdge, ...], list[str]]:
    """Open Targets "does the target drug treat this feature's disease?" edges.

    This is the Layer-2 signal the voter was designed around
    (``leak_drug_treats_disease``) and the one the taxonomic path can never
    produce: ``query_disease_hierarchy`` only relates a feature concept to its
    own parents/children, never to the prediction target, so
    ``classify_kg_signal``'s ``_connects`` check could never fire (#1607).

    Endpoint rewriting is deliberate. Open Targets speaks ChEMBL/MONDO ids,
    while the runtime's ``feature_entity_ids`` / ``target_entity_ids`` come
    from the manifest and scope_spec. An edge keyed on ChEMBL/MONDO would be
    invisible to ``_connects``. The edge asserts "target drug treats THIS
    feature's disease", so it is emitted against the identifiers both sides
    actually hold.

    Because the rewrite discards the identifiers the source actually spoke, the
    originals are carried on ``source_subject_id`` / ``source_object_id``, which
    ``cache._kg_edge_to_json`` persists. That matters more than it looks: the
    feature's CUI is mapped to a disease by a fuzzy
    ``open_targets.search_disease(preferred_name)`` lookup, and a broad or
    outright wrong EFO/MONDO match still produces a perfectly plausible
    ``object_name``. These edges drive a leakage finding, so "which disease did
    we actually match this feature to?" has to be answerable from the committed
    artifact alone — names are not auditable, ids are.

    ``KGEdge.raw`` keeps the fuller context (including the upstream row) for
    in-process callers, but is NOT persisted: ``_kg_edge_to_json`` writes a
    fixed field list that excludes it.
    """
    errors: list[str] = []
    try:
        concept_name = entity_linker.umls.cui_lookup(cui).preferred_name
    except Exception as exc:  # noqa: BLE001
        return (), [f"umls cui_lookup({cui}) failed for drug-disease: {exc}"]
    if not concept_name:
        return (), [f"no preferred name for {cui}; cannot resolve a disease id"]
    try:
        efo_id = entity_linker.open_targets.search_disease(concept_name)
    except Exception as exc:  # noqa: BLE001
        return (), [f"open_targets search_disease({concept_name!r}) failed: {exc}"]
    if not efo_id:
        # Not an error: many features are labs/utilisation concepts with no
        # disease counterpart in Open Targets.
        return (), []
    try:
        raw_edges = kg_querier.query_drug_disease_edges(target_chembl_id, efo_id)
    except Exception as exc:  # noqa: BLE001 - mirrors the taxonomic path
        return (), [f"open_targets drug-disease {target_chembl_id}/{efo_id} failed: {exc}"]

    rewritten: list[KGEdge] = []
    for edge in raw_edges:
        rewritten.append(
            KGEdge(
                subject_id=target_code,
                subject_name=edge.subject_name or target_chembl_id,
                predicate=edge.predicate,
                object_id=cui,
                object_name=edge.object_name or concept_name,
                evidence_source=edge.evidence_source,
                score=edge.score,
                pmids=edge.pmids,
                datasource=edge.datasource,
                evidence=edge.evidence,
                source_subject_id=edge.subject_id,
                source_object_id=edge.object_id,
                raw={
                    "open_targets_subject_id": edge.subject_id,
                    "open_targets_object_id": edge.object_id,
                    "resolved_from_cui": cui,
                    "resolved_disease_id": efo_id,
                    "row": edge.raw,
                },
            )
        )
    return tuple(rewritten), errors


def _build_record_live(
    fc: FeatureContract,
    *,
    manifest_fp: str,
    target_fp: str,
    target_entity_codes: list[tuple[str, str]],
    sources_attempted: tuple[str, ...],
    entity_linker: "EntityLinker",
    kg_querier: "KnowledgeGraphQuerier",
    target_chembl_id: Optional[str] = None,
    target_code: Optional[str] = None,
) -> CacheRecord:
    """Per-feature live-query path. Aggregates edges + errors across each
    of the feature's ``kg_entity_codes``.

    De-duplication contract (codex L4): a feature with multiple entity
    codes that all resolve to the same CUI (e.g., ``("ICD10CM", "L50.9")``
    + ``("UMLS", "C0042109")`` where C0042109 IS the CUI for L50.9) only
    queries the upstream KG ONCE per unique CUI. Without this, repeated
    queries produce duplicate edges in the cache, and downstream
    EnsembleVoter logic would double-count them.

    Mixed-failure reporting contract (codex M3): when ANY entity's query
    returns edges, the record's status is ``ok``, and any errors from
    OTHER entities (failed to resolve, transport raise) are surfaced via
    the ``errors`` field — NOT via a downgraded status. Operators must
    check both ``status`` and ``errors`` to spot partial failures. This
    is a deliberate design choice: a feature with ANY edge should be
    actionable for KG signal, and the partial-resolution audit trail
    lives in errors.

    Status precedence (codex M2 fix):
    - ``ok`` — at least one edge returned
    - ``queried_no_edges`` — entity(ies) resolved AND every query
      succeeded with empty result (legitimate "no taxonomic relations")
    - ``source_error`` — entity(ies) resolved BUT every successful-
      resolution's query raised; the KG layer's silent transport
      degradation hits this branch
    - ``entity_unresolved`` — every entity failed to resolve at all
    """
    aggregated_edges: list[KGEdge] = []
    errors: list[str] = []
    resolved_cuis: set[str] = set()
    queries_attempted = 0
    queries_failed = 0
    for system, code in fc.kg_entity_codes:
        cui, err = _resolve_entity_to_cui(system, code, entity_linker)
        if err is not None or cui is None:
            errors.append(err or f"({system}, {code}) returned no CUI")
            continue
        if cui in resolved_cuis:
            # codex L4: a second entity code that maps to an already-
            # queried CUI does not re-query (would produce duplicate
            # edges) — but we still record the resolution so audit
            # logs see all entity codes were processed.
            continue
        resolved_cuis.add(cui)
        edges, edge_errors = _query_edges_for_cui(cui, kg_querier)
        queries_attempted += 1
        if edge_errors:
            queries_failed += 1
        aggregated_edges.extend(edges)
        errors.extend(edge_errors)

        # Drug-disease pass (#1607). The taxonomic edges above relate the
        # feature concept to its OWN hierarchy and can never connect it to the
        # prediction target, so on their own they always classify as
        # ``no_signal``. This pass asks the question the voter's
        # ``leak_drug_treats_disease`` rule was written for.
        if target_chembl_id and target_code:
            dd_edges, dd_errors = _drug_disease_edges_for_cui(
                cui,
                entity_linker=entity_linker,
                kg_querier=kg_querier,
                target_chembl_id=target_chembl_id,
                target_code=target_code,
            )
            aggregated_edges.extend(dd_edges)
            errors.extend(dd_errors)

    if aggregated_edges:
        status: Any = "ok"
    elif resolved_cuis and queries_attempted > 0 and queries_failed == queries_attempted:
        # codex M2: every query for resolved entities raised; this is a
        # transport failure, not a legitimate empty result.
        status = "source_error"
    elif resolved_cuis:
        status = "queried_no_edges"
    elif errors:
        unresolved_signals = ("not found", "failed to resolve", "no CUI")
        unresolved = any(any(s in e for s in unresolved_signals) for e in errors)
        status = "entity_unresolved" if unresolved else "source_error"
    else:
        status = "queried_no_edges"

    return CacheRecord(
        feature_name=fc.name,
        manifest_fingerprint_sha8=manifest_fp,
        target_codes_fingerprint_sha8=target_fp,
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=tuple((t[0], t[1]) for t in fc.kg_entity_codes),
        target_entity_codes=tuple((t[0], t[1]) for t in target_entity_codes),
        sources_attempted=sources_attempted,
        status=status,
        edges=tuple(aggregated_edges),
        errors=tuple(errors),
    )


def build_cache_for_manifest(
    *,
    features: Iterable[FeatureContract],
    target_entity_codes: list[tuple[str, str]],
    out_dir: Path,
    entity_linker: Optional["EntityLinker"] = None,
    kg_querier: Optional["KnowledgeGraphQuerier"] = None,
) -> Path:
    """Build the cache file for a manifest's entity-bearing features.

    Records are emitted only for features with non-empty
    ``kg_entity_codes``.

    Two paths:

    - **Smoke path** (default; ``entity_linker`` and ``kg_querier`` both
      None): emits records with ``status="queried_no_edges"`` and an
      empty edge tuple. Used by CI smoke tests and by operators who want
      to verify the manifest fingerprint plumbing without burning UMLS
      API calls.

    - **Live path** (Item B of the engineering-actionable arc; both
      ``entity_linker`` AND ``kg_querier`` non-None): per-entity dispatch
      validates UMLS CUIs via ``UMLSClient.cui_lookup`` and resolves
      source-vocab codes via ``EntityLinker.resolve``, then queries
      ``KGQuerier.query_disease_hierarchy`` for taxonomic edges. Drug-
      disease evidence via Open Targets is a v2 punt (requires
      ChEMBL/EFO cross-walks not yet wired through the EntityLinker).

    ``sources_attempted`` reflects what was *actually* attempted: empty
    tuple in smoke mode; ``("umls_uts",)`` in live mode (we don't yet
    hit Open Targets so it's not in the attempted set — silent
    inclusion would mislead downstream audit logic).

    Returns the path of the written cache file.

    Raises ``ValueError`` if exactly one of (entity_linker, kg_querier)
    is supplied — the live path requires both, the smoke path requires
    neither.
    """
    if (entity_linker is None) != (kg_querier is None):
        raise ValueError(
            "build_cache_for_manifest requires BOTH entity_linker AND "
            "kg_querier (live mode) or NEITHER (smoke mode); supplying "
            "exactly one is ambiguous and rejected."
        )

    live_mode = entity_linker is not None and kg_querier is not None
    sources_attempted: tuple[str, ...] = ("umls_uts",) if live_mode else ()

    # Resolve the prediction target to a ChEMBL drug ONCE per build (#1607).
    # When it resolves, every feature additionally gets the Open Targets
    # "does this drug treat that disease?" pass — the only path that can yield
    # a KG signal the voter can act on. ``sources_attempted`` records exactly
    # which upstreams were consulted so a reader can tell a genuine
    # "no evidence" from "we never asked".
    target_chembl_id: Optional[str] = None
    target_code: Optional[str] = None
    target_resolution_errors: list[str] = []
    if live_mode:
        assert entity_linker is not None
        target_chembl_id, target_code, target_resolution_errors = _resolve_target_drug(
            target_entity_codes, entity_linker
        )
        if target_chembl_id:
            sources_attempted = sources_attempted + ("rxnav", "open_targets")
            logger.info(
                "KG cache: target resolved to ChEMBL %s (from %s); drug-disease pass ENABLED",
                target_chembl_id,
                target_code,
            )
        else:
            logger.warning(
                "KG cache: target did not resolve to a ChEMBL drug (%s) — only "
                "taxonomic edges will be built, which cannot produce a KG signal "
                "against the target. Errors: %s",
                target_entity_codes,
                target_resolution_errors,
            )

    features = list(features)
    manifest_fp = compute_manifest_fingerprint(features)
    target_fp = compute_target_codes_fingerprint(target_entity_codes)
    cache_path = out_dir / compose_cache_filename(manifest_fp, target_fp)

    records: list[CacheRecord] = []
    for fc in features:
        if not fc.kg_entity_codes:
            continue
        if live_mode:
            assert entity_linker is not None and kg_querier is not None
            records.append(
                _build_record_live(
                    fc,
                    manifest_fp=manifest_fp,
                    target_fp=target_fp,
                    target_entity_codes=target_entity_codes,
                    sources_attempted=sources_attempted,
                    entity_linker=entity_linker,
                    kg_querier=kg_querier,
                    target_chembl_id=target_chembl_id,
                    target_code=target_code,
                )
            )
        else:
            records.append(
                CacheRecord(
                    feature_name=fc.name,
                    manifest_fingerprint_sha8=manifest_fp,
                    target_codes_fingerprint_sha8=target_fp,
                    queried_at=datetime.now(timezone.utc),
                    feature_entity_codes=tuple((t[0], t[1]) for t in fc.kg_entity_codes),
                    target_entity_codes=tuple((t[0], t[1]) for t in target_entity_codes),
                    sources_attempted=sources_attempted,
                    status="queried_no_edges",
                    edges=(),
                    errors=(),
                )
            )

    save_cache(records, cache_path)

    summary_path = cache_path.with_suffix(".summary.md")
    _write_summary_report(summary_path, records, manifest_fp, target_fp)

    return cache_path


def _write_summary_report(
    path: Path,
    records: list[CacheRecord],
    manifest_fp: str,
    target_fp: str,
    generated_at: Optional[datetime] = None,
) -> None:
    """Write the companion .summary.md report.

    ``generated_at`` is optional. When omitted, the report does NOT
    embed a timestamp — the summary file is byte-stable across
    regenerations (the cache JSON stays byte-stable already via
    deterministic record sort + sort_keys=True). Callers that want a
    timestamp must pass one explicitly so it remains pinnable.
    """
    lines: list[str] = [
        "# KG Cache Summary",
        "",
        f"**Manifest fingerprint:** `{manifest_fp}`",
        f"**Target codes fingerprint:** `{target_fp}`",
    ]
    if generated_at is not None:
        lines.append(f"**Generated:** {generated_at.isoformat()}")
    lines.extend(
        [
            f"**Records:** {len(records)}",
            "",
            "| Feature | Status | Edges | Sources |",
            "|---|---|---|---|",
        ]
    )
    for r in sorted(records, key=lambda r: r.feature_name):
        lines.append(
            f"| `{r.feature_name}` | {r.status} | {len(r.edges)} | "
            f"{', '.join(r.sources_attempted) or '—'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-module",
        required=True,
        help="Dotted module path containing the FEATURES list "
        "(e.g. src.data.manifests.optum_feature_manifest)",
    )
    parser.add_argument(
        "--features-attr",
        default="OPTUM_FEATURES",
        help="Attribute on the manifest module (default: OPTUM_FEATURES)",
    )
    parser.add_argument(
        "--target-entity-codes",
        default="",
        help="Comma-separated SYSTEM:code list, e.g. 'RXNORM:479158,RXNORM:1011295'",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the cache file",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Run live UMLS UTS KG querying (Item B). Requires "
            "UMLS_UTS_API_KEY env var. The cache currently records edges "
            "from query_disease_hierarchy(cui) only — drug-disease "
            "evidence via Open Targets is wired through KGQuerier but "
            "NOT yet queried at build time (requires ChEMBL/EFO cross-"
            "walks not yet present in EntityLinker; v2 punt). "
            "sources_attempted in the cache will show 'umls_uts' only. "
            "Without this flag the build emits schema-and-IO smoke "
            "records (status='queried_no_edges', empty edges)."
        ),
    )
    args = parser.parse_args(argv)

    module = importlib.import_module(args.manifest_module)
    features: Any = getattr(module, args.features_attr)
    target_codes = _parse_target_codes(args.target_entity_codes)

    if args.live:
        # Lazy import: only pay the httpx import cost when actually live.
        from src.data.kg.entity_linker import EntityLinker
        from src.data.kg.kg_querier import KnowledgeGraphQuerier

        with EntityLinker() as linker:
            kg_querier = KnowledgeGraphQuerier(entity_linker=linker)
            try:
                cache_path = build_cache_for_manifest(
                    features=features,
                    target_entity_codes=target_codes,
                    out_dir=args.out,
                    entity_linker=linker,
                    kg_querier=kg_querier,
                )
            finally:
                kg_querier.close()
    else:
        cache_path = build_cache_for_manifest(
            features=features,
            target_entity_codes=target_codes,
            out_dir=args.out,
        )
    print(f"Wrote cache to {cache_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
